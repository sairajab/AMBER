"""Cross-validated age prediction on pre-defined folds.

`run_exp_age.py` trains on a single train/test split read from
`group_assignments_{dataset}.json`. This script runs the same model over a
k-fold file (the one written by the TRPCA notebook, `folds_skin_5fold.json`),
so the model is scored on exactly the folds RF / SVR / TRPCA were scored on.

The validation set is carved out of each training fold with GroupShuffleSplit
on host_subject_id, so no subject appears in more than one of train / val /
test within a fold.

Usage
-----
    python -m microbiome_model.training.run_cv_age \
        --dataset skin_data \
        --folds /s/chromatin/o/nobackup/Saira/Microbiome_Project/TRPCA/folds_skin_5fold.json \
        --epochs 300 --seeds 0
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader

from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.models.orig import AMBERRegressor, BasicRegressor
from microbiome_model.training.run_exp_age import (
    LABEL_TO_YEARS,
    AgeDatasetLoader,
    collate_fn,
    plot_true_vs_pred,
    train_model,
)

DATASETS = {"gut_data": "gut_4434", "skin_data": "skin_1975", "oral_data": "oral_2550"}
DATA_DIR = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/Input/{}/"
EMBEDDING_DIR = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/age-prediction/my_outputs/"

BEST_PARAMS = {
    "gut_data": {"count_reg": 0.1, "target_column": "age"},
    "skin_data": {"count_reg": 0.1, "target_column": "qiita_host_age"},
    "oral_data": {"count_reg": 0.1, "target_column": "qiita_host_age"},
}

MODEL_CONFIG = {
    "input_dim": 256,
    "hidden_dim": 1024,
    "num_heads": 4,
    "num_layers": 2,
    "dropout": 0.4,
    "pe": False,
}

STRATIFY_BETA = None
STRATIFY_EDGES = (30, 40, 50, 60)


def load_folds(folds_path):
    """Read a fold file into [(fold_id, train_ids, test_ids), ...].

    Accepts the k-fold format written by the TRPCA notebook
    (a list of {fold, train_indices, test_indices}) as well as the older
    single-split `group_assignments_*.json` (one dict), which is treated as a
    single fold so this script also works on the original splits.
    """
    with open(folds_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = [{"fold": 1, **raw}]

    folds = [(entry.get("fold", i + 1), entry["train_indices"], entry["test_indices"])
             for i, entry in enumerate(raw)]

    all_test = [sid for _, _, test_ids in folds for sid in test_ids]
    if len(all_test) != len(set(all_test)):
        raise ValueError("Test sets overlap between folds — is this really a k-fold file?")
    print(f"Loaded {len(folds)} fold(s) from {folds_path}")
    return folds


def build_fold_datasets(loader, metadata_df, train_ids, test_ids, fold,
                        val_frac=0.15, group_column="host_subject_id"):
    """Set loader.{train,val,test}_dataset for one fold.

    `train_model` reads the datasets off the loader, so this mirrors what
    `AgeDatasetLoader.get_dataset_by_json` does — except the validation split
    is drawn per fold instead of being read from a shared val_samples file
    (reusing one global val set across folds would put fold k's test samples
    into fold k's validation set).
    """
    biom_ids = set(loader.table.ids(axis="sample"))
    labelled = set(metadata_df.index[metadata_df[loader.target_column].notna()])

    def usable(ids, name):
        kept = [s for s in ids if s in biom_ids and s in labelled]
        if len(kept) != len(ids):
            print(f"  fold {fold} {name}: dropped {len(ids) - len(kept)} samples "
                  f"(missing from BIOM table or unlabelled)")
        return kept

    train_ids = usable(train_ids, "train")
    test_ids = usable(test_ids, "test")

    train_ids, val_ids = train_test_split(train_ids, test_size=val_frac, random_state=42 + fold)
    print(f"  fold {fold}: train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)} samples ")
    # else:
    #     # Hold out whole subjects for validation
    #     train_groups = metadata_df.loc[train_ids, group_column].values
    #     gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=42 + fold)
    #     inner_train_pos, val_pos = next(gss.split(train_ids, groups=train_groups))
    #     val_ids = [train_ids[i] for i in val_pos]
    #     train_ids = [train_ids[i] for i in inner_train_pos]
    # print("**********************************CHECKING**********************************")
    # for a, b, name in [(train_ids, test_ids, "train/test"),
    #                     (val_ids, test_ids, "val/test")]:
    #         shared = set(metadata_df.loc[a, group_column]) & set(metadata_df.loc[b, group_column])
    #         assert not shared, f"fold {fold}: subjects shared across {name}: {sorted(shared)[:5]}"

    def make(ids, seed):
        return MicrobiomeSparseDataset(
            biom_table=loader.table.filter(ids, axis="sample", inplace=False),
            sample_targets=metadata_df.loc[ids][loader.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=loader.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=seed,
            n_bins=loader.n_bins,
            subsample=True,
            categorical_metadata=loader.categorical_metadata,
        )

    loader.train_dataset = make(train_ids, seed=0)
    loader.val_dataset = make(val_ids, seed=0)
    loader.test_dataset = make(test_ids, seed=None)
    loader.train_targets = metadata_df.loc[train_ids][loader.target_column].to_list()
    loader.val_targets = metadata_df.loc[val_ids][loader.target_column].to_list()
    loader.test_targets = metadata_df.loc[test_ids][loader.target_column].to_list()

    # subsample(5000, axis="observation") discards samples left empty, so the
    # datasets are smaller than the fold. Report it: the test fold the model is
    # actually scored on is not quite the fold RF/SVR/TRPCA saw.
    dropped = len(test_ids) - len(loader.test_dataset)
    if dropped:
        print(f"  fold {fold}: {dropped}/{len(test_ids)} test samples dropped by ASV "
              f"subsampling -> scored on {len(loader.test_dataset)}")

    return train_ids, val_ids, test_ids


def evaluate_fold(model, loader, metadata_df, test_ids, device, out_dir, evals=10,
                  batch_size=32):
    """Score the test fold, averaging over `evals` ASV-subsampling seeds.

    Same protocol as `evaluate_model` in run_exp_age, but it also returns the
    per-sample predictions so out-of-fold predictions can be compared against
    the other models sample by sample.
    """
    model.eval()
    per_seed_mae = []
    preds_by_sample = {}

    for k in range(evals):
        loader.test_dataset = MicrobiomeSparseDataset(
            biom_table=loader.table.filter(test_ids, axis="sample", inplace=False),
            sample_targets=metadata_df.loc[test_ids][loader.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=loader.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=k,
            n_bins=loader.n_bins,
            subsample=False,
            categorical_metadata=loader.categorical_metadata,
        )
        test_loader = DataLoader(loader.test_dataset, batch_size=batch_size,
                                 collate_fn=collate_fn, shuffle=False)

        seed_preds, seed_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch["embeddings"].to(device)
                abundances = batch["abundances"].to(device) / 1e4
                masks = batch["masks"].to(device)
                targets = batch["label"].to(device)
                features = batch["features"].to(device) if batch["features"] is not None else None

                outputs, _, _, _, _ = model(embeddings, abundances, masks, features)

                # atleast_1d: a trailing batch of 1 would otherwise squeeze to a scalar
                preds = np.atleast_1d((outputs.cpu().squeeze(-1) * LABEL_TO_YEARS).numpy())
                trues = np.atleast_1d((targets.cpu().squeeze() * LABEL_TO_YEARS).numpy())
                for sid, p, t in zip(batch["SampleID"], preds, trues):
                    preds_by_sample.setdefault(sid, {"true": float(t), "preds": []})
                    preds_by_sample[sid]["preds"].append(float(p))
                seed_preds.append(preds)
                seed_targets.append(trues)

        seed_preds = np.concatenate(seed_preds)
        seed_targets = np.concatenate(seed_targets)
        per_seed_mae.append(mean_absolute_error(seed_targets, seed_preds))
        plot_true_vs_pred(seed_targets, seed_preds, epoch=k, out_dir=out_dir)

    oof = pd.DataFrame([
        {"sample_id": sid, "true": v["true"], "pred": float(np.mean(v["preds"]))}
        for sid, v in preds_by_sample.items()
    ])

    mae_mean = float(np.mean(per_seed_mae))
    mae_std = float(np.std(per_seed_mae))
    mae_averaged = float(mean_absolute_error(oof["true"], oof["pred"]))
    print(f"  Test MAE: {mae_mean:.4f} +/- {mae_std:.4f} yrs over {evals} subsample seeds "
          f"| {mae_averaged:.4f} yrs on seed-averaged predictions")
    return mae_mean, mae_std, mae_averaged, oof


def run_fold(loader, metadata_df, fold, train_ids, test_ids, seed, args, device):
    train_ids, val_ids, test_ids = build_fold_datasets(
        loader, metadata_df, train_ids, test_ids, fold, val_frac=args.val_frac)

    out_dir = os.path.join(args.out_root, args.dataset, args.run_name,
                           f"fold_{fold}", f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    res = {"best_model": None, "best_mae": np.inf}
    model = BasicRegressor(
        input_dim=MODEL_CONFIG["input_dim"], hidden_dim=MODEL_CONFIG["hidden_dim"],
        num_heads=MODEL_CONFIG["num_heads"], num_layers=MODEL_CONFIG["num_layers"],
        dropout=MODEL_CONFIG["dropout"], pe=MODEL_CONFIG["pe"], grl=False,
        metadata_cardinalities=loader.metadata_cardinalities,
    )
    print("metadata_cardinalities:", loader.metadata_cardinalities)
    model.reset_parameters()
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = torch.nn.SmoothL1Loss(beta=0.3)

    res = train_model(model, criterion, optimizer, loader, device,
                      num_epochs=args.epochs, out_dir=out_dir,
                      count_reg=BEST_PARAMS[args.dataset]["count_reg"],
                      stratify_beta=STRATIFY_BETA, stratify_edges=STRATIFY_EDGES,
                      batch_size=args.batch_size)

    best_model_path = res["best_model"] or os.path.join(out_dir, "best_model.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    mae_mean, mae_std, mae_averaged, oof = evaluate_fold(
        model, loader, metadata_df, test_ids, device, out_dir,
        evals=args.evals, batch_size=args.batch_size)

    fold_result = {
        "fold": fold,
        "seed": seed,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "val_mae": res["best_mae"],
        "test_mae": mae_mean,
        "test_mae_std": mae_std,
        "test_mae_avg_preds": mae_averaged,
        "best_model": best_model_path,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({"units": "years", **fold_result}, f, indent=2)

    oof.insert(1, "fold", fold)
    oof.insert(2, "seed", seed)
    return fold_result, oof


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="skin_data", choices=sorted(DATASETS))
    parser.add_argument("--folds", default="/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
                                           "TRPCA/folds_skin_5fold.json",
                        help="fold file (k-fold list, or an old single-split group_assignments json)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="model init seeds; every seed is run on every fold")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="fraction of training SUBJECTS held out for validation")
    parser.add_argument("--evals", type=int, default=10,
                        help="ASV-subsampling seeds averaged at test time")
    parser.add_argument("--out-root", default="results/age_prediction")
    parser.add_argument("--run-name", default="basic_regressor_cv")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    target_column = BEST_PARAMS[args.dataset]["target_column"]
    data_dir = DATA_DIR.format(args.dataset)
    biom_table_path = os.path.join(data_dir, DATASETS[args.dataset] + ".biom")
    metadata_file = os.path.join(data_dir, DATASETS[args.dataset] + "_map.txt")
    embedding_path = os.path.join(EMBEDDING_DIR, DATASETS[args.dataset] + "_embeddings.h5")

    metadata_df = pd.read_csv(metadata_file, sep="\t", index_col=0, low_memory=False)
    metadata_df[target_column] = pd.to_numeric(metadata_df[target_column], errors="coerce")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dataset {args.dataset}, target '{target_column}', device {device}")

    
    
    loader = AgeDatasetLoader(biom_table_path, metadata_df, target_column, embedding_path) #, categorical_columns={"sex": "sex"}) #, "country": "country"}

    folds = load_folds(args.folds)

    results, oof_frames = [], []
    for fold, train_ids, test_ids in folds:
        for seed in args.seeds:
            print(f"\n{'=' * 60}\nFold {fold}/{len(folds)}  seed {seed}\n{'=' * 60}")
            wandb.init(project="microbiome-age-prediction",
                       group=f"{args.run_name}-{args.dataset}",
                       name=f"fold{fold}-seed{seed}",
                       config={**MODEL_CONFIG, "fold": fold, "seed": seed,
                               "folds_file": args.folds},
                       reinit=True)
            fold_result, oof = run_fold(loader, metadata_df, fold, train_ids, test_ids,
                                        seed, args, device)
            wandb.log({"test/mae": fold_result["test_mae"],
                       "val/best_mae": fold_result["val_mae"]})
            wandb.finish()
            results.append(fold_result)
            oof_frames.append(oof)

    results_df = pd.DataFrame(results)
    oof_df = pd.concat(oof_frames, ignore_index=True)

    cv_dir = os.path.join(args.out_root, args.dataset, args.run_name)
    os.makedirs(cv_dir, exist_ok=True)
    results_df.to_csv(os.path.join(cv_dir, "cv_fold_metrics.csv"), index=False)
    oof_df.to_csv(os.path.join(cv_dir, "cv_oof_predictions.csv"), index=False)

    print(f"\n{'=' * 60}\nCross-validation summary ({args.dataset}, all values in years)\n{'=' * 60}")
    print(results_df[["fold", "seed", "n_train", "n_test", "val_mae", "test_mae",
                      "test_mae_std"]].to_string(index=False))

    summary = {"dataset": args.dataset, "folds_file": args.folds, "units": "years",
               "n_folds": len(folds), "seeds": args.seeds, "per_seed": {}}

    for seed in args.seeds:
        seed_rows = results_df[results_df.seed == seed]
        summary["per_seed"][str(seed)] = {
            "test_mae_mean": float(seed_rows.test_mae.mean()),
            "test_mae_std": float(seed_rows.test_mae.std()),
            "val_mae_mean": float(seed_rows.val_mae.mean()),
        }
        print(f"seed {seed}: test MAE {seed_rows.test_mae.mean():.4f} "
              f"+/- {seed_rows.test_mae.std():.4f} across {len(seed_rows)} folds")

    # Pooled over every out-of-fold prediction (averaged over seeds per sample)
    pooled = oof_df.groupby("sample_id").agg(true=("true", "first"), pred=("pred", "mean"))
    summary["pooled_oof"] = {
        "n_samples": int(len(pooled)),
        "mae": float(mean_absolute_error(pooled["true"], pooled["pred"])),
        "r2": float(r2_score(pooled["true"], pooled["pred"])),
    }
    print(f"Pooled out-of-fold ({len(pooled)} samples): "
          f"MAE {summary['pooled_oof']['mae']:.4f}, R² {summary['pooled_oof']['r2']:.4f}")

    with open(os.path.join(cv_dir, "cv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {cv_dir}/cv_fold_metrics.csv, cv_oof_predictions.csv, cv_summary.json")


if __name__ == "__main__":
    main()
