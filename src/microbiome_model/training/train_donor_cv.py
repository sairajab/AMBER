import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from microbiome_model.data.dataset import Arguments, DataProcessor
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn
from microbiome_model.eval.evaluate import build_features
from microbiome_model.models import build_model
from microbiome_model.training.train_orig import cleanup_resources, setup_seed, train_model


DATA_ROOT = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
DATA_ROOT = Path(DATA_ROOT)
N_SPLITS = 5
SEED = 42
METADATA_COLUMNS = ["env", "bi_month_name"]

ARCHITECTURES = {
    "best_model_": {"model_name": "BasicRegressorFilm", "num_layers": 4, 
                         "input_dim": 128, "hidden_dim": 512, "num_heads": 8, "dropout": 0.3, "pe": False, "grl": False, "clr": False},
    # "film_4layer_nope_grl_no_norm": {"model_name": "BasicRegressorFilm", "num_layers": 4, 
    #                      "input_dim": 128, "hidden_dim": 512, "num_heads": 8, "dropout": 0.3, "pe": False, "grl": True, "clr": False},
    #"film_2layer_nope_nogrl_no_norm": {"model_name": "BasicRegressorFilm", "num_layers": 2, 
    #                         "input_dim": 128, "hidden_dim": 512, "num_heads": 8, "dropout": 0.3, "pe": False, "grl": False, "clr": False},
    #"film_4layer_nope_grl_no_norm": {"model_name": "BasicRegressorFilm", "num_layers": 4,
    #                         "input_dim": 128, "hidden_dim": 512, "num_heads": 8, "dropout": 0.3, "pe": False, "grl": True, "clr": False},
    
}
grl_weights = [0.5, 1.0]

# MODEL_DEFAULTS = {
#     "input_dim": 128,
#     "hidden_dim": 512,
#     "num_heads": 8,
#     "dropout": 0.3,
#     "pe": False,
#     "grl": True,
#     "clr": False,
# }

TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-4,
    "reg_count_loss": 1.0,
    "num_epochs": 200,
    "use_unifrac_loss": False,
    "scale_abundance": True,
    "grl_loss_weight": 0.1,
    "subsample": True,
    "embedding_type": "DNABERT",
}


def make_train_variants(architecture_name, architecture, weights):
    if not architecture.get("grl", False):
        return [(architecture_name, dict(TRAIN_CONFIG), None)]

    return [
        (
            f"{architecture_name}_weight_{weight:g}",
            {**TRAIN_CONFIG, "grl_loss_weight": float(weight)},
            float(weight),
        )
        for weight in weights
    ]


def normalize_donor(value):
    return "" if pd.isna(value) else str(value).strip()


def make_donor_folds(donor_to_samples, n_splits=N_SPLITS, seed=SEED):
    donors = np.asarray(sorted(donor_to_samples), dtype=object)
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_indices, val_indices in splitter.split(donors):
        train_donors = set(donors[train_indices])
        val_donors = set(donors[val_indices])
        if train_donors & val_donors:
            raise RuntimeError("Donor leakage detected in a CV fold")
        train_samples = [
            sample_id for donor in sorted(train_donors)
            for sample_id in donor_to_samples[donor]
        ]
        val_samples = [
            sample_id for donor in sorted(val_donors)
            for sample_id in donor_to_samples[donor]
        ]
        folds.append((train_donors, val_donors, train_samples, val_samples))
    return folds


def load_processor():
    biom_file = DATA_ROOT / "process_data_all/exported-feature-table/feature-table.biom"
    metadata_file = DATA_ROOT / "data/new_data/merged_metadata_final.csv"
    embedding_file = DATA_ROOT / "data/embeddings/all_data.h5"
    sheds_file = DATA_ROOT / "data/new_data/metadata_sheds.tsv"
    

    for path in (biom_file, metadata_file, embedding_file, sheds_file):
        if not path.exists():
            raise FileNotFoundError(f"Required input file not found: {path}")

    #embedding_file = None # for RANDOM embeddings, we don't need to load the embedding file
    excluded_donors = {
        normalize_donor(value)
        for value in pd.read_csv(sheds_file, sep="\t")["DonorID"].dropna().unique()
    }
    args = Arguments(
        biom_file=str(biom_file),
        metadata_file=str(metadata_file),
        tree_path=None,
        embedding_file=str(embedding_file),
        embedding="DNABERT",
        heldout=None,
        sort_asvs=True,
        clr=False,
    )
    processor = DataProcessor(args)
    processor.load_data(
        multitask=True,
        column="bi_month_name",
        metadata_columns=METADATA_COLUMNS,
    )

    donor_to_samples = defaultdict(list)
    for sample_id in processor.train_data[0]:
        donor = normalize_donor(processor.sampleID_map_donorID[sample_id])
        if donor and donor not in excluded_donors:
            donor_to_samples[donor].append(sample_id)

    if len(donor_to_samples) < N_SPLITS:
        raise ValueError(
            f"Only {len(donor_to_samples)} eligible donors remain after excluding "
            f"the {len(excluded_donors)} shed donors"
        )

    eligible_samples = [
        sample_id
        for sample_ids in donor_to_samples.values()
        for sample_id in sample_ids
    ]
    processor.train_data = (
        eligible_samples,
        {sample_id: processor.sample_targets[sample_id] for sample_id in eligible_samples},
    )
    processor.test_data = ([], {})
    return processor, donor_to_samples, excluded_donors


def evaluate_donor_macro_mae(model, processor, meta_spec, device, batch_size, subsample):
    _, val_dataset = processor.sample_data(epoch=0, subsample=subsample)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=sparse_collate_fn,
    )
    donor_errors = defaultdict(list)
    sample_errors = []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            embeddings = batch["embeddings"].to(device, dtype=torch.float32)
            abundances = batch["abundances"].to(device, dtype=torch.float32) / 1e4
            masks = batch["masks"].to(device)
            features = build_features(batch, meta_spec, device)
            predictions = model(embeddings, abundances, masks, features)[0].reshape(-1)
            targets = batch["outdoor_add_0"].to(device).reshape(-1)
            errors = torch.abs(predictions - targets).cpu().numpy()
            sample_errors.extend(errors.tolist())
            for sample_id, error in zip(batch["SampleID"], errors):
                donor = normalize_donor(processor.sampleID_map_donorID[sample_id])
                donor_errors[donor].append(float(error))

    donor_maes = {
        donor: float(np.mean(errors) * 100)
        for donor, errors in donor_errors.items()
    }
    return donor_maes, float(np.mean(sample_errors) * 100)


def run_cv(architecture_names, output_dir, grl_weight_values, dry_run=False):
    processor, donor_to_samples, excluded_donors = load_processor()
    folds = make_donor_folds(donor_to_samples)
    eligible_donors = set(donor_to_samples)
    if eligible_donors & excluded_donors:
        raise RuntimeError("Excluded shed donors are present in the CV donor set")

    print(f"Excluded shed donors: {len(excluded_donors)}")
    print(f"Eligible CV donors: {len(eligible_donors)}")
    for fold_index, (train_donors, val_donors, _, _) in enumerate(folds, start=1):
        print(
            f"Fold {fold_index}: {len(train_donors)} train donors, "
            f"{len(val_donors)} validation donors"
        )
        if not (37 <= len(train_donors) <= 38 and 9 <= len(val_donors) <= 10):
            print("Warning: donor counts differ from the requested 37-38 / 9-10 split")

    if dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    fold_results = []
    donor_results = []
    model = None
    checkpoint = None

    for architecture_name in architecture_names:
        architecture = ARCHITECTURES[architecture_name]
        train_variants = make_train_variants(
            architecture_name, architecture, grl_weight_values
        )
        for variant_name, train_config, grl_weight in train_variants:
            for fold_index, (train_donors, val_donors, train_samples, val_samples) in enumerate(folds):
                setup_seed(SEED)
                if model is not None:
                    del model
                    model = None
                if checkpoint is not None:
                    del checkpoint
                    checkpoint = None
                cleanup_resources()
                train_targets = {
                    sample_id: processor.sample_targets[sample_id]
                    for sample_id in train_samples
                }
                val_targets = {
                    sample_id: processor.sample_targets[sample_id]
                    for sample_id in val_samples
                }
                processor.train_data = (train_samples, train_targets)
                processor.val_data = (val_samples, val_targets)

                meta_spec = [
                    (column, len(processor.metadata_categories[column]))
                    for column in METADATA_COLUMNS
                ]
                model_config = {
                    **architecture,
                    "metadata_columns": METADATA_COLUMNS,
                    "metadata_categories": processor.metadata_categories,
                    "metadata_cardinalities": [cardinality for _, cardinality in meta_spec],
                    "num_donors": len(train_donors),
                }
                result_dir = output_dir / variant_name / f"fold_{fold_index + 1}"
                result_dir.mkdir(parents=True)
                with (result_dir / "model_config.json").open("w") as file:
                    json.dump(model_config, file, indent=2)
                with (result_dir / "training_config.json").open("w") as file:
                    json.dump({**train_config, "seed": SEED}, file, indent=2)

                model = build_model(model_config, str(device))
                train_result = train_model(
                    model=model,
                    train_config=train_config,
                    device=device,
                    out_dir=str(result_dir),
                    data_processor=processor,
                    use_unifrac_loss=False,
                    scale_abundance=True,
                    grl=architecture.get("grl", False),
                    meta_spec=meta_spec,
                )
                checkpoint = torch.load(train_result["best_model"], map_location=device)
                model.load_state_dict(checkpoint)
                model.to(device)
                donor_maes, sample_mae = evaluate_donor_macro_mae(
                    model,
                    processor,
                    meta_spec,
                    device,
                    train_config["batch_size"],
                    train_config["subsample"],
                )
                fold_macro_mae = float(np.mean(list(donor_maes.values())))
                fold_results.append({
                    "variant": variant_name,
                    "architecture": architecture_name,
                    "grl_loss_weight": grl_weight,
                    "fold": fold_index + 1,
                    "train_donors": len(train_donors),
                    "validation_donors": len(val_donors),
                    "donor_macro_mae": fold_macro_mae,
                    "sample_mae": sample_mae,
                    "training_best_mae_normalized": float(train_result["best_mae"]),
                })
                donor_results.extend(
                    {"variant": variant_name, "architecture": architecture_name,
                     "grl_loss_weight": grl_weight, "fold": fold_index + 1,
                     "donor": donor, "mae": mae}
                    for donor, mae in donor_maes.items()
                )
                pd.DataFrame(fold_results).to_csv(output_dir / "fold_results.csv", index=False)
                pd.DataFrame(donor_results).to_csv(output_dir / "donor_results.csv", index=False)
                print(
                    f"{variant_name} fold {fold_index + 1}: "
                    f"donor-macro MAE={fold_macro_mae:.3f}; sample MAE={sample_mae:.3f}"
                )

    fold_frame = pd.DataFrame(fold_results)
    donor_frame = pd.DataFrame(donor_results)
    summary = donor_frame.groupby("variant")["mae"].agg(
        mean_donor_mae="mean", std_donor_mae="std", donors_evaluated="count"
    ).join(
        fold_frame.groupby("variant")["donor_macro_mae"].agg(
            mean_fold_mae="mean", std_fold_mae="std"
        )
    ).sort_values("mean_donor_mae")
    summary.to_csv(output_dir / "architecture_summary.csv")
    print("\nArchitecture comparison (lower donor-macro MAE is better):")
    print(summary.to_string())
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=sorted(ARCHITECTURES),
        default=list(ARCHITECTURES),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--grl-weights",
        nargs="+",
        type=float,
        default=grl_weights,
        help="GRL loss weights to compare for GRL-enabled architectures.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate donor exclusions and fold counts without training.",
    )
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        DATA_ROOT / "microbiome_model/results" / f"donor_cv_5fold_{timestamp}"
    )
    if any(weight < 0 for weight in args.grl_weights):
        parser.error("GRL weights must be non-negative")
    run_cv(args.architectures, output_dir, args.grl_weights, dry_run=args.dry_run)


if __name__ == "__main__":
    main()