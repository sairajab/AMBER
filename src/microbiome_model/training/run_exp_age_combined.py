"""Combined multi-site chronological age prediction (gut + skin + oral).

Trains a single AMBERRegressor over all three body sites at once, with the body
site passed to the model as a categorical metadata feature (alongside sex), so
the network can condition on which site a sample came from.

Everything that does the actual work is reused from run_exp_age:
    - AMBERRegressor            (models.orig)
    - MicrobiomeSparseDataset   (data.dataset_sparse)
    - collate_fn, train_model   (training.run_exp_age)
    - build_categorical_map     (training.run_exp_age)

What this file adds is only the *combination* layer:
    - MergedEmbeddingLoader     dispatches ASV lookups across three .h5 stores
    - CombinedAgeDatasetLoader  merges the three BIOM tables / metadata / splits
    - evaluate_by_site          per-site test metrics, not just a pooled number

IMPORTANT — read before running:

  * The abundance transform here mirrors whatever run_exp_age currently uses in
    its train/val loops (see _abundance_transform below). If you change it there,
    change it here too or train and test will disagree.

  * The three cohorts have very different age distributions (gut mean 47.8,
    skin 25.4, oral 28.1). Because body_site is an input feature, the model can
    score well on pooled MAE purely by learning a per-site age prior. A pooled
    number is therefore NOT comparable to your single-site results. Always read
    the per-site breakdown and the per-site mean-predictor baselines that
    evaluate_by_site prints.
"""

import json
import os
import random

import biom
import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader

from microbiome_model.data.dataset_sparse import MicrobiomeSparseDataset
from microbiome_model.data.optimized_embedding_loader import (
    MemoryMappedEmbeddingLoader as EmbeddingLoader,
)
from microbiome_model.models.orig import AMBERRegressor
from microbiome_model.training.run_exp_age import (
    LABEL_TO_YEARS,
    build_categorical_map,
    collate_fn,
    plot_true_vs_pred,
    train_model,
)

PROJECT_ROOT = "/s/chromatin/o/nobackup/Saira/Microbiome_Project"
INPUT_DIR = os.path.join(PROJECT_ROOT, "age-prediction/Input/{}/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "age-prediction/my_outputs/")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "TRPCA/2_trails_results")

# dataset dir name -> file stem
DATASETS = {
    "gut_data": "gut_4434",
    "skin_data": "skin_1975",
    "oral_data": "oral_2550",
}

# dataset dir name -> coarse body-site label used as the metadata feature
SITE_OF = {
    "gut_data": "gut",
    "skin_data": "skin",
    "oral_data": "oral",
}

# Only these metadata columns are carried across. The skin/oral map files have
# ~1100 columns each and concatenating them whole produces a mostly-NaN frame.
KEEP_COLUMNS = ["age", "sex", "host_subject_id", "body_site"]


def _abundance_transform(raw_abundances):
    """Must match the transform in run_exp_age's train/val loops.

    Dataset-side, abundances are relative and scaled to sum to 1e4 per sample
    (dataset_sparse.py), so log1p lands them in roughly 0-9.2.
    """
    return torch.log1p(raw_abundances)


class MergedEmbeddingLoader:
    """Serves ASV embeddings from several per-dataset .h5 stores.

    The three cohorts share some ASVs (6858 between gut and skin, 5224 between
    skin and oral, etc). ASV ids *are* the DNA sequences and the embeddings are
    deterministic, so a sequence present in two stores has the same vector in
    both and it does not matter which one answers. First store wins.

    Exposes the same method names as MemoryMappedEmbeddingLoader
    (`get_embedding` / `get_embeddings_batch`) so MicrobiomeSparseDataset takes
    exactly the same code path it does for single-site runs — it selects the
    batch path via hasattr(loader, 'get_embeddings_batch').
    """

    def __init__(self, embedding_paths):
        self.loaders = {}
        self.owner = {}
        for name, path in embedding_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Embeddings for '{name}' not found at {path}. Generate them with "
                    f"the single-site script first — this script never creates embeddings."
                )
            print(f"Loading embeddings for {name}: {path}")
            loader = EmbeddingLoader(path, preload_all=False)
            self.loaders[name] = loader
            index = loader.sequence_index
            new = 0
            for seq_id in index:
                if seq_id not in self.owner:
                    self.owner[seq_id] = name
                    new += 1
            print(f"   {len(index)} sequences ({new} new, "
                  f"{len(index) - new} already seen in another cohort)")
        print(f"MergedEmbeddingLoader: {len(self.owner)} unique sequences total")

    def __contains__(self, seq_id):
        return seq_id in self.owner

    def __len__(self):
        return len(self.owner)

    def get_embedding(self, seq_id):
        owner = self.owner.get(seq_id)
        if owner is None:
            raise KeyError(f"No embedding for ASV {str(seq_id)[:40]}...")
        return self.loaders[owner].get_embedding(seq_id)

    def get_embeddings_batch(self, seq_ids):
        """Group lookups by owning store so each one keeps its fancy-indexed
        batch read rather than degrading to one read per sequence."""
        out = [None] * len(seq_ids)
        by_owner = {}
        for pos, seq_id in enumerate(seq_ids):
            owner = self.owner.get(seq_id)
            if owner is None:
                raise KeyError(f"No embedding for ASV {str(seq_id)[:40]}...")
            by_owner.setdefault(owner, []).append((pos, seq_id))

        for owner, items in by_owner.items():
            positions = [p for p, _ in items]
            ids = [s for _, s in items]
            for pos, emb in zip(positions, self.loaders[owner].get_embeddings_batch(ids)):
                out[pos] = emb
        return out


class CombinedAgeDatasetLoader:
    """Merges the three cohorts into one table / metadata frame / split.

    Exposes the same surface train_model() expects from AgeDatasetLoader:
    `train_dataset`, `val_dataset`, `test_dataset`, plus `get_test_dataset(seed)`
    for the repeated-subsample test evaluation.
    """

    def __init__(self, datasets=None, target_column="age", n_bins=50,
                 categorical_columns=None, fine_body_site=False,
                 enforce_subject_disjoint=True):
        """
        Args:
            fine_body_site: if True the body-site feature distinguishes the
                sub-sites recorded in metadata (skin head/hand, oral
                tongue/mouth) giving 5 categories; gut has no body_site column
                so it stays a single category either way. Default False = the
                coarse gut/skin/oral split.
            enforce_subject_disjoint: drop train/val samples whose subject also
                appears in a later split. REQUIRED for an honest test score —
                skin and oral sample the same people, so the naive union of the
                per-cohort splits leaks 18.7% of test and 24.8% of val. Only set
                False to reproduce the contaminated numbers deliberately.
        """
        self.datasets = datasets or DATASETS
        self.target_column = target_column
        self.n_bins = n_bins
        self.fine_body_site = fine_body_site
        self.enforce_subject_disjoint = enforce_subject_disjoint

        self.tables = {}
        metadata_frames = []

        for name, stem in self.datasets.items():
            biom_path = os.path.join(INPUT_DIR.format(name), stem + ".biom")
            meta_path = os.path.join(INPUT_DIR.format(name), stem + "_map.txt")

            table = biom.load_table(biom_path)
            self.tables[name] = table
            print(f"{name}: table {table.shape}")

            md = pd.read_csv(meta_path, sep="\t", index_col=0, low_memory=False)
            metadata_frames.append(self._slim_metadata(md, name))

        self.metadata_df = pd.concat(metadata_frames, axis=0)
        if self.metadata_df.index.duplicated().any():
            dupes = self.metadata_df.index[self.metadata_df.index.duplicated()][:5]
            raise ValueError(f"Duplicate sample ids across cohorts, e.g. {list(dupes)}")
        print(f"Combined metadata: {self.metadata_df.shape[0]} samples")

        # sample_id -> 'gut'/'skin'/'oral', used for the per-site metric breakdown
        self.sample_site = self.metadata_df["dataset_site"].to_dict()

        keys = list(self.datasets)
        self.table = self.tables[keys[0]].concat(
            [self.tables[k] for k in keys[1:]], axis="sample")
        print(f"Merged table (obs x samples): {self.table.shape}")

        embedding_paths = {
            name: os.path.join(OUTPUT_DIR, stem + "_embeddings.h5")
            for name, stem in self.datasets.items()
        }
        self.embedding_loader = MergedEmbeddingLoader(embedding_paths)

        self._drop_asvs_without_embeddings()

        # Categorical metadata maps are built over the *combined* frame so that
        # a category index means the same thing in every split.
        self.categorical_columns = categorical_columns or {"sex": "sex", "body_site": "body_site"}
        self.categorical_metadata = {}
        self.metadata_cardinalities = []
        for feature_name, column in self.categorical_columns.items():
            sample_to_idx, cat_to_idx = build_categorical_map(self.metadata_df[column])
            print(f"Metadata '{feature_name}' (column '{column}'): "
                  f"{len(cat_to_idx)} categories -> {cat_to_idx}")
            self.categorical_metadata[feature_name] = sample_to_idx
            self.metadata_cardinalities.append(len(cat_to_idx))

        self.json = "combined"

    def _slim_metadata(self, md, dataset_name):
        """Reduce a cohort's metadata to the handful of columns we actually use."""
        out = pd.DataFrame(index=md.index)
        out["age"] = pd.to_numeric(md["age"], errors="coerce")
        out["sex"] = md["sex"] if "sex" in md.columns else np.nan
        out["dataset_site"] = SITE_OF[dataset_name]

        # gut_4434_map.txt has no host_subject_id at all (9 columns total). Each
        # gut sample is treated as its own subject, which is the conservative
        # assumption for group-aware splitting.
        if "host_subject_id" in md.columns:
            out["host_subject_id"] = md["host_subject_id"].astype(str)
        else:
            out["host_subject_id"] = [f"{SITE_OF[dataset_name]}_{s}" for s in md.index]

        # Coarse feature is gut/skin/oral. Fine feature appends the sub-site
        # recorded in metadata (skin: head/hand, oral: UBERON:tongue/mouth).
        if self.fine_body_site and "body_site" in md.columns:
            sub = md["body_site"].astype(str).str.replace("UBERON:", "", regex=False)
            out["body_site"] = SITE_OF[dataset_name] + "_" + sub
        else:
            out["body_site"] = SITE_OF[dataset_name]

        out = out[out["age"].notna()]
        return out

    def _drop_asvs_without_embeddings(self):
        """Keep only ASVs the merged loader can actually serve.

        A missing embedding would otherwise surface as a KeyError deep inside a
        DataLoader worker, mid-epoch.
        """
        obs_ids = list(self.table.ids(axis="observation"))
        keep = [o for o in obs_ids if o in self.embedding_loader]
        missing = len(obs_ids) - len(keep)
        if missing:
            print(f"WARNING: {missing}/{len(obs_ids)} ASVs have no embedding — dropping them")
            self.table = self.table.filter(keep, axis="observation", inplace=False)
            print(f"Table after dropping un-embedded ASVs: {self.table.shape}")
        else:
            print(f"All {len(obs_ids)} ASVs have embeddings")

    def _split_indices(self):
        """Union the three per-cohort splits into one combined split.

        Each cohort keeps exactly the train/test assignment from its own
        group_assignments_*.json, and the same validation carve-out rule as
        run_exp_age (cached val_samples_*.txt if present, otherwise a
        GroupShuffleSplit on host_subject_id with random_state=42). Doing it
        per-cohort rather than globally means a subject never straddles
        train/val/test, and the combined split stays comparable to the
        single-site runs.
        """
        train_ids, val_ids, test_ids = [], [], []
        table_samples = set(self.table.ids(axis="sample"))

        for name in self.datasets:
            split_path = os.path.join(SPLIT_DIR, f"group_assignments_{name}.json")
            with open(split_path) as f:
                assignments = json.load(f)

            cohort_train = [s for s in assignments["train_indices"]
                            if s in table_samples and s in self.metadata_df.index]
            cohort_test = [s for s in assignments["test_indices"]
                           if s in table_samples and s in self.metadata_df.index]

            short = SITE_OF[name]
            val_file = os.path.join(SPLIT_DIR, f"val_samples_{short}.txt")
            if os.path.exists(val_file):
                with open(val_file) as f:
                    cohort_val = [line.strip() for line in f if line.strip()]
                cohort_val = [s for s in cohort_val if s in cohort_train]
                print(f"{name}: validation ids read from {os.path.basename(val_file)}")
            else:
                hosts = self.metadata_df.loc[cohort_train, "host_subject_id"]
                gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
                _, val_idx = next(gss.split(cohort_train, groups=hosts))
                cohort_val = [cohort_train[i] for i in val_idx]
                print(f"{name}: validation ids from GroupShuffleSplit on host_subject_id")

            # sorted() rather than set-difference ordering so the split is
            # reproducible across interpreter runs
            cohort_train = sorted(set(cohort_train) - set(cohort_val))
            cohort_val = sorted(set(cohort_val))
            cohort_test = sorted(set(cohort_test))

            print(f"{name}: train {len(cohort_train)}  val {len(cohort_val)}  test {len(cohort_test)}")

            # Subject-level leakage inside a cohort's own published split. This
            # is inherited from group_assignments_*.json, not introduced here,
            # but it inflates that cohort's test score so it needs to be visible.
            subj = self.metadata_df["host_subject_id"]
            shared = set(subj.loc[cohort_train]) & set(subj.loc[cohort_test])
            if shared:
                n_leaked = sum(1 for s in cohort_test if subj.loc[s] in shared)
                print(f"   WARNING {name}: {len(shared)} host_subject_id(s) in both "
                      f"train and test, affecting {n_leaked} test samples")

            train_ids += cohort_train
            val_ids += cohort_val
            test_ids += cohort_test

        overlap = (set(train_ids) & set(val_ids)) | (set(train_ids) & set(test_ids)) \
            | (set(val_ids) & set(test_ids))
        if overlap:
            raise ValueError(f"Split overlap detected ({len(overlap)} samples)")

        if self.enforce_subject_disjoint:
            train_ids, val_ids, test_ids = self._enforce_subject_disjoint(
                train_ids, val_ids, test_ids)

        subj_map = self.metadata_df["host_subject_id"].to_dict()
        leaked = {subj_map[s] for s in train_ids} & {subj_map[s] for s in test_ids}
        if leaked:
            print(f"WARNING: {len(leaked)} host_subject_id(s) still appear in both "
                  f"train and test")
        else:
            print("Subject-disjointness check: train / val / test share no subject")

        return train_ids, val_ids, test_ids

    def _enforce_subject_disjoint(self, train_ids, val_ids, test_ids):
        """Make the splits subject-disjoint *across* cohorts, not just within one.

        The skin and oral cohorts sample the same individuals — 102 of skin's 103
        age-labelled host_subject_ids also appear in oral, with identical age and
        sex. Each cohort's published split is internally clean, but unioning them
        puts a person's skin sample in train and their oral sample in test, which
        lets the model memorise that person's age instead of learning from their
        microbiome. Measured on the raw union: 18.7% of test and 24.8% of val
        samples belonged to a subject also present in train.

        Resolution order is test > val > train, so the test set stays exactly the
        union of the published per-cohort test sets and per-site numbers remain
        comparable to the single-site runs. Only training data is sacrificed.
        """
        subj_map = self.metadata_df["host_subject_id"].to_dict()
        n_before = (len(train_ids), len(val_ids), len(test_ids))

        test_subjects = {subj_map[s] for s in test_ids}
        train_ids = [s for s in train_ids if subj_map[s] not in test_subjects]
        val_ids = [s for s in val_ids if subj_map[s] not in test_subjects]

        val_subjects = {subj_map[s] for s in val_ids}
        train_ids = [s for s in train_ids if subj_map[s] not in val_subjects]

        dropped_train = n_before[0] - len(train_ids)
        dropped_val = n_before[1] - len(val_ids)
        print(f"Cross-cohort subject disjointness enforced: dropped {dropped_train} "
              f"train and {dropped_val} val samples (test untouched)")
        print(f"   train {n_before[0]} -> {len(train_ids)}, "
              f"val {n_before[1]} -> {len(val_ids)}, test {len(test_ids)}")
        return train_ids, val_ids, test_ids

    def _make_dataset(self, sample_ids, seed):
        split_table = self.table.filter(sample_ids, axis="sample", inplace=False)
        return MicrobiomeSparseDataset(
            biom_table=split_table,
            sample_targets=self.metadata_df.loc[sample_ids][self.target_column].to_dict(),
            sort_asvs=True,
            embedding_loader=self.embedding_loader,
            kmer_seqs=None,
            one_hot_seqs=None,
            seed=seed,
            n_bins=self.n_bins,
            subsample=True,
            categorical_metadata=self.categorical_metadata,
        )

    def get_dataset_by_json(self):
        self.train_ids, self.val_ids, self.test_ids = self._split_indices()
        print(f"\nCOMBINED  train {len(self.train_ids)}  val {len(self.val_ids)}  "
              f"test {len(self.test_ids)}")

        self.train_dataset = self._make_dataset(self.train_ids, seed=0)
        self.val_dataset = self._make_dataset(self.val_ids, seed=0)
        self.test_dataset = self._make_dataset(self.test_ids, seed=None)

        self.train_targets = self.metadata_df.loc[self.train_ids][self.target_column].to_list()
        self.val_targets = self.metadata_df.loc[self.val_ids][self.target_column].to_list()
        self.test_targets = self.metadata_df.loc[self.test_ids][self.target_column].to_list()

        print(f"Combined train age: mean {np.mean(self.train_targets):.2f}  "
              f"std {np.std(self.train_targets):.2f}")
        for site in sorted(set(self.sample_site.values())):
            ages = [self.metadata_df.loc[s, self.target_column]
                    for s in self.train_ids if self.sample_site[s] == site]
            print(f"   {site:5s}: n={len(ages):5d}  mean {np.mean(ages):.2f}  "
                  f"std {np.std(ages):.2f}")

    def get_test_dataset(self, seed=0):
        """Rebuild the test dataset under a new subsampling seed (used by the
        repeated test evaluation). Mirrors AgeDatasetLoader.get_test_dataset."""
        self.test_dataset = self._make_dataset(self.test_ids, seed=seed)


def evaluate_by_site(model, loader, device, output_dir, n_evals=10, batch_size=32):
    """Test-set evaluation with a per-body-site MAE breakdown.

    The pooled number over a mixture of cohorts with different age
    distributions is not comparable to a single-site MAE, so the per-site rows
    and the per-site mean-predictor baselines are the numbers to read.
    """
    model.eval()
    per_seed_overall = []
    per_seed_site = {}
    last_preds = last_targets = last_sites = None

    for k in range(n_evals):
        loader.get_test_dataset(seed=k)
        test_loader = DataLoader(loader.test_dataset, batch_size=batch_size,
                                 collate_fn=collate_fn, shuffle=False)

        preds, targets, sample_ids = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                embeddings = batch["embeddings"].to(device)
                abundances = _abundance_transform(batch["abundances"].to(device))
                masks = batch["masks"].to(device)
                labels = batch["label"].to(device)
                features = batch["features"].to(device) if batch["features"] is not None else None

                outputs, _, _, _, _ = model(embeddings, abundances, masks, features)
                preds.append(outputs.cpu() * LABEL_TO_YEARS)
                targets.append(labels.cpu() * LABEL_TO_YEARS)
                sample_ids.extend(batch["SampleID"])

        preds = torch.cat(preds).squeeze().numpy()
        targets = torch.cat(targets).squeeze().numpy()
        sites = np.array([loader.sample_site[s] for s in sample_ids])

        per_seed_overall.append(float(np.mean(np.abs(preds - targets))))
        for site in np.unique(sites):
            m = sites == site
            per_seed_site.setdefault(site, []).append(
                float(np.mean(np.abs(preds[m] - targets[m]))))

        last_preds, last_targets, last_sites = preds, targets, sites

    plot_true_vs_pred(last_targets, last_preds, epoch=None, out_dir=output_dir)

    # Per-site mean-predictor baseline, computed on TRAIN ages so it is an
    # honest baseline rather than an oracle.
    train_by_site = {}
    for sid, age in zip(loader.train_ids, loader.train_targets):
        train_by_site.setdefault(loader.sample_site[sid], []).append(age)

    results = {
        "overall_mae": float(np.mean(per_seed_overall)),
        "overall_mae_std": float(np.std(per_seed_overall)),
        "per_site": {},
    }

    print("\n================ TEST RESULTS ================")
    print(f"{'site':8s} {'n':>6s} {'model MAE':>12s} {'mean-baseline':>15s}")
    for site in sorted(per_seed_site):
        site_mae = float(np.mean(per_seed_site[site]))
        site_std = float(np.std(per_seed_site[site]))
        # Baseline is scored on the samples actually evaluated, not on
        # loader.test_ids — rarefaction drops a handful of samples that end up
        # empty, so the two counts differ slightly.
        m = last_sites == site
        site_test_ages = last_targets[m]
        baseline = mean_absolute_error(
            site_test_ages, np.full(int(m.sum()), np.mean(train_by_site[site])))
        results["per_site"][site] = {
            "mae": site_mae, "mae_std": site_std,
            "n_test": int(m.sum()), "mean_baseline_mae": float(baseline),
        }
        print(f"{site:8s} {int(m.sum()):6d} {site_mae:8.3f} ± {site_std:.3f} "
              f"{baseline:15.3f}")
    print(f"{'POOLED':8s} {len(last_targets):6d} "
          f"{results['overall_mae']:8.3f} ± {results['overall_mae_std']:.3f}")
    print("==============================================\n")
    return results


if __name__ == "__main__":
    target_column = "age"
    count_reg = 0.1
    num_epochs = 300
    batch_size = 32
    stratify_beta = None          # keep None to match the single-site baselines
    stratify_edges = (30, 40, 50, 60)
    fine_body_site = False        # True -> 5 site categories instead of 3

    model_config = {
        "input_dim": 256,
        "hidden_dim": 1024,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.3,
        "pe": False,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader = CombinedAgeDatasetLoader(
        datasets=DATASETS,
        target_column=target_column,
        categorical_columns={"sex": "sex", "body_site": "body_site"},
        fine_body_site=fine_body_site,
    )
    loader.get_dataset_by_json()

    print(f"\nTrain dataset size: {len(loader.train_dataset)}")
    print(f"Val dataset size:   {len(loader.val_dataset)}")
    print(f"Test dataset size:  {len(loader.test_dataset)}")
    print(f"Metadata cardinalities: {loader.metadata_cardinalities}")

    for seed in [1, 42, 123]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # One wandb run per seed. Sharing a single run across seeds (as the
        # single-site script does) concatenates three independent trainings onto
        # one step axis and makes the loss curve look like it collapses and
        # recovers.
        run = wandb.init(
            project="microbiome-age-prediction",
            name=f"combined-3site-seed{seed}",
            group="combined_3site",
            config={**model_config, "seed": seed, "sites": list(DATASETS),
                    "count_reg": count_reg, "stratify_beta": stratify_beta,
                    "fine_body_site": fine_body_site},
            reinit=True,
        )

        model = AMBERRegressor(
            input_dim=model_config["input_dim"],
            hidden_dim=model_config["hidden_dim"],
            num_heads=model_config["num_heads"],
            num_layers=model_config["num_layers"],
            dropout=model_config["dropout"],
            pe=model_config["pe"],
            grl=False,
            metadata_cardinalities=loader.metadata_cardinalities,
        )
        model.reset_parameters()
        model = model.to(device)

        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = torch.nn.SmoothL1Loss(beta=0.3)

        out_dir = os.path.join(
            PROJECT_ROOT,
            f"microbiome_model/results/age_prediction/combined_3site/seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        res = train_model(model, criterion, optimizer, loader, device,
                          num_epochs=num_epochs, out_dir=out_dir, count_reg=count_reg,
                          stratify_beta=stratify_beta, stratify_edges=stratify_edges,
                          batch_size=batch_size)

        # Fail loudly instead of silently scoring a stale checkpoint left in
        # out_dir by an earlier run.
        if res["best_model"] is None or not np.isfinite(res["best_loss"]):
            raise RuntimeError(
                f"seed {seed}: training never produced a checkpoint "
                f"(best_loss={res['best_loss']}). Validation loss was non-finite — "
                f"no results will be written.")

        print(f"seed={seed}: best val MAE={res['best_mae']:.4f} yrs")
        model.load_state_dict(torch.load(res["best_model"], map_location=device))

        test_results = evaluate_by_site(model, loader, device, out_dir)

        summary = {
            "experiment": "combined_3site",
            "sites": list(DATASETS),
            "seed": seed,
            "units": "years",
            "count_reg": count_reg,
            "stratify_beta": stratify_beta,
            "stratify_edges": list(stratify_edges) if stratify_beta is not None else None,
            "fine_body_site": fine_body_site,
            "metadata_features": list(loader.categorical_columns),
            "metadata_cardinalities": loader.metadata_cardinalities,
            "n_train": len(loader.train_ids),
            "n_val": len(loader.val_ids),
            "n_test": len(loader.test_ids),
            "best_val_loss": res["best_loss"],
            "best_val_mae": res["best_mae"],
            "test_mae": test_results["overall_mae"],
            "test_mae_std": test_results["overall_mae_std"],
            "test_per_site": test_results["per_site"],
            "best_model": res["best_model"],
        }
        with open(os.path.join(out_dir, "results.json"), "w") as f:
            json.dump(summary, f, indent=2)

        wandb.log({
            "test/pooled_mae": test_results["overall_mae"],
            **{f"test/{site}_mae": v["mae"] for site, v in test_results["per_site"].items()},
        })
        run.finish()
