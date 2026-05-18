import os
import numpy as np
import pandas as pd
import biom
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def donor_of(sample_id, meta_data, delimiter="."):
    """Extract donor ID from a sample ID. Adjust delimiter to your format."""
    sample_to_donor = dict(zip(meta_data["SampleID"], meta_data["DonorID"]))
    return sample_to_donor.get(sample_id, None)


def build_feature_matrix(table, sample_ids, metadata_df=None,
                         use_env=False, use_bimonth=False):
    """
    Returns X aligned row-for-row with `sample_ids`.
    Abundance features + optional metadata columns.
    """
    # Abundance matrix, explicitly reindexed to the requested sample order
    abund = table.to_dataframe(dense=True).T          # [samples, asvs]
    abund = abund.reindex(sample_ids)                  # enforce exact order
    assert not abund.isnull().any().any(), "Missing samples after reindex"

    X = abund.values.astype(np.float32)

    extra = []
    if use_env or use_bimonth:
        md = metadata_df.set_index("SampleID").reindex(sample_ids)

        if use_env:
            env = (md["environment"] == "outdoor").astype(int).values.reshape(-1, 1)
            extra.append(env)

        if use_bimonth:
            bimonth_oh = pd.get_dummies(md["bimonth"], prefix="bm")
            # Ensure all 6 categories always present (consistent columns across folds)
            for b in range(1, 7):
                col = f"bm_{b}"
                if col not in bimonth_oh.columns:
                    bimonth_oh[col] = 0
            bimonth_oh = bimonth_oh[[f"bm_{b}" for b in range(1, 7)]]
            extra.append(bimonth_oh.values)

    if extra:
        X = np.hstack([X] + extra)

    return X


def load_data(biom_file_path, target_df, rarefaction_depth=5000, seed=42):
    """Load table once, rarefy once (deterministically), build target map."""
    table = biom.load_table(biom_file_path)
    table = table.filter(lambda v, i, m: v.sum() > 0, axis="sample")

    # Deterministic single rarefaction shared across all folds.
    # If you want to match SeqMiT's per-epoch rarefaction, that's a different
    # design choice — but for a fair, reproducible RF baseline, fix it once.
    np.random.seed(seed)
    table = table.subsample(rarefaction_depth, axis="sample", with_replacement=True)

    target_map = dict(zip(target_df["SampleID"], target_df["outdoor_add_0"]))

    # Keep only samples present in both table and targets
    table_ids = set(table.ids(axis="sample"))
    valid_ids = [sid for sid in target_map if sid in table_ids]

    return table, target_map, valid_ids


# ---------------------------------------------------------------------------
# Nested LOBO CV
# ---------------------------------------------------------------------------

def nested_lobo_cv(biom_file_path, target_df, output_dir="rf_results",
                   metadata_path=None, use_env=False, use_bimonth=False,
                   rarefaction_depth=5000, n_inner_folds=4, seed=42):

    os.makedirs(output_dir, exist_ok=True)

    table, target_map, valid_ids = load_data(
        biom_file_path, target_df, rarefaction_depth, seed
    )

    metadata_df = pd.read_csv(metadata_path) if (use_env or use_bimonth) else None

    # Donor ID per sample (exact match, not substring)
    donors = np.array([donor_of(sid, target_df) for sid in valid_ids])
    unique_donors = sorted(set(donors))
    print(f"{len(valid_ids)} samples across {len(unique_donors)} donors")

    param_grid = {
        "max_depth": [None, 8, 16],
        "max_features": ["sqrt", 0.2, 1.0],
        "min_samples_leaf": [1, 3],
    }

    results = {}
    print(unique_donors)

    # ---- Outer loop: leave one donor out ----
    for held_donor in unique_donors:
        test_mask = donors == held_donor
        train_mask = ~test_mask

        test_ids = [sid for sid, m in zip(valid_ids, test_mask) if m]
        train_ids = [sid for sid, m in zip(valid_ids, train_mask) if m]

        if len(test_ids) == 0:
            print(f"Skipping {held_donor}: no test samples")
            continue

        X_train = build_feature_matrix(table, train_ids, metadata_df,
                                       use_env, use_bimonth)
        X_test = build_feature_matrix(table, test_ids, metadata_df,
                                      use_env, use_bimonth)

        y_train = np.array([target_map[s] for s in train_ids]) / 100.0
        y_test = np.array([target_map[s] for s in test_ids]) / 100.0

        # ---- Inner loop: group-aware CV over training donors only ----
        train_donor_labels = np.array([donor_of(s, target_df) for s in train_ids])
        n_train_donors = len(set(train_donor_labels))
        inner_splits = min(n_inner_folds, n_train_donors)

        inner_cv = GroupKFold(n_splits=inner_splits)

        rf = RandomForestRegressor(
            n_estimators=500,
            criterion="absolute_error",
            random_state=seed,
            n_jobs=-1,
        )

        grid = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=inner_cv.split(X_train, y_train, groups=train_donor_labels),
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            refit=True,
        )

        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        mae = mean_absolute_error(y_test * 100, y_pred * 100)

        results[held_donor] = {
            "mae": float(mae),
            "n_test": len(test_ids),
            "best_params": grid.best_params_,
        }
        print(f"{held_donor:>5} | MAE={mae:7.2f} | n_test={len(test_ids):3d} "
              f"| {grid.best_params_}")

    # ---- Summary ----
    maes = np.array([r["mae"] for r in results.values()])
    print("\n" + "=" * 50)
    print(f"LOBO mean MAE : {maes.mean():.2f} ± {maes.std():.2f}")
    print(f"Median MAE    : {np.median(maes):.2f}")
    print(f"Worst donor   : {max(results, key=lambda k: results[k]['mae'])} "
          f"({maes.max():.2f})")
    print(f"Best donor    : {min(results, key=lambda k: results[k]['mae'])} "
          f"({maes.min():.2f})")

    results_df = pd.DataFrame([
        {"donor": k, "mae": v["mae"], "n_test": v["n_test"],
         "best_params": str(v["best_params"])}
        for k, v in results.items()
    ])
    results_df.to_csv(os.path.join(output_dir, "lobo_results.csv"), index=False)

    return results, results_df


if __name__ == "__main__":
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
    
    biom_file_sheds = os.path.join(data_dir, "data/table_sheds_dada2.biom")
    sheds_file = os.path.join(data_dir, "data/new_data/metadata_sheds.tsv")
    sheds_df = pd.read_csv(sheds_file, sep="\t")
    #sheds_df.index = "13810."+ sheds_df.index.astype(str) 
    print(sheds_df.head())

    nested_lobo_cv(
        biom_file_path=biom_file_sheds,
        target_df=sheds_df,
        output_dir=os.path.join(data_dir, "microbiome_mode/results/rf_results_abundance"),
        use_env=False, use_bimonth=False,
    )

    # --- RF + Env ---
    # nested_lobo_cv(..., metadata_path="../data/metadata.csv",
    #                use_env=True, use_bimonth=False,
    #                output_dir="rf_results_env")

    # --- RF + Env & Bimonth ---
    # nested_lobo_cv(..., metadata_path="../data/metadata.csv",
    #                use_env=True, use_bimonth=True,
    #                output_dir="rf_results_env_bimonth")