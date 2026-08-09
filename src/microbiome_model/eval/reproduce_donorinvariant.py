"""Reproduce the original DonorInvariant evaluation (evaluation_results_0, MAE 44.21).

evaluate_grl.py can no longer produce this number for three independent reasons,
all of which fail silently:

  1. best_run_dir is assigned three times in its __main__ (lines ~356-358) and the
     LAST assignment wins, so it points at multitask_all_nope_bimonth_env_features
     rather than DonorInvariant. The file was edited 12 minutes after
     evaluation_results/ was written.

  2. It builds BasicRegressorBest(128, 1024, 4, 2, 0.3, pe=False) (line ~226).
     The DonorInvariant checkpoints are the commented-out line above it:
     BasicRegressorGRL(256, 512, 4, 2, 0.3, pe=True, num_subjects=106).
     Verified: GRL loads with strict=True; Best fails with 87 shape mismatches.

  3. THE ONE THAT CHANGES THE NUMBERS. dataset.py's `bi_month_name` map used to
     emit 6-dim one-hot vectors (that block is still there, commented out) and now
     emits scalar indices 0-5. BasicRegressorGRL still declares
     `gating_feature = nn.Linear(6, input_dim//2)` and feeds `features` into it
     raw. The weight shape (128, 6) stays valid either way, so nothing errors --
     the gate just computes something completely different from what was trained.
     BasicRegressorBest was later patched to re-derive the one-hot
     (F.one_hot(features[:, 1:], num_classes=6)); BasicRegressorGRL never was.

This script leaves evaluate_grl.py and dataset.py untouched and applies all three
corrections locally, then diffs against evaluation_results_0/predictions.csv,
which is the authoritative record of what the original run produced.

Usage:
    python -m microbiome_model.eval.reproduce_donorinvariant            # all donors
    python -m microbiome_model.eval.reproduce_donorinvariant D4 D5      # subset
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from microbiome_model.data.dataset import Arguments, DataProcessor
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn
from microbiome_model.eval.evaluate_grl import predict
from microbiome_model.models.zoo import BasicRegressorGRL

BASE = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
BIOM_FILE = os.path.join(BASE, "process_data_all/exported-feature-table/feature-table.biom")
METADATA_FILE = os.path.join(BASE, "data/new_data/combined_metadata_merged.tsv")
EMBEDDING_PATH = os.path.join(BASE, "data/embeddings/all_data.h5")

# FIX 1: the experiment these checkpoints actually belong to.
EXPERIMENT_DIR = os.path.join(BASE, "microbiome_model/results/DonorInvariant")
REFERENCE_CSV = os.path.join(EXPERIMENT_DIR, "evaluation_results_0", "predictions.csv")

ALL_DONORS = [f"D{i}" for i in range(4, 31)]
N_BIMONTH = 6

# FIX 4 (found empirically).  `subsample=True` calls
# biom Table.subsample(5000, axis="observation"), which on this table keeps only
# the ~482 of 755,658 observations whose total count reaches 5000 -- i.e. it
# discards 99.94% of the feature table -- and then drops any sample left empty.
# That is what shrinks the test set from 1031 to 932 samples.
#
#   subsample=False -> reproduces evaluation_results/ (April) EXACTLY.
#   subsample=True  -> reproduces evaluation_results_0/'s sample COUNTS exactly,
#                      but its MAEs only for some donors, because the surviving
#                      482-ASV draw swings a single donor's MAE by 5-11 years
#                      depending on seed. evaluation_results_0's 44.21 is one
#                      sample from that distribution, not a stable quantity.
SUBSAMPLE = False


def build_model(device):
    """FIX 2: the architecture the checkpoints were saved from."""
    return BasicRegressorGRL(256, 512, 4, 2, 0.3, pe=True, num_subjects=106).to(device)


def restore_onehot_metadata(data_processor):
    """FIX 3: re-expand bi_month_name indices back to the 6-dim one-hot the
    checkpoints were trained with.

    Must run after load_data() and before sample_test_data(), which passes
    self.sample_seasons through as `sample_metadata`.
    """
    seasons = data_processor.sample_seasons
    sample = next(iter(seasons.values()))
    if np.ndim(sample) > 0 and np.shape(sample)[-1] == N_BIMONTH:
        print("  metadata already one-hot; leaving as is")
        return 0
    eye = np.eye(N_BIMONTH, dtype=np.float32)
    data_processor.sample_seasons = {k: eye[int(v)] for k, v in seasons.items()}
    return len(seasons)


def best_run_for(donor_id, runs=3):
    best_mae, best_dir = float("inf"), None
    for i in range(runs):
        run_dir = os.path.join(EXPERIMENT_DIR, donor_id, f"run_{i + 1}")
        res_path = os.path.join(run_dir, "res.json")
        if not os.path.exists(res_path):
            continue
        with open(res_path) as f:
            res = json.load(f)
        if res["best_mae"] < best_mae:
            best_mae, best_dir = res["best_mae"], run_dir
    return best_dir, best_mae


def evaluate_donor(donor_id, device):
    best_dir, best_mae = best_run_for(donor_id)
    if best_dir is None:
        print(f"{donor_id}: no res.json found, skipping")
        return None

    model = build_model(device)
    model.load_state_dict(torch.load(os.path.join(best_dir, "model.pt"), map_location=device))
    model.eval()

    args = Arguments(biom_file=BIOM_FILE, metadata_file=METADATA_FILE, tree_path=None,
                     embedding_file=EMBEDDING_PATH, embedding="DNABERT",
                     heldout=donor_id, sort_asvs=True)
    dp = DataProcessor(args)
    dp.load_data(multitask=True, column="bi_month_name")
    restore_onehot_metadata(dp)

    test_dataset = dp.sample_test_data(subsample=SUBSAMPLE)
    if len(test_dataset) == 0:
        print(f"{donor_id}: no test data")
        return None

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False,
                                         num_workers=0, collate_fn=sparse_collate_fn)
    labels, preds, sample_ids = predict(model, loader, device)

    # evaluate_grl.py applies np.abs() to BOTH before scaling by 100. Kept here so
    # the comparison is like-for-like; `raw` shows what it would be without it.
    return pd.DataFrame({
        "SampleID": sample_ids,
        "DonorID": donor_id,
        "Actual": np.abs(labels) * 100,
        "Predicted": np.abs(preds) * 100,
        "Predicted_raw": preds * 100,
        "val_mae": best_mae,
    })


def main(donors):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nExperiment: {EXPERIMENT_DIR}\nDonors: {len(donors)}\n")

    frames = []
    for donor_id in donors:
        print(f"--- {donor_id}")
        df = evaluate_donor(donor_id, device)
        if df is not None:
            mae = np.mean(np.abs(df.Actual - df.Predicted))
            print(f"  n={len(df)}  MAE={mae:.3f}")
            frames.append(df)
    if not frames:
        print("nothing evaluated")
        return
    got = pd.concat(frames, ignore_index=True)

    # Compare per-donor MAE, not per-sample: evaluation_results_0 has no SampleID
    # column and its `Actual` values repeat within a donor, so a (donor, actual)
    # join is not unique and silently mismatches rows.
    mine = got.groupby("DonorID").apply(
        lambda d: float(np.mean(np.abs(d.Actual - d.Predicted))))
    counts = got.groupby("DonorID").size()

    print("\n" + "=" * 78)
    print(f"subsample={SUBSAMPLE}")
    print(f"{'donor':6s} {'n':>4s} {'this run':>10s} | {'eval_0':>9s} {'diff':>7s} | "
          f"{'April':>9s} {'diff':>7s}")
    refs = {}
    for tag, name in (("eval_0", "evaluation_results_0"), ("april", "evaluation_results")):
        with open(os.path.join(EXPERIMENT_DIR, name, "per_donor_maes.json")) as f:
            refs[tag] = json.load(f)

    hits = {"eval_0": 0, "april": 0}
    for donor in sorted(mine.index, key=lambda d: int(d[1:])):
        m = mine[donor]
        r0, ra = refs["eval_0"].get(donor), refs["april"].get(donor)
        d0, da = abs(m - r0), abs(m - ra)
        hits["eval_0"] += d0 < 0.01
        hits["april"] += da < 0.01
        print(f"{donor:6s} {counts[donor]:4d} {m:10.4f} | {r0:9.4f} {d0:7.4f} | "
              f"{ra:9.4f} {da:7.4f}")

    pooled = float(np.mean(np.abs(got.Actual - got.Predicted)))
    print(f"\n{'POOLED':6s} {len(got):4d} {pooled:10.4f} | "
          f"{refs['eval_0']['mean_absolute_error']:9.4f} "
          f"{abs(pooled - refs['eval_0']['mean_absolute_error']):7.4f} | "
          f"{refs['april']['mean_absolute_error']:9.4f} "
          f"{abs(pooled - refs['april']['mean_absolute_error']):7.4f}")
    n = len(mine)
    print(f"\ndonors matching within 0.01 -- eval_0: {hits['eval_0']}/{n}   "
          f"April: {hits['april']}/{n}")

    out = os.path.join(EXPERIMENT_DIR, "evaluation_results_reproduced")
    os.makedirs(out, exist_ok=True)
    got.to_csv(os.path.join(out, "predictions.csv"), index=False)
    per_donor = {d: float(v) for d, v in mine.items()}
    per_donor["mean_absolute_error"] = pooled
    per_donor["_subsample"] = SUBSAMPLE
    with open(os.path.join(out, "per_donor_maes.json"), "w") as f:
        json.dump(per_donor, f, indent=4)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ALL_DONORS)
