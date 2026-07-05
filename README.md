# AMBER – Abundance-Modulated Bi-EncodeR

## Setup

```bash
conda env create -f environment.yml
conda activate microbiome_model
```

This installs the pinned dependencies and the `microbiome_model` package itself
(editable install, from `pyproject.toml`).

Note: `src/microbiome_model/baselines/rf.py` additionally depends on `qiime2`,
which is not pip/conda-forge installable and requires a separate QIIME 2
environment; it is not covered by `environment.yml`.

## Training

```bash
python src/microbiome_model/training/train_orig.py
```

See `job.sh` for an example SLURM submission.
