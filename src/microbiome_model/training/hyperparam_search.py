"""
Hyperparameter search using Optuna for microbiome regression model.
Searches over model architecture, training params, and GRL on/off.
"""

import os
import json
import random
import logging
import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# ── your existing imports ──
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# ---- project imports ----
from microbiome_model.models import build_model
from microbiome_model.models.orig import BasicRegressor, BasicRegressorwithUnifrac, BasicRegressorNew
from microbiome_model.models.zoo import GeneralizedRegressor, BasicRegressorGRL, CLR, ClusteredRegressor, SoftADDContrastiveLoss, ContrastiveEncoder
from microbiome_model.data.dataset import Arguments, DataProcessor
from microbiome_model.losses.core import compute_count_loss, PairwiseLoss
from microbiome_model.eval.evaluate import predict
from microbiome_model.utils.misc import _mean_absolute_error, compute_diversity_indices, create_augmented_batch_DDD 
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn, DonorAwareSampler
from microbiome_model.training.train_orig import train_model, cleanup_resources



def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except:
        pass


def create_objective(
    biom_file, metadata_file, embedding_path, heldout,
    device, output_dir, data_dir
):
    """
    Returns an Optuna objective function closed over the fixed data paths.
    """

    def objective(trial):
        # ─── Sample hyperparameters ───
        grl = trial.suggest_categorical("grl", [True, False])
        input_dim = trial.suggest_categorical("input_dim", [128, 256, 512])
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.05)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        reg_count_loss = trial.suggest_float("reg_count_loss", 0.01, 1.0, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        pe = trial.suggest_categorical("pe", [True, False])
        grl_loss_weight = trial.suggest_categorical("grl_loss_weight", [0.01,0.1, 0.5, 1.0]) if grl else 0.0

        seed = 42
        setup_seed(seed)

        # ─── Data loading ───
        args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=heldout,
            sort_asvs=True,
        )
        data_processor = DataProcessor(args)
        data_processor.load_data(multitask=True, column="bi_month_name")

        train_donors = set(
            data_processor.sampleID_map_donorID[sid]
            for sid in data_processor.train_data[0]
        )

        # ─── Model config ───
        model_config = {
            "model_name": "BasicRegressor",
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dropout": dropout,
            "pe": pe,
            "multitask": True,
            "data": "all_data",
            "grl": grl,
            "num_donors": len(train_donors),
            "pretrained_weights": os.path.join(
                data_dir,
                "microbiome_model/results/MaskedPretrainSkin/pretrained_best.pt",
            ),
        }

        model = build_model(model_config, str(device))

        # ─── Trial output dir ───
        trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_dir, exist_ok=True)

        # Save configs for reproducibility
        trial_params = {
            **model_config,
            "learning_rate": learning_rate,
            "reg_count_loss": reg_count_loss,
            "batch_size": batch_size,
            "grl_loss_weight": grl_loss_weight,
            "seed": seed,
        }
        training_config = {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "reg_count_loss": reg_count_loss,
                "early_stop_warmup": 50,
                "patience": 30,
                "num_epochs": 300,
                "use_unifrac_loss": False,
                "scale_abundance": True
            }
        with open(os.path.join(trial_dir, "trial_params.json"), "w") as f:
            json.dump(trial_params, f, indent=4)

        # ─── Train ───
        try:
            res = train_model(
                model=model,
                train_config=training_config,
                device=device,
                out_dir=trial_dir,
                data_processor=data_processor,
                use_unifrac_loss=False,
                scale_abundance=True,
                grl=grl,
            )
            val_mae = res["best_mae"]

        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()  # This will show the FULL error
            val_mae = float("inf")

        finally:
            del model
            cleanup_resources()

        # Save result
        with open(os.path.join(trial_dir, "result.json"), "w") as f:
            json.dump({"val_mae": val_mae, **trial_params}, f, indent=4)

        return val_mae

    return objective


def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ─── Paths ───
    data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
    biom_file = os.path.join(data_dir, "process_data_all/exported-feature-table/feature-table.biom")
    metadata_file = os.path.join(data_dir, "data/new_data/combined_metadata_merged.tsv")
    embedding_path = os.path.join(data_dir, "data/embeddings/all_data.h5")

    heldout = "D15"
    experiment_name = "hyperparam_search_D15_50trials"
    output_dir = os.path.join(data_dir, f"microbiome_model/results/{experiment_name}/")
    os.makedirs(output_dir, exist_ok=True)

    # ─── Optuna study ───
    study = optuna.create_study(
        study_name=experiment_name,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        storage=f"sqlite:///{output_dir}/optuna.db",
        load_if_exists=True,
    )

    objective = create_objective(
        biom_file=biom_file,
        metadata_file=metadata_file,
        embedding_path=embedding_path,
        heldout=heldout,
        device=device,
        output_dir=output_dir,
        data_dir=data_dir,
    )

    study.optimize(objective, n_trials=50, gc_after_trial=True)

    # ─── Report best ───
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    best = study.best_trial
    print(f"  Val MAE: {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save summary
    summary = {
        "best_val_mae": best.value,
        "best_params": best.params,
        "best_trial_number": best.number,
        "n_trials": len(study.trials),
    }
    with open(os.path.join(output_dir, "best_trial_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Save all trials as CSV
    df = study.trials_dataframe()
    df.to_csv(os.path.join(output_dir, "all_trials.csv"), index=False)

    # ─── Retrain best config with 3 seeds, full epochs ───
    print("\n" + "=" * 60)
    print("RETRAINING BEST CONFIG WITH MULTIPLE SEEDS")
    print("=" * 60)

    best_params = best.params
    seed_values = [42, 1337, 2048]

    for i, seed in enumerate(seed_values):
        print(f"\nRetrain run {i+1}/3, seed={seed}")
        setup_seed(seed)
        cleanup_resources()

        args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=heldout,
            sort_asvs=True,
        )
        data_processor = DataProcessor(args)
        data_processor.load_data(multitask=True, column="bi_month_name")

        train_donors = set(
            data_processor.sampleID_map_donorID[sid]
            for sid in data_processor.train_data[0]
        )

        model_config = {
            "model_name": "BasicRegressor",
            "input_dim": 256,
            "hidden_dim": best_params["hidden_dim"],
            "num_heads": best_params["num_heads"],
            "num_layers": best_params["num_layers"],
            "dropout": best_params["dropout"],
            "pe": best_params["pe"],
            "multitask": True,
            "data": "all_data",
            "grl": best_params["grl"],
            "num_donors": len(train_donors),
            "pretrained_weights": os.path.join(
                data_dir,
                "microbiome_model/results/MaskedPretrainSkin/pretrained_best.pt",
            ),
        }

        model = build_model(model_config, str(device))

        result_dir = os.path.join(output_dir, "best_retrain", f"seed_{seed}")
        os.makedirs(result_dir, exist_ok=True)

        with open(os.path.join(result_dir, "config.json"), "w") as f:
            json.dump({**model_config, **best_params, "seed": seed}, f, indent=4)

        res = train_model(
            model=model,
            train_config={
                "batch_size": best_params["batch_size"],
                "learning_rate": best_params["learning_rate"],
                "reg_count_loss": best_params["reg_count_loss"],
                "early_stop_warmup": 50,
                "patience": 30,
                "num_epochs": 300,
                "grl_loss_weight": best_params["grl_loss_weight"]
            },
            device=device,
            out_dir=result_dir,
            data_processor=data_processor,
            use_unifrac_loss=False,
            scale_abundance=True,
            grl=best_params["grl"],
        )

        res["seed"] = seed
        with open(os.path.join(result_dir, "result.json"), "w") as f:
            json.dump(res, f, indent=4)

        print(f"  Seed {seed} -> best_mae: {res['best_mae']:.4f}")
        del model
        cleanup_resources()


if __name__ == "__main__":
    main()