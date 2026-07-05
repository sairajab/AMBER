import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH (so `import microbiome_model...` works)
ROOT = Path(__file__).resolve().parents[3]  # .../microbiome_model (project root)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import re
import json
import shutil
from glob import glob
from datetime import datetime
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt

import seaborn as sns

# ---- project imports ----
from microbiome_model.models import build_model
from microbiome_model.models.zoo import SampleLevelRegressor, GeneralizedRegressor
from microbiome_model.models.orig import BasicRegressor, BasicRegressorwithUnifrac
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn
from microbiome_model.data.dataset import DataProcessor, Arguments
from microbiome_model.utils.misc import _mean_absolute_error
from microbiome_model.training.pre_training_masked import MaskedAbundancePretraining

def build_features(batch, meta_spec, device):
    """Build [B, n_meta] integer feature tensor. None if meta_spec empty.
    MUST match the training-time builder exactly."""
    if not meta_spec:
        return None
    cols = []
    for name, _card in meta_spec:
        if name == "env":
            cols.append(batch['env'].to(device).long())
        elif name == "bimonth":
            cols.append(batch['season'].to(device).long())
        else:
            raise ValueError(f"Unknown metadata feature: {name}")
    return torch.stack(cols, dim=1)

def predict(model, loader, device, scale_abundances=False, val=False,
            multitask=False, meta_spec=None):
    """Generate predictions for a dataset"""

        #### D20 unique features
    with open("../D12_unique_features.txt", "r") as f:
        d20_unique_features = [line.strip() for line in f]
        d20_unique_features = set(d20_unique_features)
        
    model = model.to(device)
    model.eval()
    predictions = []
    labels = []
    criterion = nn.MSELoss()
    test_loss = 0
    all_ids = []
    per_sample_preds = {}
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(device, dtype=torch.float32)
            abundances = batch['abundances'].to(device) / 1e4 if scale_abundances else batch['abundances'].to(device)
            masks = batch['masks'].to(device)
            targets = batch['outdoor_add_0'].to(device)
            sample_ids = batch["SampleID"]
            all_ids.extend(batch["SampleID"])
            env_idx = batch['env'].to(device, dtype=torch.float32)  # 0 for indoor, 1 for outdoor
            seqids = batch['seqs_ids']  # list of lists of ASV IDs for each sample in the batch
            features = build_features(batch, meta_spec, device)

            # print("Num non-zero ASVs per sample:", (abundances > 0).sum(-1).float().mean())
            # print("Top-1 abundance value:", abundances.max(-1).values.mean())

            # for i in range(embeddings.shape[0]):
            #     seqs = seqids[i]  # list of ASV IDs for this sample
            #     j = []
            #     k =0
            #     for s in seqs:
            #         if s in d20_unique_features:
            #             j.append(k)
            #         k += 1
            #     # set masks for unique features to zero to see if model relies on them
            #     # mask before
            #     print(f"Before masking, sample {sample_ids[i]} has {masks[i].sum().item()} ASVs present.")
            #     for idx in j:
            #         masks[i, idx] = 0
            #     print(f"After masking D20 unique features, sample {sample_ids[i]} has {masks[i].sum().item()} ASVs present.")
                

            

            # if d20_unique_features is not None:
            #     for i in range(embeddings.shape[0]):
            #         if sample_ids[i] == "13810.SKF.SHED.D12.2020.T12":
            #             print(abundances[i])
            #         seqs = seqids[i]
            #         for k, s in enumerate(seqs):
            #             if s in d20_unique_features:
            #                 masks[i, k] = 0
            #                 embeddings[i, k] = 0
            #                 abundances[i, k] = 0
            
            # valid_mask = masks[i].bool()
            # valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            # valid_abundances = abundances[i, valid_indices, 0]
            # sorted_order = torch.argsort(valid_abundances, descending=True)
            
            # # reorder embeddings, abundances, masks
            # embeddings[i, :len(valid_indices)] = embeddings[i, valid_indices[sorted_order]]
            # abundances[i, :len(valid_indices)] = abundances[i, valid_indices[sorted_order]]
            # masks[i, :len(valid_indices)] = 1
            # masks[i, len(valid_indices):] = 0  # push masked positions to end



            outputss = model(embeddings, abundances, masks, features)
            outputs = outputss[0]
            if multitask:
                p_indoor = outputs[0]
                p_outdoor = outputs[1]
                mask_in = (env_idx == 0).view(-1)
                mask_out = (env_idx == 1).view(-1)
                all_outputs = torch.cat([p_indoor[mask_in], p_outdoor[mask_out]])
                all_targets = torch.cat([targets[mask_in], targets[mask_out]])
                outputs = all_outputs
                targets = all_targets
            
            if val:
                for i in range(targets.shape[0]):
                    seqs = seqids[i]  # list of ASV IDs for this sample
                    uniq_feats = 0
                    for s in seqs:
                        if s in d20_unique_features:
                            uniq_feats += 1
                        

                    per_sample_preds[sample_ids[i]] = [float(outputs[i].cpu()), float(targets[i].cpu()), uniq_feats]
                
            # loss = criterion(outputs, targets)
            # test_loss += loss.item()        
            labels.append(targets.cpu().numpy().reshape(-1))
            predictions.append(outputs.cpu().numpy().reshape(-1))
            
    # test_loss /= len(loader)
    
    # print("Loss ", test_loss)

    print("Unique SampleIDs in predict():", len(set(all_ids)))
    print("Expected SampleIDs:", len(loader)) 
    return np.concatenate(labels), np.concatenate(predictions), per_sample_preds

def get_residual_plot(predictions, labels, fname):
    # Residuals
    residuals = (np.array(predictions) - np.array(labels)) / 100.0  # Scale back to original values
    y_pred = np.array(predictions) / 100.0  # Scale back to original values
    # Option 1: residuals vs prediction
    plt.figure()
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals (y_pred - y_true)")
    plt.title("Residual Plot")
    plt.savefig(fname)
    plt.show()
        
def evaluate_asv_encoder(donor_ids, res="results/", find_best = True):
    train_asv_encoder = True
    one_hot_seqs = None
    embedding_path = None
    eval_runs = 3
    all_labels = []
    all_predictions = []
    results = defaultdict(float)
    mean_mae = 0
    for donor_id in donor_ids: 
        for _ in range(eval_runs):
            abundance_table, train_data , test_data, one_hot_seqs, distances = load_data(heldout = donor_id, train_encoder=True)
            test_loader = sample_test_data(abundance_table, test_data, embedding_path = embedding_path, one_hot_seqs=one_hot_seqs, random_vector=random_vec) 
            runs = 2   
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            best_mae = 1000
            best_i = 0
            for i in range(1, runs+1):
                dir = f"{res}/{donor_id}/run_{i}/"
                with open(dir + 'res.json') as f:
                    config = json.load(f)
                    mae = float(config['best_mae'])
                    print(f"MAE for {donor_id} : {mae}")
                if mae < best_mae:
                    best_mae = mae
                    best_run = dir
                    best_i = i                

            model_files = best_run + "model.pt"
            print(f"Best run: {best_run}", best_mae)
            print(f"Done for {donor_id} : {best_mae}")
            model = SampleLevelRegressor(use_nt_encoder=train_asv_encoder, pe=True)

            ## print model parameters
            print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
            print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
            model.load_state_dict(torch.load(model_files))
            #print(model)
            labels, predictions = predict(model, test_loader , device)
                
            all_labels.extend(labels*100)
            all_predictions.extend(predictions*100)
                
            mae = _mean_absolute_error(predictions*100, labels*100,f'{best_run}/test.png')
            print(f'Test MAE run {best_i}: {mae} : {model._count_alpha}')
            mae_f = float(mae)
            results[donor_id] += mae_f
                
        results[donor_id] = results[donor_id]/eval_runs
        mean_mae += results[donor_id]
        _mean_absolute_error( all_predictions, all_labels,f'{res}/test_all.png')
        pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/orig_results.csv", index=True)
        
        get_residual_plot(all_predictions, all_labels, f"{res}/residuals.png")
        print(f"Mean MAE: {mean_mae / len(donor_ids)}")
        results["mean_mae"] = mean_mae / len(donor_ids)
        # save in file
        with open(f"{res}/results.json", 'w') as f:
            json.dump(results, f, indent=4)
            
def evaluate_all_runs(donor_ids, res,  metadata_file, biom_file, embedding_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for donor_id in donor_ids:
        print(f"\n=== Donor {donor_id} ===")

        args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
                )
        data_processor = DataProcessor(args)
        data_processor.load_data(multitask=True, column="bi_month_name")
            
        test_dataset = data_processor.sample_test_data()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=sparse_collate_fn)
            

        # evaluate every run
        for i in range(1, 4):  # adjust number of runs
            run_dir = f"{res}/{donor_id}/run_{i}/"
            model_file = run_dir + "model.pt"

            model = BasicRegressor(input_dim=128, pe=False).to(device)
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.eval()

            labels, predictions, _ = predict(model, test_loader, device, multitask=True)

            mae_t = float(_mean_absolute_error(predictions*100, labels*100, f'{run_dir}/test.png'))
            print(f"Donor {donor_id} | Run {i} | Test MAE: {mae_t:.4f}")

            # (optional) also save per-run results
            with open(run_dir + "test_eval.json", "w") as f:
                json.dump({"test_mae": float(mae_t)}, f, indent=2)

def evaluate_basic(donor_ids, res, metadata_file, biom_file, embedding_path):
    results = {}
    results_valid = {}
    all_labels = []
    all_predictions = []
    all_sample_ids = []
    eval_runs = 10
    runs = 3
    multitask = False
    mean_mae = 0


    

    for donor_id in donor_ids:
        donor_labels = []
        donor_preds = []
        donor_sample_ids = []

        # ---- Find best run (lowest val MAE) ----
        best_mae = float("inf")
        best_run_dir = None
        best_run_id = None

        for i in range(1, runs+1):
            dir = f"{res}/{donor_id}/run_{i}/"
            with open(dir + 'res.json') as f:
                config = json.load(f)
                mae = float(config['best_mae'])
                if mae < best_mae:
                    best_mae = mae
                    best_run_dir = dir
                    best_run_id = i

        print(f"Best run for donor {donor_id}: {best_run_dir} (val MAE {best_mae})")

        # ---- Evaluate best run multiple times ----
        total_mae = []
        for k in range(eval_runs):
            train_config_path = os.path.join(best_run_dir, "training_config.json")
            if os.path.exists(train_config_path):
                with open(train_config_path) as f:
                    train_config = json.load(f)
            
                if "embedding_path" in train_config:
                    print(f"Using embedding path from train_config.json: {train_config['embedding_path']}")
                    embedding_path = train_config["embedding_path"]
            print("embedding path: ", embedding_path)
            args = Arguments(
                biom_file=biom_file,
                metadata_file=metadata_file,
                tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
                heldout=donor_id,
                sort_asvs=True,
                clr=False,
            )
            data_processor = DataProcessor(args)
            data_processor.load_data(multitask=True, column="bi_month_name")
            
            test_dataset = data_processor.sample_test_data(epoch=k)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=sparse_collate_fn
            )

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cfg_path = os.path.join(best_run_dir, "model_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    model_cfg = json.load(f)
                    if "model_name" not in model_cfg:
                        model_cfg["model_name"] = "BasicRegressor"
            else:
                logging.warning(
                    "model_config.json not found in %s; falling back to default BasicRegressor config.",
                    best_run_dir,
                )
                model_cfg = {"model_name": "BasicRegressor", "input_dim": 128, "pe": False}
            model_a = build_model(model_cfg, str(device))
            checkpoint = torch.load(os.path.join(best_run_dir, "model.pt"), map_location=device)
            model_a.load_state_dict(checkpoint, strict=True)
            model_a.to(device)
            cards = model_cfg.get("metadata_cardinalities", [2,6])
            CARD_TO_NAME = {2: "env", 6: "bimonth"}
            if cards:
                meta_spec = [(CARD_TO_NAME[c], c) for c in cards]
            else:
                meta_spec = []

            print(f"meta_spec: {meta_spec}")


            labels, predictions, per_sample_pred = predict(
                model_a, test_loader, device, scale_abundances=True, val = True, multitask=multitask, meta_spec=meta_spec
            )
            # # labels = np.expm1(labels)
            # # predictions = np.expm1(predictions)
            # for i in per_sample_pred:
            #     per_sample_pred[i][0] = float(np.expm1(per_sample_pred[i][0]))
            #     per_sample_pred[i][1] = float(np.expm1(per_sample_pred[i][1]))



            with open(f"{best_run_dir}/test_predictions.json", "w") as fs:
                json.dump(per_sample_pred,  fs, indent=4)
            # Get sample IDs in the same order as predictions
            sample_ids = list(per_sample_pred.keys()) if isinstance(per_sample_pred, dict) else test_dataset.sample_ids

            mae_t = _mean_absolute_error(predictions * 100, labels * 100, f'{best_run_dir}/test.png')
            total_mae.append(float(mae_t))
            print(f"Donor {donor_id} | MAE: {mae_t}")

            donor_labels.extend(labels * 100)
            donor_preds.extend(predictions * 100)
            donor_sample_ids.extend(sample_ids)

        avg_mae = np.mean(total_mae)
        std_mae = np.std(total_mae)
        results[donor_id] = avg_mae
        results_valid[donor_id] = best_mae
        mean_mae += avg_mae

        print(f"Donor {donor_id} | Average MAE over {eval_runs} runs: {avg_mae:.4f} ± {std_mae:.4f}")


        all_labels.extend(donor_labels)
        all_predictions.extend(donor_preds)
        all_sample_ids.extend(donor_sample_ids)

    # ---- After all donors ----
    _plot_face_hip_colored(all_predictions, all_labels, all_sample_ids, f'{res}/test_all_facehip.png')
    get_residual_plot(all_predictions, all_labels, f"{res}/residuals.png")
    _plot_residuals_face_hip(all_predictions, all_labels, all_sample_ids, f"{res}/residuals_facehip.png")

    results["mean_mae"] = mean_mae / len(donor_ids)
    
    pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/orig_results.csv", index=True)

    with open(f"{res}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    results_valid["mean_mae"] = sum(results_valid.values()) / len(donor_ids)
    with open(f"{res}/results_valid.json", 'w') as f:
        json.dump(results_valid, f, indent=4)

    print(f"Mean MAE: {results['mean_mae']}")


def _plot_face_hip_colored(predictions, labels, sample_ids, save_path):
    """Scatter plot of predicted vs true ADD, colored by face/hip body site."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    
    # Identify face vs hip from sample IDs
    is_face = np.array(['SKF' in str(sid) for sid in sample_ids])
    is_hip = np.array(['SKH' in str(sid) for sid in sample_ids])
    is_other = ~(is_face | is_hip)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if is_face.any():
        face_mae = np.mean(np.abs(predictions[is_face] - labels[is_face]))
        ax.scatter(labels[is_face], predictions[is_face], 
                   alpha=0.6, s=30, c='#E63946', label=f'Face (n={is_face.sum()}, MAE={face_mae:.1f})')
    if is_hip.any():
        hip_mae = np.mean(np.abs(predictions[is_hip] - labels[is_hip]))
        ax.scatter(labels[is_hip], predictions[is_hip], 
                   alpha=0.6, s=30, c='#457B9D', label=f'Hip (n={is_hip.sum()}, MAE={hip_mae:.1f})')
    if is_other.any():
        ax.scatter(labels[is_other], predictions[is_other], 
                   alpha=0.6, s=30, c='gray', label=f'Other (n={is_other.sum()})')
    
    # Diagonal reference line
    lo = min(labels.min(), predictions.min())
    hi = max(labels.max(), predictions.max())
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.5, label='y = x')
    
    ax.set_xlabel('True ADD')
    ax.set_ylabel('Predicted ADD')
    ax.set_title('Predicted vs True ADD by Body Site')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_residuals_face_hip(predictions, labels, sample_ids, save_path):
    """Residual plot colored by face/hip body site."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    predictions = np.array(predictions).flatten()
    labels = np.array(labels).flatten()
    residuals = predictions - labels
    
    is_face = np.array(['SKF' in str(sid) for sid in sample_ids])
    is_hip = np.array(['SKH' in str(sid) for sid in sample_ids])
    is_other = ~(is_face | is_hip)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if is_face.any():
        ax.scatter(labels[is_face], residuals[is_face],
                   alpha=0.6, s=30, c='#E63946', label=f'Face (n={is_face.sum()})')
    if is_hip.any():
        ax.scatter(labels[is_hip], residuals[is_hip],
                   alpha=0.6, s=30, c='#457B9D', label=f'Hip (n={is_hip.sum()})')
    if is_other.any():
        ax.scatter(labels[is_other], residuals[is_other],
                   alpha=0.6, s=30, c='gray', label=f'Other (n={is_other.sum()})')
    
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('True ADD')
    ax.set_ylabel('Residual (Predicted - True)')
    ax.set_title('Residuals by Body Site')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_basic_old(donor_ids, res, metadata_file, biom_file, embedding_path):           
    results = {}
    results_valid = {}
    all_labels = []
    all_predictions = []
    eval_runs = 10
    runs = 1
    multitask = False
    mean_mae = 0
    
    for donor_id in donor_ids:
        donor_labels = []
        donor_preds = []

        # ---- Find best run (lowest val MAE) ----
        best_mae = float("inf")
        best_run_dir = None
        best_run_id = None

        for i in range(1, runs+1):
            dir = f"{res}/{donor_id}/run_{i}/"
            with open(dir + 'res.json') as f:
                config = json.load(f)
                mae = float(config['best_mae'])
                if mae < best_mae:
                    best_mae = mae
                    best_run_dir = dir
                    best_run_id = i

        print(f"Best run for donor {donor_id}: {best_run_dir} (val MAE {best_mae})")

        # ---- Evaluate best run multiple times ----
        total_mae = 0
        for k in range(eval_runs):
            args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
            sort_asvs=True,
            clr=True,
                )
            data_processor = DataProcessor(args)
            data_processor.load_data(multitask=True, column="bi_month_name")
            
            test_dataset = data_processor.sample_test_data(epoch=k)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, collate_fn=sparse_collate_fn)
            

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cfg_path = os.path.join(best_run_dir, f"model_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    model_cfg = json.load(f)
                    if "model_name" not in model_cfg:
                        model_cfg["model_name"] = "BasicRegressor"
            else:
                logging.warning(
                    "model_config.json not found in %s; falling back to default BasicRegressor config.",
                    best_run_dir,
                )
                model_cfg = {"model_name": "BasicRegressor", "input_dim": 128, "pe": False}
            model_a = build_model(model_cfg, str(device))
            checkpoint = torch.load(os.path.join(best_run_dir, "model.pt"), map_location=device)
            model_a.load_state_dict(checkpoint, strict=True)


            model_a.to(device)

            labels, predictions, per_sample_pred = predict(model_a, test_loader, device, scale_abundances=False, multitask=multitask)

            mae_t = _mean_absolute_error(predictions*100, labels*100, f'{best_run_dir}/test.png')
            total_mae += float(mae_t)
            # for j in range(len(predictions)):

            #     print(f"Donor {donor_id} | Sample {j} | True: {labels[j][0]:.2f} | Pred: {predictions[j][0]:.2f}")
            print(f"Donor {donor_id} | MAE: {mae_t}")

            donor_labels.extend(labels*100)
            donor_preds.extend(predictions*100)

        avg_mae = total_mae / eval_runs
        results[donor_id] = avg_mae
        results_valid[donor_id] = best_mae
        mean_mae += avg_mae

        # collect global predictions for residual plot
        all_labels.extend(donor_labels)
        all_predictions.extend(donor_preds)

    # ---- After all donors ----
    _mean_absolute_error(all_predictions, all_labels, f'{res}/test_all.png')
    get_residual_plot(all_predictions, all_labels, f"{res}/residuals.png")

    results["mean_mae"] = mean_mae / len(donor_ids)
    
    pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/orig_results.csv", index=True)

    with open(f"{res}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    results_valid["mean_mae"] = sum(results_valid.values()) / len(donor_ids)
    with open(f"{res}/results_valid.json", 'w') as f:
        json.dump(results_valid, f, indent=4)

    print(f"Mean MAE: {results['mean_mae']}")

 

if __name__ == "__main__":

        from sklearn.model_selection import train_test_split
        folds = 3
        random_vec = False
#         donor_ids = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
#  'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
        #donor_ids = ["D27", "D15","D12", "D24"]#, "D15",'D19', "D22", "D27", ]  #

        data_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
        output_dir = os.path.join(data_dir, "microbiome_model/results/ModelSelection/tmp")
        output_dir = os.path.join(data_dir, "microbiome_model/results/all_data_seqmit_env_bimonth_attnpool_1/") #clusteredbaseline2
        biom_file = os.path.join(data_dir, "process_data_all/exported-feature-table/feature-table.biom")
        biom_file_sheds = os.path.join(data_dir, "data/new_data/feature-table/feature-table.biom")
        sheds_file = os.path.join(data_dir, "data/new_data/metadata_sheds.tsv")
        metadata_file = os.path.join(data_dir, "data/new_data/combined_metadata_merged.tsv")
        embedding_path = os.path.join(data_dir, "data/embeddings/all_data.h5") #all_data_s_dnabert
        donor_ids = pd.read_csv(sheds_file, sep="\t")['DonorID'].unique().tolist()
        #donor_ids = ["D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16"]
        donor_ids =    ['D24', 'D15', 'D20', 'D27', "D13", 'D22', "D5"]#,"D27", "D13","D15", "D20", "D24"] #, "D15", "D25"]  # , "D9"] #, "D26", "D9", "D27"] #[,"D25", "D29", "D15"] #,"D28", "D27", "D7","D25", "D15", "D13"] #, "D28", "D25", "D15", "D10", "D7"] #, "D25", "D28"]
        evaluate_basic(donor_ids, output_dir, metadata_file, biom_file, embedding_path)
        #evaluate_all_runs(donor_ids, output_dir, metadata_file, biom_file, embedding_path)
