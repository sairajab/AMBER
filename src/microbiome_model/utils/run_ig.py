import torch
from captum.attr import DeepLiftShap, IntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
#from dataset import load_data, sample_test_data
import pandas as pd

from microbiome_model.data.dataset import DataProcessor, Arguments
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn
from microbiome_model.models.orig import BasicRegressor
from microbiome_model.eval.evaluate import predict
import json
import torch.nn as nn
from torch.optim import AdamW
import os

donor_ids = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
 'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
indoor_ids = ['D7', 'D5', 'D9', 'D14', 'D16', 'D18', 'D20', 'D22', 'D24', 'D26',
       'D28', 'D30']
outdoor_uncaged = ['D6', 'D10', 'D12']
outdoor_caged = ['D8', 'D4', 'D11', 'D13', 'D15', 'D17', 'D19', 'D21', 'D23', 'D25',
       'D27', 'D29']
all_outdoor = outdoor_uncaged + outdoor_caged
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings, abundances, features, env, masks=None):
        outputs = self.model(embeddings, abundances, masks, features, env=env)[0]
        return outputs

def computer_attributions(wrapped_model, test_loader, donor_id, attribution_method="IG", device='cuda'):
    if attribution_method == "Shap":
        attributor = DeepLiftShap(wrapped_model)
    elif attribution_method == "IG":
        attributor = IntegratedGradients(wrapped_model, multiply_by_inputs=True)
    else:
        raise ValueError("Unsupported attribution method. Use 'DeepLiftShap' or 'IG'.")

    all_attr_embeddings = []
    all_attr_abundances = []
    all_attr_features = []
    all_attr_env = []
    all_seq_ids = []
    all_sample_ids = []

    for batch in tqdm(test_loader):
        embeddings = batch['embeddings'].to(device).requires_grad_()
        abundances = batch['abundances'].to(device).requires_grad_() / 1e4
        masks = batch['masks'].to(device)
        sample_ids = batch['SampleID']
        seq_ids = batch['seqs_ids']
        features = batch['season'].to(device, dtype=torch.float32).requires_grad_()
        env = batch['env'].to(device, dtype=torch.float32).requires_grad_()
        all_seq_ids.append(seq_ids)
        all_sample_ids.append(sample_ids)

        baseline_embed = torch.zeros_like(embeddings)
        baseline_abund = torch.zeros_like(abundances)
        baseline_features = torch.zeros_like(features)
        #baseline_env = torch.zeros_like(env)
        ### with 0.5 
        baseline_env = torch.full_like(env, 0.5)

        inputs = (embeddings, abundances, features, env)
        baselines = (baseline_embed, baseline_abund, baseline_features, baseline_env)

        attributions = attributor.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=(masks,),
            n_steps=100
        )

        all_attr_embeddings.append(attributions[0].detach().cpu())
        all_attr_abundances.append(attributions[1].detach().cpu())
        all_attr_features.append(attributions[2].detach().cpu())
        all_attr_env.append(attributions[3].detach().cpu())
        del embeddings, abundances, features, env, masks, attributions
        torch.cuda.empty_cache()


    return all_attr_embeddings, all_attr_abundances, all_attr_features, all_attr_env, all_seq_ids, all_sample_ids


base_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
data_dir = os.path.join(base_dir, "data/new_data")
res = "../results/all_data_ST_pe_3/"
metadata_file = os.path.join(data_dir, "combined_metadata_merged.tsv")

res = "heldout_orig_softplus/"
res = "finetune_dnabert_results/more_encoder_layers_filtered_biom_mae_bigN_nonzero_asvs/"
res = "../results/multitask_all_nope_bimonth_env_features/"
res = os.path.join(base_dir, "microbiome_model", "results", "all_data_ST_pe_3")
multitask = True

best_runs = {}
biom_file = os.path.join(base_dir,"process_data_all/exported-feature-table/feature-table.biom")
sheds_file = os.path.join(data_dir, "metadata_sheds.tsv")
embedding_path = os.path.join(base_dir, "data/embeddings/all_data.h5")
all_samples_asv_importance = {}

for donor_id in all_outdoor:  # or indoor_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    avg_mae = 0
    best_mae = 1000
    runs = 3
    best_i = 0
    for i in range(1, runs+1):
        dir = f"{res}/{donor_id}/run_{i}/"
                    
        with open(dir + 'res.json') as f:
            config = json.load(f)
            mae = float(config['best_mae'])
            if mae < best_mae:
                best_mae = mae
                best_run = dir
                best_i = i
    
    if os.path.exists(best_run + "attributions_asv_10.json"):
        print(f"Attributions already exist for donor {donor_id}, run {best_i}. Skipping...")
        best_runs[donor_id] = best_i
        continue
                
    model_files = best_run + "model.pt"
    print(f"Best run: {best_run}", best_mae)
    # Instantiate and load model
    model = BasicRegressor(input_dim=512, hidden_dim=512,num_heads=4, num_layers=2, pe=False) #768
    model.load_state_dict(torch.load(model_files, map_location=device))
    
    args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
            sort_asvs=True, 
                )
        
        # Load Data
    data_processor = DataProcessor(args)
        # Assuming load_data handles the splitting based on 'heldout' arg
    data_processor.load_data(multitask=multitask, column="bi_month_name")
        
    test_dataset = data_processor.sample_test_data()
    if len(test_dataset) == 0:
            print(f"Warning: No test data found for donor {donor_id}")
            continue
            
    test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=2, 
            shuffle=False, 
            num_workers=0, 
            collate_fn=sparse_collate_fn
        )
    

    # Move model to the correct device
    model = model.to(device)
    # Set up attribution method
    model.eval()
    wrapped_model = WrappedModel(model).to(device)
    all_attr_embeddings, all_attr_abundances, all_attr_features, all_attr_env, all_seq_ids, all_sample_ids = computer_attributions(
        wrapped_model, test_loader, donor_id, attribution_method="IG", device=device
    )

    per_sample_asv_importance = {}
    per_sample_abundance_importance = {}
    per_sample_feature_importance = {}
    per_sample_env_importance = {}

    for i in range(len(all_attr_embeddings)):
        sample_ids = all_sample_ids[i]
        per_batch_seq_ids = all_seq_ids[i]
        
        for j in range(all_attr_embeddings[i].shape[0]):
            sample_id = sample_ids[j]
            
            # --- ASV embedding attributions ---
            per_sample_asv_importance.setdefault(sample_id, {})
            if sample_id not in all_samples_asv_importance:
                all_samples_asv_importance[sample_id] = {}
            
            per_sample_seq_ids = per_batch_seq_ids[j]
            attr = all_attr_embeddings[i][j].abs().mean(dim=-1).numpy()
            attr_thresh = np.max(attr) * 0.70
            for k in range(len(per_sample_seq_ids)):
                seq_id = per_sample_seq_ids[k]
                if attr[k] < attr_thresh:
                    continue
                
                per_sample_asv_importance[sample_id][seq_id] = float(attr[k])
                all_samples_asv_importance[sample_id][seq_id] = float(attr[k])
            
            # --- Abundance attributions (per ASV) ---
            per_sample_abundance_importance.setdefault(sample_id, {})
            abund_attr = all_attr_abundances[i][j].abs().squeeze(-1).numpy()  # [L]
            for k in range(len(per_sample_seq_ids)):
                seq_id = per_sample_seq_ids[k]
                per_sample_abundance_importance[sample_id][seq_id] = float(abund_attr[k])
            
            # --- Feature attributions (season/bi-month) ---
            feat_attr = all_attr_features[i][j].abs().numpy()  # [6]
            per_sample_feature_importance[sample_id] = feat_attr.tolist()
            
            # --- Env attribution (indoor/outdoor) ---
            env_attr = all_attr_env[i][j].abs().item()
            per_sample_env_importance[sample_id] = env_attr

    del model, wrapped_model, all_attr_embeddings, all_attr_abundances, all_attr_features, all_attr_env
    torch.cuda.empty_cache()

    # Save all results
    with open(f"{res}/{donor_id}/run_{best_i}/attributions_asv_100.json", 'w') as f:
        json.dump(per_sample_asv_importance, f, indent=4)

    with open(f"{res}/{donor_id}/run_{best_i}/attributions_abundance_100.json", 'w') as f:
        json.dump(per_sample_abundance_importance, f, indent=4)

    with open(f"{res}/{donor_id}/run_{best_i}/attributions_features_100.json", 'w') as f:
        json.dump(per_sample_feature_importance, f, indent=4)

    with open(f"{res}/{donor_id}/run_{best_i}/attributions_env_100.json", 'w') as f:
        json.dump(per_sample_env_importance, f, indent=4)
    
    best_runs[donor_id] = best_i
    
    
print("DONE")