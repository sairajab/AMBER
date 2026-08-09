import json
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, r2_score
from scipy import stats
from tqdm import tqdm

# --- IMPORTS FROM YOUR PROJECT ---
# Adjust these imports to match your actual file structure
from microbiome_model.models.zoo import GeneralizedRegressor,  CLR, ClusteredRegressor, BasicRegressorGRL, BasicRegressorBest
from microbiome_model.models.orig import BasicRegressor, BasicRegressorwithUnifrac
from microbiome_model.data.dataset import Arguments, DataProcessor
from microbiome_model.data.dataset_sparse import collate_fn as sparse_collate_fn
from torch.utils.data import DataLoader

# --- HELPER FUNCTIONS ---
def _mean_absolute_error(predictions, labels, plot_path):
    """Calculates MAE and saves a scatter plot."""
    mae = mean_absolute_error(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(labels, predictions, alpha=0.5)
    
    # Plot diagonal line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]
    plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    plt.xlim(lims)
    plt.ylim(lims)
    
    plt.xlabel('Actual (Degree Days)')
    plt.ylabel('Predicted (Degree Days)')
    plt.title(f'MAE: {mae:.2f}')
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()
    
    return mae

def predict(model, loader, device, multitask=False):
    """Runs inference on a loader and returns flat arrays of labels and preds."""
    model.eval()
    all_preds = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in loader:
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device) / 1e4  # Rescale if you did this during training
            masks = batch['masks'].to(device)
            features = batch['season'].to(device)
            env = batch['env'].to(device, dtype=torch.float32)
            targets = batch['outdoor_add_0'].to(device)
            
            # Forward pass
            # We ignore the other returns (counts, subject_logits, etc.)
            outputs = model(embeddings, abundances, masks, features, env=env)
            preds = outputs[0]  # Assuming the first output is predictions
            
            all_preds.append(preds.cpu().numpy().flatten())
            all_labels.append(targets.cpu().numpy().flatten())
            all_sample_ids.append(batch['SampleID'])
            
    return np.concatenate(all_labels), np.concatenate(all_preds), np.concatenate(all_sample_ids)

# --- INVARIANCE CHECK (t-SNE) ---
def evaluate_invariance(model, val_loader, device, output_dir):
    """
    Extracts latent features from the model to visualize if 
    subjects are mixed (good) or clustered (bad/leakage).
    """
    print("Extracting features for t-SNE...")
    model.eval()
    features_list = []
    ages_list = []
    subjects_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="t-SNE Extraction"):
            embeddings = batch['embeddings'].to(device)
            abundances = batch['abundances'].to(device)
            masks = batch['masks'].to(device)
            features = batch['season'].to(device)
            env = batch['env'].to(device, dtype=torch.float32)
            
            # --- MANUALLY RUN ENCODER PARTS TO GET LATENT VECTOR ---
            # (Matches BasicRegressor logic)
            input_embeddings = model.scale_embeddings(embeddings)
            
            if env is not None and features is not None:
                env_embeddings = model.env_encoder(env.unsqueeze(-1))
                meta_embeddings = torch.cat([model.gating_feature(features), env_embeddings], dim=1)
                feature_embeddings = torch.sigmoid(meta_embeddings)
                input_embeddings = input_embeddings * feature_embeddings.unsqueeze(1)
            
            batch_size = input_embeddings.shape[0]
            query_token = model.query_vector.expand(batch_size, -1, -1)
            asv_embeddings = torch.cat([query_token, input_embeddings], dim=1)

            # Abundance processing
            ab_t = abundances.transpose(1, 2)
            ab_t = torch.nn.functional.pad(ab_t, pad=(1, 0), mode='constant', value=1)
            ab_t = ab_t.transpose(1, 2)
            
            cls_mask = torch.ones(batch_size, 1, device=masks.device, dtype=masks.dtype)
            masks_aug = torch.cat([cls_mask, masks], dim=1)
            attention_mask = ~masks_aug.bool()

            if model.pe:
                weighted_embeddings = asv_embeddings + model.pos_embedding(asv_embeddings) * ab_t
            else:
                weighted_embeddings = asv_embeddings + asv_embeddings * ab_t

            # Sequence Attention
            count_embeddings = model.sequence_attention(weighted_embeddings, src_key_padding_mask=attention_mask)
            count_alpha = torch.nn.functional.softplus(model._count_alpha)
            sequence_encoded = asv_embeddings + count_embeddings * count_alpha

            # Sample Attention
            target_encoded = model.sample_attention(sequence_encoded, src_key_padding_mask=attention_mask)
            summary_token = target_encoded[:, 0, :]
            latent_features = model.norm(summary_token)
            # -------------------------------------------------------

            features_list.append(latent_features.cpu().numpy())
            ages_list.append(batch['outdoor_add_0'].cpu().numpy())
            
            # Extract Subject ID
            # Assumes SampleID format "DonorID_Timepoint"
            
            subjects = [data_processor.sampleID_map_donorID[s] for s in batch["SampleID"]]
            subjects_list.extend(subjects)

    features = np.concatenate(features_list)
    ages = np.concatenate(ages_list).ravel()
    print(f"Extracted features shape: {features.shape}")
    print(f"Ages shape: {ages.shape}    Subjects extracted: {len(set(subjects_list))}")

    
    print(f"Running t-SNE on {features.shape[0]} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot A: Color by Age
    sns.scatterplot(x=features_2d[:,0], y=features_2d[:,1], hue=ages, palette="viridis", s=60, alpha=0.8, ax=axes[0])
    axes[0].set_title("Latent Space by ADD (Target)")
    
    # Plot B: Color by Subject
    # Only show top 15 subjects to avoid messy legends
    top_subjects = pd.Series(subjects_list).value_counts().index[:15]
    mask = np.isin(subjects_list, top_subjects)
    
    sns.scatterplot(x=features_2d[mask,0], y=features_2d[mask,1], hue=np.array(subjects_list)[mask], s=60, alpha=0.8, ax=axes[1])
    axes[1].set_title("Latent Space by Subject (Should be Mixed)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_invariance_check.png"))
    print(f"t-SNE plots saved to {output_dir}")
    # plt.show() # Uncomment if running in notebook


# --- MAIN EVALUATION FUNCTION ---
def evaluated_on_heldout(donor_ids, metadata_file, biom_file, embedding_path, experiment_dir):
    
    

    total_mae = 0
    all_donor_labels = []
    all_donor_preds = []
    all_donor_ids = []

    multitask = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model Once (assuming architecture is fixed)
    # Note: Ensure input_dim/num_subjects matches training
    print("Loading Model...")
    per_donor_maes = {}
    val_maes = {}
    output_dir = os.path.join(experiment_dir, "evaluation_results")
    os.makedirs(output_dir, exist_ok=True)

    df_meta = pd.read_csv(metadata_file, sep='\t')
    print(df_meta.head())
    # Iterate over held-out donors
    all_sample_ids = []
    for donor_id in tqdm(donor_ids, desc="Evaluating Donors"):

        runs = 3
        best_run = None
        best_mae = float('inf')
        for i in range(runs):
            run_dir = os.path.join(experiment_dir,donor_id,f"run_{i+1}")
            with open(os.path.join(run_dir, "res.json"), 'r') as f:
                res = json.load(f)
            if res['best_mae'] < best_mae:
                best_mae = res['best_mae']
                best_run = run_dir

        best_model_path = os.path.join(best_run, "model.pt")
        print(f"Best model for donor {donor_id} found at {best_model_path} with MAE {best_mae:.4f}")

        # We need to know num_subjects to initialize the adversarial head, 
        # even if we don't use it for inference.
        # Quick hack: Read metadata to count unique donors
        
        n_subjects = len(df_meta['host_subject_id'].unique())
        
        #model = BasicRegressor(input_dim=768, hidden_dim=1024, num_heads=8, num_layers=4, dropout=0.3, pe = True, grl=True,  unique_donors_train=n_subjects)
        #model = CLR()
        #model = ClusteredRegressor()

        #model = BasicRegressorGRL(256, 512, 4, 2, 0.3, pe=True, num_subjects=106) # domain invariant
        model = BasicRegressorBest(128, 1024, 4, 2, 0.3, pe=False)

        print(f"Loading model for donor {donor_id} from {best_model_path}...")

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.to(device)
        model.eval()
            


        for i in range(1):
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
                batch_size=16, 
                shuffle=False, 
                num_workers=0, 
                collate_fn=sparse_collate_fn
            )
            
            # Predict
            labels, predictions, sample_ids = predict(model, test_loader, device)
            
            # Rescale (Assuming you divided by 100 during training)
            labels_real = np.abs(labels) * 100
            preds_real = np.abs(predictions) * 100
            
            # Store for aggregation
            all_donor_labels.extend(labels_real)
            all_donor_preds.extend(preds_real)
            all_donor_ids.extend([donor_id] * len(labels_real))
            all_sample_ids.extend(sample_ids)
            
            # Individual Donor MAE
            donor_mae = mean_absolute_error(labels_real, preds_real)
            print(f"Donor {donor_id} MAE: {donor_mae:.2f}")
            per_donor_maes[donor_id] = donor_mae
            val_maes[donor_id] = best_mae

    # --- AGGREGATED RESULTS ---
    all_donor_labels = np.array(all_donor_labels)
    all_donor_preds = np.array(all_donor_preds)
    
    final_mae = mean_absolute_error(all_donor_labels, all_donor_preds)
    final_r2 = r2_score(all_donor_labels, all_donor_preds)
    
    print("="*30)
    print(f"OVERALL RESULTS on {len(donor_ids)} Donors")
    print(f"MAE: {final_mae:.4f} Degree Days")
    print(f"R2:  {final_r2:.4f}")
    print("="*30)

    # # Save Prediction CSV
    df_results = pd.DataFrame({
        'SampleID': all_sample_ids,
        'DonorID': all_donor_ids,
        'Actual': all_donor_labels,
        'Predicted': all_donor_preds
    })
    df_results.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    _mean_absolute_error(all_donor_preds, all_donor_labels, os.path.join(output_dir, "overall_scatter.png"))

    # Plot Error by Donor (Boxplot)
    plt.figure(figsize=(12, 6))
    df_results['AbsError'] = np.abs(df_results['Actual'] - df_results['Predicted'])
    sns.boxplot(x='DonorID', y='AbsError', data=df_results)
    plt.xticks(rotation=45)
    plt.title("Absolute Error Distribution by Donor")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_donor.png"))
    per_donor_maes["mean_absolute_error"] = final_mae
    with open(os.path.join(output_dir, "per_donor_maes.json"), 'w') as f:
        json.dump(per_donor_maes, f, indent=4)
    val_maes["mean_absolute_error"] = np.mean(list(val_maes.values()))
    with open(os.path.join(output_dir, "validation_maes.json"), 'w') as f:
        json.dump(val_maes, f, indent=4)


def evaluate_invariance_fn(donor_id, data_processor, device, experiment_dir, run=1):
    print(type(experiment_dir), type(donor_id), type(run))

    model_path = os.path.join(experiment_dir, donor_id, f"run_{run}", "model.pt") 

    n_subjects = 106
    model = BasicRegressor(input_dim=256, hidden_dim=512, num_heads=4, num_layers=2, dropout=0.3, pe = True, num_subjects=n_subjects)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
            

    output_dir = os.path.join(experiment_dir, donor_id, f"run_{run}", "invariance_check")
    os.makedirs(output_dir, exist_ok=True)
    _, val_dataset = data_processor.sample_data(epoch=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn)
    evaluate_invariance(model, val_loader, device, output_dir)


if __name__ == "__main__":
    
    # --- CONFIGURATION ---
    # Update these paths to match your system exactly
    base_dir = "/s/chromatin/o/nobackup/Saira/Microbiome_Project/"
    data_dir = os.path.join(base_dir, "data/new_data")
    
    biom_file = os.path.join(base_dir,"process_data_all/exported-feature-table/feature-table.biom")
    metadata_file = os.path.join(data_dir, "combined_metadata_merged.tsv")
    sheds_file = os.path.join(data_dir, "metadata_sheds.tsv")
    embedding_path = os.path.join(base_dir, "data/embeddings/all_data.h5")
    
    # Where your trained model is saved
    best_run_dir = os.path.join(base_dir, "microbiome_model/results/ModelSelection/baseline_metadata/")
    best_run_dir = os.path.join(base_dir, "microbiome_model/results//DonorInvariant/")
    best_run_dir = os.path.join(base_dir, "microbiome_model/results/multitask_all_nope_bimonth_env_features") #domain invariant
    #best_run_dir = os.path.join(base_dir, "microbiome_model/results/ModelSelection/bestmodelGRL") #clusteredbaseline2

    # --- DONOR SELECTION ---
    # You can load from file or define manually
    # if os.path.exists(sheds_file):
    #     donor_ids = pd.read_csv(sheds_file, sep="\t")['DonorID'].unique().tolist()
    # else:
    #     # Fallback manual list
    #     donor_ids = ["D7", "D4", "D12"] 

    ## evaluate invariance 
    donor_id = "D24"  # Example donor for invariance check

    args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=donor_id,
                )
        
        # Load Data
    data_processor = DataProcessor(args)
    data_processor.load_data(multitask=True, column="bi_month_name")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #evaluate_invariance_fn(donor_id, data_processor, device, best_run_dir, run=1)

    print(f"Invariance evaluation completed for donor {donor_id}. Check the generated t-SNE plots for insights on subject mixing in the latent space.")
    print("\n\n" + "="*50 + "\n\n")
    donor_ids = ["D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25",
                 "D26", "D27", "D28", "D29", "D30"] 

    #donor_ids = ["D15"]


    print(f"Starting evaluation on {len(donor_ids)} donors...")
    
    evaluated_on_heldout(
        donor_ids=donor_ids,
        metadata_file=metadata_file,
        biom_file=biom_file,
        embedding_path=embedding_path,
        experiment_dir=best_run_dir,
    )
