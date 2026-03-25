import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from models import SampleLevelRegressor
from datetime import datetime
from other_models import random_forest
from sklearn.metrics import mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import load_data, sample_data, sample_test_data
import os 
import math 
import torch.optim as optim
from losses import compute_count_loss, PairwiseLoss
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import _mean_absolute_error, predict
from other_models import RandomForestModel
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestRegressor

def float_mask(tensor , dtype=torch.float32):

    return (tensor != 0).to(dtype)

def compute_unifrac_loss(y_true, embeddings, pairwise_loss_fn):

    loss = pairwise_loss_fn(y_true, embeddings)  # [B, B]
    mask = float_mask(loss)                      # [B, B]
    num_samples = mask.sum()
    total_loss = loss.sum()
    return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0, device=loss.device)


class WarmupCosineDecay:
    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=0.001, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.min_lr = min_lr

    def lr_lambda(self, step):
        """Defines learning rate schedule"""
        if step < self.warmup_steps:
            return self.base_lr
        else:
            # Cosine decay phase
            decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * decay_ratio))  # Cosine decay

    def get_scheduler(self):
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)



def train_model_rf_style(model, abundance_table, train_data, val_data, out_dir,
                         num_epochs=100, early_stop_warmup=10, patience=10,
                         kmer_seqs=None, embedding_path="../data/embeddings/dnabert_embeddings.h5", random_vector=False):

    os.makedirs(out_dir, exist_ok=True)

    best_val_mae = float('inf')
    patience_counter = 0
    best_model_path = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Sample fresh data
        train_loader, val_loader = sample_data(
            abundance_table, train_data, val_data,
            kmer_seqs=kmer_seqs,
            embedding_path=embedding_path,
            random_vector=random_vector
        )

        # Fit model on current train data
        model.fit(train_loader)
        
        # Evaluate on validation set
        val_preds = model.predict(val_loader)
        _, val_labels = model._flatten_dataloader(val_loader)  # or however you get true labels
        val_mae = mean_absolute_error(val_labels, val_preds)

        print(f"\nEpoch {epoch+1}/{num_epochs} - Validation MAE: {val_mae:.4f}")

        # Save loss
        train_losses.append(0.0)  # RF has no internal training loss
        val_losses.append(val_mae)

        # Save best model
        if val_mae < best_val_mae:
            print("New best model found.")
            best_val_mae = val_mae
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_model_path = f'{out_dir}/best_model.pkl'
            import joblib
            joblib.dump(model.model, best_model_path)
            patience_counter = 0
        else:
            if epoch >= early_stop_warmup:
                patience_counter += 1
                print(f"No improvement. Patience counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # Plot validation MAE
    plt.figure(figsize=(8, 6))
    plt.plot(val_losses, label='Validation MAE', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/val_mae_curve.png')
    plt.close()

    # Save losses
    pd.DataFrame({'val_mae': val_losses}).to_csv(f'{out_dir}/val_mae.csv', index=False)

    return {
        "best_mae": float(best_val_mae),
        "best_model": best_model_path
    }

def test_rf():
        random_vec = False
        donor_ids = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
 'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
        results = {}
        res = "rf_results"
        for donor_id in donor_ids:

            out_dir = os.path.join(res, donor_id)
            
            abundance_table, train_data, test_data, embedding_path, _  = load_data(heldout=donor_id, kmer_embedding=False)
            test_loader = sample_test_data(abundance_table, test_data, embedding_path = embedding_path, random_vector=random_vec)
            
            model = RandomForestModel(n_estimators=100, max_depth=10)
            # Load raw sklearn model
            raw_model = joblib.load(f'{out_dir}/best_model.pkl')
            model = RandomForestModel(n_estimators=100, max_depth=10)
            model.model = raw_model
            
            mae = model.evaluate(test_loader, out_dir)
            results[donor_id] = mae

        
        pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/rf_results.csv", index=True)
        


def median_model(train_data, test_data, out_dir):
    # Extract the median of the training data
    train_samples, train_target = train_data
    val_samples, val_target = test_data
    
    print(f"Number of training samples: {len(train_samples)}")
    targets = np.array(list(train_target.values()))
    val_targets = np.array(list(val_target.values()))
    # Compute the median for each sample
    medians = np.median(targets, axis=0)

    # Predict on validation set
    val_preds = np.full_like(val_targets, medians)

    # Compute MAE
    mae = _mean_absolute_error(val_targets*100, val_preds*100, f'{out_dir}/median_model.png')
    
    print(f"Median model MAE: {mae}")

    return {
        "best_mae": mae,
        "best_model": None  # No model to save for median model
    }

def test_median_model():
    donor_ids = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
 'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
    results = {}
    res = "median_results"
    for donor_id in donor_ids:

        out_dir = os.path.join(res, donor_id)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        
        abundance_table, train_data, test_data, embedding_path, _  = load_data(heldout=donor_id, kmer_embedding=False)
        train_samples , train_target = train_data
        num_train_samples = list(range(len(train_samples)))
        
        train_ids, val_ids = train_test_split(num_train_samples, test_size=0.1 , random_state=42)   
            # Split train and validation sets
        X_train, X_val = [train_samples[i] for i in train_ids], [train_samples[i] for i in val_ids]
        y_train, y_val = {i:train_target[i] for i in X_train}, {i:train_target[i] for i in X_val}
        mae = median_model(train_data, test_data, out_dir)
        results[donor_id] = mae

    
    pd.DataFrame.from_dict(results, orient='index').to_csv(f"{res}/median_results.csv", index=True)
import biom            
def get_data_2(heldout, table_path, target_path):
    
    abundance_table = biom.load_table(table_path)#.subsample(5000)
    print("Abundance table", abundance_table.shape)
    targets_df = pd.read_csv(target_path)

    sample_ids = abundance_table.ids(axis='sample')  # list of all sample IDs
    sample_targets = dict(zip(targets_df['SampleID'], targets_df['outdoor_add_0']))
    
    shared_ids = np.intersect1d(sample_ids, list(sample_targets.keys()))

    table = abundance_table.filter(shared_ids, axis='sample', inplace=False)
    table = table.filter(
    lambda val, id_, metadata: val.sum() != 0,
    axis='observation',
)
    print("Abundance table after filtering", table.shape)
    # Separate heldout samples for test set
    test_ids = [sid for sid in shared_ids if heldout in sid]
    train_val_ids = [sid for sid in shared_ids if sid not in test_ids]

    # have both sets together
    train_ids, val_ids = train_test_split(
    train_val_ids, 
    test_size=0.1, 
    random_state=42
)

    # Convert to numpy for fast lookup
    sample_ids_array = np.array(shared_ids)

    # Find positions in the matrix
    train_indices = np.where(np.isin(sample_ids_array, train_ids))[0]
    val_indices = np.where(np.isin(sample_ids_array, val_ids))[0]
    test_indices = np.where(np.isin(sample_ids_array, test_ids))[0]
    train_val_indices = np.where(np.isin(sample_ids_array, train_val_ids))[0]


    table_data = table.matrix_data.tocoo()
    table_data = table_data.tocsr()
    
    X_train = table_data[:, train_val_indices].transpose()
    X_val = table_data[:, val_indices].transpose()
    X_test = table_data[:, test_indices].transpose()
    
    
    
    y_train = [sample_targets[sid] for sid in train_val_ids]
    y_val = [sample_targets[sid] for sid in val_ids]
    y_test = [sample_targets[sid] for sid in test_ids]


    return X_train, y_train, X_val, y_val, X_test, y_test

def get_data(heldout, table_path, target_path):
    # Load biom table
    abundance_table = biom.load_table(table_path)  # no subsampling
    print("Abundance table before filtering", abundance_table.shape)

    # Load target metadata
    targets_df = pd.read_csv(target_path)
    sample_targets = dict(zip(targets_df['SampleID'], targets_df['outdoor_add_0']))

    # Keep only samples that exist in both biom and target file, preserving biom order
    biom_sample_ids = list(abundance_table.ids(axis='sample'))
    shared_ids = [sid for sid in biom_sample_ids if sid in sample_targets]

    # Filter biom table to shared IDs
    table = abundance_table.filter(shared_ids, axis='sample', inplace=False)
    # Remove features with zero total counts
    table = table.filter(
        lambda val, id_, metadata: val.sum() != 0,
        axis='observation',
    )
    print("Abundance table after filtering", table.shape)

    # Split samples into heldout donor vs. rest
    test_ids = [sid for sid in shared_ids if heldout in sid]
    train_val_ids = [sid for sid in shared_ids if sid not in test_ids]

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.1,
        random_state=42
    )

    # Get feature matrix (sparse csr) with samples as columns
    table_data = table.matrix_data.tocsr()

    # Convert to sample-by-feature (rows = samples, cols = features)
    X_train = table_data[:, [shared_ids.index(sid) for sid in train_val_ids]].transpose()
    X_val   = table_data[:, [shared_ids.index(sid) for sid in val_ids]].transpose()
    X_test  = table_data[:, [shared_ids.index(sid) for sid in test_ids]].transpose()

    # Targets aligned by sample ID
    y_train = [sample_targets[sid] for sid in train_val_ids]
    y_val   = [sample_targets[sid] for sid in val_ids]
    y_test  = [sample_targets[sid] for sid in test_ids]

    return X_train, y_train, X_val, y_val, X_test, y_test    
    
    


def main():
    mode = "train"
    # Setup
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    from sklearn.model_selection import KFold

    all_heldouts = ['D19','D7','D8','D22','D13','D15','D28','D10','D17','D11','D4','D26','D23','D29','D27','D20','D6',
                    'D25','D30','D5','D21','D18','D14','D12','D24','D9','D16']
    #done = ['D10', 'D11','D7', 'D8', 'D22'] #
    #heldout_samples = list(set(all_heldouts) - set(done))
    heldout_samples = all_heldouts
    print(f"Running for heldout samples: {heldout_samples}")
    output_dir = "rf_results_no_embeddngs"
    random_vector = False
    kmer_embedding = False
    embedding_path = None
    kmer_seqs = None
    rarefied_table_path = "../data/table_sheds_dada2.biom"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # Create a directory for each heldout sample
    test_results = {}
    for j in range(len(heldout_samples)):
        
        heldout = heldout_samples[j]
        print(f"Running for heldout sample: {heldout}")
        out_dir = os.path.join(output_dir, heldout)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        X_train, y_train, _, _, X_test, y_test = get_data(heldout, rarefied_table_path, "../data/target_df.csv")

        print(X_train.shape)
        #print(X_val.shape)
        print(X_test.shape)
        rf = RandomForestRegressor(n_estimators=100, 
    						   max_features=0.2, 
    						   max_depth= None, 
    						   random_state=999, 
    						   criterion='absolute_error', 
    						   bootstrap=False,
                               n_jobs=-1)
        #rf = RandomForestRegressor(n_estimators=500, random_state=999, criterion='absolute_error')
        rf.fit(X_train, y_train)
            
        # val_preds = rf.predict(X_val)
        # val_mae = _mean_absolute_error(y_val, val_preds, f'{out_dir}/val_mae_run.png')
        # print(f"Validation MAE: {val_mae}")
        
        test_preds = rf.predict(X_test)
        test_mae = _mean_absolute_error(test_preds,y_test,f'{out_dir}/test_mae.png')
        print(f"Test MAE: {test_mae}")
        test_results[heldout] = test_mae
        
    # Save results
    results_df = pd.DataFrame.from_dict(test_results, orient='index', columns=['MAE'])
    results_df.to_csv(f"{output_dir}/rf_results_3.csv", index=True)



if __name__ == "__main__":
    main()
    #test_rf()
    #test_median_model()
