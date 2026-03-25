import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from tqdm import tqdm
from model_orig import BasicRegressor, BasicRegressorwithUnifrac
from datetime import datetime
from other_models import random_forest
from sklearn.metrics import mean_absolute_error
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from dataset import Arguments, DataProcessor
import os 
import math 
import torch.optim as optim
from losses import compute_count_loss, PairwiseLoss
from sklearn.model_selection import train_test_split
import json
import matplotlib.pyplot as plt
import pandas as pd
from evaluate import _mean_absolute_error, predict
import random
import re
from dataset_sparse import collate_fn as sparse_collate_fn
from torch.utils.data import Dataset, DataLoader


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



def train_model(model, num_epochs, learning_rate, device, out_dir, data_processor, use_unifrac_loss=False): #abundance_table , distances, train_data, val_data, use_unifrac_loss=False, embedding_path=None,kmer_seqs=None, random_vector=False, kmer_embedding=False): 
        """Training function similar to before but adapted for sample-level predictions"""
        model = model.to(device)
        #criterion = nn.SmoothL1Loss(beta=2) #MSELoss()
        criterion = nn.MSELoss()
        pairwise_loss_fn = PairwiseLoss()
        count_loss_type = "mse"
        
        print("Training model...")
        print(model)
        train_losses = []
        val_losses = []
        train_maes = []
        val_maes = []
        warmup_steps = 10000 
        total_steps = 30000
        lrs = []
        
        # Initialize datasets and loaders with resource management
        train_dataset = None
        val_dataset = None
        train_loader = None
        val_loader = None
        multitask = False
    
    #try:
        # steps_per_epoch = len(train_loader)  # Number of batches per epoch
        # total_steps = num_epochs * steps_per_epoch  # Total training steps

        # # Set warmup steps dynamically (e.g., first 10% of training)
        # warmup_steps = int(0.1 * total_steps)  

        # Initialize optimizer and scheduler
        # optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.00001)
        # #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.001)
        # # Cosine Decay with Restarts Scheduler (Equivalent to CosineDecayRestarts in TF)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, 
        #     T_0=warmup_steps,  # Number of steps before first restart
        #     T_mult=1)  
        
        # Multiplicative factor for decay period
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer, mode='min', patience=5, factor=0.5)
        learning_rate = 1e-4      # Slightly higher than 1e-5 for meaningful updates
        weight_decay = 1e-3       # Regularization to stabilize training
        eta_min = 1e-6            # Minimum LR at the end of cosine cycle
        T_0 = 10                  # Number of epochs before first restart
        T_mult = 2                # Multiply T_0 after each restart

        # ----- Optimizer -----
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate)

        # ----- Scheduler -----
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )

        best_val_loss = float('inf')
        early_stop_warmup = 50
        patience = 30
        patience_counter = 0
        best_mae = 1000
        step = 0
        reg_count_loss = 1 # Weight for count regularization loss
        train_dataset, val_dataset = data_processor.sample_data(epoch=0)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn)

        distances_target = None
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0
            per_batch_count_loss = 0
            per_batch_unifrac_loss = 0
            #train_loader, _ = sample_data(abundance_table , train_data, val_data,epoch,kmer_seqs=kmer_seqs, embedding_path=embedding_path, random_vector=random_vector, batch_size=4)
            train_dataset.sample_epoch_init(epoch)
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False, collate_fn=sparse_collate_fn)

            predictions = []
            labels = []
            indoor_loss = 0
            outdoor_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
                embeddings = batch ['embeddings'].to(device)
                abundances = batch['abundances'].to(device)
                targets = batch['outdoor_add_0'].to(device)
                masks = batch['masks'].to(device)
                sample_ids = batch["SampleID"]
                env = batch['env'].to(device, dtype=torch.float32)  # 0 for indoor, 1 for outdoor
                features= batch['season'].to(device)  # seasonal features
                
                # donor_ids = [int(re.search(r'D(\d+)', sid).group(1)) for sid in sample_ids]
                # donor_ids = torch.tensor(donor_ids).to(device)
                donor_ids = None
                # On train donors and D13:
                # print("Num non-zero ASVs per sample:", (abundances > 0).sum(-1).float().mean())
                # print("Top-1 abundance value:", abundances.max(-1).values.mean())

                #print(embeddings.shape, abundances.shape, targets.shape, masks.shape)
                if torch.isnan(targets).any():
                    print("NaN detected in targets!")
                
                optimizer.zero_grad()
                outputs , counts_pred , unifrac_embeddings = model(embeddings, abundances, masks, features , env=env)
                
                # if torch.isnan(outputs).any():
                #     print("NaN detected in outputs!")
            
                count_loss = compute_count_loss(abundances, counts_pred, loss_type=count_loss_type).mean() 
                if multitask:
                    p_indoor, p_outdoor = outputs
                    mask_in = (env == 0)
                    mask_out = (env == 1)

                    loss_in = F.mse_loss(p_indoor[mask_in], targets[mask_in]) if mask_in.any() else 0
                    loss_out = F.mse_loss(p_outdoor[mask_out], targets[mask_out]) if mask_out.any() else 0
                    
                    n_in = mask_in.sum() + 1e-8
                    n_out = mask_out.sum() + 1e-8
                    total = n_in + n_out
                    w_in = n_in / total
                    w_out = n_out / total
                    loss = w_in * loss_in + w_out * loss_out
                    all_preds = torch.cat([p_indoor[mask_in], p_outdoor[mask_out]])
                    all_targets = torch.cat([targets[mask_in], targets[mask_out]])

                    if loss_in != 0:
                        indoor_loss += loss_in.item()
                    if loss_out != 0:
                        outdoor_loss += loss_out.item()
                else:
                    loss = criterion(outputs, targets)
                    all_preds = outputs
                    all_targets = targets
                    
                if use_unifrac_loss:
                    distances_target = torch.from_numpy(distances.filter(sample_ids).data)
                    unifrac_loss = compute_unifrac_loss(distances_target, unifrac_embeddings, pairwise_loss_fn)
                    per_batch_unifrac_loss += unifrac_loss.item()
                    loss = loss + (0.1 * unifrac_loss)
                
                

                predictions.append(all_preds.cpu().detach().numpy())
                labels.append(all_targets.cpu().detach().numpy())
                
                
                loss = loss + (reg_count_loss * count_loss)
                #print(f"Loss: {loss.item()}")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # scheduler.step()  # Step the scheduler after each optimizer step
                
                # # Increase step counter and log learning rate periodically
                step += 1
                # if step % 101 == 0:
                #     print(f"Step {step}: Learning Rate = {scheduler.get_last_lr()[0]:.6f}")
                #     print(f"Step {step}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")
                # lrs.append(optimizer.param_groups[0]['lr'])
                
                train_loss += loss.item()
                per_batch_count_loss += count_loss.item()
                    # Step scheduler ONCE per epoch, not per batch
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            
            print("total steps ", step)
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            train_mae = mean_absolute_error(labels ,predictions)
            train_maes.append(train_mae)
            train_loss /= len(train_loader)
            per_batch_count_loss /= len(train_loader)
            per_batch_unifrac_loss /= len(train_loader)
            indoor_loss /= len(train_loader)
            outdoor_loss /= len(train_loader)
            print(f"Indoor Loss: {indoor_loss:.4f}, Outdoor Loss: {outdoor_loss:.4f}")

            predictions = []
            labels = []

            # Validation
            model.eval()
            val_loss = 0
            per_batch_count_loss_val = 0
            per_batch_unifrac_loss_val = 0
            all_ids = []
            with torch.no_grad():
                for batch in val_loader:
                    embeddings = batch['embeddings'].to(device)
                    abundances = batch['abundances'].to(device)
                    targets = batch['outdoor_add_0'].to(device)
                    masks = batch['masks'].to(device)
                    all_ids.extend(batch["SampleID"])
                    sample_ids = batch["SampleID"]
                    env = batch['env'].to(device, dtype=torch.float32)  # 0 for indoor, 1 for outdoor
                    features= batch['season'].to(device)  # seasonal features

                    outputs,counts_pred, unifrac_embeddings = model(embeddings, abundances, masks, features, env=env, donor_ids=None)
                    count_loss = compute_count_loss(abundances, counts_pred ,loss_type=count_loss_type).mean()
                    if multitask:
                        p_indoor, p_outdoor = outputs
                        mask_in = (env == 0)
                        mask_out = (env == 1)
                        loss_in = F.mse_loss(p_indoor[mask_in], targets[mask_in]) if mask_in.any() else 0
                        loss_out = F.mse_loss(p_outdoor[mask_out], targets[mask_out]) if mask_out.any() else 0
                        n_in = mask_in.sum() + 1e-8
                        n_out = mask_out.sum() + 1e-8
                        total = n_in + n_out
                        w_in = n_in / total
                        w_out = n_out / total
                        loss = w_in * loss_in + w_out * loss_out
                        all_preds = torch.cat([p_indoor[mask_in], p_outdoor[mask_out]])
                        all_targets = torch.cat([targets[mask_in], targets[mask_out]])
                    else:
                        loss = criterion(outputs, targets)
                        all_preds = outputs
                        all_targets = targets
                        
                    if use_unifrac_loss:
                        sample_ids = batch["SampleID"]
                        distances_target = torch.from_numpy(distances.filter(sample_ids).data).to(device)
                        unifrac_loss = compute_unifrac_loss(distances_target, unifrac_embeddings, pairwise_loss_fn)
                        per_batch_unifrac_loss_val += unifrac_loss.item()
                        loss = loss + (0.1 * unifrac_loss)
                        
                    loss = loss + (reg_count_loss * count_loss) 
                    
                    val_loss += loss.item()
                    per_batch_count_loss_val += count_loss.item()
                    

                    predictions.append(all_preds.cpu().detach().numpy())
                    labels.append(all_targets.cpu().detach().numpy())

            print("All IDs in validation set: ", len(all_ids))
            val_loss /= len(val_loader)
            per_batch_count_loss_val /= len(val_loader)
            per_batch_unifrac_loss_val /= len(val_loader)
            labels = np.concatenate(labels)
            predictions = np.concatenate(predictions)
            val_mae = mean_absolute_error(labels ,predictions)
            val_maes.append(val_mae)
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Count Loss: {per_batch_count_loss:.4f} , Unifrac Loss: {per_batch_unifrac_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Validation Loss: {val_loss:.4f}, Count Loss: {per_batch_count_loss_val:.4f}, Unifrac Loss: {per_batch_unifrac_loss_val:.4f}, Validation MAE: {val_mae:.4f}")
            print(f"Alpha: {model._count_alpha.item():.4f}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            #scheduler.step(val_loss)  # Step the scheduler with validation loss
            # Early stopping
            if val_mae < best_mae:
                best_val_loss = val_loss

                best_mae = val_mae
                if epoch >= early_stop_warmup:
                    patience_counter = 0
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                torch.save(model.state_dict(), f'{out_dir}/model.pt')
                best_model_path = f'{out_dir}/model.pt'
                print("Running predictions on validation set")
                labels_p, predictions_p = predict(model, val_loader  , device, multitask=multitask)
                err = _mean_absolute_error(predictions_p, labels_p, f"{out_dir}/best_model_valid_2.png")
                err = _mean_absolute_error(predictions_p*100, labels_p*100, f"{out_dir}/best_model_valid.png")
                print(f"MAE on validation set: {err}")

            else:
                if epoch >= early_stop_warmup:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print('Early stopping triggered')
                        break
            # Plot training and validation loss
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label='Train Loss', marker='o')
        plt.plot(val_losses, label='Validation Loss', marker='s')
        plt.plot(train_maes, label='Train MAE', marker='x')
        plt.plot(val_maes, label='Validation MAE', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/loss_curve.png')  # Save to output directory
        plt.close()  # Close the figure to free memory
            
            # Save lrs to CSV
        pd.DataFrame({'lrs': lrs}).to_csv(f'{out_dir}/lrs.csv', index=False)
        pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'{out_dir}/losses.csv', index=False)

        return {"best_loss" : float(best_val_loss), "best_mae" : float(best_mae) , "best_model" : best_model_path}
        
    # except Exception as e:
    #     print(f"Error during training: {e}")
    #     raise e
    
    #finally:
        # Clean up resources
        try:
            # Close any embedding loaders
            if hasattr(train_dataset, 'embedding_loader') and hasattr(train_dataset.embedding_loader, 'file'):
                if train_dataset.embedding_loader.file is not None:
                    train_dataset.embedding_loader.file.close()
            if hasattr(val_dataset, 'embedding_loader') and hasattr(val_dataset.embedding_loader, 'file'):
                if val_dataset.embedding_loader.file is not None:
                    val_dataset.embedding_loader.file.close()
        except:
            pass
        
        # Delete datasets and loaders
        del train_dataset, val_dataset, train_loader, val_loader
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def cleanup_resources():
    """Clean up resources between training runs"""
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Close any remaining file handles
    try:
        import resource
        print(f"Open file handles: {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except:
        pass

def setup_seed(seed):
    """Set random seed for reproducibility"""
    import os
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

def main():
    setup_seed(42) # Set a global seed for reproducibility
    
    mode = "train"
    # Setup
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #train_loader, val_loader, test_loader = reload_data()
    
    # Initialize and train model
    
    from sklearn.model_selection import KFold
    ## all outdoors ###
    # output_dir = "finetune_dnabert_results/all_outdoors/"
    # biom_file = "../data/new_data/table_all/feature-table.biom"
    # metadata_file = "../data/new_data/metadata_all_outdoor.tsv"
    # embedding_path = "../data/embeddings/al_outdoors.h5"
    # heldout_samples =['D4', 'D17', 'D29', 'D12', 'D6',
    #    'D8', 'D10', 'D13', 'D15', 'D19', 'D21', 'D23', 'D25', 'D27',
    #    'D11']
    ### all outdoors ###
    ## sheds data ###
    output_dir = "../results/multitask_all_nope_bimonth_env_features/"
    biom_file = "../../process_data_all/exported-feature-table/feature-table.biom"
    #biom_file = "../../data/new_data/feature-table/feature-table.biom"
    sheds_file = "../../data/new_data/metadata_sheds.tsv"
    metadata_file = "../../data/new_data/combined_metadata_merged.tsv"
    embedding_path = "../../data/embeddings/all_data.h5"
    heldout_samples = pd.read_csv(sheds_file, sep="\t")['DonorID'].unique().tolist()


    #pd.read_csv(metadata_file)[""]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    seed_values = [42, 1337, 2048]

    # Create a directory for each heldout sample
    for j in range(len(heldout_samples)):
        
        heldout = heldout_samples[j]
        print(f"Running for heldout sample: {heldout}")
        out_dir = os.path.join(output_dir, heldout)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
            

        runs = 3

        #abundance_table, train_data , test_data, embedding_path, distances = load_data(args
        
        
        best_mae = 1000
        multitask = True
        #for fold, (train_idx, val_idx) in enumerate(kf.split(train_samples)):
        for i in range(runs):
            print(f"Run {i+1}/{runs} for heldout sample {heldout}")
            
            # Clean up resources before each run
            cleanup_resources()
            args = Arguments(
            biom_file=biom_file,
            metadata_file=metadata_file,
            tree_path=None,
            embedding_file=embedding_path,
            embedding="DNABERT",
            heldout=heldout
                )
            data_processor = DataProcessor(args)
            data_processor.load_data(multitask=multitask, column="bi_month_name")
            
            # Set a different seed for each run
            # Split train and validation sets
            print("IIIIIIIIIIIII", i)
            setup_seed(seed_values[i])  # Different seed for each run
            model = BasicRegressor(input_dim=128, pe=False)
            result_dir = os.path.join(out_dir, f"run_{i+1}")
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            
            #try:
            res = train_model(
                    model=model,
                    num_epochs=1000,
                    learning_rate=0.00001,
                    device=device,
                    out_dir=result_dir,
                    data_processor=data_processor
                )
            print(res)            
            res["seed"] = seed_values[i]
                
                # Save results
            with open(result_dir + "/res.json", "w") as json_file:
                    json.dump(res, json_file, indent=4)
                    
            # except Exception as e:
            #     print(f"Error in run {i+1}: {e}")
            #     cleanup_resources()
            #     #continue
            
            # finally:
            #     # Clean up after each run
            #     del model
            #     cleanup_resources()



if __name__ == "__main__":
    main()
    

