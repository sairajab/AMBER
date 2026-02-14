import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import biom
import pandas as pd
from sklearn.model_selection import train_test_split
from evaluate import _mean_absolute_error
from sklearn.model_selection import PredefinedSplit

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)
        self.max_len = 0
    
    def fit(self, train_loader):
        X, y = self._flatten_dataloader(train_loader)
        self.model.fit(X, y)

    def predict(self, loader):
        X, _ = self._flatten_dataloader(loader)
        preds = self.model.predict(X)
        return preds

    def evaluate(self, loader, out_dir):
        X, y_true = self._flatten_dataloader(loader)
        preds = self.model.predict(X)
        print(out_dir)
        mae = _mean_absolute_error(preds *100,y_true*100,  out_dir + "/test.png")
        return mae

    def _flatten_dataloader(self, loader):
        # Collapse batched embeddings using mean pooling
        X, y = [], []

        for batch in loader:
            emb = batch['embeddings']  # [B, L, D]
            mask = batch['abundances']#.squeeze()  # [B, L, 1]
            
            pooled = (emb * mask)#.sum(dim=2) / mask.sum(dim=2).clamp(min=1e-6)

            X.append(pooled.cpu().numpy())
            y.append(batch['outdoor_add_0'].cpu().numpy())
            print("batching")
        
        # max_len = max(arr.shape[1] for arr in X)
        # print("Max length ", max_len)
        # padded_arrays = []
        # for arr in X:
        #     pad_width = ((0, 0), (0, max_len - arr.shape[1]), (0, 0))  # pad only dim 1
        #     padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=0)
        #     padded = padded.sum(dim=2)
        #     padded_arrays.append(padded)
        
        X = np.vstack(X)
        X = X.sum(axis=2)
        if self.max_len == 0:
            self.max_len = X.shape[1]
        print("Max length ", self.max_len)
        print("X shape ", X.shape)
        pad_len = self.max_len - X.shape[1]  # how much to pad along the embedding dimension

        if pad_len > 0:
            pad_width = ((0, 0), (0, pad_len))  # pad axis=1 (columns)
            padded = np.pad(X, pad_width=pad_width, mode='constant', constant_values=0)
            X = padded
            
        y = np.concatenate(y).ravel()
        print("X shape ", X.shape,)
        print("y shape ", y.shape)
        return X, y


# Function to extract data from DataLoader into NumPy arrays
def extract_data(biom_file_path,target_file_path, test_host_id = "D12"):
    
    table = biom.load_table(biom_file_path)
    table = table.filter(lambda val, id_, md: val.sum() > 0, axis='sample')
    table = table.subsample(5000, axis='sample', with_replacement=True)
    # abundance_data = table.to_dataframe()
    
    
    targets_df = pd.read_csv(target_file_path)
    sample_targets = dict(zip(targets_df['SampleID'], targets_df['outdoor_add_0']))
    
    test_ids = []
    train_samples = []

    sample_ids = list(sample_targets.keys())
    
    for id in sample_ids:
        if test_host_id in id:
            test_ids.append(id)
        else:
            train_samples.append(id) 
    
    X_train, y_train = [], []
    
    
    train_val_table = table.filter(train_samples, inplace = False)
    
    train_ids, val_ids = train_test_split(train_samples, test_size=0.1, random_state=42)



    # print("Train ids ", train_ids)
    # print("Val ids ", val_ids)
    
    train_table = train_val_table.filter(train_ids, inplace = False)

    for train_id in train_ids:

    
        targets = sample_targets[train_id] / 100.0  # Normalize by 100
        y_train.append(targets)  # Convert tensor to NumPy
        
    abundances = train_table.to_dataframe().T
    X_train.append(abundances)  # Convert tensor to NumPy 
    
    X_val, y_val = [], []
    
    val_table = train_val_table.filter(val_ids, inplace = False)
    
    for val_id in val_ids:
        #abundances = abundance_data[val_id]
        targets = sample_targets[val_id] / 100.0
        
        #X_val.append(abundances)  # Convert tensor to NumPy
        y_val.append(targets)  # Convert tensor to NumPy
    X_val.append(val_table.to_dataframe().T)  # Convert tensor to NumPy
    
    X_test, y_test = [], []
    
    test_table = table.filter(test_ids, inplace = False)
    
    #test_table = test_table.subsample(5000, axis='sample', with_replacement=True)
    for test_id in test_ids:

        #abundances = abundance_data[test_id]
        targets = sample_targets[test_id] / 100.0

    
        #X_test.append(abundances)  # Convert tensor to NumPy
        y_test.append(targets)  # Convert tensor to NumPy
    X_test.append(test_table.to_dataframe().T)  # Convert tensor to NumPy
    
    return (np.vstack(X_train), np.vstack(y_train)), (np.vstack(X_val), np.vstack(y_val)), (np.vstack(X_test), np.vstack(y_test))

def random_forest(biom_file_path,target_file_path, output_dir = "rf_results"):
    
    import os
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    param_grid = {"max_depth": [None, 4],
              "max_features": ['auto', 0.2],
              "bootstrap": [True, False]}
    
    heldout_samples = ['D7','D8','D22','D13','D15', 'D28', 'D10', 'D17', 'D11''D19','D4','D26','D23','D29','D27','D20','D6','D25','D30',
                        'D5','D21','D18','D14','D12','D24','D9','D16']
    
    results = {}
    # Create a directory for each heldout sample
    for j in range(len(heldout_samples)):
        
        xy_train, xy_val , xy_test = extract_data(biom_file_path,target_file_path, test_host_id = heldout_samples[j])
        X_train, y_train = xy_train
        X_test, y_test = xy_test
        X_val, y_val = xy_val
        print("Train data size ", X_train.shape , y_train.shape, len(X_train))
        print("Test data size ", X_test.shape , y_test.shape)
        print("Val data size ", X_val.shape , y_val.shape)
        
        X_all = np.concatenate([X_train, X_val])
        y_all = np.concatenate([y_train, y_val])
        
        # Create test_fold array: -1 for train, 0 for val
        test_fold = [-1] * len(X_train) + [0] * len(X_val)
        
        ps = PredefinedSplit(test_fold)
        

        

        rf = RandomForestRegressor(n_estimators=500, random_state=999, criterion='absolute_error')
        #grid search cv using rf and the hyperparameter grid on the inner_cv training set
        rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=ps, n_jobs=-1, scoring='neg_mean_absolute_error')
        print("Running grid search ... ")
        rf_grid.fit(X_all, y_all)
        
        res = ",".join(("{}={}".format(*i) for i in rf_grid.best_params_.items()))

        print(res)
        
        yhat = rf_grid.predict(X_test)
        #results = pd.DataFrame(y.iloc[test_ids])
        #results['predicted_add'] = np.array(yhat)
        # results.index.name = None
        # print(results.to_csv(header=False), file=f)
        MAE = _mean_absolute_error(yhat*100, y_test*100, f'heldout_samples[j].png')
        print(f"Prediction score (MAE): {heldout_samples[j]}" ,MAE)
        results[heldout_samples[j]] = MAE
        
    print("Results: ", results)
    return results

if __name__ == "__main__":
    biom_file_path = '../filtered_table.biom'
    target_file_path = '../data/target_df.csv'
    output_dir='rf_results'
    # Run the random forest model
    results = random_forest(biom_file_path, target_file_path, output_dir=output_dir)
    # Print the results
    print("Final results: ", results)
    # Save the results to a CSV file
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['MAE'])
    results_df.to_csv(output_dir + '/rf_results.csv', index=True)

