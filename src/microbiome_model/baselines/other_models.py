import os
import numpy as np
import pandas as pd
import biom
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from evaluate import _mean_absolute_error


def donor_of(sample_id,metadata, delim='.'):
    """Extract donor ID from a sample ID. ADJUST delim to your ID format."""
    sample_to_donor = dict(zip(metadata["sample_name"], metadata["host_subject_id"]))
    return sample_to_donor.get(sample_id, None)

def extract_data_fixed(biom_file_path, target_file_path, test_host_id,
                        use_metadata=True):
    table = biom.load_table(biom_file_path)
    table = table.filter(lambda val, id_, md: val.sum() > 0, axis='sample')
    table = table.subsample(5000, axis='sample', with_replacement=True)

    targets_df = pd.read_csv(target_file_path, sep='\t')
    targets_df = targets_df[targets_df['dataset_type'] == "train"]

    # Drop duplicate sample names up front — prevents silent .loc misalignment
    targets_df = targets_df.drop_duplicates(subset='sample_name')

    sample_targets = dict(zip(targets_df['sample_name'], targets_df['add_0c']))

    # --- Sample IDs present in BOTH table and targets ---
    table_ids = set(table.ids(axis='sample'))
    sample_ids = [i for i in sample_targets if i in table_ids]
    table = table.filter(sample_ids, inplace=False)

    # --- Train/test split by EXACT donor match (not substring) ---
    train_samples = [i for i in sample_ids if donor_of(i, targets_df) != test_host_id]
    test_ids      = [i for i in sample_ids if donor_of(i, targets_df) == test_host_id]

    if len(test_ids) == 0:
        raise ValueError(f"No test samples for donor {test_host_id}. "
                         f"Check donor_of() delimiter.")

    # --- Abundance matrices, explicitly ordered to match sample lists ---
    full_df = table.to_dataframe().T
    train_df = full_df.loc[train_samples]
    test_df  = full_df.loc[test_ids]

    assert list(train_df.index) == train_samples, "Train X/y misalignment"
    assert list(test_df.index)  == test_ids,      "Test X/y misalignment"

    X_train = csr_matrix(train_df.values)
    X_test  = csr_matrix(test_df.values)
    y_train = np.array([sample_targets[i] for i in train_samples])
    y_test  = np.array([sample_targets[i] for i in test_ids])

    # --- Optional metadata: env + bimonth one-hot ---
    if use_metadata:
        meta_df = targets_df[['sample_name', 'bi_month_name', 'env']] \
            .set_index('sample_name')

        assert meta_df.index.is_unique, "Duplicate sample names in metadata"

        meta_df['env'] = (meta_df['env'].str.strip().str.lower()
                          .map({'indoor': 0, 'outdoor': 1}))

        meta_oh = pd.get_dummies(meta_df, columns=['bi_month_name'],
                                 prefix='bi_month')

        missing_train = set(train_samples) - set(meta_oh.index)
        missing_test  = set(test_ids) - set(meta_oh.index)
        assert not missing_train, f"Train IDs missing metadata: {missing_train}"
        assert not missing_test,  f"Test IDs missing metadata: {missing_test}"

        meta_train = meta_oh.loc[train_samples].astype(float)
        meta_test  = meta_oh.loc[test_ids].astype(float)

        X_train = hstack([X_train, csr_matrix(meta_train.values)])
        X_test  = hstack([X_test,  csr_matrix(meta_test.values)])

    return (X_train, y_train), (X_test, y_test)


def random_forest(biom_file_path, target_file_path, heldout_samples,
                   output_dir="rf_results", use_metadata=True):
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for hd in heldout_samples:
        (X_train, y_train), (X_test, y_test) = extract_data_fixed(
            biom_file_path, target_file_path,
            test_host_id=hd, use_metadata=use_metadata,
        )
        y_train = y_train.ravel()
        y_test  = y_test.ravel()

        print(f"{hd}: train {X_train.shape}, test {X_test.shape}")

        rf = RandomForestRegressor(
            n_estimators=500,
            max_features=0.2,
            max_depth=None,
            bootstrap=False,
            criterion='absolute_error',
            random_state=999,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        yhat = rf.predict(X_test)

        # Save predictions for this donor
        pd.DataFrame({'true': y_test, 'predicted': yhat}).to_csv(
            os.path.join(output_dir, f'new_{hd}_predictions.csv'), index=False
        )


        mae = _mean_absolute_error(yhat, y_test,
                                   os.path.join(output_dir, f'new_{hd}.png'))
        print(f"  MAE({hd}) = {mae}  (n_test={len(y_test)})")
        results[hd] = float(mae)

    maes = np.array(list(results.values()))
    print("\n" + "=" * 50)
    print(f"LOBO mean MAE : {maes.mean():.2f} ± {maes.std():.2f}")
    print(f"Median MAE    : {np.median(maes):.2f}")
    print(f"Worst donor   : {max(results, key=results.get)} ({maes.max():.2f})")
    print(f"Best donor    : {min(results, key=results.get)} ({maes.min():.2f})")
    return results


def median_baseline(metadata_file, heldout_samples):
    targets_df = pd.read_csv(metadata_file, sep='\t')
    targets_df = targets_df[targets_df['dataset_type'] == "train"]
    targets_df = targets_df.drop_duplicates(subset='sample_name')
    sample_targets = dict(zip(targets_df['sample_name'], targets_df['add_0c']))
    medians = {}
    results = {}
    lengths = {}
    for hd in heldout_samples:
        test_ids  = [i for i in sample_targets if donor_of(i, targets_df) == hd]
        train_ids = [i for i in sample_targets if donor_of(i, targets_df) != hd]

        y_train = np.array([sample_targets[i] for i in train_ids])
        y_test  = np.array([sample_targets[i] for i in test_ids])

        # TRUE median — optimal constant predictor under MAE
        pred = np.median(y_train)
        yhat = np.full(len(y_test), pred)

        mae = mean_absolute_error(y_test, yhat)
        print(f"Median baseline MAE({hd}) = {mae:.2f}")
        results[hd] = mae
        medians[hd] = pred
        lengths[hd] = len(y_test)

    maes = np.array(list(results.values()))
    print(f"\nMedian baseline mean MAE: {maes.mean():.2f} ± {maes.std():.2f}")
    return results, medians, lengths


if __name__ == "__main__":
    biom_file_path  = '../../process_data_all/rarefied-table-processed.biom'
    target_file_path = '../../data/new_data/combined_metadata_merged.tsv'
    output_dir = 'rf_results'

    heldout_samples = ['D7','D8','D22','D13','D15','D28','D10','D17','D11',
                       'D19','D4','D26','D23','D29','D27','D20','D6','D25',
                       'D30','D5','D21','D18','D14','D12','D24','D9','D16']

    # --- RF with metadata ---
    # results = random_forest(biom_file_path, target_file_path,
    #                         heldout_samples, output_dir=output_dir,
    #                         use_metadata=True)
    # pd.DataFrame.from_dict(results, orient='index', columns=['MAE']) \
    #     .to_csv(os.path.join(output_dir, 'rf_with_metadata.csv'))

    # #--- RF abundance only ---
    # results = random_forest(biom_file_path, target_file_path,
    #                         heldout_samples, output_dir='rf_results_abund',
    #                         use_metadata=False)

    # --- Median baseline ---
    results, medians, lengths = median_baseline(target_file_path, heldout_samples)
    pd.DataFrame.from_dict(results, orient='index', columns=['MAE']) \
        .to_csv(os.path.join(output_dir, 'new_median_baseline.csv'))
    pd.DataFrame.from_dict(medians, orient='index', columns=['Median']) \
        .to_csv(os.path.join(output_dir, 'new_train_medians.csv'))
    pd.DataFrame.from_dict(lengths, orient='index', columns=['Length']) \
        .to_csv(os.path.join(output_dir, 'new_test_lengths.csv'))
