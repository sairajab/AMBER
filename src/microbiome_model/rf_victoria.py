#!/usr/bin/env python

from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import LeaveOneGroupOut, KFold, GridSearchCV, GroupKFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import make_regression
from scipy.stats import randint as sp_randint
from matplotlib import pyplot as plt
from collections import defaultdict

# qiime imports
import qiime2
from qiime2 import Artifact, Metadata

# General Tool Imports
import numpy as np
import pandas as pd
import collections
from pickle import load, dump
import warnings
import biom
import scipy
import joblib



#Import data
sheds_data = biom.load_table('../data/table_sheds_dada2.biom').to_dataframe()
print('data shape before filtering samples', sheds_data.shape)
sheds_metadata = Metadata.load('../metadata_sheds.tsv').to_dataframe()

# filter metadata to only contain samples in the data table
sheds_metadata = sheds_metadata.loc[sheds_metadata.index.isin(sheds_data.columns)].copy()

# add metadata
sheds_metadata['env']= 'indoor'
sheds_metadata.loc[sheds_metadata['indoor_outdoor'].str.contains('outdoor', na=False), 'env']= 'outdoor'
sheds_metadata['facility']= 'STAFS'

# filter samples
sheds_metadata = sheds_metadata.loc[sheds_metadata['body_site'].isin(['R.skin.face', 'R.skin.hip'])].copy()

print('data shape before filtering samples', sheds_data.shape)
#filter data table and remove columns with zeros
sheds_data = sheds_data[sheds_metadata.index.tolist()]
print('data shape before removing zero sum features', sheds_data.shape)
sheds_data = sheds_data.loc[sheds_data.sum(axis=1) != 0, :]

print(sheds_metadata.DonorID.value_counts())
print(sheds_metadata.sample_type.value_counts())
print(sheds_metadata.indoor_outdoor.value_counts())
print(sheds_metadata.sequencing_run.value_counts())
print('unique fts', sheds_data.shape[0])

#get data in correct format
sheds_matrix = scipy.sparse.csr_matrix(sheds_data.T.values)
# reset metadata index to match sheds data just in case order matters
sheds_metadata = sheds_metadata.reindex(index = sheds_data.columns.tolist())

# build model
X = sheds_matrix
y = sheds_metadata['outdoor_add_0']
y = (y.astype(float))
print("Y shape", y.shape, sheds_data.shape)

groups = sheds_metadata['DonorID']

# outer_cv creates folds by leave-one-body-out for estimating generalization error
# the number of folds is the number of bodies
outer_cv = LeaveOneGroupOut().split(X, y, groups=groups)

# prints the number of folds in the outer loop
# another sanity check, should be equal to number of bodies
print("Number of outer folds to perform: ", LeaveOneGroupOut().get_n_splits(X, y, groups=groups))

results = sheds_metadata.copy()
importances_df = pd.DataFrame(index = sheds_data.index)

count=1
for train_ids, test_ids in outer_cv:
    mean_y = 0
    rf = RandomForestRegressor(n_estimators=500, 
    						   max_features=0.2, 
    						   max_depth= None, 
    						   random_state=999, 
    						   criterion='absolute_error', 
    						   bootstrap=False,
                               n_jobs=-1)
    print(X[train_ids, :].shape, y.iloc[train_ids].shape, X[test_ids,:].shape, y.iloc[test_ids].shape)
    print(X[train_ids, :].toarray()[0:5,0:5], y.iloc[train_ids].to_list()[0:5])
    rf.fit(X[train_ids, :], y.iloc[train_ids])
    yhat = rf.predict(X[test_ids,:])
    results.loc[results.index[test_ids], 'predicted_add']=np.array(yhat)
    mean_y += yhat.mean()
    print("mean ", mean_y)
    print("MAE: ", mean_absolute_error(y.iloc[test_ids], yhat))
    print(yhat)
    # Determine important features
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
   
    general_importances = []
    subcount = 0
    for i in indices:
        general_importances += (importances_df.index[i], importances[indices[subcount]])
        subcount += 1

    general_importances_df = pd.DataFrame(np.array(general_importances).reshape(int(len(general_importances)/2),2))
    general_importances_df = general_importances_df.rename({0: "feature", 1: "importance"},axis='columns')
    general_importances_df.set_index('feature', inplace=True)
    importances_df['importance_{}'.format(count)]=general_importances_df['importance']
    print(count)
    count+=1

    
#results.to_csv('indoor_outdoor_output_2.csv')
#importances_df.to_csv('indoor_outdoor_imporances_2.csv')