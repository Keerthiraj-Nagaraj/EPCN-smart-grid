#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 05 06:20:26 2020

@author: keerthiraj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from tqdm import tqdm
from helper_functions_tecd import diag_load
from helper_functions_tecd import RXdistance, class_metrics

from sklearn.metrics import f1_score, confusion_matrix

sim_start_time = time.time()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#%%

# =============================================================================
# # Reading CSV files
# =============================================================================
orz = pd.read_csv('../csv files/orz.csv')
err_ins = pd.read_csv('../csv files/err_ins.csv')
err_id = pd.read_csv("../csv files/err_id.csv", header = None)
se_dist = pd.read_csv("../csv files/se_dist.csv")
z = pd.read_csv('../csv files/z.csv')

# =============================================================================
# Defining training size
# =============================================================================
train_size = 1800  #2 hours for initial training


etas = [9]
betas = [90]
alphas = [0.00008]

eta_val = etas[0]
alpha_val = alphas[0]
beta_val = betas[0]


z = z.drop(z.columns[0], axis=1) # remove column 0 (the feature numbers)
z_arr = np.array(z)


ed = []

for i in range(len(z)):
     ed.append([int(z_arr[i][1]), int(z_arr[i][2]), i])

feat = []

feat_dict = {}


for j in range(1,119):
     
     feat_ind = []
     
     for i in range(len(ed)):
          if ed[i][0] == j:
               feat_ind.append(ed[i][2])
     
     feat_dict[j] = feat_ind  
     feat.append(feat_ind)     


#%%
# =============================================================================
# # Preparing the dataset
# =============================================================================
orz = orz.drop(orz.columns[0], axis=1) # remove column 0 (the feature numbers)
orz_arr = np.array(orz)

# =============================================================================
# #Removing zero varinace features
# =============================================================================

data = np.transpose(orz_arr)
cov_ind = np.var(data, axis = 0)
zero_variance_features = np.where((cov_ind == 0))
zero_variance_features = list(zero_variance_features[0])

df_data = pd.DataFrame(data)
df_data.drop(df_data.columns[zero_variance_features],axis=1,inplace=True)
data = np.array(df_data)

# =============================================================================
# #Normalize
# =============================================================================

for i in range(data.shape[1]):
     data[:,i] = (data[:,i] - np.mean(data[:,i]))/(np.std(data[:,i]))

# =============================================================================
# # Getting ground truth
# =============================================================================
err_ins = err_ins.drop(err_ins.columns[0], axis=1)  # remove column 0 (the feature numbers)
err_in = np.array(err_ins.values) # convert to numpy array

# =============================================================================
# #State estimator predictions and Ground truth for training 
# =============================================================================
err_id = np.array(err_id.iloc[1,1::].values)

y_se = [0] * len(err_id)
y_se = np.array(y_se)

for i,val in enumerate(err_id):
    if val > 0:
        y_se[i] = 1
    

se_dist = se_dist.drop(se_dist.columns[0], axis=1)  # remove column 0 (the feature numbers)
se_d = np.array(se_dist.values) # convert to numpy array
se_d = np.transpose(se_d)
se_distance = se_d
se_distance = se_distance.reshape(len(se_distance),)


y_data = []

for i in range(len(err_in[0])):
    if (err_in[0][i] > 0):
        y_data.append(1)
    else:
        y_data.append(0)

y_data = np.array(y_data)


#%%

print('classifiers')
# =============================================================================
# Defining model parameters

names = ["KNN-1", "KNN-5", "KNN-10", "KNN-20", "Naive Bayes", "SVC", "mlpnn", "AdaBoost" ]

classifiers = [
    KNeighborsClassifier(n_neighbors = 1, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 5, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 10, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 20, weights = 'distance'),
    
    GaussianNB(var_smoothing=1e-06),               
    SVC(kernel="rbf", gamma= 'auto', probability = True, class_weight = 'balanced', 
        verbose = True, decision_function_shape = 'ovo'),
    MLPClassifier(hidden_layer_sizes=(250, 100, 50), activation='relu', solver='adam', alpha=1, max_iter=1000),
    AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=500) , n_estimators=500, random_state=0)]

names = ["mlpnn"]

classifiers = [
    MLPClassifier(hidden_layer_sizes=(250, 100, 50), activation='relu', solver='adam', alpha=1, max_iter=1000)]
# =============================================================================

#%%



# names = ["KNN-3"]

# classifiers = [
#     KNeighborsClassifier(n_neighbors = 3, weights = 'distance')]

cv_num = 15


num_of_trials = len(names) * len(alphas) * len(etas) * len(betas) * cv_num


for name, clf in tqdm(zip(names, classifiers), desc = 'classifier'):
    
    result_table = []
    
    trial_num = 0
    
    start_data_index = 0
    start_test_index = 1800
    stop_data_index = 7599 #21599 #7599

    for crossvalidation_number in tqdm(range(cv_num), desc = 'CV'):
         
         # print(start_data_index, start_test_index, stop_data_index)
                                  
         x_train = data[start_data_index:start_test_index, :]
         x_test = data[start_test_index:stop_data_index, :]
         
         y_train = y_data[start_data_index:start_test_index]
         y_test = y_data[start_test_index:stop_data_index]
         
         x_train = np.array(x_train)
         y_train = np.array(y_train)   
         x_test = np.array(x_test)   
         y_test = np.array(y_test)

         clf.fit(x_train, y_train)
         y_pred = clf.predict(x_test)
           
         cm = confusion_matrix(y_test, y_pred)

         acc, prec, rec, f1 = class_metrics(cm)

         result_table.append([float(acc), float(prec), float(rec), float(f1)])
         
         trial_num += 1
         
         
         start_data_index += 1000
         start_test_index += 1000
         stop_data_index += 1000
                
         
    result_table = np.array(result_table)
    means_res = np.mean(result_table, axis = 0)
    means_res = means_res.reshape(1,len(means_res))
    
    std_res = np.std(result_table, axis = 0)
    std_res = std_res.reshape(1,len(std_res))
    
    full_result_table = np.concatenate((result_table, means_res), axis = 0)
    full_result_table = np.concatenate((full_result_table, std_res), axis = 0)
    
    result_table_pd = pd.DataFrame(full_result_table)
    csv_ml_filename = 'ml_' + name + '_result_table_dailyload_cv.csv'
    result_table_pd.to_csv(csv_ml_filename)

# =============================================================================
# Saving results to CSV file
# =============================================================================




sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time

print('Total simulation duration', sim_time) 