#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:17:21 2019

@author: keerthiraj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from tqdm import tqdm
from helper_functions import diag_load
from helper_functions import RXdistance

from sklearn.metrics import f1_score, confusion_matrix

sim_start_time = time.time()

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

#etas = [5, 7.5, 10] first experiment
#betas = [225, 450, 900] first experiment
#alphas = [0.00001, 0.00005, 0.0001, 0.0005, 0.001] first experiment

#4
#etas = [4, 6, 8, 12, 16]
#betas = [15, 90, 225]
#alphas = [0, 0.00001, 0.00005, 0.0001, 0.1]


etas = [9]
betas = [90]
alphas = [0.00008]

#
#etas = [7.5]
#betas = [225]
#alphas = [0.00005]

# =============================================================================
# Features for each region in the smart grid
# =============================================================================

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
## =============================================================================
## Defining deatures for CorrDet
## =============================================================================
#feat = []
#
#feat_ind = []
#
#for i in range(1):
#     for j in range(data.shape[1]):
#          feat_ind.append(j)
#     feat.append(feat_ind)


# =============================================================================
# Defining arrays to store results
# =============================================================================

#%%
start_data_index = 0
start_test_index = 1800
stop_data_index = 21599 #7599


cv_num = 1


num_of_trials = len(alphas) * len(etas) * len(betas) * cv_num

ecd_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))
y_pred_overall = np.zeros((stop_data_index - train_size, num_of_trials))


y_test_overall = np.zeros((stop_data_index - train_size, num_of_trials))
se_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))


result_table = np.zeros((num_of_trials, 5))

trial_num = 0


overall_f1 = []

overall_cm_dict = {}

for crossvalidation_number in tqdm(range(cv_num), desc = 'CV'):
     
     print(start_data_index, start_test_index, stop_data_index)
     


     for eta_val in tqdm(etas, desc = 'etas'):
          
          for beta_val in tqdm(betas, 'betas'):
               
               for alpha_val in tqdm(alphas, desc = 'alphas'):
     
     
                    cross_valid_exp_num = 1
                    
                    for trail_num_cv in range(cross_valid_exp_num):
                         
#                         rolled_data = np.roll(data, -trail_num_cv * train_size, axis = 0)
#                         rolled_y_data = np.roll(y_data, -trail_num_cv * train_size, axis = 0)
#                         rolled_se_distance = np.roll(se_distance, -trail_num_cv * train_size, axis = 0)
                         
                         x_train = data[start_data_index:start_test_index, :]
                         x_test = data[start_test_index:stop_data_index, :]
                         
                         y_train = y_data[start_data_index:start_test_index]
                         y_test = y_data[start_test_index:stop_data_index]
                         
                         x_train = np.array(x_train)
                         y_train = np.array(y_train)   
                         x_test = np.array(x_test)   
                         y_test = np.array(y_test)
                         
                         
                         se_test = se_distance[start_test_index:stop_data_index]
                         se_test = np.array(se_test)
                         
                    # =============================================================================
                    #      Identifying normal sampels for training
                    # =============================================================================
                         idx_normal = np.where(y_train == 0)[0]
                         idx_normal_train = [x for x in idx_normal if x< train_size]
                         
                         
                         rx_matrix_train = np.zeros((len(x_train), len(feat)))
                         rx_matrix = np.zeros((len(x_test), len(feat)))
                         anomaly_matrix = np.zeros((len(x_test), len(feat)))
                         thr_matrix = np.zeros((len(x_test), len(feat)))
                         
                         #len(feat)
                         
                         num_of_regions = len(feat)
                         
                         for i in range(num_of_regions):
                              
                    # =============================================================================
                    #           #Define training and testing set for each region
                    # =============================================================================
                              
                              curr_train = x_train[:, feat[i]]
                              curr_test = x_test[:, feat[i]]
                              
                              curr_train_all = curr_train
                         
                              curr_train = curr_train[idx_normal_train,:]
                              curr_train = np.transpose(curr_train)
                         
                              
                              curr_mu = np.mean(curr_train, axis = 1) # initial mean
                              curr_cov = np.cov(curr_train, rowvar=True) # initial covariance
                              curr_cov = diag_load(curr_cov) # load the matrix diagonal
                              curr_icov = LA.pinv(curr_cov) # initial inverse covariance
                              
                    # =============================================================================
                    #         Training decision scores  
                    # =============================================================================
                              curr_rxdist_train = []
                              
                              for x_sample in curr_train_all:               
                                  rxdist = RXdistance(x_sample, curr_mu, curr_icov)              
                                  curr_rxdist_train.append(rxdist[0][0])
                              
                              curr_rxdist_train = np.array(curr_rxdist_train)
                              rx_matrix_train[:,i] = curr_rxdist_train
                         
                              rxdist_all = list(curr_rxdist_train)
                              
                              rx_anomaly = []
                              curr_rxdist = []
                              thr_list = []
                              
                              rxdist_normal = list(curr_rxdist_train[idx_normal_train])
                              
                              thr_update_list_len = beta_val
                              
                              alpha = alpha_val
                              
                              
                              for k, x_sample in enumerate(curr_test):
                                   
                                  rxdist = RXdistance(x_sample, curr_mu, curr_icov)              
                                  curr_rxdist.append(rxdist[0][0])             
                                  curr_thr = np.mean(rxdist_normal[-thr_update_list_len:]) + eta_val * np.std(rxdist_normal[-thr_update_list_len:])
                                  thr_list.append(curr_thr)
                                               
                                  if rxdist < curr_thr:
                                       
                                       if alpha != 0:
                                            
                                            mu = (1-alpha) * curr_mu + alpha * x_sample
                                            
                                            icov = (1/(1-alpha)) * (curr_icov - ( (x_sample - mu) * np.transpose((x_sample*mu)) / ( ((1-alpha)/alpha) + np.transpose((x_sample*mu)) * (x_sample - mu))))
                                        
                                            curr_mu = mu
                                            curr_icov = icov
                                       
                                       rx_anomaly.append(0)
                                       rxdist_normal.append(rxdist)                   
                                  else:
                                       rx_anomaly.append(1)
                                       
                                  rxdist_all.append(rxdist)
                                  
                              thr_list = np.array(thr_list)
                              thr_matrix[:,i] = thr_list
                                           
                              curr_rxdist = np.array(curr_rxdist)
                              rx_matrix[:,i] = curr_rxdist                    
                              rx_anomaly = np.array(rx_anomaly)            
                              anomaly_matrix[:,i] = rx_anomaly
                                       
                         rx_full = np.concatenate((rx_matrix_train, rx_matrix), axis = 0)    
                         an = np.sum(anomaly_matrix, axis = 1)
                         
                    # =============================================================================
                    #      Prediction
                    # =============================================================================
                         y_pred = an > 0
                         y_pred = y_pred*1
                         
                         y_pred = np.array(y_pred)
                         
                         ecd_score_daily = []
                    
                         for m in range(len(y_test)):
                              if y_pred[m] == 0:
                                   ecd_score_daily.append(np.min(rx_matrix[m,:]))
                              else:
                                   ecd_score_daily.append(np.max(rx_matrix[m,:]))
                         
                         ecd_score_daily = np.array(ecd_score_daily)
                    
                         ecd_decision_scores[:,trial_num] = ecd_score_daily
                         
                         se_decision_scores[:,trial_num] = se_test
                         
                         y_test_overall[:,trial_num] = y_test
                         
                         y_pred_overall[:,trial_num] = y_pred     
                         
                         
                         save_rx_file_name = 'experiment_csv_files/rx_matrix_' + str(crossvalidation_number) + '_' + str(eta_val) + '.csv'
                         save_thr_file_name = 'experiment_csv_files/thr_matrix_' + str(crossvalidation_number) + '_' + str(eta_val) +'.csv'
                         save_an_file_name = 'experiment_csv_files/an_matrix_' + str(crossvalidation_number) + '_' + str(eta_val) +'.csv'
                         
                         rx_matrix_pd = pd.DataFrame(rx_matrix)
                         rx_matrix_pd.to_csv(save_rx_file_name)
                         
                         
                         thr_matrix_pd = pd.DataFrame(thr_matrix)
                         thr_matrix_pd.to_csv(save_thr_file_name)
                         
                         an_matrix_pd = pd.DataFrame(anomaly_matrix)
                         an_matrix_pd.to_csv(save_an_file_name)
                         
                         f1 = f1_score(y_test, y_pred)
                         overall_f1.append(f1)
                         
                         
                         cm = confusion_matrix(y_test, y_pred)
                         overall_cm_dict[crossvalidation_number] = cm
                         
                         
                         curr_result = [eta_val, beta_val, alpha_val, crossvalidation_number, f1]
                         
                         print('eta, beta, alpha, f1 : ', curr_result)
                         print('CM : ', cm)
                         result_table[trial_num,:] = [eta_val, beta_val, alpha_val, crossvalidation_number, f1]
                         
                         trial_num += 1
                         
                         
                         # start_data_index += 1000
                         # start_test_index += 1000
                         # stop_data_index += 1000
                         
print(overall_f1)
print(overall_cm_dict)
# =============================================================================
# Saving results to CSV file
# =============================================================================
ecdt_dailyload_pd = pd.DataFrame(ecd_decision_scores)
ecdt_dailyload_pd.to_csv('ECD-AS_decision_scores_dailyload_cv_exp.csv')


ecdt_dailyload_f1_pd = pd.DataFrame(overall_f1)
ecdt_dailyload_f1_pd.to_csv('ECD-AS_f1_scores_dailyload_cv_exp.csv')

result_table_pd = pd.DataFrame(result_table)
result_table_pd.to_csv('ECD-AS_result_table_dailyload_cv_exp.csv')

y_pred_pd = pd.DataFrame(y_pred_overall)
y_pred_pd.to_csv('ECD-AS_y_pred_overall_dailyload_cv_exp.csv')


result_table_pd = pd.DataFrame(result_table)
result_table_pd.to_csv('ECD-AS_result_table_dailyload_cv_exp.csv')


y_test_pd = pd.DataFrame(y_test_overall)
y_test_pd.to_csv('ECD-AS_y_test_overall_dailyload_cv_exp.csv')

se_dailyload_pd = pd.DataFrame(se_decision_scores)
se_dailyload_pd.to_csv('SE_decision_scores_dailyload_cv_exp.csv')

     
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time

print('Total simulation duration', sim_time) 