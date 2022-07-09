#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:09:07 2020

@author: keerthiraj
"""


# =============================================================================
# Importing necessary libraries
# =============================================================================

import sys
sys.path.append('../helper files')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
from numpy import linalg as LA
from tqdm import tqdm

from helper_functions import diag_load, plot_confusion_matrix, RXdistance, class_metrics
from sklearn.metrics import f1_score, confusion_matrix

sim_start_time = time.time()


# In[]

# =============================================================================
# # Reading CSV files
# =============================================================================

mitm = pd.read_csv('../data csv files/PC_mitm_21599_v1_point1percent_noise.csv') # IAT Injection
err_ins = pd.read_csv('../data csv files/err_ins_dos.csv')
z = pd.read_csv('../data csv files/z_dos.csv')

# In[]
# =============================================================================
# Defining hyper parameters
# =============================================================================
train_size = 1800  #2 hours worth of samples for initial training

etas = [7]
betas = [90]
alphas = [0.00008]

cm_title = "ECD-AS MITM PC" 
# =============================================================================
# Measurements for each region in the smart grid
# =============================================================================
# In[]

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
          if ed[i][0] == j or ed[i][1] == j:
               feat_ind.append(ed[i][2])     
     feat_dict[j] = feat_ind  
     feat.append(feat_ind)     


# In[]

# =============================================================================
# # Preparing the dataset
# =============================================================================

orz = mitm
orz = orz.drop(orz.columns[0], axis=1) # remove column 0 (the feature numbers)
orz_arr = np.array(orz)

# In[]
# =============================================================================
# #Removing zero varinace features and normalizing individual datasets
# =============================================================================

# MITM data

data_sg = np.transpose(orz_arr)
cov_ind = np.var(data_sg, axis = 0)
zero_variance_features = np.where(cov_ind == 0)
zero_variance_features = list(zero_variance_features[0])

df_data_sg = pd.DataFrame(data_sg)
df_data_sg.drop(df_data_sg.columns[zero_variance_features],axis=1,inplace=True)
data_sg = np.array(df_data_sg)

#Normalize
for i in range(data_sg.shape[1]):
     data_sg[:,i] = (data_sg[:,i] - np.mean(data_sg[:,i]))/(np.std(data_sg[:,i]))


data_sg = np.transpose(data_sg)


# In[]
# =============================================================================
# # Getting ground truth
# =============================================================================
err_ins = err_ins.drop(err_ins.columns[0], axis=1)  # remove column 0 (the feature numbers)
err_in = np.array(err_ins.values) # convert to numpy array


y_data = []

for i in range(len(err_in[0])):
    if (err_in[0][i] > 0):
        y_data.append(1)
    else:
        y_data.append(0)

y_data = np.array(y_data)


# In[]
# =============================================================================
# Defining arrays to store results
# =============================================================================

data = data_sg

start_data_index = 0
start_test_index = 1800
stop_data_index = 21599 #12599 #7599


cv_num = 10


num_of_trials = len(alphas) * len(etas) * len(betas) * cv_num

ecd_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))
y_pred_overall = np.zeros((stop_data_index - train_size, num_of_trials))


y_test_overall = np.zeros((stop_data_index - train_size, num_of_trials))
se_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))


result_table = np.zeros((num_of_trials + 2, 12))

trial_num = 0

# data = np.transpose(data)

overall_f1 = []
overall_cm_dict = {}


eta_val = etas[0]
beta_val = betas[0]
alpha_val = alphas[0]


for crossvalidation_number in tqdm(range(cv_num), desc = 'CV'):
     
     cm_title_1 = cm_title + '_' + str(crossvalidation_number)
     print(start_data_index, start_test_index, stop_data_index)
     
     for eta_val in tqdm(etas, desc = 'etas'):
          
          for beta_val in tqdm(betas, 'betas'):
              
              for alpha_val in tqdm(alphas, desc = 'alphas'):
                  
                  x_train = data[start_data_index:start_test_index, :]
                  x_test = data[start_test_index:stop_data_index, :]
                 
                  y_train = y_data[start_data_index:start_test_index]
                  y_test = y_data[start_test_index:stop_data_index]
                 
                  x_train = np.array(x_train)
                  y_train = np.array(y_train)   
                  x_test = np.array(x_test)   
                  y_test = np.array(y_test)
     
# =============================================================================
#      Identifying normal sampels for training
# =============================================================================
                  idx_normal = np.where(y_train == 0)[0]
                  idx_normal_train = [x for x in idx_normal if x< train_size]
                 
                 
                  rx_matrix_train = np.zeros((len(x_train), len(feat)))
                  rx_matrix = np.zeros((len(x_test), len(feat)))
                  anomaly_matrix = np.zeros((len(x_test), len(feat)))
                  thr_matrix = np.zeros((len(x_test), len(feat)))
     
     
                  num_of_regions = len(feat)
     
                  for i in range(num_of_regions):
                      
# =============================================================================
## Define training and testing set for each region
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
# Prediction
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
                 
                 
                  y_test_overall[:,trial_num] = y_test
                  y_pred_overall[:,trial_num] = y_pred     
                 
                 
                  f1 = f1_score(y_test, y_pred)
                  overall_f1.append(f1)
                 
                 
                  cm = confusion_matrix(y_test, y_pred)
                  tn, fp, fn, tp = cm.ravel()
                  acc, prec, recall, f1 = class_metrics(cm)
                 
                  target_names = ['Normal', 'Anomalies']
                  plot_confusion_matrix(cm, target_names, title=cm_title_1)
                 
                  overall_cm_dict[crossvalidation_number] = cm
                 
                 
                  curr_result = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                 
                  print('eta, beta, alpha, tn, fp, fn, tp, acc, prec, recall, f1: ', curr_result)
                 
                  result_table[trial_num,:] = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                 
                  trial_num += 1
                           
                  start_data_index += 1000
                  start_test_index += 1000
                  stop_data_index += 1000



                         
print(overall_f1)
print(overall_cm_dict)
# =============================================================================
# Saving results to CSV file
# =============================================================================

file_title = './local_csv_files/' + cm_title 


result_table[trial_num,:] = np.mean(result_table[:num_of_trials,:], axis = 0)
trial_num += 1
result_table[trial_num,:] = np.std(result_table[:num_of_trials,:], axis = 0)
result_table_pd = pd.DataFrame(result_table)
result_table_pd.to_csv(file_title + 'result_table' +  '.csv')

     
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time

print 'Total simulation duration', sim_time 
