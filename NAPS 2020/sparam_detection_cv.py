#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  20 23:57:01 2020

@author: keerthiraj
"""

print('sparam_detection_CV')
print('')



# =============================================================================
# Importing necessary libraries
# =============================================================================

# ECD functions

import sys
sys.path.append('../helper files')
from helper_functions import diag_load, plot_confusion_matrix, RXdistance, class_metrics

#%%

# basic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from numpy import linalg as LA

import time
from tqdm import tqdm
warnings.filterwarnings("ignore") 
#%%

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

#%%

def df_to_array(dataframe):
    
    arr = dataframe.drop(dataframe.columns[0], axis = 1)
    arr = np.array(arr)
    
    arr = np.transpose(arr)
           
    return arr

#%%
sim_start_time = time.time()

# Setting global parameters


#%%

train_size_fdi_percen = 0.6
train_size_param_percen = 0.15

etas = [10.0]
betas = [450]
alphas = [8e-5]

eta_val = etas[0]
beta_val = betas[0]
alpha_val = alphas[0]

suffix_name = '_eta_10_beta_900_alpha_8e_5_CV'

#%%

print('Spatial connections for the grid')

z = pd.read_csv('../data csv files latest/z_ml.csv')

z = z.drop(z.columns[0], axis=1) # remove column 0 (the feature numbers)
z_arr = np.array(z)

ed = []

for i in range(len(z)):
     ed.append([int(z_arr[i][1]), int(z_arr[i][2]), i])


number_of_regions = 118

feat = []
feat_dict = {}
feat_len = []

for j in range(1,number_of_regions+1):
     feat_ind = []
     for i in range(len(ed)):
          if ed[i][0] == j:
               feat_ind.append(ed[i][2])     
     feat_dict[j] = feat_ind  
     feat.append(feat_ind)  
     feat_len.append(len(feat_ind))

#%%

# =============================================================================
# Importing and preparing test day data and historical data
# =============================================================================

print('Importing and preparing test day data and historical data...')


orz_pd = pd.read_csv('../data csv files latest/att_01-08-2018_sparam.csv')

orz_pd_1d = pd.read_csv('../data csv files latest/orz_01-07-2018.csv')
orz_pd_2d = pd.read_csv('../data csv files latest/orz_01-06-2018.csv')
orz_pd_3d = pd.read_csv('../data csv files latest/orz_01-05-2018.csv')
orz_pd_4d = pd.read_csv('../data csv files latest/orz_01-04-2018.csv')
orz_pd_5d = pd.read_csv('../data csv files latest/orz_01-03-2018.csv')
orz_pd_6d = pd.read_csv('../data csv files latest/orz_01-02-2018.csv')
orz_pd_7d = pd.read_csv('../data csv files latest/orz_01-01-2018.csv')

# orz_pd_1y = pd.read_csv('../data csv files latest/orz_01-02-2017.csv')
# orz_pd_2y = pd.read_csv('../data csv files latest/orz_01-04-2016.csv')
#%%

se_residual_pd = pd.read_csv('../data csv files latest/sparam_residual.csv')
err_in_pd = pd.read_csv('../data csv files latest/err_ins_sparam.csv')


se_score = pd.read_csv('../data csv files latest/sparam_se_score.csv')
se_threshold = pd.read_csv('../data csv files latest/sparam_se_threshold.csv')


#%%

line_buses_sparam = pd.read_csv('../data csv files latest/line_buses_sparam.csv')
line_buses = df_to_array(line_buses_sparam)

#%%

print('Converting dataframe to arrays ... ')

orz = df_to_array(orz_pd)

orz_1d = df_to_array(orz_pd_1d)
orz_2d = df_to_array(orz_pd_2d)
orz_3d = df_to_array(orz_pd_3d)
orz_4d = df_to_array(orz_pd_4d)
orz_5d = df_to_array(orz_pd_5d)
orz_6d = df_to_array(orz_pd_6d)
orz_7d = df_to_array(orz_pd_7d)
# orz_1y = df_to_array(orz_pd_1y)
# orz_2y = df_to_array(orz_pd_2y)

#%%
se_residual = df_to_array(se_residual_pd)
se_residual = se_residual[:,:orz.shape[1]]

err_in = df_to_array(err_in_pd)

#%%

print('SE predictions')

se_score = df_to_array(se_score)
se_threshold = df_to_array(se_threshold)

se_pred = np.zeros((len(se_score),))
se_pred[se_score.reshape(len(se_score),) > se_threshold[0][0]] = 1


#%%

# full_orz = np.dstack((orz, orz_1d, orz_2d, orz_3d, orz_4d, orz_5d, orz_6d, orz_7d, orz_1y, orz_2y))
full_orz = np.dstack((orz, orz_1d, orz_2d, orz_3d, orz_4d, orz_5d, orz_6d, orz_7d))


#%%

print('bus wise multiple target regression with multiple linear regresor....')

result_len = 3

mod_results = np.zeros((number_of_regions, result_len))

curr_day_prev_sample = 1

ml_est = np.zeros(orz.shape)
ml_res = np.zeros(orz.shape)


for bus_number in tqdm(range(number_of_regions), desc = 'buses'):
   
    # =============================================================================
    # # Data manipulation to get training and testing sets
    # =============================================================================
    
    x_full_currbus = full_orz[:,feat[bus_number],:]
    
    ydata_currbus = x_full_currbus[:,:, 0]  #current day measurement as targets
    
    xdata_currbus = x_full_currbus[:,:,1:] #previous 7 day measurements as inputs
    
    xdata_currbus_reshaped = xdata_currbus.reshape(xdata_currbus.shape[0], xdata_currbus.shape[1] * xdata_currbus.shape[2])
    
    
    
    xtrain, xtest, ytrain, ytest = train_test_split(xdata_currbus_reshaped, ydata_currbus, test_size=(1-train_size_fdi_percen), 
                                                    shuffle = False, random_state=42)
    
    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    
    # =============================================================================
    # Model fitting and prediction
    # =============================================================================
    
    reg = MultiOutputRegressor(LinearRegression(fit_intercept=True, normalize=False)).fit(xtrain_scaled, ytrain)
    ypred = reg.predict(xtest_scaled)
    
    
    weight_matrix = np.zeros((ytest.shape[1], xtest.shape[1])) 
    # ytest.shape[1] - #measurements in current bus; 1 for intercept, and xtest.shape[1] for # of total inputs (#days * #measurements), 
    
    for reg_model_num in range(ytest.shape[1]):
        # weight_matrix[reg_model_num,0] = reg.estimators_[reg_model_num].intercept_
        weight_matrix[reg_model_num,:] = reg.estimators_[reg_model_num].coef_
        
    
    res = np.abs(ytest - ypred)
    mse = mean_squared_error(ytest, ypred)
    res_mean = np.mean(res)
    res_std = np.std(res)
    
    mod_results[bus_number,:] = [mse, res_mean, res_std]
    
    ml_est[len(ytrain):, feat[bus_number]] = ypred
    ml_res[len(ytrain):, feat[bus_number]] = res
           
        

#%%

print('Residual space analysis')
# =============================================================================
# Residual ECD-AS
# =============================================================================

se_residual_test = np.abs(se_residual[len(ytrain):, :])
ml_res_test = np.abs(ml_res[len(ytrain):, :])

err_in_test = err_in[len(ytrain):]

train_size = int(np.floor(len(ytest)*train_size_param_percen))

#%%

# se_std = 0.01

# for i in range(se_residual_test.shape[1]):
#       se_residual_test[:,i] = (se_residual_test[:,i])/(se_std * np.abs(orz[len(ytrain_all):,i]))

#%%

for i in range(ml_res_test.shape[1]):
      se_residual_test[:,i] = (se_residual_test[:,i] - np.mean(se_residual_test[:train_size,i]))/(np.std(se_residual_test[:train_size,i]))
    

for i in range(ml_res_test.shape[1]):
      ml_res_test[:,i] = (ml_res_test[:,i] - np.mean(ml_res_test[:train_size,i]))/(np.std(ml_res_test[:train_size,i]))

#%%

res_dif = np.abs(se_residual_test - ml_res_test)

res_dif_sample_wise_mean = np.mean(res_dif, axis = 1)
res_dif_sample_wise_std = np.std(res_dif, axis = 1)


#%%

y_att = []

for i in range(len(err_in)):
    if (err_in[i][0] != 0):
        y_att.append(1)
    else:
        y_att.append(0)

y_att = np.array(y_att)

y_att_test = y_att[len(ytrain):]

#%%

data = np.transpose(res_dif)

cov_ind = np.var(data, axis = 1)

zero_variance_features = np.where(cov_ind == 0)
zero_variance_features = list(zero_variance_features[0])

df_data = pd.DataFrame(data)
df_data.drop(df_data.columns[zero_variance_features],axis=1,inplace=True)
data = np.array(df_data)

#Normalize

for i in range(data.shape[1]):
     data[:,i] = (data[:,i] - np.mean(data[:train_size,i]))/(np.std(data[:train_size,i]))


data = np.transpose(data)

#%%

res_data = data

y_data = y_att_test
se_distance = se_score[len(ytrain):]


#%%


start_data_index = 0
start_test_index = 1640
stop_data_index = 3640 #7599 #total 8640


cv_num = 5

num_of_trials = len(alphas) * len(etas) * len(betas) * cv_num

# ecd_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))
y_pred_overall = np.zeros((stop_data_index - train_size, num_of_trials))


y_test_overall = np.zeros((stop_data_index - train_size, num_of_trials))
# se_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))

result_table = np.zeros((num_of_trials + 2, 12))
se_result_table = np.zeros((num_of_trials + 2, 12))

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
                                                 
                         x_train = data[start_data_index:start_test_index, :]
                         x_test = data[start_test_index:stop_data_index, :]
                         
                         y_train = y_data[start_data_index:start_test_index]
                         y_test = y_data[start_test_index:stop_data_index]
                         
                         x_train = np.array(x_train)
                         y_train = np.array(y_train)   
                         x_test = np.array(x_test)   
                         y_test = np.array(y_test)
                         
                         
                         se_test = se_distance[start_test_index:stop_data_index]
                         se_test = np.array(se_test).reshape(len(se_test),)
                         
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
                         
                              
                              if curr_train.shape[0] > 1:
                             
                                 curr_mu = np.mean(curr_train, axis = 1) # initial mean
                                 curr_cov = np.cov(curr_train, rowvar=True) # initial covariance
                                 curr_cov = diag_load(curr_cov) # load the matrix diagonal
                                 curr_icov = LA.pinv(curr_cov) # initial inverse covariance
                                 
                              else:
                                  curr_mu = np.mean(curr_train, axis = 1)
                                  curr_cov = np.var(curr_train)
                                  curr_icov = 1/curr_cov
                              
                    # =============================================================================
                    #         Training decision scores  
                    # =============================================================================
                              curr_rxdist_train = []
                              
                              for x_sample in curr_train_all:               
                                  
                                  rxdist = RXdistance(x_sample, curr_mu, curr_icov)  
                                  
                                  if x_sample.shape[0] > 1:
                                      curr_rxdist_train.append(rxdist[0][0])
                                  else:
                                      curr_rxdist_train.append(rxdist[0])
                                  
                              
                              curr_rxdist_train = np.array(curr_rxdist_train)
                              # rx_matrix_train[:,i] = curr_rxdist_train
                         
                              # rxdist_all = list(curr_rxdist_train)
                              
                              rx_anomaly = []
                              curr_rxdist = []
                              thr_list = []
                              
                              rxdist_normal = list(curr_rxdist_train[idx_normal_train])
                              
                              thr_update_list_len = beta_val
                              
                              alpha = alpha_val
                              
                              
                              for k, x_sample in enumerate(curr_test):
                                   
                                  rxdist = RXdistance(x_sample, curr_mu, curr_icov)     
                                  
                                  
                                  if x_sample.shape[0] > 1:
                                      curr_rxdist.append(rxdist[0][0])
                                  else:
                                      curr_rxdist.append(rxdist[0])           
                                  
                                
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
                                       
                                  # rxdist_all.append(rxdist)
                                  
                              thr_list = np.array(thr_list)
                              thr_matrix[:,i] = thr_list
                                           
                              curr_rxdist = np.array(curr_rxdist)
                              rx_matrix[:,i] = curr_rxdist                    
                              rx_anomaly = np.array(rx_anomaly)            
                              anomaly_matrix[:,i] = rx_anomaly
                                       
                         # rx_full = np.concatenate((rx_matrix_train, rx_matrix), axis = 0)    
                         an = np.sum(anomaly_matrix, axis = 1)
                         
                    # =============================================================================
                    #      Prediction
                    # =============================================================================
                         y_pred = an > 0
                         y_pred = y_pred*1
                         
                         y_pred = np.array(y_pred)
                        
                         
                         
                         se_pred = np.zeros((len(y_pred),))
                         se_pred[se_test > se_threshold[0][0]] = 1
                         cm_se = confusion_matrix(y_test, se_pred)
                         tn_se, fp_se, fn_se, tp_se = cm_se.ravel()                         
                         acc_se, prec_se, recall_se, f1_se = class_metrics(cm_se)                         
                         se_result_table[trial_num,:] = [eta_val, beta_val, alpha_val, crossvalidation_number, tn_se, fp_se, fn_se, tp_se, acc_se, prec_se, recall_se, f1_se]
                         se_curr_result = [tn_se, fp_se, fn_se, tp_se, acc_se, prec_se, recall_se, f1_se]                       
                         print('tn, fp, fn, tp, acc, prec, recall, f1: ')                     
                         print('SE - ', se_curr_result)
                         
                         cm = confusion_matrix(y_test, y_pred)
                         tn, fp, fn, tp = cm.ravel()                         
                         acc, prec, recall, f1 = class_metrics(cm)                  
                         curr_result = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                         print('ML - ', curr_result)
                         result_table[trial_num,:] = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                         
                         trial_num += 1
                    
                         
                         start_data_index += 1000
                         start_test_index += 1000
                         stop_data_index += 1000                         


#%%
                         
# =============================================================================
# Saving results to CSV file
# =============================================================================

result_table[trial_num,:] = np.mean(result_table[:num_of_trials,:], axis = 0)
se_result_table[trial_num,:] = np.mean(se_result_table[:num_of_trials,:], axis = 0)

trial_num += 1
result_table[trial_num,:] = np.std(result_table[:num_of_trials,:], axis = 0)
se_result_table[trial_num,:] = np.std(se_result_table[:num_of_trials,:], axis = 0)


#%%


result_table_pd = pd.DataFrame(result_table)
result_table_pd.to_csv('sparam_result_table_ml_cv.csv')


se_result_table_pd = pd.DataFrame(se_result_table)
se_result_table_pd.to_csv('sparam_result_table_se_cv.csv')


#%%

# print('train 60 percent')
        
# print(eta_val, beta_val, alpha_val)                            

# rx_full = np.concatenate((rx_matrix_train, rx_matrix), axis = 0)
# an = np.sum(anomaly_matrix, axis = 1)
# r_pred = np.zeros((len(rlab_test),))
# r_pred[an>0] = 1
     
# cm_res = confusion_matrix(rlab_test, r_pred)
# print(cm_res)

# acc, prec, recall, f1 = class_metrics(cm_res)
# print(acc, prec, recall, f1)

# res_cm_plot_name = 'train_norm_cm_res_sparam' + suffix_name

# target_names = ['Normal', 'Anomalies']
# plot_confusion_matrix(cm_res, target_names, title=res_cm_plot_name)
     
# #%%

# print('Overall - SE results ')

# se_pred_test = se_pred[(len(ytrain) + train_size):]

# cm_se = confusion_matrix(rlab_test, se_pred_test)
# print(cm_se)

# acc, prec, recall, f1 = class_metrics(cm_se)
# print(acc, prec, recall, f1)

# se_cm_plot_name = 'train_norm_cm_sparam_se' + suffix_name

# target_names = ['Normal', 'Anomalies']
# plot_confusion_matrix(cm_se, target_names, title=se_cm_plot_name)
# #%%
# # mod_results = pd.DataFrame(mod_results)
# # mod_results.to_csv('../result csv files/MTR-MLR-01-08-2018-results.csv')

# # ml_est_pd = pd.DataFrame(ml_est[len(ytest):,:])
# # ml_est_pd.to_csv('../result csv files/MTR-MLR-01-08-2018-estimates.csv')

# # ml_res_pd = pd.DataFrame(ml_res[len(ytest):,:])
# # ml_res_pd.to_csv('../result csv files/MTR-MLR-01-08-2018-residuals.csv')


#%%
sim_end_time = time.time()
sim_time = sim_end_time - sim_start_time

print('Total simulation duration', sim_time) 
