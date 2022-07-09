#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 19:06:35 2020

@author: keerthiraj
"""

# =============================================================================
# ML SG SDN python function definitions
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools

import pandas as pd
import time
from tqdm import tqdm

from sklearn.metrics import f1_score, confusion_matrix

import random
import os

from scipy.io import loadmat

import warnings
warnings.filterwarnings("ignore")

#%%

# =============================================================================
# A function to convert dataframe to array with options to whether transpose or drop 1st column
# =============================================================================

def df_to_array(dataframe, trans = True, drop_column = True):
    
    if drop_column:
        arr = dataframe.drop(dataframe.columns[0], axis = 1)
        arr = np.array(arr)
    else:
        arr = np.array(dataframe)
        
    if trans:
        arr = np.transpose(arr)      
    
    return arr

#%%

# =============================================================================
# A function to convert .mat file into csv files
# =============================================================================

def mat_to_csv(from_filename,
               var_name,
               to_filename,
               print_keys = True):

     #'../data mat files/01-15-2018_grouped_data.mat')
    
    mat_filename = '../data mat files/' + from_filename + '.mat'
    matdata = loadmat(mat_filename)
    
    if print_keys:
        print(matdata.keys())
    
    var = matdata[var_name] # 'err_ins_FDI'
    var = pd.DataFrame(var)
    
    csv_filename = '../data csv files/' + to_filename + '.csv'
    var.to_csv(csv_filename)    

    print('Variable from mat file converted to csv file')
    
#%%

def arr_to_csv(arr, csv_filename):
    
    arr_pd = pd.DataFrame(arr)
    arr_pd.to_csv(csv_filename)
    
    # print('csv file created')
    


#%%
# =============================================================================
# A function to get grid connections from z file (connection information)
# =============================================================================

def get_grid_connections(z):
    
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

    return feat

#%%

# =============================================================================
# A function to remove zero variance measurements if there are any in the measurement data..
# =============================================================================

def remove_zero_var_meas(data):
    
    cov_ind = np.var(data, axis = 0)
    zero_variance_features = np.where((cov_ind == 0))
    zero_variance_features = list(zero_variance_features[0])
    
    df_data = pd.DataFrame(data)
    df_data.drop(df_data.columns[zero_variance_features],axis=0,inplace=True)
    data = np.array(df_data)
    
    return data

#%%

# =============================================================================
# A function to get ground truth vector from err_ins_FDI file
# =============================================================================

def get_ground_truth_from_pd(err_ins, drop_column = True, trans = True):
    
    if drop_column:
        err_in = err_ins.drop(err_ins.columns[0], axis=1)  # remove column 0 (the feature numbers#%%
        err_in = np.array(err_in)
        
    else:
        err_in = np.array(err_ins)
    
    if trans:
        err_in = np.transpose(err_in)

    err_in = err_in.reshape(len(err_in),)
    
    y_data = []
    
    for i in range(len(err_in)):
        if (err_in[i] > 0):
            y_data.append(1)
        else:
            y_data.append(0)
    
    y_data = np.array(y_data)

    return y_data



def get_ground_truth_from_arr(err_ins):
    
    # if drop_column:
    #     err_in = err_ins.drop(err_ins.columns[0], axis=1)  # remove column 0 (the feature numbers#%%
    #     err_in = np.array(err_in)
        
    # else:
    #     err_in = np.array(err_ins)
    
    # if trans:
    #     err_in = np.transpose(err_in)

    err_in = err_ins.copy()
    
    y_data = []
    
    for i in range(len(err_in)):
        if (err_in[i] > 0):
            y_data.append(1)
        else:
            y_data.append(0)
    
    y_data = np.array(y_data)

    return y_data


#%%

# =============================================================================
# A function to encrypt/mutate measurement data - exponential distribution
# =============================================================================

def mutation_params_exp(mutation_sd,
                    mutation_mean,
                    interval = 100,
                    transmutes_per_interval = 50,
                    psr_seq = np.arange(0,500),
                    random_seed_coeff = 1):
    
    indices = set(np.arange(0,interval))
    
    ind_list = []
    ind_list_2 = []
    value_list = []
    value_list_2 = []
    
    for i in range(len(psr_seq)):
        
        random.seed(psr_seq[i] * random_seed_coeff)
        
        curr_ind = list(np.sort([x + (interval * i) for x in random.sample(indices, transmutes_per_interval)]))
        
        random.seed((psr_seq[i] + 1) * random_seed_coeff)
        
        curr_ind_2 = list(np.sort([x + (interval * i) for x in random.sample(indices, transmutes_per_interval)]))        
        
        np.random.seed(psr_seq[i]* random_seed_coeff)
        
        
        curr_val = list(np.random.normal(mutation_mean, mutation_sd ,transmutes_per_interval))
        
        curr_val = list(np.random.exponential(scale = mutation_sd, size = transmutes_per_interval))
        
        curr_val_2 = list(np.random.exponential(scale = 0.5 * mutation_sd, size = transmutes_per_interval))
        
        ind_list.extend(curr_ind)
        ind_list_2.extend(curr_ind_2)
        value_list.extend(curr_val)
        value_list_2.extend(curr_val_2)
        
        
    return ind_list, ind_list_2, value_list, value_list_2


#%%


def mutation_params_normal(mutation_sd,
                    mutation_mean,
                    interval = 100,
                    transmutes_per_interval = 50,
                    psr_seq = np.arange(0,500),
                    random_seed_coeff = 1):
    
    indices = set(np.arange(0,interval))
    
    ind_list = []
    # ind_list_2 = []
    value_list = []
    # value_list_2 = []
    
    for i in range(len(psr_seq)):
        
        random.seed(psr_seq[i] * random_seed_coeff)
        
        curr_ind = list(np.sort([x + (interval * i) for x in random.sample(indices, transmutes_per_interval)]))
        
        random.seed((psr_seq[i] + 1) * random_seed_coeff)
        
        # curr_ind_2 = list(np.sort([x + (interval * i) for x in random.sample(indices, transmutes_per_interval)]))        
        
        np.random.seed(psr_seq[i]* random_seed_coeff)
        
        
        curr_val = list(np.random.normal(mutation_mean, mutation_sd ,transmutes_per_interval))
        
        # curr_val_2 = list(np.random.normal(mutation_mean, 0.75 * mutation_sd ,transmutes_per_interval))
        
        
        ind_list.extend(curr_ind)
        # ind_list_2.extend(curr_ind_2)
        value_list.extend(curr_val)
        # value_list_2.extend(curr_val_2)
        
        
    return ind_list, value_list

#%%


# =============================================================================================
# Code to add slow rate fdi attacks to measurement data
# ==============================================================================================

def add_slowfdi_attacks(measurement_data,
                        err_size,
                        err_ins):
    
    attacked_data = measurement_data.copy()
    
    for sample_num in range(attacked_data.shape[0]):
        
        if err_ins[sample_num] > 0:
            
            attack_measurement = err_ins[sample_num] - 1 # -1 is matlab to python conversion
            
            curr_meas_val = attacked_data[sample_num, attack_measurement].copy()
            
            attacked_data[sample_num, attack_measurement] = curr_meas_val + (err_size[sample_num] * abs(curr_meas_val))/100
    
    return attacked_data
   
#%%


def generate_err_ins(attack_len = 150,
                     attack_gap_sz = 2000,
                     attack_meas_number = 394,
                     attack_start = 2000,
                     attack_end = 18000,
                     attack_array_len = 21600,
                     seed_val = 42
                     ):
    
    np.random.seed(seed_val)
    
    attack_start_sample = attack_start # 0
    attack_end_sample = attack_end # len(temp_err_in)
    
    temp_err_in = np.zeros((attack_array_len,)) #np.zeros((orz.shape[0],))
    
    attack_measurement_number = attack_meas_number #use matlab index for consistency
    
    attack_length = attack_len
    
    attack_gap_size = attack_gap_sz
    
    curr_range_start = attack_start_sample
    curr_range_stop = attack_start_sample + attack_gap_size
    
    for i in range(int((attack_end_sample - attack_start_sample)/attack_gap_size)):
        
        curr_start = np.random.randint(curr_range_start, curr_range_stop)
        
        temp_err_in[curr_start:(curr_start+attack_length+1)] = attack_measurement_number
        
        curr_range_start += attack_gap_size
        curr_range_stop += attack_gap_size
        
    return temp_err_in





#%%

def generate_error_sizes(err_in, sd_start = 1, sd_end = 6, err_len = 51, err_shape = 'linear'):

    err_sz = np.zeros(err_in.shape) 

    if err_shape == 'linear':
        sz = list(np.linspace(sd_start, sd_end, err_len))
    else:
        sz = sorted(10 * np.logspace(sd_start, sd_end, err_len, base = sd_end/10.0))
        
    
    add_attack = -1
    
    for i in range(len(err_in)):
        
        if err_in[i] > 0:
            add_attack += 1
            if add_attack >= 0:
                # print(add_attack)
                err_sz[i] = sz[add_attack] 
        else:
            add_attack = -1

    return err_sz

#%%

# =============================================================================================
# Code to add SMART rate fdi attacks to measurement data
# ==============================================================================================

def add_smartfdi_attacks(measurement_data,
                        err_size,
                        err_ins,
                        add_val_to = 'mean',
                        window_size = 100):
    
    normal_data = measurement_data.copy()
    
    final_attacked_data = measurement_data.copy()
    
    win_size = window_size
    
    attack_start_flag = 0
    
    for sample_num in range(1, normal_data.shape[0]):
        
        
        if err_ins[sample_num] > 0:
            
            attack_measurement = int(err_ins[sample_num] - 1) # -1 is matlab to python conversion
            
            if attack_start_flag == 0:
                
                # print(sample_num, win_size, attack_measurement)
                
                curr_meas_mean = np.mean(normal_data[sample_num - win_size:sample_num, attack_measurement]) 
                curr_meas_std = np.std(normal_data[sample_num - win_size:sample_num, attack_measurement]) 
                attack_start_flag = 1
                
            curr_meas_val = normal_data[sample_num, attack_measurement].copy()
            
            if add_val_to == 'mean':
                
                final_attacked_data[sample_num, attack_measurement] = curr_meas_mean + (err_size[sample_num] * abs(curr_meas_std))
                
            else:
                
                final_attacked_data[sample_num, attack_measurement] = curr_meas_val + (err_size[sample_num] * abs(curr_meas_std))
            
            
        else:
            attack_start_flag = 0
    
    return final_attacked_data
   


#%%

# =============================================================================
# A function to normalize a given array - subtract mean and divide by std in each column
# =============================================================================

def get_normalized(arr):
    
    for i in range(arr.shape[1]):
          arr[:,i] = (arr[:,i] - np.mean(arr[:,i]))/(np.std(arr[:,i]))
        
    return arr


#%%


# =============================================================================
# A function to calculate classification metrics from confusion matrix
# =============================================================================
def class_metrics(cm):
               
     tn, fp, fn, tp = cm.ravel()
     
     tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
     
     acc = (tp+tn)/(tp+fp+fn+tn)
     
     if (tp+fp) == 0:
         pre = 0.0
     else:
         pre = tp/(tp+fp)
     
     if (tp+fn) == 0:
         rec = 0.0
     else:
         rec = tp/(tp+fn)
     
     if (pre + rec) == 0:
         f1 = 0.0
     else:
         f1 = (2.0*pre*rec) / (pre+rec)
          
     
#          print 'Acc, Pre, Rec, F1'
     return [acc, pre, rec, f1]

#%%

# Diagonal loading
# Inputs: covariance matrix, condition number maximum threshold, step size
# Output: Diagonally loaded covariance matrix
# This method adds a small value to the diagonal of a covariance matrix 
# to ensure that the matrix is invertible (not ill-conditioned)



def diag_load(cov, cond_thres = 100000, lam = 1e-14):  
    eig_val = LA.eig(cov)[0] # get all eigenvalues
#    print(eig_val)
    eig_max = max(eig_val) # largest eigenvalue 
    eig_min = min(eig_val) # smallest eigenvalue
    cond_num = abs(eig_max/eig_min) # condition number
    
    while cond_num > cond_thres:   
        eig_max = eig_max + lam 
        eig_min = eig_min + lam
        cond_num = abs(eig_max/eig_min)
        lam = 2 * lam
        
    cov = cov + lam * np.identity(len(cov))
    return cov


#%%

# Calculating RX distance
# Input: data sample, mean and inverse covariance marix
# Output: RX distance
def RXdistance(x, mu, icov):
    x = x.reshape(len(x), 1) # data sample
    mu = mu.reshape(len(mu), 1) # mean 
    x_mu = x - mu # sample - mean
    x_mu_tr = (x - mu).reshape(1, len(x)) # (sample - mean) transpose
    md_sq = x_mu_tr.dot(icov).dot(x_mu) # Mahalanobis distance squared
    
    return md_sq



#%%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    if not os.path.isdir('./cm_files'):
    	os.mkdir('./cm_files')

    title = './cm_files/' + title

    plt.savefig(title +".png")

#%%

# Output = Result matrix for now
# =============================================================================

def perform_ecd_as(train_size,
                   alphas,
                   betas,
                   etas,
                   data,
                   y_data,
                   feat,
                   start_data_index = 0,
                   start_test_index = 8100,
                   stop_data_index = 21600,
                   cross_val_num = 1,
                   cross_valid = False):

    start_data_index = start_data_index
    start_test_index = train_size
    stop_data_index = stop_data_index #data.shape[0] #21600 #21600 #12600
    
    cv_num = cross_val_num
    
    num_of_trials = len(alphas) * len(etas) * len(betas) * cv_num
    
    ecd_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))
    y_pred_overall = np.zeros((stop_data_index - train_size, num_of_trials))
    
    
    y_test_overall = np.zeros((stop_data_index - train_size, num_of_trials))
    # se_decision_scores = np.zeros((stop_data_index - train_size, num_of_trials))
    
    result_table = np.zeros((num_of_trials+2, 12))
    
    trial_num = 0
    
    
    # eta_val = etas[0]
    # beta_val = betas[0]
    # alpha_val = alphas[0]
    
    
    for crossvalidation_number in range(cv_num):
         
         for eta_val in etas:
              
              for beta_val in betas:
                  
                  for alpha_val in alphas:
                      
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
                           # print(i)
                          
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
                           rx_matrix_train[:,i] = curr_rxdist_train.reshape(len(curr_rxdist_train),)
                     
                           rxdist_all = list(curr_rxdist_train)
                          
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
                                   
                               rxdist_all.append(rxdist)
                               
                           thr_list = np.array(thr_list)
                           thr_matrix[:,i] = thr_list
                                       
                           curr_rxdist = np.array(curr_rxdist).reshape(len(curr_rxdist),)
                           rx_matrix[:,i] = curr_rxdist                    
                           rx_anomaly = np.array(rx_anomaly)            
                           anomaly_matrix[:,i] = rx_anomaly
                           
                           
                                                        
                      # rx_full = np.concatenate((rx_matrix_train, rx_matrix), axis = 0)    
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
                                          
                      # f1 = f1_score(y_test, y_pred)
                      # overall_f1.append(f1)
                                          
                      cm = confusion_matrix(y_test, y_pred)
                      tn, fp, fn, tp = cm.ravel()
                      acc, prec, recall, f1 = class_metrics(cm)
                     
                      # target_names = ['Normal', 'Anomalies']
                      # plot_confusion_matrix(cm, target_names, title=cm_title_1)
                     
                      # overall_cm_dict[crossvalidation_number] = cm
                     
                      curr_result = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                     
                      print('eta, beta, alpha, tn, fp, fn, tp, acc, prec, recall, f1: ', curr_result)
                     
                      result_table[trial_num,:] = [eta_val, beta_val, alpha_val, crossvalidation_number, tn, fp, fn, tp, acc, prec, recall, f1]
                     
                      trial_num += 1
                      
                      if cross_valid == True:
                          
                          start_data_index += 1000
                          start_test_index += 1000
                          stop_data_index += 1000
    
                             

    return result_table


#%%



def train_model(train_data, label_data, train_size = 1800, feat = 118):
    
    idx_normal = np.where(label_data == 0)[0]
    idx_normal_train = [x for x in idx_normal if x< train_size]
   
    rx_matrix_train = np.zeros((len(train_data), len(feat)))
    
    
    
    
    
    
#%%

