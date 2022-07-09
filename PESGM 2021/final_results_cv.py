#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 02:55:46 2020

@author: keerthiraj
"""

# =============================================================================
# Final cross validation results
# =============================================================================


import sys
sys.path.append('../helper files')

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat

#%%

import helper_functions as hf

#%%

import warnings
warnings.filterwarnings("ignore")

#%%

print('..............RFDI 3..............', time.time())


print('..............LOADING DATA..............', time.time())

# =============================================================================
# Data for TX side - original measurement data 
# =============================================================================

attack_model = 'rfdi 3'

from_filename = 'SE_mutated_attacked_and_then_decrypted_attack_len_50_sz_6sd_s4_sdval_2_trans_75'

mat_filename = '../data mat files/' + attack_model + '/' + from_filename + '.mat'
matdata = loadmat(mat_filename)

# print(matdata.keys())

#%%

se_score = np.array(matdata['J']).transpose().reshape(21600,) 
se_thr = np.array(matdata['chai']).transpose().reshape(21600,) 
# var = pd.DataFrame(var)

se_pred = np.zeros(se_score.shape)

for i,scr in enumerate(se_score):
    
    if se_score[i] > se_thr[i]:
        se_pred[i] = 1

#%%


#%%

# print('..............LOADING DATA..............', time.time())

# =============================================================================
# Data for TX side - original measurement data 
# =============================================================================

orz_pd = pd.read_csv('../data csv files/org_z_gaussian_ml_s4_0_3_ed.csv')
orz = hf.df_to_array(orz_pd,  trans = False)

#%%

# =============================================================================
# Data for attack generation
# =============================================================================


err_ins_pd = pd.read_csv('../data csv files/err_ins_s4_0_3_ed.csv')


#%%
# =============================================================================
# data for RX side
# =============================================================================

z = pd.read_csv('../data csv files/z.csv')
feat = hf.get_grid_connections(z)

# err_ins = pd.read_csv('../data csv files/err_ins_s4_0_3_ed.csv')


#%%

train_size = 8100 #8100  #9 hours for initial training

#%%

print('..............ENCRYPTING MEASUREMENT DATA AT TX..............', time.time())

# =============================================================================
# Measurement data transmuation - encryption
# =============================================================================

full_indices = set(np.arange(0, len(orz)))

mutation_mean_val = 0
mutation_sd_val = 2

interval_val = 100
transmutes_per_interval_val = 75

mutated_orz = np.zeros(orz.shape)

lazy = [int((x**2 + x + 2)/2) for x in range(orz.shape[0]//interval_val)]

mutate_indices_tx = np.zeros((orz.shape[0]//interval_val * transmutes_per_interval_val, orz.shape[1]))
mutate_values_tx = np.zeros((orz.shape[0]//interval_val * transmutes_per_interval_val, orz.shape[1]))


for meas in range(orz.shape[1]):
    
    curr_meas = np.copy(orz[:,meas])
    
    curr_meas_mutated = np.copy(curr_meas)
    
    ind_list, value_list = hf.mutation_params_normal(random_seed_coeff = meas,
                                            interval = interval_val,
                                            transmutes_per_interval = transmutes_per_interval_val,
                                            psr_seq = lazy,
                                            mutation_mean = 0,#np.mean(curr_meas), #[:train_size]), #mutation_mean_val,
                                            mutation_sd = mutation_sd_val * np.std(curr_meas[:train_size]))
        

    #mutating real data
    curr_meas_mutated[ind_list] = np.array(list(curr_meas[ind_list] + value_list))
    
    mutate_indices_tx[:,meas] = np.array(ind_list)    #.reshape(len(ind_list),1)
    mutate_values_tx[:,meas] = np.array(value_list)   #.reshape(len(value_list),1)
    
    mutated_orz[:,meas] = np.copy(curr_meas_mutated)

# hf.arr_to_csv(mutated_orz, csv_filename = 'local csv files/normal_mutated_org_z_gaussian_ml_s4_0_3.csv')

#%%


print('..............GENERATING SMART RAMP FDI ATTACKs..............', time.time())
# ===================================================================================================================
# Attack generation 
# ===================================================================================================================

err_ins = hf.df_to_array(err_ins_pd, trans = False)
err_ins = err_ins.reshape(len(err_ins),)

#%%
ramp_attack_length = 50

# err_ins = hf.generate_err_ins(attack_len = ramp_attack_length,
#                               attack_gap_sz = 1000,
#                               attack_meas_number = 394,
#                               attack_start = 7000,
#                               attack_end = 20000,
#                               attack_array_len = 21600,
#                               seed_val = 42)

    
#%%

err_sz_linear = hf.generate_error_sizes(err_ins, sd_start = 1, sd_end = 6, 
                                        err_len = (ramp_attack_length+1), err_shape = 'linear')

err_sz_smooth = hf.generate_error_sizes(err_ins, sd_start = 1, sd_end = 6, 
                                        err_len = (ramp_attack_length+1), err_shape = 'smooth')



#%%

# attack_model = 'rfdi 1'

if (attack_model == 'rfdi 1'):
    
    err_sz = err_sz_linear.copy()
    to_attack = 'meas'
    
elif attack_model == 'rfdi 2':
    
    err_sz = err_sz_smooth.copy()
    to_attack = 'meas'

elif attack_model == 'rfdi 3':
    
    err_sz = err_sz_linear.copy()
    to_attack = 'mean'

else:
    err_sz = err_sz_smooth.copy()
    to_attack = 'mean'


#%%

# Adding attacks to original data

attacked_org_data = hf.add_smartfdi_attacks(measurement_data = orz, err_size = err_sz, 
                                            err_ins = err_ins, add_val_to = to_attack, window_size = 50)     


#%%

# Adding attacks to modified data data

attacked_mutated_data = hf.add_smartfdi_attacks(measurement_data = mutated_orz, err_size = err_sz, 
                                                err_ins = err_ins, add_val_to = to_attack, window_size = 50)
     


#%%

print('..............PREPARING DATA AT RX..............', time.time())
# =============================================================================
# RX side - 
# =============================================================================

org = orz.copy()
org = hf.remove_zero_var_meas(org)

#%%

att = attacked_org_data.copy()
att = hf.remove_zero_var_meas(att)

#%%

attacked_orz =  attacked_mutated_data.copy()
attacked_orz = hf.remove_zero_var_meas(attacked_orz)

#%%

# =============================================================================
# Defining hyper-parameters
# =============================================================================

train_size = 8100 #8100  #9 hours for initial training

# etas = [9]# 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
# betas =  [450]# [90]# #[90] #[15, 45, 90, 150, 300, 450]  #[90]
# alphas = [0.00008]# [0.00008]


#%%

print('..............DECRYPTING DATA AT RX..............', time.time())

# =============================================================================
# RX side
# =============================================================================

mutation_sd_val = 2

interval_val = 100
transmutes_per_interval_val = 75

lazy = [int((x**2 + x + 2)/2) for x in range(attacked_orz.shape[0]//interval_val)]

unmutated_orz = np.zeros(attacked_orz.shape)

mutate_indices_rx = np.zeros((attacked_orz.shape[0]//interval_val * transmutes_per_interval_val, attacked_orz.shape[1]))
mutate_values_rx = np.zeros((attacked_orz.shape[0]//interval_val * transmutes_per_interval_val, attacked_orz.shape[1]))


for meas in range(attacked_orz.shape[1]):
    
    
    curr_org_meas = np.copy(org[:,meas])
    
    curr_rx_meas = np.copy(attacked_orz[:,meas])
    
    curr_meas_unmutated = np.copy(curr_rx_meas)
 

    ind_list, value_list = hf.mutation_params_normal(random_seed_coeff = meas,
                                            interval = interval_val,
                                            transmutes_per_interval = transmutes_per_interval_val,
                                            psr_seq = lazy,
                                            mutation_mean = 0, #np.mean(curr_rx_meas),
                                            mutation_sd = mutation_sd_val * np.std(curr_org_meas[:train_size]))
    
    curr_meas_unmutated[ind_list] = np.array(list(curr_rx_meas[ind_list] - value_list))  

    unmutated_orz[:,meas] = curr_meas_unmutated
    

#%%

final_dataset = org.copy()
final_dataset[train_size:,:] = np.copy(unmutated_orz[train_size:, :])

#%%

attack_meas_number = 393


#%%

data = att.copy()
data = hf.get_normalized(data)

#%%

y_data = hf.get_ground_truth_from_arr(err_ins)

#%%


from sklearn.metrics import confusion_matrix

se_test = np.copy(y_data)

start_se_ind = 8100
stop_se_ind = 16600

f1_se = []
prec_se = []
rec_se = []

for i in range(5):
    
    cm = confusion_matrix(se_test[start_se_ind:stop_se_ind], se_pred[start_se_ind:stop_se_ind])
    tn, fp, fn, tp = cm.ravel()
    acc, prec, recall, f1 = hf.class_metrics(cm)
    
    prec_se.append(prec)
    rec_se.append(recall)
    f1_se.append(f1)
    
    start_se_ind += 1000
    stop_se_ind += 1000


print(attack_model)


print('SE prec mean - ', np.mean(prec_se))
print('SE prec std - ', np.std(prec_se))

print('SE rec mean - ', np.mean(rec_se))
print('SE rec std - ', np.std(rec_se))


print('SE f1 mean - ', np.mean(f1_se))
print('SE f1 std - ', np.std(f1_se))




#%%

train_size = 8100 #8100  #9 hours for initial training

etas = [4]#[3,4,5] # 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
betas =  [450]#[150, 450]# [90]# #[90] #[15, 45, 90, 150, 300, 450]  #[90]
alphas = [0.00008]# [0.00008]


# print('ORG detection began at - ', time.time())

print('..............ATTACK DETECTION FOR ORIGINAL DATA..............', time.time())

result_table_att = hf.perform_ecd_as(train_size,
                    alphas,
                    betas,
                    etas,
                    data,
                    y_data,
                    feat,
                    start_data_index = 0,
                    start_test_index = train_size,
                    stop_data_index = 16600,
                    cross_val_num = 5,
                    cross_valid = True)

#%%

cross_val_num = 5
trial_num = 5

result_table_att[trial_num,:] = np.mean(result_table_att[:cross_val_num,:], axis = 0)
trial_num += 1
result_table_att[trial_num,:] = np.std(result_table_att[:cross_val_num,:], axis = 0)

result_att_file = 'local csv files/crossval/' + attack_model + '/att_rfdi_cv.csv'

hf.arr_to_csv(result_table_att, csv_filename = result_att_file)



#%%

# print('ORG detection completed at - ', time.time())

# print('ETA - 6 decrypt')

train_size = 8100 #8100  #9 hours for initial training

etas = [9]#[3,4,5,6,7,8,9]# 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
betas =  [450] #[150, 450]# [90]# #[90] #[15, 45, 90, 150, 300, 450]  #[90]
alphas = [0.00008]# [0.00008]


data = unmutated_orz.copy()
data = hf.get_normalized(data)

#%%

print('..............ATTACK DETECTION FOR ENCRYPT/DECRYPT DATA..............', time.time())

result_table_final = hf.perform_ecd_as(train_size,
                   alphas,
                   betas,
                   etas,
                   data,
                   y_data,
                   feat,
                   start_data_index = 0,
                   start_test_index = train_size,
                   stop_data_index = 16600,
                   cross_val_num = 5,
                   cross_valid = True)

#%%


# print('Original attacked data - f1score - ', list(result_table_att[0][11:]))

# print('Encrypt/Decrypt attacked data - f1score - ', list(result_table_final[0][11:]))

print('..............SAVING RESULTS DATA..............', time.time())

cross_val_num = 5
trial_num = 5

result_table_final[trial_num,:] = np.mean(result_table_final[:cross_val_num,:], axis = 0)
trial_num += 1
result_table_final[trial_num,:] = np.std(result_table_final[:cross_val_num,:], axis = 0)

result_final_file = 'local csv files/crossval/' + attack_model + '/decrypt_rfdi_cv.csv'

hf.arr_to_csv(result_table_final, csv_filename = result_final_file)


# result_table_pd = pd.DataFrame(result_table)
# result_table_pd.to_csv('local csv files/result_table_gauss_s4_0_3_betas_8100_2.csv')

#%%

#%%

print('ETAs experiments for Smart RAMP Attacks simulation completed at - ', time.time())

# result_table_pd = pd.DataFrame(org)
# result_table_pd.to_csv('local csv files/original_measurements_s4_smooth.csv')

# result_table_pd = pd.DataFrame(attacked_org_data)
# result_table_pd.to_csv('local csv files/original_attacked_s4_smoothFDI_err_sz_1_to_5.csv')


# result_table_pd = pd.DataFrame(unmutated_orz)
# result_table_pd.to_csv('local csv files/mutated_attacked_and_then_decrypted_smoothFDI_err_sz_1_to_5.csv')


#%%