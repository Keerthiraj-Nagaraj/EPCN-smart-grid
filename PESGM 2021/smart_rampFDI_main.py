#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:31:51 2020

@author: keerthiraj
"""


# =============================================================================================================
# ED ECD AS full flow : TX - measurement encryption, Attack generation and RX- decryption + attack detection
# =============================================================================================================

import sys
sys.path.append('../helper files')

#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#%%

import helper_functions as hf

#%%

import warnings
warnings.filterwarnings("ignore")

#%%

print('..............LOADING DATA..............', time.time())

# =============================================================================
# Data for TX side - original measurement data 
# =============================================================================

orz_pd = pd.read_csv('../data csv files/org_z_gaussian_ml_s4_0_3_ed.csv')
orz = hf.df_to_array(orz_pd,  trans = False)

#%%

# =============================================================================
# Data for attack generation
# =============================================================================

# err_sz_pd = pd.read_csv('../data csv files/err_sz_FDI_s4_0_3_ed.csv')
# err_sz = hf.df_to_array(err_sz_pd,  trans = False)

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
mutation_sd_val = 3

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


print('..............GENERATING SMART RAMP ATTACKs - TYPE 4..............', time.time())
# ===================================================================================================================
# Attack generation 
# ===================================================================================================================

err_ins = hf.df_to_array(err_ins_pd, trans = False)
err_ins = err_ins.reshape(len(err_ins),)

#%%
ramp_attack_length = 75

err_ins = hf.generate_err_ins(attack_len = ramp_attack_length,
                              attack_gap_sz = 1000,
                              attack_meas_number = 394,
                              attack_start = 7000,
                              attack_end = 20000,
                              attack_array_len = 21600,
                              seed_val = 42)

    
#%%

err_sz_linear = hf.generate_error_sizes(err_ins, sd_start = 1, sd_end = 6, 
                                        err_len = (ramp_attack_length+1), err_shape = 'linear')

err_sz_smooth = hf.generate_error_sizes(err_ins, sd_start = 1, sd_end = 6, 
                                        err_len = (ramp_attack_length+1), err_shape = 'smooth')

err_sz = err_sz_smooth.copy()

#%%


# Adding attacks to original data

attacked_org_data = hf.add_smartfdi_attacks(measurement_data = orz, err_size = err_sz, 
                                            err_ins = err_ins, add_val_to = 'mean', window_size = 50)     


#%%

# Adding attacks to modified data data

attacked_mutated_data = hf.add_smartfdi_attacks(measurement_data = mutated_orz, err_size = err_sz, 
                                                err_ins = err_ins, add_val_to = 'mean', window_size = 50)
     


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

mutation_sd_val = 3

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

# result_table_pd = pd.DataFrame(org)
# result_table_pd.to_csv('local csv files/original_measurements_s4.csv')

result_table_pd = pd.DataFrame(err_ins)
result_table_pd.to_csv('local csv files/attacked meas latest/err_ins_FDI_type_4_attack_len_75_sz_6sd_s4.csv')

result_table_pd = pd.DataFrame(err_sz)
result_table_pd.to_csv('local csv files/attacked meas latest/err_sz_FDI_type_4_attack_len_75_sz_6sd_s4.csv')

result_table_pd = pd.DataFrame(attacked_org_data)
result_table_pd.to_csv('local csv files/attacked meas latest/original_attacked_type_4_attack_len_75_sz_6sd_s4.csv')

result_table_pd = pd.DataFrame(unmutated_orz)
result_table_pd.to_csv('local csv files/attacked meas latest/mutated_attacked_and_then_decrypted_type_4_attack_len_75_sz_6sd_s4.csv')

#%%
'''
data = att.copy()
data = hf.get_normalized(data)

#%%

y_data = hf.get_ground_truth_from_arr(err_ins)

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
                    stop_data_index = 21600,
                    cross_valid = False)

#%%

# print('ORG detection completed at - ', time.time())

# print('ETA - 6 decrypt')

train_size = 8100 #8100  #9 hours for initial training

etas = [8, 9]#[3,4,5,6,7,8,9]# 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
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
                   stop_data_index = 21600,
                   cross_valid = False)

#%%


# print('Original attacked data - f1score - ', list(result_table_att[0][11:]))

# print('Encrypt/Decrypt attacked data - f1score - ', list(result_table_final[0][11:]))

print('..............SAVING RESULTS DATA..............', time.time())

hf.arr_to_csv(result_table_att, 
              csv_filename = 'local csv files/results ramp attacks/att_type_4_attack_len_75_sz_6sd_s4_sdval_3_trans_75.csv')

hf.arr_to_csv(result_table_final, 
              csv_filename = 'local csv files/results ramp attacks/decrypt_type_4_attack_len_75_sz_6sd_s4_sdval_3_trans_75.csv')


# result_table_pd = pd.DataFrame(result_table)
# result_table_pd.to_csv('local csv files/result_table_gauss_s4_0_3_betas_8100_2.csv')


#%%

print('ETAs experiments for Smart RAMP Attacks simulation completed at - ', time.time())

# result_table_pd = pd.DataFrame(org)
# result_table_pd.to_csv('local csv files/original_measurements_s4_smooth.csv')

# result_table_pd = pd.DataFrame(attacked_org_data)
# result_table_pd.to_csv('local csv files/original_attacked_s4_smoothFDI_err_sz_1_to_5.csv')


# result_table_pd = pd.DataFrame(unmutated_orz)
# result_table_pd.to_csv('local csv files/mutated_attacked_and_then_decrypted_smoothFDI_err_sz_1_to_5.csv')


#%%

'''