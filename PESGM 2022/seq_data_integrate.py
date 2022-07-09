#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:28:51 2021

@author: keerthiraj
"""


# =============================================================================================================
# 
# =============================================================================================================

import sys
sys.path.append('helper files')

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

orz_pd = pd.read_csv('../data csv files/TD_multi_part1.csv')
orz = hf.df_to_array(orz_pd,  trans = False)

#%%

# for i in range(orz.shape[1]):
#     plt.figure()
#     plt.plot(orz[:,i], '.')
#     title = 'measurement_' + str(i)
    
#     plt.title(title)
#     figname = 'figures/td_measurement_' + str(i) + '.png'
#     plt.savefig(figname)


#%%

# =============================================================================
# Data for attack generation
# =============================================================================

# err_sz_pd = pd.read_csv('../data csv files/err_sz_FDI_s4_0_3_ed.csv')
# err_sz = hf.df_to_array(err_sz_pd,  trans = False)

err_ins_pd = pd.read_csv('data csv files/err_ins.csv')

z = pd.read_csv('data csv files/z.csv')
feat = hf.get_grid_connections(z)

# err_ins = pd.read_csv('../data csv files/err_ins_s4_0_3_ed.csv')


#%%

train_size = 1800 #1800  #2 hours for initial training

#%%

# ===================================================================================================================
# Attack generation 
# ===================================================================================================================

err_ins = hf.df_to_array(err_ins_pd, trans = False)
err_ins = err_ins.reshape(len(err_ins),)

#%%

print('..............PREPARING DATA AT RX..............', time.time())
# =============================================================================
# RX side - 
# =============================================================================

org = orz.copy()
org = hf.remove_zero_var_meas(org)

#%%

# =============================================================================
# Defining hyper-parameters
# =============================================================================


# etas = [9]# 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
# betas =  [450]# [90]# #[90] #[15, 45, 90, 150, 300, 450]  #[90]
# alphas = [0.00008]# [0.00008]

#%%

data = org.copy()
data = hf.get_normalized(data)

#%%

y_data = hf.get_ground_truth_from_arr(err_ins)

#%%

train_size = 1800

number_of_regions = 118

train_data = np.zeros((len(train_size), len(number_of_regions)))

sample_data_point = np.ones((1, len(number_of_regions)))

label_data = y_data



data_incoming = 1

eta = 9#[3,4,5,6,7,8,9]# 11, 12, 13] #[9] #gauss #[7, 8, 9, 10, 11, 12, 13]# ; [3, 4, 4.5, 5, 5.5, 6, 7]
beta =  450 #[150, 450]# [90]# #[90] #[15, 45, 90, 150, 300, 450]  #[90]
alpha = 0.00008# [0.00008]

anomalies = []

sample_id = 0

#if data_incoming:
for sample_data_point in data:
    
    if sample_id <= train_size:
        #save train data
        train_data[sample_id,:] = sample_data_point
        sample_id += 1
    elif sample_id == train_size:
        mu_list, icov_list, rxdist_normal = hf.train_model(train_data, label_data)
        sample_id += 1
    else:
        thr_list = hf.compute_threshold(rxdist_normal, beta)
        
        anomaly_decision, anomaly_regions = hf.sample_inference(sample_data_point, 
                                                                mu_list,
                                                                icov_list,
                                                                feat, 
                                                                thr_list, 
                                                                number_of_regions)
        
        if anomaly_decision:
            anomalies.append([sample_id, anomaly_regions])
        else:
            mu_list, icov_list = hf.update_model(sample_data_point,
                                                 mu_list,
                                                 icov_list,
                                                 alpha,
                                                 feat,
                                                 number_of_regions)
            
        sample_id += 1

#%%

# print('Original attacked data - f1score - ', list(result_table_att[0][11:]))

# print('Encrypt/Decrypt attacked data - f1score - ', list(result_table_final[0][11:]))

# print('..............SAVING RESULTS DATA..............', time.time())

# hf.arr_to_csv(result_table_att, 
#               csv_filename = 'local csv files/results ramp attacks/att_type_4_attack_len_75_sz_6sd_s4_sdval_3_trans_75.csv')

# hf.arr_to_csv(result_table_final, 
#               csv_filename = 'local csv files/results ramp attacks/decrypt_type_4_attack_len_75_sz_6sd_s4_sdval_3_trans_75.csv')


# result_table_pd = pd.DataFrame(result_table)
# result_table_pd.to_csv('local csv files/result_table_gauss_s4_0_3_betas_8100_2.csv')


#%%

# print('ETAs experiments for Smart RAMP Attacks simulation completed at - ', time.time())

# result_table_pd = pd.DataFrame(org)
# result_table_pd.to_csv('local csv files/original_measurements_s4_smooth.csv')

# result_table_pd = pd.DataFrame(attacked_org_data)
# result_table_pd.to_csv('local csv files/original_attacked_s4_smoothFDI_err_sz_1_to_5.csv')


# result_table_pd = pd.DataFrame(unmutated_orz)
# result_table_pd.to_csv('local csv files/mutated_attacked_and_then_decrypted_smoothFDI_err_sz_1_to_5.csv')


#%%
