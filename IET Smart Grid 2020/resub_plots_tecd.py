#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:08:25 2020

@author: keerthiraj
"""


# =============================================================================
# Resubmission plots
# =============================================================================



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions_tecd import class_metrics, mat2csv
from sklearn.metrics import confusion_matrix


#%%

# file_name = 'informed_redundancy_case118_drift_10000_wzeroinj_sdscaled_daily'
# var = 'chithII'
# csv_file = 'chithII_daily'
    
# mat2csv(file_name, var, csv_file)


#%%

# =============================================================================
# Drift Vs Dailyload
# =============================================================================

orz_drift_pd = pd.read_csv('../csv files/orz_drift.csv')
orz_daily_pd = pd.read_csv('../csv files/orz.csv')

#%%

orz_drift_pd = orz_drift_pd.drop(orz_drift_pd.columns[0], axis=1) # remove column 0 (the feature numbers)
orz_drift = np.array(orz_drift_pd)
orz_drift = np.transpose(orz_drift)
orz_drift = orz_drift[:,:691]


orz_daily_pd = orz_daily_pd.drop(orz_daily_pd.columns[0], axis=1) # remove column 0 (the feature numbers)
orz_daily = np.array(orz_daily_pd)
orz_daily = np.transpose(orz_daily)

#%%

measurement_vals = [337, 341, 0, 108, 215]

measurement_name = ['Real Power Flow', 'Reactive Power Flow', 'Real Power Injection', 'Reactive Power Injection', 'Voltage Magnitude']

# plt.figure(figsize=(6, 5))
     
drift_label = 'drift load'
daily_label = 'daily load'


f, axs = plt.subplots(2, 3, figsize=(15, 10), squeeze = True)
i = 2

ax1 = axs[0, i-2]

ax1.plot(orz_drift[:,measurement_vals[i]], marker = '.', color = 'tab:orange', label = drift_label, linewidth=0.5, markersize=4)
# ax1.set_xlabel('sample #')
ax1.set_ylabel('Measurement value') 
ax1.legend()
ax1.set_title(measurement_name[i])

ax2 = axs[1, i-2]
ax2.plot(orz_daily[:,measurement_vals[i]], marker = 'd', color = 'tab:blue', label = daily_label, linewidth=0.5, markersize=4)
ax2.set_xlabel('sample #')
ax2.set_ylabel('Measurement value') 
ax2.legend()
ax2.set_title(measurement_name[i])
    
for i in range(3, len(measurement_name)):
    
    ax1 = axs[0, i-2]
    
    ax1.plot(orz_drift[:,measurement_vals[i]], marker = '.', color = 'tab:orange', label = drift_label, linewidth=0.5, markersize=4)
    # ax1.set_xlabel('sample #')
    # ax1.set_ylabel('Measurement value') 
    ax1.legend()
    ax1.set_title(measurement_name[i])
    
    ax2 = axs[1, i-2]
    ax2.plot(orz_daily[:,measurement_vals[i]], marker = 'd', color = 'tab:blue', label = daily_label, linewidth=0.5, markersize=4)
    ax2.set_xlabel('sample #')
    # ax2.set_ylabel('Measurement value') 
    ax2.legend()
    ax2.set_title(measurement_name[i])
    
    # ax1.set_title('drift load data')
    # ax2.set_title('daily load data')

plot_name = 'figures/only_3_subplots_drift_vs_daily.pdf'
plt.tight_layout()
plt.savefig(plot_name, dpi = 900)         

#%%

# =============================================================================
# SE Vs CD Vs ECD VS tECD
# =============================================================================

train_size = 1800

se_rx_mat = pd.read_csv('../csv files/se_dist.csv')
se_rx_mat = se_rx_mat.drop(se_rx_mat.columns[0], axis =1)
se_rx_mat = np.array(se_rx_mat)
se_rx_mat = np.transpose(se_rx_mat)
se_rx_mat = se_rx_mat.reshape(len(se_rx_mat),)
se_rx_mat = se_rx_mat[train_size:]

se_thr_mat = pd.read_csv('chithII_daily.csv')
se_thr_mat = se_thr_mat.drop(se_thr_mat.columns[0], axis =1)
se_thr_mat = np.array(se_thr_mat)

#%%

cd_rx_mat = pd.read_csv('CDT_decision_scores_redundancy.csv')
cd_thr_mat = pd.read_csv('CDT_thr_overall_redundancy.csv')

cd_rx_mat = cd_rx_mat.drop(cd_rx_mat.columns[0], axis =1)
cd_rx_mat = np.array(cd_rx_mat).reshape(len(cd_rx_mat),)

cd_thr_mat = cd_thr_mat.drop(cd_thr_mat.columns[0], axis =1)
cd_thr_mat = np.array(cd_thr_mat)


#%%

ecd_rx_mat = pd.read_csv('ECD_rx_mat_redundancy_optimal_thresholds.csv')
ecd_thr_mat = pd.read_csv('ECD_thr_overall_redundancy_optimal_thresholds.csv')

ecd_rx_mat = ecd_rx_mat.drop(ecd_rx_mat.columns[0], axis =1)
ecd_rx_mat = np.array(ecd_rx_mat)
ecd_rx_mat = ecd_rx_mat[train_size:, :]

ecd_thr_mat = ecd_thr_mat.drop(ecd_thr_mat.columns[0], axis =1)
ecd_thr_mat = np.array(ecd_thr_mat).reshape(len(ecd_thr_mat),)


#%%

eta_val = 9
beta_val = 90
alpha_val = 0.00008

rx_file_name = 'rx_matrix_' + str(eta_val) + '_' + str(beta_val) + '_' + str(alpha_val) + '.csv'
thr_file_name = 'thr_matrix_' + str(eta_val) + '_' + str(beta_val) + '_' + str(alpha_val) + '.csv'

tecd_rx_mat = pd.read_csv(rx_file_name)
tecd_thr_mat = pd.read_csv(thr_file_name)

tecd_rx_mat = tecd_rx_mat.drop(tecd_rx_mat.columns[0], axis =1)
tecd_rx_mat = np.array(tecd_rx_mat)
# tecd_rx_mat = np.transpose(tecd_rx_mat)

tecd_thr_mat = tecd_thr_mat.drop(tecd_thr_mat.columns[0], axis =1)
tecd_thr_mat = np.array(tecd_thr_mat)
# tecd_thr_mat = np.transpose(tecd_thr_mat)


#%%
 
score_label = 'decision score'
thr_label = 'threshold'

region_number = 1

f, axs = plt.subplots(1, 4, figsize=(16, 4))

ax1 = axs[0]

se_thr = [se_thr_mat[0][0]] * len(se_rx_mat)

ax1.plot(se_rx_mat, '.-', label = score_label, linewidth=0.1, markersize=4)
ax1.plot(se_thr, '-k', label = thr_label, linewidth=2, markersize=4)
ax1.set_xlabel('testing sample #')
ax1.set_ylabel('Decision score') 
ax1.legend()
ax1.set_title('State Estimator', fontsize=16)


ax1 = axs[1]

cd_thr = [cd_thr_mat[0][0]] * len(se_rx_mat)

ax1.plot(cd_rx_mat, '.-', label = score_label, linewidth=0.3, markersize=4)
ax1.plot(cd_thr, '-k', label = thr_label, linewidth=2, markersize=4)
ax1.set_xlabel('testing sample #')
# ax1.set_ylabel('Mahalanobis distance score') 
ax1.legend()
ax1.set_title('CorrDet detector', fontsize=16)

ax1 = axs[2]

ecd_thr = [ecd_thr_mat[region_number]] * len(se_rx_mat)

ax1.plot(ecd_rx_mat[:, region_number], '.-', label = score_label, linewidth=0.1, markersize=4)
ax1.plot(ecd_thr, '-k', label = thr_label, linewidth=2, markersize=4)
ax1.set_xlabel('testing sample #')
# ax1.set_ylabel('Decision score') 
ax1.legend()

curr_title = 'ECD detector - bus: ' + str(region_number+1) 
ax1.set_title(curr_title, fontsize=16)


ax1 = axs[3]

# cd_thr = [cd_thr_mat[0][0]] * len(se_rx_mat)

ax1.plot(tecd_rx_mat[:, region_number], '.-', label = score_label, linewidth=0.3, markersize=4)
ax1.plot(tecd_thr_mat[:, region_number], '-k', label = 'adaptive threshold', linewidth=2, markersize=4)
ax1.set_xlabel('testing sample #')
# ax1.set_ylabel('Mahalanobis distance score') 
ax1.legend()
curr_title = 'ECD-AS detector - bus: ' + str(region_number+1) 
ax1.set_title(curr_title, fontsize=16)

plot_name = 'figures/score_v_thr_results_region_' + str(region_number+1) + '.pdf'
plt.tight_layout()
plt.savefig(plot_name, dpi = 900)         


#%%





# ab = 1

# score_label = 'decision score: region = ' + str(ab+1)
# thr_label = 'threshold: region = ' + str(ab+1)

# plt.plot(tecd_rx_mat[:,ab], '.-', label = score_label, linewidth=0.5, markersize=2)
# plt.plot(tecd_thr_mat[:,ab], '--', label = thr_label)
     

# #title_name = 'Decision score Vs threshold for multiple regions'
# #plt.title(title_name)
# plt.xlabel('testing sample #')
# plt.ylabel('Mahalanobis distance score') 
# plt.xlim(0, 19800)
# # plt.ylim(0, 1000)
# plt.legend()
# plot_name = 'scoreVsthr_figures_resub/score_v_thr_regions_old' + str(eta_val) + '_' + str(beta_val) + '_' + str(alpha_val) + '.png'
# plt.savefig(plot_name, dpi = 1200)         

#%%




