#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:17:21 2019

@author: keerthiraj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy import linalg as LA

import itertools

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

# Read the Input Dataset
orz = pd.read_csv('orz_redundancy.csv')
err_ins = pd.read_csv('err_ins_redundancy.csv')
err_id = pd.read_csv("err_id_redundancy.csv", header = None)
se_dist = pd.read_csv("se_dist_redundancy.csv")
z = pd.read_csv('z_redundancy.csv')


thr_val = pd.read_csv('best_threshold_ecdt_redundancy.csv')


train_size = 3000


# Diagonal loading
# Inputs: covariance matrix, condition number maximum threshold, step size
# Output: Diagonally loaded covariance matrix
# This method adds a small value to the diagonal of a covariance matrix 
# to ensure that the matrix is invertible (not ill-conditioned)
def diag_load(cov, cond_thres = 100000, lam = 1e-14):  
    eig_val = LA.eig(cov)[0] # get all eigenvalues
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
    plt.savefig(title+".png")



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

reg = {}

for i in range(len(z)):
     reg[i] = []

for i,f in enumerate(feat):
     for each_feat in f:
          if i not in reg[each_feat]:
               reg[each_feat].append(i)
          

#Creating threshold vector
               
thr_val = thr_val.drop(thr_val.columns[0], axis=1)
thr_val = np.array(thr_val).reshape(len(thr_val),)



# Preparing the dataset
orz = orz.drop(orz.columns[0], axis=1) # remove column 0 (the feature numbers)
orz_arr = np.array(orz)

#Removing zero varinace features

data = np.transpose(orz_arr)

cov_ind = np.var(data, axis = 0)

zero_variance_features = np.where(cov_ind == 0)
zero_variance_features = list(zero_variance_features[0])

df_data = pd.DataFrame(data)
df_data.drop(df_data.columns[zero_variance_features],axis=1,inplace=True)
data = np.array(df_data)

#Normalize

for i in range(data.shape[1]):
     data[:,i] = (data[:,i] - np.mean(data[:,i]))/(np.std(data[:,i]))

# Getting ground truth
err_ins = err_ins.drop(err_ins.columns[0], axis=1)  # remove column 0 (the feature numbers)
err_in = np.array(err_ins.values) # convert to numpy array

#State estimator predictions
err_id = np.array(err_id.iloc[1,1::].values)

y_se = [0] * len(err_id)
y_se = np.array(y_se)

for i,val in enumerate(err_id):
    if val > 0:
        y_se[i] = 1
    



plt.figure()
plt.plot(data[:,339], '-')
plt.xlabel('sample number')
plt.ylabel('Normalized real power flow value')
plt.savefig('rpf_2_5_3.png', dpi = 600)

se_dist = se_dist.drop(se_dist.columns[0], axis=1)  # remove column 0 (the feature numbers)
se_d = np.array(se_dist.values) # convert to numpy array
se_d = np.transpose(se_d)
se_distance = se_d
se_distance = se_distance.reshape(len(se_distance),)


y_feat = []

for i in range(len(err_in[0])):
     if i > train_size and err_in[0][i] > 0:
        y_feat.append(err_in[0][i])


y_feat = np.array(y_feat)

y_data = []

for i in range(len(err_in[0])):
    if (err_in[0][i] > 0):
        y_data.append(1)
    else:
        y_data.append(0)

y_data = np.array(y_data)

data_splits = []

start = 0
end = 1000

for i in range(10):
     data_splits.append([start, end])
     start += 1000
     end += 1000


auc_arr = []

se_decision_scores = np.zeros((len(y_data) - train_size, 10))

ecd_decision_scores = np.zeros((len(y_data) - train_size, 10))

an_overall = np.zeros((len(y_data) - train_size, 10))

y_test_overall = np.zeros((len(y_data) - train_size, 10))

y_pred_overall = np.zeros((len(y_data) - train_size, 10))

thr_overall = np.zeros((len(feat), 10))


for j in range(10):
          
     x_train = []
     y_train = []
     
     x_test = []
     y_test = []
     
     se_test = []
     
     se_pred_train = []
     se_pred_test = []
     
     for k in range(10):
          
          if k<3:
               x_train.extend(data[data_splits[k][0]:data_splits[k][1], :])
               y_train.extend(y_data[data_splits[k][0]:data_splits[k][1]])
               se_pred_train.extend(y_se[data_splits[k][0]:data_splits[k][1]])
          else:
               x_test.extend(data[data_splits[k][0]:data_splits[k][1], :])
               y_test.extend(y_data[data_splits[k][0]:data_splits[k][1]])
               se_test.extend(se_distance[data_splits[k][0]:data_splits[k][1]])
               se_pred_test.extend(y_se[data_splits[k][0]:data_splits[k][1]])
               
     x_train = np.array(x_train)
     y_train = np.array(y_train)   
     x_test = np.array(x_test)   
     y_test = np.array(y_test)
     
     print(x_train[0,0])
     
        
     se_pred_train = np.array(se_pred_train)
     se_pred_test = np.array(se_pred_test)
     
     se_test = np.array(se_test)
     
     idx_normal = np.where(y_train == 0)[0]
     idx_normal_train = [x for x in idx_normal if x< train_size]
     
     rx_matrix_train = np.zeros((len(x_train), len(feat)))
     rx_matrix = np.zeros((len(x_test), len(feat)))
     anomaly_matrix = np.zeros((len(x_test), len(feat)))
     thr_list = []
     
     cov_dict = {}
     
     
     for i in range(len(feat)):
          
          #Define current training and testing set
          
          curr_train = x_train[:, feat[i]]
          curr_test = x_test[:, feat[i]]
          
          curr_train_all = curr_train
          
          curr_train = curr_train[idx_normal_train,:]
          curr_train = np.transpose(curr_train)
     
          
          curr_mu = np.mean(curr_train, axis = 1) # initial mean
          curr_cov = np.cov(curr_train, rowvar=True) # initial covariance
          curr_cov = diag_load(curr_cov) # load the matrix diagonal
          curr_icov = LA.pinv(curr_cov) # initial inverse covariance
          
          cov_dict[i] = curr_cov 
     
          curr_rxdist_train = []
          
          for x_sample in curr_train_all:
              rxdist = RXdistance(x_sample, curr_mu, curr_icov)
              curr_rxdist_train.append(rxdist[0][0])
          
          curr_rxdist_train = np.array(curr_rxdist_train)
          
          rx_matrix_train[:,i] = curr_rxdist_train     
          
          curr_rxdist = []
          
          for x_sample in curr_test:
              rxdist = RXdistance(x_sample, curr_mu, curr_icov)
              curr_rxdist.append(rxdist[0][0])
          
          curr_rxdist = np.array(curr_rxdist)
          
          rx_matrix[:,i] = curr_rxdist
     
     
          curr_thr = np.mean(curr_rxdist) + thr_val[j] * np.std(curr_rxdist)
          
          thr_list.append(curr_thr)
          
          rx_anomaly = []
          
          for dist in curr_rxdist:
               if dist > curr_thr:
                    rx_anomaly.append(1)
               else:
                    rx_anomaly.append(0)
                    
          rx_anomaly = np.array(rx_anomaly)      
          
          anomaly_matrix[:,i] = rx_anomaly
          
          if i % 10 == 0:
               print 'Experiment for ', i, ' number of spatial regions completed'
     
       
     rx_full = np.concatenate((rx_matrix_train, rx_matrix), axis = 0)
     
     an = np.sum(anomaly_matrix, axis = 1)
     
     an_reg = np.sum(anomaly_matrix, axis = 0)
     
     y_pred = []
     
     for val in an:
          if val > 0:
               y_pred.append(1)
          else:
               y_pred.append(0)
     
     
     y_pred = np.array(y_pred)
          
          
     f1 = f1_score(y_test, y_pred)
     
     cm = confusion_matrix(y_test, y_pred)
     
     
     target_names = ['Normal', 'Anomalies']
#     print(classification_report(y_test, y_pred, target_names=target_names))
     
     cm = confusion_matrix(y_test, y_pred)
     
     plot_confusion_matrix(cm, target_names, title='Confusion_Matrix_ECD_redundancy_'+str(i))
     
     
     ecdt_redundancy = []
     
     for m in range(len(y_test)):
          if y_pred[m] == 0:
               ecdt_redundancy.append(np.min(rx_matrix[m,:]))
          else:
               ecdt_redundancy.append(np.max(rx_matrix[m,:]))
     
     ecdt_redundancy = np.array(ecdt_redundancy)


     ecd_decision_scores[:,j] = ecdt_redundancy
     
     se_decision_scores[:,j] = se_test


     an_overall[:,j] = an
     
     y_test_overall[:,j] = y_test
     
     y_pred_overall[:,j] = y_pred
     
     thr_overall[:,j] = np.array(thr_list)
     
     
     data_splits = np.roll(data_splits, shift= -1, axis=0)

     
thr_list_df = pd.DataFrame(thr_overall)
thr_list_df.to_csv('ECD_thr_overall_redundancy_optimal_thresholds.csv')


an_df = pd.DataFrame(an_overall)
an_df.to_csv('ECD_anomaly_predictions_redundancy_optimal_thresholds.csv')


ecdt_redundancy_pd = pd.DataFrame(ecd_decision_scores)
ecdt_redundancy_pd.to_csv('ECD_decision_scores_redundancy_optimal_thresholds.csv')

y_test_pd = pd.DataFrame(y_test_overall)
y_test_pd.to_csv('ECD_y_test_overall_redundancy.csv')

y_pred_pd = pd.DataFrame(y_pred_overall)
y_pred_pd.to_csv('ECD_y_pred_overall_redundancy_optimal_thresholds.csv')


se_redundancy_pd = pd.DataFrame(se_decision_scores)
se_redundancy_pd.to_csv('SE_decision_scores_redundancy.csv')

feat_df = pd.DataFrame(feat)
feat_df.to_csv('Spatial_regions_features_mapping.csv')

#for i in range(118):
#     
#     th = [thr_list[i]] * 10000
#     th = np.array(th).reshape(10000,)
#     
#     plt.figure()
#     plt.plot(rx_full[:,i])
#     plt.plot(th, '--k')
#     name = 'ecd_figures_sg/ecd_sg' + str(i) + '.png'
#     plt.savefig(name)
#





#Fusion results

#se_predictions = pd.read_csv('se_pred_best_test_redundancy.csv')
#se_pred = np.array(se_predictions.iloc[:,1].values)

#
#se_dist_train = se_distance[:train_size]
#
#f1_se = []
#thr_se_arr = []
#
#for thr_se in range(300, 380):          
#     y_pred_se_train = []
#     
#     
#     for i in se_dist_train:
#          if i > thr_se:
#               y_pred_se_train.append(1)
#          else:
#               y_pred_se_train.append(0)
#                    
#          
#     y_pred_se_train = np.array(y_pred_se_train)
#               
#     f1_se_temp = f1_score(y_train, y_pred_se_train)
#     
#     f1_se.append(f1_se_temp)
#     
#     thr_se_arr.append(thr_se)
#    
#
#best_thr_se  = thr_se_arr[np.argmax(f1_se)]
#
##best_thr_se = 350
#
#y_pred_se = []
#
#se_dist_test = se_distance[train_size:]
#
#for i in se_dist_test:
#     if i > best_thr_se:
#          y_pred_se.append(1)
#     else:
#          y_pred_se.append(0)
#               
#     
#y_pred_se = np.array(y_pred_se)
#
#se_pred = y_pred_se



# Ground truth Vs State estimator 

#print(" ")
#
#cm_SE = confusion_matrix(y_test, se_pred_test)
#print("Ground truth Vs State Estimator")
#target_names = ['Normal', 'Anomalies']
#print(classification_report(y_test, se_pred_test, target_names=target_names))
#plot_confusion_matrix(cm_SE, target_names, title='Confusion_Matrix_SE_ECDT_redundancy')
#
#
## Ground truth Vs Combined results (AND)
#print(" ")
#y_and = [0] * len(se_pred_test)
#y_and = np.array(y_and)
#
#for i in range(len(y_and)):
#    if y_pred[i] and se_pred_test[i] == 1:
#        y_and[i] = 1
#        
#
#cm_and = confusion_matrix(y_test, y_and)
#print("Ground truth Vs Combined results (AND)")
#target_names = ['Normal', 'Anomalies']
#print(classification_report(y_test, y_and, target_names=target_names))
#plot_confusion_matrix(cm_and, target_names, title='Confusion_Matrix_ECDT_redundancy_AND')
#
#
## Ground truth Vs Combined results (OR)
#print(" ")
#y_or = [0] * len(se_pred_test)
#y_or = np.array(y_or)
#
#for i in range(len(y_or)):
#    if y_pred[i] or se_pred_test[i] == 1:
#        y_or[i] = 1
#        
#
#cm_or = confusion_matrix(y_test, y_or)
#
#print("Ground truth Vs Combined results (OR)")
#target_names = ['Normal', 'Anomalies']
#print(classification_report(y_test, y_or, target_names=target_names))
#plot_confusion_matrix(cm_or, target_names, title='Confusion_Matrix_ECDT_redundancy_OR')
#
#
#
#
#y_measurement = []
#
#for i in range(train_size, len(err_in[0])):
#        y_measurement.append(err_in[0][i])
#
#
#y_measurement = np.array(y_measurement)
#
#
#fn_sample_numbers = []
#fp_sample_numbers = []
#
#for i in range(len(y_pred)):
#     if y_pred[i] == 0 and y_test[i] == 1:
#          fp_sample_numbers.append(i)
#     elif y_pred[i] == 1 and y_test[i] == 0:
#          fn_sample_numbers.append(i)
#
#
#fn_regions = []
#
#for i in fn_sample_numbers:
#     print(np.sum(anomaly_matrix[i,:]))
#     fn_regions.append(np.argmax(anomaly_matrix[i,:]))
#
##fn_regions = [x+1 for x in fn_regions]
#
#fp_regions = []
#fp_measurement = []
#
#
#for i in fp_sample_numbers:
#     fp_measurement.append(y_measurement[i])
#     fp_regions.append(reg[y_measurement[i]-1])
#
#
#
#fp_region_list = []
#
#for i in fp_regions:
#     for j in i:
#          fp_region_list.append(j)
#
#
##fp_region_list = [x+1 for x in fp_region_list]
#
#fn_set = list(set(fn_regions))
#fp_set = list(set(fp_region_list))
#
#fp_count = {}
#
#for i,val in enumerate(fp_set):
#     fp_count[val] = fp_region_list.count(val)
#     
#
#error_regions_list = []
#
#error_regions_list.extend(fn_regions)
#
#error_regions_list.extend(fp_region_list)
#
#region_wise_error_count = []
#
#for i in range(len(feat)):
#     region_wise_error_count.append(error_regions_list.count(i))
#
#
#region_wise_length = []
#
#for i in feat:
#     region_wise_length.append(len(i))
#
#
##case118_redundancy_10000_20190515T161311.mat
#
#anomaly_matrix_df = pd.DataFrame(anomaly_matrix)
#anomaly_matrix_df.to_csv('anomaly_matrix_09_05_redundancy_ECDT.csv')
#
#
#rx_matrix_df = pd.DataFrame(rx_matrix)
#rx_matrix_df.to_csv('rx_matrix_09_05_redundancy_ECDT.csv')
#
#
#rx_full_df = pd.DataFrame(rx_full)
#rx_full_df.to_csv('rx_full_09_05_redundancy_ECDT.csv')




#
#fn_sample_number_df = pd.DataFrame(fn_sample_numbers)
#fn_sample_number_df.to_csv('FN_sample_number_08_21_redundancy_cordet.csv')
#
#fn_regions_df = pd.DataFrame(fn_regions)
#fn_regions_df.to_csv('FN_regions_08_21_redundancy_cordet.csv')
#
#fp_sample_number_df = pd.DataFrame(fp_sample_numbers)
#fp_sample_number_df.to_csv('FP_sample_number_08_21_redundancy_cordet.csv')
#
#import csv
#
#w = csv.writer(open("features_regions_08_21_cordet.csv", "w"))
#for key, val in feat_dict.items():
#     w.writerow([key, val])
#
#fp_regions_dict = {}
#
#for i in range(len(fp_regions)):
#     fp_regions_dict[i] = fp_regions[i]
#
#w1 = csv.writer(open("FP_regions_08_21_cordet.csv", "w"))
#for key, val in fp_regions_dict.items():
#     w1.writerow([key, val])