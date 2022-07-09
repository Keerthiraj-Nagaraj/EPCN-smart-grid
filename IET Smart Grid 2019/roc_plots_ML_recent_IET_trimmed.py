#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 23:06:36 2019

@author: keerthiraj
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import precision_recall_curve





se_redundancy = pd.read_csv('SE_decision_scores_redundancy.csv')
se_redundancy = se_redundancy.drop(se_redundancy.columns[0], axis=1) # remove column 0 (the feature numbers)
se_redundancy = np.array(se_redundancy).reshape(len(se_redundancy), se_redundancy.shape[1])


cdt_redundancy = pd.read_csv('CDT_decision_scores_redundancy.csv')
cdt_redundancy = cdt_redundancy.drop(cdt_redundancy.columns[0], axis=1) # remove column 0 (the feature numbers)
cdt_redundancy = np.array(cdt_redundancy).reshape(len(cdt_redundancy), cdt_redundancy.shape[1])



ecdt_redundancy = pd.read_csv('ECD_decision_scores_redundancy_optimal_thresholds.csv')
ecdt_redundancy = ecdt_redundancy.drop(ecdt_redundancy.columns[0], axis=1) # remove column 0 (the feature numbers)
ecdt_redundancy = np.array(ecdt_redundancy).reshape(len(ecdt_redundancy), ecdt_redundancy.shape[1])




ml_redundancy = pd.read_csv('predict_proba_ml_recent.csv')
ml_redundancy = ml_redundancy.drop(ml_redundancy.columns[0], axis=1) # remove column 0 (the feature numbers)
ml_redundancy = np.array(ml_redundancy).reshape(len(ml_redundancy), ml_redundancy.shape[1])



y_test_redundancy = pd.read_csv('ECD_y_test_overall_redundancy.csv')
y_test_redundancy = y_test_redundancy.drop(y_test_redundancy.columns[0], axis=1) # remove column 0 (the feature numbers)
y_test_redundancy = np.array(y_test_redundancy).reshape(len(y_test_redundancy), y_test_redundancy.shape[1])



names = ["KNN-1", "KNN-3",
         "Decision Tree", "AdaBoost",
         "Naive Bayes"]

names = ["KNN-1", "KNN-3", "KNN-5", "KNN-10", "AdaBoost", "Naive Bayes", "SVC"]




knn_1 = ml_redundancy[:,0:10]
knn_3 = ml_redundancy[:,10:20]
knn_5 = ml_redundancy[:,20:30]
knn_10 = ml_redundancy[:,30:40]
ada = ml_redundancy[:,40:50]
nbayes = ml_redundancy[:,50:60]
svc = ml_redundancy[:,60:70]


for i in range(ecdt_redundancy.shape[1]):
     ecdt_redundancy[:,i] = (ecdt_redundancy[:,i] - np.mean(ecdt_redundancy[:,i]))/(np.std(ecdt_redundancy[:,i]))


for i in range(se_redundancy.shape[1]):
     se_redundancy[:,i] = (se_redundancy[:,i] - np.mean(se_redundancy[:,i]))/(np.std(se_redundancy[:,i]))
     

fused_rx = np.add(ecdt_redundancy, se_redundancy)


rx_val = np.concatenate((knn_1, knn_3, knn_5, knn_10, ada, nbayes, svc, se_redundancy, ecdt_redundancy, fused_rx, cdt_redundancy), axis=1)

y_test = np.concatenate((y_test_redundancy, y_test_redundancy, y_test_redundancy, 
                         y_test_redundancy, y_test_redundancy, y_test_redundancy, 
                         y_test_redundancy, y_test_redundancy, y_test_redundancy,
                         y_test_redundancy, y_test_redundancy), axis=1)

rx_val = np.array(rx_val)
y_test = np.array(y_test)

#symbols = ['.-r', 'g', '.--b', '.-m', 'k', '.--c']
#
#symbols = []
#symbols.extend(['r']*10)
#symbols.extend(['g']*10)
#symbols.extend(['b']*10)
#symbols.extend(['c']*10)

#
plt.style.use('ggplot')


index_val = np.arange(0,y_test.shape[1])

count = 0

plt.figure()

for i in index_val:
     rx = rx_val[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     #plt.figure()
     plt.plot(fpr, tpr)
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     #plt.title('ROC curve')
     plt.grid(True)
     count = count + 1
#     plt.savefig('roc_constant.png', dpi = 600)



plt.figure()



tprs_knn_1 = []

aucs_knn_1 = []

mean_fpr_knn_1 = np.linspace(0, 1, 100)


for i in range(10):
     rx = knn_1[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_knn_1.append(interp(mean_fpr_knn_1, fpr, tpr))
     tprs_knn_1[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_knn_1.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_knn_1 = np.mean(tprs_knn_1, axis=0)
#mean_tpr_knn_1[-1] = 1.0
mean_auc_knn_1 = auc(mean_fpr_knn_1, mean_tpr_knn_1)
std_auc_knn_1 = np.std(aucs_knn_1)


mfpr_range_limit = np.where(mean_fpr_knn_1 < 0.2)

mean_fpr_knn_1, mean_tpr_knn_1 = mean_fpr_knn_1[:len(mfpr_range_limit[0])], mean_tpr_knn_1[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_knn_1, mean_tpr_knn_1, color='tab:brown', label=r'K-NN (K=1) Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_knn_1, std_auc_knn_1), lw=2, alpha=.8)

std_tpr_knn_1 = np.std(tprs_knn_1, axis=0)
tprs_upper_knn_1 = np.minimum(mean_tpr_knn_1 + std_tpr_knn_1[:len(mfpr_range_limit[0])], 1)
tprs_lower_knn_1 = np.maximum(mean_tpr_knn_1 - std_tpr_knn_1[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_knn_1, tprs_lower_knn_1, tprs_upper_knn_1, color='grey', alpha=.4)




tprs_knn_3 = []

aucs_knn_3 = []

mean_fpr_knn_3 = np.linspace(0, 1, 100)


for i in range(10):
     rx = knn_3[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_knn_3.append(interp(mean_fpr_knn_3, fpr, tpr))
     tprs_knn_3[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_knn_3.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_knn_3 = np.mean(tprs_knn_3, axis=0)
#mean_tpr_knn_3[-1] = 1.0
mean_auc_knn_3 = auc(mean_fpr_knn_3, mean_tpr_knn_3)
std_auc_knn_3 = np.std(aucs_knn_3)


mfpr_range_limit = np.where(mean_fpr_knn_3 < 0.2)

mean_fpr_knn_3, mean_tpr_knn_3 = mean_fpr_knn_3[:len(mfpr_range_limit[0])], mean_tpr_knn_3[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_knn_3, mean_tpr_knn_3, color='magenta', label=r'K-NN (K=3) Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_knn_3, std_auc_knn_3), lw=2, alpha=.8)

std_tpr_knn_3 = np.std(tprs_knn_3, axis=0)
tprs_upper_knn_3 = np.minimum(mean_tpr_knn_3 + std_tpr_knn_3[:len(mfpr_range_limit[0])], 1)
tprs_lower_knn_3 = np.maximum(mean_tpr_knn_3 - std_tpr_knn_3[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_knn_3, tprs_lower_knn_3, tprs_upper_knn_3, color='grey', alpha=.4)




tprs_knn_5 = []

aucs_knn_5 = []

mean_fpr_knn_5 = np.linspace(0, 1, 100)


for i in range(10):
     rx = knn_5[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_knn_5.append(interp(mean_fpr_knn_5, fpr, tpr))
     tprs_knn_5[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_knn_5.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_knn_5 = np.mean(tprs_knn_5, axis=0)
#mean_tpr_knn_5[-1] = 1.0
mean_auc_knn_5 = auc(mean_fpr_knn_5, mean_tpr_knn_5)
std_auc_knn_5 = np.std(aucs_knn_5)


mfpr_range_limit = np.where(mean_fpr_knn_5 < 0.2)

mean_fpr_knn_5, mean_tpr_knn_5 = mean_fpr_knn_5[:len(mfpr_range_limit[0])], mean_tpr_knn_5[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_knn_5, mean_tpr_knn_5, color='purple', label=r'K-NN (K=5) Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_knn_5, std_auc_knn_5), lw=2, alpha=.8)

std_tpr_knn_5 = np.std(tprs_knn_5, axis=0)
tprs_upper_knn_5 = np.minimum(mean_tpr_knn_5 + std_tpr_knn_5[:len(mfpr_range_limit[0])], 1)
tprs_lower_knn_5 = np.maximum(mean_tpr_knn_5 - std_tpr_knn_5[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_knn_5, tprs_lower_knn_5, tprs_upper_knn_5, color='grey', alpha=.4)




tprs_nbayes = []

aucs_nbayes = []

mean_fpr_nbayes = np.linspace(0, 1, 100)


for i in range(10):
     rx = nbayes[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_nbayes.append(interp(mean_fpr_nbayes, fpr, tpr))
     tprs_nbayes[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_nbayes.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_nbayes = np.mean(tprs_nbayes, axis=0)
#mean_tpr_nbayes[-1] = 1.0
mean_auc_nbayes = auc(mean_fpr_nbayes, mean_tpr_nbayes)
std_auc_nbayes = np.std(aucs_nbayes)


mfpr_range_limit = np.where(mean_fpr_nbayes < 0.2)

mean_fpr_nbayes, mean_tpr_nbayes = mean_fpr_nbayes[:len(mfpr_range_limit[0])], mean_tpr_nbayes[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_nbayes, mean_tpr_nbayes, color='darkgreen', label=r'GNB Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_nbayes, std_auc_nbayes), lw=2, alpha=.8)

std_tpr_nbayes = np.std(tprs_nbayes, axis=0)
tprs_upper_nbayes = np.minimum(mean_tpr_nbayes + std_tpr_nbayes[:len(mfpr_range_limit[0])], 1)
tprs_lower_nbayes = np.maximum(mean_tpr_nbayes - std_tpr_nbayes[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_nbayes, tprs_lower_nbayes, tprs_upper_nbayes, color='grey', alpha=.4)




tprs_ada = []

aucs_ada = []

mean_fpr_ada = np.linspace(0, 1, 100)


for i in range(10):
     rx = ada[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_ada.append(interp(mean_fpr_ada, fpr, tpr))
     tprs_ada[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_ada.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_ada = np.mean(tprs_ada, axis=0)
#mean_tpr_ada[-1] = 1.0
mean_auc_ada = auc(mean_fpr_ada, mean_tpr_ada)
std_auc_ada = np.std(aucs_ada)


mfpr_range_limit = np.where(mean_fpr_ada < 0.2)

mean_fpr_ada, mean_tpr_ada = mean_fpr_ada[:len(mfpr_range_limit[0])], mean_tpr_ada[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_ada, mean_tpr_ada, color='tab:orange', label=r'ADA Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_ada, std_auc_ada), lw=2, alpha=.8)

std_tpr_ada = np.std(tprs_ada, axis=0)
tprs_upper_ada = np.minimum(mean_tpr_ada + std_tpr_ada[:len(mfpr_range_limit[0])], 1)
tprs_lower_ada = np.maximum(mean_tpr_ada - std_tpr_ada[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_ada, tprs_lower_ada, tprs_upper_ada, color='grey', alpha=.4)





tprs_svc = []

aucs_svc = []

mean_fpr_svc = np.linspace(0, 1, 100)


for i in range(10):
     rx = svc[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_svc.append(interp(mean_fpr_svc, fpr, tpr))
     tprs_svc[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_svc.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_svc = np.mean(tprs_svc, axis=0)
#mean_tpr_svc[-1] = 1.0
mean_auc_svc = auc(mean_fpr_svc, mean_tpr_svc)
std_auc_svc = np.std(aucs_svc)


mfpr_range_limit = np.where(mean_fpr_svc < 0.2)

mean_fpr_svc, mean_tpr_svc = mean_fpr_svc[:len(mfpr_range_limit[0])], mean_tpr_svc[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_svc, mean_tpr_svc, color='black', label=r'SVC Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_svc, std_auc_svc), lw=2, alpha=.8)

std_tpr_svc = np.std(tprs_svc, axis=0)
tprs_upper_svc = np.minimum(mean_tpr_svc + std_tpr_svc[:len(mfpr_range_limit[0])], 1)
tprs_lower_svc = np.maximum(mean_tpr_svc - std_tpr_svc[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_svc, tprs_lower_svc, tprs_upper_svc, color='grey', alpha=.4)





tprs_se = []

aucs_se = []

mean_fpr_se = np.linspace(0, 1, 100)


for i in range(10):
     rx = se_redundancy[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_se.append(interp(mean_fpr_se, fpr, tpr))
     tprs_se[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_se.append(roc_auc)
     
     plt.plot(fpr, tpr, 'r', lw=1, alpha=0.1)
     

mean_tpr_se = np.mean(tprs_se, axis=0)
#mean_tpr_se[-1] = 1.0
mean_auc_se = auc(mean_fpr_se, mean_tpr_se)
std_auc_se = np.std(aucs_se)


mfpr_range_limit = np.where(mean_fpr_se < 0.2)

mean_fpr_se, mean_tpr_se = mean_fpr_se[:len(mfpr_range_limit[0])], mean_tpr_se[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_se, mean_tpr_se, color='darkred', label=r'SE Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_se, std_auc_se), lw=2, alpha=.8)

std_tpr_se = np.std(tprs_se, axis=0)
tprs_upper_se = np.minimum(mean_tpr_se + std_tpr_se[:len(mfpr_range_limit[0])], 1)
tprs_lower_se = np.maximum(mean_tpr_se - std_tpr_se[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_se, tprs_lower_se, tprs_upper_se, color='grey', alpha=.4)



tprs_cdt = []

aucs_cdt = []

mean_fpr_cdt = np.linspace(0, 1, 100)


for i in range(10):
     rx = cdt_redundancy[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     tprs_cdt.append(interp(mean_fpr_cdt, fpr, tpr))
     
     tprs_cdt[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_cdt.append(roc_auc)
     
     plt.plot(fpr, tpr, 'c', lw=1, alpha=0.1)
     

mean_tpr_cdt = np.mean(tprs_cdt, axis=0)
#mean_tpr_cdt[-1] = 1.0
mean_auc_cdt = auc(mean_fpr_cdt, mean_tpr_cdt)
std_auc_cdt = np.std(aucs_cdt)



mfpr_range_limit = np.where(mean_fpr_cdt < 0.2)

mean_fpr_cdt, mean_tpr_cdt = mean_fpr_cdt[:len(mfpr_range_limit[0])], mean_tpr_cdt[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_cdt, mean_tpr_cdt, color='cyan', 
         label=r'CD Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_cdt, std_auc_cdt), lw=2, alpha=.8)

std_tpr_cdt = np.std(tprs_cdt, axis=0)
tprs_upper_cdt = np.minimum(mean_tpr_cdt + std_tpr_cdt[:len(mfpr_range_limit[0])], 1)
tprs_lower_cdt = np.maximum(mean_tpr_cdt - std_tpr_cdt[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_cdt, tprs_lower_cdt, tprs_upper_cdt, color='grey', alpha=.4)



tprs_ecd = []

aucs_ecd = []

mean_fpr_ecd = np.linspace(0, 1, 100)


for i in range(10):
     rx = ecdt_redundancy[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)
     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]
     
     tprs_ecd.append(interp(mean_fpr_ecd, fpr, tpr))
     tprs_ecd[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_ecd.append(roc_auc)
     
     plt.plot(fpr, tpr,'g', lw=1, alpha=0.1)
     

mean_tpr_ecd = np.mean(tprs_ecd, axis=0)
#mean_tpr_ecd[-1] = 1.0
mean_auc_ecd = auc(mean_fpr_ecd, mean_tpr_ecd)
std_auc_ecd = np.std(aucs_ecd)


mfpr_range_limit = np.where(mean_fpr_ecd < 0.2)

mean_fpr_ecd, mean_tpr_ecd = mean_fpr_ecd[:len(mfpr_range_limit[0])], mean_tpr_ecd[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_ecd, mean_tpr_ecd, color='lime', 
         label=r'ECD Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_ecd, std_auc_ecd), lw=2, alpha=.8)

std_tpr_ecd = np.std(tprs_ecd, axis=0)
tprs_upper_ecd = np.minimum(mean_tpr_ecd + std_tpr_ecd[:len(mfpr_range_limit[0])], 1)
tprs_lower_ecd = np.maximum(mean_tpr_ecd - std_tpr_ecd[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_ecd, tprs_lower_ecd, tprs_upper_ecd, color='grey', alpha=.4,
                 label=r'$\pm$ 1 std. dev.')



tprs_fused = []

aucs_fused = []

mean_fpr_fused = np.linspace(0, 1, 100)


for i in range(10):
     rx = fused_rx[:,i]
     fpr, tpr, threshold = roc_curve(y_test[:,i], rx)

     fpr_range_limit = np.where(fpr < 0.2)
     
     fpr, tpr = fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])]

     tprs_fused.append(interp(mean_fpr_fused, fpr, tpr))
     tprs_fused[-1][0] = 0.0
     
     roc_auc = 5 * auc(fpr[:len(fpr_range_limit[0])], tpr[:len(fpr_range_limit[0])])
     aucs_fused.append(roc_auc)
     
     plt.plot(fpr, tpr, 'b', lw=1, alpha=0.1)
     

mean_tpr_fused = np.mean(tprs_fused, axis=0)
#mean_tpr_fused[-1] = 1.0
mean_auc_fused = auc(mean_fpr_fused, mean_tpr_fused)
std_auc_fused = np.std(aucs_fused)

mfpr_range_limit = np.where(mean_fpr_fused < 0.2)

mean_fpr_fused, mean_tpr_fused = mean_fpr_fused[:len(mfpr_range_limit[0])], mean_tpr_fused[:len(mfpr_range_limit[0])]

plt.plot(mean_fpr_fused, mean_tpr_fused, color='darkblue', 
         label=r'Fused Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc_fused, std_auc_fused), lw=2, alpha=.8)

std_tpr_fused = np.std(tprs_fused, axis=0)

tprs_upper_fused = np.minimum(mean_tpr_fused + std_tpr_fused[:len(mfpr_range_limit[0])], 1)
tprs_lower_fused = np.maximum(mean_tpr_fused - std_tpr_fused[:len(mfpr_range_limit[0])], 0)
plt.fill_between(mean_fpr_fused, tprs_lower_fused, tprs_upper_fused, color='grey', alpha=.4)

#plt.plot([0, 0.2], [0, 1], linestyle='--', lw=2, color='k',
#         label='Chance', alpha=.8)




plt.xlim([-0.005, 0.2])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.title('Receiver operating characteristics')
plt.legend(bbox_to_anchor=(1.75, 0.5), loc="center right", fontsize=9)
plt.savefig('ROC_IET_ML_comp_final_trimmed.png', dpi = 600, bbox_inches="tight")

