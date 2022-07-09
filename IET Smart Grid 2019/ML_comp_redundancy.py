# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 16:17:21 2019

@author: keerthiraj
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import f1_score, auc, roc_curve
#
##
#from dbn.tensorflow import SupervisedDBNClassification


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import IsolationForest


#
#classifiers = [
#    KNeighborsClassifier(1),
#    KNeighborsClassifier(3),
##    KNeighborsClassifier(5),
#    SVC(kernel="rbf", gamma=2, C=1),
#    DecisionTreeClassifier(max_depth=500),                  
#    AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=500) , n_estimators=200, random_state=0),
#    GaussianNB(var_smoothing=1e-06),
#    SupervisedDBNClassification(hidden_layers_structure=[256, 256],
#                                         learning_rate_rbm=0.05,
#                                         learning_rate=0.1,
#                                         n_epochs_rbm=10,
#                                         n_iter_backprop=100,
#                                         batch_size=32,
#                                         activation_function='relu',
#                                         dropout_p=0.2)]

# Read the Input Dataset
orz = pd.read_csv('orz_redundancy.csv')
err_ins = pd.read_csv('err_ins_redundancy.csv')
err_id = pd.read_csv("err_id_redundancy.csv", header = None)
se_dist = pd.read_csv("se_dist_redundancy.csv")
z = pd.read_csv('z_redundancy.csv')


thr_val = pd.read_csv('best_threshold_ecdt_redundancy.csv')


train_size = 3000



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



print 'classifiers'



# =============================================================================
# Defining model parameters

names = ["KNN-1", "KNN-3", "KNN-5", "KNN-10", "AdaBoost", "Naive Bayes", "SVC"]


classifiers = [
    KNeighborsClassifier(n_neighbors = 1, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 3, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 5, weights = 'distance'),
    KNeighborsClassifier(n_neighbors = 10, weights = 'distance'),
    
                
    AdaBoostClassifier(base_estimator= DecisionTreeClassifier(max_depth=500) , n_estimators=500, random_state=0),
    
    GaussianNB(var_smoothing=1e-06),
    
#    IsolationForest(n_estimators=10, max_samples = 2000, contamination = 0.05, 
#                    max_features = data.shape[1], bootstrap = True),
                    
    SVC(kernel="rbf", gamma= 'auto_deprecated', probability = True, class_weight = 'balanced', 
        verbose = True, decision_function_shape = 'ovo'), ]

# =============================================================================


data_splits = []

start = 0
end = 1000

for i in range(10):
     data_splits.append([start, end])
     start += 1000
     end += 1000




se_decision_scores = np.zeros((len(y_data) - train_size, 10))

ecd_decision_scores = np.zeros((len(y_data) - train_size, 10))

an_overall = np.zeros((len(y_data) - train_size, 10))

y_test_overall = np.zeros((len(y_data) - train_size, 10))

y_pred_overall = np.zeros((len(y_data) - train_size, 10))

thr_overall = np.zeros((len(feat), 10))



predict_proba_overall =  np.zeros((len(y_data) - train_size, 10 * len(names)))


cm_dict = {}
f1_dict = {}
auc_dict = {}

col_count = 0

predict_proba_dict = {}

for name, clf in tqdm(zip(names, classifiers), desc = 'classifier'):
     
     print name
     
     data_splits = []
     
     start = 0
     end = 1000
     
     for i in range(10):
          data_splits.append([start, end])
          start += 1000
          end += 1000
     
#     plt.figure()

     for j in tqdm(range(10), desc = 'iter'):
               
          x_train = []
          y_train = []
          
          x_test = []
          y_test = []
          
          se_test = []
          
          se_pred_train = []
          se_pred_test = []
          
          for k in range(10):
               
               if k< int(train_size/1000):
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
                    
#          if name in names[:4]:
#              clf.set_params(metric_params = {'V' : np.cov(x_train, rowvar=False)})
#          
          clf.fit(x_train, y_train)
#          score = clf.score(x_test, y_test)
          y_pred = clf.predict(x_test)
          
#          if name == "IsolationForest":
#              y_pred = clf.predict(x_test)
#              predict_proba_overall[:, col_count] = y_pred
#          else:
          y_pred_proba = clf.predict_proba(x_test)
     
          predict_proba_dict[name] = y_pred_proba[:,1]
     
          predict_proba_overall[:, col_count] = y_pred_proba[:,1]
          
          dict_key = name + '-' + str(j)
          
          print dict_key
          
          f1 = f1_score(y_test, y_pred)  
          cm = confusion_matrix(y_test, y_pred)
          
          cm_dict[dict_key] = cm
          f1_dict[dict_key] = f1
          
#          fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:,1])
#          
#          roc_auc = auc(fpr, tpr)
#
#
#          auc_dict[dict_key] = roc_auc          
#
#
#          
#          lab = name + '-' + str(j) +  '-:-' + str(roc_auc)
     
#          plt.plot(fpr, tpr, lw=1, alpha=0.6, label = lab)
         
          col_count += 1
          
          data_splits = np.roll(data_splits, shift= -1, axis=0)
#


predict_proba_overall_pd = pd.DataFrame(predict_proba_overall)
#predict_proba_overall_pd.to_csv('predict_proba_ml_recent.csv')

#

names_all = []

for name in names:
     for i in range(10):
          names_all.append(name)

plt.figure()


auc_dict = {}

for i in range(predict_proba_overall.shape[1]):
     
     dec_score = predict_proba_overall[:,i]
     
     fpr, tpr, threshold = roc_curve(y_test, dec_score)
     
     roc_auc = auc(fpr, tpr)

     dict_key = names_all[i] + '-' + str(i)
     
     auc_dict[dict_key] = roc_auc          

     
#     lab = names_all[i] + '-' + str(j) +  '-:-' + str(roc_auc)

     plt.plot(fpr, tpr, lw=1, alpha=0.6)
#

import csv



with open('ML_comp_redundancy_confusion_matrices.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in cm_dict.items():
       writer.writerow([key, value])

     
with open('ML_comp_redundancy_f1_scores.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in f1_dict.items():
       writer.writerow([key, value])
       
       
with open('ML_comp_redundancy_AUC_scores.csv', 'w') as csv_file:  
    writer = csv.writer(csv_file)
    for key, value in auc_dict.items():
       writer.writerow([key, value])
       


## %%
#names = ["DBNN"]
#
#
#classifiers = [SupervisedDBNClassification(hidden_layers_structure=[256, 256],
#                                         learning_rate_rbm=0.05,
#                                         learning_rate=0.1,
#                                         n_epochs_rbm=10,
#                                         n_iter_backprop=100,
#                                         batch_size=32,
#                                         activation_function='relu',
#                                         dropout_p=0.2)]
##
##names = ["RBF-SVC"]
##
##
##classifiers = [ SVC(kernel="rbf", gamma=2, C=1, probability = True, class_weight = {0: 0.95, 1:0.05})]
#
## %%
#    
#for name, clf in zip(names, classifiers):
#     
#     print(name)
#     
#     clf.fit(x_train, y_train)
#     
#     y_pred_proba = clf.predict_proba(x_test)
#     
#     f1 = f1_score(y_test, y_pred)  
#     cm = confusion_matrix(y_test, y_pred)
#     
#     
#     predict_proba_dict[name] = y_pred_proba[:,1]
#     
#     fpr, tpr, threshold = roc_curve(y_test, y_pred_proba[:,1])
#          
#     roc_auc = auc(fpr, tpr)
#
#
#     auc_dict[name] = roc_auc          
#
#
#          
#     lab = name + '-:-' + str(roc_auc)
#     
#     plt.plot(fpr, tpr, lw=1, alpha=0.8, label = lab)
##     
#
#
#
##     
#for name, clf in zip(names, classifiers):
#     
#     print(name)
#     y_pred = clf.predict(x_test)
#     
#     
##          y_pred_proba = clf.predict_proba(x_test)
#     target_names = ['Normal', 'Attacked']
#     print(classification_report(y_test, y_pred, target_names=target_names)) 
#     
##     print confusion_matrix(y_test, y_pred)
#
#
##     
##
##
##
##plt.xlim([-0.05, 1.05])
##plt.ylim([-0.05, 1.05])
##plt.xlabel('False Positive Rate')
##plt.ylabel('True Positive Rate')
##plt.grid(True)
##plt.title('Receiver operating characteristics')
##plt.legend(loc="lower right", fontsize = 8)
