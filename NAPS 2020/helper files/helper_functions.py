#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:27:25 2020

@author: keerthiraj
"""


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools


import os

import pandas as pd
from scipy.io import loadmat

#%%


def mat2csv(filename, var, csv_file):
    
    mat_filename = '../data mat files/' + filename + '.mat'
    data = loadmat(mat_filename)
    orz = data[var]
    orz = pd.DataFrame(orz)
    
    csv_filename = '../data csv files/' + csv_file + '.csv'
    
    # print(data.keys())
    
    orz.to_csv(csv_filename)

#%%

def class_metrics(cm):
               
     tn, fp, fn, tp = cm.ravel()
     
     tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
     
     acc = (tp+tn)/(tp+fp+fn+tn)
     
     pre = tp/(tp+fp)
     
     rec = tp/(tp+fn)
     
     f1 = (2.0*pre*rec) / (pre+rec)
          
     
#          print 'Acc, Pre, Rec, F1'
     return [acc, pre, rec, f1]

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

    if not os.path.isdir('./cm_files'):
    	os.mkdir('./cm_files')

    title = './cm_files/' + title

    plt.savefig(title +".png")

