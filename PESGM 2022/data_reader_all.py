# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 05:09:41 2021

@author: keert
"""

# =============================================================================
# CECD-AS Real time integration
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import csv
import os

#%%

import sys
sys.path.append('../helper files')

import helper_functions as hf

#%%

import warnings
warnings.filterwarnings("ignore")


#%%


err_ins_pd = pd.read_csv('../data csv files/err_ins.csv')
err_ins = hf.df_to_array(err_ins_pd, trans = True)
err_ins = err_ins.reshape(len(err_ins),)

y_data = hf.get_ground_truth_from_arr(err_ins)

z = pd.read_csv('../data csv files/z.csv')
feat = hf.get_grid_connections(z)

train_size = 1800 #1800  #2 hours for initial training


#%%

class Watcher(object):
    def __init__(self):
        self._cached_stamp = 0
        self.filename = '' #'local csv files/data_source_1.csv'

    def changecheck(self):
        stamp = os.stat(self.filename).st_mtime
        out = 0
        
        if self._cached_stamp is not None:
            if stamp != self._cached_stamp:
                self._cached_stamp = stamp
                # File has changed, so do something...
                #print(stamp)
                out = 1
        
        return out    


#%%


filename1 = 'local csv files/data_source_1_sg.csv'
# filename2 = 'local csv files/data_source_2_fdi_iat.csv'
# filename3 = 'local csv files/data_source_2_fdi_td.csv'

watch1 = Watcher() 
watch1.filename = filename1

# watch2 = Watcher() 
# watch2.filename = filename2

# watch3 = Watcher() 
# watch3.filename = filename3

data_sg = []

data_iat = []

data_td = []

#%%

while True: 
    try: 
        time.sleep(1) 
        #print(pub.changecheck())
        
        # if watch1.changecheck() or watch2.changecheck() or watch3.changecheck():
            
        if watch1.changecheck():
            with open(filename1, "r") as f:
                for line in f: pass
                print('SG data collected: ', line[:50], ' ....') #this is the last line of the file            
                data_sg.append(line)
            # else:
            #     pass
                    
            # if watch2.changecheck():
            #     with open(filename2, "r") as f:
            #         for line in f: pass
            #         print('IAT data collected')#line) #this is the last line of the file            
            #         data_iat.append(line)
            # else:
            #     pass
                    
            # if watch3.changecheck():
            #     with open(filename3, "r") as f:
            #         for line in f: pass
            #         print('TD data collected')#line) #this is the last line of the file            
            #         data_td.append(line)     
            # else:
            #     pass
        else:
            pass
            #print(" ")
            #print('Waiting for next data sample')
                
    except KeyboardInterrupt: 
        print('\nDone') 
        break 
    except: 
        print(f'Unhandled error: {sys.exc_info()[0]}')
    
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

