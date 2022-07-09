# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 05:09:41 2021

@author: keert
"""

import numpy as np
import csv
import pandas as pd
import time

#%%



data_pd_iat = pd.read_csv('../data csv files/IAT_multi_part_dos.csv')
data_pd_iat = data_pd_iat.drop(data_pd_iat.columns[0], axis = 1)
data_iat = np.array(data_pd_iat)

#%%

data_pd_td = pd.read_csv('../data csv files/TD_multi_part_dos.csv')
data_pd_td = data_pd_td.drop(data_pd_td.columns[0], axis = 1)
data_td = np.array(data_pd_td)

#%%

for i in range(len(data_iat)):
    fp_iat =  open("local csv files/data_source_3_dos_iat.csv", "a", newline="")
    writer = csv.writer(fp_iat, delimiter=",")
        # writer.writerow(["your", "header", "foo"])  # write header
    writer.writerow(data_iat[i,:])
    fp_iat.close()
    
    fp_td =  open("local csv files/data_source_3_dos_td.csv", "a", newline="")
    writer = csv.writer(fp_td, delimiter=",")
        # writer.writerow(["your", "header", "foo"])  # write header
    writer.writerow(data_td[i,:])
    fp_td.close()
    
    
    time.sleep(4)

#%%
