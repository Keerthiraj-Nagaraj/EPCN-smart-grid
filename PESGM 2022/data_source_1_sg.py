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

data_pd = pd.read_csv('../data csv files/att_z_gaussian_ml_s4_0_3_ed.csv')
data_pd = data_pd.drop(data_pd.columns[0], axis = 1)
data = np.array(data_pd)

#%%

for row in data:
    fp =  open("local csv files/data_source_1_sg.csv", "a", newline="")
    writer = csv.writer(fp, delimiter=",")
        # writer.writerow(["your", "header", "foo"])  # write header
    writer.writerow(row)
    fp.close()
    time.sleep(4)

#%%