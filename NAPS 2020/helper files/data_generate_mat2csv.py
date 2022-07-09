# -*- coding: utf-8 -*-
"""
Created on Mon Dec  04 08:38:22 2019

@author: keert
"""

# Generating CSV files from .mat file
import pandas as pd
from scipy.io import loadmat

#%%


def mat2csv_org(filename, var, csv_file):
    
    mat_filename = '../data mat files latest/' + filename + '.mat'
    data = loadmat(mat_filename)
    orz = data[var]
    orz = pd.DataFrame(orz)
    
    csv_filename = '../data csv files latest/' + csv_file + '.csv'
    
    # print(data.keys())
    
    orz.to_csv(csv_filename)

#%%
# data = loadmat('../data mat files/01-01-2018.mat')

# data1 = loadmat('../data mat files/01-02-2017.mat')

# data2 = loadmat('../data mat files/01-04-2016.mat')

# data3 = loadmat('../data mat files/01-08-2018.mat')

# data4 = loadmat('../data mat files/01-09-2017.mat')

# data5 = loadmat('../data mat files/01-11-2016.mat')

#%%

matdata = loadmat('../data mat files latest/01-08-2018.mat')

print(matdata.keys())



#%%

var = 'err_ins_FDI'

csv_file = 'err_ins_sparam_mixed_new_noise'

orz = matdata[var]
orz = pd.DataFrame(orz)

csv_filename = '../data csv files/' + csv_file + '.csv'
# print(data.keys())
orz.to_csv(csv_filename)

#%%

file = '01-07-2018'
var = 'org_z_ml'

csv_file = 'orz_01-07-2018'

print(file, var, csv_file)
mat2csv_org(file, var, csv_file)

print("Done!")
#%%