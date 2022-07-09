# -*- coding: utf-8 -*-
"""
Created on Mon Dec  04 08:38:22 2019

@author: keert
"""

# Generating CSV files from .mat file
import pandas as pd
from scipy.io import loadmat

import matplotlib.pyplot as plt

print("loading matlab file...")

#%%
data = loadmat('../data mat files/01-08-2018.mat')

#%%
data1 = loadmat('../data mat files/01-02-2017.mat')

data2 = loadmat('../data mat files/01-04-2016.mat')

data3 = loadmat('../data mat files/01-08-2018.mat')

data4 = loadmat('../data mat files/01-09-2017.mat')

data5 = loadmat('../data mat files/01-11-2016.mat')

#%%

dat = data['org_z_ml']
dat1 = data1['org_z_ml']
dat2 = data2['org_z_ml']

dat3 = data3['org_z_ml']
dat4 = data4['org_z_ml']
dat5 = data5['org_z_ml']

plt.figure()
plt.plot(dat[0,:], '.r')
plt.title('2018-1')
plt.savefig('2018-1.png')

plt.figure()
plt.plot(dat1[0,:], '.b')
plt.title('2017-1')
plt.savefig('2017-1.png')

plt.figure()
plt.plot(dat2[0,:], '.g')
plt.title('2016-1')
plt.savefig('2016-1.png')


plt.figure()
plt.plot(dat3[0,:], '.r')
plt.title('2018-2')
plt.savefig('2018-2.png')


plt.figure()
plt.plot(dat4[0,:], '.b')
plt.title('2017-2')
plt.savefig('2017-2.png')


plt.figure()
plt.plot(dat5[0,:], '.g')
plt.title('2016-2')
plt.savefig('2016-2.png')


#%%

# measurement type 1: feature 333
# measurement type 2: feature 337
# measurement type 3: feature 340
# measurement type 4: feature: 334
# measurement type 5: feature 0
# measurement type 6: feature 108
# measurement type 7: feature: 215

meas = [333, 337, 340, 334, 0, 108, 215]

for feat in meas:
    plt.figure()
    plt.plot(dat3[feat,:])
    title = 'feature number : ' + str(feat)
    plt.title(title)
    
    figname = title + '.png'
    # plt.savefig(figname)



#%%

print("Done!")
