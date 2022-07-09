# -*- coding: utf-8 -*-
"""
Created on Tue Mar  31 15:41:22 2020

@author: keerthiraj
"""

# Generating CSV files from .mat file
import pandas as pd
from scipy.io import loadmat

print("loading matlab file...")
data = loadmat('../mat files/01-01-2018_mfdi_mdos.mat')

print("generating orz .csv...")
orz = data['att_z_ml']
orz = pd.DataFrame(orz)
orz.to_csv('../data csv files/orz_mfdi_mdos.csv')

print("generating err_ins_DoS_attack.csv...")
err_ins = data['err_ins_FDI']
err_ins = pd.DataFrame(err_ins)
err_ins.to_csv('../data csv files/err_ins_mfdi_mdos.csv')
#

print("generating err_ins_DoS_attack.csv...")
err_ins = data['DoS_attack']
err_ins = pd.DataFrame(err_ins)
err_ins.to_csv('../data csv files/err_ins_mdos_mfdi.csv')

print("generating err_ins_DOS_.csv...")
err_attack = data['err_ins_DoS']
err_attack = pd.DataFrame(err_attack)
err_attack.to_csv('../data csv files/err_dos_attack_mdos_mfdi.csv')


print("generating err_ins_DOS_.csv...")
err_attack = data['err_ins_DoS']
err_attack = pd.DataFrame(err_attack)
err_attack.to_csv('../data csv files/err_attack_inj_multi.csv')

print("generating err_type_.csv...")
err_type = data['err_type']
err_type = pd.DataFrame(err_type)
err_type.to_csv('../data csv files/err_type_mfdi_mdos.csv')


print("generating z.csv...")
z  = data['z_ml']
z = pd.DataFrame(z)
z.to_csv('../data csv files/z_mfdi_mdos.csv')


print("Done!")
