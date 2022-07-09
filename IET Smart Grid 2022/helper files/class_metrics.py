#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 20:43:42 2019

@author: keerthiraj
"""


for i in range(20):
     
     print 'Re-do?'

     redo = input()
     
     if redo == 1:
     
          print 'Input TN - True Negatives'
          
          tn = float(input())
          
          
          print 'Input FP - False Positives'
          
          fp =float(input())
          
          
          print 'Input FN - False  Negatives'
          
          fn = float(input())
          
          
          print 'Input TP - True Positives'
          
          tp = float(input())
          
          acc = (tp+tn)/(tp+fp+fn+tn)
          
          pre = tp/(tp+fp)
          
          rec = tp/(tp+fn)
          
          f1 = (2.0*pre*rec) / (pre+rec)
          
          print 'Acc, Pre, Rec, F1'
          print acc, pre, rec, f1
     else:
          print 'done'
          break

