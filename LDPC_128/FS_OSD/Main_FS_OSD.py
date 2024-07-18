# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import time
T1 = time.process_time()
import sys
import globalmap as GL
import fs_testing as FS_test

sys.argv = "python 2.0 3.0 6 1 12 CCSDS_ldpc_n128_k64.alist NMS-1".split()

#setting a batch of global parameters
GL.global_setting(sys.argv) 
selected_ds,snr_list = GL.data_setting()

#beta list for controlloing trade-off between performance and complexity
beta_list = [0.1]*len(snr_list)
for i in range(len(snr_list)):
   snr = round(snr_list[i],2)
   beta = beta_list[i]
   FS_test.fs_osd(snr,beta,selected_ds[i]) 
   
T2 =time.process_time()
print('\nRunning time:%s seconds!'%(T2 - T1))