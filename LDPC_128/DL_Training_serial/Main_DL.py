# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:09:34 2023

@author: zidonghua_30
"""
import sys
import globalmap as GL
import nn_training as CNN_RNN
import predict_phase as Predict
import os

# Set KMP_DUPLICATE_LIB_OK environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.argv = "python 2.7 2.7 100 12 CCSDS_ldpc_n128_k64.alist NMS-1".split()

#setting a batch of global parameters
GL.global_setting(sys.argv) 
selected_ds = GL.data_setting()

DIA = True

restore_info = []
indicator_list = []
prefix_list = []

if DIA:
    # one and only one of following will be true
    CNN_indicator = True
    RNN1_indicator = False
    RNN2_indicator = False
    indicator_list = [CNN_indicator,RNN1_indicator,RNN2_indicator]
    prefix_list = ['model_cnn','model_rnn1','model_rnn2']
    restore_info = GL.logistic_setting_model(indicator_list,prefix_list)
#determine the priority of order pattern list
if GL.get_map('reliability_enhance'):
    CNN_RNN.Training_NN(selected_ds,restore_info,indicator_list,prefix_list,DIA) 
if GL.get_map('prediction_order_pattern'):
    restore_predict_model_info = GL.set_predict_model(DIA)
    restore_list = [restore_info,restore_predict_model_info]
    Predict.train_sliding_window(restore_list,indicator_list,prefix_list,DIA)