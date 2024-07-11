# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 13:11:35 2021

@author: Administrator
"""
import globalmap as GL
import numpy as np

#np.random.seed(0)


def testing_data_generating(code,SNR,max_frame):
    #retrieving global paramters of the code
    n = code.check_matrix_column
    k = code.k    
    sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR/10)))
    testing_data_labels = np.zeros((max_frame,n),dtype=np.int64)
    #starting off the data generating process
    mean = 1   
    if GL.get_map('Rayleigh_fading'):  
        points = max_frame*n # 栅网采样点数 
        f_sample = 1024  # 采样频率      
        time_length = points/f_sample  # 仿真时长
        frequency = 64  # 信号频率
        samples_per_period = f_sample / frequency  # 每周期采样点数
        period_num = time_length * frequency  # 共采样多少周期
        duration = GL.get_map('duration')
        randn = np.random.normal       
        h = np.zeros(points,dtype=complex)
        for i in range(int(period_num / duration)):
            h.real[i*int(samples_per_period * duration):(i+1) * int(samples_per_period * duration)] = randn()
            h.imag[i*int(samples_per_period * duration):(i+1) * int(samples_per_period * duration)] = randn()
        h = h / np.sqrt(2)  #normalization so as to compare with equivalent awgn SNR
        #
        h = np.reshape(h,[-1,n])
        channel_information = np.abs(h) + randn(0,sigma,size=(max_frame,n))
        #channel_information = 1 + randn(0,sigma,size=(max_frame,n))
    else: #default is awgn channel
        channel_information =  np.random.normal(mean,sigma,size=(max_frame,n))
    if not GL.get_map('ALL_ZEROS_CODEWORD_TESTING'):
        rand_message = np.random.randint(0,2,size=[max_frame,k],dtype=int)
        codewords = rand_message.dot(code.G)%2
        testing_data = np.where(codewords==0,channel_information,-channel_information)
        testing_data_labels = codewords
    else:
        testing_data = channel_information
    # Whether to use channel estimation as initialization of input(LLR)
    #if GL.get_map('SIGMA_SCALING_TESTING'):
        #testing_data= 2.0*testing_data/(sigma*sigma)
    return testing_data,testing_data_labels
