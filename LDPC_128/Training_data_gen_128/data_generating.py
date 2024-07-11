# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 20:18:23 2022

@author: lgw
"""
import numpy as np
import math
from scipy import integrate
import globalmap as GL
import tensorflow as tf

np.random.seed(0)

def f1(x,mid_sigma): 
  y=2/(x**2)*f_w(x,mid_sigma)
  return y
def f2(x,mid_sigma):
  y=4*(1/(x**2)+1/(x**4))*f_w(x,mid_sigma)
  return y
def f_w(x,mid_sigma):
  t = abs(x-mid_sigma)
  y= math.exp(-t)
  return y
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
  feats,labels = data
  ndatas = len(labels)
  with tf.io.TFRecordWriter(out_filename) as file_writer:
      for inx in range(ndatas):   
          feature,label = feats[inx], labels[inx]
          feat_shape = feats[inx].shape
          record_bytes = tf.train.Example(features=tf.train.Features(feature={
              "feature": tf.train.Feature(float_list=tf.train.FloatList(value=feature)),
              "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
              "shape":  tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
          })).SerializeToString()    
          file_writer.write(record_bytes)    
      
  
def training_data_generating(code,SNRs,max_frame):
    #retrieving global paramters of the code
    n = code.check_matrix_column
    k = code.k 
    training_data_labels = np.zeros((max_frame,n),dtype=np.int64)
    noise = np.random.randn(max_frame,n)
    #starting off the data generating process
    SNR1 = SNRs[0]
    SNR2 = SNRs[1]
    
    sigma1 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR1/10)))
    sigma2 =  np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(SNR2/10)))
    mid_SNR = (SNR1+SNR2)/2
    mid_sigma = np.sqrt(1. / (2 * (float(k)/float(n)) * 10**(mid_SNR/10)))
  
    #weight_coefficient for valid density
    if SNR1!=SNR2:
        tmp,_ = integrate.quad(f_w,sigma1,sigma2,args=(mid_sigma))
        weight_coefficient = 1/tmp
        tmp_mean,_ = integrate.quad(f1, sigma1,sigma2,args=(mid_sigma))
        new_mean = weight_coefficient*tmp_mean
        tmp_variance,_ = integrate.quad(f2, sigma1,sigma2,args=(mid_sigma))
        new_variance  = weight_coefficient*tmp_variance- new_mean**2
        #print(new_mean,new_variance)
        sigma = np.sqrt(new_variance)
    else:
        sigma = sigma1
        new_mean = 1
    noise *= sigma 
    noise +=new_mean
    # generate random codewords
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        training_data = noise
    else:
        rand_message = np.random.randint(0,2,size=[max_frame,k],dtype=int)
        codewords = rand_message.dot(code.G)%2
        training_data = np.where(codewords==0,noise,-noise)
        training_data_labels = codewords

  
    return training_data,training_data_labels
