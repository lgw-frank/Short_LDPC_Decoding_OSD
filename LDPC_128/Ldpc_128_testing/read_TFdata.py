# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:03:55 2021

@author: Administrator
"""
import tensorflow as tf


def parse_exmp(serial_exmp,code_length):
     feats = tf.io.parse_single_example(serial_exmp, features={'feature':tf.io.FixedLenFeature([code_length], tf.float32),\
     'label':tf.io.FixedLenFeature([code_length],tf.int64), 'shape':tf.io.FixedLenFeature([], tf.int64)})
     soft_input = feats['feature']
     label = feats['label']
     shape = tf.cast(feats['shape'], tf.int32)
     return soft_input, label, shape

  
def get_dataset(fname,code_length):
    dataset = tf.data.TFRecordDataset(fname)
    return dataset.map(lambda x:parse_exmp(x,code_length)) # use padded_batch method if padding needed


  
def data_handler(code_length,file_name,batch_size=1): 
# training dataset
      dataset_train = get_dataset(file_name,code_length)        
      dataset_train = dataset_train.batch(batch_size,drop_remainder=False).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset_train
