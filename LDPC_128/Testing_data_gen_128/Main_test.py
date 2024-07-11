# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:31:50 2021

@author: Administrator
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 16:17:48 2021

@author: Administrator
"""
import tensorflow as tf
import numpy as np
import sys
import os
# Run as follows:
np.set_printoptions(precision=3)
import fill_matrix_info as Fill_matrix
import globalmap as GL
import data_generating as Data_gen

sys.argv = "python 2.0 3.0 6 100 1000 CCSDS_ldpc_n128_k64.alist".split()
                                 
#command line arguments
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('snr_num', int(sys.argv[3]))
GL.set_map('batch_size', int(sys.argv[4]))

GL.set_map('testing_batch_number', float(sys.argv[5]))
GL.set_map('H_filename', sys.argv[6])

GL.set_map('Rayleigh_fading', False)

batch_size = GL.get_map('batch_size')
test_batch = GL.get_map('testing_batch_number')
nDatas_test = test_batch*batch_size 

if GL.get_map('Rayleigh_fading'):  
    duration = 1  # Rayleigh fading duration   
    GL.set_map('duration',duration)
    suffix = 'Rayleigh_awgn_duration_'+str(GL.get_map('duration'))
else:
    suffix = 'Awgn'

# setting global parameters

GL.set_map('ALL_ZEROS_CODEWORD_TESTING', False)

GL.set_map('portion_dis', '0.05 0.075 0.2 0.5 0.75 1.0') #for generating testing data of various sizes to be aligned with  sys.argv argument

H_filename=GL.get_map('H_filename')


code = Fill_matrix.Code(H_filename)
GL.set_map('code_parameters', code)

#training setting
#retrieving global paramter values of the code
n = code.check_matrix_column

snr_lo = round(GL.get_map('snr_lo'),2)
snr_hi = round(GL.get_map('snr_hi'),2)
snr_num = GL.get_map('snr_num')
SNRs = np.linspace(snr_lo,snr_hi, snr_num)
   
def get_tfrecords_example(feature, label):
    tfrecords_features = {}
    feat_shape = feature.shape
    tfrecords_features['feature'] = tf.train.Feature(float_list=tf.train.FloatList(value=feature))
    tfrecords_features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
    tfrecords_features['shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=list(feat_shape)))
    return tf.train.Example(features = tf.train.Features(feature = tfrecords_features))
#writing all data to tfrecord file
def make_tfrecord(data, out_filename):
    feats, labels = data
    tfrecord_wrt = tf.io.TFRecordWriter(out_filename)
    ndatas = len(labels)
    for inx in range(ndatas):
        exmp = get_tfrecords_example(feats[inx], labels[inx])
        exmp_serial = exmp.SerializeToString()
        tfrecord_wrt.write(exmp_serial)
    tfrecord_wrt.close()
#create directory if not existence   
file_dir = './data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)

string_list = (GL.get_map('portion_dis')).split()
portion_list = np.array(string_list).astype(float) 
i = 1
for SNR in reversed(SNRs):  
    print(f'Generating test data for {round(SNR,1)}dB')
    percentage = portion_list[-i]
    # make test set
    max_frame = int(percentage*nDatas_test)
    test_data,test_labels = Data_gen.testing_data_generating(code, SNR,max_frame )
    data = (test_data,test_labels)
    if GL.get_map('ALL_ZEROS_CODEWORD_TESTING'):
        ending = 'allzero'
    else:
        ending = 'nonzero'
    filepath = file_dir+'test-'+ending+str(round(SNR,2))+'dB-'+suffix+'.tfrecord'
    make_tfrecord(data, filepath)
    i +=1
print("Data for testing are generated successfully!")

