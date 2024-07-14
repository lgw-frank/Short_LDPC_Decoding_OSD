# -*- coding: utf-8 -*-
import time
T1 = time.process_time()
import os
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5)
#import matplotlib
import sys
import fill_matrix_info as Fill_matrix
import globalmap as GL
import display_selection as Display
import ms_test as Decoder_module
import read_TFdata as Reading
import tensorflow as tf

# Belief propagation using TensorFlow.Run as follows:
# python main.py 0 1.5 3 5 100 1000 10 FG-LDPC273_191 FG-LDPC273_191.g.mat SNNMS/NNMS  191
#          1 2  3 4  5  6  7   8         9          10     11
sys.argv = "python 2.0 3.0 6 1000 100 12 CCSDS_ldpc_n128_k64.alist NMS-1".split()

# setting global parameters

GL.set_map('ALL_ZEROS_CODEWORD_TESTING', False)

#command line arguments
GL.set_map('snr_lo', float(sys.argv[1]))
GL.set_map('snr_hi', float(sys.argv[2]))
GL.set_map('snr_num', int(sys.argv[3]))
GL.set_map('unit_batch_size', int(sys.argv[4]))
GL.set_map('num_batch_test', int(sys.argv[5]))
GL.set_map('num_iterations', int(sys.argv[6]))
GL.set_map('H_filename', sys.argv[7])
GL.set_map('selected_decoder_type', sys.argv[8])

GL.set_map('decoding_threshold',40000)

GL.set_map('ALL_ZEROS_CODEWORD_TESTING',False)

GL.set_map('Rayleigh_fading', False)

if GL.get_map('Rayleigh_fading'):
    GL.set_map('duration', 1)
    suffix = 'Rayleigh_awgn_duration_'+str(GL.get_map('duration'))
else:
    suffix = 'Awgn'

decoder_type = GL.get_map('selected_decoder_type')
n_iteration = GL.get_map('num_iterations')
snr_lo = round(GL.get_map('snr_lo'),2)
snr_hi = round(GL.get_map('snr_hi'),2)
H_filename=GL.get_map('H_filename')
code = Fill_matrix.Code(H_filename)
GL.set_map('code_parameters', code)
n_dims = GL.get_map('code_parameters').check_matrix_column

if decoder_type != 'SPA':
    restore_ckpts_dir = '../Ldpc_'+str(n_dims)+'_training/ckpts/'+decoder_type+'/'+str(n_iteration)+'th'+'/'
    #instance of Model creation
    test_Model = Decoder_module.Decoding_model()
    # saved restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=test_Model)
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        print('Loading the saved model!')
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()
else:
    test_Model = Decoder_module.Decoding_model()

# the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
unit_batch_size = GL.get_map('unit_batch_size')

snr_lo = round(GL.get_map('snr_lo'),2)
snr_hi = round(GL.get_map('snr_hi'),2)
snr_num = GL.get_map('snr_num')
SNRs = np.linspace(snr_lo,snr_hi,snr_num)

data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
Display.display_selection()


logdir = './log/'
if not os.path.exists(logdir):
    os.makedirs(logdir)
log_filename = logdir+'FER-'+decoder_type+'-'+str(n_iteration)+'th'+'.txt'

#passing the step of restoring decoding with saved parameters for types of SPA or MS
start_test_step = 0
Test_total_fer = 0.
Test_total_ber = 0.
# reading in training/validating data;make dataset iterator
if GL.get_map('ALL_ZEROS_CODEWORD_TESTING'):
    ending = 'allzero'
else:
    ending = 'nonzero'
FER_list = []
for SNR in SNRs:
    snr = round(SNR,2)
    output_dir = data_dir+'/'+str(decoder_type)+'/'+str(n_iteration)+'th/'+str(snr)+'dB/'
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    counter = 0
    Test_total_fer = 0.
    Test_total_ber = 0.
    iput_file =  data_dir +'test-'+ending+str(snr)+'dB-'+suffix+'.tfrecord'
    sigma = np.sqrt(1. / (2 * (float(code.k)/float(code.check_matrix_column)) * 10**(SNR/10)))
    GL.set_map('noise_standard_variance',sigma)
    dataset_test = Reading.data_handler(code.check_matrix_column,iput_file,unit_batch_size)
    input_list = list(dataset_test.as_numpy_iterator())
    num_counter = len(input_list)
    #decoding_failure container
    buffer_inputs = []
    buffer_labels = []
    undetected_sum = 0
    for i in range(num_counter):
        inputs = input_list[i]
        fer,ber,undetected_count,buffer = test_Model(inputs[0],inputs[1])
        #print(test_Model.trainable_variables)
        buffer_inputs.append(buffer[0])
        buffer_labels.append(buffer[1])
        Test_total_fer = Test_total_fer+fer
        Test_total_ber = Test_total_ber+ber
        undetected_sum += undetected_count
        counter += 1
        #print(test_Model.trainable_variables)
        if counter % 100 == 0:
          print("%.4f codewords tested, FER:%.4f"%(counter*unit_batch_size,Test_total_fer/counter))
        if Test_total_fer > GL.get_map('decoding_threshold')/unit_batch_size:
          break
    average_test_fer1 = Test_total_fer/counter
    average_test_ber1 = Test_total_ber/counter
    average_undetected_fer = undetected_sum/(counter*unit_batch_size)
    print(counter," batches tested!")
    print ("FER %.4f, BER %.4f,UFER %.4f"%(average_test_fer1,average_test_ber1,average_undetected_fer))
    FER_list.append((snr,round(average_test_fer1,5)))
    with open(log_filename,'a+') as f:
      f.write('\nFor %.1fdB summary:\n'%round(SNR,2))
      f.write("FER %.4f, BER %.4f,UFER %.6f"%(average_test_fer1,average_test_ber1,average_undetected_fer)+'\n')
    #Start off postprocessing
    buffer = (buffer_inputs,buffer_labels)
    updated_buffer = test_Model.postprocess_failure_cases(buffer)

    if GL.get_map('ALL_ZEROS_CODEWORD_TESTING'):
        file_name = 'ldpc-allzero-retest.tfrecord'
    else:
        file_name = 'ldpc-nonzero-retest.tfrecord'
    output_file = output_dir+file_name
    Decoder_module.save_decoded_data(updated_buffer,output_file,SNR,log_filename,n_iteration+1)
print((f"FER_list:{FER_list}"))
with open(log_filename,'a+') as f:
  f.write(f"FER_list:{FER_list}")
  
T2 =time.process_time()
print('程序运行时间:%s秒' % (T2 - T1))