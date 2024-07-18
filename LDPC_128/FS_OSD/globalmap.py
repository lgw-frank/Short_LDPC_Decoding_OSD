"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
import numpy as np
import os,sys

map = {}
def set_map(key, value):
    map[key] = value
def del_map(key):
    try:
        del map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")
def get_map(key):
    try:
        if key in "all":
            return map
        return map[key]
    except KeyError :
        print ("key:'"+str(key)+"' non-existence")

#global parameters setting
def global_setting(argv):
    #command line arguments
    set_map('snr_lo', float(argv[1]))
    set_map('snr_hi', float(argv[2]))
    set_map('snr_num',int(argv[3]))
    set_map('unit_batch_size',int(argv[4]))
    set_map('num_iterations', int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type',argv[7])
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)   
    
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    set_map('order_limit',3)
    set_map('termination_num_threshlod',100)
    set_map('miracle_view',False) 
    set_map('convention_osd',False)
    set_map('fs_osd',True)
    set_map('d_min',14)
    set_map('tau_psc',30)

def data_setting():
    code = get_map('code_parameters')
    n_dims = code.check_matrix_column
    batch_size = get_map('unit_batch_size')
    snr_num = get_map('snr_num')
    snr_lo = get_map('snr_lo')
    snr_hi = get_map('snr_hi')
    snr_list = np.linspace(snr_lo,snr_hi,snr_num)
    n_iteration = get_map('num_iterations')
    list_length = n_iteration+1
    data_handler_list = []
    data_dir = '../Testing_data_gen_'+str(n_dims)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    decoder_type = get_map('selected_decoder_type')
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'ldpc-allzero-retest.tfrecord'
    else:
        file_name = 'ldpc-nonzero-retest.tfrecord'    
    for i in range(snr_num):
        snr = str(round(snr_list[i],2))
        input_dir = data_dir+decoder_type+'/'+str(n_iteration)+'th/'+snr+'dB/'
        # reading in training/validating data;make dataset iterator
        file_dir = input_dir+file_name
        dataset_test = Reading.data_handler(code.check_matrix_column,file_dir,batch_size*list_length)
        #dataset_test = dataset_test.take(10)
        # dataset_test = dataset_test.cache()
        data_handler_list.append(dataset_test)
    return data_handler_list,snr_list               
                   