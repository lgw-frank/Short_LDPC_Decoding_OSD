"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix
import os
import numpy as np

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
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)      
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    set_map('print_interval',100)
    set_map('record_interval',100) 
    set_map('convention_path',False)
    
    set_map('termination_threshold',500)
    set_map('threshold_sum',3)     #threshold for sum of number of non-zero elements across sections
    set_map('training_snr',2.7)
    set_map('segment_num',6)
    set_map('soft_margin',0.9)  
    set_map('decoding_length',30)  
    set_map('sliding_win_width',5)  #ensure the setting is an odd number

def secure_segment_threshold():
    num_seg = get_map('segment_num')  # Assuming this function returns an integer
    code = get_map('code_parameters')  # Assuming this function returns an object with attribute 'k'
    allocation_length = code.k - 1
    basic_length = list(range(1, num_seg))  # Convert range to a list
    num_basic = sum(basic_length)
    
    # Calculate initial segment sizes
    segment_size_list = [int(allocation_length / num_basic * b) for b in basic_length]
    
    # Adjust the last segment size
    segment_size_list[-1] += allocation_length - sum(segment_size_list)
    
    # Add one head
    segment_size_whole = np.insert(segment_size_list, 0, 1)
    
    # Calculate boundaries
    boundary_MRB =  np.insert(np.cumsum(segment_size_whole),0,0)
    
    return segment_size_whole, boundary_MRB  
 
def logistic_setting_model(indicator_list,prefix_list):
    for i,element in enumerate(indicator_list):
        if element == True:
            prefix = prefix_list[i]
            break
    n_iteration = get_map('num_iterations')
    training_snr = get_map('training_snr')
    snr_lo = training_snr
    snr_hi = training_snr
    snr_info = '/'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    ckpts_dir = '../DL_Training_serial/ckpts/'+prefix+snr_info+str(n_iteration)+'th'+'/'
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)   
    ckpt_nm = 'ldpc-ckpt'  
    restore_model_step = 'latest'
    restore_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_model_info
def set_predict_model(DIA):
    if DIA: 
        nn_type = 'fcn'
    else:
        nn_type ='benchmark'
    decoding_length = get_map('decoding_length')
    n_iteration = get_map('num_iterations')
    #snr_lo = round(get_map('snr_lo'),2)
    #snr_hi = round(get_map('snr_hi'),2)
    threshold_sum = get_map('threshold_sum')
    training_snr = get_map('training_snr')
    snr_lo = training_snr
    snr_hi = training_snr
    snr_info = '/'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    ckpts_dir = '../DL_Training_serial/ckpts/'+nn_type+snr_info+str(n_iteration)+'th/len-'+str(decoding_length)+'-order-'+str(threshold_sum)+'/'
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)   
    ckpt_nm = 'ldpc-ckpt'  
    restore_model_step = 'latest'
    restore_predict_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_predict_model_info  
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
        data_handler_list.append(dataset_test)
    return data_handler_list,snr_list               
                   