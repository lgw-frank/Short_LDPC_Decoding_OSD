"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import tensorflow as tf
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
    set_map('unit_batch_size', int(argv[3]))
    set_map('num_iterations', int(argv[4]))
    set_map('H_filename', argv[5])
    set_map('selected_decoder_type',argv[6])  
    
    # the training/testing paramters setting when selected_decoder_type= Combination of VD/VS/SL,HD/HS/SL
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)   
    set_map('epochs',100)    
    set_map('initial_learning_rate', 0.001)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)  
    set_map('termination_step',2000)
    set_map('termination_prediction_step',2000)
    
    set_map('threshold_sum',3)     #threshold for sum of number of non-zero elements across sections
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    set_map('convention_path',False)    
    
    #store it onto global space
    set_map('code_parameters', code)
    set_map('print_interval',100)
    set_map('record_interval',100) 
    
    set_map('nn_train',True)
    set_map('segment_num',6)
    set_map('reliability_enhance',False)
    set_map('prediction_order_pattern',True)    
    set_map('regulation_weight',10.) 
    set_map('decoding_length',30) 
    set_map('regnerate_training_samples',True)
    set_map('sliding_win_width',5)  #the setting can be optimized to further improve FER
    
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
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    # training_snr = get_map('training_snr')
    # snr_lo = training_snr
    # snr_hi = training_snr
    snr_info = '/'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    ckpts_dir = './ckpts/'+prefix+snr_info+str(n_iteration)+'th/'
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
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    threshold_sum = get_map('threshold_sum')
    # training_snr = get_map('training_snr')
    # snr_lo = training_snr
    # snr_hi = training_snr
    snr_info = '/'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    ckpts_dir = './ckpts/'+nn_type+snr_info+str(n_iteration)+'th/len-'+str(decoding_length)+'-order-'+str(threshold_sum)+'/'
    #create the directory if not existing
    if not os.path.exists(ckpts_dir):
        os.makedirs(ckpts_dir)   
    ckpt_nm = 'ldpc-ckpt'  
    restore_model_step = ''
    restore_predict_model_info = [ckpts_dir,ckpt_nm,restore_model_step]
    return restore_predict_model_info  

def save_model_dir(prefix):
    n_iteration = get_map('num_iterations')
    ckpts_dir = './ckpts/'+prefix+'/'+str(n_iteration)+'th'+'/'
    # checkpoint
    checkpoint_prefix = os.path.join(ckpts_dir,'ckpt')
    if not os.path.exists(os.path.dirname(checkpoint_prefix)):
        os.makedirs(os.path.dirname(checkpoint_prefix))
    return checkpoint_prefix

def data_setting():
    list_length =   get_map('num_iterations') + 1
    batch_size = get_map('unit_batch_size')*list_length
    code = get_map('code_parameters')
    code_length = code.check_matrix_column
    #training data directory
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    data_dir = '../Training_data_gen_'+str(code_length)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'+str(n_iteration)+'th'+'/'+decoder_type+'/'
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'ldpc-allzero-retrain.tfrecord'
    else:
        file_name = 'ldpc-nonzero-retrain.tfrecord' 
    file_dir = data_dir+file_name
    dataset_train = Reading.data_handler(code.check_matrix_column,file_dir,batch_size)
    #dataset_train = dataset_train.take(4)
    selected_ds = dataset_train.cache()
    return selected_ds                   
                    
def optimizer_setting():
    #optimizing settings
    decay_rate = get_map('decay_rate')
    initial_learning_rate = get_map('initial_learning_rate')
    decay_steps = get_map('decay_step')
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,staircase=True)
    return exponential_decay

def log_setting(restore_info,checkpoint,prefix):
    (ckpts_dir,ckpt_nm,_) = restore_info
    n_iteration = get_map('num_iterations')
    # summary recorder
    # Create the log directory
    log_dir = './tensorboard/'+prefix+'/'+str(n_iteration)+'th'+'/'
    os.makedirs(log_dir, exist_ok=True)
    # Set up TensorBoard writer
    summary_writer = tf.summary.create_file_writer(log_dir)     # the parameter is the log folder we created
    manager_current = tf.train.CheckpointManager(checkpoint, directory=ckpts_dir, checkpoint_name=ckpt_nm, max_to_keep=5)
    logger_info = (summary_writer,manager_current)
    return logger_info