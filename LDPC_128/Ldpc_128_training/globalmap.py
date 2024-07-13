"""
Created on Thu Nov 11 23:58:09 2021

@author: Administrator
"""# dictionary operations including adding,deleting or retrieving
import os
import tensorflow as tf
import read_TFdata as Reading
import fill_matrix_info as Fill_matrix

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
    set_map('num_batch_train', int(argv[4]))
    set_map('num_iterations', int(argv[5]))
    set_map('H_filename', argv[6])
    set_map('selected_decoder_type', argv[7])
    
    # the training/testing paramters setting for selected_decoder_type
    set_map('loss_process_indicator', True)
    set_map('ALL_ZEROS_CODEWORD_TRAINING', False)
    set_map('epochs',100)
    set_map('initial_learning_rate', 0.01)
    set_map('decay_rate', 0.95)
    set_map('decay_step', 500)
    set_map('termination_step',1200)
    #filling parity check matrix info
    H_filename = get_map('H_filename')
    code = Fill_matrix.Code(H_filename)
    #store it onto global space
    set_map('code_parameters', code)
    set_map('print_interval',50)
    set_map('record_interval',50)
    
def logistic_setting():
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    ckpts_dir = './ckpts/'+decoder_type+'/'+str(n_iteration)+'th'+'/'
    ckpts_dir_par = './ckpts/'+decoder_type+'/'+str(n_iteration)+'th'+'/par/'
    #create the directory if not existing
    if not os.path.exists(ckpts_dir_par):
        os.makedirs(ckpts_dir_par)  
    ckpt_nm = 'ldpc-ckpt'  
    restore_step = 'latest'
    restore_info = [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step]
    return restore_info

def training_setting():  
    #training setting
    multiplier = 1  #combine several batches into one big batch to guarantee stability of gradient search
    #training initials
    start_step = 0
    train_steps = get_map('num_batch_train')*get_map('epochs')   
    start_info = [start_step,multiplier,train_steps]
    return start_info
 
def data_setting(code,unit_batch_size):
    #training data directory
    code_length = code.check_matrix_column
    snr_lo = round(get_map('snr_lo'),2)
    snr_hi = round(get_map('snr_hi'),2)
    n_iteration = get_map('num_iterations')
    data_dir = '../Training_data_gen_'+str(code_length)+'/data/snr'+str(snr_lo)+'-'+str(snr_hi)+'dB/'
    # reading in training/validating data;make dataset iterator
    if get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'ldpc-train-allzero.tfrecord'
    else:
        file_name = 'ldpc-train-nonzero.tfrecord'
    file_dir = data_dir+file_name
    dataset_train = Reading.data_handler(code_length,file_dir,unit_batch_size)
    #preparing batch iterator of data file
    #dataset_train = dataset_train.take(50)
    selected_ds = dataset_train.cache()
    decoder_type = get_map('selected_decoder_type')
    gen_data_dir = data_dir + str(n_iteration)+'th/'+decoder_type+'/'
    if not os.path.exists(gen_data_dir):
      os.makedirs(gen_data_dir)
    return gen_data_dir,selected_ds

def optimizer_setting():
    #optimizing settings
    decay_rate = get_map('decay_rate')
    initial_learning_rate = get_map('initial_learning_rate')
    decay_steps = get_map('decay_step')
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps, decay_rate,staircase=True)
    return exponential_decay

def log_setting(restore_info,checkpoint):
    n_iteration = get_map('num_iterations')
    decoder_type = get_map('selected_decoder_type')
    (ckpts_dir,ckpt_nm,_,_) = restore_info
    # summary recorder
    summary_writer = tf.summary.create_file_writer('./tensorboard/'+str(decoder_type)+'/'+str(n_iteration)+'th'+'/')     # the parameter is the log folder we created
    # tf.summary.trace_on(graph=True, profiler=True)  # Open Trace option, then the dataflow graph and profiling information can be recorded
    #sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    manager_current = tf.train.CheckpointManager(checkpoint, directory=ckpts_dir, checkpoint_name=ckpt_nm, max_to_keep=5)
    logger_info = (summary_writer,manager_current)
    return logger_info