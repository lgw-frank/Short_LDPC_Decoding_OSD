# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 21:29:58 2022

@author: lgw
"""
import numpy as np
np.set_printoptions(precision=3)
#import matplotlib
import tensorflow  as tf
import globalmap as GL
import ms_decoder_dense as Decoder_module
def training_stage(restore_info):
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size)

    start_info = GL.training_setting()
    exponential_decay = GL.optimizer_setting()
    #instance of Model creation   
    Model = Decoder_module.Decoding_model()
    optimizer = tf.keras.optimizers.legacy.Adam(exponential_decay)
    # save restoring info
    checkpoint = tf.train.Checkpoint(myAwesomeModel=Model, myAwesomeOptimizer=optimizer)
    logger_info = GL.log_setting(restore_info,checkpoint)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step] = restore_info  
    if restore_step:
        start_step,ckpt_f = Decoder_module.retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()
        start_info[0] = start_step
        #model = Model.print_model()
        #print_flops(model)
        #Model.obtain_paras()
    if GL.get_map('loss_process_indicator'):
        Model = Decoder_module.training_block(start_info,Model,optimizer,\
                    exponential_decay,iterator,logger_info,restore_info)
    return Model

def post_process_input(Model):
    unit_batch_size = GL.get_map('unit_batch_size')
    code = GL.get_map('code_parameters')
    data_dir,iterator = GL.data_setting(code,unit_batch_size)
    #acquiring erroneous cases with necessary modification or perturbation
    GL.set_map('loss_process_indicator', False)
    buffer_list = Decoder_module.postprocess_training(Model,iterator)
    if GL.get_map('ALL_ZEROS_CODEWORD_TRAINING'):
        file_name = 'ldpc-allzero-retrain.tfrecord'
    else:
        file_name = 'ldpc-nonzero-retrain.tfrecord'
    retrain_file_dir = data_dir+file_name
    Decoder_module.save_decoded_data(buffer_list[0],buffer_list[1],retrain_file_dir)
    print("Collecting targeted cases of decoding is finished!")


