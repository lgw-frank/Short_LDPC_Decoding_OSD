# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 12:46:41 2024

@author: zidonghua_30
"""
import globalmap as GL
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
import nn_net as CRNN_DEF
#from collections import Counter,defaultdict,OrderedDict
#import numpy as np
import math
import pickle
import  os
class CustomCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_categorical_accuracy', **kwargs):
        super(CustomCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.cat_acc_metric = tf.metrics.CategoricalAccuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.cat_acc_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.cat_acc_metric.result()

    def reset_states(self):
        self.cat_acc_metric.reset_states()
        
def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) 
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f

def collect_inputs(file_name): 
    with open(file_name, "rb") as fh:
        _ = pickle.load(fh)
        _ = pickle.load(fh)
        _ = pickle.load(fh)
        inteval_info_list = pickle.load(fh) 
    input_list = []
    label_list = []
    for  i in range(len(inteval_info_list)):
        batch_input = inteval_info_list[i]
        for element in batch_input:     
            input_minmax = list(element[0])
            label = element[1]
            restructured_input = tf.reshape(tf.concat(input_minmax,axis=-1)[1:],[1,-1])
            input_list.append(restructured_input)
            label_list.append(label)
    volume_inputs = tf.concat(input_list,axis=0)
    original_labels = tf.concat(label_list,axis=0)
    # acquire weights for imbalanced dataset 
    total_samples = len(original_labels)
    limit_label_index = tf.reduce_max(original_labels)+1
    volume_labels = tf.where(original_labels>=0,original_labels,limit_label_index)
    num_classes = limit_label_index+1 # another class for failure cases
    one_hot_matrix = tf.keras.utils.to_categorical(volume_labels, num_classes)
    count_class = tf.reduce_sum(one_hot_matrix,axis=0)
    class_weight = (total_samples) / (num_classes * count_class)
    weight_samples = tf.reduce_sum(one_hot_matrix*class_weight,axis=-1)
    return volume_inputs,weight_samples,volume_labels,num_classes
        
def predict_main(restore_predict_info,DIA):
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = str(snr_lo)+"-"+str(snr_hi)
    decoding_length = GL.get_map('decoding_length')
    batch_size = GL.get_map('unit_batch_size')
    if DIA: 
        nn_bit_wise_type = 'model_cnn'
    else:
        nn_bit_wise_type ='benchmark'
    data_file_dir = './log/'+nn_bit_wise_type+'/'+snr_info+'dB/'+"order-pattern-mimmax-len("+str(decoding_length)+')-'+nn_bit_wise_type+".pkl"
    if not os.path.exists(data_file_dir):
        print('Directiory not found, please check again!')
        exit(-1)
    volume_inputs,weight_samples,volume_labels,num_classes = collect_inputs(data_file_dir)
    num_counter = math.ceil(len(volume_labels)/batch_size)
    nn_predic_type = 'baseline'
    if DIA:
        epochs = GL.get_map('epochs') 
        nn_predic_type = 'fcn'
        fcn = CRNN_DEF.predict_phase(decoding_length,num_classes)  #considering the rightmost and failure together.
        step = 0
        exponential_decay = GL.optimizer_setting()
        optimizer = tf.keras.optimizers.Adam(exponential_decay)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=fcn, myAwesomeOptimizer=optimizer)
        summary_writer,manager_current = GL.log_setting(restore_predict_info,checkpoint,nn_predic_type)
        # Instantiate the metric
        custom_accuracy = CustomCategoricalAccuracy()
        cce = CategoricalCrossentropy()
        #unpack related info for restoraging
        [ckpts_dir,ckpt_nm,restore_step] = restore_predict_info  
        if restore_step:
            print('Load the previous saved model from disk!')
            step,ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
            status = checkpoint.restore(ckpt_f)
            status.expect_partial()  
        #initialize starting point
        start_epoch = step//num_counter
        residual = step%num_counter
        jump_loop = False
        if GL.get_map('nn_train'):  
            if step < GL.get_map('termination_prediction_step'):             
                for epoch in range(start_epoch,epochs):
                    print("\nStart of epoch %d:" % epoch) 
                    custom_accuracy.reset_states()
                    for i in range(residual,num_counter):
                        step = step + 1
                        inputs = volume_inputs[i*batch_size:(i+1)*batch_size]
                        batch_weight = weight_samples[i*batch_size:(i+1)*batch_size]
                        labels = volume_labels[i*batch_size:(i+1)*batch_size]
                        with tf.GradientTape() as tape:
                            outputs = fcn(inputs)
                            loss = cce(y_true=labels,y_pred=outputs,sample_weight=batch_weight)
                            custom_accuracy.update_state(y_true=labels,y_pred=outputs,sample_weight=batch_weight)
                        total_variables = fcn.trainable_variables         
                        grads = tape.gradient(loss,total_variables)
                        grads_and_vars=zip(grads, total_variables)
                        capped_gradients = [(tf.clip_by_norm(grad,5e2), var) for grad, var in grads_and_vars if grad is not None]
                        optimizer.apply_gradients(capped_gradients)   
                        if step % print_interval == 0:   
                            total_metric_value = custom_accuracy.result()
                            print('Step:%d  Loss:%.3f Precision:%.3f'%(step,loss.numpy(),total_metric_value))
                            #_ = evaluate_MRB_bit(inputs,labels)
                            #_ = evaluate_MRB_bit(refined_inputs,labels)                                                               
                        if step % record_interval == 0:
                            manager_current.save(checkpoint_number=step)                   
                        if step >= GL.get_map('termination_step'):
                            jump_loop = True
                            break
                    residual = 0
                    if jump_loop:
                        break
                #save the latest setting
                manager_current.save(checkpoint_number=step) 
    #verifying trained pars from start to end    
    success_sum = 0
    fail_sum = 0
    actual_size = 0
    for i in range(num_counter):
        inputs = volume_inputs[i*batch_size:(i+1)*batch_size]
        labels = volume_labels[i*batch_size:(i+1)*batch_size]
        batch_count = len(labels)
        actual_size += batch_count
        outputs = fcn(inputs)
        hard_prediction_index = tf.cast(tf.argmax(outputs,axis=-1),tf.int32)
        success_counter = tf.reduce_sum(tf.where(labels==hard_prediction_index,1,0))
        fail_counter = tf.reduce_sum(tf.where(labels==hard_prediction_index,0,1))
        success_sum += success_counter
        fail_sum += fail_counter
        #print(success_sum.numpy(),fail_sum.numpy())
        if (i+1)%10 == 0:
            print(f'S:{success_sum} F:{fail_sum}') 

    success_rate = success_sum/actual_size
    fail_rate = fail_sum/actual_size
    print('Summary of validation:')
    print(f'Total counts:{actual_size}')
    print(f'S:{success_sum} F:{fail_sum}') 
    print(f'Success rate:{success_rate:.4f} Fail rate:{fail_rate:.4f}')
    #save on disk files
    log_dir = './log/'+nn_predic_type+'/'+snr_info+'dB/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    with open(log_dir+'length-'+str(decoding_length)+'prediction-ratio.txt', "a+") as f:
        f.write('\nFor Summery of '+snr_info+'dB:\n')
        f.write(f'S:{success_sum} F:{fail_sum}\n')   
        f.write(f'Success rate:{success_rate:.4f} Fail rate:{fail_rate:.4f}')
