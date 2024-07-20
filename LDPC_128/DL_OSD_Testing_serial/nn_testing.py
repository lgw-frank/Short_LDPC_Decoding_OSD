# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
import nn_net as CRNN_DEF
import ordered_statistics_decoding as OSD_mod
import re
import pickle
import os
from collections import Counter,defaultdict,OrderedDict
import numpy as  np
import ast
import time

# import numpy as np
# import ms_decoder_dense as MDL
#import ordered_statistics_decoding as OSD_Module
#import ordered_statistics_decoding as OSD_Module

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return ckpt_f    

def NN_gen(restore_list,indicator_list):  
    sliding_win_width = GL.get_map('sliding_win_width')
    cnn = CRNN_DEF.conv_bitwise()
    rnn1 = CRNN_DEF.rnn_one()
    rnn2 = CRNN_DEF.rnn_two()
    nn_list = [cnn,rnn1,rnn2] 
    for i,element in enumerate(indicator_list):
        if element == True:
            nn = nn_list[i]
            break
    checkpoint = tf.train.Checkpoint(myAwesomeModel=nn)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,restore_step] = restore_list[0]  
    if restore_step:
        ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial() 
        
    fcn = CRNN_DEF.Predict_outlier_light(sliding_win_width)
    checkpoint = tf.train.Checkpoint(myAwesomeModel=fcn)
    #unpack related info for restoraging
    [ckpts_dir,ckpt_nm,restore_step] = restore_list[1]  
    if restore_step:
        ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial()     
    return nn,fcn

def query_convention_path(indicator_list,prefix_list,DIA):
    nn_type = 'benchmark'
    order_sum = GL.get_map('threshold_sum')+1
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    decoding_path = []
    for i in range(order_sum):
        for j1 in range(order_sum):
            for j2 in range(order_sum):                
                for j3 in range(order_sum):
                    if j1+j2+j3<=i:
                        decoding_path.append([j1,j2,j3])
    decoding_path,_ = tf.raw_ops.UniqueV2(x=decoding_path,axis=[0])
    return decoding_path,nn_type
#fixed scheduling for decoding path    
def query_decoding_path(indicator_list,prefix_list,DIA):
    snr_lo = 2.7
    snr_hi = 2.7
    snr_info = str(snr_lo)+"-"+str(snr_hi)
    #query decoding path
    nn_type = 'benchmark'
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    decoder_type = GL.get_map('selected_decoder_type')
    log_dir = '../DL_Training_serial/log/'+decoder_type+'/'+snr_info+'dB/'
    file_name = log_dir+"dist-error-pattern-"+nn_type+".pkl"
    with open(file_name, "rb") as fh:
        _ = pickle.load(fh)
        _ = pickle.load(fh)    
        _ = pickle.load(fh)
        _ = pickle.load(fh)    
        _ = pickle.load(fh)
        pattern_dict = pickle.load(fh)
    # Sort the dictionary keys based on their values in descending order
    string_pattern = sorted(pattern_dict, key=pattern_dict.get, reverse=True)
    #turn into digits-type list
    decoding_path = []
    for element in string_pattern:
        distilled_digits = re.findall(r"\w+",element)
        num_group = [int(element) for element in distilled_digits]
        decoding_path.append(num_group)
    residual_path = filter_order_patterns(decoding_path)  
    return residual_path,nn_type  

def filter_order_patterns(decoding_path):
    residual_path = [] 
    threshold_sum = GL.get_map('threshold_sum')
    nomial_path_length = GL.get_map('decoding_length')  
    for order_pattern in decoding_path:
        if sum(order_pattern) <= threshold_sum:
            residual_path.append(order_pattern) 
    return residual_path[:nomial_path_length]
      
def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss

def calculate_list_cross_entropy_ber(input_list,labels):
    cross_entropy_list = []
    ber_list = []
    for i in range(len(input_list)):
        cross_entropy_element = calculate_loss(input_list[i],labels).numpy()
        cross_entropy_list.append(cross_entropy_element)
        current_hard_decision = tf.where(input_list[i]>0,0,1)
        compare_result = tf.where(current_hard_decision!=labels,1,0)
        num_errors = tf.reduce_sum(compare_result)
        ber_list.append(num_errors)
    return cross_entropy_list,ber_list

#fixed scheduling for decoding path    
def generate_teps(osd,residual_path):
    num_segments = GL.get_map('segment_num')
    #segmentation of MRB
    _,boundary_MRB = GL.secure_segment_threshold()
    range_list = [range(boundary_MRB[i],boundary_MRB[i+1]) for i in range(num_segments)]
    #generate all possible error patterns of mrb
    error_pattern_list = []
    erro_pattern_size_list = []
    for j in range(len(residual_path)):
        element = osd.error_pattern_gen(residual_path[j],range_list)
        error_pattern_list.append(element)
        erro_pattern_size_list.append(element.shape[0])
    acc_block_size = np.insert(np.cumsum(erro_pattern_size_list),0,0)  
    return error_pattern_list,acc_block_size       

def Testing_OSD(snr,selected_ds,restore_list,indicator_list,prefix_list,DIA):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    order_sum = GL.get_map('threshold_sum')
    Convention_path_indicator = GL.get_map('convention_path')
    soft_margin = GL.get_map('soft_margin')
    osd_instance = OSD_mod.osd(code)
    if Convention_path_indicator:
        residual_path,nn_type = query_convention_path(indicator_list,prefix_list,DIA)
    else:    
        residual_path,nn_type = query_decoding_path(indicator_list,prefix_list,DIA)
    
    print(f'Actual decoding path:{len(residual_path)}',residual_path)
    
    teps_list,acc_block_size = generate_teps(osd_instance,residual_path)
    tep_info = (teps_list,acc_block_size)
    #acquire decoding 
    if DIA:
        nn,fcn = NN_gen(restore_list,indicator_list) 
    logdir = './log/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_filename = logdir+'OSD-'+str(order_sum)+'-'+nn_type+'.txt'  
    list_length = GL.get_map('num_iterations')+1
    
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)      
    
    fail_sum = 0
    correct_sum = 0
    windows_sum = 0
    complexity_sum = 0
    actual_size = 0
    cross_entropy_list_sum = [0.]*(list_length+1)
    ber_list_sum = [0]*(list_length+1)
    
    for i in range(num_counter):   
        if DIA:
            squashed_inputs,inputs,labels = nn.preprocessing_inputs(input_list[i])
            new_inputs = nn(squashed_inputs)
            #nn.print_model()
        else:
            labels = input_list[i][1][0::list_length]
            new_inputs = input_list[i][0][0::list_length]
        actual_size += labels.shape[0]
        
        input_data_list = [input_list[i][0][j::list_length] for j in range(list_length)]
        input_data_list.append(new_inputs)
        cross_entropy_list,ber_list = calculate_list_cross_entropy_ber(input_data_list,labels)
        # Element-wise addition using a loop
        cross_entropy_list_sum = [a + b for a, b in zip(cross_entropy_list, cross_entropy_list_sum)] 
        ber_list_sum = [a + b for a, b in zip(ber_list, ber_list_sum)]
        # Alternatively, using map and lambda function
        # result = list(map(lambda x, y: x + y, list1, list2))
        #OSD processing
        correct_counter,fail_counter,windows_size,complexity_size = osd_instance.sliding_osd(fcn,input_list[i][0],new_inputs,labels,tep_info)
        correct_sum += correct_counter
        fail_sum += fail_counter  
        windows_sum += windows_size
        complexity_sum += complexity_size
        if (i+1)%10 == 0:
            average_size = round(complexity_sum/actual_size,4)
            wins_size = round(windows_sum/actual_size,4)
            print(f'\nFor {snr:.1f}dB order_sum:{order_sum} len:{len(residual_path)} soft_margin:{soft_margin}:')
            print(f'Selected actual path:{residual_path}')
            print(f'--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}')      
            average_loss_list  = [cross_entropy_list_sum[j]/actual_size for j in range(list_length+1)] 
            average_ber_list  = [ber_list_sum[j]/(actual_size*code.check_matrix_column) for j in range(list_length+1)]
            formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
            formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
            print(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}')
            T2 =time.process_time()
            print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
        if i == num_counter-1 or fail_sum >= GL.get_map('termination_threshold'):
            break
    T2 =time.process_time()
    FER = round(fail_sum/actual_size,5)  
    average_size = round(complexity_sum/actual_size,4)
    wins_size = round(windows_sum/actual_size,4)
    average_loss_list  = [cross_entropy_list_sum[j]/actual_size for j in range(list_length+1)]  
    average_ber_list  = [ber_list_sum[j]/(actual_size*code.check_matrix_column) for j in range(list_length+1)]  
    print('\nFor %.1fdB (order_sum:%d) '%(snr,order_sum)+nn_type+':\n')
    print('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')          
    print(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}')
    formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
    formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
    print(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}')
    print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
    with open(log_filename,'a+') as f:
        f.write(f'For {snr:.1f}dB order_sum:{order_sum} len:{len(residual_path)} soft_margin:{soft_margin}:\n')
        f.write(f'Selected actual path:{residual_path}\n')
        f.write('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')         
        f.write(f'FER:{FER}--> S/F:{correct_sum} /{fail_sum} Avr TEPs:{average_size} Wins:{wins_size}\n')
        formatted_floats_ce = [" ".join(["{:.3f}".format(value) for value in average_loss_list])]
        formatted_floats_ber = [" ".join(["{:.3f}".format(value) for value in average_ber_list])]
        f.write(f'avr CE per itr:\n{formatted_floats_ce} \nBER:{formatted_floats_ber}\n')
        f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!\n')
    return FER,log_filename

  
def evaluate_MRB_bit(updated_inputs,labels):
    inputs_abs = tf.abs(updated_inputs)
    code = GL.get_map('code_parameters')
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(updated_inputs,order_index,batch_dims=1)
    order_inputs_hard = tf.where(order_inputs>0,0,1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32)
    cmp_result = tf.reduce_sum(tf.where(order_inputs_hard[:,-code.k:] == order_labels[:,-code.k:],0,1),axis=-1).numpy()
    Demo_result=Counter(cmp_result) 
    #print(Demo_result)
    return Demo_result
def dic_union(dicA,dicB):
    for key,value in dicB.items():
        if key in dicA:
            dicA[key] += value
        else:
            dicA[key] = value
    return dict(sorted(dicA.items(), key=lambda d:d[0])) 
    

def stat_pro_osd(inputs,labels):
    #initialize of mask
    code = GL.get_map('code_parameters')
    order_H_list,order_inputs,order_labels = check_matrix_reorder(inputs,labels)
    updated_MRB_list = []
    swap_len_list = []
    for i in range(inputs.shape[0]):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        M,record_col_index = code.gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.check_matrix_column))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] = index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        #print(Updated_MRB)
        updated_MRB_list.append(updated_MRB) 
        swap_indicator = np.where(updated_MRB>=code.check_matrix_column-code.k,0,1)
        swap_sum = sum(swap_indicator)
        swap_len_list.append(swap_sum)
    return updated_MRB_list,swap_len_list 

def check_matrix_reorder(inputs,labels):
    code = GL.get_map('code_parameters')
    expanded_H = tf.expand_dims(code.H,axis=0)
    #query the least reliable independent positions
    lri_p = tf.argsort(abs(inputs),axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
    order_labels = tf.gather(labels,lri_p,batch_dims=1)
    batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
    tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
    order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
    return order_H_list,order_inputs,order_labels

def gf2elim(M):
      m,n = M.shape
      i=0
      j=0
      record_col_exchange_index = []
      while i < m and j < n:
          #print(M)
          # find value and index of largest element in remainder of column j
          if np.max(M[i:, j]):
              k = np.argmax(M[i:, j]) +i
        # swap rows
              #M[[k, i]] = M[[i, k]] this doesn't work with numba
              if k !=i:
                  temp = np.copy(M[k])
                  M[k] = M[i]
                  M[i] = temp              
          else:
              if not np.max(M[i, j:]):
                  M = np.delete(M,i,axis=0) #delete a all-zero row which is redundant
                  m = m-1  #update according info
                  continue
              else:
                  column_k = np.argmax(M[i, j:]) +j
                  temp = np.copy(M[:,column_k])
                  M[:,column_k] = M[:,j]
                  M[:,j] = temp
                  record_col_exchange_index.append((j,column_k))
      
          aijn = M[i, j:]
          col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
          col[i] = 0 #avoid xoring pivot row with itself
          flip = np.outer(col, aijn)
          M[:, j:] = M[:, j:] ^ flip
          i += 1
          j +=1
      return M,record_col_exchange_index 
  
