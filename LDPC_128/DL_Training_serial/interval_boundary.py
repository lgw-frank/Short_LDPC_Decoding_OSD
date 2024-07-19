# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 22:33:54 2024

@author: lgw
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
#from tensorflow.keras import  metrics
import nn_net as CRNN_DEF
#from collections import Counter,defaultdict,OrderedDict
import numpy as np
import pickle,re
import  os
import ordered_statistics_decoding as OSD_mod

def check_matrix_reorder(original_inputs,inputs,labels):
    code = GL.get_map('code_parameters')
    expanded_H = tf.expand_dims(code.H,axis=0)
    #query the least reliable independent positions
    lri_p = tf.argsort(abs(inputs),axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
    order_original_inputs = tf.gather(original_inputs,lri_p,batch_dims=1)
    order_labels = tf.gather(labels,lri_p,batch_dims=1)
    batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
    tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
    order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
    return order_H_list,order_original_inputs,order_inputs,order_labels

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
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = str(snr_lo)+"-"+str(snr_hi)
    #query decoding path
    nn_type = 'benchmark'
    if DIA: 
        for i,element in enumerate(indicator_list):
            if element == True:
                nn_type = prefix_list[i]
                break
    decoder_type = GL.get_map('selected_decoder_type')
    log_dir = './log/'+decoder_type+'/'+snr_info+'dB/'
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
            

def NN_gen(restore_info,indicator_list):  
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
    [ckpts_dir,ckpt_nm,restore_step] = restore_info  
    if restore_step:
        _,ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
        status = checkpoint.restore(ckpt_f)
        status.expect_partial() 
    return nn

def query_samples(selected_ds,restore_info,residual_path,nn_type,option_tuple):
    code = GL.get_map('code_parameters')
    order_sum = GL.get_map('threshold_sum')
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = '/'+str(snr_lo)+"-"+str(snr_hi)+'dB/'
    decoding_length = GL.get_map('decoding_length')
    print('Selectd decoding path:\n',residual_path)
    (indicator_list,DIA) = option_tuple
    osd_instance = OSD_mod.osd(code)
    teps_list,boundary_error_patterns = generate_teps(osd_instance,residual_path)
    tep_info = (teps_list,boundary_error_patterns)
    #acquire decoding 
    if DIA:
        nn = NN_gen(restore_info,indicator_list) 
    list_length = GL.get_map('num_iterations')+1
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)      
    fail_sum = 0
    correct_sum = 0
    undetect_sum = 0
    # validating the effect by finding the consumed number of TEPs before hitting the authentic EP
    validate_size = 0
    record_list = []
    for i in range(num_counter):
        labels = input_list[i][1][0::list_length]
        if DIA:
            squashed_inputs,original_inputs,_ = nn.preprocessing_inputs(input_list[i])
            updated_inputs = nn(squashed_inputs)
        else:
            original_inputs = input_list[i][0][0::list_length]
            updated_inputs = input_list[i][0][list_length-1::list_length]  
        validate_size += updated_inputs.shape[0]
        #preparing training samples
        records,cf_counter = query_teps_dis(original_inputs,updated_inputs,labels,tep_info) 
        record_list.append(records)
        correct_sum += cf_counter[0]
        fail_sum += cf_counter[1]
        undetect_sum += cf_counter[2]
        if (i+1)%2 == 0: 
            print(correct_sum,fail_sum,undetect_sum)
    #save aquired data in disk
    logdir = './log/'+nn_type+snr_info
    if not os.path.exists(logdir):
        os.makedirs(logdir)    
    nominal_actual_length =  'order-pattern-len'+str(decoding_length)+'-'
    file_name = logdir+nominal_actual_length+nn_type+".pkl"
    saved_summary = (order_sum,validate_size,correct_sum,fail_sum,undetect_sum)
    records_matrix = np.concatenate(record_list,axis=0)
    print(f'Correct:{correct_sum},Failed:{fail_sum},Undetected:{undetect_sum}')
    with open(file_name, "wb") as fh:
        pickle.dump(saved_summary,fh)
        pickle.dump(records_matrix,fh)
    return saved_summary, records_matrix


def reform_inputs(records_matrix): 
    #plot_rows(records_matrix)
    #ground label indicator vector
    suc_class_indicator = tf.where(records_matrix[:,-1:]!=-1,1, 0)
    input_list = records_matrix[:,:-1] 
    min_value = tf.reduce_min(input_list,axis=-1,keepdims=True)
    window_width = GL.get_map('sliding_win_width')
    window_width_pos = window_width+1
    window_list = []
    for i in range(input_list.shape[1]-window_width+1):
        window = input_list[:,i:i+window_width]  # Adjust window position
        window_min = tf.reduce_min(window,axis=-1,keepdims=True) 
        #indicator_vector
        indicator_min = tf.where(min_value==window_min,1,0)
        label_indicator = tf.cast(indicator_min*suc_class_indicator,tf.float32)
        positions = tf.fill([window.shape[0],1],float(i))
        expanded_window = tf.concat([window,positions],axis=1)
        window_tuple = tf.concat((expanded_window,label_indicator),axis=1)
        window_list.append(window_tuple)       
    windows_matrix = tf.reshape(window_list,[-1,window_width_pos+1])
    #sorting the matrix
    #sorting the matrix
    volume_data = tf.sort(windows_matrix[:,:-2],axis=1,direction='ASCENDING')
    volume_inputs = tf.concat([volume_data,windows_matrix[:,-2:-1]],axis=1)
    volume_labels = windows_matrix[:,-1]
    return volume_inputs,volume_labels

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
    acc_size_list = np.insert(np.cumsum(erro_pattern_size_list),0,0)  
    return error_pattern_list,acc_size_list   

def query_teps_dis(original_inputs,updated_inputs,labels,tep_info):
    code = GL.get_map('code_parameters')
    #OSD processing
    #first arrangment of H
    swap_info = check_matrix_reorder(original_inputs,updated_inputs,labels)
    order_H_list,order_original_inputs_list,order_updated_inputs_list,order_labels_list = swap_info
    tep_blocks,acc_block_size = tep_info
    blocks_matrix = tf.concat(tep_blocks,axis=0)
    record_list = []
    success_counter = 0
    fail_counter = 0
    undetect_counter = 0
    actual_size = updated_inputs.shape[0]
    for i in range(actual_size):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        #second arrangement of H
        reduce_H,record_col_index = gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.check_matrix_column))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] = index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
        mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
        updated_index_order = tf.concat([index_order[:(code.check_matrix_column-code.k)],mrb_order],axis=0)
        #third arrangement of H
        final_H2 = tf.gather(reduce_H[:,-code.k:],mrb_swapping_index,axis=1)   
        renewed_original_inputs = tf.gather(order_original_inputs_list[i],updated_index_order)
        renewed_updated_inputs = tf.gather(order_updated_inputs_list[i],updated_index_order)    
        renewed_labels  = tf.cast(tf.gather(order_labels_list[i],updated_index_order),dtype=tf.int32)
        # setting anchoring point    
        renewed_updated_hard = tf.where(renewed_updated_inputs>0,0,1)
        baseline_mrb = tf.reshape(renewed_updated_hard[-code.k:],[1,-1]) 
        #estimations of codeword candidate
        error_pattern_matrix = blocks_matrix
        estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+baseline_mrb)%2)
        estimated_lrb_matrix = tf.matmul(tf.cast(final_H2,tf.int32),estimated_mrb_matrix)%2
        codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
        expand_codeword_candidate_matrix = tf.concat([codeword_candidate_matrix,tf.reshape(renewed_labels,[1,-1])],axis=0)
        order_hard = tf.where(renewed_original_inputs>0,0,1)  
        discrepancy_matrix = tf.cast((expand_codeword_candidate_matrix + order_hard)%2,dtype=tf.float32)
        soft_discrepancy_sum = tf.reduce_sum(discrepancy_matrix*abs(renewed_original_inputs),axis=-1)
        temp_min_index = tf.argmin(soft_discrepancy_sum[:-1])
        locate_phase = -1
        if soft_discrepancy_sum[temp_min_index] < soft_discrepancy_sum[-1]:
            #tf.print(soft_discrepancy_sum[temp_min_index],soft_discrepancy_sum[-1])
            undetect_counter += 1
            continue
        if soft_discrepancy_sum[temp_min_index] == soft_discrepancy_sum[-1]:
            locate_phase = 1
        block_min_sum = [tf.reduce_min(soft_discrepancy_sum[acc_block_size[k]:acc_block_size[k+1]]) for k in range(len(tep_blocks))]
        record_unit = np.append(block_min_sum,locate_phase)
        record_list.append(record_unit)
        success_decoding_indicator = (locate_phase > -1)
        if success_decoding_indicator:
            success_counter += 1
        else:
            fail_counter += 1   
    cf_counter = (success_counter,fail_counter,undetect_counter)
    record_matrix = np.vstack(record_list)
    return record_matrix,cf_counter