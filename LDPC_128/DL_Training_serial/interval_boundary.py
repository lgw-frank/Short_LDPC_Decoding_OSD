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
from itertools import combinations
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
    num_blocks = GL.get_map('num_blocks')
    tep_blocks,acc_block_size = partition_counter(pattern_dict, num_blocks) 
    return tep_blocks,acc_block_size,nn_type  

def string2digits(source_str):
    distilled_digits = re.findall(r"\w+",source_str)
    num_group = [int(element) for element in distilled_digits]
    return num_group

def remove_duplicate_rows(binary_matrix):
    seen = set()
    unique_rows = []
    for row in binary_matrix:
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)
    unique_matrix = np.array(unique_rows)
    return unique_matrix

def merge_matrix(matrix1, matrix2):
    # Find distinct rows in matrix2 that are not in matrix1
    submatrix1 = matrix1[:,:-1]
    derived_matrix = np.array([row for row in matrix2 if not np.any(np.all(submatrix1 == row, axis=1))])
    unique_new_matrix = remove_duplicate_rows(derived_matrix)
    filled_ones = tf.ones([unique_new_matrix.shape[0],1],dtype=tf.int32)
    extended_matrix2 = tf.concat([unique_new_matrix,filled_ones],axis=1)
    # Append the unique rows to matrix1
    result_matrix = np.vstack((matrix1, extended_matrix2))
    return result_matrix

    
def filter_and_sort_counter(counter, threshold):
    code = GL.get_map('code_parameters')
    # Sort the filtered items by count in descending order (you can change this to ascending if needed)
    sorted_items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    pattern_list = [string2digits(sorted_items[i][0]) for i in range(len(sorted_items))]
    pattern_matrix = tf.reshape(pattern_list,[-1,code.k])
    count_list = [int(sorted_items[i][1]) for i in range(len(sorted_items))]
    count_vector = tf.reshape(count_list,[-1,1])
    # Calculate the sum of elements in each row
    row_sums = np.sum(pattern_matrix, axis=1)
    # Filter out rows where the sum is more than 2
    filtered_matrix = pattern_matrix[row_sums <= threshold]
    filtered_count_vec = count_vector[row_sums <= threshold]
    pattern_info_matrix = tf.concat([filtered_matrix,filtered_count_vec],axis=-1)
    return pattern_info_matrix

def left_row_shift(row, nth_bit):
    #number of non-zero elements 
    num_non_zero = np.sum(row)
    process_indicator = False
    if nth_bit <= num_non_zero:       
        # Find indices of all nonzero elements
        nonzero_indices = np.where(row != 0)[0]   
        if len(set(range(nonzero_indices[nth_bit-1]+1))-set(nonzero_indices[:nth_bit]))!=0:
            location_index = nth_bit-1
            nth_index = nonzero_indices[location_index]
            location_pos = nth_index
            #push to the leftmost side position
            while True: 
                location_pos -= 1
                if row[location_pos] == 0:
                    row[location_pos] = 1
                    row[nth_index] = 0
                    nth_index = location_pos
                    process_indicator = True 
                if location_pos == 0:
                    break
    return process_indicator, row
        
def left_shift_matrix(original_matrix,nth_bit):
    matrix = np.copy(original_matrix)
    new_row_list = []
    for row_idx in range(matrix.shape[0]):
        current_row = matrix[row_idx][:-1]
        process_indicator, row = left_row_shift(current_row,nth_bit)
        if process_indicator:
            new_row_list.append(row)
    new_matrix = np.reshape(new_row_list,[-1,matrix.shape[1]-1])
    return new_matrix

def extending_patterns(pattern_info_matrix,nth_bit):
    #complement all other nearest neighbors within given order-p to control complexity
    dilated_matrix = left_shift_matrix(pattern_info_matrix,nth_bit)
    residual_path_matrix = merge_matrix(pattern_info_matrix,dilated_matrix)
    return residual_path_matrix
    

def partition_counter(error_patterns_counter, num_blocks): 
    num_blocks = GL.get_map('num_blocks')
    threshold_sum = GL.get_map('threshold_sum')
    
    # Sort the Counter items by count in descending order
    blocks_list = [] 
    acc_sum = 0 
    block = []   
    pattern_info_matrix = filter_and_sort_counter(error_patterns_counter,threshold_sum)
    if GL.get_map('extending_tep'):
        nth_bit = GL.get_map('nth_bit')
        extended_pattern_matrix = extending_patterns(pattern_info_matrix,nth_bit)
    else:
        extended_pattern_matrix = pattern_info_matrix
    sorted_pattern_matrix = reorder_matrix_by_index_sum_tf(extended_pattern_matrix)
    counts_sum = sum(sorted_pattern_matrix[:,-1])
    for i in range(sorted_pattern_matrix.shape[0]):  
        block_sum = sorted_pattern_matrix[i][-1]
        if (acc_sum+block_sum)*num_blocks > counts_sum:
            block.append(sorted_pattern_matrix[i][:-1])
            blocks_list.append(block)
            counts_sum -= (acc_sum+block_sum)
            acc_sum = 0
            num_blocks -= 1
            block = []
        else:
            acc_sum += block_sum
            block.append(sorted_pattern_matrix[i][:-1]) 
    if len(block):
        blocks_list.append(block)        
    block_sums = [len(blocks_list[i]) for i in range(len(blocks_list))]
    acc_block_size = np.insert(np.cumsum(block_sums),0,0)
    tep_blocks = [sorted_pattern_matrix[acc_block_size[j]:acc_block_size[j+1]][:,:-1] for j in range(len(block_sums))]
    return tep_blocks,acc_block_size

def reorder_matrix_by_index_sum_tf(pattern_info_matrix):
    matrix = pattern_info_matrix[:,:-1]
    occurrence_vec = pattern_info_matrix[:,-1:]
    # Create a range tensor for column indices
    col_indices = tf.range(matrix.shape[1])
    # Expand dims for broadcasting and compute the index sum for each row
    index_sums = tf.reduce_sum(matrix * col_indices-int(1e3)*occurrence_vec, axis=1)
    # Get the sorted indices based on the index sums
    sorted_indices = tf.argsort(index_sums, axis=0, direction='ASCENDING')
    # Gather the rows based on the sorted indices
    sorted_matrix = tf.gather(pattern_info_matrix, sorted_indices)
    return sorted_matrix


def binary_sequences_within_hamming_weight1(length, max_hamming_weight):
    all_sequences = []
    
    for hamming_weight in range(max_hamming_weight + 1):
        positions = list(combinations(range(length), hamming_weight))
        num_sequences = len(positions)
        
        sequences = np.zeros((num_sequences, length), dtype=int)
        
        for i, pos in enumerate(positions):
            sequences[i, list(pos)] = 1
        
        all_sequences.extend([''.join(map(str, seq)) for seq in sequences])
    
    return all_sequences

def binary_sequences_within_hamming_weight(length, max_hamming_weight):
    def next_combination(x):
        u = x & -x
        v = u + x
        return v + (((v ^ x) // u) >> 2)
    
    all_sequences = []
    
    for hamming_weight in range(max_hamming_weight + 1):
        if hamming_weight == 0:
            # Add the all-zeros sequence
            all_sequences.append([0] * length)
            continue
        if hamming_weight > length:
            continue
        
        start = (1 << hamming_weight) - 1
        end = start << (length - hamming_weight)
        
        x = start
        while x <= end:
            binary_str = bin(x)[2:].zfill(length)
            all_sequences.append([int(bit) for bit in binary_str])
            x = next_combination(x)   
    # Convert the list of lists to a NumPy array
    TEP_tensor = np.array(all_sequences, dtype=int) 
    return TEP_tensor

def hamming_weight(row):
    """Compute the Hamming weight (number of 1s) of a binary row."""
    return np.sum(row)

def index_sum(row):
    """Calculate the sum of indices of non-zero bits, starting from the leftmost bit (index 0)."""
    return np.sum(np.nonzero(row)[0]) if np.any(row) else 0

def subtract_and_sort_matrices(A, B):
    # Convert rows to sets of tuples for fast set operations
    set_A = {tuple(row) for row in A}
    set_B = {tuple(row) for row in B}
    
    # Subtract sets to get rows in B but not in A
    set_diff = set_B - set_A
    
    # Convert the set difference back to a numpy array
    diff_matrix = np.array([list(row) for row in set_diff])
    
    # Sort the resulting matrix first by Hamming weight, then by index sum of non-zero bits
    sorted_matrix = sorted(diff_matrix, key=lambda row: (hamming_weight(row), index_sum(row)))  
    return np.array(sorted_matrix)


def filter_order_patterns(decoding_path):
    residual_path = []
    code = GL.get_map('code_parameters') 
    threshold_sum = GL.get_map('threshold_sum')
    nomial_path_length = GL.get_map('decoding_length')
    for order_pattern in decoding_path:
        if sum(order_pattern) <= threshold_sum:
           residual_path.append(order_pattern) 
    #for one case one order pattern, complement all other quailifed TEPs within given order-p
    stacked_path = np.stack(residual_path)
    if len(residual_path) < nomial_path_length:
        sequences = binary_sequences_within_hamming_weight(code.k,threshold_sum)
        updated_residual_path_matrix = subtract_and_sort_matrices(stacked_path,sequences)
        #reconstruct residual path
        residual_path_matrix = np.vstack((stacked_path,updated_residual_path_matrix))
        decoding_path = residual_path_matrix[:nomial_path_length]
    else:
        decoding_path = stacked_path[:nomial_path_length]
    return  decoding_path
            

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

#fixed scheduling for decoding path    
def generate_teps(osd,residual_path):
    num_segments = GL.get_map('segment_num')
    #segmentation of MRB
    _,boundary_MRB= GL.secure_segment_threshold()
    range_list = [range(boundary_MRB[i],boundary_MRB[i+1]) for i in range(num_segments)]
    #generate all possible error patterns of mrb
    error_pattern_list = [osd.error_pattern_gen(residual_path[j],range_list) for j in range(len(residual_path))]
    return error_pattern_list  

def query_samples(selected_ds,restore_info,tep_info,nn_type,option_tuple):
    order_sum = GL.get_map('threshold_sum')
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    snr_info = '/'+str(snr_lo)+"-"+str(snr_hi)+'dB/'
    num_blocks = GL.get_map('num_blocks')
    (tep_blocks,acc_block_size) = tep_info
    blocks_matrix = tf.concat(tep_blocks,axis=0)
    print('Selectd decoding path:\n',blocks_matrix)
    (indicator_list,DIA) = option_tuple
    
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
    #for i in range(10):
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
    if GL.get_map('extending_tep'):
        str_ext = 'extended'
    else:
        str_ext = 'proper'
    nominal_actual_length =  'order-pattern-len'+str(num_blocks)+'-'+str_ext+'-'
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
