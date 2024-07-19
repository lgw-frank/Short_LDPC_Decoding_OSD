# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
from tensorflow.keras import  metrics
import nn_net as CRNN_DEF
from collections import Counter,defaultdict,OrderedDict
import numpy as np
import pickle,re
import  os
from typing import Any, Dict,Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
#from tensorflow.python.trackable.data_structures import NoDependency
#from sympy.utilities.iterables import multiset_permutations
from itertools import combinations
import math
import ast
from scipy.special import comb

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
def identify_mrb(order_H_list):
    #initialize of mask
    code = GL.get_map('code_parameters')
    updated_index_list = []
    updated_M_list = []
    swap_len_list = []
    swap_lrb_position_list = []
    for i in range(order_H_list.shape[0]):
        # H assumed to be full row rank to obtain its systematic form
        tmp_H = np.copy(order_H_list[i])
        #reducing into row-echelon form and record column 
        #indices involved in pre-swapping
        swapped_H,record_col_index = gf2elim(tmp_H) 
        index_length = len(record_col_index)
        #update all swapping index
        index_order = np.array(range(code.check_matrix_column))
        for j in range(index_length):
            tmpa = record_col_index[j][0]
            tmpb = record_col_index[j][1]
            index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
        #udpated mrb indices
        updated_MRB = index_order[-code.k:]
        updated_LRB = index_order[:code.k]
        mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
        mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
        updated_index_order = tf.concat([index_order[:(code.check_matrix_column-code.k)],mrb_order],axis=0)
        updated_M = tf.gather(swapped_H[:,-code.k:],mrb_swapping_index,axis=1)    
        updated_index_list.append(updated_index_order)  
        updated_M_list.append(updated_M)
        swap_indicator = np.where(updated_MRB>=code.check_matrix_column-code.k,0,1)
        # focus of rear part of positions in LRB plus those positions swapped from MRB      
        discrinating_size = GL.get_map('sense_region')
        jump_point = (code.check_matrix_column-code.k)-discrinating_size
        swap_lrb_indicator = np.where(updated_LRB>=jump_point,1,0)
        swap_sum = sum(swap_indicator)
        swap_len_list.append(swap_sum)
        swap_lrb_position_list.append(swap_lrb_indicator)
    return updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list  

# def place_ones(size,order):
#   total_list = []
#   for i in range(order+1):
#       tmp_point = size*[0,]
#       if i:
#           tmp_point[:i] = i*(1,)
#       perm_a = list(multiset_permutations(tmp_point))
#       total_list += perm_a  
#   return np.reshape(total_list,[-1,size])

def generate_binary_arrays(length, hamming_weight):
    if hamming_weight < 0 or hamming_weight > length:
        return []
    all_arrays = []
    # Generate all combinations of indices with the specified hamming weight
    index_combinations = list(combinations(range(length), hamming_weight))
    for indices in index_combinations:
        binary_array = np.zeros(length, dtype=int)
        binary_array[list(indices)] = 1
        all_arrays.append((np.sum(np.nonzero(binary_array)[0]), tf.constant(binary_array, dtype=tf.int32)))
    # Sort by the sum of indices of nonzero elements in ascending order
    sorted_arrays = sorted(all_arrays, key=lambda x: x[0]) 
    return [arr[1] for arr in sorted_arrays]

def group_binary_arrays(arrays, batch_block_size):
    num_groups = math.ceil(len(arrays)/batch_block_size)
    grouped_arrays = [[] for _ in range(num_groups)]  
    for i, arr in enumerate(arrays):
        group_index = i // batch_block_size
        grouped_arrays[group_index].append(arr) 
    return grouped_arrays,num_groups

def analyze_distribution(original_counter, cluster_sequence,cluster_group):
    # Convert string keys to lists of integers
    new_list = [(list(map(int, key.split())), value) for key, value in original_counter.items()]
    # Get items in descending order of values
    items_descending = sorted(new_list, key=lambda x: x[1], reverse=True)
    group_occurrences = [0] * len(cluster_sequence)  
    # Generate a list of cumulative sums
    cumulative_sums = [0]+[sum(cluster_group[:i+1]) for i in range(len(cluster_group))]
    for item_tuple in items_descending:
        #print(item_tuple)
        #print(group_occurrences)
        indices_list = item_tuple[0]
        if indices_list == [-1]:
            start_index = cumulative_sums[0]
            end_index = cumulative_sums[1]
        else:
            shift = len(indices_list)
            if shift >= len(cumulative_sums)-1:
                continue
            start_index = cumulative_sums[shift]
            end_index = cumulative_sums[shift+1]
        increment = item_tuple[1]
        jump_indicator = False
        for i, group in enumerate(cluster_sequence[start_index:end_index]):
            for binary_array in group:
                if indices_list == [-1] or tf.reduce_all(tf.gather(binary_array, indices_list)!=0):
                    group_occurrences[start_index+i] += increment
                    jump_indicator = True
                    break
            if jump_indicator == True:
                break
    # Print the occurrences for each group
    print(group_occurrences)
    #for i, occurrences in enumerate(group_occurrences):
        #print(f"Group {i + 1}: {occurrences} occurrences")
    return group_occurrences




def execute_osd(inputs,labels):
    code = GL.get_map('code_parameters')
    order_list= check_matrix_reorder(inputs,labels)
    order_H_list,order_inputs,order_labels = order_list
    updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = identify_mrb(order_H_list)
    input_size = inputs.shape[0]
    order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
    order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]
    order_input_matrix = tf.reshape(order_input_list,[-1,code.check_matrix_column])
    order_label_matrix = tf.reshape(order_label_list,[-1,code.check_matrix_column])
    order_hard_matrix = tf.where(order_input_matrix>0,0,1)
    #record error bit positions of MRB
    bit_records = tf.where(order_hard_matrix[:,-code.k:]==order_label_matrix[:,-code.k:],0,1)
    # Find the indices of nonzero elements
    nonzero_indices = tf.where(bit_records != 0)
    # Group the indices by row using a dictionary
    count_dict = {}
    for row_index, col_index in nonzero_indices.numpy():
        if row_index in count_dict:
            count_dict[row_index].append(col_index)
        else:
            count_dict[row_index] = [col_index]
    # Convert the lists of column indices to strings
    count_dict_str = {row_index: ' '.join(map(str, col_indices)) for row_index, col_indices in count_dict.items()}
    # Use Counter to count occurrences
    Demo_result = Counter(count_dict_str.values())
    # Manually set count for a specific item of all zero TEP
    item_to_add = '-1'
    count_to_set = inputs.shape[0]-sum(Demo_result.values())
    Demo_result[item_to_add] = count_to_set
    print(Demo_result)
    return Demo_result
    
        
def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
    """Counts and returns model FLOPs.
  Args:
    model: A model instance.
    inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
      shape specifications to getting corresponding concrete function.
    output_path: A file path to write the profiling results to.
  Returns:
    The model's FLOPs.
  """
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            print('Failed to count model FLOPs with error %s, because the build() '
                 'methods in keras layers were not called. This is probably because '
                 'the model was not feed any input, e.g., the max train step already '
                 'reached before this run.', e)
            return None
    return None
def print_flops(model):
    flops = try_count_flops(model)
    print(flops/1e3,"K Flops")
    return None
def print_model_summary(model):
    # Create an instance of the model
    #model = ResNet(num_blocks=3, filters=64, kernel_size=3)
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # Print model summary
    list_length = GL.get_map('num_iterations')+1
    stripe = GL.get_map('stripe')
    model.build(input_shape=(None, list_length,stripe))  # Assuming input shape is (batch_size, sequence_length, input_dim)
    model.summary()
    return None
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

def calculate_loss(inputs,labels):
    labels = tf.cast(labels,tf.float32)  
    #measure discprepancy via cross entropy metric which acts as the loss definition for deep learning per batch         
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-inputs, labels=labels))
    return  loss

def custom_loss(y_pred, y_true):
        # Sort the indices of y_pred based on the absolute values
        sorted_indices = tf.argsort(tf.abs(y_pred), axis=-1, direction='DESCENDING')
        # Compute probabilities of being 1 using sigmoid function
        probability_matrix = 1-tf.sigmoid(y_pred)
        
        
        # Extract the indices of the first 100 bits with the largest magnitudes
        top_indices = sorted_indices[:, :118]
        
        # Sort y_pred based on the indices
        sorted_y_pred = tf.gather(probability_matrix, sorted_indices, batch_dims=1)
        
        # Extract the signs of these bits from both y_true and sorted_y_pred
        signs_true = tf.gather(y_true, top_indices, batch_dims=1)
        
        # Compute MSE loss for the signs of the selected bits
        mse_loss = tf.keras.losses.mean_squared_error(signs_true, sorted_y_pred[:,:118])
        
        # Sum up the MSE loss for the selected bits
        total_loss = tf.reduce_sum(mse_loss)
        return total_loss


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



def Training_NN(selected_ds,restore_info,indicator_list,prefix_list,DIA=False):
    snr_lo = round(GL.get_map('snr_lo'),2)
    snr_hi = round(GL.get_map('snr_hi'),2)
    list_length = GL.get_map('num_iterations')+1
    decoder_type = GL.get_map('selected_decoder_type')
    snr_info = str(snr_lo)+'-'+str(snr_hi)
    print_interval = GL.get_map('print_interval')
    record_interval = GL.get_map('record_interval')
    prefix = 'benchmark'
    #query of size of input feedings
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    if DIA:
        epochs = GL.get_map('epochs') 
        cnn = CRNN_DEF.conv_bitwise()
        rnn1 = CRNN_DEF.rnn_one()
        rnn2 = CRNN_DEF.rnn_two()
        nn_list = [cnn,rnn1,rnn2]    
        for i,element in enumerate(indicator_list):
            if element == True:
                nn = nn_list[i]
                prefix = prefix_list[i]
                break
        step = 0
        exponential_decay = GL.optimizer_setting()
        optimizer = tf.keras.optimizers.legacy.Adam(exponential_decay)
        checkpoint = tf.train.Checkpoint(myAwesomeModel=nn, myAwesomeOptimizer=optimizer)
        summary_writer,manager_current = GL.log_setting(restore_info,checkpoint,prefix)
        #unpack related info for restoraging
        [ckpts_dir,ckpt_nm,restore_step] = restore_info  
        if restore_step:
            print('Load the previous saved model from disk!')
            step,ckpt_f = retore_saved_model(ckpts_dir,restore_step,ckpt_nm)
            status = checkpoint.restore(ckpt_f)
            status.expect_partial()  
        loss_meter = metrics.Mean()
        #initialize starting point
        start_epoch = step//num_counter
        residual = step%num_counter
        jump_loop = False
        if GL.get_map('nn_train'):  
            if step < GL.get_map('termination_step'):
                for epoch in range(start_epoch,epochs):
                    print("\nStart of epoch %d:" % epoch) 
                    for i in range(residual,num_counter):
                        squashed_inputs,inputs,labels = nn.preprocessing_inputs(input_list[i])
                        step = step + 1
                        with tf.GradientTape() as tape:
                            refined_inputs = nn(squashed_inputs)
                            loss = calculate_loss(refined_inputs,labels)
                            #loss = custom_loss(refined_inputs, labels)
                            loss_meter.update_state(loss) 
                        total_variables = nn.trainable_variables         
                        grads = tape.gradient(loss,total_variables)
                        grads_and_vars=zip(grads, total_variables)
                        capped_gradients = [(tf.clip_by_norm(grad,5e2), var) for grad, var in grads_and_vars if grad is not None]
                        optimizer.apply_gradients(capped_gradients)   
                        if step % print_interval == 0:   
                            print('Step:%d  Loss:%.3f'%(step,loss.numpy()))
                            #_ = evaluate_MRB_bit(inputs,labels)
                            #_ = evaluate_MRB_bit(refined_inputs,labels)                                                               
                        if step % record_interval == 0:
                            manager_current.save(checkpoint_number=step)                   
                        if step >= GL.get_map('termination_step'):
                            jump_loop = True
                            break
                        loss_meter.reset_states()  
                    residual = 0
                    if jump_loop:
                        break
                #save the latest setting
                manager_current.save(checkpoint_number=step) 

    #verifying trained pars from start to end    
    dic_sum = {}
    dic_sum_initial = {}
    dic_sum_end = {}
    actual_size = 0
    pattern_cnt = Counter()
    swap_list = []
    loss_sum = 0.
    #query partition of MRB
    segment_size_list,boundary_MRB = GL.secure_segment_threshold()
    for i in range(num_counter):
        labels = input_list[i][1][list_length-1::list_length]
        if DIA:
            squashed_inputs,inputs,_ = nn.preprocessing_inputs(input_list[i])
            updated_inputs = nn(squashed_inputs)
            #nn.print_model()
            loss = calculate_loss(updated_inputs,labels)
        else:
            updated_inputs = input_list[i][0][list_length-1::list_length]
            loss = calculate_loss(updated_inputs,labels)
        loss_sum += loss   
        actual_size += updated_inputs.shape[0]

        cmp_results = evaluate_MRB_bit(updated_inputs,labels)
        dic_sum = dic_union(dic_sum,cmp_results)
        cmp_results = evaluate_MRB_bit(input_list[i][0][0::list_length],labels)
        dic_sum_initial = dic_union(dic_sum_initial,cmp_results)
        cmp_results = evaluate_MRB_bit(input_list[i][0][list_length-1::list_length],labels)
        dic_sum_end = dic_union(dic_sum_end,cmp_results)
        #query pattern distribution
        pattern_cnt,swap_len_list = generate_pattern_dist(updated_inputs,labels,pattern_cnt,boundary_MRB) 
        swap_list += swap_len_list
        if (i+1)%100 == 0:
            print(dic_sum) 
            #print(pattern_cnt) 
    # Merge identical keys by summing their counts
    merged_counter = Counter()
    for key, value in pattern_cnt.items():
        # Replace spaces with commas and evaluate as a Python expression
        key_with_commas = key.replace(' ', ',')
        key_list = ast.literal_eval(key_with_commas)
        summed_key = sum(key_list)
        merged_counter[summed_key] += value  
    #calculate ratioed yielding
    pattern_size_list = []
    for element in list(pattern_cnt.keys()):
        distilled_digits = re.findall(r"\w+",element)
        num_group = [int(element) for element in distilled_digits]
        pattern_size = np.prod(comb(segment_size_list,num_group))
        pattern_size_list.append(pattern_size)
    # Update counts by dividing each by individual digits
    updated_counts = {key: value / pattern_size_list[index] for index,(key,value) in enumerate(pattern_cnt.items())}
    # Create a new Counter instance with updated values
    updated_counter = Counter(updated_counts)

    average_loss = loss_sum/actual_size
    average_swaps = sum(swap_list)/len(swap_list)
    print(f'Total counts:{actual_size}')
    print(f'average swaps:{average_swaps:.4f} \naverage loss:{average_loss:.4f}')
    print('Summary for 0-th dist:',dic_sum_initial)
    print('Summary for T-th dist:',dic_sum_end)
    print('Summary before GE for '+prefix+':',dic_sum)
    print('Summary of after GE for '+prefix+':',merged_counter)
    print(f'Summary of decoding_pth:\n{pattern_cnt}')
    print('Summary of decoding_pth:')
    # Get items sorted by counts in descending order
    sorted_items = updated_counter.most_common()
    # Print the items in descending order of counts
    for key, value in sorted_items:
        print(f'{key}: {value:.4f}')
    #save on disk files
    log_dir = './log/'+decoder_type+'/'+snr_info+'dB/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  
    with open(log_dir+"dist-error-pattern-"+prefix+".pkl", "wb") as fh:
        pickle.dump(actual_size,fh)
        pickle.dump(dic_sum_initial,fh)        
        pickle.dump(dic_sum_end,fh)
        pickle.dump(dic_sum,fh)
        pickle.dump(merged_counter,fh)
        pickle.dump(updated_counter,fh)   
        
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
    
def generate_pattern_dist(inputs,labels,pattern_cnt,boundary_MRB):
    Updated_MRB_list,swap_len_list = stat_pro_osd(inputs,labels)    
    pattern_cnt = evaluate_MRB_pattern(inputs,labels,Updated_MRB_list,swap_len_list,pattern_cnt,boundary_MRB) 
    return pattern_cnt,swap_len_list


def evaluate_MRB_pattern(inputs,labels, updated_mrb_list,swap_len_list,pattern_cnt,boundary_MRB):
    inputs_abs = tf.abs(inputs)
    order_index = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
    order_inputs = tf.gather(inputs,order_index,batch_dims=1)
    order_labels = tf.cast(tf.gather(labels,order_index,batch_dims=1),tf.int32)
     
    updated_mrb_list = [np.array(list(element)) for element in updated_mrb_list]
    reorder_index = tf.sort(tf.reshape(updated_mrb_list,[inputs.shape[0],-1]),axis=-1,direction='ASCENDING')
    reorder_inputs = tf.gather(order_inputs,reorder_index,batch_dims=1)
    reorder_inputs_hard = tf.where(reorder_inputs>0,0,1)
    reorder_labels = tf.cast(tf.gather(order_labels,reorder_index,batch_dims=1),tf.int32)
    #difference matrix
    diff_matrix = tf.where(reorder_inputs_hard == reorder_labels,0,1)
    block_sum_list = []
    for i in range(len(boundary_MRB)-1):
        element = np.sum(diff_matrix[:,boundary_MRB[i]:boundary_MRB[i+1]],axis=-1,keepdims=True)
        block_sum_list.append(element)
    compound_block = tf.concat(block_sum_list,axis=-1)       
    for i in range(compound_block.shape[0]):
        #word = str(compound_block[i].numpy())
        word = ','.join(map(str,compound_block[i].numpy()))
        pattern_cnt[word] +=1  
    return pattern_cnt


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


