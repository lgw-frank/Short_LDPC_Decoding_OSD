# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
#import nn_net as CRNN_DEF

import os
import numpy as  np
import scipy.stats as stats
import math
from collections import Counter
import convention_osd as cnv_OSD
import time

def retore_saved_model(restore_model_info):
    restore_ckpts_dir,ckpt_nm,restore_step = restore_model_info
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


def com_mrb_prob(noise_variance,input_soft_info): 
    code = GL.get_map('code_parameters')
    prob_mrb = 1.0
    for i in range(code.k):
        ith_prob = 1-tf.sigmoid(-4*noise_variance*abs(input_soft_info[i]))
        prob_mrb = prob_mrb*ith_prob  
    return prob_mrb      
        
        
def pb_osd(snr,selected_ds):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    #n_iteration = GL.get_map('num_iterations')
    #list_length = n_iteration+1
    order_limit = GL.get_map('order_limit')
    #inverse the noise variance
    gamma = 10**(snr/10)
    noise_variance = 1/gamma
    #preparing input data 
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)
    fail_sum = 0
    correct_sum = 0
    counter_teps_sum = 0
    counter_suc_sum1 = 0
    counter_suc_sum2 = 0
    if GL.get_map('miracle_view'):
        counter_stat = Counter()
    if GL.get_map('convention_osd'):
        order_limit = GL.get_map('order_limit')
        convention_counter = Counter()
        convetion_success = 0
        convetion_fail = 0
    memory_sum = 0    
    for i in range(num_counter):
        #hard-decision of received sequence
        input_data = input_list[i][0][0]
        input_label = input_list[i][1][0]          
        updated_inputs,updated_labels,reduced_G = swapped_info(input_data,input_label)
        if GL.get_map('miracle_view'):
            counter_stat,mrb_error_num = miracle_view(updated_inputs,updated_labels,reduced_G,counter_stat)
            if (i+1)%1000 == 0:
                print('\nFor %.1fdB (order_limit:%d) summary:'%(snr,order_limit))
                print('--> Distribution:',counter_stat)  
            continue
        if GL.get_map('convention_osd'):
            error_patterns_matrix = cnv_OSD.generate_teps(order_limit)
            boundary_list = cnv_OSD.query_boundary(order_limit)
            decoding_info = updated_inputs,updated_labels,reduced_G,error_patterns_matrix,boundary_list
            correct_indicator,teps_size,belonged_phase = cnv_OSD.convention_osd_main(decoding_info)
            convention_counter.update([belonged_phase])
            if correct_indicator:
                convetion_success += 1
            else:
                convetion_fail +=1
            if (i+1)%10 == 0:
                total_num =convetion_fail + convetion_success
                success_rate = round(convetion_success/total_num,4)
                average_num_teps = counter_teps_sum/total_num
                print('\nFor %.1fdB (order_limit:%d) summary:'%(snr,order_limit))
                print(f'--> S/F:{convetion_success}/{convetion_fail} FER:{1-success_rate:.4f}')  
                print(f'Cost of num of TEPs:{teps_size}')
                T2 =time.process_time()
                print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')

        if GL.get_map('pb_osd'):
            updated_hard_desicions = tf.cast(tf.where(updated_inputs>0,0,1),tf.int32)
            #generate all-zero tep mapped codeword estimate
            optimal_codeword = tf.matmul(tf.reshape(updated_hard_desicions[:code.k],[1,-1]),reduced_G)%2
            hard_distance = (optimal_codeword+updated_hard_desicions)%2
            #initialize the weighted Hamming weight
            w_dmin = tf.reduce_sum(tf.cast(hard_distance,tf.float32)*abs(updated_inputs))        
            
            #early_termination_indicator = False
            #initialize searching list
            starting_point = code.k-1
            tep_matrix = create_binary_tensor(code.k,starting_point)
            cost_tep_num = 0
            early_stopping = False
            #common terms
            p1 = mean_lrb_prob(noise_variance,updated_inputs)
            para_list = [p1,0.5]
            pt = mean_mrb_prob(noise_variance,updated_inputs)  #for calculation of p_e^suc and p_e^pro
            p_t_suc,p_t_pro,N_max = calculate_two_thresholds(pt)
            spl_mrb_prob = com_mrb_prob(noise_variance,updated_inputs)
            for j in range(N_max-1):
                tep_matrix,selected_tep,comparison_counter = optimal_tep_sequence(updated_inputs,tep_matrix) 
                memory_sum += comparison_counter
                #generate codeword candidate
                mrb_candidate = (updated_hard_desicions[:code.k]+selected_tep)%2
                codeword_candidate = tf.matmul(tf.reshape(mrb_candidate,[1,-1]),reduced_G)%2
                hard_distance = (codeword_candidate+updated_hard_desicions)%2
                wrapped_info = (selected_tep,noise_variance,updated_inputs,spl_mrb_prob,w_dmin,para_list)
                p_e_pro = acquire_prob_promising(wrapped_info)
                if p_e_pro < p_t_pro:
                        early_stopping = True
                        cost_tep_num = j+1
                        break
                #generate codeword candidate
                mrb_candidate = (updated_hard_desicions[:code.k]+selected_tep)%2
                codeword_candidate = tf.matmul(tf.reshape(mrb_candidate,[1,-1]),reduced_G)%2
                hard_distance = (codeword_candidate+updated_hard_desicions)%2
                w_de = tf.reduce_sum(tf.cast(hard_distance,tf.float32)*abs(updated_inputs))
                counter_suc_sum1 += 1
                if w_de < w_dmin:                   
                    optimal_codeword = codeword_candidate
                    w_dmin = w_de
                    hard_distance = tf.squeeze(hard_distance)
                    p_e_suc = acquire_p_e_suc(noise_variance,updated_inputs,spl_mrb_prob,hard_distance)
                    counter_suc_sum2 += 1
                    if p_e_suc > p_t_suc:
                        early_stopping = True
                        cost_tep_num = j+1
                        #print('Early stopping decoding occurred!')
                        break           
            #print(f'{w_dmin.numpy():.2f}',end=' ')
            print('.',end='')
            if early_stopping:            
                counter_teps_sum += cost_tep_num
            else:
                counter_teps_sum += N_max
                print('\n********full traversing TEPs**********')
               
            if tf.reduce_sum(abs(optimal_codeword-tf.cast(updated_labels,tf.int32))):
                fail_sum += 1
            else:
                correct_sum +=1
            if (i+1)%10 == 0:
                total_num = correct_sum+fail_sum
                success_rate = round(correct_sum/total_num,4)
                average_num_teps = counter_teps_sum/total_num
                average_num_memory = memory_sum/total_num
                average_suc_counter1 = counter_suc_sum1/total_num
                average_suc_counter2 = counter_suc_sum2/total_num
                print('\nFor %.1fdB (order_limit:%d) summary:'%(snr,order_limit))
                print(f'--> S/F:{correct_sum}/{fail_sum} FER:{1-success_rate:.4f}')  
                print(f'Num of TEPs:{average_num_teps:.2f} Maintained_list:{average_num_memory:.2f} Average_suc: {average_suc_counter1:.2f}/{average_suc_counter2:.2f}')
                T2 =time.process_time()
                print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')
            if fail_sum >= GL.get_map('termination_num_threshlod'):
                break                                         
    
    if GL.get_map('miracle_view'):
        print('\nFor miracle view %.1fdB (order_limit:%d) '%(snr,order_limit)+':')
        # Sum up all values
        total_sum = sum(counter_stat.values())
        print(f'total_sum:{total_sum}')
        # Initialize an accumulator
        accumulated_values = 0
        # Sort items by keys in ascending order
        sorted_items = sorted(counter_stat.items())
        # Iterate through sorted items and accumulate values gradually
        for key, value in sorted_items:
            accumulated_values += value
            print(f"order-{key}: Accumulated Ratio: {accumulated_values/total_sum:.4f}")      
    if GL.get_map('pb_osd'):
        logdir = './log/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_filename = logdir+'PB-OSD-order-'+str(order_limit)+'.txt'   
        actual_size = correct_sum+fail_sum   
        FER = round(fail_sum/actual_size,4)
        average_size = round(counter_teps_sum/actual_size,5)
        average_num_memory = round(memory_sum/actual_size,5)
        average_suc_counter1 = round(counter_suc_sum1/actual_size,5)
        average_suc_counter2 = round(counter_suc_sum2/actual_size,5)
        print('\nFor PB-OSD %.1fdB (order_limit:%d) '%(snr,order_limit)+':\n')
        print('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')   
        print(f'FER:{FER:.4f} Average TEPs:{average_size:.2f} Maintained_list_len:{average_num_memory:.2f} Average_suc: {average_suc_counter1:.2f}/{average_suc_counter2:.2f}')
        T2 =time.process_time()
        print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!')
        with open(log_filename,'a+') as f:
          f.write('\nFor PB-OSD %.1fdB (order_limit:%d) summary:\n'%(snr,order_limit))
          f.write(f'--> S/F:{correct_sum}/{fail_sum}\n')         
          f.write(f'FER:{FER:.5f} Average TEPs:{average_size:.2f} Maintained_list_len:{average_num_memory:.2f} Average_suc: {average_suc_counter1:.2f}/{average_suc_counter2:2f}\n')
          f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/actual_size:.4f}!\n')
    if GL.get_map('convention_osd'):
        logdir = './log/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_filename = logdir+'CNV-OSD-order-'+str(order_limit)+'.txt' 
        T2 =time.process_time()
        total_sum = (convetion_success+convetion_fail)
        FER = round(convetion_fail/total_sum,4)
        print('\nFor CNV-OSD %.1fdB (order_limit:%d) '%(snr,order_limit)+':\n')
        print('----> S:'+str(convetion_success)+' F:'+str(convetion_fail)+'\n')     
        print('Distribution of phases:'+str(convention_counter)+'\n')
        print('FER:'+str(FER)+' Average TEPs size:',teps_size,'\n')
        print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_sum:.4f}!')
        with open(log_filename,'a+') as f:
          f.write('\nFor CNV-OSD %.1fdB (order_limit:%d) summary:\n'%(snr,order_limit))
          f.write('----> S:'+str(convetion_success)+' F:'+str(convetion_fail)+'\n')   
          f.write('Distribution of phases:'+str(convention_counter)+'\n')
          f.write('FER:'+str(FER)+' Average TEPs size:'+str(teps_size)+'\n')
          f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_sum:.4f}!\n')

def full_gf2elim(M):
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

def identify_mrb(order_inputs,order_G):
    #initialize of mask
    code = GL.get_map('code_parameters')
    # G assumed to be full row rank to obtain its systematic form
    tmp_G = np.copy(order_G)
    #reducing into row-echelon form and record column 
    #indices involved in pre-swapping
    swapped_G,record_col_index = full_gf2elim(tmp_G) 
    index_length = len(record_col_index)
    #update all swapping index
    index_order = np.array(range(code.check_matrix_column))
    for j in range(index_length):
        tmpa = record_col_index[j][0]
        tmpb = record_col_index[j][1]
        index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
    #udpated mrb indices
    updated_MRB = index_order[:code.k]
    mrb_swapping_index = tf.argsort(updated_MRB,axis=0,direction='ASCENDING')
    mrb_order = tf.sort(updated_MRB,axis=0,direction='ASCENDING')
    #next to aquire the permuation-mapped matrix
    identity_matrix = np.identity(code.k, dtype=int)
    updated_mrb_matrix = tf.gather(identity_matrix,mrb_swapping_index,axis=1)  
       
    #updated lrb indices
    updated_LRB = index_order[code.k:]
    lrb_swapping_index = tf.argsort(updated_LRB,axis=0,direction='ASCENDING')
    lrb_order = tf.sort(updated_LRB,axis=0,direction='ASCENDING')    
    interm_lrb_matrix = tf.gather(swapped_G[:,code.k:],lrb_swapping_index,axis=1)  
    # Perform matrix multiplication in Galois field (GF(2))
    # notice: the left inverse is exactly the transpose matrix in GF(2) 
    updated_lrb_matrix = tf.math.mod(tf.matmul(tf.cast(tf.transpose(updated_mrb_matrix),tf.int32), tf.cast(interm_lrb_matrix,tf.int32)), 2)    
    
    updated_G = tf.concat([identity_matrix,updated_lrb_matrix],axis=1)  
    
    updated_index_order = tf.concat([mrb_order,lrb_order],axis=0)
    
    return updated_G,updated_index_order 
                        
def swapped_info(inputs,labels):
    code = GL.get_map('code_parameters')
    inputs_abs = abs(inputs)
    #pi one permutation
    reorder_index = tf.argsort(inputs_abs,axis=-1,direction='DESCENDING')
    order_inputs = tf.gather(inputs,reorder_index)
    order_labels = tf.gather(labels,reorder_index)
    order_G = tf.gather(code.G,reorder_index,axis=1).numpy()
    #print(order_H.dot(order_G.T)%2)
    #Gaussian elimination
    reduced_G,updated_index_order = identify_mrb(order_inputs,order_G)
    #pi two permutation as result of GE   
    updated_inputs = tf.gather(order_inputs,updated_index_order,axis=0)
    updated_labels = tf.gather(order_labels,updated_index_order,axis=0)
    return updated_inputs,updated_labels,reduced_G 

def format_list_print(list_data):
    formatted_values = [f'{abs(value):.4f}' for value in list_data]
    result_str = ', '.join(formatted_values)
    print(f'formated_list: {result_str}')

def find_smallest_index(tep_sample,count):
    # Mask non-zero elements
    non_zero_mask = tf.math.not_equal(tep_sample, 0)
    non_zero_values = tf.boolean_mask(tep_sample, non_zero_mask)
    
    # Find the indices of the smallest two non-zero elements
    sorted_indices = tf.argsort(non_zero_values)
    smallest_two_indices = tf.gather(tf.where(non_zero_mask), sorted_indices[:count])
    # Convert the result to a numpy array for easier printing
    smallest_two_indices_np = smallest_two_indices.numpy()
    return smallest_two_indices_np

def create_binary_tensor(mrb_length, indices):
    # Create a binary tensor with all zeros
    binary_tensor = tf.zeros(mrb_length, dtype=tf.int32)
    # Make sure indices is one-dimensional
    indices = tf.reshape(indices, [-1])
    # Set non-zero elements at specified indices
    binary_tensor = tf.tensor_scatter_nd_update(binary_tensor, tf.expand_dims(indices, axis=1), tf.ones_like(indices, dtype=tf.int32))
    binary_tensor = tf.reshape(binary_tensor,[1,-1])
    return binary_tensor
def delete_row(matrix, row_index_to_delete):
    # Use tf.range to create a range of indices
    indices_to_keep = tf.range(matrix.shape[0]) != tf.cast(row_index_to_delete,tf.int32)
    # Use tf.boolean_mask to keep all rows except the one to be deleted
    new_matrix = tf.boolean_mask(matrix, indices_to_keep, axis=0)
    return new_matrix
def append_row_if_not_exists(matrix, new_row):
    size = tf.size(matrix)
    if tf.equal(size, 0):
        new_matrix = tf.reshape(new_row,[1,-1])
    else:
        # Check if the new row already exists in any row of the matrix
        exists_check = tf.reduce_any(tf.reduce_all(tf.equal(matrix, new_row), axis=1))
        # If the new row doesn't exist, append it to the matrix
        if not exists_check.numpy():
            new_matrix = tf.concat([matrix, new_row], axis=0)
    return new_matrix

def optimal_tep_sequence(input_soft_info,tep_matrix):
    code = GL.get_map('code_parameters')
    order_limit = GL.get_map('order_limit')
    abs_mrb = tf.reshape(abs(input_soft_info[:code.k]),[1,-1])  
    min_index = tf.argmin(tf.reduce_sum(abs_mrb*tf.cast(tep_matrix,tf.float32),axis=-1))
    if tep_matrix.shape[0]==1:
      comparison_counter = 1
    else:
      comparison_counter = 2
    selected_tep = tep_matrix[min_index]
    # Find the indices of non-zero elements
    nonzero_indices = tf.where(tep_matrix[min_index] != 0) 
    udpated_tep_matrix = delete_row(tep_matrix,min_index)
    #find if the extended tep exists
    if  nonzero_indices[-1] < code.k-1 and len(nonzero_indices) < order_limit:
        new_indices = np.append(nonzero_indices, code.k-1)
        extended_tep = create_binary_tensor(code.k,new_indices)
        udpated_tep_matrix = append_row_if_not_exists(udpated_tep_matrix, extended_tep)
    #find if the adjacient tep exists
    if len(nonzero_indices) > 1:
        if nonzero_indices[-1]-nonzero_indices[-2] > 1:
            right_most_index = nonzero_indices[-1]-1
            new_indices = np.append(nonzero_indices[:-1], right_most_index) 
            adjacient_tep = create_binary_tensor(code.k,new_indices)
            udpated_tep_matrix = append_row_if_not_exists(udpated_tep_matrix, adjacient_tep)
    else:
        new_index = nonzero_indices[-1]-1
        if new_index > -1 :
            adjacient_tep = create_binary_tensor(code.k,new_index)
            udpated_tep_matrix = append_row_if_not_exists(udpated_tep_matrix, adjacient_tep)
        
    return udpated_tep_matrix,selected_tep,comparison_counter

def beta_acquire(input_soft_info,tep,w_d_min):
    code = GL.get_map('code_parameters')
    reliability_lrb_mean = tf.reduce_mean(abs(input_soft_info[code.k:]))
    reliability_mrb_sum = tf.reduce_sum(abs(input_soft_info[:code.k])*tf.cast(tep,tf.float32))
    tmp = max(0,tf.floor((w_d_min-reliability_mrb_sum)/reliability_lrb_mean))
    beta = min(tmp,code.check_matrix_column-code.k)
    return beta
def mean_lrb_prob(noise_variance,input_soft_info):   
    code = GL.get_map('code_parameters')
    #average probability of bits in lrb portion
    prob_lrb = tf.sigmoid(-4*noise_variance*abs(input_soft_info[code.k:]))
    prob_lrb_mean = tf.reduce_mean(prob_lrb)   
    return prob_lrb_mean
# approximate the mixture of two binomials of cdfs:(n1,p1), (n2,p2), here p1= prob_lrb_mean, p2 =1/2
def approximate_binomials_cdf(cdf_info):
    (n1,p1,n2,p2,w1,w2,beta) = cdf_info
    mean = w1*n1*p1+w2*n2*p2
    std = tf.sqrt(w1**2*n1*p1*(1-p1)+w2**2*n2*p2*(1-p2))
    # Compute the Q-function using tf.math.qtf
    # Compute the Q-function using 1 - tf.math.erf
    z_value = (beta-mean)/std
    q_function_result = 1 - tf.math.erf(z_value / tf.sqrt(2.0))
    return q_function_result

def acquire_p_e_suc(noise_variance,input_soft_info,spl_mrb_prob,codeword_distance):
    code = GL.get_map('code_parameters')
    tep_p = tep_prob(noise_variance,input_soft_info,spl_mrb_prob,codeword_distance[:code.k])
    ratio_factor = (1-tep_p)/tep_p
    #to ensure numerical stability, we use exp(lg(x)) = x
    prob_product = 1.0
    for i in range(code.k,code.check_matrix_column):
        if codeword_distance[i] == 0:
            prob_factor = 2*(1-tf.sigmoid(-4*noise_variance*abs(input_soft_info[i])))
        else:
            prob_factor = 2*(tf.sigmoid(-4*noise_variance*abs(input_soft_info[i])))
        prob_product = prob_product*prob_factor
    p_e_suc = 1/(1+ratio_factor/prob_product)
    return p_e_suc



# acquiring the probability of error pattern given the i-th bits are erroneous conditioned on reliabilities.
def tep_prob(noise_variance,input_soft_info,spl_mrb_prob,tep):
    code = GL.get_map('code_parameters')
    reliability_sum = tf.reduce_sum(abs(input_soft_info[:code.k])*tf.cast(tep,tf.float32))
    tep_p = tf.math.exp(-4*noise_variance*reliability_sum)*spl_mrb_prob
    return tep_p

#acquire p_e^pro
def acquire_prob_promising(wrapped_info): 
    (selected_tep,noise_variance,input_soft_info,spl_mrb_prob,w_dmin,para_list) = wrapped_info
    code = GL.get_map('code_parameters')
    n = code.check_matrix_column-code.k
    p1,p2 = para_list[0],para_list[1]
    w1 = tep_prob(noise_variance, input_soft_info,spl_mrb_prob,selected_tep)
    w2 = 1- w1
    beta = beta_acquire(input_soft_info, selected_tep, w_dmin)
    binomial_sum = 0.    
    for p,w in [(p1,w1),(p2,w2)]:  
        binomial_cdf = stats.binom.cdf(beta, n, p)
        binomial_sum += w*binomial_cdf
    #print("Weighted Binomial CDF sum:", binomial_sum)
    return tf.cast(binomial_sum,tf.float64)

def mean_mrb_prob(noise_variance,input_soft_info):   
    code = GL.get_map('code_parameters')
    #average probability of bits in mrb portion
    prob_mrb = tf.sigmoid(-4*noise_variance*abs(input_soft_info[:code.k]))
    prob_mrb_mean = tf.reduce_mean(prob_mrb)   
    return prob_mrb_mean        

def approximate_single_binomial(cdf_info):
    # Parameters for the binomial distribution and the expected argumented value required
    (n,p,order_limit) = cdf_info
    # Continuity-corrected normal approximation
    approximation = stats.norm.cdf((order_limit + 0.5 - n * p) / tf.sqrt(n * p * (1 - p))) 
    # Direct computation of binomial CDF for k=3
    # binomial_distribution = tfp.distributions.Binomial(total_count=n, probs=p)
    # binomial_cdf = binomial_distribution.cdf(order_limit)
    binomial_cdf = stats.binom.cdf(order_limit, n, p)
    return approximation,binomial_cdf

# Function to calculate binomial coefficient
def binomial_coefficient(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        
def calculate_two_thresholds(pt):
    code = GL.get_map('code_parameters')
    order_limit = GL.get_map('order_limit')
    cdf_info = (code.k,pt,order_limit) 
    niu = approximate_single_binomial(cdf_info)[1]
    p_t_suc = 0.99*niu
    #p_t_suc = max(0.99*niu,0.9)
    # Calculate binomial coefficient
    comb_sum = 0
    for i in range(order_limit+1):
        comb_count = binomial_coefficient(code.k, i)
        comb_sum += comb_count
    p_t_pro = 0.002*tf.sqrt((1-niu)/comb_sum)
    #p_t_pro = 0.0002*tf.sqrt((1-niu)/comb_sum)
    N_max = comb_sum
    return p_t_suc,p_t_pro,N_max

def miracle_view(updated_inputs,updated_labels,reduced_G,counter_stat):
    code = GL.get_map('code_parameters') 
    updated_hard_desicions = tf.cast(tf.where(updated_inputs>0,0,1),tf.int32)
    #codeword_estimate = tf.matmul(tf.reshape(updated_hard_desicions[:code.k],[1,-1]),reduced_G)%2
    #distance_hard = (updated_hard_desicions+codeword_estimate)%2
    authentic_error_pattern = (updated_hard_desicions[:code.k]+tf.cast(updated_labels[:code.k],tf.int32))%2
    mrb_error_num = tf.reduce_sum(authentic_error_pattern).numpy()
    counter_stat.update([mrb_error_num])
    
    return counter_stat,mrb_error_num