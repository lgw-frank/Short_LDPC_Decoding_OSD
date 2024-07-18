# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:53:46 2022

@author: zidonghua_30
"""
#from hyperts.toolbox import from_3d_array_to_nested_df
import globalmap as GL
import tensorflow as tf
# import ordered_statistics_decoding as OSD_mod
# import re
# import pickle
import os
import time
import numpy as  np
import scipy.stats as stats
import math
from collections import Counter
import convention_osd as cnv_OSD
from itertools import combinations        

def acquire_pnc_boundary(descending_input):
    order_limit = GL.get_map('order_limit')
    code = GL.get_map('code_parameters')
    dimension_k = code.k
    boundary_list = []
    for i in range(order_limit):
        element = sum(abs(descending_input)[dimension_k-(i+1):dimension_k])
        boundary_list.append(element)
    return boundary_list

def generate_sequential_teps(max_value,max_loops):
    code = GL.get_map('code_parameters')
    dimension_k = code.k
    tep_matrix_list = []
    for num_loops in range(1,max_loops+1):
        # List to store the vectors
        vectors = []  
        # Generate combinations of iterators without replacement
        for combination in combinations(range(max_value), num_loops):
            # Create an array of zeros
            vector = np.zeros(max_value, dtype=int)    
            # Set the elements corresponding to the indices in the combination to '1'
            vector[list(combination)] = 1
            # Append the vector to the list
            vectors.append(vector[::-1])
        tep_matrix = tf.reshape(vectors,[-1,dimension_k])
        tep_matrix_list.append(tep_matrix)
    return tep_matrix_list
    
def one_tep_compare(updated_inputs,nth_tep,reduced_G,threshold):
    code = GL.get_map('code_parameters')
    diemsnion_k = code.k
    updated_hard_desicions = tf.cast(tf.where(updated_inputs>0,0,1),tf.int32)
    #generate all-zero tep mapped codeword estimate
    new_mrb = (updated_hard_desicions[:diemsnion_k]+nth_tep)%2
    optimal_codeword = tf.matmul(tf.reshape(new_mrb,[1,-1]),reduced_G)%2
    hard_distance = (optimal_codeword+updated_hard_desicions)%2
    #initialize the weighted Hamming weight
    w_dmin = tf.reduce_sum(tf.cast(hard_distance,tf.float32)*abs(updated_inputs)) 
    early_stopping = False
    if float(tf.reduce_sum(hard_distance)) < threshold:
        early_stopping = True
    return early_stopping,optimal_codeword,w_dmin

       
        
def fs_osd(snr,beta,selected_ds):
    start_time = time.process_time()
    code = GL.get_map('code_parameters')
    dimension_n = code.check_matrix_column
    dimension_k = code.k
    #n_iteration = GL.get_map('num_iterations')
    #list_length = n_iteration+1
    order_limit = GL.get_map('order_limit')
    #preparing input data 
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list)
    fail_sum = 0
    correct_sum = 0
    counter_teps_sum = 0
    if GL.get_map('miracle_view'):
        counter_stat = Counter()
    if GL.get_map('convention_osd'):
        order_limit = GL.get_map('order_limit')
        convention_counter = Counter()
        convetion_success = 0
        convetion_fail = 0 
    optimal_list = []
    tau_psc = GL.get_map('tau_psc')
    #calculate tau
    tau_e = math.floor(GL.get_map('d_min')-1)/2
    tep_matrix_list = generate_sequential_teps(dimension_k,order_limit)
    for i in range(num_counter):
        print('.',end='')
        early_jumping = False
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
                total_num = (convetion_success+convetion_fail)
                success_rate = round(convetion_success/total_num,4)
                average_num_teps = counter_teps_sum/total_num
                print('\nFor %.1fdB (order_limit:%d) summary:'%(snr,order_limit))
                print('--> S/F:',convetion_success,'/',convetion_fail,'Success rate:',success_rate)  
                print(f'Cost of num of TEPs:{teps_size}')
                T2 =time.process_time()
                print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')

            continue

        if GL.get_map('fs_osd'):
            boundary_list = acquire_pnc_boundary(updated_inputs)
            all_zero_tep = [0]*dimension_k
            early_stopping,optimal_codeword,w_dmin = one_tep_compare(updated_inputs,all_zero_tep,reduced_G,tau_e)
            num_teps = 1
            if early_stopping:
                optimal_list.append(optimal_codeword) 
            else:
                s_low_bound_list = [x + beta*(dimension_n-dimension_k) for x in boundary_list]
                for j in range(order_limit):
                    if s_low_bound_list[j] < w_dmin:
                        teps_list = tep_matrix_list[j]
                        for nth_tep in teps_list:
                            num_teps += 1
                            early_stopping,new_codeword,new_w_d = one_tep_compare(updated_inputs,nth_tep,reduced_G,tau_e)
                            if early_stopping:
                                optimal_list.append(new_codeword) 
                                early_jumping = True
                                break
                            second_early_stopping,_,_ = one_tep_compare(updated_inputs,nth_tep,reduced_G,tau_psc)                        
                            if second_early_stopping:
                                if new_w_d < w_dmin:
                                    w_dmin = new_w_d
                                    optimal_codeword = new_codeword
                        if early_jumping:
                            break
                    else:
                        optimal_list.append(optimal_codeword) 
                        early_jumping = True
                        break
                if not early_jumping:
                    optimal_list.append(optimal_codeword) 
            counter_teps_sum +=  num_teps     
            if tf.reduce_sum(abs(optimal_codeword-tf.cast(updated_labels,tf.int32))):
                fail_sum += 1
            else:
                correct_sum +=1
            if (i+1)%10 == 0:
                total_num = (correct_sum+fail_sum)
                success_rate = round(correct_sum/total_num,4)
                average_num_teps = counter_teps_sum/total_num
                print('\nFor %.1fdB (order_limit:%d) summary:'%(snr,order_limit))
                print(f'--> S/F:{correct_sum}/{fail_sum} Error rate:{1-success_rate:.4f}')  
                print(f'Num of TEPs:{average_num_teps:.2f}')
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
        #print(counter_stat.most_common())
        # Get the keys
        # keys = list(counter_stat.keys())
        # print("Keys:", keys)      
    if GL.get_map('fs_osd'):
        logdir = './log/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_filename = logdir+'FS-OSD-order-'+str(order_limit)+'.txt' 
        total_num = (correct_sum+fail_sum)
        FER = round(fail_sum/total_num,4)
        average_size = round(counter_teps_sum/total_num,5)
        print('\nFor FS-OSD %.1fdB (order_limit:%d) '%(snr,order_limit)+':\n')
        print('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')   
        print(f'FER:{FER:.2f} Average TEPs:{average_size:.2f} \n')
        T2 =time.process_time()
        print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')
        with open(log_filename,'a+') as f:
          f.write('\nFor FS-OSD %.1fdB (order_limit:%d) summary:\n'%(snr,order_limit))
          f.write('----> S:'+str(correct_sum)+' F:'+str(fail_sum)+'\n')         
          f.write(f'FER:{FER:.2f} Average TEPs:{average_size:.2f}\n')
          f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!\n')
    if GL.get_map('convention_osd'):
        logdir = './log/'
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        log_filename = logdir+'CNV-OSD-order-'+str(order_limit)+'.txt' 
        total_num = (convetion_success+convetion_fail)
        FER = round(convetion_fail/total_num,4)
        print('\nFor Conv-OSD %.1fdB (order_limit:%d) '%(snr,order_limit)+':\n')
        print('----> S:'+str(convetion_success)+' F:'+str(convetion_fail)+'\n')     
        print('Distribution of phases:'+str(convention_counter)+'\n')
        print('FER:'+str(FER)+' Average TEPs size:',teps_size,'\n')
        T2 =time.process_time()
        print(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')
        with open(log_filename,'a+') as f:
          f.write('\nFor CNV-OSD %.1fdB (order_limit:%d) summary:\n'%(snr,order_limit))
          f.write('----> S:'+str(convetion_success)+' F:'+str(convetion_fail)+'\n')   
          f.write('Distribution of phases:'+str(convention_counter)+'\n')
          f.write('FER:'+str(FER)+' Average TEPs size:'+str(teps_size)+'\n')
          f.write(f'Running time:{T2 - start_time} seconds with mean time {(T2 - start_time)/total_num:.4f}!')

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

def miracle_view(updated_inputs,updated_labels,reduced_G,counter_stat):
    code = GL.get_map('code_parameters') 
    updated_hard_desicions = tf.cast(tf.where(updated_inputs>0,0,1),tf.int32)
    #codeword_estimate = tf.matmul(tf.reshape(updated_hard_desicions[:code.k],[1,-1]),reduced_G)%2
    #distance_hard = (updated_hard_desicions+codeword_estimate)%2
    authentic_error_pattern = (updated_hard_desicions[:code.k]+tf.cast(updated_labels[:code.k],tf.int32))%2
    mrb_error_num = tf.reduce_sum(authentic_error_pattern).numpy()
    #print('miracle result:',mrb_error_num)
    # if mrb_error_num<=1:
    #     print('miracle result:',mrb_error_num)
    #     #print(authentic_error_pattern.numpy())
    #     guess_whd = np.sum(tf.cast(distance_hard,tf.float32)*abs(updated_inputs))
    #     distance_authentic = (updated_hard_desicions+tf.cast(updated_labels,tf.int32))%2
    #     truth_whd = np.sum(tf.cast(distance_authentic,tf.float32)*abs(updated_inputs))
    #     print(f'guess_whd:{guess_whd} truth_whd:{truth_whd}')
    counter_stat.update([mrb_error_num])
    
    return counter_stat,mrb_error_num