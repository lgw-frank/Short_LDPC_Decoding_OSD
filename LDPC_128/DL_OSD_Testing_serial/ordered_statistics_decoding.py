
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:34:01 2023

@author: Administrator
"""

import tensorflow as tf
import numpy as np
import globalmap as GL
from itertools import combinations,chain
import itertools as it
from scipy.special import comb
#from sympy.utilities.iterables import multiset_permutations

# input operation:1)magnitude swapping 2)Gaussian elimination swapping 3)MRB magnitude swapping
class osd:
    def __init__(self,code):
        self.original_H = code.H  
        self.n_dims = code.check_matrix_column
        self.k = code.k
        self.m = self.n_dims - self.k         
    #magnitude of signals
    def mag_input_gen(self,inputs):
        inputs_abs = abs(inputs)
        reorder_index_batch = tf.argsort(inputs_abs,axis=-1,direction='ASCENDING')
        return reorder_index_batch         
   
    def check_matrix_reorder(self,iteration_inputs,inputs,labels):
        expanded_H = tf.expand_dims(self.original_H,axis=0)
        list_length = GL.get_map('num_iterations')+1
        #query the least reliable independent positions
        lri_p = self.mag_input_gen(inputs)
        order_inputs = tf.gather(inputs,lri_p,batch_dims=1)
        order_original_list = [tf.gather(iteration_inputs[i::list_length],lri_p,batch_dims=1)  for i in range(list_length)]
        order_labels = tf.gather(labels,lri_p,batch_dims=1)
        batched_H = tf.transpose(tf.tile(expanded_H,[inputs.shape[0],1,1]),perm=[0,2,1])
        tmp_H_list = tf.gather(batched_H,lri_p,batch_dims=1)
        order_H_list = tf.transpose(tmp_H_list,perm=[0,2,1])
        return order_H_list,order_inputs,order_original_list,order_labels 
    
    def identify_mrb(self,order_H_list):
        #initialize of mask
        code = GL.get_map('code_parameters')
        threshold_sum = GL.get_map('threshold_sum')
        updated_index_list = []
        updated_M_list = []
        swap_len_list = []
        swap_lrb_position_list = []
        for i in range(order_H_list.shape[0]):
            # H assumed to be full row rank to obtain its systematic form
            tmp_H = np.copy(order_H_list[i])
            #reducing into row-echelon form and record column 
            #indices involved in pre-swapping
            swapped_H,record_col_index = self.full_gf2elim(tmp_H) 
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
            jump_point = (code.check_matrix_column-code.k)-4*threshold_sum
            swap_lrb_indicator = np.where(updated_LRB>=jump_point,1,0)
            swap_sum = sum(swap_indicator)
            swap_len_list.append(swap_sum)
            swap_lrb_position_list.append(swap_lrb_indicator)
        return updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list      
    def error_pattern_gen(self,direction,range_list):
        code = GL.get_map('code_parameters')
        def function1_inline(i,value):
            if value:        
                tmp = combinations(range_list[i],value)
            else:
                tmp = [-1]
            return tmp
        itering_list = [ function1_inline(i,value) for i,value in enumerate(direction)]
        combination_join = list(it.product(*itering_list))
        filtered_comb = [list(filter(lambda x:x!=-1,combination_element)) for combination_element in combination_join]
        error_patterns = np.zeros(shape=[len(combination_join),code.k],dtype=int)      
        def function2_inline(i,sequence):
            indices = list(chain.from_iterable(sequence))
            error_patterns[i, indices] = 1
        [ function2_inline(i,sequence) for i, sequence in enumerate(filtered_comb)]    
                      
        return error_patterns 

    def collect_tep(self,decoding_path):
        Convention_path_indicator = GL.get_map('convention_path')
        code = GL.get_map('code_parameters')
        factor_gap = GL.get_map('delimiter')
        #segmentation of MRB
        delimiter1 = code.k//factor_gap
        delimiter2 = 3*delimiter1
        LR_part = range(delimiter1)
        MR_part = range(delimiter1,delimiter2)
        HR_part = range(delimiter2,code.k)
        range_seg = [LR_part,MR_part,HR_part]
        #trim decoding path if requested
        if Convention_path_indicator:          
            valid_length = len(decoding_path)
        else:
            valid_length = GL.get_map('decoding_length')         
        proper_error_pattern_list = [self.error_pattern_gen(decoding_path[j],range_seg) for j in range(valid_length)]
        proper_error_pattern_matrix = tf.concat(proper_error_pattern_list,axis=0)
        return proper_error_pattern_matrix


    def best_estimating(self,order_list):
        order_original_input_list,order_label_list,candidate_list = order_list
        correct_counter = 0
        fail_counter = 0  
        input_size =len(order_label_list)
        order_hard_list1 = [tf.where(order_original_input_list[i]>0,0,1) for i in range(len(order_original_input_list))]
        for i in range(input_size):
            #selection best estimation 
            discrepancy_matrix = tf.cast((candidate_list[i] + order_hard_list1[i])%2,dtype=tf.float32)
            soft_discrepancy_sum = tf.reduce_sum(discrepancy_matrix*abs(tf.expand_dims(order_original_input_list[i],axis=0)),axis=-1)
            #selecting the best estimation  
            estimated_index = tf.argmin(soft_discrepancy_sum)
            #print('optimal index:',estimated_index.numpy())
            cmp_result = (candidate_list[i][estimated_index] == order_label_list[i])
            if (tf.reduce_all(cmp_result)):
                correct_counter += 1
            else:
                fail_counter += 1  
        return correct_counter,fail_counter
    
    def sliding_window_ops(self,fcn,window,global_min,k): 
        early_termination = False
        sorted_window = tf.sort(tf.reshape(window,[1,-1]),direction='ASCENDING')
        expanded_sorted_window = tf.reshape(np.append(sorted_window, float(k)),[1,-1])
        output_prb = tf.squeeze(fcn(expanded_sorted_window))
        win_min = tf.reduce_min(window)
        #if output_prb[0]-output_prb[1] > 0.95: 
        if output_prb[1] > GL.get_map('soft_margin'): 
            early_termination = True        
        global_min = min(global_min,win_min)
        return early_termination,global_min

    def acquire_min(self,error_pattern_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric): #minimum of teps belonging to some order pattern
            estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+initial_mrb)%2)
            estimated_lrb_matrix = tf.matmul(tf.cast(M_matrix,tf.int32),estimated_mrb_matrix)%2        
            codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
            #order pattern min-value
            #selection best estimation 
            discrepancy_matrix = tf.cast((codeword_candidate_matrix + order_hard_original)%2,dtype=tf.float32)    
            row_sums = tf.reduce_sum(discrepancy_matrix*tf.expand_dims(mag_metric,axis=0),axis=-1) 
            min_sum = tf.reduce_min(row_sums)
            return min_sum
    
    def sliding_osd(self,fcn,input_list,inputs,labels,tep_info):
        code = GL.get_map('code_parameters')   
        success_dec = 0
        failure_dec = 0
        sliding_win_width = GL.get_map('sliding_win_width')
        order_list= self.check_matrix_reorder(input_list,inputs,labels)
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = self.identify_mrb(order_H_list)
        input_size = inputs.shape[0]        
        teps_list,acc_block_size = tep_info
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[0][i],updated_index_list[i]) for i in range(input_size)]  
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]                  
        complexity_sum = 0
        windows_sum = 0
        for i in range(input_size):
            print('.',end='')
            #ground truth
            order_hard_original = tf.where(order_original_input_list[i]>0,0,1)
            discrepancy_truth = tf.cast((order_label_list[i] + order_hard_original)%2,dtype=tf.float32)
            mag_metric = abs(order_original_input_list[i])
            discrepancy_sum_truth = tf.reduce_sum(discrepancy_truth*mag_metric)
            M_matrix = updated_M_list[i]  
            #serial processing each codeword            
            order_hard = tf.where(order_input_list[i]>0,0,1)
            initial_mrb = order_hard[-code.k:]
            #initialize the window content
            window = []    
            for error_pattern_matrix in teps_list[:sliding_win_width]:
                min_sum = self.acquire_min(error_pattern_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric)
                window.append(min_sum)  
            global_min = tf.reduce_min(window)
            for k in range(len(teps_list)-sliding_win_width+1):    
                deep_limit = k+sliding_win_width
                if k != 0:
                    #update window content
                    #estimations of codeword candidate
                    error_pattern_matrix = teps_list[sliding_win_width+k-1]
                    min_sum = self.acquire_min(error_pattern_matrix,initial_mrb,M_matrix,order_hard_original,mag_metric)
                    window.append(min_sum) 
                    # Keep the last four elements plus the new one
                    window = window[-sliding_win_width:]
                    if min_sum > global_min:
                        continue
                #tf.print(window)   
                early_termination,global_min = self.sliding_window_ops(fcn,window,global_min,k)
                if early_termination:
                    break   
            window_num = deep_limit - sliding_win_width+1
            #tf.print(window_num)
            complexity_sum += acc_block_size[deep_limit]
            windows_sum += window_num
            if global_min == discrepancy_sum_truth:
                success_dec += 1
            else:
                failure_dec += 1
        return success_dec,failure_dec,windows_sum,complexity_sum
     
    def full_gf2elim(self,M):
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

    def execute_osd(self,input_list,inputs,labels,proper_error_pattern_matrix):
        threshold_indicator = GL.get_map('threshold_indicator')
        threshold_sum = GL.get_map('threshold_sum')
        code = GL.get_map('code_parameters')   
        list_length = GL.get_map('num_iterations')+1
        order_list= self.check_matrix_reorder(input_list,inputs,labels)
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list,swap_len_list,swap_lrb_position_list = self.identify_mrb(order_H_list)
        input_size = inputs.shape[0]

        
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[i][j],updated_index_list[j]) for i in range(list_length) for j in range(input_size)]                

        def function_inline(i):
            print('.',end='')
            #serial processing each codeword
            order_hard = tf.where(order_input_list[i]>0,0,1)
            M_matrix = updated_M_list[i]
            #generate all possible error patterns of mrb            
            error_pattern_matrix = proper_error_pattern_matrix       
            # setting starting point                                              
            initial_mrb = order_hard[-code.k:]
            initial_lrb = tf.reshape(order_hard[:code.check_matrix_column-code.k],[-1,1])
            codeword_lrb = tf.matmul(tf.reshape(initial_mrb,[1,-1]),tf.cast(M_matrix,tf.int32),transpose_b=True)%2
            codeword_candidate_matrix = tf.concat([codeword_lrb,tf.reshape(initial_mrb,[1,-1])],axis=1)
            #estimations of codeword candidate
            estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+initial_mrb)%2)
            estimated_lrb_matrix = tf.matmul(tf.cast(M_matrix,tf.int32),estimated_mrb_matrix)%2        
            #new branch to exclude some low probability test error patterns
            if threshold_indicator:
                swap_lrb_position_vector = tf.reshape(swap_lrb_position_list[i],[-1,1])
                binary_lrb_sum= (initial_lrb+estimated_lrb_matrix)%2
                focus_position_diff_sum = tf.reduce_sum(binary_lrb_sum*swap_lrb_position_vector,axis=0)
                residual_index = tf.squeeze(tf.where(focus_position_diff_sum <= 2*threshold_sum))
                if len(residual_index) < 2:
                    residual_index = [0,1,2,3]
                estimated_mrb_matrix = tf.gather(estimated_mrb_matrix,residual_index,axis=1)
                estimated_lrb_matrix = tf.gather(estimated_lrb_matrix,residual_index,axis=1)
            candidate_size = estimated_mrb_matrix.shape[1]
            #print('candidate_size:',candidate_size)
            codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
            return codeword_candidate_matrix,candidate_size
        tuple_list = [function_inline(i) for i in range(input_size)]
        candidate_list = [tuple_list[i][0] for i in range(input_size)]
        candidate_size_list = [tuple_list[i][1] for i in range(input_size)]
        print('\n')
        candiate_sum_size = tf.reduce_sum(candidate_size_list)      
        return order_original_input_list,order_label_list,candidate_list,candiate_sum_size