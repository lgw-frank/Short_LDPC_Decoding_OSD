
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
from collections import Counter
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
    def stat_pre_osd(self,inputs):
        #initialize of mask
        order_H_list = self.check_matrix_reorder(inputs)
        n_S = GL.get_map('swapping_threshold')
        swap_length_list = []
        initial_index_list = []
        for i in range(inputs.shape[0]):
            # H assumed to be full row rank to obtain its systematic form
            tmp_H = np.copy(order_H_list[i])
            #reducing into row-echelon form and record column 
            #indices involved in pre-swapping
            M,record_col_index = self.medium_gf2elim(tmp_H)  
            initial_index_list.append(record_col_index[n_S])
            #incoporate dependent indices belonged columns
            index_length = len(record_col_index)
            swap_length_list.append(index_length)
        #summmary of failure report
        Demo_result=Counter(swap_length_list)
        Demo_result=dict(Demo_result) 
        print(Demo_result)            
        return Demo_result,initial_index_list            
    def stat_pro_osd(self,inputs):
        #initialize of mask
        code = GL.get_map('code_parameters')
        order_H_list = self.check_matrix_reorder(inputs)
        updated_MRB_list = []
        for i in range(inputs.shape[0]):
            # H assumed to be full row rank to obtain its systematic form
            tmp_H = np.copy(order_H_list[i])
            #reducing into row-echelon form and record column 
            #indices involved in pre-swapping
            M,record_col_index = self.full_gf2elim(tmp_H) 
            index_length = len(record_col_index)
            #update all swapping index
            index_order = np.array(range(code.check_matrix_column))
            for j in range(index_length):
                tmpa = record_col_index[j][0]
                tmpb = record_col_index[j][1]
                index_order[tmpa],index_order[tmpb] =  index_order[tmpb],index_order[tmpa]   
            #udpated mrb indices
            updated_MRB = index_order[-code.k:]
            #print(Updated_MRB)
            updated_MRB_list.append(updated_MRB)         
        return updated_MRB_list 
    
    def identify_mrb(self,order_H_list):
        #initialize of mask
        code = GL.get_map('code_parameters')
        updated_index_list = []
        updated_M_list = []
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
            mrb_swapping_index = tf.argsort(index_order[-code.k:],axis=0,direction='ASCENDING')
            mrb_order = tf.sort(index_order[-code.k:],axis=0,direction='ASCENDING')
            updated_index_order = tf.concat([index_order[:(code.check_matrix_column-code.k)],mrb_order],axis=0)
            updated_M = tf.gather(swapped_H[:,-code.k:],mrb_swapping_index,axis=1)    
            updated_index_list.append(updated_index_order)  
            updated_M_list.append(updated_M)
        return updated_index_list,updated_M_list
    
    def combination_mrb(self,range_start, range_end, level):
        if level:
            initial_range = range(range_start, range_end)
            itering = combinations(initial_range, level)
            Clist = list(itering)
            adding_index = tf.reshape(Clist,shape=[-1,level])
            adding_list = []
            for i in range(len(adding_index[:,])):
                adding_vector = np.isin(initial_range, adding_index[i])
                adding_list.append(adding_vector)
            mrb_operand_matrix = tf.where(tf.reshape(adding_list, shape=[-1,len(initial_range)]),1,0)
        else:
            mrb_operand_matrix = np.zeros(shape=[1,range_end-range_start],dtype=int)
        return mrb_operand_matrix

    def error_pattern_gen(self,direction,range_list):
        code = GL.get_map('code_parameters')
        def function1_inline(i,value):
            if value:
                tmp = combinations(range_list[i],value)
            else:
                tmp = [-1]
            return tmp
        itering_list = [ function1_inline(i,value) for i,value in enumerate(direction)]   
        #combination_join = list(it.product(itering_list[0],itering_list[1],itering_list[2],itering_list[3],itering_list[4],itering_list[5]))
        combination_join = list(it.product(*itering_list))
        filtered_comb = [list(filter(lambda x:x!=-1,combination_element)) for combination_element in combination_join]
        error_patterns = np.zeros(shape=[len(combination_join),code.k],dtype=int)      
        def function2_inline(i,sequence):
            indices = list(chain.from_iterable(sequence))
            error_patterns[i, indices] = 1
        [ function2_inline(i,sequence) for i, sequence in enumerate(filtered_comb)]                       
        return error_patterns 

    def reform_inputs(self,inputs,labels,input_list_complex,updated_index_list): 
        input_list = input_list_complex[0]
        #first time swapping due to magnitude
        list_length = GL.get_map('num_iterations')+1
        lri_p = self.mag_input_gen(inputs)
        order_labels = tf.gather(labels,lri_p,batch_dims=1)
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(labels.shape[0])]
        reconstructed_input_list = [tf.gather(input_list[i::list_length],lri_p,batch_dims=1) for i in range(list_length)]
        #second time swapping due to Gaussian elimination and mrb forming per codeword
        flattened_input_list = [tf.gather(reconstructed_input_list[i][j],updated_index_list[j]) \
                                for i in range(list_length) for j in range(inputs.shape[0])]
        order_hard_list = [tf.where(flattened_input_list[i]>0,0,1) for i in range(len(flattened_input_list))]
        return flattened_input_list,order_hard_list,order_label_list
        
    def tailored_pattern(self,direction,error_patterns_list):
        code = GL.get_map('code_parameters')
        seg_num = GL.get_map('seg_num')
        itering_list = []
        for seg_counter in range(seg_num):
            order = direction[seg_counter]
            tmp = np.arange(len(error_patterns_list[seg_counter][order]))
            itering_list.append(tmp)
        combination_join = list(it.product(*itering_list))
        error_pattern_list = []
        for combination_element in combination_join:
            partial_list = [(error_patterns_list[i][direction[i]][combination_element[i]])  for i in range(seg_num)]
            new_error_pattern = tf.reduce_sum(tf.reshape(partial_list,[-1,code.k]),axis=0)
            error_pattern_list.append(new_error_pattern)
        erro_pattern_matrix = tf.reshape(error_pattern_list,shape=[-1,code.k])   
        return erro_pattern_matrix        
        
    def execute_osd2(self,inputs,labels,direction_matrix,valid_length):
        code = GL.get_map('code_parameters')
        order_H_list,order_inputs,order_labels = self.check_matrix_reorder(inputs,labels)
        updated_index_list,updated_M_list = self.identify_mrb(order_H_list)
        #segmentation of MRB
        k1 = GL.get_map('delimiter')
        k2 = 2*k1
        range0 = range(k1)
        range1 = range(k1,k2)
        range2 = range(k2,code.k)
        range_list = [range0,range1,range2]
        if valid_length < 1:
            print('Error occurred')
            exit(-1)
        candidate_list = []
        for i in range(inputs.shape[0]):
            print('.',end='')
            #serial processing each codeword
            order_input = tf.gather(order_inputs[i],updated_index_list[i])
            order_hard = tf.where(order_input>0,0,1)
            M_matrix = updated_M_list[i]
            #generate all possible error patterns of mrb
            error_pattern_matrix = []
            for j in range(valid_length):   
                direction = direction_matrix[j]
                new_error_patterns = self.error_pattern_gen(direction,range_list)
                if len(new_error_patterns) == 1:
                    error_pattern_matrix = new_error_patterns
                else:
                    error_pattern_matrix = tf.concat([error_pattern_matrix,new_error_patterns],axis=0)
            # setting starting point                                              
            initial_mrb = order_hard[-code.k:]
            codeword_lrb = tf.matmul(tf.reshape(initial_mrb,[1,-1]),tf.cast(M_matrix,tf.int32),transpose_b=True)%2
            codeword_candidate_matrix = tf.concat([codeword_lrb,tf.reshape(initial_mrb,[1,-1])],axis=1)
            #estimations of codeword candidate
            estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+initial_mrb)%2)
            estimated_lrb_matrix = tf.matmul(tf.cast(M_matrix,tf.int32),estimated_mrb_matrix)%2
            codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
            candidate_list.append(codeword_candidate_matrix)
        return candidate_list,updated_index_list
    

    def execute_osd(self,input_list,inputs,labels,direction_matrix,valid_length):
        code = GL.get_map('code_parameters')    
        list_length = GL.get_map('num_iterations')+1
        order_list= self.check_matrix_reorder(input_list,inputs,labels)
        order_H_list,order_inputs,order_original_list,order_labels = order_list
        updated_index_list,updated_M_list = self.identify_mrb(order_H_list)
        #segmentation of MRB
        fixed_gap = GL.get_map('delimiter')
        delimiter1 = fixed_gap
        delimiter2 = 2*fixed_gap+delimiter1
        LR_part = range(delimiter1)
        MR_part = range(delimiter1,delimiter2)
        HR_part = range(delimiter2,code.k)
        range_list = [LR_part,MR_part,HR_part]
        input_size = inputs.shape[0]
        if valid_length < 1:
            print('Error occurred')
            exit(-1)
        order_input_list = [tf.gather(order_inputs[i],updated_index_list[i]) for i in range(input_size)]
        order_label_list = [tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32) for i in range(input_size)]
        order_original_input_list = [tf.gather(order_original_list[i][j],updated_index_list[j]) for i in range(list_length) for j in range(input_size)]
        
        def function3_inline(i):
            #serial processing each codeword
            order_hard = tf.where(order_input_list[i]>0,0,1)
            M_matrix = updated_M_list[i]
            #generate all possible error patterns of mrb
            error_pattern_list = [self.error_pattern_gen(direction_matrix[j],range_list) for j in range(valid_length)]
            error_pattern_matrix = tf.concat(error_pattern_list,axis=0)
            print('error pattern length:',error_pattern_matrix.shape[0])
            # setting starting point                                              
            initial_mrb = order_hard[-code.k:]
            codeword_lrb = tf.matmul(tf.reshape(initial_mrb,[1,-1]),tf.cast(M_matrix,tf.int32),transpose_b=True)%2
            codeword_candidate_matrix = tf.concat([codeword_lrb,tf.reshape(initial_mrb,[1,-1])],axis=1)
            #estimations of codeword candidate
            estimated_mrb_matrix = tf.transpose((tf.cast(error_pattern_matrix,tf.int32)+initial_mrb)%2)
            estimated_lrb_matrix = tf.matmul(tf.cast(M_matrix,tf.int32),estimated_mrb_matrix)%2
            codeword_candidate_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
            return codeword_candidate_matrix
        candidate_list = [function3_inline(i) for i in range(input_size)]
        
        return order_original_input_list,order_label_list,candidate_list

    def execute_osd_4(self,inputs,labels,direction_matrix,valid_length):  
        code = GL.get_map('code_parameters')
        order_H_list,order_inputs,order_labels = self.check_matrix_reorder(inputs,labels)
        updated_index_list,updated_M_list = self.identify_mrb(order_H_list)
        #segmentation of MRB
        k1 = GL.get_map('delimiter')
        k2 = 2*k1
        correct_counter = 0
        fail_counter = 0
        if valid_length < 1:
            print('Error occurred')
            exit(-1)
        for i in range(inputs.shape[0]):
            order_input = tf.gather(order_inputs[i],updated_index_list[i])
            order_hard = tf.where(order_input>0,0,1)
            order_label = tf.cast(tf.gather(order_labels[i],updated_index_list[i]),dtype=tf.int32)
            M_matrix = updated_M_list[i]
            initial_mrb = order_hard[-code.k:]
            codeword_lrb = tf.matmul(tf.reshape(initial_mrb,[1,-1]),M_matrix,transpose_b=True)%2
            codeword_candidate_matrix = tf.concat([codeword_lrb,tf.reshape(initial_mrb,[1,-1])],axis=1)
            counter = 0
            for j in range(valid_length-1):      
                direction = direction_matrix[j+1]
                mrb_operand_matrix1 = self.combination_mrb(0, k1, direction[0])
                mrb_operand_matrix2 = self.combination_mrb(k1, k2, direction[1])
                mrb_operand_matrix3 = self.combination_mrb(k2, code.k, direction[2])
                mrb_shift_list = []
                for t1,t2,t3 in it.product(range(mrb_operand_matrix1.shape[0]), range(mrb_operand_matrix2.shape[0]), \
                    range(mrb_operand_matrix3.shape[0])):
                        mrb_shift = tf.concat([mrb_operand_matrix1[t1],mrb_operand_matrix2[t2],mrb_operand_matrix3[t3]],axis=0)
                        mrb_shift_list.append(mrb_shift)
                        counter += 1
                mrb_shift_matrix = tf.reshape(mrb_shift_list,shape=[-1,code.k])
                estimated_mrb_matrix = tf.transpose((mrb_shift_matrix+initial_mrb)%2)
                estimated_lrb_matrix = tf.matmul(M_matrix,estimated_mrb_matrix)%2
                codeword_matrix = tf.transpose(tf.concat([estimated_lrb_matrix,estimated_mrb_matrix],axis=0))
                codeword_candidate_matrix = tf.concat([codeword_candidate_matrix,codeword_matrix],axis=0)
            #print('counter:',counter)
            #verify codeword is valid
            # update_H = tf.concat([tf.eye(code.check_matrix_row,dtype=tf.int32),M_matrix],axis=1)
            # print(tf.reduce_sum(tf.matmul(update_H,codeword_candidate_matrix,transpose_b=True)%2))
            #selection best estimation
            discrepancy_matrix = tf.cast((codeword_candidate_matrix + order_hard)%2,dtype=tf.float32)
            soft_discrepancy_sum = tf.reduce_sum(abs(discrepancy_matrix*order_input),axis=-1)
            estimated_index = tf.argmin(soft_discrepancy_sum)
            if (tf.reduce_all(codeword_candidate_matrix[estimated_index] == order_label)):
                correct_counter += 1
            else:
                fail_counter += 1
        return correct_counter,fail_counter    
    
    def medium_gf2elim(self,M):
          m,n = M.shape
          i=0
          j=0
          record_col_index = []
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
                  else:
                      record_col_index.append(i)
                      i += 1
                      j +=1                  
                  continue   
          
              aijn = M[i, j:]
              col = np.copy(M[:, j]) #make a copy otherwise M will be directly affected
              col[i] = 0 #avoid xoring pivot row with itself
              flip = np.outer(col, aijn)
              M[:, j:] = M[:, j:] ^ flip
              i += 1
              j +=1
          return M,record_col_index  
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
