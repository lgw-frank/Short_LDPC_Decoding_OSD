# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 01:22:27 2024

@author: lgw
"""
import globalmap as GL
import tensorflow as tf
import numpy as  np
from itertools import combinations
import math

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
    sorted_arrays = sorted(all_arrays, key=lambda x: -x[0]) 
    array_list = [arr[1] for arr in sorted_arrays]
    return array_list
# Function to calculate binomial coefficient
def binomial_coefficient(n, k):
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
        
def generate_teps(order_limit):
    code = GL.get_map('code_parameters')
    array_list = []
    for i in range(order_limit+1):
        array_list_element = generate_binary_arrays(code.k,i)
        array_list.append(array_list_element)
    array_matrix = tf.concat(array_list,axis=0)
    return array_matrix
def query_boundary(order_limit):
    code = GL.get_map('code_parameters')
    boundary_list = []
    counter_pos = 0
    for i in range(order_limit+1):
        shift = binomial_coefficient(code.k,i)
        counter_pos += shift
        boundary_list.append(counter_pos)
    return boundary_list
        
def convention_osd_main(wrapped_input):
    updated_inputs,updated_original_inputs,updated_labels,reduced_G,error_patterns_matrix,boundary_list = wrapped_input
    code = GL.get_map('code_parameters')
    correct_indicator = False
    # setting starting point  
    updated_hard_decisions = tf.where(updated_inputs>0,0,1) 
    updated_labels = tf.cast(tf.reshape(updated_labels,shape=[1,-1]),tf.int32)                                     
    initial_mrb = tf.reshape(updated_hard_decisions[:code.k],[1,-1])
    #estimations of other codeword candidate
    estimated_mrb_matrix = (tf.cast(error_patterns_matrix,tf.int32)+initial_mrb)%2
    codeword_candidates_matrix = tf.matmul(estimated_mrb_matrix,tf.cast(reduced_G,tf.int32))%2  
    discrepancy_matrix = tf.cast((codeword_candidates_matrix + updated_hard_decisions)%2,dtype=tf.float32)
    soft_discrepancy_sum = tf.reduce_sum(discrepancy_matrix*abs(updated_original_inputs),axis=-1)
    #selecting the best estimation  
    estimated_index = tf.argmin(soft_discrepancy_sum)
    #distance_hard = (updated_hard_decisions+codeword_estimate)%2
    #print(np.sum(tf.cast(distance_hard,tf.float32)*abs(updated_inputs)))
    belonged_phase = -1
    cmp_result = (codeword_candidates_matrix[estimated_index:estimated_index+1] == updated_labels)      
    if (tf.reduce_all(cmp_result)):
        #print(f'soft_diff:{soft_discrepancy_sum[estimated_index]:.4f} index:{estimated_index}')
        #decide which order the chosen index belongs to
        for i in range(len(boundary_list)):
            if estimated_index < boundary_list[i]:
                belonged_phase = i   
                break
        correct_indicator = True
    teps_size = codeword_candidates_matrix.shape[0]
    return correct_indicator,teps_size,belonged_phase