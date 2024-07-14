# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
import data_generating as Data_gen
from keras.constraints import non_neg
from tensorflow.keras.layers import Dense     # for the hidden layer
#from distfit import distfit
   
class neural_network_process(tf.keras.Model):
    def __init__(self):
        super(neural_network_process, self).__init__()    
        self.hidden_layer = Dense(4,kernel_constraint= non_neg(),activation="linear",use_bias=False)
        self.output_layer = Dense(1,kernel_constraint= non_neg(),activation="linear",use_bias=False)
    def build(self,input_shape):
        pass 
    def call(self, inputs):
        x = self.hidden_layer(inputs) 
        output= self.output_layer(x)
        return output  
    
class Decoding_model(tf.keras.Model):
    def __init__(self):
        super(Decoding_model,self).__init__()
        self.layer = Decoder_Layer()            
    def call(self,inputs,labels): 
        soft_output_list =  self.layer(inputs,labels)
        fer,ber,undected_count,indices = self.get_eval(soft_output_list,labels)
        buffer = self.collect_failed_output_selective(soft_output_list,labels,indices)
        return fer,ber,undected_count,buffer 
    
    def get_eval(self,soft_output_list,labels):
        code = GL.get_map('code_parameters')
        soft_output = soft_output_list[-1]
        tmp = tf.cast(tf.where(soft_output>0,0,1),tf.int64)
        err_batch = tf.where(tmp == labels,0,1)
        err_bit_sum = tf.reduce_sum(err_batch,axis=-1)
        success_index = tf.where(err_bit_sum==0)
        #exclude the undetected errors
        syndrome = tf.reduce_sum(tf.matmul(tmp,code.H,transpose_b=True)%2,axis=-1)
        qualifed_index = tf.where(syndrome==0)
        #query undetected_index 
        # Find elements in A that are not in B
        not_in_success_index = tf.boolean_mask(qualifed_index, ~tf.reduce_any(tf.equal(qualifed_index[:, tf.newaxis], success_index), axis=1))
        if not_in_success_index:
            print("Undetected Elements:", not_in_success_index.numpy())
        index = tf.where(syndrome!=0)
        FER = 1-len(success_index)/labels.shape[0]
        BER = tf.reduce_sum(err_bit_sum)/(labels.shape[0]*labels.shape[1])
        return FER, BER,len(not_in_success_index),index   
    def collect_failed_output_selective(self,soft_output_list,labels,index):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])     
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels 
    #squashing data appropriately
    def postprocess_failure_cases(self,buffer):
        #collecting erroneous decoding info
        buffer_inputs = [j for i in buffer[0] for j in i]
        buffer_labels = [j for i in buffer[1] for j in i]
        return buffer_inputs,buffer_labels

class Decoder_Layer(tf.keras.layers.Layer):
    def __init__(self,initial_value = -0.048):
        super().__init__()
        self.decoder_type = GL.get_map('selected_decoder_type')
        self.num_iterations = GL.get_map('num_iterations')
        self.code = GL.get_map('code_parameters')
        self.feature_len = self.code.max_chk_degree-1
        self.initials = initial_value
    #V:vertical H:Horizontal D:dynamic S:Static  /  VSSL: Vertical Static/Dynamic Shared Layer
    def build(self, input_shape):       
        if GL.get_map('selected_decoder_type') in ['NMS-1']:
            self.shared_check_weight = self.add_weight(name='decoder_check_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
        if GL.get_map('selected_decoder_type') in ['NMS-2']:
           self.shared_bit_weight = self.add_weight(name='decoder_bit_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
           self.shared_check_weight = self.add_weight(name='decoder_check_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
        if GL.get_map('selected_decoder_type') in ['NMS-3']:
           self.shared_bit_weight1 = self.add_weight(name='decoder_bit_normalized factor1',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
           self.shared_bit_weight2 = self.add_weight(name='decoder_bit_normalized factor2',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
           self.shared_check_weight = self.add_weight(name='decoder_check_normalized factor',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
        if GL.get_map('selected_decoder_type') in ['NMS-r']:
           self.shared_bit_weight1 = self.add_weight(name='decoder_bit_normalized factor1',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
           self.shared_bit_weight2 = self.add_weight(name='decoder_bit_normalized factor2',shape=[1],trainable=True,initializer=tf.keras.initializers.Constant(self.initials ))    
           x_train = tf.random.normal(shape=(1,self.feature_len),dtype=tf.float32)
           tmp_model = neural_network_process()
           _ = tmp_model(x_train)
           self.nn = tmp_model   
         # Code for model call (handles inputs and returns outputs)
    def call(self,soft_input,labels):    
        bp_result = self.belief_propagation_op(soft_input,labels)
        return bp_result[4]
       
                
# builds a belief propagation TF graph

    def belief_propagation_op(self,soft_input, labels):
        soft_output_list = [soft_input]
        init_value = tf.zeros(soft_input.shape,dtype=tf.float32)
        for _ in range(self.num_iterations):         
            soft_output_list.append(init_value)
        return tf.while_loop(
            self.continue_condition, # iteration < max iteration?
            self.belief_propagation_iteration, # compute messages for this iteration
            loop_vars = [
                soft_input, # soft input for this iteration
                labels,
                0, # iteration number
                tf.zeros([soft_input.shape[0],self.code.check_matrix_row,self.code.check_matrix_column],dtype=tf.float32)    ,# cv_matrix
                soft_output_list,  # soft output for this iteration
            ]
            )
            
    # compute messages from variable nodes to check nodes
    def compute_vc(self,cv_matrix, soft_input,iteration):
        normalized_tensor = 1.0
        check_matrix_H = tf.cast(self.code.H,tf.float32)
        if GL.get_map('selected_decoder_type') in ['NMS-2']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight)
        if GL.get_map('selected_decoder_type') in ['NMS-3','NMS-r']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight1)            
        soft_input_weighted = soft_input*normalized_tensor           
        temp = tf.reduce_sum(cv_matrix,1)                        
        temp = temp+soft_input_weighted
        temp = tf.expand_dims(temp,1)
        temp = temp*check_matrix_H
        vc_matrix = temp - cv_matrix
        return vc_matrix  
    # compute messages from check nodes to variable nodes
    def compute_cv(self,vc_matrix,iteration):
        if GL.get_map('selected_decoder_type') in ['NMS-r']:
            cv_matrix = self.compute_cv1(vc_matrix,iteration)
        else:
            cv_matrix = self.compute_cv2(vc_matrix,iteration)
        return cv_matrix
        
    def compute_cv1(self,vc_matrix,iteration):
        #feature_len = GL.get_map('feature_len')
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp = tf.reduce_prod(vc_matrix_sign,axis=-1)
        temp = tf.expand_dims(temp,axis=-1)
        transition_sign_matrix = temp*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #oprations on magnitudes
        expanded_H = tf.tile(tf.expand_dims(check_matrix_H,axis=0),(vc_matrix.shape[0],1,1))
        reformed_H = tf.reshape(expanded_H,[-1,self.code.max_chk_degree])
        index_matrix = tf.where(reformed_H)
        nonzero_matrix =tf.reshape(tf.math.abs(vc_matrix)[expanded_H != 0],[-1,self.code.max_chk_degree])
        a_eye = tf.eye(self.code.max_chk_degree)
        ones = tf.ones(self.code.max_chk_degree)
        def expanded_row(i):
            test_data_t = tf.transpose(nonzero_matrix)
            mask_row = ones - a_eye[i]    
            left_matrix = tf.transpose(tf.boolean_mask(test_data_t,mask_row))  
            stacked_item = tf.sort(left_matrix,axis=-1,direction='ASCENDING')
            list_row = self.nn(stacked_item)
            return list_row
        list_d = list(map(expanded_row,range(self.code.max_chk_degree)))
        list_d = tf.concat(list_d,axis=1)
        cv_sparse_flattened = tf.squeeze(tf.reshape(list_d,[-1,1]))
        sp_input = tf.SparseTensor(dense_shape=reformed_H.shape,values=cv_sparse_flattened,indices = index_matrix)
        cv_matrix_dense = tf.sparse.to_dense(sp_input)
        cv_matrix = tf.stop_gradient(result_sign_matrix)*tf.reshape(cv_matrix_dense,shape=vc_matrix.shape)  
        return cv_matrix  
    # compute messages from check nodes to variable nodes
    def compute_cv2(self,vc_matrix,iteration):
        normalized_tensor = 1.0
        check_matrix_H = self.code.H
        #operands sign processing 
        supplement_matrix = tf.cast(1-check_matrix_H,dtype=tf.float32)
        supplement_matrix = tf.expand_dims(supplement_matrix,0)
        sign_info = supplement_matrix + vc_matrix
        vc_matrix_sign = tf.sign(sign_info)
        temp1 = tf.reduce_prod(vc_matrix_sign,2)
        temp1 = tf.expand_dims(temp1,2)
        transition_sign_matrix = temp1*check_matrix_H
        result_sign_matrix = transition_sign_matrix*vc_matrix_sign 
        #preprocessing data for later calling of top k=2 largest items
        back_matrix = tf.where(check_matrix_H==0,-1e30-1,0.)
        back_matrix = tf.expand_dims(back_matrix,0)
        vc_matrix_abs = tf.abs(vc_matrix)
        vc_matrix_abs_clip = tf.clip_by_value(vc_matrix_abs, 0, 1e30)
        vc_matrix_abs_minus = -tf.abs(vc_matrix_abs_clip)
        vc_decision_matrix = vc_matrix_abs_minus+back_matrix
        min_submin_info = tf.nn.top_k(vc_decision_matrix,k=2)
        min_column_matrix = -min_submin_info[0][:,:,0]
        min_column_matrix = tf.expand_dims(min_column_matrix,2)
        min_column_matrix = min_column_matrix*check_matrix_H
        second_column_matrix = -min_submin_info[0][:,:,1]
        second_column_matrix = tf.expand_dims(second_column_matrix,2)
        second_column_matrix = second_column_matrix*check_matrix_H  
        result_matrix = tf.where(vc_matrix_abs_clip>min_column_matrix,min_column_matrix,second_column_matrix)
        if GL.get_map('selected_decoder_type') in ['NMS-1','NMS-2','NMS-3']:
            normalized_tensor = tf.nn.softplus(self.shared_check_weight)       
        cv_matrix = normalized_tensor *result_matrix*tf.stop_gradient(result_sign_matrix)         
        return cv_matrix
    
    #common sense definition taking into account all bits of codewords         
    # def calculation_loss(self,soft_output,labels,loss):
    #      #cross entroy
    #     labels = tf.cast(labels,tf.float32)
    #     CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
    #     return CE_loss+loss

    #combine messages to get posterior LLRs
    def marginalize(self,cv_matrix, soft_input,iteration,soft_output_list):
        normalized_tensor = 1.0
        if GL.get_map('selected_decoder_type') in ['NMS-2']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight)
        if GL.get_map('selected_decoder_type') in ['NMS-3','NMS-r']:
            normalized_tensor = tf.nn.softplus(self.shared_bit_weight2)
        temp = tf.reduce_sum(cv_matrix,1)
        soft_output = temp+normalized_tensor*soft_input
        soft_output_list[iteration+1] = soft_output
   
    def continue_condition(self,soft_input,labels,iteration, cv_matrix,soft_output_list):
        condition = (iteration < self.num_iterations) 
        return condition
    
    def belief_propagation_iteration(self,soft_input, labels, iteration, cv_matrix,soft_output_list):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix, soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration)      
        # get output for this iteration
        self.marginalize(cv_matrix, soft_input,iteration,soft_output_list) 
        iteration += 1   
        return soft_input, labels, iteration, cv_matrix,soft_output_list

#common sense definition taking into account all bits of codewords         
def calculation_loss(soft_output,labels):
     #cross entroy
    labels = tf.cast(labels,tf.float32)
    CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
    return CE_loss
#save modified data for postprocessing
def save_decoded_data(updated_buffer,file_dir,snr,log_filename,list_length):
    stacked_buffer_info = tf.stack(updated_buffer[0])
    stacked_buffer_label = tf.stack(updated_buffer[1])
    CE_loss_list = []
    for i in range(list_length):
        cross_entropy_bits = stacked_buffer_info[i::list_length]
        cross_entropy_labels = stacked_buffer_label[i::list_length]
        CE_loss = calculation_loss(cross_entropy_bits,cross_entropy_labels)
        CE_loss_list.append((CE_loss/cross_entropy_bits.shape[0]).numpy())
    print(CE_loss_list)
    # Notation text
    notation_text = "CE list:"
    with open(log_filename,'a+') as f:
      f.write(str(cross_entropy_bits.shape[0])+'tested:\n')
      f.write("# " + notation_text + '\n')
      f.write(' '.join(map(str, CE_loss_list)) + '\n')  # Join list elements with space as delimiter
    print("%.4f tested\nCE_list:%s"%(cross_entropy_bits.shape[0],str(CE_loss_list)))
    print("Data for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=file_dir)    
    snr = str(round(snr,2))
    print('For '+ snr +"dB:Data storing finished!")
