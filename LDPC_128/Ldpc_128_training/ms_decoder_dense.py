# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 10:41:33 2022

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
from tensorflow.keras.layers import Dense     # for the hidden layer
import data_generating as Data_gen
from keras.constraints import non_neg
    
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
        super().__init__()
        self.layer = Decoder_Layer()
    def call(self,inputs): 
        bp_result = self.layer(inputs)
        return bp_result 
    
    def collect_failed_input_output(self,soft_output_list,labels,index):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(list_length):
                buffer_inputs.append(soft_output_list[j][i])    
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels     
    def collect_failed_input_output2(self,soft_output_list,labels,index):
        list_length = self.layer.num_iterations + 1
        buffer_inputs = []
        buffer_labels = []
        indices = tf.squeeze(index,1).numpy()
        for i in indices:
            for j in range(1,list_length):
                buffer_inputs.append(soft_output_list[j][i]-soft_output_list[0][i])#save the difference     
                buffer_labels.append(labels[i])
        return buffer_inputs,buffer_labels 
    def get_eval(self,soft_output_list, labels):
        soft_output = soft_output_list[-1]
        tmp = tf.cast((soft_output < 0),tf.bool)
        label_bool = tf.cast(labels, tf.bool)
        err_batch = tf.math.logical_xor(tmp, label_bool)
        FER_data = tf.reduce_any(err_batch,1)
        index = tf.where(FER_data)
        BER_data = tf.reduce_sum(tf.cast(err_batch, tf.int64))
        FER = tf.math.count_nonzero(FER_data)/soft_output.shape[0]
        BER = BER_data/(soft_output.shape[0]*soft_output.shape[1])
        return FER, BER,index    

 
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
    def call(self,inputs):
        soft_input = inputs[0]
        labels = inputs[1]     
        bp_result = self.belief_propagation_op(soft_input,labels)
        soft_output_list,label,loss = bp_result[5],bp_result[1],bp_result[4]
        return soft_output_list,label,loss
       
                
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
                0.,# loss
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
    def calculation_loss(self,soft_output,labels,loss):
         #cross entroy
        labels = tf.cast(labels,tf.float32)
        CE_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-soft_output, labels=labels)) 
        return CE_loss+loss

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
        return soft_output  
   
    def continue_condition(self,soft_input,labels,iteration, cv_matrix, loss,soft_output_list):
        condition = (iteration < self.num_iterations) 
        return condition
    
    def belief_propagation_iteration(self,soft_input, labels, iteration, cv_matrix, loss,soft_output_list):
        # compute vc
        vc_matrix = self.compute_vc(cv_matrix, soft_input,iteration)
        # compute cv
        cv_matrix = self.compute_cv(vc_matrix,iteration)      
        # get output for this iteration
        soft_output = self.marginalize(cv_matrix, soft_input,iteration,soft_output_list) 
        iteration += 1   
        loss = self.calculation_loss(soft_output,labels,loss)
        return soft_input, labels, iteration, cv_matrix, loss,soft_output_list

def retore_saved_model(restore_ckpts_dir,restore_step,ckpt_nm):
    print("Ready to restore a saved latest or designated model!")
    ckpt = tf.train.get_checkpoint_state(restore_ckpts_dir)
    if ckpt and ckpt.model_checkpoint_path: # ckpt.model_checkpoint_path means the latest ckpt
      if restore_step == 'latest':
        ckpt_f = tf.train.latest_checkpoint(restore_ckpts_dir)
        start_step = int(ckpt_f.split('-')[-1]) + 1
      else:
        ckpt_f = restore_ckpts_dir+ckpt_nm+'-'+restore_step
        start_step = int(restore_step)+1
      print('Loading wgt file: '+ ckpt_f)   
    else:
      print('Error, no qualified file found')
    return start_step,ckpt_f
#save modified data for postprocessing
def save_decoded_data(buffer_inputs,buffer_labels,file_dir):
    #code = GL.get_map('code_parameters')
    stacked_buffer_info = tf.stack(buffer_inputs)
    stacked_buffer_label = tf.stack(buffer_labels)
    print(" Data for retraining  with %d cases to be stored " % stacked_buffer_info.shape[0])
    data = (stacked_buffer_info.numpy(),stacked_buffer_label.numpy())
    Data_gen.make_tfrecord(data, out_filename=file_dir)    
    print("Data storing finished!")

#postprocessing after first stage training
def postprocess_training(Model,iterator):
    #collecting erroneous decoding info
    buffer_inputs = []
    buffer_labels = []
    #query of size of input feedings
    input_list = list(iterator.as_numpy_iterator())
    num_counter = len(input_list) 
    for i in range(num_counter):
        if not (i+1) % 100:
            print("Total ",i+1," batches are processed!")
        inputs = input_list[i]
        soft_output_list,label,_ = Model(inputs)
        _,_,indices= Model.get_eval(soft_output_list,label)
        buffer_inputs_tmp,buffer_labels_tmp = Model.collect_failed_input_output(soft_output_list,label,indices)   
        buffer_inputs.append( buffer_inputs_tmp)
        buffer_labels.append(buffer_labels_tmp)
    buffer_inputs = [j for i in buffer_inputs for j in i]
    buffer_labels = [j for i in buffer_labels for j in i]
    return buffer_inputs,buffer_labels
    
#main training process
def training_block(start_info,Model,optimizer,exponential_decay,selected_ds,log_info,restore_info):
    #query of size of input feedings
    input_list = list(selected_ds.as_numpy_iterator())
    num_counter = len(input_list) 
    start_step,multiplier,train_steps = start_info
    summary_writer,manager_current = log_info
    ckpts_dir,ckpt_nm,ckpts_dir_par,restore_step= restore_info
    batch_index = start_step
    termination_indicator = False
    start_step = start_step%num_counter
    while True:
        for i in range(start_step,num_counter):
                loss_mini_total = 0.0
                fer_mini_total = 0.0
                ber_mini_total = 0.0
                grads_flag = True
                for _ in range(multiplier):
                    with tf.GradientTape() as tape:
                        inputs = input_list[i]
                        soft_output_list,label,loss = Model(inputs)
                        fer,ber,_= Model.get_eval(soft_output_list,label)
                        loss_mini_total = loss_mini_total+loss
                        fer_mini_total = fer_mini_total+fer
                        ber_mini_total= ber_mini_total+ber
                        grads = tape.gradient(loss,Model.variables)
                        if grads_flag:
                            grads_mini_total = grads
                            grads_flag = False                        
                        else:
                            grads_mini_total = grads_mini_total+ grads
                my_new_list = [grad_iterator/multiplier for grad_iterator in grads_mini_total if grad_iterator is not None]
                grads_and_vars=zip(my_new_list, Model.variables)
                capped_gradients = [(tf.clip_by_norm(grad,5), var) for grad, var in grads_and_vars if grad is not None]
                #capped_gradients = [(tf.clip_by_value(grad,-1,1), var) for grad, var in grads_and_vars if grad is not None]
                optimizer.apply_gradients(capped_gradients)
                with summary_writer.as_default():                               # the logger to be used
                  tf.summary.scalar("loss", loss_mini_total/multiplier, step=batch_index)
                  tf.summary.scalar("FER", fer_mini_total/multiplier, step=batch_index)  # you can also add other indicators below
                  tf.summary.scalar("BER", ber_mini_total/multiplier, step=batch_index)  # you can also add other indicators below     
                # log to stdout 
                print_interval = GL.get_map('print_interval')
                record_interval = GL.get_map('record_interval')
                batch_index = batch_index+1
                if batch_index % print_interval== 0 or batch_index == train_steps-1: 
                    print("Step%4d: lr:%.4f Loss:%.4f FER:%.4f BER:%.4f"%\
                    (batch_index,exponential_decay(batch_index),loss_mini_total/multiplier, fer_mini_total.numpy()/multiplier, ber_mini_total.numpy()/multiplier) ) 
                  #record best ber/fer parameters in files       
                    manager_current.save(checkpoint_number=batch_index)
                if batch_index % record_interval == 0:
                    print("For all layers at the %4d-th step:"%batch_index)
                    for variable in Model.variables:
                        #print(variable.name+' '+str(variable.numpy()))  
                        print(str(variable.numpy()))  
                    with open(ckpts_dir_par+'values.txt','a+') as f:
                        f.write("For all layers at the %4d-th step:\n"%batch_index)
                        for variable in Model.variables:
                            f.write(variable.name+' '+str(variable.numpy())) 
                        f.write('\n')  
                if batch_index >=min(train_steps,GL.get_map('termination_step')):
                    termination_indicator = True
                    break
                if batch_index%num_counter==0:
                    start_step=0
        if termination_indicator:
            break
    print("Final selected parameters:")
    for weight in  Model.layer.get_weights():
      print(weight)
    return Model          