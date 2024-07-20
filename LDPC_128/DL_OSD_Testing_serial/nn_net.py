# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:54:10 2023

@author: Administrator
"""
import tensorflow as tf
import globalmap as GL
from tensorflow.keras import  layers
from tensorflow import keras 
from tensorflow.keras.layers import Input, Dense, Dropout, Attention,Layer,PReLU
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential 
from tensorflow.keras import regularizers
import os   
class TransformerLayer(Layer):
    def __init__(self, units, num_heads, dropout_rate=0.1):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(units, num_heads)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)  # Adding layer normalization here
        self.dense1 = Dense(units, activation='relu')
        self.dense2 = Dense(units)
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)  # Adding layer normalization here

    def call(self, inputs):
        attention_output = self.multi_head_attention(inputs)
        attention_output = self.dropout1(attention_output)
        attention_output = self.norm1(inputs + attention_output)

        feed_forward_output = self.dense2(self.dropout2(self.dense1(attention_output)))
        output = self.norm2(attention_output + feed_forward_output)
        return output
# Example usage
# input_data = tf.random.uniform((32, 10, 64))  # Example input data: 32 samples, 10 timesteps, 64 features
# transformer_layer = TransformerLayer(units=64, num_heads=8)
# output = transformer_layer(input_data)
class MultiHeadAttention(Layer):
    def __init__(self, units, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.units = units
        self.num_heads = num_heads
        self.depth = units // num_heads
        self.attention_heads = [SelfAttention(units=self.depth) for _ in range(num_heads)]
        self.dense = Dense(units)
    
    def call(self, inputs):
        attention_outputs = [attention(inputs) for attention in self.attention_heads]
        concatenated_outputs = tf.concat(attention_outputs, axis=-1)
        projected_outputs = self.dense(concatenated_outputs)
        return projected_outputs

# # Example usage
# input_data = tf.random.uniform((32, 10, 64))  # Example input data: 32 samples, 10 timesteps, 64 features
# multi_head_attention_layer = MultiHeadAttention(units=64, num_heads=8)
# output = multi_head_attention_layer(input_data)

class SelfAttention(Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
        self.W_query = Dense(units)
        self.W_key = Dense(units)
        self.W_value = Dense(units)
        self.dropout = Dropout(0.2)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, inputs):
        query = self.W_query(inputs)
        key = self.W_key(inputs)
        value = self.W_value(inputs)
        
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores_scaled = attention_scores / tf.sqrt(float(self.units))
        attention_weights = self.softmax(attention_scores_scaled)
        
        context_vector = tf.matmul(attention_weights, value)
        return context_vector


class TemperatureLayer(layers.Layer):
    def __init__(self):
        super(TemperatureLayer, self).__init__()
    def build(self, input_shape): 
        initializer = tf.keras.initializers.Constant(1.0) 
        self.temperature = self.add_weight(name = 'Temperature', 
        shape = [1], 
        initializer = initializer, trainable = True) 
        super(TemperatureLayer, self).build(input_shape)
     
    def call(self, inputs):
        return inputs*self.temperature

class Predict_outlier(tf.keras.Model):
    def __init__(self, sliding_win_width):
        super(Predict_outlier, self).__init__()
        self.input_width = sliding_win_width
        self.dense1 = layers.Dense(2 * self.input_width, kernel_initializer = 'normal',\
             use_bias=False, activation='relu', input_shape=(self.input_width,),kernel_regularizer=regularizers.l2(0.01),name='1st_Layer')  # Adding L2 regularization
        self.dense2 = layers.Dense(self.input_width, kernel_initializer = 'normal',use_bias=False, activation='relu',\
                        kernel_regularizer=regularizers.l2(0.01),name='2nd_Layer')
        self.dense3 = layers.Dense(2,use_bias=False,activation='softmax',name='Output')
        self.temperature_layer = TemperatureLayer()
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        outputs = self.dense3(self.temperature_layer(x))
        return outputs
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.input_width,), dtype='float32', name='Input')
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(self.temperature_layer(x))
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.summary()
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'predict_outlier_'+str(self.input_width)+'.h5'
        # Save the model in HDF5 format
        model.save(dot_img_file)     
        #tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        #return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
    # def print_paras_summary(self):       
    #     model = Sequential() 
    #     model.add(self.dense1(32, input_shape = (16,))) 
    #     model.add(self.dense2(8, activation = 'softmax')) 
    #     model.summary()
    
class Predict_outlier_light(tf.keras.Model):
    def __init__(self, sliding_win_width):
        super(Predict_outlier_light, self).__init__()
        self.input_width = sliding_win_width+1
        self.dense1 = layers.Dense(self.input_width, kernel_initializer = 'normal',\
             use_bias=False, activation='linear', input_shape=(self.input_width,),kernel_regularizer=regularizers.l2(0.01),name='1st_Layer')  # Adding L2 regularization
        #self.dropout = tf.keras.layers.Dropout(0.5)  # Define dropout layer
        self.dense2 = layers.Dense(2,use_bias=False, activation='softmax',name='Output')
        #self.temperature_layer = TemperatureLayer()
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.dropout(x)
        outputs = self.dense2(x)
        return outputs
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.input_width,), dtype='float32', name='Input')
        x = self.dense1(inputs) 
        outputs = self.dense2(self.temperature_layer(x))
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        model.summary()
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'predict_outlier_'+str(self.input_width)+'.h5'
        # Save the model in HDF5 format
        model.save(dot_img_file)     
        #tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        #return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
    # def print_paras_summary(self):       
    #     model = Sequential() 
    #     model.add(self.dense1(32, input_shape = (16,))) 
    #     model.add(self.dense2(8, activation = 'softmax')) 
    #     model.summary()
        
class conv_bitwise(keras.Model):
    def __init__(self):
        super(conv_bitwise, self).__init__()   
        code = GL.get_map('code_parameters')
        list_length = GL.get_map('num_iterations')+1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.H = code.H
        self.activation = tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25))
        #self.attention_layer = SelfAttention(units=32)
    def build(self,input_shape):
        self.cnv_one = layers.Conv1D(filters=8,kernel_size=3,strides=1,\
                                     padding="valid",activation='linear',use_bias=False,name='1st_layer')
        self.cnv_two = layers.Conv1D(4,3,1,padding="valid",activation='linear',use_bias=False,name='2nd_layer')
        self.cnv_three = layers.Conv1D(2,3,1,padding="valid",activation='linear',use_bias=False,name='3rd_layer')  
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(1,activation='linear', name='Dense')
    def call(self, inputs):
        x = self.cnv_one(inputs) 
        x = self.cnv_two(x)
        x = self.cnv_three(x)
        x = self.flatten(x)
        outputs = tf.reshape(self.dense(x),[-1,self.n_dims])
        return outputs  
    def preprocessing_inputs(self,input_slice):
        original_input = input_slice[0]
        original_label = input_slice[1]
        file_input_data = tf.reshape(original_input,[-1,self.list_length,self.n_dims])   
        distortred_inputs = tf.transpose(file_input_data,perm=[0,2,1])
        labels= original_label[0::self.list_length]
        #inputs = original_input[self.list_length-1::self.list_length]
        inputs = original_input[0::self.list_length] #inital received message
        #preprocessing of input data
        squashed_inputs = tf.reshape(distortred_inputs,[-1,self.list_length,1])
        return squashed_inputs,inputs,labels 
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        #inputs = tf.reshape(inputs,[-1,self.list_length,1])
        x = self.cnv_one(inputs) 
        x =self.cnv_two(x)
        x = self.cnv_three(x)
        x = self.flatten(x)
        outputs = tf.reshape(self.dense(x),[-1,self.n_dims])
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy()) 

# Define your model
# Assuming the data is organized in X_train, y_train, X_val, y_val
# y_train and y_val should be one-hot encoded if using softmax        
class PredictPhase0(keras.Model):
    def __init__(self, decoding_length, practical_capacity):
        super(PredictPhase0, self).__init__()
        self.width = 2 * decoding_length - 1
        self.output_width = practical_capacity  # increment for the failure case
        self.temperature = 5.
        # First sublayer with linear activation
        self.dense1 = layers.Dense(2*self.output_width, use_bias=False,activation='linear',input_shape=(self.width,))
        self.dense_2 = layers.Dense(self.output_width, use_bias=False, activation='linear')
        # Second sublayer with softmax activation
        self.dense_3 = layers.Softmax()
    def call(self, inputs):
        #x = Attention()([inputs, inputs])
        x = self.dense1(inputs)
        #x = Dropout(0.2)(x)
        x = self.dense_2(x)/self.temperature
        #x = self.dense_2(x)
        outputs = self.dense_3(x)
        return outputs
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs  
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
        
        
class PredictPhase(keras.Model):
    def __init__(self, decoding_length, practical_capacity):
        super(PredictPhase, self).__init__()
        self.width = 2 * decoding_length - 1
        self.output_width = practical_capacity  # increment for the failure case
        self.temperature = 10.
        self.dense1 = layers.Dense(self.output_width, use_bias=False,activation='linear',input_shape=(self.width,))
        #self.dense2 = layers.Dense(self.output_width, use_bias=False, activation='linear')
        self.dense3 = layers.Dense(self.output_width, activation='softmax')
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.dense2(x)
        outputs = self.dense3(x / self.temperature)
        return outputs
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs  
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
        
class PredictPhase2(keras.Model):
    def __init__(self, decoding_length, practical_capacity):
        super(PredictPhase2, self).__init__()
        self.width = 2 * decoding_length - 1
        self.output_width = practical_capacity  # increment for the failure case
        self.temperature = 100.
        self.dense1 = layers.Dense(4*self.output_width, use_bias=False,activation='linear',input_shape=(self.width,))
        #self.dense2 = layers.Dense(2*self.output_width, activation='linear',kernel_initializer=tf.keras.initializers.HeNormal())
        #self.dense3 = layers.Dense(self.output_width, kernel_initializer=tf.keras.initializers.HeNormal(),activation='softmax')  # Output layer for softmax probabilities
        # First sublayer with linear activation
        self.dense_31 = layers.Dense(self.output_width, use_bias=False, activation='linear')
        # Second sublayer with softmax activation
        self.dense_32 = layers.Softmax()
    def call(self, inputs):
        x = self.dense1(inputs)
        #x = self.dense2(x)
        x = self.dense_31(x)/self.temperature
        outputs = self.dense_32(x)
        #outputs = self.dense3(x)
        return outputs
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs  
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
        
class PredictPhase3(keras.Model):
    def __init__(self,decoding_length,practical_capacity):
        super(PredictPhase3, self).__init__()   
        self.width = 2*decoding_length-1
        self.output_width = practical_capacity #increment for the failure case
    def build(self,input_shape):
        self.dense1 = layers.Dense(4, activation='linear', input_shape=(self.width,))
        self.dense2 = layers.Dense(2, activation='linear', input_shape=(self.width,))
        self.dense3 = layers.Dense(self.output_width, activation='softmax')  # Output layer for softmax probabilities
    def call(self, inputs):
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs   
    def print_model(self):
        inputs = tf.keras.Input(shape=(self.list_length,1,), dtype='float32', name='input')
        x = self.dense1(inputs) 
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs  
        #x= tf.squeeze(x,axis=-1)   
        #output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
        #outputs = tf.reshape(x,shape=[-1,self.n_dims])
        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
        fig_dir = './figs/'
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        dot_img_file = fig_dir+'model_cnn_'+str(self.n_dims)+'.png'
        tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True) 
        return model
        # query total number of trainable parameters
    def obtain_paras(self):
        total_params = tf.reduce_sum([tf.reduce_prod(var.shape) for var in self.trainable_variables])
        print('Total params:', total_params.numpy())
       
class rnn_one(keras.Model):
    def __init__(self):
        super(rnn_one, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.batch_size = GL.get_map('unit_batch_size')
        self.lstm_layer1 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=True)
        self.lstm_layer2 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=False)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)        
        #self.lstm1 =  keras.layers.Bidirectional(self.lstm_layer1,merge_mode='concat')
        #self.lstm_layer2 = keras.layers.LSTM(int(2*self.n_dims),activation="elu",use_bias=False,dropout=0.0,return_sequences=True)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)
        self.simple_rnn1 = keras.layers.SimpleRNN(8*self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=True)
        #self.simple_rnn2 = keras.layers.SimpleRNN(2*n,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_rnn3 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        #self.simple_gru1 = keras.layers.GRU(int(4*n),activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_gru1 = keras.layers.GRU(self.n_dims,activation="linear",return_sequences=True,use_bias=False)
        self.simple_gru2 = keras.layers.GRU(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        self.output_layer = keras.layers.Dense(self.n_dims,activation="linear",use_bias=False)
    
    def call(self, inputs):
        x = self.simple_gru1(inputs) 
        x = self.simple_gru2(x)
        outputs= self.output_layer(x)
        return   outputs
    def preprocessing_inputs(self,input_slice):
        reformed_inputs = tf.reshape(input_slice[0],shape=[-1,self.list_length,self.n_dims])
        inputs = input_slice[0][self.list_length-1::self.list_length,:]
        labels = input_slice[1][self.list_length-1::self.list_length,:]
        return reformed_inputs,inputs,labels
    
class rnn_two(keras.Model):
    def __init__(self):
        super(rnn_two, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.batch_size = GL.get_map('unit_batch_size')
        self.lstm_layer1 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=True)
        self.lstm_layer2 = keras.layers.LSTM(int(self.n_dims),use_bias=True,dropout=0.0,return_sequences=False)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)        
        #self.lstm1 =  keras.layers.Bidirectional(self.lstm_layer1,merge_mode='concat')
        #self.lstm_layer2 = keras.layers.LSTM(int(2*self.n_dims),activation="elu",use_bias=False,dropout=0.0,return_sequences=True)
        #self.lstm2 =  keras.layers.Bidirectional(self.lstm_layer2)

        
        self.simple_rnn1 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_rnn2 = keras.layers.SimpleRNN(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        #self.simple_gru1 = keras.layers.GRU(int(4*n),activation="linear",dropout=0.0,return_sequences=True,use_bias=False)
        self.simple_gru1 = keras.layers.GRU(self.n_dims,activation="linear",return_sequences=True,use_bias=False)
        self.simple_gru2 = keras.layers.GRU(self.n_dims,activation="linear",dropout=0.0,return_sequences=False,use_bias=False)
        self.output_layer = keras.layers.Dense(self.n_dims,activation="linear",use_bias=False)
    
    def call(self, inputs):
        x = self.simple_rnn1(inputs) 
        x = self.simple_rnn2(x)
        outputs= self.output_layer(x)
        return   outputs
    def preprocessing_inputs(self,input_slice):
        reformed_inputs = tf.reshape(input_slice[0],shape=[-1,self.list_length,self.n_dims])
        inputs = input_slice[0][self.list_length-1::self.list_length,:]
        labels = input_slice[1][self.list_length-1::self.list_length,:]
        return reformed_inputs,inputs,labels
                         
class rnn_three(keras.Model):
    def __init__(self):
        super(rnn_three, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.lstm_layer = keras.layers.LSTM(7,use_bias=True,dropout=0.0,return_sequences=True,unroll=True)   
        self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)
        self.rnn_2nd = model_rnn_2nd()
    def build(self,input_shape):
        self.coefficient_list = []
        for i in range(self.list_length):
            self.coefficient_list.append(self.add_weight(name='normalized factor'+str(i),shape=[],trainable=True))    
    def call(self, trim_info_list):
        outputs_list = []
        batch_size = trim_info_list[0].shape[0]//self.list_length
        for i in range(self.n_dims):
            inputs_concat =trim_info_list[i]
            inputs = tf.reshape(inputs_concat,shape=[batch_size,self.list_length,-1,inputs_concat.shape[2]])
            inputs = tf.transpose(inputs,perm=[0,2,1,3])
            inputs = tf.reshape(inputs,[-1,self.list_length,inputs.shape[3]])
            inputs_sub_x = inputs[:,:,1:]
            inputs_sub_y = tf.reshape(inputs[:,:,0],[batch_size,-1,self.list_length])
            x = self.lstm_layer(inputs_sub_x) 
            x= self.output_layer(x)
            x = tf.reshape(x,[batch_size,-1,self.list_length])
            output_element = inputs_sub_y+tf.expand_dims(self.coefficient_list,axis=0)*x
            
            reduce_output = tf.transpose(tf.reduce_mean(output_element,axis=1,keepdims=True),perm=[0,2,1])
            outputs_list.append(reduce_output)   
        final_output = self.rnn_2nd(outputs_list,batch_size)  
        return   final_output
    def preprocessing_inputs(self,input_slice):
        code = GL.get_map('code_parameters')
        check_matrix_H = code.H
        original_input = input_slice[0]
        original_label = input_slice[1]
        expanded_input = tf.expand_dims(original_input,axis=1)
        information_matrix = expanded_input*check_matrix_H
        trim_info_list = []
        for i in range(code.check_matrix_column):
            selected_info_col = information_matrix[:,:,i:i+1]
            part1 = information_matrix[:,:,:i]
            part2 = information_matrix[:,:,i+1:]
            new_info_matrix = tf.concat([selected_info_col,part1, part2], axis=-1)
            #assuming constant row weight    
            compressed_info_matrix = tf.reshape(new_info_matrix[new_info_matrix!=0.],\
                                                       shape=[original_input.shape[0],-1,self.row_weight])  
            
            check_matrix_col = tf.reshape(check_matrix_H[:,i],[-1,1])
            trimmed_info_matrix = compressed_info_matrix*tf.cast(check_matrix_col,tf.float32)
            trimmed_info_array = tf.reshape(trimmed_info_matrix[trimmed_info_matrix!=0.],\
                                                 shape=[original_input.shape[0],-1,self.row_weight])
            trim_info_list.append(trimmed_info_array)
        return trim_info_list,original_input[self.list_length-1::self.list_length],original_label[self.list_length-1::self.list_length]
    
class model_rnn_2nd(keras.Model):
    def __init__(self):
        super(model_rnn_2nd, self).__init__()    
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.lstm_layer = keras.layers.LSTM(self.list_length,use_bias=True,dropout=0.0,return_sequences=False,unroll=True)   
        self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)  
    def call(self, input_list,batch_size):
        input_array = tf.reshape(input_list,shape=[-1,batch_size,self.list_length,1])
        input_array = tf.transpose(input_array,perm=[1,0,2,3])
        input_array = tf.reshape(input_array,shape=[-1,self.list_length,1])
        x = self.lstm_layer(input_array) 
        x= self.output_layer(x)
        outputs = tf.reshape(x,shape=[batch_size,-1])
        return  outputs
    
class cnv_nn(keras.Model):
    def __init__(self):
        super(cnv_nn, self).__init__()   
        code = GL.get_map('code_parameters')
        list_length =   GL.get_map('num_iterations') + 1
        self.n_dims = code.check_matrix_column
        self.list_length = list_length
        self.row_weight = code.max_chk_degree
        self.batch_size = GL.get_map('unit_batch_size')
        # self.cnv_one = layers.Conv2D(filters=32,kernel_size=(3,1),strides=(1,1),\
        #                 kernel_initializer='glorot_uniform' ,padding="valid",\
        #                 kernel_regularizer=l2(0.0005),activation="relu",\
        #                 kernel_constraint=maxnorm(3),use_bias=False)
        #init = tf.constant_initializer(np.identity(n))

# tf.nn.conv1d(value, filters, stride, padding)
        self.cnv_one_triple = layers.Conv2D(filters=32,kernel_size=[3,self.row_weight],strides=3,\
                                     padding="valid",activation='elu',use_bias=True)

        self.cnv_one_quintuple = layers.Conv2D(filters=32,kernel_size=[5,self.row_weight],strides=5,\
                                     padding="valid",activation='elu',use_bias=True)            
        self.cnv_two = layers.Conv2D(64,[3,1],1,padding="valid",activation='elu',use_bias=True)
        self.cnv_three = layers.Conv2D(32,[3,1],1,padding="valid",activation='elu',use_bias=True)
        self.cnv_four = layers.Conv2D(8,[3,1],1,padding="valid",activation='elu',use_bias=True)
        self.cnv_five = layers.Conv2D(1,[3,1],1,padding="valid",activation='linear',use_bias=True)   
        
    def preprocessing_inputs(self,input_slice):
        code = GL.get_map('code_parameters')
        check_matrix_H = code.H
        original_input = input_slice[0]
        original_label = input_slice[1]
        expanded_input = tf.expand_dims(original_input,axis=1)
        information_matrix = expanded_input*check_matrix_H
        trim_info_list = []
        for i in range(code.check_matrix_column):
            selected_info_col = information_matrix[:,:,i:i+1]
            part1 = information_matrix[:,:,:i]
            part2 = information_matrix[:,:,i+1:]
            new_info_matrix = tf.concat([selected_info_col,part1, part2], axis=-1)
            #assuming constant row weight    
            compressed_info_matrix = tf.reshape(new_info_matrix[new_info_matrix!=0.],\
                                                       shape=[original_input.shape[0],-1,self.row_weight])  
            
            check_matrix_col = tf.reshape(check_matrix_H[:,i],[-1,1])
            trimmed_info_matrix = compressed_info_matrix*tf.cast(check_matrix_col,tf.float32)
            compressed_trimmed_info = tf.reshape(trimmed_info_matrix[trimmed_info_matrix!=0.],\
                                                 shape=[original_input.shape[0],-1,self.row_weight])
            trimmed_info_array = tf.reshape(compressed_trimmed_info,shape=[self.batch_size ,-1,self.row_weight])
            trim_info_list.append(trimmed_info_array)
        return trim_info_list,original_label[::9,:]
           
    def call(self, trim_info_list):
        outputs = []
        for i in range(self.n_dims):
            inputs = tf.expand_dims(trim_info_list[i],axis=-1)
            if inputs.shape[1] == 3*self.list_length:
                x = self.cnv_one_triple(inputs)  
            elif inputs.shape[1] == 5*self.list_length:
                x = self.cnv_one_quintuple(inputs) 
            else:
                print('No defined intput dimensions! Error occurred !!')
            x =self.cnv_two(x)
            x =self.cnv_three(x)
            x =self.cnv_four(x)
            x =self.cnv_five(x)
            x= tf.squeeze(x,axis=[-1,-2])
            outputs.append(x)    
        outputs = tf.transpose(tf.reshape(outputs,[-1,x.shape[0]]))
        return  outputs
##########################################################################################################    
            
# class satisfied_checks(keras.Model):
#     def __init__(self):
#         super(satisfied_checks, self).__init__()    
#         code = GL.get_map('code_parameters')
#         self.n_dims = code.check_matrix_column
#         self.hidden_layer1 = keras.layers.Dense(int(self.n_dims/4),activation="linear",use_bias=True)
#         self.hidden_layer2 = keras.layers.Dense(int(self.n_dims/16),activation="linear",use_bias=True)
#         self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)
#     def build(self,input_shape):
#         pass 
#     def call(self, inputs):
#         x = self.hidden_layer1(inputs) 
#         x = self.hidden_layer2(x) 
#         output= self.output_layer(x)
#         return   output
    
# class unsatisfied_checks(keras.Model):
#     def __init__(self):
#         super(unsatisfied_checks, self).__init__()    
#         code = GL.get_map('code_parameters')
#         self.n_dims = code.check_matrix_column
#         self.hidden_layer1 = keras.layers.Dense(int(self.n_dims/4),activation="linear",use_bias=True)
#         self.hidden_layer2 = keras.layers.Dense(int(self.n_dims/16),activation="linear",use_bias=True)
#         self.output_layer = keras.layers.Dense(1,activation="linear",use_bias=True)
#     def build(self,input_shape):
#         pass 
#     def call(self, inputs):
#         x = self.hidden_layer1(inputs) 
#         x = self.hidden_layer2(x) 
#         output= self.output_layer(x)
#         return   output

# class NN_model(keras.Model):
#     def __init__(self):
#         super(NN_model, self).__init__()    
#         code = GL.get_map('code_parameters')
#         self.n_dims = code.check_matrix_column
#         self.m_dims = code.check_matrix_row
#         self.u_model = unsatisfied_checks()
#         self.s_model = satisfied_checks() 
#         self.hyper_par = GL.get_map('hyper_par')
#         self.H = code.H
#         self.row_d = code.max_chk_degree
#     def build(self,input_shape):
#         # introding coefficent to link two models's output
#         self.coefficients = []
#         for i in range(3):
#             #self.coefficients.append(self.add_weight(name='normalized factor'+str(i),shape=[],initializer="zeros",trainable=True)) 
#             self.coefficients.append(self.add_weight(name='normalized factor'+str(i),shape=[],trainable=True)) 

#     def shrink_op(self,stimulus):
#         shrink_list = []
#         for i in range(self.n_dims):
#             if i == 0:
#                 shrink_element = stimulus[:,i+1:]
#             elif i == self.n_dims-1:
#                 shrink_element = stimulus[:,:i]
#             else:
#                 shrink_element = tf.concat([stimulus[:,:i],stimulus[:,i+1:]],axis=1) 
#             shrink_list.append(shrink_element)
#         return shrink_list
        
#     def preprocessing(self,inputs):
#         #shrink check parity matrix H and inputs
#         shrink_H_list = self.shrink_op(self.H)
#         shrink_input_list = self.shrink_op(inputs)
#         shrink_H_tensor = tf.reshape(shrink_H_list,[self.n_dims,self.m_dims,-1])
#         shrink_input_tensor = tf.reshape(shrink_input_list,[self.n_dims,inputs.shape[0],-1])
        
#         expanded_inputs = tf.expand_dims(shrink_input_tensor,axis=2)
#         expanded_H = tf.cast(tf.expand_dims(shrink_H_tensor,axis=1),tf.float32)
#         info_matrix = expanded_inputs*expanded_H   
#         shrink_abs = tf.abs(info_matrix)
#         shrink_sign = tf.where(info_matrix<0,1,0)
#         sorted_abs = tf.sort(shrink_abs,axis=-1,direction='DESCENDING')[:,:,:,:self.row_d-1]
#         return sorted_abs, shrink_sign
        
#     def call(self, inputs,labels):
#         #labels = tf.cast(labels,tf.float32)  
#         #calculate syndrome
#         hard_decision = tf.where(inputs>0,0,1)  
#         hard_syndrome = tf.matmul(hard_decision,self.H.T)%2   
#         sorted_abs, shrink_sign = self.preprocessing(inputs)
#         # streamlined to two models for further refining
#         decision_sign = tf.where(tf.reduce_sum(shrink_sign,axis=-1)%2==0,1.,-1.)
#         u_stream = self.u_model(tf.reshape(sorted_abs,[-1,self.row_d-1]))
#         s_stream = self.s_model(tf.reshape(sorted_abs,[-1,self.row_d-1]))
#         #reordered info 
#         reorder_sign = tf.transpose(tf.reshape(decision_sign,shape=[self.n_dims,inputs.shape[0],-1]),perm=[1,2,0])
#         u_reorder_stream = tf.transpose(tf.reshape(u_stream,shape=[self.n_dims,inputs.shape[0],-1]),perm=[1,2,0])
#         s_reorder_stream = tf.transpose(tf.reshape(s_stream,shape=[self.n_dims,inputs.shape[0],-1]),perm=[1,2,0])
#         expanded_syndrome = tf.expand_dims(hard_syndrome,axis=-1)
#         filter_one = tf.cast(expanded_syndrome*tf.expand_dims(self.H,axis=0),tf.float32)
#         filter_two = tf.cast((1-expanded_syndrome)*tf.expand_dims(self.H,axis=0),tf.float32)
#         bits_weight_tensor = (self.coefficients[1]*filter_one*u_reorder_stream+self.coefficients[2]*filter_two*s_reorder_stream)*reorder_sign
#         bits_estimation = self.coefficients[0]*inputs+tf.reduce_sum(bits_weight_tensor,axis=1) 
  
#         # lri_p = tf.argsort(tf.abs(bits_estimation),axis=-1)
#         # order_estimation = tf.gather(bits_estimation,lri_p,batch_dims=1)
#         # order_label = tf.gather(labels,lri_p,batch_dims=1)
#         # loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=-order_estimation[:,64:], labels=order_label[:,64:]))
#         # #setup of target input, hyper_par in [0,1]
#         hypered_shrink = inputs*self.hyper_par
#         hypered_dilate = inputs
#         input_bits = tf.where(inputs>0,0,1)
#         target_input = tf.where(input_bits!=labels,hypered_shrink,hypered_dilate)
#         loss = tf.reduce_sum(tf.abs(target_input-bits_estimation))
#         #loss = tf.reduce_sum(tf.abs(target_input-bits_estimation))
#         # #target_prob = tf.sigmoid(target_input)
#         # #measure discprepancy with loss          
#         # lri_p = tf.argsort(tf.abs(target_input),axis=-1)
#         # order_estimation = tf.gather(bits_estimation,lri_p,batch_dims=1)
#         # order_target = tf.gather(target_input,lri_p,batch_dims=1)
#         # order_label = tf.gather(labels,lri_p,batch_dims=1)
#         # loss_tensor = tf.where(order_label==0,order_estimation-order_target,order_target-order_estimation)
#         # #query of median point
#         # #loss = -tf.reduce_sum(loss_tensor[:,int(self.n_dims/2):])    
#         # loss = -tf.reduce_min(tf.reduce_min(loss_tensor[:,int(self.n_dims/2):],axis=-1))
#         return  bits_estimation,loss