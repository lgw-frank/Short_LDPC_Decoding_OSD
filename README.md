# Short_LDPC_Decoding_OSD
The source code was uploaded and subject to updates occasionally. The project is about how to boost FER of short LDPC codes by the combination of normalized min-sum and ordered statistics decoding variant supported by two simple neural network models. A training and testing recipe in ./LDPC_128 was given to guide interested readers to replicate all decoding results in my submitted paper, a condensed merge of https://doi.org/10.48550/arXiv.2307.06575 and https://doi.org/10.48550/arXiv.2404.14165. Besides, the interested readers are strongly recommended to read the How_to_use_info file in each module of ./LDPC_128/Module_name/ for proper usage. For any readers interested in replicating the decoding results, do not hesitate to leave a note to lgw.frank@gmail.com when puzzled by the code logics or other questions.

How to deal with potential conflicts encountered attributed to upgradation of new package version:
1)ImportError: keras.optimizers.legacy is not supported in Keras 3. When using tf.keras, to continue using a tf.keras.optimizers.legacy optimizer, you can install the tf_keras package (Keras 2) and set the environment variable TF_USE_LEGACY_KERAS=True to configure TensorFlow to use tf_keras when accessing tf.keras:
Solution: 
from tensorflow.keras.optimizers import Adam
......
optimizer = Adam(learning_rate=0.001)
# Use the new Adam optimizer to substitute for the original one:'optimizer = tf.keras.optimizers.legacy.Adam(exponential_decay)' in nn_training.py file
2)AttributeError: 'Mean' object has no attribute 'reset_states' 
solution: replace reset_states with  reset_state
