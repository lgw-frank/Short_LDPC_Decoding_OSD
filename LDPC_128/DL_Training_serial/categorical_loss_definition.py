import tensorflow as tf

# Example data
y_true = tf.one_hot([1, 0, 2], depth=3)
y_pred = tf.nn.softmax([[0.1, 0.8, 0.1], [0.9, 0.1, 0.0], [0.2, 0.2, 0.6]])

# Define weights for each class
class_weights = tf.constant([1.0, 2.0, 0.5])

# Calculate element-wise categorical crossentropy without weights
crossentropy_loss = tf.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0)

# Apply weights manually
weighted_loss = crossentropy_loss * class_weights

# Get the scalar sum of the weighted losses
total_loss = tf.reduce_sum(weighted_loss)

print('Element-wise Categorical Crossentropy Loss:', crossentropy_loss.numpy())
print('Element-wise Weighted Categorical Crossentropy Loss:', weighted_loss.numpy())
print('Scalar Sum of Weighted Losses:', total_loss.numpy())
import tensorflow as tf

# Example data
y_true = tf.one_hot([1, 0, 2, 1, 2], depth=3)

# Calculate class frequencies
class_frequencies = tf.reduce_sum(y_true, axis=0)

# Calculate inverse class frequencies as weights
class_weights = 1.0 / class_frequencies.numpy()

print('Class Frequencies:', class_frequencies.numpy())
print('Inverse Class Weights:', class_weights)
import tensorflow as tf

# Example data
batch_size = 5
num_classes = 3

# Simulated batch of data
y_true_batch = tf.one_hot([1, 0, 2, 1, 2], depth=num_classes)
y_pred_batch = tf.nn.softmax([[0.1, 0.8, 0.1], [0.9, 0.1, 0.0], [0.2, 0.2, 0.6], [0.5, 0.4, 0.1], [0.3, 0.3, 0.4]])

# Fixed weights for each class
class_weights = tf.constant([1.0, 2.0, 0.5])

# Repeat the fixed weights to match the batch size
batch_weights = tf.tile(tf.expand_dims(class_weights, 0), [batch_size, 1])

# Calculate element-wise categorical crossentropy with fixed weights
crossentropy_loss = tf.losses.categorical_crossentropy(y_true_batch, y_pred_batch, from_logits=False, label_smoothing=0, sample_weight=batch_weights)

# Get the scalar sum of the weighted losses
total_loss = tf.reduce_sum(crossentropy_loss)

print('Element-wise Categorical Crossentropy Loss:', crossentropy_loss.numpy())
print('Scalar Sum of Weighted Losses:', total_loss.numpy())
