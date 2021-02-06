from __future__ import print_function
import numpy as np
import tensorflow as tf
import sys
import pdb
sys.path.append('../')
import functools

def lazy_property(function):
  attribute = '_cache_' + function.__name__
  
  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, function(self))
    return getattr(self, attribute)
  return decorator

seed = 0

class MLP:
  
  def __init__(self, x, y, network_architecture, drop_keep=1.0):
    self.network_architecture=network_architecture
    self.x = x
    self.y = y
    self.drop_keep = drop_keep
    
    self._initialize_network_params()
    self.logits
    self.max_logits
    self.prediction
    self.cost
    self.accuracy
    self.calc_acc()
    self.calc_auc()
    
  def _initialize_network_params(self):
    n_input    = self.network_architecture['n_input']
    n_hidden_1 = self.network_architecture['num_neurons_1']
    n_hidden_2 = self.network_architecture['num_neurons_2']
    n_classes  = self.network_architecture['num_classes']
    # Previously initialized using random normal
    # Default here is Xavier (tensorflow default)
    # Default initialization works much better 
    h1    = tf.get_variable("h1",    shape=[n_input, n_hidden_1])
    h2    = tf.get_variable("h2",    shape=[n_hidden_1, n_hidden_2])
    h_out = tf.get_variable("h_out", shape=[n_hidden_2, n_classes])
    b1    = tf.get_variable("b1",    shape=[n_hidden_1])
    b2    = tf.get_variable("b2",    shape=[n_hidden_2])
    b_out = tf.get_variable("b_out", shape=[n_classes])
    """
    h1    = tf.Variable(tf.random_normal([n_input, n_hidden_1], seed=seed))
    h2    = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], seed=seed+1))
    h_out = tf.Variable(tf.random_normal([n_hidden_2, n_classes], seed=seed+2))
    b1    = tf.Variable(tf.random_normal([n_hidden_1], seed=seed+3))
    b2    = tf.Variable(tf.random_normal([n_hidden_2], seed=seed+4))
    b_out = tf.Variable(tf.random_normal([n_classes], seed=seed+5))
    """
    self.weights = {'h1': h1, 'h2': h2, 'out': h_out}
    self.biases  = {'b1': b1, 'b2': b2, 'out': b_out}
  
  @lazy_property
  def logits(self):
    # Hidden layer with ReLU activation
    layer_1 = tf.nn.xw_plus_b(self.x, self.weights['h1'], self.biases['b1'])
    layer_1 = tf.nn.relu(layer_1, name="layer_1")
    layer_1 = tf.nn.dropout(layer_1, self.drop_keep)
    
    # Hidden layer with ReLU activation
    layer_2 = tf.nn.xw_plus_b(layer_1, self.weights['h2'], self.biases['b2'])
    layer_2 = tf.nn.relu(layer_2, name="layer_2")
    layer_2 = tf.nn.dropout(layer_2, self.drop_keep)
    
    # Output layer with linear activation
    out_layer = tf.nn.xw_plus_b(layer_2, self.weights['out'], self.biases['out'], name="output_layer")
    return out_layer
  
  @lazy_property
  def max_logits(self):
    # Note: returns max probs, rather than logits right now...
    max_logits = tf.reduce_max(tf.nn.softmax(self.logits), 1, name="max_logits")
    return max_logits

  @lazy_property
  def prediction(self):
    prediction = tf.argmax(self.logits, 1, name="prediction")
    return prediction
  
  @lazy_property
  def cost(self):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
                                                      labels=self.y)
    return tf.reduce_mean(loss)
  
  @lazy_property
  def accuracy(self):
    temp_labels = tf.argmax(self.y, 1)
    correct_predictions = tf.equal(self.prediction, temp_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
    return accuracy
  
  def calc_acc(self):
    self.acc, self.acc_update_op = tf.contrib.metrics.streaming_accuracy(predictions=tf.to_float(self.prediction), 
                                                                    labels=tf.to_float(tf.argmax(self.y,1)))

  def calc_auc(self):
    self.auc, self.auc_update_op = tf.contrib.metrics.streaming_auc(tf.to_float(self.prediction), 
                                                                    tf.to_float(tf.argmax(self.y,1)))
                                                                        
  @staticmethod
  def _weight_and_bias(in_size, out_size, seed_shift):
    weight = tf.truncated_normal([in_size, out_size], stddev=0.01, seed=seed+seed_shift)
    bias = tf.constant(0.1, shape=[out_size])
    return tf.Variable(weight), tf.Variable(bias)

