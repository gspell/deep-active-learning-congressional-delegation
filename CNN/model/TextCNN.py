# Imports
import numpy as np
import tensorflow as tf
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

class TextCNN:
  
  def __init__(self, x, cfg, drop_keep=1.0):
    
    self.inputs=x
    self.drop_keep = drop_keep
    
    self._initialize_network_params(cfg)
    self.source_embs
    self.emb_inputs
    self.convolved_pooled_outputs

  def _initialize_network_params(self, cfg):
    self.seq_length   = cfg['seq_length']
    self.vocab_size   = cfg['vocab_size']
    self.emb_size     = cfg['emb_size']
    self.filter_sizes = cfg['filter_sizes']
    self.num_filters  = cfg['num_filters']
  
  @lazy_property  
  def source_embs(self):
    shape=[self.vocab_size, self.emb_size]
    # Randomly initialize for now -- assigned pretrained can occur elsewhere
    # TODO: investigate this random initialization
    #init = tf.random_uniform(shape, -1.0, 1.0, seed=seed)
    #embs = tf.Variable(init, trainable=True, name="W_emb")
    embs = tf.get_variable("W_emb", shape=shape)
    return embs
  
  @lazy_property
  def emb_inputs(self):
    embedded_inputs = tf.nn.embedding_lookup(self.source_embs, self.inputs)
    return tf.expand_dims(embedded_inputs, -1)
    
  @lazy_property
  def convolved_pooled_outputs(self):
    convolved_outputs = []
    for i, filter_size in enumerate(self.filter_sizes):
      with tf.variable_scope("convolution-%s" % filter_size):
        output = self.basic_conv(filter_size, self.emb_inputs, i)
        convolved_outputs.append(output)
    num_filters_total = self.num_filters * len(self.filter_sizes)
    h_pool = tf.concat(convolved_outputs, 3)
    h_pool_drop = tf.nn.dropout(h_pool, self.drop_keep, seed=seed)
    return tf.reshape(h_pool_drop, [-1, num_filters_total])

  def basic_conv(self, filter_size, x, i = 0):
    shape     = [filter_size, self.emb_size, 1, self.num_filters]
    strides   = [1, 1, 1, 1]
    pool_size = [1, self.seq_length - filter_size + 1, 1, 1]
    
    #init      = tf.truncated_normal(shape, stddev=0.1, seed = seed + i)
    #W = tf.Variable(init, name="W")
    #b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
    W = tf.get_variable("W", shape=shape)
    b = tf.get_variable("b", shape=[self.num_filters])

    conv = tf.nn.conv2d(x, W, strides=strides, padding="VALID", name="conv")
    h    = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
    pool = tf.nn.max_pool(h, pool_size, strides=strides,
                          padding="VALID", name="pool")
    return pool
    

        
