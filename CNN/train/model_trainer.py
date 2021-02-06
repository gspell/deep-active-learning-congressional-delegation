#!/usr/bin/env python

"""
@author: Greg Spell, December 13, 2017

The following is for my own use as I explore TensorFlow and develop
a coding style I like within the framework. 

Many parts of this resemble the base of "Tensorbase" by my friend
and colleague, Dan Salo. 

"""

import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm, trange

class ModelTrainer():
  
  def __init__(self, train_cfg, restore=False):
    
    self.restore = restore
    self.global_step = tf.Variable(0, trainable=False)
    self.train_cfg   = train_cfg
    self.batch_size  = train_cfg['batch_size']
    
    self._set_seed()
    self._data()
    self._network()
    self._model_metrics()
    self._optimizer()
    self.sess = self._configure_session()
    self._summaries()
    self.saver, self.writer = self._set_saver_writer()
    self._initialize_model()
    
  def __enter__(self):
    return self
  
  def __exit__(self, *err):
    self.close()
  
  def _set_seed(self, seed=None):
    if seed is None:
      seed = self.train_cfg['dataset_seed']
    print("\nSetting numpy random seed to {}".format(seed))
    print("Setting tensorflow random seed to {}".format(seed))
    #tf.random.set_seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
  def _data(self):
    """ Define data (placeholders) """
    raise NotImplementedError
    
  def _network(self):
    """ Define network """
    raise NotImplementedError
  
  def _model_metrics(self):
    """ Define metrics such as loss and accuracy """
    raise NotImplementedError
    
  def _optimizer(self):
    """ Define Optimizer """
    raise NotImplementedError
  
  def _summaries(self):
    """ Define summaries to output to Tensorboard """
    raise NotImplementedError
    
  def _configure_session(self):
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth=True
    return tf.Session(config=session_config)
  
  def _set_saver_writer(self):
    saver  = tf.train.Saver()
    writer = tf.summary.FileWriter(self.train_cfg['log_dir'], self.sess.graph)
    return saver, writer

  def _initialize_model(self):
    print("Initializing local variables")
    self.local_var_init_op = tf.local_variables_initializer()
    self.sess.run(tf.local_variables_initializer())
    print("Initializing global variables")
    if self.restore is True:
      self._restore_from_meta_graph()  
    else:
      all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      self.sess.run(tf.global_variables_initializer())
      self.print_log("Model training from scratch")

  def _restore_from_meta_graph(self):
    print("Restoring from a meta graph")
    filename = self.train_cfg["restore_file"]
    restore_saver = tf.train.import_meta_graph(filename)
    restore_saver.restore(self.sess, filename[:-5]) # remove ".meta" from name
    msg = "Model restored from {}"
    self.print_log(msg.format(self.train_cfg["restore_file"]))
    
  def _init_remaining_variables(self):
    uninit_vars = self.sess.run(tf.report_uninitialized_variables())
    print("\nRemains to initialize {} variables".format(len(uninit_vars)))
    vars_list = list()
    for v in uninit_vars:
      var = v.decode("utf-8")
      vars_list.append(var)
    uninit_vars_tf = [v for v in tf.global_variables() if
                      v.name.split(":")[0] in vars_list]
    self.sess.run(tf.variables_initializer(var_list=uninit_vars_tf))
  
  def evaluate(self):
    raise NotImplementedError
  
  def _get_batches_per_epoch():
    raise NotImplementedError
    
  def _run_train_iter(self, feed_dict):
    raise NotImplementedError
  
  def _get_train_feed_dict(self):
    raise NotImplementedError
  
  def _get_eval_feed_dict(self):
    raise NotImplementedError
  
  def _final_eval(self, avg_train_cost):
    raise NotImplementedError
  
  def _conclude_training(self):
    self._save_model(train_stage=self.epoch)

  def _save_model(self, train_stage):
    self.print_log("\nSaving model checkpoint")
    checkpoint_name = self.train_cfg['log_dir'] + 'part_%d' % train_stage + '.ckpt'
    save_path = self.saver.save(self.sess, checkpoint_name)
    self.print_log("Model saved to file: %s" % save_path)
  
  def close_model(self):
    print("Closing tensorflow session and reseting default graph\n")
    self.sess.close()
    tf.reset_default_graph()

  @staticmethod
  def print_log(message):
    """ Print message to terminal and to logging document if applicable """
    print(message)
    #logging.info(message)
