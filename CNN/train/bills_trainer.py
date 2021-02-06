from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import argparse
from tqdm import tqdm, trange
import random
import pickle
import sys
import time

sys.path.append('../')
from bills_config import cfg
from config_utils import get_train_cfg, get_CNN_params, get_MLP_params
from data.load_bills_data import read_bills_data
from model.MLP_classifier import MLP
from model.TextCNN import TextCNN
from train.model_trainer import ModelTrainer 

class BillTrainer(ModelTrainer):
  def __init__(self, train_cfg, CNN_params, MLP_params):
    self.lr         = train_cfg['learning_rate']
    self.data_dir   = train_cfg['data_dir']
    self.CNN_params = CNN_params
    self.MLP_params = MLP_params
    super().__init__(train_cfg, restore=train_cfg['restore'])
    self._initialize_word_embeddings(self.CNN_params['emb_file'])

  def _load_dataset(self):
    self.data = read_bills_data(self.data_dir)
    
  def _data(self):
    input_shape = [None, self.CNN_params["seq_length"]]
    label_shape = [None, self.MLP_params["num_classes"]]
    self.inputs = tf.placeholder(tf.int32,   input_shape, name="inputs")
    self.labels = tf.placeholder(tf.float32, label_shape, name="labels")
    self.drop_keep_prob = tf.placeholder(tf.float32, name="drop_keep_prob")
    
  def _network(self):
    with tf.variable_scope('model'):
      text_features   = self._text_feature_extractor()
      self.classifier = self._make_classifier(text_features)
  
  def _text_feature_extractor(self):
    self.cnn      = TextCNN(self.inputs, self.CNN_params)
    text_features = self.cnn.convolved_pooled_outputs
    return text_features
  
  def _initialize_word_embeddings(self, embedding_filename):
    if embedding_filename is not None:
      with open(embedding_filename, "rb") as pickle_file:
        initW = pickle.load(pickle_file)
      msg   = "Initializing pickled word embeddings from {}"
      print(msg.format(embedding_filename))
      self.sess.run(self.cnn.source_embedding.assign(initW))
    else:
      print("Not using pretrained word embeddings")
      
  def _make_classifier(self, text_features):
    num_filter_types  = len(self.CNN_params["filter_sizes"])
    num_total_filters = self.CNN_params['num_filters'] * num_filter_types
    self.MLP_params['n_input'] = num_total_filters
    mlp_classifier = MLP(text_features, self.labels,
                         self.MLP_params, self.drop_keep_prob)
    return mlp_classifier
  
  def _model_metrics(self):
    self.loss = self.classifier.cost
    self.acc_update_op = self.classifier.acc_update_op
    self.acc = self.classifier.acc
    
  def _optimizer(self):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      opt = tf.train.AdamOptimizer(self.lr)
      self.train_op = opt.minimize(self.loss, global_step=self.global_step)
  
  def _summaries(self):
    loss_summary = tf.summary.scalar("loss", self.loss)
    acc_summary  = tf.summary.scalar("acc",  self.acc)
    
    # Train Summaries
    self.train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(self.train_cfg['log_dir'], "train")
    self.train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                      self.sess.graph)
    
    # Eval Summaries
    self.eval_summary_op = tf.summary.merge([loss_summary, acc_summary])
    eval_summary_dir = os.path.join(self.train_cfg['log_dir'], "eval")
    self.eval_summary_writer = tf.summary.FileWriter(eval_summary_dir,
                                                     self.sess.graph)
  
  def _get_batches_per_epoch(self):
    return int(np.ceil(self.data.train.num_examples/self.batch_size))
  
  def train(self):
    avg_train_cost, avg_train_acc, train_step = 0., 0.0, 0
    num_batches_per_epoch = self._get_batches_per_epoch()
    for self.epoch in trange(1, self.train_cfg['num_epochs']+1, desc="Epochs"):
      for i in range(num_batches_per_epoch):
        # TODO: put in own function
        feed_dict             = self._get_train_feed_dict(drop_keep=self.train_cfg["drop_keep"])
        batch_cost, batch_acc = self._run_train_iter(feed_dict)
        #avg_train_cost += batch_cost / (num_batches_per_epoch)
        #avg_train_acc  += batch_acc  / (num_batches_per_epoch)
        avg_train_cost += batch_cost / self.train_cfg["num_eval"]
        avg_train_acc  += batch_acc  / self.train_cfg["num_eval"]
        train_step += 1
        if (train_step % self.train_cfg['num_eval'] == 0):
          # TODO: put in own function
          eval_cost, eval_acc = self.evaluate()
          self._display_eval(train_step, avg_train_cost, eval_cost,
                             avg_train_acc, eval_acc)
          avg_train_cost, avg_train_acc = 0.0, 0.0
          self.sess.run(self.local_var_init_op)
    final_metrics = self._final_eval(train_step)
    self._conclude_training()
    return final_metrics

  def evaluate(self):
    self.sess.run(self.local_var_init_op)
    cost, test_acc = self._run_eval_iter()
    return cost, test_acc

  def _final_eval(self, step):
    print("\n")
    avg_train_cost, train_acc = self._get_train_metrics()
    avg_eval_cost,  test_acc  = self._get_eval_metrics()
    self._display_eval(step, avg_train_cost, avg_eval_cost, train_acc, test_acc)
    return train_acc, test_acc
  
  def _get_train_metrics(self):
    num_batches_per_epoch = self._get_batches_per_epoch()
    avg_train_cost, avg_train_acc = 0., 0.
    for i in range(num_batches_per_epoch):
      feed_dict = self._get_train_feed_dict(drop_keep=1.0)
      outputs = [self.global_step, self.loss, self.classifier.accuracy]
      #batch_cost, batch_acc = self._run_train_iter(feed_dict)
      _, batch_cost, batch_acc = self.sess.run(outputs, feed_dict)
      avg_train_cost += batch_cost / (num_batches_per_epoch)
      avg_train_acc  += batch_acc / (num_batches_per_epoch)
    #train_acc = self.sess.run(self.classifier.acc)
    return avg_train_cost, avg_train_acc

  def _get_eval_metrics(self):
    eval_cost, test_acc = self.evaluate()
    #avg_eval_cost = eval_cost / self.data.test.num_examples
    avg_eval_cost = eval_cost
    return avg_eval_cost, test_acc
      
  def _run_train_iter(self, feed_dict):
    outputs= [self.train_op, self.global_step, self.loss,
              self.train_summary_op, self.acc_update_op]
    _, step, cost, summary, _ = self.sess.run(outputs, feed_dict)
    self.train_summary_writer.add_summary(summary=summary, global_step=step)
    acc = self.sess.run(self.classifier.acc)
    return cost, acc
  
  def _run_eval_iter(self):
    feed_dict = self._get_eval_feed_dict()
    """
    outputs   = [self.global_step, self.loss,
                 self.eval_summary_op, self.acc_update_op]
    """
    outputs   = [self.global_step, self.loss,
                 self.eval_summary_op, self.classifier.accuracy]
    #step, cost, summary, _ = self.sess.run(outputs, feed_dict)
    step, cost, summary, acc = self.sess.run(outputs, feed_dict)
    self.eval_summary_writer.add_summary(summary, step)
    #acc = self.sess.run(self.classifier.acc)
    return cost, acc
     
  def _get_train_feed_dict(self, drop_keep):
    batch_x, batch_y, _ = self.data.train.next_batch(self.batch_size)
    return {self.inputs: batch_x, self.labels: batch_y,
            self.drop_keep_prob: drop_keep}
  
  def _get_eval_feed_dict(self):
    eval_data = self.data.test
    return {self.inputs: eval_data.documents, self.labels: eval_data.labels,
            self.drop_keep_prob: 1.0}
  
  def _display_eval(self, step, train_cost, eval_cost, train_acc, test_acc):
    msg = "Step: {0:d} \t {1:.3f} \t {2:.3f} \t {3:.3f} \t {4:.3f}"
    tqdm.write(msg.format(step, train_cost, eval_cost, train_acc, test_acc))

def set_environ_vars(args):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
  print("\nUsing GPU {}".format(args['gpu']))

def train_bills_model(current_train_cfg):
  train_cfg = get_train_cfg(current_train_cfg)
  CNN_params = get_CNN_params()
  MLP_params = get_MLP_params()
  trainer = BillTrainer(train_cfg, CNN_params, MLP_params)
  trainer._load_dataset()
  train_acc, test_acc = trainer.train()
  trainer.close_model()
  return train_acc, test_acc

def record_final_acc(filename, accuracy):
  with open(filename, "ab") as f:
    np.savetxt(f, np.expand_dims(accuracy,axis=0),
               fmt="%.3f", delimiter=", ", newline="\n")
    
def main():
  # Parse Arguments
  parser = argparse.ArgumentParser(description="Bill Classification Arguments")
  parser.add_argument('-g', '--gpu', default=1) # Specify which GPU to use
  parser.add_argument('-r', '--restore', default=0) # Restore from a model
  parser.add_argument('-s', '--seed', default=0) # Int for setting random seed
  args = vars(parser.parse_args())
  train_cfg = {}
  if args['restore'] == 0:
    train_cfg['restore'] = False
  else:
    train_cfg['restore'] = True
  
  train_cfg['dataset_seed'] = int(args['seed'])
  set_environ_vars(args)
  train_acc, test_acc = train_bills_model(train_cfg)
  
  filename="test.csv"
  print(test_acc)
  record_final_acc(filename, test_acc)
  
if __name__ == '__main__':
  main()
