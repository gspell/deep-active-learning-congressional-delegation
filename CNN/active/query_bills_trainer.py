from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import argparse
from tqdm import tqdm, trange
import random

import sys
sys.path.append('../')
import time

from bills_config import cfg
from config_utils import get_train_cfg, get_CNN_params, get_MLP_params 
from train.bills_trainer import BillTrainer
from data.load_bills_data import read_bills_data
from active_learning_utils import make_first_query, get_idxs_to_query

class QueryBillsTrainer(BillTrainer):
  def __init__(self, train_cfg, CNN_params, MLP_params, active_params):
    self.num_to_start       = active_params['num_to_start']
    self.num_each_query     = active_params['num_each_query']
    self.full_training_size = active_params["full_training_size"]
    self._set_seed(seed=train_cfg['dataset_seed'])
    self.num_total_queries = self._compute_num_queries()
    self.unqueried_idxs, self.queried_idxs = self._perform_first_query()
    self.num_queries_performed = 0.
    print("\nThe following TRAINING indices were initially unqueried (sorted):")
    print(np.sort(self.unqueried_idxs))
    super().__init__(train_cfg, CNN_params, MLP_params)
  
  def _load_dataset(self):
    self.data = read_bills_data(self.data_dir, queried_idxs = self.queried_idxs)
    print("\nTraining set is {} examples".format(self.data.train._num_examples))
  
  def _compute_num_queries(self):
    num_unqueried_init = self.full_training_size - self.num_to_start
    num = int(np.ceil(num_unqueried_init / self.num_each_query))
    print("\nWill perform {} queries.".format(num))
    return num
  
  def _perform_first_query(self):
    unqueried = np.arange(int(self.full_training_size))
    queried   = np.random.choice(unqueried, self.num_to_start, replace=False)
    unqueried = np.setdiff1d(unqueried, queried)
    return unqueried, queried

  def perform_random_query(unqueried, num_to_query):
    idxs_queried = np.random.choice(unqueried, num_to_query, replace=False)
    return idxs_queried

  def _perform_query(self):
    print("Have performed {} queries so far.".format(self.num_queries_performed))
    idxs_to_query = get_idxs_to_query(self.num_total_queries,
                                      self.num_queries_performed,
                                      self.num_each_query, self.unqueried_idxs)
    print("\nQueried the following training examples:")
    print(np.sort(idxs_to_query))
    self.num_queries_performed += 1
    print("Queried {} training examples".format(len(idxs_to_query)))
    self.queried_idxs   = np.concatenate((self.queried_idxs, idxs_to_query))
    self.unqueried_idxs = np.setdiff1d(self.unqueried_idxs, idxs_to_query)

  def train_all_queries(self):
    accuracies = []
    train_acc, test_acc = self.train()
    accuracies.append(test_acc)
    for query_iteration in range(self.num_total_queries):
      self._perform_query()
      self.close_model()
      super().__init__(self.train_cfg, self.CNN_params, self.MLP_params)
      msg = "Currently have {} unqueried documents\n"
      print(msg.format(len(self.unqueried_idxs)))
      train_acc, test_acc = self.train()
      accuracies.append(test_acc)
    return accuracies

def train_once(current_train_cfg, run_num):
  train_cfg = get_train_cfg(current_train_cfg)
  train_cfg['log_dir'] = update_logging_directory(run_num, train_cfg['log_dir'])
  CNN_params = get_CNN_params()
  MLP_params = get_MLP_params()
  active_params = get_active_params()
  trainer = QueryBillsTrainer(train_cfg, CNN_params, MLP_params, active_params)
  accuracies = trainer.train_all_queries()
  trainer.close_model()
  return accuracies

def get_active_params():
  active_params = {}
  active_params['num_to_start'] = 20
  active_params['num_each_query'] = 10
  active_params["full_training_size"] = 1228
  return active_params

def set_environ_vars(args):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])

def update_logging_directory(run_num, base_directory):
  update_dir = os.path.join(base_directory, "random/run"+str(run_num)+"/")
  return update_dir
     
def main():
  # Parse Arguments
  parser = argparse.ArgumentParser(description="Bill Classification Arguments")
  parser.add_argument('-g', '--gpu', default=1) # Specify which GPU to use
  parser.add_argument('-r', '--restore', default=0) # Restore from a model
  parser.add_argument('-s', '--seed', default=0)
  args = vars(parser.parse_args())
  train_cfg = {}
  if args['restore'] == 0:
    train_cfg['restore'] = False
  else:
    train_cfg['restore'] = True
  
  train_cfg['dataset_seed'] = int(args['seed'])
  # train_cfg['query_seed'] = int(args['seed'])
  set_environ_vars(args)
  
  filename="random_queries.csv"
  i = int(args['seed'])
  accuracies = train_once(train_cfg, i)
  with open(filename, "ab") as f:
    np.savetxt(f, np.expand_dims(accuracies, axis=0), fmt="%.3f",
               delimiter=", ", newline="\n")
  print(accuracies)
if __name__ == '__main__':
  main()
