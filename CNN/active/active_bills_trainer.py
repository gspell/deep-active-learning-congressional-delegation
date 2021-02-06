import numpy as np
import tensorflow as tf
import pdb
import os
import argparse
import time
from tqdm import tqdm, trange
import random

import sys
sys.path.append('../')

from bills_config import cfg
from config_utils import get_train_cfg, get_CNN_params, get_MLP_params 
from query_bills_trainer import QueryBillsTrainer
from data.load_bills_data import read_bills_data
from active_learning_utils import update_unqueried_list

class ActiveBillsTrainer(QueryBillsTrainer):
  def __init__(self, train_cfg, CNN_params, MLP_params, active_params):
    super().__init__(train_cfg, CNN_params, MLP_params, active_params)
    self.test_doc_ids = self.examine_test_set()

  def _perform_query(self):
    print("Have performed {} queries so far.".format(self.num_queries_performed))
    idxs_to_query = self._get_active_queries()
    self.num_queries_performed += 1
    print("\nQueried the following training examples:")
    print(np.sort(idxs_to_query))
    print("Queried {} training examples".format(len(idxs_to_query)))
    self.queried_idxs = np.concatenate((self.queried_idxs, idxs_to_query))
    self.unqueried_idxs = update_unqueried_list(self.unqueried_idxs, idxs_to_query)
    
  def _get_query_feed_dict(self):
    batch_x, batch_y, batch_ids = self.data.unqueried.next_batch(self.batch_size)
    feed = {self.inputs:batch_x, self.labels:batch_y, self.drop_keep_prob: 1.0}
    return feed, batch_ids
  
  def examine_test_set(self):
    test_doc_ids = self.data.test.doc_ids
    msg = "\nTest set comprises {} documents of IDS (sorted):"
    print(msg.format(len(test_doc_ids)))
    print(np.sort(test_doc_ids))
    return test_doc_ids

  def _get_unqueried_logits(self):
    num_unqueried = self.data.unqueried.num_examples
    num_query_batches = int(np.ceil(num_unqueried / self.batch_size))
    print("Have to do {} batches to determine queries".format(num_query_batches))
    unqueried_logits, idxs_this_query = np.array([]), np.array([])
    for i in range(num_query_batches):
      feed_dict, batch_ids = self._get_query_feed_dict()
      batch_logits = self.sess.run(self.classifier.max_logits, feed_dict)
      unqueried_logits = np.concatenate((unqueried_logits, batch_logits))
      idxs_this_query  = np.concatenate((idxs_this_query,  batch_ids))
    return unqueried_logits, idxs_this_query.astype(int)
  
  def _select_max_logits(self, unqueried_logits):
    """ Lol, not even max logits. Min probs... """
    """ Must have called it max logits because for the chosen class """
    # TODO: rename method since misnomer.
    # TODO: stop saying logits, and use probs instead...
    idxs = np.argpartition(unqueried_logits, self.num_each_query)
    idxs = idxs[:self.num_each_query]
    return idxs
  
  def _get_active_queries(self):
    if self.num_queries_performed == self.num_total_queries:
      print("Performed all queries")
      idx_to_query = None
    elif self.num_queries_performed ==(self.num_total_queries - 1):
      print("Peforming the final query")
      idx_to_query = self.unqueried_idxs
    else:
      print("Choosing {} examples to query".format(self.num_each_query))
      unqueried_logits, idxs_this_query = self._get_unqueried_logits()
      max_logit_idxs = self._select_max_logits(unqueried_logits)
      idx_to_query = idxs_this_query[max_logit_idxs]
    return idx_to_query

def train_once(current_train_cfg, run_num):
  train_cfg = get_train_cfg(current_train_cfg)
  train_cfg['log_dir'] = update_logging_directory(run_num, train_cfg['log_dir'])
  CNN_params = get_CNN_params()
  MLP_params = get_MLP_params()
  active_params = get_active_params()
  trainer = ActiveBillsTrainer(train_cfg, CNN_params, MLP_params, active_params)
  accuracies = trainer.train_all_queries()
  trainer.close_model()
  return accuracies

def get_active_params():
  active_params = {}
  active_params['num_to_start'] = 20
  active_params['num_each_query'] = 10
  active_params['full_training_size'] = 1228
  return active_params

def set_environ_vars(args):
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])
  # print("\nUsing GPU {}".format(args['gpu']))

def update_logging_directory(run_num, base_directory):
  update_dir = os.path.join(base_directory, "active/run"+str(run_num)+"/")
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
  #train_cfg['query_seed'] = int(args['seed'])
  set_environ_vars(args)
  
  filename="active_queries.csv"
  i = train_cfg['dataset_seed']
  accuracies = train_once(train_cfg, i)
  with open(filename, "ab") as f:
    np.savetxt(f, np.expand_dims(accuracies, axis=0), fmt="%.3f", delimiter=", ", newline="\n")
  print(accuracies)
if __name__ == '__main__':
  main()
