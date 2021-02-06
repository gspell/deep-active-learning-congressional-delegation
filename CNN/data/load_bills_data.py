import numpy as np
import os
import pickle
import sys

import pdb

# Ensure the "CNN/data" directory is in path for the imports
sys.path.append('../')
import data.neural_text_preprocessing
from data.dataset_utils import extract_labels, split_dataset, dense_to_one_hot

from data.DataSet import DataSet, DataSets

def examine_test_set(test_ids):
  msg = "\nTest set comprises {} documents of IDs (sorted):"
  print(msg.format(len(test_ids)))
  print(np.sort(test_ids))

def load_unlabeled_bills(train_dir):
  data_sets = DataSets()
  train_x, train_y, train_ids, test_x, test_y, test_ids = load_pickled_data(train_dir)
  try:
    np.shape(train_y)[1]
  except:
    train_y = dense_to_one_hot(train_y.astype(int))
    test_y  = dense_to_one_hot(test_y.astype(int))
  data_sets.unlabeled = DataSet(train_x, train_y, train_ids)
  #data_sets.unlabeled = DataSet(test_x, test_y, test_ids)
  """
  data_sets.train = DataSet(train_x, train_y, train_ids)
  data_sets.test  = DataSet(test_x,  test_y,  test_ids)
  """
  return data_sets

def load_unlabeled_corpus_111(train_dir):
  data_sets = DataSets()
  x, y, ids = load_pickled_corpus111_data(train_dir)
  try:
    np.shape(train_y)[1]
  except:
    y = dense_to_one_hot(y.astype(int))
  data_sets.unlabeled = DataSet(x, y, ids)
  return data_sets

def load_pickled_corpus111_data(corpus_111_dir):
  filename = os.path.join(corpus_111_dir, "all_sequences.pickle")
  with open(filename, "rb") as pickle_file:
    x = pickle.load(pickle_file)
  filename = os.path.join(corpus_111_dir, "all_labels.pickle")
  with open(filename, "rb") as pickle_file:
    y = pickle.load(pickle_file)
  filename = os.path.join(corpus_111_dir, "all_indices.pickle")
  with open(filename, "rb") as pickle_file:
    ids = pickle.load(pickle_file)
  return x, y, ids

def load_pickled_data(train_dir):
  filename = os.path.join(train_dir, "train_sequences.pickle")
  with open(filename, "rb") as pickle_file:
    train_x = pickle.load(pickle_file)
  filename = os.path.join(train_dir, "train_labels.pickle")
  with open(filename, "rb") as pickle_file:
    train_y = pickle.load(pickle_file)
  filename = os.path.join(train_dir, "train_indices.pickle")
  with open(filename, "rb") as pickle_file:
    train_ids = pickle.load(pickle_file)
  filename = os.path.join(train_dir, "test_sequences.pickle")
  with open(filename, "rb") as pickle_file:
    test_x = pickle.load(pickle_file)
  filename = os.path.join(train_dir, "test_labels.pickle")
  with open(filename, "rb") as pickle_file:
    test_y = pickle.load(pickle_file)
  filename = os.path.join(train_dir, "test_indices.pickle")
  with open(filename, "rb") as pickle_file:
    test_ids = pickle.load(pickle_file)
  return train_x, train_y, train_ids, test_x, test_y, test_ids

def read_bills_data(train_dir, queried_idxs=None):
  data_sets = DataSets()
  train_x, train_y, train_ids, test_x, test_y, test_ids = load_pickled_data(train_dir)
  try:
    np.shape(train_y)[1]
  except:
    train_y = dense_to_one_hot(train_y)
    test_y  = dense_to_one_hot(test_y)
  if queried_idxs is not None:
    all_train_idxs = np.arange(len(train_y))
    queried_docs   = train_x[queried_idxs]
    queried_labels = train_y[queried_idxs]
    unqueried_idxs = np.setdiff1d(all_train_idxs, queried_idxs)
    
    remaining_docs   = train_x[unqueried_idxs]
    remaining_labels = train_y[unqueried_idxs]
    
    data_sets.train     = DataSet(queried_docs,   queried_labels,   queried_idxs)
    data_sets.unqueried = DataSet(remaining_docs, remaining_labels, unqueried_idxs)
  else:
    data_sets.train = DataSet(train_x, train_y, train_ids)    
  data_sets.test = DataSet(test_x, test_y, test_ids)

  return data_sets
