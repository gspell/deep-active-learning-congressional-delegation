import numpy as np
import pdb
import csv

def dense_to_one_hot(labels_dense, num_classes=2):
  """Convert class labels from scalars to one-hot vectors.
  Note that the class labels MUST BE INTS """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def extract_labels(filename, one_hot=False):
  """Extract the labels into a 1D (or 2D if one_hot) uint8 numpy array [index]."""
  #print('Loading labels from {}'.format(filename))
  labels = []
  with open(filename) as f:
      reader = csv.reader(f, delimiter='\n')
      for row in reader:
          labels.extend([int(label) for label in row])
  del(f)
  labels = np.array(labels)
  print(np.shape(labels))
  if one_hot:
    return dense_to_one_hot(labels)
  return labels

def split_dataset(data, percent_holdout=0.15, permutation=None):
  #print("\nSplitting into training and evaluation sets")
  num_holdout = int(len(data) * percent_holdout)
  data = shuffle_data_indices(data, permutation)
  train_set = data[:-num_holdout]
  test_set = data[-num_holdout:]
  print("Using {} examples for training; {} for evaluation".format(len(train_set), len(test_set)))
  return train_set, test_set

def shuffle_data_indices(data, permutation=None):
  if permutation is None:
    # np.random.shuffle(data) # Does this in-place. Not necessarily desirable!
    permutation = get_shuffling_permutation(len(data))
  data = [data[i] for i in permutation] # This is for if data is of type list
    # If data is an np.array, can do data[perm] directly
    # What are the advantages of one over the other above????
  return data

def get_shuffling_permutation(permutation_length):
  sequence = np.arange(permutation_length)
  np.random.shuffle(sequence)
  return sequence
  
def combine_data_items_into_list(*args):
  """ Returns a zipped list of data items (so that inputs, labels are together rather than parallel)"""
  return list(zip(*args))

def separate_data_items_from_list(data):
  """ Returns an array of unzipped data items (say, inputs and labels separate) """
  return (np.squeeze(data_item) for data_item in zip(*data))
