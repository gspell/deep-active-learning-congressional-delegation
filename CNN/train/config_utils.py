import numpy as np
from bills_config import cfg

def get_train_cfg(train_cfg):
  train_cfg['log_dir']       = cfg.LOGGING_DIRECTORY
  train_cfg['data_dir']      = cfg.DATA_DIRECTORY
  train_cfg['restore_file']  = cfg.RESTORE_FILE
  train_cfg['batch_size']    = cfg.TRAIN.BATCH_SIZE
  train_cfg['num_epochs']    = cfg.TRAIN.NUM_EPOCHS
  train_cfg['learning_rate'] = cfg.TRAIN.LEARNING_RATE
  train_cfg['num_display']   = cfg.TRAIN.NUM_DISPLAY
  train_cfg['num_eval']      = cfg.TRAIN.NUM_EVAL
  train_cfg['drop_keep']     = cfg.TRAIN.DROP_KEEP
  return train_cfg

def get_CNN_params():
  CNN_params = {}
  CNN_params['seq_length']   = cfg.CNN.SEQUENCE_LENGTH
  CNN_params['vocab_size']   = cfg.CNN.VOCAB_SIZE
  CNN_params['filter_sizes'] = cfg.CNN.FILTER_SIZES
  CNN_params['num_filters']  = cfg.CNN.NUM_FILTERS
  CNN_params['emb_size']     = cfg.CNN.EMBEDDING_SIZE
  CNN_params['emb_file']     = cfg.CNN.EMBEDDING_FILE
  return CNN_params

def get_MLP_params():
  MLP_params = {}
  MLP_params['num_neurons_1'] = cfg.MLP.NUM_NEURONS_1
  MLP_params['num_neurons_2'] = cfg.MLP.NUM_NEURONS_2
  MLP_params['num_classes']   = cfg.MLP.NUM_CLASSES
  return MLP_params
