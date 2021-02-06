# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:45 2018

@author: Gregory Spell
"""

""" Bill Classification configuration system

This file specifies default config options for the convolutional feature
extractor and MLP classifier for legislative text

"""

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
cfg = __C

###############################################################################
# TRAINING OPTIONS
###############################################################################

__C.TRAIN = edict()

__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.BATCH_SIZE = 64
__C.TRAIN.NUM_EPOCHS = 35
__C.TRAIN.NUM_DISPLAY = 5 
__C.TRAIN.NUM_EVAL = 1
__C.TRAIN.DROP_KEEP = 0.85

###############################################################################
# FILES AND DIRECTORIES
###############################################################################
# This could be more elegant with fancier pathing or defining home directories
# Change these to whatever you like!
__C.LOGGING_DIRECTORY = "../../results/active_learning_results/QUERY/"
__C.DATA_DIRECTORY = "../../datasets/cnn_processed_data/Labeled/"
__C.RESTORE_FILE = "../../trained_models/temp.ckpt"

###############################################################################
# CNN ARCHITECTURE
###############################################################################

__C.CNN = edict()

__C.CNN.SEQUENCE_LENGTH = 2000
__C.CNN.VOCAB_SIZE = 5775 # TODO be replaced with directory to vocabulary
__C.CNN.EMBEDDING_SIZE = 300
__C.CNN.FILTER_SIZES = [1, 2, 3, 4, 5]
__C.CNN.NUM_FILTERS = 64
__C.CNN.EMBEDDING_FILE = None
###############################################################################
# MLP ARCHITECTURE
###############################################################################

__C.MLP = edict()
__C.MLP.NUM_NEURONS_1 = 64
__C.MLP.NUM_NEURONS_2 = 32
__C.MLP.NUM_CLASSES = 2
