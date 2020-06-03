import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
import pickle
import numpy as np
import pandas as pd
import csv
import time
from pathlib import Path

import utils

inp_dir = 'data/tfds'
op_dir = 'data/zcaCIFAR/'

_, dataset_info = tfds.load(name="cifar10", split='train[:1%]', batch_size=-1, data_dir=inp_dir, with_info=True)

# compute dimensions using the dataset information
num_classes = dataset_info.features['label'].num_classes
height, width, channels = dataset_info.features['image'].shape
num_pixels = height * width * channels

train_data = tfds.load(name="cifar10", split='train[:100%]', batch_size=-1, data_dir=inp_dir, with_info=False)
train_data = tfds.as_numpy(train_data)

test_data = tfds.load(name="cifar10", split='test[:100%]', batch_size=-1, data_dir=inp_dir, with_info=False)
test_data = tfds.as_numpy(test_data)

# full train set:
train_x, train_y = train_data['image'], train_data['label']
test_x, test_y = test_data['image'], test_data['label']

# bring data in the range [0,1]
def normalize_data(x):
  return (x - x.min()) / (x.max() - x.min())

def zca_whiten_images(x):
  #taking the per-pixel mean across the entire batch
  x = x - x.mean(axis=0)
  cov = np.cov(x, rowvar=False)
  # calculate the singular values and vectors of the covariance matrix and use them to rotate the dataset.
  U,S,V = np.linalg.svd(cov)
  # add epsilon to prevent division by zero (using default value from the paper). Whitened image depends on epsilon and batch_size. 
  epsilon = 0.1
  x_zca = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(x.T).T
  # rescale whitened image to range [0,1]
  x_zca = normalize_data(x_zca)
  #reshaping to [NHWC] will be done in conv fwd pass
  
  return x_zca

def preprocess_data(x, y, batch_size, apply_whitening = True):
    xtmp = []
    n_batches = len(x) // batch_size
    for i in range(n_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        x_batch = normalize_data(x_batch)
        x_batch = np.reshape(x_batch, (-1, num_pixels))
        
        if(apply_whitening):
            x_batch = zca_whiten_images(x_batch)
        
        xtmp.extend(x_batch)
    
    return xtmp, y

# whitening is depended on the batchsize. According to the paper, it is better to have a larger batchsize
batch_size = 2000
prep_start = time.time()

print('starting preprocessing data...')
train_x, train_y = preprocess_data(train_x, train_y, batch_size)
test_x, test_y= preprocess_data(test_x, test_y, batch_size)
print('time to process data: ', time.time() - prep_start)

Path(op_dir).mkdir(exist_ok=True)

pickle.dump(zip(train_x, train_y), open(op_dir+'train.pkl', 'wb'))
pickle.dump(zip(test_x, test_y), open(op_dir+'test.pkl', 'wb'))
print('written processed dataset into pickle file.')