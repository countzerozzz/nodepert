import numpy as np
import pandas as pd
import jax.numpy as jnp
from jax import random
import pickle
import re
import itertools

import tensorflow_datasets as tfds
import models.fc as fc

seed = 5
inp_dir = 'data/zcaCIFAR/'

data = pickle.load(open(inp_dir + 'train.pkl', 'rb'))
train_x, train_y = list(zip(*data))

data = pickle.load(open(inp_dir + 'test.pkl', 'rb'))
test_x, test_y = list(zip(*data))

num_classes = 10
num_pixels = 3072

# select which split of the data to use:
trainsplit = 'train[:100%]'
testsplit = 'test[:100%]'

# create a one-hot encoding of y
def one_hot(y, num_classes, dtype=np.float32):
  return jnp.array(y[:, None] == jnp.arange(num_classes), dtype)

# this dataloader is about 10 times slower than tfds, probably as it isn't pre-fetching batches.
def get_data_batches(batchsize=100, split='train[:100%]'):
    if(re.search('train', split)):
        x = jnp.array(train_x)
        y = one_hot(np.asarray(train_y), 10)

    elif(re.search('test', split)):
        x = jnp.array(test_x)
        y = one_hot(np.asarray(test_y), 10)
    
    else:
        raise NameError("wrong dataset split requested!")
    
    #regex to choose num_batches according to the split passed to the function
    num_batches = int(int(re.findall('[0-9]+', split)[0])/100*(len(y)/batchsize))
    num_train = num_batches * batchsize

    try:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(num_train)        
        # keep getting batches until you get to the end.
        while True:
            for i in range(num_batches):
                batch_idx = perm[i * batchsize:(i + 1) * batchsize]
                yield x[batch_idx], y[batch_idx]
                #if generator has reached the number of batches in an epoch, raise exception to break out
                if(i == num_batches-1):
                    raise StopIteration
    except:
        pass
