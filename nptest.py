# !python --version
import pdb as pdb # pdb.set_trace()
import numpy as np
import jax as jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
from jax.ops import index, index_add, index_update
import matplotlib.pyplot as pp
import matplotlib
import time, copy
import pickle
# import tensorflow as tf

import data.mnistloader as data
import train
import models.fc as fc
import models.optim as optim
import models.losses as losses
import models.metrics as metrics

import importlib
importlib.reload(data)
importlib.reload(train)
importlib.reload(fc)
importlib.reload(losses)
importlib.reload(metrics)
importlib.reload(optim)

randkey = random.PRNGKey(0)
randkey = random.PRNGKey(int(time.time()))

# define some high level constants
config = {}
config['num_epochs'] = num_epochs = 10000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 5000, 5000, 5000, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

forward = fc.batchforward
optimizer = optim.npupdate
# optimizer = optim.sgdupdate
optimstate = {  'lr' : 4e-6, 't' : 0 }

# now train
params, optimstate, exp_data = train.train( params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)

# save out results of experiment
# pickle.dump(exp_data, open("explogs/exp_data.pickle", "wb"))
# pickle.dump(params, open("explogs/params.pickle", "wb"))
