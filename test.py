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
# import tensorflow as tf

import importlib

import models.fc as fc
import data.mnistloader as data
from models.losses import batchceloss
from train import train

importlib.reload(fc)
# importlib.reload(data)

randkey = random.PRNGKey(0)

#define some high level constants
config = {}
config['num_epochs'] = num_epochs = 500
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

#build our network
layer_sizes = [data.num_pixels, 300, 300, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)

tmpdata = data.get_data_batches()
x, y = next(tmpdata)

#define forward, loss, and update functions:
#note that this doesn't currently generate new noise every time.
randkey, _ = random.split(randkey)
def forward(x, params):
    h, a, xi = fc.batchnoisyforward(x, params, randkey)
    print(xi[0][1])
    return h, a

def loss(x, y, params):
    h, a = forward(x, params)
    celoss = batchceloss(h[-1], y).sum()
    return celoss

@jit
def sgdupdate(x, y, params, optimstate=None):
    print('building sgdupdate.')
    #learning rate:
    step_size = 1e-3
    grads = grad(loss, argnums = (2))(x, y, params)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, grads)], optimstate

#then train
params, optimstate, exp_data = train(params, forward, data, config, sgdupdate, verbose=True)
