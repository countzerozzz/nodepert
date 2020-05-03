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

import train
import models.fc as fc
import models.optim as optim
import data.mnistloader as data
import models.losses as losses
import models.metrics as metrics

import importlib
importlib.reload(train)
importlib.reload(fc)
importlib.reload(losses)
importlib.reload(metrics)

from models.losses import batchceloss
from models.losses import batchmseloss
from models.metrics import accuracy


randkey = random.PRNGKey(0)
randkey = random.PRNGKey(int(time.time()))

#define some high level constants
config = {}
config['num_epochs'] = num_epochs = 2000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

#build our network
layer_sizes = [data.num_pixels, 300, 300, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)

forward = fc.batchforward
noisyforward=fc.batchnoisyforward


tmpdata = data.get_data_batches()
x, y = next(tmpdata)
x2, y2 = next(tmpdata)


def loss(x, y, params):
    h, a = forward(x, params)
    celoss = batchceloss(h[-1], y).sum()
    return celoss

# @jit
def sgdupdate(x, y, params, optimstate=None):
    print('building sgdupdate.')
    #learning rate:
    lr = 1e-3
    grads = grad(loss, argnums = (2))(x, y, params)
    return [(w - lr * dw, b - lr * db)
            for (w, b), (dw, db) in zip(params, grads)], optimstate

def sgdgrads(x, y, params, optimstate=None):
    print('building sgdgrads.')
    grads = grad(loss, argnums = (2))(x, y, params)
    return grads

@jit
def npupdate(x, y, params, randkey, optimstate=None):
  print('building npupdate')
  lr = 5e-5
  sigma = fc.nodepert_noisescale
  randkey, _ = random.split(randkey)
  h, a, xi = noisyforward(x, params, randkey)
  noisypred = h[-1]
  h, a = forward(x, params)
  pred = h[-1]

  loss = jnp.mean(jnp.square(pred - y),1)
  noisyloss = jnp.mean(jnp.square(noisypred - y),1)

  lossdiff = (noisyloss - loss)/(sigma**2)

  gradnp=[]
  for ii in range(len(params)):
    tmp = jnp.einsum('ij,i->ij', xi[ii], lossdiff)
    dw = jnp.einsum('ij,ik->kj', h[ii], tmp)
    db = jnp.mean(tmp, 0)
    gradnp.append((dw,db))

  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params,gradnp)], gradnp, optimstate

# then train
params, optimstate, exp_data = train.train(params, forward, data, config, npupdate, randkey, verbose=True)
