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

import models.fc as fc
import data.mnistloader as data
import models.optim as optim
from models.losses import batchceloss
from models.metrics import accuracy
from train import train

randkey = random.PRNGKey(0)

#define some high level constants
config = {}
config['num_epochs'] = num_epochs = 10
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

#build our network
layer_sizes = [data.num_pixels, 300, data.num_classes]
(randkey, _) = random.split(randkey)
params = fc.init(layer_sizes, randkey)

forward = fc.batchforward
noisyforward=fc.batchnoisyforward

tmpdata = data.get_data_batches()
x, y = next(tmpdata)


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
def npupdate(x, y, params, optimstate=None):
  print('building npupdate')
  lr = 2e-4
  sigma = fc.nodepert_noisescale
  h, a, xi = noisyforward(x, params, randkey)
  noisypred = h[-1]
  h, a = forward(x, params)
  pred = h[-1]

  loss = jnp.mean(jnp.square(pred - y),1)
  noisyloss = jnp.mean(jnp.square(noisypred - y),1)

  k = (noisyloss - loss)/(sigma**2)

  # batchcompute_gradsnp=jit(vmap(compute_gradsnp, in_axes=(0,0,0), out_axes=(0)))
  # grad_np=jnp.mean(batchcompute_gradsnp(h, xi, k), 0)

  grad_np=[]
  for ii in range(len(params)):
    # squiggle=xi[i]
    dw = jnp.mean(jnp.einsum('ki,kj,k->kji', h[ii], xi[ii], k), 0)
    db = jnp.mean(jnp.einsum('ki,k->ki', xi[ii], k), 0)
    grad_np.append((dw,db))


  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params,grad_np)], optimstate

# compute on larger batches and show that it is converging with sgd

# gradssgd = sgdgrads(x, y, params)
# gradsnp = npupdate(x, y, params)


# then train
params, optimstate, exp_data = train(params, forward, data, config, npupdate, verbose=True)
