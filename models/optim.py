import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import sigmoid
import models.fc as fc
import models.losses as losses

batchmseloss = losses.batchmseloss
forward = fc.batchforward
noisyforward=fc.batchnoisyforward

# this is terrible! we should factor out the loss in this file or something
@jit
def loss(x, y, params):
    h, a = forward(x, params)
    loss = batchmseloss(h[-1], y).sum()
    return loss

@jit
def sgdupdate(x, y, params, randkey, optimstate=None):
    print('building sgdupdate.')
    #learning rate:
    lr = 1e-3
    grads = grad(loss, argnums = (2))(x, y, params)
    return [(w - lr * dw, b - lr * db)
            for (w, b), (dw, db) in zip(params, grads)], grads, optimstate

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

  grads=[]
  for ii in range(len(params)):
    tmp = jnp.einsum('ij,i->ij', xi[ii], lossdiff)
    dw = jnp.einsum('ij,ik->kj', h[ii], tmp)
    db = jnp.mean(tmp, 0)
    grads.append((dw,db))

  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params, grads)], grads, optimstate
