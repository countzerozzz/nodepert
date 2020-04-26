import numpy as np
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp


#define element-wise relu:
def relu(x):
  return jnp.maximum(0,x)

# helper function to init weights and biases:
def init_layer(m, n, randkey):
  w_key, b_key = random.split(randkey)

  # use uniform he style scaling (not glorot):
  std = np.sqrt(2.0 / m)
  weights = std*random.normal(w_key, (n, m))
  biases = jnp.zeros((n,))

  return weights, biases

#init all the weights in a network of given size:
def init(sizes, key):
  keys = random.split(key, len(sizes))
  params = [init_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
  return params

# build the forward pass for a single image:
def forward(x, params):
  h = []; a = [];
  h.append(x)

  for (w, b) in params[:-1]:
    a.append(jnp.dot(w, h[-1]) + b)
    h.append(relu(a[-1]))

  w, b = params[-1]
  logits = jnp.dot(w, h[-1]) + b
  a.append(logits)
  logsoftmax = logits - logsumexp(logits)
  h.append(logsoftmax)
  return h, a

#upgrade to handle batches using 'vmap'
batchforward = jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))
