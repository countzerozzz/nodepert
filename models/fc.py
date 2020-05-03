import numpy as np
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp

# define element-wise relu:
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

# init all the weights in a network of given size:
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
  act = jnp.dot(w, h[-1]) + b
  a.append(act)
  # logsoftmax = a[-1] - logsumexp(a[-1])
  # h.append(logsoftmax)
  output = jnp.tanh(a[-1])
  h.append(output)
  return h, a

# upgrade to handle batches using 'vmap'
batchforward = jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))

# compute norms of parameters (frobenius norm for weiths, L2 for biases)
def compute_norms(params):
    norms = [(jnp.linalg.norm(ww), jnp.linalg.norm(bb)) for (ww, bb) in params]
    return norms

###################################
# node perturbation functionality #
###################################

nodepert_noisescale = 1e-6

#currently the way that random numbers are handled here isn't very safe :(

#also, it's terrible to have two functions for forward passes.
#eventually we should make a single function and give an option to sample noise...
#otherwise we have to manually track changes between these...

# noisy forward pass:
def noisyforward(x, params, randkey):
  h = []; a = []; xi = [];
  h.append(x)

  for (w, b) in params[:-1]:
    act = jnp.dot(w, h[-1]) + b
    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale*random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    h.append(relu(a[-1]))

  w, b = params[-1]
  act = jnp.dot(w, h[-1]) + b
  randkey, _ = random.split(randkey)
  noise = nodepert_noisescale*random.normal(randkey, act.shape)
  xi.append(noise)
  a.append(act + noise)
  # logsoftmax = a[-1] - logsumexp(a[-1])
  # h.append(logsoftmax)
  output = jnp.tanh(a[-1])
  h.append(output)
  return h, a, xi

batchnoisyforward = jit(vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0)))
