import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import sigmoid

# define element-wise relu:
def relu(x):
    return jnp.maximum(0, x)


# helper function to init weights and biases:
def init_layer(m, n, randkey):
    w_key, _ = random.split(randkey)

    # use Xavier normal initialization
    std = np.sqrt(2.0 / (m + n))
    weights = std * random.normal(w_key, (n, m))
    biases = jnp.zeros((n,))

    return weights, biases


# init all the weights in a network of given size:
def init(sizes, key):
    keys = random.split(key, len(sizes))
    params = [init_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return params


def copyparams(params):
    paramcopy = [(ww.copy(), bb.copy()) for (ww, bb) in params]
    return paramcopy


# build the forward pass for a single image:
def forward(x, params):
    h = []
    a = []
    h.append(x)

    for (w, b) in params[:-1]:
        a.append(jnp.dot(w, h[-1]) + b)
        h.append(relu(a[-1]))

    w, b = params[-1]
    act = jnp.dot(w, h[-1]) + b
    a.append(act)
    # logsoftmax = a[-1] - logsumexp(a[-1])
    # h.append(logsoftmax)
    output = sigmoid(a[-1])
    h.append(output)
    return h, a


batchforward = jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))
# upgrade to handle batches using 'vmap'

# compute norms of parameters (frobenius norm for weiths, L2 for biases)
@jit
def compute_norms(params):
    norms = [(jnp.linalg.norm(ww), jnp.linalg.norm(bb)) for (ww, bb) in params]
    return norms


###################################
# node perturbation functionality #
###################################

nodepert_noisescale = 1e-4

# currently the way that random numbers are handled here isn't very safe :(
# this is now better, but not good...

# also, it's terrible to have two functions for forward passes.
# eventually we should make a single function and give an option to sample noise...
# otherwise we have to manually track changes between these...
# this is getting worse! we're already trying both sigmoid and softmax

# new noisy forward pass:
def noisyforward(x, params, randkey):
    h = []
    a = []
    xi = []
    aux = []
    h.append(x)

    for (w, b) in params[:-1]:
        h[-1] = jax.lax.stop_gradient(h[-1])
        act = jnp.dot(w, h[-1]) + b
        randkey, _ = random.split(randkey)
        noise = nodepert_noisescale * random.normal(randkey, act.shape)
        xi.append(noise)
        a.append(act + noise)
        aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))
        h.append(relu(a[-1]))

    h[-1] = jax.lax.stop_gradient(h[-1])
    w, b = params[-1]
    act = jnp.dot(w, h[-1]) + b
    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale * random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))

    # logsoftmax = a[-1] - logsumexp(a[-1])
    # h.append(logsoftmax)
    output = sigmoid(a[-1])
    h.append(output)
    return h, a, xi, aux


batchnoisyforward = jit(
    vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
)

################################
#     linear forward pass      #
################################


def linforward(x, params):
    h = []
    a = []
    h.append(x)

    for (w, b) in params[:-1]:
        a.append(jnp.dot(w, h[-1]) + b)
        h.append(a[-1])

    w, b = params[-1]
    act = jnp.dot(w, h[-1]) + b
    a.append(act)
    h.append(a[-1])
    return h, a


batchlinforward = jit(vmap(linforward, in_axes=(0, None), out_axes=(0, 0)))


def noisylinforward(x, params, randkey):
    h = []
    a = []
    xi = []
    aux = []
    h.append(x)

    for (w, b) in params[:-1]:
        h[-1] = jax.lax.stop_gradient(h[-1])
        act = jnp.dot(w, h[-1]) + b
        randkey, _ = random.split(randkey)
        noise = nodepert_noisescale * random.normal(randkey, act.shape)
        xi.append(noise)
        a.append(act + noise)
        aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))
        h.append(a[-1])

    h[-1] = jax.lax.stop_gradient(h[-1])
    w, b = params[-1]
    act = jnp.dot(w, h[-1]) + b
    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale * random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))

    h.append(a[-1])
    return h, a, xi, aux


batchnoisylinforward = jit(
    vmap(noisylinforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0))
)
