import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import relu

nodepert_noisescale = 1e-4


def copyparams(params):
    paramcopy = [(ww.copy(), bb.copy()) for (ww, bb) in params]
    return paramcopy

# compute norms of parameters (frobenius norm for weights, L2 for biases)
@jit
def compute_norms(params):
    norms = [(jnp.linalg.norm(ww), jnp.linalg.norm(bb)) for (ww, bb) in params]
    return norms


# build the forward pass for a single image:
# !!!IMPORTANT: any changes made to the forward pass need to be reflected in the noisy forward function as well.
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
    output = jax.nn.sigmoid(a[-1])
    h.append(output)
    return h, a

# upgrade to handle batches using 'vmap'
def build_batchforward():
    return jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))


###################################
# node perturbation functionality #
###################################

# noisy forward pass:
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
    output = jax.nn.sigmoid(a[-1])
    # output = jax.nn.relu(a[-1])
    h.append(output)
    return h, a, xi, aux

def build_batchnoisyforward():
    return jit(vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0)))


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

def build_batchlinforward():
    return jit(vmap(linforward, in_axes=(0, None), out_axes=(0, 0)))


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

def build_batchnoisylinforward():
    return jit(vmap(noisylinforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0)))
