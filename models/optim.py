import jax
import jax.numpy as jnp
from jax import random
from jax import grad
from jax import jit
import models.fc as fc

import models.losses as losses

# defaults
mseloss = losses.batchmseloss
forward = fc.batchforward
noisyforward = fc.batchnoisyforward


@jit
def loss(x, y, params):
    h, _ = forward(x, params)
    loss = mseloss(h[-1], y).mean()
    # loss = celoss(h[-1], y).mean()
    return loss


@jit
def nploss(x, y, params, randkey):
    #   sigma = fc.nodepert_noisescale
    randkey, _ = random.split(randkey)

    # forward pass with noise
    h, a, xi, aux = noisyforward(x, params, randkey)
    noisypred = h[-1]

    # forward pass with no noise
    h, a = forward(x, params)
    pred = h[-1]

    loss = mseloss(pred, y)
    noisyloss = mseloss(noisypred, y)
    lossdiff = noisyloss - loss

    lossdiff = jax.lax.stop_gradient(lossdiff)
    loss = jnp.mean(lossdiff * jnp.sum(jnp.asarray(aux), 0))
    return loss


@jit
def npupdate(x, y, params, randkey, optimstate):
    print("building np update")
    lr = optimstate["lr"]
    wd = optimstate.get("wd", 0)
    if "linear" in optimstate:
        global forward
        global noisyforward
        forward = fc.batchlinforward
        noisyforward = fc.batchnoisylinforward

    grads = grad(nploss, argnums=(2))(x, y, params, randkey)
    return (
        [
            (w - lr * dw - wd * w, b - lr * db - wd * b)
            for (w, b), (dw, db) in zip(params, grads)
        ],
        grads,
        optimstate,
    )


@jit
def sgdupdate(x, y, params, randkey, optimstate):
    print("building sgd update")
    lr = optimstate["lr"]
    wd = optimstate.get("wd", 0)
    # this is a hack for quickly including a linear FC networks.
    if "linear" in optimstate:
        global forward
        forward = fc.batchlinforward

    grads = grad(loss, argnums=(2))(x, y, params)
    return (
        [
            (w - lr * dw - wd * w, b - lr * db - wd * b)
            for (w, b), (dw, db) in zip(params, grads)
        ],
        grads,
        optimstate,
    )


# This is the old, by hand way we used to compute the np updates:
# inefficient, but useful for debugging

# @jit
# def oldnpupdate(x, y, params, randkey, optimstate):
#   print('building npupdate')
#   lr = optimstate['lr']
#   sigma = fc.nodepert_noisescale
#   randkey, _ = random.split(randkey)
#
#   # forward pass with noise
#   h, a, xi = noisyforward(x, params, randkey)
#   noisypred = h[-1]
#
#   # forward pass with no noise
#   h, a = forward(x, params)
#   pred = h[-1]
#
#   # should call loss function code here:
#   loss = jnp.mean(jnp.square(pred - y),1)
#   noisyloss = jnp.mean(jnp.square(noisypred - y),1)
#   lossdiff = (noisyloss - loss)/(sigma**2)
#
#   grads=[]
#   for ii in range(len(params)):
#     dh = jnp.einsum('ij,i->ij', xi[ii], lossdiff)
#     dw = jnp.einsum('ij,ik->kj', h[ii], dh) / x.shape[0]
#     db = jnp.mean(dh, 0)
#     grads.append((dw,db))
#
#   return [(w - lr * dw, b - lr * db)
#           for (w, b), (dw, db) in zip(params, grads)], grads, optimstate
