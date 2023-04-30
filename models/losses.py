import jax.numpy as jnp
from jax import vmap

# mse loss:
def mseloss(pred, target):
    loss = jnp.mean(jnp.square(pred - target))
    return loss


batchmseloss = vmap(mseloss, in_axes=(0, 0), out_axes=0)
