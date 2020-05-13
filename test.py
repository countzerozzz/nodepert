import jax
import jax.numpy as jnp
from jax import random

randkey = random.PRNGKey(0)


randkey, _ = random.split(randkey)
W = random.normal(randkey, (3, 5))

randkey, _ = random.split(randkey)
x = random.normal(randkey, (5,))

randkey, _ = random.split(randkey)
noise = random.normal(randkey, (3,))



def myfunc(W, x):
    y = jnp.dot(W,x) + noise
    loss = jnp.sum(y*noise)
    return loss


dlossdW = jax.grad(myfunc, 0)
