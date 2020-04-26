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

randkey = random.PRNGKey(0)

layer_sizes = [data.num_pixels, 300, 300, data.num_classes]
(randkey, _) = random.split(randkey)
params = fc.init_network(layer_sizes, randkey)

#define some high level constants
num_epochs = 500
batchsize = 100
num_classes = data.num_classes

#learning rate:
step_size = 1e-3

batch = data.get_data_batches()
x, y = next(batch)

h = fc.batchforward(x, params)
