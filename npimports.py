import pdb as pdb
import numpy as np
import jax as jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
from jax.ops import index, index_add, index_update
import matplotlib.pyplot as pp
import matplotlib
import time, copy
import pickle

import data.mnistloader as data
import train
import models.fc as fc
import models.optim as optim
import models.losses as losses
import models.metrics as metrics

import importlib
importlib.reload(data)
importlib.reload(train)
importlib.reload(fc)
importlib.reload(losses)
importlib.reload(metrics)
importlib.reload(optim)
