import pdb as pdb
import numpy as np
import jax as jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
from jax.example_libraries import optimizers
# from jax.ops import index, index_add, index_update # deprecated; use jnp.index_exp instead
from jax.scipy.special import logsumexp
from jax.nn import sigmoid
from jax.nn import softmax
from jax.lib import xla_bridge
import pandas as pd
import time, copy, re
import pickle
import csv
import itertools
import os
from pathlib import Path

import data_loaders
import train
import utils
import models.fc as fc
import models.conv as conv
import models.optim as optim
import models.losses as losses
import models.metrics as metrics

# change here directly when we want to perform experiments with different datasets.
import data_loaders.mnist_loader as data
dataset = "MNIST"

# import data_loaders.fmnist_loader as data
# dataset = 'f-MNIST'

# import data_loaders.cifar10_loader as data
# dataset = 'CIFAR10'

# import data_loaders.tiny_imgnet_loader as data
# dataset = "Tiny-ImageNet"

import importlib

importlib.reload(data_loaders)
importlib.reload(train)
importlib.reload(utils)
importlib.reload(fc)
importlib.reload(conv)
importlib.reload(losses)
importlib.reload(metrics)
importlib.reload(optim)
importlib.reload(data)
