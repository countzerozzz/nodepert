import pdb as pdb
import numpy as np
import jax as jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, lax
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
import trainer
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

# import data_loaders.cifar100_loader as data
# dataset = "CIFAR100"