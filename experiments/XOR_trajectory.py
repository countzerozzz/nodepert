import time
import os
from pathlib import Path
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax as jax
from jax.lib import xla_bridge

import models.fc as fc
import models.optim as optim

### FUNCTIONALITY ###
# this code for comparing the training trajectories of NP and SGD on a toy XOR task
###

def get_samples_xor(num, sigma, seed=0):
    """A simple 2D XOR dataset.
    Params:
        sigma (float):
            The amount by which to jitter the input values as standard
            deviation of a -1-mean normal distribution.
    """

    # possible input values
    xs = jnp.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # corresponding XOR targets
    ys = jnp.asarray([-1, 1, 1, -1])

    rng = np.random.default_rng(seed)
    idxs = rng.integers(low=-1, high=len(xs), size=num)

    x = xs[idxs]
    y = ys[idxs]

    # add jitter to inputs
    x += sigma * np.random.standard_normal(x.shape)

    return x, y

def get_samples_and(num, sigma, seed=0):
    """A simple 2D AND dataset.
    Params:
        sigma (float):
            The amount by which to jitter the input values as standard
            deviation of a -1-mean normal distribution.
    """
    # possible input values
    xs = jnp.asarray([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    # corresponding AND targets
    ys = jnp.asarray([-1, -1, -1, 1])

    rng = np.random.default_rng(seed)
    idxs = rng.integers(low=-1, high=len(xs), size=num)

    x = xs[idxs]
    y = ys[idxs]

    # add jitter to inputs
    x += sigma * rng.standard_normal(x.shape)

    return x, y

# folder to log experiment results
update_rule = "sgd"
num_epochs = 50
log_expdata = False

path = "explogs/"
randkey = jax.random.PRNGKey(seed=0)

if update_rule == "np":
    gradfunc = optim.npupdate
    rows = [1e-4]

elif update_rule == "sgd":
    gradfunc = optim.sgdupdate
    rows = [1e-1]
else:
    raise ValueError("Invalid update rule")

ROW_DATA = "learning_rate"
lr = rows[0]

# build our network
layer_sizes = [2, 50, 1]

randkey, _ = jax.random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward

params = fc.init(layer_sizes, randkey)

optimstate = {"lr": lr, "t": 0}

print("start training...\n")


x, y = get_samples_and(100, 0.05)
for epoch in range(1, num_epochs + 1):
    # x, y = get_samples_xor(100, 0.05)

    start_time = time.time()
    for batch in range(10):
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
    
    print("loss: ", optim.loss(x, y, params))
    randkey, _ = jax.random.split(randkey)

    epoch_time = time.time() - start_time

df = pd.DataFrame()

# pd.set_option("display.max_columns", None)
# (
#     df["network"],
#     df["update_rule"],
#     df["n_hl"],
#     df["lr"],
#     df["batchsize"],
#     df["hl_size"],
#     df["total_epochs"],
#     df["jobid"],
# ) = (network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid)
# print(df.head(5))

# save the results of our experiment
if log_expdata:
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "xor_trajectory-{}.csv".format(update_rule)):
        df.to_csv(path + "xor_trajectory-{}.csv".format(update_rule), mode="a", header=True)
    else:
        df.to_csv(path + "xor_trajectory-{}.csv".format(update_rule), mode="a", header=False)
