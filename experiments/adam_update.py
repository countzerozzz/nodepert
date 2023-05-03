import pdb as pdb
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path
import os
import numpy as np
import itertools
import time
import jax
from jax.example_libraries import optimizers
import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer
import nodepert.model.fc as fc

### FUNCTIONALITY ###
# this code updates network weights with an Adam-like update rule using the NP gradients. Storing the final accuracy reached by the network after some
# number of epochs. Paper: https://arxiv.org/pdf/1412.6980.pdf
# pass update_rule as adam-np or adam-sgd or np or sgd
###

# parse arguments:
args = utils.parse_args()

network = args.network
if network == "conv":
    raise ValueError("This experiment is only for FC networks")

dataset = args.dataset
update_rule = args.update_rule

randkey = random.PRNGKey(args.jobid)

# define training configs
train_config = {'num_epochs': args.num_epochs, 
                'batchsize': args.batchsize, 
                'compute_norms': False, 
                'save_trajectory': False}

# folder to log experiment results
path = f"explogs/{args.network}/"

# load the dataset:
match dataset.lower():
    case "mnist":
        import data_loaders.mnist_loader as data
    case "fmnist":
        import data_loaders.fmnist_loader as data
    case "cifar10":
        import data_loaders.cifar10_loader as data
    case "cifar100":
        import data_loaders.cifar100_loader as data

print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?

num = 25  # number of learning rates
rows = np.logspace(start=-5, stop=-1, num=num, endpoint=True, base=10, dtype=np.float32)
# adam usually requires a smaller learning rate
if "adam" in update_rule:
    rows = np.logspace(
        start=-6, stop=-2, num=num, endpoint=True, base=10, dtype=np.float32
    )

lr = rows[args.jobid % len(rows)]

# build our network
layer_sizes = [data.num_pixels] + [args.hl_size] * args.n_hl + [data.num_classes]

randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
optim.forward = fc.build_batchforward()
optim.noisyforward = fc.build_batchnoisyforward()

if "np" in update_rule:
    gradfunc = optim.npupdate
elif "sgd" in update_rule:
    gradfunc = optim.sgdupdate

params = fc.init(layer_sizes, randkey)
itercount = itertools.count()

@jax.jit
def update(i, grads, opt_state):
    return opt_update(i, grads, opt_state)


# initialize the built-in JAX adam optimizer
opt_init, opt_update, get_params = optimizers.adam(lr)

opt_state = opt_init(params)
itercount = itertools.count()

optimparams = {"lr": lr, "t": 0, "wd": args.wd}
test_acc = []

for epoch in range(1, args.num_epochs + 1):
    start_time = time.time()
    test_acc.append(trainer.compute_metrics(params, optim.forward, data)[1])

    print("EPOCH ", epoch)
    for x, y in data.get_rawdata_batches(batchsize=args.batchsize, split=data.trainsplit):
        x, y = data.prepare_data(x, y)
        randkey, _ = random.split(randkey)
        # get the gradients, throw away the traditional weight updates
        new_params, grads = gradfunc(x, y, params, randkey, optimparams)

        if "adam" in update_rule:
            # pass the gradients to the JAX adam function
            opt_state = update(next(itercount), grads, opt_state)
            new_params = get_params(opt_state)

        params = new_params

    epoch_time = time.time() - start_time
    print(
        "epoch training time: {}\n test acc: {}\n".format(
            round(epoch_time, 2), round(test_acc[-1], 3)
        )
    )

df = pd.DataFrame()
pd.set_option("display.max_columns", None)
# take the mean of the last 5 epochs as the final accuracy
df["final_acc"] = [np.mean(test_acc[-5:])]
# store meta data about the experiment
for arg in vars(args):
    if network == "conv" and (arg == "hl_size" or arg == "n_hl"):
        continue
    df[f"{arg}"] = getattr(args, arg)
print(df.head(5))

# save the results of our experiment
if args.log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "adam_update.csv"):
        use_header = True

    df.to_csv(path + "adam_update.csv", mode="a", header=use_header)
