import pdb as pdb
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path
import os
import numpy as np

import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer
import nodepert.model.fc as fc
import nodepert.model.conv as conv

### FUNCTIONALITY ###
# A much larger, all convolutional network, with good performance with SGD. Implementation details as mentioned in the paper:
# https://arxiv.org/pdf/1412.6806.pdf
# We use the same network architecture, however, exclude the following while training: dropout, SGD + momentum, decaying LR
###

# parse arguments:
args = utils.parse_args()

args.network = "all-cnn-net"
dataset = args.dataset
update_rule = args.update_rule

randkey = random.PRNGKey(args.jobid)

# define training configs
train_config = {'num_epochs': args.num_epochs, 
                'batchsize': args.batchsize, 
                'compute_norms': False, 
                'save_trajectory': False}

# folder to log experiment results
path = "explogs/conv/wd/"

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

conv.height = data.height
conv.width = data.width
conv.channels = data.channels
# num = 7 # number of learning rates
# rows = np.logspace(-6, -3, num, endpoint=True, dtype=np.float32)

if update_rule == "sgd":
    # rows = np.concatenate(([0.05, 0.07], np.arange(1e-1, 1, 2e-1)))
    rows = np.array([0.005, 0.01, 0.05, 0.1, 0.5])
    optimizer = optim.sgdupdate

elif update_rule == "np":
    # rows = np.linspace(0.01, 0.5, num)
    rows = np.array([8e-5, 1e-4, 2e-4, 3e-4])
    optimizer = optim.npupdate
else:
    raise ValueError("Unknown update rule")

row_id = args.jobid % len(rows)
lr = rows[row_id]

# len(convout_channels) has to be same as convlayer_sizes!
# convout_channels = [num_channels] * conv_depth
convout_channels = [96, 96, 192, 192, 192, 192, 10]

# format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [
    (5, 5, data.channels, convout_channels[0]),
    (3, 3, convout_channels[0], convout_channels[1]),  # stride = 2
    (5, 5, convout_channels[1], convout_channels[2]),
    (3, 3, convout_channels[2], convout_channels[3]),  # stride = 2
    (3, 3, convout_channels[3], convout_channels[4]),
    (1, 1, convout_channels[4], convout_channels[5]),
    (1, 1, convout_channels[5], convout_channels[6]),
]

down_factor = 4
fclayer_sizes = [
    int(
        (data.height / down_factor)
        * (data.width / down_factor)
        * convlayer_sizes[-1][-1]
    ),
    data.num_classes,
]

randkey = random.PRNGKey(args.jobid)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?
print("conv architecture {}, fc layer {}".format(convlayer_sizes, fclayer_sizes))
print("model params: ", utils.get_params_count(params))

# get forward pass, optimizer, and optimizer state + params
optim.forward = conv.build_batchforward()
optim.noisyforward = conv.build_batchnoisyforward()

optimparams = {"lr": args.lr, "wd": args.wd}

# now train
params, expdata = trainer.train(
    params, optim.forward, data, train_config, optimizer, optimparams, randkey, verbose=False
)

df = pd.DataFrame.from_dict(expdata)

# store meta data about the experiment:
for arg in vars(args):
    if (arg == "hl_size" or arg == "n_hl"):
        continue
    df[f"{arg}"] = getattr(args, arg)
pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment:
if args.log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "all-cnn-a-wd.csv"):
        use_header = True

    df.to_csv(path + "all-cnn-a-wd.csv", mode="a", header=use_header)
