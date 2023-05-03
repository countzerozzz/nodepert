import pdb as pdb
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path
import os

import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer
import nodepert.model.fc as fc
import nodepert.model.conv as conv

### FUNCTIONALITY ###
# this code is for comparing a fixed, small conv network performance for SGD vs NP
###

# parse arguments:
args = utils.parse_args()

args.network = "conv-base"
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

conv.height = data.height
conv.width = data.width
conv.channels = data.channels

if update_rule == "sgd":
    rows = [0.005, 0.01, 0.05, 0.1, 0.5]
    optimizer = optim.sgdupdate
elif update_rule == "np":
    rows = [1e-3, 2e-3, 3e-3]
    optimizer = optim.npupdate
else:
    raise ValueError("Unknown update rule")

ROW_DATA = "learning_rate"
row_id = args.jobid % len(rows)
lr = rows[row_id]

# len(convout_channels) has to be same as convlayer_sizes!
convout_channels = [32, 32, 32]

# format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [
    (3, 3, data.channels, convout_channels[0]),
    (3, 3, convout_channels[0], convout_channels[1]),
    (3, 3, convout_channels[1], convout_channels[2]),
]

down_factor = 2
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

# num_params = utils.get_params_count(params)
num_conv_layers = len(convlayer_sizes)

print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?
print("conv architecture {}, fc layer {}".format(convlayer_sizes, fclayer_sizes))
# print('model params: ', num_params)

# get forward pass, optimizer, and optimizer state + params
forward = conv.build_batchforward()
optim.forward = conv.build_batchforward()
optim.noisyforward = conv.build_batchnoisyforward()


optimparams = {"lr": lr, "wd": args.wd}

print("total parameters:", utils.get_params_count(params))

# now train
params, expdata = trainer.train(
    params, forward, data, train_config, optimizer, optimparams, randkey, verbose=False
)

df = pd.DataFrame.from_dict(expdata)

# store meta data about the experiment:
for arg in vars(args):
    if (arg == "hl_size" or arg == "n_hl"):
        continue
    df[f"{arg}"] = getattr(args, arg)
pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment
if args.log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "conv-base-wd.csv"):
        use_header = True

    df.to_csv(path + "conv-base-wd.csv", mode="a", header=use_header)
