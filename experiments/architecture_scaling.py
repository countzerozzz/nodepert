import pdb as pdb
import jax as jax
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path
import numpy as np

import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer
import nodepert.build_network.fc as fc


args = utils.parse_args()

if args.network == "conv":
    raise ValueError("This experiment is only for FC networks")

dataset = args.dataset
update_rule = args.update_rule
randkey = random.PRNGKey(args.jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.
num = 50  # number of learning rates

rows = np.logspace(-4, 0, num, endpoint=True, dtype=np.float32)
ROW_DATA = "learning_rate"
row_id = args.jobid % len(rows)
lr = rows[row_id]

path = "explogs/scaling/"
# define training configs
train_config = {'num_epochs': args.num_epochs, 
                'batchsize': args.batchsize, 
                'compute_norms': False, 
                'save_trajectory': False}

print(f"Dataset: {dataset}")
print(f"Network: {args.network}")
print(f"Update rule: {update_rule}")
# display backend:
print(f"Running on: {xla_bridge.get_backend().platform}\n")

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

# load the network:
randkey, subkey = random.split(randkey)

params, forward, noisyforward = fc.init(subkey, args, data)
optim.forward = forward
optim.noisyforward = noisyforward

# load the optimizer:
if update_rule == "np":
    optimizer = optim.npupdate
elif update_rule == "sgd":
    optimizer = optim.sgdupdate

# define the optimizer hyper parameters:
optimparams = {"lr": args.lr, "wd": args.wd}

# now train:
params, expdata = trainer.train(
    params, optim.forward, data, train_config, optimizer, optimparams, randkey, verbose=False
)

df = pd.DataFrame.from_dict(expdata)

# store meta data about the experiment:
for arg in vars(args):
    df[f"{arg}"] = getattr(args, arg)
# save the results of our experiment:

pd.set_option("display.max_columns", None)
print(df.head(5))

if args.log_expdata:
    logdata_path = Path(path)
    logdata_path.mkdir(parents=True, exist_ok=True)

    csv_file = logdata_path / "depth-scaling.csv"
    write_header = not csv_file.exists()

    df.to_csv(csv_file, mode="a", header=write_header)