import pdb as pdb
import jax as jax
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path

import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer


args = utils.parse_args()

network = args.network
dataset = args.dataset
update_rule = args.update_rule
randkey = random.PRNGKey(args.jobid)

path = "explogs/"
# define training configs
train_config = {'num_epochs': args.num_epochs, 
                'batchsize': args.batchsize, 
                'compute_norms': False, 
                'save_trajectory': False}

print(f"Dataset: {dataset}")
print(f"Network: {network}")
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
    case _ :
        raise ValueError(f"Dataset {dataset} not supported")

# load the network:
randkey, subkey = random.split(randkey)
match network:
    case "fc":
        import nodepert.build_network.fc as fc
        params, forward, noisyforward = fc.init(subkey, args, data)
        optim.forward = forward
        optim.noisyforward = noisyforward
    case "linfc":
        import nodepert.build_network.linfc as linfc
        params, forward, noisyforward = linfc.init(subkey, args, data)
        optim.forward = forward
        optim.noisyforward = noisyforward
    case "conv":
        import nodepert.build_network.conv as conv
        params, forward, noisyforward = conv.init(subkey, args, data)
        optim.forward = forward
        optim.noisyforward = noisyforward
    case _ :
        raise ValueError(f"chose a valid network (fc, linfc, conv), {network} not supported")

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
    if network == "conv" and (arg == "hl_size" or arg == "n_hl"):
        continue
    df[f"{arg}"] = getattr(args, arg)
# save the results of our experiment:

pd.set_option("display.max_columns", None)
print(df.head(5))

if args.log_expdata:
    logdata_path = Path(path)
    logdata_path.mkdir(parents=True, exist_ok=True)

    csv_file = logdata_path / f"{network}-expdata.csv"
    write_header = not csv_file.exists()

    df.to_csv(csv_file, mode="a", header=write_header)