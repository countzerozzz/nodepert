import pdb as pdb
import jax as jax
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path

import utils
import model.trainer as trainer
import model.fc as fc
import model.conv as conv
import model.optim as optim


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
match dataset:
    case "MNIST":
        import data_loaders.mnist_loader as data
    case "fMNIST":
        import data_loaders.fmnist_loader as data
    case "CIFAR10":
        import data_loaders.cifar10_loader as data
    case "CIFAR100":
        import data_loaders.cifar100_loader as data

# load the network:
randkey, subkey = random.split(randkey)
match network:
    case "fc":
        import configs.fc as fc
        params, forward, noisyforward = fc.init(subkey, args, data)
        optim.forward = forward
        optim.noisyforward = noisyforward
    case "linfc":
        import configs.linfc as linfc
        params, forward, noisyforward = linfc.init(subkey, args, data)
        optim.forward = forward
        optim.noisyforward = noisyforward
    case "conv":
        import configs.conv as conv
        params, forward, noisyforward = conv.init(subkey, args, data)
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
df["dataset"] = dataset
for arg in vars(args):
    if network == "conv" and (arg == "hl_size" or arg == "n_hl"):
        continue
    df[f"{arg}"] = getattr(args, arg)

pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment:
if args.log_expdata:
    logdata_path = Path(path)
    logdata_path.mkdir(parents=True, exist_ok=True)

    csv_file = logdata_path / f"{network}-expdata.csv"
    write_header = not csv_file.exists()

    df.to_csv(csv_file, mode="a", header=write_header)