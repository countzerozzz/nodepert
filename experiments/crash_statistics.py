import pdb as pdb
import numpy as np
from jax import random
from jax.lib import xla_bridge
import pandas as pd
from pathlib import Path
import time
import pickle

import nodepert.utils as utils
import nodepert.optim as optim
import nodepert.trainer as trainer
import nodepert.build_network.fc as fc 
import experiments.crash_utils as crash_utils

### FUNCTIONALITY ###
# this code measures the grad_norm, (noise of gradient)_norm, sign symmetry and angle between 'true' gradient and gradient estimates when the network crashes.
# there is a crash if the current accuracy is less than 'x%' from the max accuracy. When a crash is detected, the network restarts from the last checkpoint
# and then calculates the dynamics during the crash.
###

args = utils.parse_args()

network = args.network
if args.network == "conv":
    raise NotImplementedError("This experiment is not implemented for convolutional networks yet.")

dataset = args.dataset
update_rule = args.update_rule
batchsize = args.batchsize
randkey = random.PRNGKey(args.jobid)

path = "explogs/crash_dynamics/"
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


split_percent = "[:10%]"
data_points = 600
interval = 1  # interval of batches to compute grad dynamics over after a crash.
# set flags for which of the metrics to compute during the crash, 
# gradient statistics (norm of gradient, norm of difference to the true gradient, sign symmetry, angle with true gradient)
# params_stats (norms and variance of weights)
# activity_stats (activity of the network)

calculate_stats = {
    "grad_stats": True,
    "params_stats": True,
    "activity_stats": True,
}

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = "learning_rate"
# rows = np.logspace(start=-3, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
# rows = np.linspace(0.005, 0.025, num=20)
rows = [0.01, 0.015, 0.02, 0.025]

row_id = args.jobid % len(rows)
lr = rows[row_id]
print("learning rate", lr)

# percent of the data to use for training, and calculate the crash dynamics over. manually set to 10%, 6000 data points (mnist).
num_batches = int(data_points / batchsize)

layer_sizes = [data.num_pixels] + args.n_hl * [args.hl_size] + [data.num_classes]
params, forward, noisyforward = fc.init(randkey, args, data)
optim.forward = forward
optim.noisyforward = noisyforward

if update_rule == "np":
    gradfunc = optim.npupdate
elif update_rule == "sgd":
    gradfunc = optim.sgdupdate

optimparams = {"lr": lr, "wd": args.wd}

test_acc = []

# define metrics for measuring a crash
high = -1
crash = False
stored_epoch = -1

# log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(0, args.num_epochs):
    crash=True
    break
    start_time = time.time()
    test_acc.append(
        trainer.compute_metrics(params, forward, data, split_percent=split_percent)[1]
    )
    print("EPOCH {}\ntest acc: {}%".format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    if high - test_acc[-1] > 25:
        print("crash detected! Resetting params")
        crash = True
        break

    # every 5 epochs checkpoint the network params
    if epoch % 5 == 0:
        Path(path + "model_params/").mkdir(parents=True, exist_ok=True)
        pickle.dump(params, open(path + "model_params/" + str(args.jobid) + ".pkl", "wb"))
        stored_epoch = epoch

    for x, y in data.get_rawdata_batches(
        batchsize=args.batchsize, split="train" + split_percent
    ):
        x, y = data.prepare_data(x, y)
        randkey, _ = random.split(randkey)
        params, grads = gradfunc(x, y, params, randkey, optimparams)

    epoch_time = time.time() - start_time
    print("epoch training time: {}s\n".format(round(epoch_time, 2)))

train_df = pd.DataFrame()
train_df["test_acc"] = test_acc

if crash:
    params = pickle.load(open(path + "model_params/" + str(args.jobid) + ".pkl", "rb"))
    test_acc, epochs = [], []

    dataframes = {}
    xl, yl = data.prepare_data(*next(iter(data.get_rawdata_batches(batchsize=data_points, split=data.trainsplit))))

    for ii in range(stored_epoch, stored_epoch + 5):
        print("calculating dynamics for epoch {}.".format(ii))
        for batch_id in range(num_batches):
            x, y = xl[batch_id * batchsize: (batch_id + 1) * batchsize], yl[batch_id * batchsize: (batch_id + 1) * batchsize]
            randkey, _ = random.split(randkey)
            _, sgdgrad = optim.sgdupdate(x, y, params, randkey, optimparams)
            params, npgrad = optim.npupdate(x, y, params, randkey, optimparams)

            if batch_id % interval == 0:
                test_acc.append(trainer.compute_metrics(params, forward, data, split_percent)[1])
                _, truegrad = optim.sgdupdate(xl, yl, params, randkey, optimparams)

                epoch = round(ii + (batch_id + 1) / num_batches, 3)
                temp_dfs = {}
                if(calculate_stats["grad_stats"]):
                    print("calculating grad stats")
                    temp_dfs.update(crash_utils.compute_grad_stats(npgrad, sgdgrad, truegrad))
                if(calculate_stats["params_stats"]):
                    print("calculating params stats")
                    temp_dfs.update(crash_utils.compute_params_stats(params))
                if(calculate_stats["activity_stats"]):
                    print("calculating activity stats")
                    temp_dfs.update(crash_utils.compute_activity_stats(params, x))

                epochs.append(epoch)


else:
    print("no crash detected, exiting...")
    exit()

pd.set_option("display.max_columns", None)



# store meta data about the experiment:
for arg in vars(args):
    train_df[f"{arg}"] = getattr(args, arg)

print("train df:\n", train_df.head(5))
pd.set_option("display.max_columns", None)

# save the results of our experiment:
if args.log_expdata:
    logdata_path = Path(path)
    logdata_path.mkdir(parents=True, exist_ok=True)

    # for df_name in df_names:
    #     print(f"{df_name}:\n{dataframes[df_name].head(5)}\n")
    #     csv_file = logdata_path / f"{df_name}-expdata.csv"
    #     write_header = not csv_file.exists()
    #     dataframes[df_name].to_csv(csv_file, mode="a", header=write_header)


    # csv_file = logdata_path / f"{df_name}-expdata.csv"
    # write_header = not csv_file.exists()
    # dataframes[df_name].to_csv(csv_file, mode="a", header=write_header)

