import numpy as np
import jax.numpy as jnp
from jax import random
from models.metrics import accuracy
from models.fc import compute_norms
import utils
import time

def compute_metrics(params, forward, data, split_percent="[:100%]"):
    # run through the training set and compute the metrics:
    train_acc = []
    for x, y in data.get_rawdata_batches(batchsize=100, split="train" + split_percent):
        x, y = data.prepare_data(x, y)
        h, a = forward(x, params)
        train_acc.append(accuracy(h[-1], y))
    train_acc = 100 * np.mean(train_acc)

    # run through the test set and compute the metrics:
    test_acc = []
    for x, y in data.get_rawdata_batches(batchsize=100, split="test" + split_percent):
        x, y = data.prepare_data(x, y)
        h, a = forward(x, params)
        test_acc.append(accuracy(h[-1], y))
    test_acc = 100 * np.mean(test_acc)

    return train_acc, test_acc


def train(params, forward, data, config, optimizer, optimparams, randkey, verbose=True):
    num_epochs = config["num_epochs"]
    batchsize = config["batchsize"]

    train_acc, test_acc = compute_metrics(params, forward, data)
    param_norms, grad_norms = compute_norms(params), None

    # Initialize experiment data dictionary
    expdata = {
        "epoch": [0],
        "epoch_time": [0.0],
        "train_acc": [train_acc],
        "test_acc": [test_acc],
        **({"param_norms": [param_norms], "grad_norms": [grad_norms]} if config.get("compute_norms") else {}),
        **({"trajectory": [utils.params_to_npvec(params)]} if config.get("save_trajectory") else {}),
    }

    print("start training...\n")

    for epoch in range(1, num_epochs + 1):

        start_time = time.time()

        # run through the data and train!
        for x, y in data.get_rawdata_batches(
            batchsize=batchsize, split=data.trainsplit
        ):
            x, y = data.prepare_data(x, y)
            randkey, _ = random.split(randkey)
            params, grads = optimizer(x, y, params, randkey, optimparams)

        # compute metrics and norms:
        train_acc, test_acc = compute_metrics(params, forward, data)

        epoch_time = time.time() - start_time

        # log experiment data:
        expdata["epoch"].append(epoch)
        expdata["epoch_time"].append(epoch_time)
        expdata["train_acc"].append(train_acc)
        expdata["test_acc"].append(test_acc)

        if config.get("compute_norms"):
            param_norms = compute_norms(params)
            grad_norms = compute_norms(grads)
            expdata["param_norms"].append(param_norms)
            expdata["grad_norms"].append(grad_norms)

        if config.get("save_trajectory"): 
            expdata["trajectory"].append(utils.params_to_npvec(params))

        if verbose:
            print("\nEpoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))

            if config["compute_norms"]:
                print("Norm of all params {}".format(jnp.asarray(param_norms).sum()))
                print("Norm of all grads {}".format(jnp.asarray(grad_norms).sum()))

        else:
            print(".", end="")

    print("finished training")

    return params, expdata
