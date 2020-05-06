import numpy as np
import jax.numpy as jnp
from jax import random
from models.metrics import accuracy
from models.fc import compute_norms
import time
import pdb

def train(params, forward, data, config, optimizer, optimstate, randkey, verbose=True):
  num_epochs = config['num_epochs']
  batchsize = config['batchsize']

  expdata = {}
  expdata['epoch'] = []
  expdata['epoch_time'] = []
  expdata['train_acc'] = []
  expdata['test_acc'] = []
  expdata['param_norms'] = []
  expdata['grad_norms'] = []

  print('start training...\n')

  for epoch in range(num_epochs):
    start_time = time.time()

    # run through the data and train!
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    # run through the training set and compute the metrics:
    train_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.trainsplit):
        h, a = forward(x, params)
        train_acc.append(accuracy(h[-1], y))
    train_acc = np.mean(train_acc)

    # run through the test set and compute the metrics:
    test_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.testsplit):
      h, a = forward(x, params)
      test_acc.append(accuracy(h[-1], y))
    test_acc = np.mean(test_acc)

    # compute norms so that we can see if they blow up:
    param_norms = compute_norms(params)
    grad_norms = compute_norms(grads)

    # get data to test whether we're saturating our nonlinearites:
    tmpdata = data.get_data_batches(batchsize=100, split=data.trainsplit)
    x, y = next(tmpdata)
    h, a = forward(x, params)

    epoch_time = time.time() - start_time

    # log the experiment data:
    expdata['epoch'].append(epoch)
    expdata['epoch_time'].append(epoch_time)
    expdata['train_acc'].append(np.mean(train_acc))
    expdata['test_acc'].append(np.mean(test_acc))
    expdata['param_norms'].append(param_norms)
    expdata['grad_norms'].append(grad_norms)

    if(verbose):
      print("\nEpoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("Training set accuracy {}".format(train_acc))
      print("Test set accuracy {}".format(test_acc))
      print("Norm of all params {}".format(jnp.asarray(param_norms).sum()))
      print("Norm of all grads {}".format(jnp.asarray(grad_norms).sum()))
      print("Norm of penultimate layer {}".format(jnp.linalg.norm(h[-2][0,:])))
      print("Sample penultimate layer {}".format(h[-2][0,0:5]))
      print("Sample final layer {}".format(h[-1][0,0:5]))
    else:
      print('.', end='')

  print('finished training')

  return params, optimstate, expdata
