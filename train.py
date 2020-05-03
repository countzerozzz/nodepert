import numpy as np
import jax.numpy as jnp
from jax import random
from models.metrics import accuracy
from models.fc import compute_norms
import time
import pdb

def train(params, forward, data, config, optimizer, randkey, optimstate=None, verbose=False):
  num_epochs = config['num_epochs']
  batchsize = config['batchsize']

  exp_data = {}
  exp_data['epoch'] = []
  exp_data['epoch_time'] = []
  exp_data['train_acc'] = []
  exp_data['test_acc'] = []
  exp_data['param_norms'] = []
  exp_data['grad_norms'] = []

  print('start training...\n')

  for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainpercent):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    # run through the training set and compute the metrics:
    train_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.trainpercent):
        h, a = forward(x, params)
        train_acc.append(accuracy(h[-1], y))
    train_acc = np.mean(train_acc)

    # run through the test set and compute the metrics:
    test_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.testpercent):
      h, a = forward(x, params)
      test_acc.append(accuracy(h[-1], y))
    test_acc = np.mean(test_acc)

    epoch_time = time.time() - start_time

    param_norms = compute_norms(params)
    grad_norms = compute_norms(grads)

    # test whether we're saturating our tanh() or having all 0s in relu layer:
    tmpdata = data.get_data_batches(batchsize=100, split=data.trainpercent)
    x, y = next(tmpdata)
    h, a = forward(x, params)

    # log the experiment data:
    exp_data['epoch'].append(epoch)
    exp_data['epoch_time'].append(epoch_time)
    exp_data['train_acc'].append(np.mean(train_acc))
    exp_data['test_acc'].append(np.mean(test_acc))
    exp_data['param_norms'].append(param_norms)
    exp_data['grad_norms'].append(grad_norms)

    if(verbose):
      print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("Training set accuracy {}".format(train_acc))
      print("Test set accuracy {}".format(test_acc))
      # print("Final layer weight norm {}".format(param_norms[-1][0]))
      # print("Final layer bias norm {}".format(param_norms[-1][0]))
      print("Norm of all params {}".format(jnp.asarray(param_norms).sum()))
      print("Norm of all grads {}".format(jnp.asarray(grad_norms).sum()))
      print("Norm of penultimate layer {}".format(jnp.linalg.norm(h[-2][0,:])))
      print("Sample penultimate layer {}".format(h[-2][0,0:5]))
      print("Sample final layer {}".format(h[-1][0,0:5]))
    else:
      print('.', end='')

  print('finished training')

  return params, optimstate, exp_data
