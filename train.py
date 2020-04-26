import numpy as np
import jax.numpy as jnp
from models.metrics import accuracy
import time

def train(params, forward, data, config, optimizer=None, optimstate=None, verbose=False):
  num_epochs = config['num_epochs']
  batchsize = config['batchsize']

  exp_data = {}
  exp_data['epoch'] = []
  exp_data['epoch_time'] = []
  exp_data['train_acc'] = []
  exp_data['test_acc'] = []

  print('start training...\n')

  for epoch in range(num_epochs):
    start_time = time.time()
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainpercent):
        params, optimstate = optimizer(x, y, params, optimstate)

    #run through the training set and compute the metrics:
    train_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.trainpercent):
        h, a = forward(x, params)
        train_acc.append(accuracy(h[-1], y))
    train_acc = np.mean(train_acc)

    #run through the test set and compute the metrics:
    test_acc = []
    for x, y in data.get_data_batches(batchsize=1000, split=data.testpercent):
      h, a = forward(x, params)
      test_acc.append(accuracy(h[-1], y))
    test_acc = np.mean(test_acc)

    epoch_time = time.time() - start_time

    #log the experiment data:
    exp_data['epoch'].append(epoch)
    exp_data['epoch_time'].append(epoch_time)
    exp_data['train_acc'].append(np.mean(train_acc))
    exp_data['test_acc'].append(np.mean(test_acc))

    if(verbose):
      print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
      print("Training set accuracy {}".format(train_acc))
      print("Test set accuracy {}".format(test_acc))
    else:
      print('.', end='')

  print('finished training')

  return params, optimstate, exp_data