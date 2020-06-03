import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import time
import pandas as pd
import pickle
from pathlib import Path

import utils

seed=0
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

data_dir = 'data/tfds'

_, dataset_info = tfds.load(name="cifar10", split='train[:1%]', batch_size=-1, data_dir=data_dir, with_info=True)

# compute dimensions using the dataset information
num_classes = dataset_info.features['label'].num_classes
height, width, channels = dataset_info.features['image'].shape
num_pixels = height * width * channels

train_data = tfds.load(name="cifar10", split='train[:100%]', batch_size=-1, data_dir=data_dir, with_info=False)
train_data = tfds.as_numpy(train_data)

test_data = tfds.load(name="cifar10", split='test[:100%]', batch_size=-1, data_dir=data_dir, with_info=False)
test_data = tfds.as_numpy(test_data)

# full train set:
train_x, train_y = train_data['image'], train_data['label']
test_x, test_y = test_data['image'], test_data['label']

# compute essential statistics, per channel for the dataset on the full trainset:
chmean = np.mean(train_x, axis=(0,1,2))
chstd = np.std(train_x, axis=(0,1,2), keepdims=True)
data_minval = train_x.min()
data_maxval = train_x.max()

#normalize data to [0,1] range
def normalize_data(x, minval, maxval):
  return (x - minval) / (maxval - minval)

def channel_standardization(x, chmean, chstd):
  return (x - chmean) / chstd

train_x = channel_standardization(train_x, chmean, chstd)
test_x = channel_standardization(test_x, chmean, chstd)
train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)
    
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()
start_time = time.time()
convchannels = 32
actfunc = 'relu'
loss_func = 'mse'
lr = 0.1

if(loss_func == 'mse'):
    loss_fn = tf.keras.losses.MeanSquaredError()

else:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

path = 'explogs/cnn-benchmark/'

model = tf.keras.models.Sequential()

if (network  == 'fc'):
    train_x = np.reshape(train_x, (-1, 3072))
    test_x = np.reshape(test_x, (-1, 3072))
    model.add(tf.keras.layers.Dense(hl_size, input_dim=3072, activation=actfunc))
    for i in range(n_hl - 1):
        model.add(tf.keras.layers.Dense(hl_size, activation=actfunc))
    model.add(tf.keras.layers.Dense(10))

elif(network == 'conv'):
    #* Keras default values of initializer, [kernel: glorot, bias: zeros]
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), strides=1, activation=actfunc, input_shape=(32, 32, 3), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), activation=actfunc))
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), activation=actfunc))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='sigmoid'))

if (update_rule == 'sgd'):
    optim = tf.keras.optimizers.SGD(learning_rate = lr)

model.compile(optimizer=optim,
              loss=loss_fn, 
              metrics=['accuracy'])

print(model.summary())
start_time = time.time()

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batchsize,
                    validation_data=(test_x, test_y), verbose = 0)

print(history.history['val_accuracy'])
print(model.evaluate(test_x, test_y, batch_size=1000))

elapsed_time = time.time() - start_time
total_params = model.count_params()
print(total_params)
print(elapsed_time)

def file_writer(path):
    if(network == 'fc'):
        meta_info = ["network ", network," update_rule ",update_rule.upper()," depth ", str(n_hl), " width ", str(hl_size), " lr ", str(lr),
              " num_epochs ", str(num_epochs)]

    else:
        meta_info = ["network", network, ROW_DATA, rows[row_id], COL_DATA, cols[col_id]]

    with open(path+ROW_DATA+'-'+COL_DATA+'.csv', 'a') as csvFile:
        writer = csv.writer(csvFile, lineterminator='\n')
        writer.writerow(meta_info)
        writer.writerow(history.history['val_accuracy'])
        writer.writerow([str(total_params)])
        writer.writerow("")
        csvFile.flush()

    csvFile.close()
    return

if(log_expdata):
    Path(path).mkdir(exist_ok=True)
    file_writer(path)

# #* code for loading from zca pickle
# seed = 5
# inp_dir = 'data/zcaCIFAR/'

# data = pickle.load(open(inp_dir + 'train.pkl', 'rb'))
# train_x, train_y = list(zip(*data))
# train_x = np.reshape(train_x, (-1, 32, 32, 3))
# train_y = np.reshape(train_y, (-1))
# print(train_x.shape)

# data = pickle.load(open(inp_dir + 'test.pkl', 'rb'))
# test_x, test_y = list(zip(*data))
# test_x = np.reshape(test_x, (-1, 32, 32, 3))
# test_y = np.reshape(test_y, (-1))
# print(test_x.shape)