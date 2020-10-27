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

print(tf.config.list_physical_devices('GPU'))
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


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


optimizer, lr, batchsize, num_epochs, network, dropout, final_actfunc, loss_func, log_expdata, jobid = utils.parse_conv_args_tf()

np.random.seed(jobid)
tf.compat.v1.set_random_seed(jobid)

#* learning rates for MSE / SGD
rows = [0.25, 0.1, 0.05, 0.01]

#* learning rates for Adam
# rows = [0.0005, 0.001, 0.005, 0.01]

#* learning rates for cross entropy
# rows = [0.1, 0.05, 0.01, 0.005]

ROW_DATA = 'learning_rate'
row_id = jobid % len(rows)
lr = rows[row_id]

start_time = time.time()
actfunc = 'relu'

if(loss_func == 'cross-entropy'):
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

else:
    loss_fn = tf.keras.losses.MeanSquaredError()
    
path = 'explogs/conv/'

model = tf.keras.models.Sequential()

if (network  == 'fc'):
    train_x = np.reshape(train_x, (-1, 3072))
    test_x = np.reshape(test_x, (-1, 3072))
    model.add(tf.keras.layers.Dense(hl_size, input_dim=3072, activation=actfunc))
    for i in range(n_hl - 1):
        model.add(tf.keras.layers.Dense(hl_size, activation=actfunc))
    model.add(tf.keras.layers.Dense(10))

elif(network == 'conv-small'):
    convchannels = 32
    #* Keras default values of initializer, [kernel: glorot, bias: zeros]
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), strides=1, activation=actfunc, input_shape=(32, 32, 3), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), activation=actfunc))
    model.add(tf.keras.layers.Conv2D(convchannels, (3, 3), activation=actfunc))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation=final_actfunc))

elif(network == 'All-CNN-A'):
    #* All-CNN-A architecture as in paper https://arxiv.org/pdf/1412.6806.pdf (without final 6x6 global averaging penultimate layer)
    model.add(tf.keras.layers.Conv2D(96, (5, 5), strides=1, activation=actfunc, padding='same', input_shape=(32, 32, 3), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    if(dropout):
        model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(96, (3, 3), strides=2, activation=actfunc, padding='same'))
    if(dropout):
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, (5, 5), strides=1, activation=actfunc, padding='same'))
    model.add(tf.keras.layers.Conv2D(192, (3, 3), strides=2, activation=actfunc, padding='same'))
    if(dropout):
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv2D(192, (3, 3), strides=1, activation=actfunc, padding='same'))
    model.add(tf.keras.layers.Conv2D(192, (1, 1), strides=1, activation=actfunc, padding='same'))
    model.add(tf.keras.layers.Conv2D(10, (1, 1), strides=1, activation=actfunc, padding='same'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation=final_actfunc))

if (optimizer == 'sgd'):
    optim = tf.keras.optimizers.SGD(learning_rate = lr)

elif (optimizer == 'sgd-momentum'):
    optim = tf.keras.optimizers.SGD(learning_rate = lr, momentum=0.9)

elif (optimizer == 'adam'):
    optim = tf.keras.optimizers.Adam(learning_rate = lr)

model.compile(optimizer=optim,
              loss=loss_fn, 
              metrics=['accuracy'])

print(model.summary())
start_time = time.time()

time_callback = TimeHistory()

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batchsize, callbacks = [time_callback],
                    validation_data=(test_x, test_y), verbose = 0)

# print(model.evaluate(test_x, test_y, batch_size=1000))

df = pd.DataFrame()
df['train_acc'] = history.history['accuracy']
df['test_acc'] = history.history['val_accuracy']
df['epoch'] = np.arange(1, len(history.history['val_accuracy']) + 1)
df['epoch_time'] = time_callback.times
df['total_params'] = model.count_params()
df['optimizer'], df['lr'], df['batchsize'], df['num_epochs'], df['network'], df['dropout'], df['final_actfunc'], df['loss_func'], df['jobid'] = \
                                        optimizer, lr, batchsize, num_epochs, network, dropout, final_actfunc, loss_func, jobid

pd.set_option('display.max_columns', None)
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'convTF.csv')):
        use_header = True
    
    df.to_csv(path + 'convTF.csv', mode='a', header=use_header)


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