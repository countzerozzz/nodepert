import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import time
import pandas as pd
import pickle

import utils

# data_dir = 'data/tfds'

# _, dataset_info = tfds.load(name="cifar10", split='train[:1%]', batch_size=-1, data_dir=data_dir, with_info=True)

# # compute dimensions using the dataset information
# num_classes = dataset_info.features['label'].num_classes
# height, width, channels = dataset_info.features['image'].shape
# num_pixels = height * width * channels

# train_data = tfds.load(name="cifar10", split='train[:100%]', batch_size=-1, data_dir=data_dir, with_info=False)
# train_data = tfds.as_numpy(train_data)

# test_data = tfds.load(name="cifar10", split='test[:100%]', batch_size=-1, data_dir=data_dir, with_info=False)
# test_data = tfds.as_numpy(test_data)

# # full train set:
# train_x, train_y = train_data['image'], train_data['label']
# test_x, test_y = test_data['image'], test_data['label']

# train_x = train_x / 255
# test_x = test_x / 255

#! If it's using ZCA input then it fails to init conv layer: cudNN fails, but only sometimes. why?
seed = 5
inp_dir = 'data/zcaCIFAR/'

data = pickle.load(open(inp_dir + 'train.pkl', 'rb'))
train_x, train_y = list(zip(*data))
train_x = np.reshape(train_x, (-1, 32, 32, 3))
train_y = np.reshape(train_y, (-1))
print(train_x.shape)

data = pickle.load(open(inp_dir + 'test.pkl', 'rb'))
test_x, test_y = list(zip(*data))
test_x = np.reshape(test_x, (-1, 32, 32, 3))
test_y = np.reshape(test_y, (-1))
print(test_x.shape)

network, update_rule, n_hl, lr, batch_size, hl_size, num_epochs, log_expdata = utils.parse_args()

start_time = time.time()

path = 'explogs/'

model = tf.keras.models.Sequential()

if (network  == 'fc'):
    model.add(tf.keras.layers.Dense(hl_size, input_dim=3072, activation='relu'))
    for i in range(n_hl - 1):
        model.add(tf.keras.layers.Dense(hl_size, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

elif(network == 'conv'):
    #* Keras default values of initializer, [kernel: glorot, bias: zeros]
    model.add(tf.keras.layers.Conv2D(32, (3, 3), strides=1, activation='relu', input_shape=(32, 32, 3), kernel_initializer='glorot_uniform', bias_initializer='zeros'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10))

#* pass learning_rate=lr into SGD for setting custom learning rate
if (update_rule == 'sgd'):
    optim = tf.keras.optimizers.SGD(learning_rate = lr)

elif (update_rule == 'adam'):
    optim = tf.keras.optimizers.Adam(learning_rate = lr)

model.compile(optimizer=optim,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              #* while using MSE, have to make the labels into one-hot (uncomment to_categorical line on top!)
            #   loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', 'accuracy'])

print(model.summary())
start_time = time.time()

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(test_x, test_y), verbose = 1)

print(history.history['acc'])
print(history.history['val_acc'])
print(model.evaluate(test_x, test_y, batch_size=1000))

elapsed_time = time.time() - start_time
print(elapsed_time)


def file_writer(path):
    params = ["update_rule ",update_rule.upper()," depth ", str(n_hl), " width ", str(hl_size), " lr ", str(lr),
              " num_epochs ", str(num_epochs), " batch_size ",str(batch_size)]

    with open(path+'cnnCIFAR.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        writer.writerow(history.history['val_acc'])
        writer.writerow(str(elapsed_time))
        writer.writerow("")
        csvFile.flush()

    csvFile.close()
    return

if(log_expdata):
    file_writer(path)

