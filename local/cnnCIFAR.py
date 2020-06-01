import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import time

import utils

data_dir = '/nfs/ghome/live/yashm/Desktop/research/nodepert/data/tfds'
# data_dir = '/tmp/tfds'

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

def normalize_data(x):
  return (x - x.min()) / (x.max() - x.min())

def zca_whiten_images(x):
  #taking the per-pixel mean across the entire batch
  x = x - x.mean(axis=0)
  cov = np.cov(x, rowvar=False)
  # calculate the singular values and vectors of the covariance matrix and use them to rotate the dataset.
  U,S,V = np.linalg.svd(cov)
  # add epsilon to prevent division by zero (using default value from the paper). Whitened image depends on epsilon and batch_size. 
  epsilon = 0.1
  x_zca = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(x.T).T
  # rescale whitened image to range [0,1]
  x_zca = normalize_data(x_zca)
  #reshaping to [NHWC] will be done in conv fwd pass
  
  return x_zca

def preprocess_data(x, y, batch_size, apply_whitening = True):
    xtmp = []
    n_batches = len(x) // batch_size
    for i in range(n_batches):
        x_batch = x[i*batch_size:(i+1)*batch_size]
        x_batch = normalize_data(x_batch)
        x_batch = np.reshape(x_batch, (-1, num_pixels))

        if(apply_whitening):
            x_batch = zca_whiten_images(x_batch)
        xtmp.extend(x_batch)
    #make y as one_hot encoding if using MSE
    # y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    xtmp = np.array(xtmp)
    if(network == 'conv'):
        xtmp = np.reshape(xtmp, (-1, height, width, channels))
    
    return xtmp, y

network, update_rule, n_hl, lr, batch_size, hl_size, num_epochs, log_expdata = utils.parse_args()

train_x, train_y = preprocess_data(train_x, train_y, batch_size, network)
test_x, test_y = preprocess_data(test_x, test_y, batch_size, network)

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
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              #* while using MSE, have to make the labels into one-hot (uncomment to_categorical line on top!)
            #   loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', 'accuracy'])

print(model.summary())
start_time = time.time()

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(test_x, test_y), verbose = 1)

print(history.history['accuracy'])
print(history.history['loss'])
# print(model.evaluate(test_x, test_y, batch_size=1000))

def file_writer():
    params = ["update_rule ",update_rule.upper()," depth ", str(n_hl), " width ", str(hl_size), " lr ", str(lr),
              " num_epochs ", str(num_epochs), " batch_size ",str(batch_size)]

    with open(path+'cnnCIFAR.csv', 'a') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(params)
        writer.writerow(history.history['accuracy'])
        writer.writerow(str(elapsed_time))
        writer.writerow("")
        csvFile.flush()

    csvFile.close()
    return

if(log_expdata):
    file_writer()

print(time.time() - start_time)
