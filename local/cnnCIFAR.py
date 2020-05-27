import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import csv
import time

import utils

start_time = time.time()

# data_dir = '/tmp/tfds'
data_dir = '/nfs/ghome/live/yashm/Desktop/research/nodepert/data/tfds'

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

#* currently no ZCA whitening on images
train_x = train_x / 255
test_x = test_x / 255

# train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
# test_y = tf.keras.utils.to_categorical(test_y, num_classes=num_classes)

# network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata = utils.parse_args()

network = 'conv' # 'fc'
#* when setting the network to 'fc', uncomment these
# n_hl = 2
# hl_size = 500

batchsize = 32
num_epochs = 10
log_expdata = False


path = 'explogs/'

model = tf.keras.models.Sequential()

if (network  == 'fc'):
    train_x = np.reshape(train_x, (len(train_x), num_pixels))
    test_x = np.reshape(test_x, (len(test_x), num_pixels))

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


# learning_rate=lr
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
              #* while using MSE, have to make the labels into one-hot (uncomment to_categorical line on top!)
            #   loss=tf.keras.losses.MeanSquaredError(),
              metrics=['mse', 'accuracy'])

print(model.summary())
start_time = time.time()

history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=batchsize,
                    validation_data=(test_x, test_y), verbose = 0)

elapsed_time = time.time() - start_time

print(history.history['accuracy'])
print(history.history['loss'])
# print(model.evaluate(test_x, test_y, batch_size=1000))

def file_writer():
    params = ["update_rule ",update_rule.upper()," depth ", str(n_hl), " width ", str(hl_size), " lr ", str(lr),
              " num_epochs ", str(num_epochs), " batchsize ",str(batchsize)]

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
