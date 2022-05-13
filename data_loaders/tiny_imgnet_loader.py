import numpy as np
import jax.numpy as jnp
import tensorflow as tf

import tensorflow_datasets as tfds
from tiny_imagenet import TinyImagenetDataset

tiny_imagenet_builder = TinyImagenetDataset()

tiny_imagenet_builder.download_and_prepare()

# these are dictionary of labels, images, and ids
train_dataset = tiny_imagenet_builder.as_dataset(split="train")
validation_dataset = tiny_imagenet_builder.as_dataset(split="validation")

assert isinstance(train_dataset, tf.data.Dataset)
assert isinstance(validation_dataset, tf.data.Dataset)

num_classes = 200
height = width = 64
channels = 3
num_pixels = height * width * channels

# select which split of the data to use:
trainsplit = "train"

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

# normalize data to [0,1] range
def normalize_data(x, minval=0, maxval=255):
    return (x - minval) / (maxval - minval)

def prepare_data(x, y):
    x = normalize_data(x)
    x = np.reshape(x, (len(x), num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y

def get_rawdata_batches(batchsize=100, split=trainsplit):

    # you can build up an arbitrary tf.data input pipeline
    if split.find("train") != -1:
        ds = train_dataset.batch(batchsize).prefetch(1)

    elif split.find("validation") != -1: 
        ds = validation_dataset.batch(batchsize).prefetch(1)

    else:
        print("wrong split specified")
        exit()

    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)

# for a_train_example in train_dataset.take(5):
#     image, label, id = a_train_example["image"], a_train_example["label"], a_train_example["id"]
#     print(f"Image Shape - {image.shape}")
#     print(f"Label - {label.numpy()}")
#     print(f"Id - {id.numpy()}")

# # print info about the data
# print(tiny_imagenet_builder.info)

