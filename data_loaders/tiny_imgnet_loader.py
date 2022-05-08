import os
import numpy as np
import jax.numpy as jnp
import tensorflow as tf

import tensorflow_datasets as tfds
from tiny_imagenet import TinyImagenetDataset

tiny_imagenet_builder = TinyImagenetDataset()

tiny_imagenet_builder.download_and_prepare()

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
testsplit = "test"

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# normalize data to [0,1] range
def normalize_data(x, minval=0, maxval=255):
    return (x - minval) / (maxval - minval)


def prepare_data(x, y):

    x = normalize_data(x)

    x = np.reshape(x, (-1, num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y


def get_rawdata_batches(batchsize=100, split=trainsplit):

    # you can build up an arbitrary tf.data input pipeline
    if "train" in split:
        ds = train_dataset.batch(batchsize).prefetch(1)

    elif "test" in split:
        ds = validation_dataset.batch(batchsize).prefetch(1)

    else:
        print("wrong split specified")
        exit()

    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)


# create a generator that normalizes the data and makes it into JAX arrays
def get_data_batches(batchsize=100, split="train"):

    ds = get_rawdata_batches(batchsize=batchsize, split=split)
    # tmp = next(ds)
    # x, y = tmp['image'], tmp['label']
    # x, y = prepare_data(x, y)
    # print(x.shape, y.shape)

    try:
        # keep getting batches until you get to the end.
        while True:
            tmp = next(ds)
            x, y = tmp["image"], tmp["label"]
            x, y = prepare_data(x, y)
            yield (x, y)
    except:
        pass


# if __name__ == "__main__":
#     # get_data_batches()

#     for x, y in get_data_batches(batchsize=1000, split=testsplit):
#         print(x.shape, y.shape)

#     print('done')
