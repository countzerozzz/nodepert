import numpy as np
import jax.numpy as jnp

import tensorflow_datasets as tfds

data_dir = "data/tfds"

# fetch full dataset and info for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# you can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
    
# get the dataset information first:
_, dataset_info = tfds.load(
    name="mnist", split="train[:1%]", batch_size=-1, data_dir=data_dir, with_info=True
)

# compute dimensions using the dataset information
num_classes = dataset_info.features["label"].num_classes
height, width, channels = dataset_info.features["image"].shape
num_pixels = height * width * channels

# select which split of the data to use:
trainsplit = "train[:100%]"
testsplit = "test[:100%]"

train_data = tfds.load(
    name="mnist", split=trainsplit, batch_size=-1, data_dir=data_dir, with_info=False
)
train_data = tfds.as_numpy(train_data)

# full train set:
train_images = train_data["image"]
num_train = len(train_images)

# compute essential statistics for the dataset on the full trainset:
data_minval = train_images.min()
data_mean = train_images.mean()
data_maxval = train_images.max()
data_stddev = train_images.std()

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# standadize data to have 0 mean and unit standard deviation
def standardize_data(x, data_mean, data_stddev):
    return (x - data_mean) / data_stddev


def prepare_data(x, y):
    x = standardize_data(x, data_mean, data_stddev)
    x = np.reshape(x, (len(x), num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y


def get_rawdata_batches(batchsize=100, split="train[:100%]", return_tfdata=False):
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name="mnist", split=split, as_supervised=True, data_dir=data_dir)

    # you can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batchsize).prefetch(1)

    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    if not return_tfdata:
        ds = tfds.as_numpy(ds)

    return ds
