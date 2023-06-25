import numpy as np
import jax.numpy as jnp

import tensorflow_datasets as tfds

data_dir = "data/tfds"

# fetch full dataset and info for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# you can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy

# get the dataset information first:
_, dataset_info = tfds.load(
    name="cifar10", split="train[:1%]", batch_size=-1, data_dir=data_dir, with_info=True
)

# compute dimensions using the dataset information
num_classes = dataset_info.features["label"].num_classes
height, width, channels = dataset_info.features["image"].shape
num_pixels = height * width * channels

# select which split of the data to use:
trainsplit = "train[:100%]"
testsplit = "test[:100%]"

train_data = tfds.load(
    name="cifar10", split=trainsplit, batch_size=-1, data_dir=data_dir, with_info=False
)
train_data = tfds.as_numpy(train_data)

# full train set:
train_images = train_data["image"]
num_train = len(train_images)

# compute essential statistics, per channel for the dataset on the full trainset:
chmean = np.mean(train_images, axis=(0, 1, 2))
chstd = np.std(train_images, axis=(0, 1, 2), keepdims=True)
data_minval = train_images.min()
data_maxval = train_images.max()

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# normalize data to [0,1] range
def normalize_data(x, minval, maxval):
    return (x - minval) / (maxval - minval)


def channel_standardization(x, chmean, chstd):
    return (x - chmean) / chstd


# from paper: K. K. Pal and K. S. Sudeep, “Preprocessing for image classification by convolutional neural networks,” 2016 IEEE
def zca_whiten_images(x):
    x = np.reshape(x, (-1, num_pixels))
    x = normalize_data(x, data_minval, data_maxval)
    # taking the per-pixel mean across the entire batch
    x = x - x.mean(axis=0)
    cov = np.cov(x, rowvar=False)
    # calculate the singular values and vectors of the covariance matrix and use them to rotate the dataset.
    U, S, V = np.linalg.svd(cov)
    # add epsilon to prevent division by zero (using default value from the paper). Whitened image depends on epsilon and batch_size.
    epsilon = 0.1
    x_zca = U.dot(np.diag(1.0 / np.sqrt(S + epsilon))).dot(U.T).dot(x.T).T
    # rescale whitened image to range [0,1]
    x_zca = normalize_data(x_zca, x_zca.min(), x_zca.max())
    # reshaping to [NHWC] will be done in conv fwd pass

    return x_zca


def prepare_data(x, y, preprocess="standardize"):

    if preprocess.lower() == "zca":
        x = zca_whiten_images(x)

    elif preprocess.lower() == "normalize":
        x = normalize_data(x, data_minval, data_maxval)

    else:
        # passed x should be of the form [NHWC]
        x = channel_standardization(x)

    x = np.reshape(x, (-1, num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y


def get_rawdata_batches(batchsize=100, split="train[:100%]", return_tfdata=False):
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name="cifar10", split=split, as_supervised=True, data_dir=data_dir)

    # you can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batchsize).prefetch(1)

    if not return_tfdata:
        ds = tfds.as_numpy(ds)
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return ds

