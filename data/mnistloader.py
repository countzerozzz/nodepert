import numpy as np
import jax.numpy as jnp

import tensorflow_datasets as tfds
data_dir = '/tmp/tfds'

# fetch full dataset and info for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# you can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy

# get the dataset information first:
_, dataset_info = tfds.load(name="mnist", split='train[:1%]', batch_size=-1, data_dir=data_dir, with_info=True)

# compute dimensions using the dataset information
num_classes = dataset_info.features['label'].num_classes
h, w, c = dataset_info.features['image'].shape
num_pixels = h * w * c

# select which split of the data to use:
trainpercent = 'train[:100%]'
testpercent = 'test[:100%]'

train_data = tfds.load(name="mnist", split=trainpercent, batch_size=-1, data_dir=data_dir, with_info=False)
train_data = tfds.as_numpy(train_data)

# full train set:
train_images = train_data['image']

# compute essential statistics for the dataset on the full trainset:
data_minval = train_images.min()
data_mean = train_images.mean()
data_minval = train_images.max()
data_stddev = train_images.std()

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

# normalize data:
def normalize_data(x, data_mean, data_stddev):
  return (x - data_mean)/data_stddev

def prepare_data(x, y):
    x = normalize_data(x, data_mean, data_stddev)
    x = np.reshape(x, (len(x), num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y

def get_rawdata_batches(batchsize=100, split='train[:100%]'):
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name='mnist', split=split, as_supervised=True, data_dir=data_dir)

  # you can build up an arbitrary tf.data input pipeline
  ds = ds.batch(batchsize).prefetch(1)

  # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
  return tfds.as_numpy(ds)

# create a generator that normalizes the data and makes it into JAX arrays
def get_data_batches(batchsize=100, split='train[:100%]'):
    ds = get_rawdata_batches(batchsize, split)

    # at the end of the dataset a 'StopIteration' exception is raised
    try:
        # keep getting batches until you get to the end.
        while True:
            x, y = next(ds)
            x, y = prepare_data(x, y)
            yield (x, y)
    except:
        pass
