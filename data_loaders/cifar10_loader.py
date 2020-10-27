import numpy as np
import jax.numpy as jnp

import tensorflow_datasets as tfds

data_dir = 'data/tfds'
# data_dir = '/tmp/tfds'

# fetch full dataset and info for evaluation
# tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
# you can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy

# get the dataset information first:
_, dataset_info = tfds.load(name="cifar10", split='train[:1%]', batch_size=-1, data_dir=data_dir, with_info=True)

# compute dimensions using the dataset information
num_classes = dataset_info.features['label'].num_classes
height, width, channels = dataset_info.features['image'].shape
num_pixels = height * width * channels

# select which split of the data to use:
trainsplit = 'train[:100%]'
testsplit = 'test[:100%]'

train_data = tfds.load(name="cifar10", split=trainsplit, batch_size=-1, data_dir=data_dir, with_info=False)
train_data = tfds.as_numpy(train_data)

# full train set:
train_images = train_data['image']
num_train = len(train_images)

# compute essential statistics, per channel for the dataset on the full trainset:
chmean = np.mean(train_images, axis=(0,1,2))
chstd = np.std(train_images, axis=(0,1,2), keepdims=True)
data_minval = train_images.min()
data_maxval = train_images.max()

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

#normalize data to [0,1] range
def normalize_data(x, minval, maxval):
  return (x - minval) / (maxval - minval)

def channel_standardization(x, chmean, chstd):
  return (x - chmean) / chstd

# from paper: K. K. Pal and K. S. Sudeep, “Preprocessing for image classification by convolutional neural networks,” 2016 IEEE
def zca_whiten_images(x):
  x = np.reshape(x, (-1, num_pixels))
  x = normalize_data(x, data_minval, data_maxval)
  #taking the per-pixel mean across the entire batch
  x = x - x.mean(axis=0)
  cov = np.cov(x, rowvar=False)
  # calculate the singular values and vectors of the covariance matrix and use them to rotate the dataset.
  U,S,V = np.linalg.svd(cov)
  # add epsilon to prevent division by zero (using default value from the paper). Whitened image depends on epsilon and batch_size. 
  epsilon = 0.1
  x_zca = U.dot(np.diag(1.0/np.sqrt(S + epsilon))).dot(U.T).dot(x.T).T
  # rescale whitened image to range [0,1]
  x_zca = normalize_data(x_zca, x_zca.min(), x_zca.max())
  #reshaping to [NHWC] will be done in conv fwd pass
  
  return x_zca

def prepare_data(x, y, preprocess):
    
    if(preprocess.lower() == 'zca'):
      x = zca_whiten_images(x)
    
    elif(preprocess.lower() == 'normalize'):
      x = normalize_data(x, data_minval, data_maxval)

    else:
      # passed x should be of the form [NHWC]
      x = channel_standardization(x, chmean, chstd)

    x = np.reshape(x, (-1, num_pixels))
    x = jnp.asarray(x)
    y = one_hot(y, num_classes)
    return x, y

def get_rawdata_batches(batchsize=100, split='train[:100%]'):
  # as_supervised=True gives us the (image, label) as a tuple instead of a dict
  ds = tfds.load(name='cifar10', split=split, as_supervised=True, data_dir=data_dir)

  # you can build up an arbitrary tf.data input pipeline
  ds = ds.batch(batchsize).prefetch(1)

  # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
  return tfds.as_numpy(ds)

# create a generator that normalizes the data and makes it into JAX arrays
def get_data_batches(batchsize=100, split='train[:100%]', preprocess = 'standardize'):
    ds = get_rawdata_batches(batchsize, split)

    # at the end of the dataset a 'StopIteration' exception is raised
    try:
        # keep getting batches until you get to the end.
        while True:
            x, y = next(ds)
            x, y = prepare_data(x, y, preprocess)
            yield (x, y)
    except:
        pass
