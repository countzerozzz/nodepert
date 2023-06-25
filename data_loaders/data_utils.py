
import numpy as np
import jax.numpy as jnp

# create a one-hot encoding of x of size k:
def one_hot(x, k, dtype=np.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


# normalize data to [0,1] range
def normalize_data(x, minval, maxval):
    return (x - minval) / (maxval - minval)


# standadize data to have 0 mean and unit standard deviation
def standardize_data(x, data_mean, data_stddev):
    return (x - data_mean) / data_stddev


def channel_standardize_data(x, chmean, chstd):
    return (x - chmean) / chstd


# from paper: K. K. Pal and K. S. Sudeep, “Preprocessing for image classification by convolutional neural networks,” 2016 IEEE
def zca_whiten_images(x, num_pixels, data_minval, data_maxval):
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
