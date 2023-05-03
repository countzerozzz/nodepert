import jax.random as random
import numpy as np
import jax.numpy as jnp
import nodepert.model.fc as fc

# helper function to init weights and biases:
def init_single_fc_layer(m, n, randkey):
    w_key, _ = random.split(randkey)

    # use Xavier normal initialization
    std = np.sqrt(2.0 / (m + n))
    weights = std * random.normal(w_key, (n, m))
    biases = jnp.zeros((n,))

    return weights, biases

# init all the weights in a network of given size:
def init_fc_params(sizes, key):
    keys = random.split(key, len(sizes))
    params = [init_single_fc_layer(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
    return params


def init(randkey, args, data):

    # build our network
    n_hl = args.n_hl
    hl_size = args.hl_size
    layer_sizes = [data.num_pixels] + [hl_size] * n_hl + [data.num_classes]

    randkey, _ = random.split(randkey)
    params = init_fc_params(layer_sizes, randkey)
    print(f"fully connected network structure: {layer_sizes}\n")

    # get forward pass, optimizer, and optimizer state + params
    forward = fc.build_batchforward()
    noisyforward = fc.build_batchnoisyforward()

    return params, forward, noisyforward