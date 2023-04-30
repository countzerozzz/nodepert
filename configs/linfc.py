import jax.random as random
import models.fc as fc

def init(randkey, args, data):

    # build our network
    n_hl = args.n_hl
    hl_size = args.hl_size
    layer_sizes = [data.num_pixels] + [hl_size] * n_hl + [data.num_classes]

    randkey, _ = random.split(randkey)
    params = fc.init(layer_sizes, randkey)
    print(f"fully connected network structure: {layer_sizes}\n")

    # get forward pass, optimizer, and optimizer state + params
    forward = fc.build_batchlinforward()
    noisyforward = fc.build_batchnoisylinforward()

    return params, forward, noisyforward