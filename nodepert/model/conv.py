import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit

nodepert_noisescale = 1e-5

# defaults defined for MNIST:
height = 28
width = 28
channels = 1

# define element-wise relu:
def relu(x):
    return jnp.maximum(0, x)


# helper function to init conv layer:
def init_single_convlayer(
    kernel_height, kernel_width, input_channels, output_channels, randkey
):

    # NOTICE: ordering changes from function argument ordering!!
    # random normal sampling of parameters
    w_key, _ = random.split(randkey)
    std = np.sqrt(
        2.0 / (kernel_height * kernel_width * (input_channels + output_channels))
    )
    kernel = std * random.normal(
        w_key, (kernel_height, kernel_width, output_channels, input_channels)
    )

    # for conv layers there are typically only 1 bias per channel
    # biases = jnp.zeros((output_channels,))
    biases = std * random.normal(w_key, (output_channels,))

    return kernel, biases


# init all the conv layers in a network of given size:
def init_convlayers(sizes, key):
    keys = random.split(key, len(sizes))
    # pdb.set_trace()
    params = []
    for ii in range(len(sizes)):
        params.append(init_single_convlayer(*sizes[ii], keys[ii]))
    return params


# build the conv forward pass for a single image:
# !!!IMPORTANT: any changes made to the forward pass need to be reflected in the noisy forward function as well.
def forward(x, params):
    # reshape the input to be num_images x height x width x channels (NHWC)
    x = x.reshape(
        1, height, width, channels
    ).astype(np.float32)

    # transpose to NCHW
    x = jnp.transpose(x, [0, 3, 1, 2])
    h = []
    a = []
    h.append(x)
    curr_height = height
    curr_width = width

    for ind, (kernel, biases) in enumerate(params[:-1]):
        stride = 1
        # downsample by 2 in the second and fourth conv layers
        if ind == 1 or ind == 3:
            stride = 2
            curr_height = int(curr_height / 2)
            curr_width = int(curr_width / 2)

        convout_channels = kernel.shape[-2]
        # output (lhs) will be in the form NCHW

        act = jax.lax.conv(
            h[-1],  # lhs = NCHW image tensor
            kernel.transpose(
                [2, 3, 0, 1]
            ),  # rhs = IOHW conv kernel tensor [according to JAX page]
            (stride, stride),  # window strides
            "SAME", # padding mode
        ) 

        act = act + jnp.repeat(biases, curr_height * curr_width).reshape(
            1, convout_channels, curr_height, curr_width
        )

        a.append(act)
        h.append(relu(a[-1]))

    w, b = params[-1]
    act = jnp.dot(w, h[-1].flatten()) + b
    a.append(act)

    output = jax.nn.sigmoid(a[-1])
    h.append(output)

    return h, a

# upgrade to handle batches using 'vmap'
def build_batchforward():
    return jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))

# new noisy forward pass:
def noisyforward(x, params, randkey):
    x = x.reshape(
        1, height, width, channels
    ).astype(np.float32)  # NHWC

    x = jnp.transpose(x, [0, 3, 1, 2])

    h = []
    a = []
    xi = []
    aux = []
    h.append(x)

    curr_height = height
    curr_width = width

    for ind, (kernel, biases) in enumerate(params[:-1]):
        stride = 1
        if ind == 1 or ind == 3:
            stride = 2
            curr_height = int(curr_height / 2)
            curr_width = int(curr_width / 2)

        convout_channels = kernel.shape[-2]
        h[-1] = jax.lax.stop_gradient(h[-1])

        act = jax.lax.conv(
            h[-1],  # lhs = NCHW image tensor
            kernel.transpose(
                [2, 3, 0, 1]
            ),  # rhs = IOHW conv kernel tensor [according to JAX page]
            (stride, stride),  # window strides
            "SAME",
        )  # padding mode

        act = act + jnp.repeat(biases, curr_height * curr_width).reshape(
            1, convout_channels, curr_height, curr_width
        )

        randkey, _ = random.split(randkey)
        noise = nodepert_noisescale * random.normal(randkey, act.shape)
        xi.append(noise)
        a.append(act + noise)
        aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))
        h.append(relu(a[-1]))

    h[-1] = jax.lax.stop_gradient(h[-1])
    w, b = params[-1]
    act = jnp.dot(w, h[-1].flatten()) + b
    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale * random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    aux.append(jnp.sum(a[-1] * noise * (1 / (nodepert_noisescale ** 2))))

    output = jax.nn.sigmoid(a[-1])
    h.append(output)

    return h, a, xi, aux

# upgrade to handle batches using 'vmap'
def build_batchnoisyforward():
    return jit(vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0)))
