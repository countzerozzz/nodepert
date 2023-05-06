import jax.random as random
import numpy as np
import nodepert.network_init.fc as fc
import nodepert.model.conv as conv

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
def init_conv_params(sizes, key):
    keys = random.split(key, len(sizes))
    # pdb.set_trace()
    params = []
    for ii in range(len(sizes)):
        params.append(init_single_convlayer(*sizes[ii], keys[ii]))
    return params


def init(randkey, args, data):

    # build our network
    convout_channels = [32, 32, 32]
    # format (kernel height, kernel width, input channels, output channels)
    convlayer_sizes = [(3, 3, prev_ch, curr_ch) for prev_ch, curr_ch in zip([data.channels] + convout_channels[:-1], convout_channels)]
    # because of striding, we need to downsample the final fc layer
    down_factor = 2
    fclayer_sizes = [
        int(
            (data.height / down_factor)
            * (data.width / down_factor)
            * convlayer_sizes[-1][-1]
        ),
        data.num_classes,
    ]

    key, key2 = random.split(randkey)
    convparams = init_conv_params(convlayer_sizes, key)
    fcparams = fc.init_single_fc_layer(fclayer_sizes[0], fclayer_sizes[1], key2)
    params = convparams + [fcparams]
    print(f"conv architecture {convlayer_sizes}, FC layer {fclayer_sizes}\n")

    conv.height = data.height
    conv.width = data.width
    conv.channels = data.channels

    forward = conv.build_batchforward()
    noisyforward = conv.build_batchnoisyforward()

    return params, forward, noisyforward