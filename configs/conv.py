import jax.random as random
import model.fc as fc
import model.conv as conv

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
    convparams = conv.init_convlayers(convlayer_sizes, key)
    fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], key2)
    params = convparams + [fcparams]
    print(f"conv architecture {convlayer_sizes}, FC layer {fclayer_sizes}\n")

    conv.height = data.height
    conv.width = data.width
    conv.channels = data.channels

    forward = conv.build_batchforward()
    noisyforward = conv.build_batchnoisyforward()

    return params, forward, noisyforward