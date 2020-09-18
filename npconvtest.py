import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

# define training configs
config = {}
config['num_epochs'] = num_epochs = 10
config['batchsize'] = batchsize = 100
config['num_classes'] = data.num_classes
config['compute_norms'] = False

#length of convout_channels has to be same as convlayer_sizes!
convout_channels = [32, 32, 32]

#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, data.channels, convout_channels[0]),
                   (3, 3, convout_channels[0], convout_channels[1]),
                   (3, 3, convout_channels[1], convout_channels[2])]

fclayer_sizes = [data.height * data.width * convlayer_sizes[-1][-1], data.num_classes]

randkey, _ = random.split(randkey)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

# get forward pass, optimizer, and optimizer state + params
forward = conv.batchforward
optim.forward = conv.batchforward
optim.noisyforward = conv.batchnoisyforward

# optimizer = optim.sgdupdate
optimizer = optim.npupdate
optimstate = { 'lr' : 2e-4, 't' : 0 }

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)
