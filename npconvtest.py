import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

# define training configs
config = {}
config['num_epochs'] = num_epochs = 2000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, 1, 32), (3, 3, 32, 32), (3, 3, 32, 32)]
fclayer_sizes = [conv.imgheight*conv.imgwidth*convlayer_sizes[-1][-1] , data.num_classes]

randkey, _ = random.split(randkey)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

# get forward pass, optimizer, and optimizer state + params
forward = optim.forward = conv.batchforward
optim.forward = conv.batchforward
optim.noisyforward = conv.batchnoisyforward

optimizer = optim.npupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)
