import npimports
from npimports import *

randkey = random.PRNGKey(int(time.time()))

# define training configs
config = {}
config['num_epochs'] = num_epochs = 20
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 500, 500, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
optimizer = optim.sgdupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

# now train
params, optimstate, exp_data = train.train( params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)
