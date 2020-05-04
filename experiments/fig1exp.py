import npimports
from npimports import *

# define some high level constants
config = {}
config['num_epochs'] = num_epochs = 10000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 100, 100, 100, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

forward = fc.batchforward
optimizer = optim.npupdate
# optimizer = optim.sgdupdate
optimstate = {  'lr' : 4e-6, 't' : 0 }

# now train
params, optimstate, expdata = train.train(params, forward, data, config, optimizer, randkey, verbose=True)

# make experiment log directory and write data
pickle.dump(expdata, open("explogs/exp_data.pickle", "wb"))
pickle.dump(params, open("explogs/params.pickle", "wb"))


layer_sizes = [data.num_pixels, 1000, 1000, 1000, data.num_classes]


layer_sizes = [data.num_pixels, 10000, 10000, 10000, data.num_classes]
