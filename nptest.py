import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

log_expdata = False
path = 'explogs/train/'

# define training configs
config = {}
config['num_epochs'] = num_epochs = 2000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 500, 500, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
# forward = fc.batchforward
forward = optim.forward = fc.batchforward
optim.forward = fc.batchforward
optim.noisyforward = fc.batchnoisyforward

optimizer = optim.npupdate
optimstate = { 'lr' : 5e-3, 'wd' : 1e-4, 't' : 0 }

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)

# save out results of experiment
if(log_expdata):
    pickle.dump(expdata, open(path + "100e.pickle", "wb"))

# plotting:
# import matplotlib.pyplot as pp
# pp.plot(expdata['train_acc'])
# pp.plot(expdata['test_acc'])
# pp.show()
