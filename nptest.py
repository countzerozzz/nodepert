import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

log_expdata=False
path = 'explogs/train/'
# define training configs
config = {}
config['num_epochs'] = num_epochs = 100
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 500, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
optimizer = optim.npupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

# use this if you don't want to wait as long:
# data.trainsplit = 'train[:5%]'
# data.testsplit = 'test[:5%]'

# now train
params, optimstate, expdata = train.train( params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = False)

# save out results of experiment
if(log_expdata):
    pickle.dump(expdata, open(path + "100e.pickle", "wb"))

# plotting:
# import matplotlib.pyplot as pp
# pp.plot(expdata['train_acc'])
# pp.plot(expdata['test_acc'])
# pp.show()
