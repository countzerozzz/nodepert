import npimports
import importlib
importlib.reload(npimports)
from npimports import *

randkey = random.PRNGKey(0)

# define training configs
config = {}
config['num_epochs'] = num_epochs = 100
config['batchsize'] = batchsize = 100

config['num_classes'] = num_classes = data.num_classes
# build our network
layer_sizes = [data.num_pixels, 50, 50, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

optimstate = { 'lr' : 1e-3, 't' : 0 }

x, y = next(data.get_data_batches())

randkey, _ = random.split(randkey)

h, a, xi, aux = fc.batchnewnoisyforward(x, params, randkey)

optim.nploss(x, y, params, randkey)

optim.newnpupdate(x, y, params, randkey, optimstate)

# optim.


# # get forward pass, optimizer, and optimizer state + params
# forward = fc.batchforward
# optimizer = optim.npupdate
# optimstate = { 'lr' : 1e-3, 't' : 0 }
#
# # use this if you don't want to wait as long:
# # data.trainsplit = 'train[:5%]'
# # data.testsplit = 'test[:5%]'
#
# # now train
# params, optimstate, expdata = train.train( params,
#                                             forward,
#                                             data,
#                                             config,
#                                             optimizer,
#                                             optimstate,
#                                             randkey,
#                                             verbose = True)


# W = random.normal(randkey, (3, 5))
#
# randkey, _ = random.split(randkey)
# x = random.normal(randkey, (5,))
#
# randkey, _ = random.split(randkey)
# noise = random.normal(randkey, (3,))
#
#
#
# def myfunc(W, x):
#     y = jnp.dot(W,x) + noise
#     loss = jnp.sum(y*noise)
#     return loss
#
#
# dlossdW = jax.grad(myfunc, 0)
