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
# layer_sizes = [data.num_pixels, 50, 50, data.num_classes]
# randkey, _ = random.split(randkey)
# params = fc.init(layer_sizes, randkey)
# print("Network structure: {}".format(layer_sizes))

optimstate = { 'lr' : 1e-3, 't' : 0 }

x, y = next(data.get_data_batches())

randkey, _ = random.split(randkey)

#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, 1, 32), (3, 3, 32, 32), (3, 3, 32, 32)]
fclayer_sizes = [conv.imgheight*conv.imgwidth*convlayer_sizes[-1][-1] , data.num_classes]

randkey, _ = random.split(randkey)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

convnetparams = convparams
convnetparams.append(fcparams)

# h, a = conv.batchforward(x, convnetparams)

h, a, xi, aux = conv.batchnoisyforward(x, convnetparams, randkey)




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
