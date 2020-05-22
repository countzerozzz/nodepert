import npimports
import importlib
importlib.reload(npimports)
from npimports import *

randkey = random.PRNGKey(int(time.time()))

# define training configs
config = {}
config['num_epochs'] = num_epochs = 1000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
# layer_sizes = [data.num_pixels, 500, 500, data.num_classes]
# randkey, _ = random.split(randkey)
# params = fc.init(layer_sizes, randkey)
#
# randkey, _ = random.split(randkey)




#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, 1, 32), (3, 3, 32, 32), (3, 3, 32, 32)]
fclayer_sizes = [conv.imgheight*conv.imgwidth*convlayer_sizes[-1][-1] , data.num_classes]

randkey, _ = random.split(randkey)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

convnetparams = convparams
convnetparams.append(fcparams)

# get forward pass, optimizer, and optimizer state + params
forward = conv.batchforward
optimizer = optim.npupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

# use this if you don't want to wait as long:
# data.trainsplit = 'train[:5%]'
# data.testsplit = 'test[:5%]'

x, y = next(data.get_data_batches())

# params, grads, optimstate = optimizer(x, y, convparams, randkey, optimstate)

# now train
params, optimstate, expdata = train.train(  convparams,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)



# plotting:
# import matplotlib.pyplot as pp
# pp.plot(expdata['train_acc'])
# pp.plot(expdata['test_acc'])
# pp.show()
