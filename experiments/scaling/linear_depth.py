import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import data_loaders.mnistloader as data
#parse arguments
config = {}
update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], log_expdata = utils.parse_args()
path = 'explogs/scaling/depth/'
seed=int(time.time())
randkey = random.PRNGKey(seed)

# build our network
layer_sizes = [data.num_pixels]
for i in range(n_hl):
    layer_sizes.append(hl_size)
layer_sizes.append(data.num_classes)

randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
if(update_rule == 'np'):
    optimizer = optim.npupdate
elif(update_rule == 'sgd'):
    optimizer = optim.sgdupdate

# Note: The way of creating a linear fwd pass is currently a hack! - value of 'linear' in the optimstate dictionary has no use (dummy val). 
# The optim.npupdate function checks if 'linear' is present as a key in this dictionary and if so, calls the fc.batchlinforward. This hacky 
# method is used as the npupdate function is jitted and branching can't be made by evaluating a value passed to the function.

optimstate = { 'lr' : lr, 't' : 0 }

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = False)

# save out results of experiment
if(log_expdata):
    elapsed_time = np.sum(expdata['epoch_time'])
    meta_data=update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], elapsed_time
    utils.file_writer(path+'n_hl'+str(n_hl)+'.pkl', expdata, meta_data)
