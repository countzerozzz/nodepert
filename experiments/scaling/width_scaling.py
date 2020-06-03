import npimports
import importlib
importlib.reload(npimports)
from npimports import *

#parse arguments
config = {}
update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], log_expdata = utils.parse_args()
path = 'explogs/scaling/width/'
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
    utils.file_writer(path+'hl_size'+str(hl_size)+'.pkl', expdata, meta_data)
