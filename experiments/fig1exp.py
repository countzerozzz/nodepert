import npimports
from npimports import *

randkey = random.PRNGKey(int(time.time()))

data.trainsplit = 'train[:10%]'
data.testsplit = 'test[:10%]'

# define training configs
config = {}
config['num_epochs'] = num_epochs = 200
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 100, 100, data.num_classes]
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
optimizer = optim.npupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

learning_rates = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]

# define the experiment results directory
path = "explogs/fig1exp/"
try:
    os.mkdir(path)
except OSError:
    print ("creation of directory %s failed" % path)
else:
    print ("created the directory %s " % path)

npparams = {}
npexpdata = {}

for lr in learning_rates:
    optimstate['lr'] = lr
    randkey, _ = random.split(randkey)
    params = fc.init(layer_sizes, randkey)

    # now train
    params, optimstate, expdata = train.train( params,
                                               forward,
                                               data,
                                               config,
                                               optimizer,
                                               optimstate,
                                               randkey,
                                               verbose = True)
    npparams.update({lr : params})
    npexpdata.update({lr : expdata})

# save out results of experiment
pickle.dump(npexpdata, open(path + "npexpdata.pickle", "wb"))
pickle.dump(npparams, open(path + "npparams.pickle", "wb"))



# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
optimizer = optim.sgdupdate
optimstate = { 'lr' : 5e-5, 't' : 0 }

sgdparams = {}
sgdexpdata = {}

for lr in learning_rates:
    optimstate['lr'] = lr
    randkey, _ = random.split(randkey)
    params = fc.init(layer_sizes, randkey)

    # now train
    params, optimstate, expdata = train.train( params,
                                               forward,
                                               data,
                                               config,
                                               optimizer,
                                               optimstate,
                                               randkey,
                                               verbose = True)
    sgdparams.update({lr : params})
    sgdexpdata.update({lr : expdata})

# save out results of experiment
pickle.dump(sgdexpdata, open(path + "sgdexpdata.pickle", "wb"))
pickle.dump(sgdparams, open(path + "sgdparams.pickle", "wb"))
