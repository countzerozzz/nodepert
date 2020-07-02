import npimports
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

# define training configs
config = {}
config['num_epochs'] = num_epochs = 2000
config['batchsize'] = batchsize = 100
config['num_classes'] = num_classes = data.num_classes

# build our network
layer_sizes = [data.num_pixels, 500, 500, data.num_classes]
print("Network structure: {}".format(layer_sizes))

randkey, _ = random.split(randkey)
origparams = fc.init(layer_sizes, randkey)

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchforward
optim.forward = fc.batchforward
optim.noisyforward = fc.batchnoisyforward

optimizer = optim.npupdate
optimstate = { 'lr' : 5e-3, 'wd' : 1e-4, 't' : 0 }

weightdecays = [0.0, 1e-5, 1e-4, 1e-3]

# define the experiment results directory
path = "explogs/weightdecay/"
try:
    os.mkdir(path)
except OSError:
    print ("creation of directory %s failed" % path)
else:
    print ("created the directory %s " % path)

npwdparams = {}
npwdexpdata = {}

for wd in weightdecays:
    optimstate['wd'] = wd
    params = fc.copyparams(origparams)

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
pickle.dump(npwdexpdata, open(path + "npwdexpdata.pickle", "wb"))
pickle.dump(npwdparams, open(path + "npwdparams.pickle", "wb"))
