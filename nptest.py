import npimports
import importlib

importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

log_expdata = False
path = "explogs/train/"

# define training configs
config = {}
config["num_epochs"] = num_epochs = 15
config["batchsize"] = batchsize = 100
config["compute_norms"] = False

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

optimizer = optim.sgdupdate
optimstate = {"lr": 0.25, "wd": 0, "t": 0}

# now train
params, optimstate, expdata = train.train(
    params, forward, data, config, optimizer, optimstate, randkey, verbose=False
)

# save out results of experiment
if log_expdata:
    Path(path).mkdir(exist_ok=True)
    pickle.dump(expdata, open(path + "traindata.pkl", "wb"))

# plotting:
# import matplotlib.pyplot as pp
# pp.plot(expdata['train_acc'])
# pp.plot(expdata['test_acc'])
# pp.show()
