import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# set the 'seed' for our experiment
# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

log_expdata = False
path = 'explogs/'

# parse FC network arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# define training configs
config = {}
config['num_epochs'] = num_epochs = 10
config['batchsize'] = batchsize = 100
config['compute_norms'] = False

# build our network
layer_sizes = [data.num_pixels, 500, 500, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))
print(xla_bridge.get_backend().platform) # are we running on CPU or GPU?

# get forward pass, optimizer, and optimizer state + params

forward = optim.forward = fc.batchforward
optim.forward = fc.batchforward
optim.noisyforward = fc.batchnoisyforward

if(update_rule == 'np'):
    optimizer = optim.npupdate
elif(update_rule == 'sgd'):
    optimizer = optim.sgdupdate

optimstate = { 'lr' : lr}

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = False)

df = pd.DataFrame.from_dict(expdata)
df['dataset'] = npimports.dataset
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
pd.set_option('display.max_columns', None)
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'fc-test.csv')):
        use_header = True
    
    df.to_csv(path + 'fc-test.csv', mode='a', header=use_header)