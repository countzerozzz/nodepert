import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code is for finding the scalability of node perturbation with depth (constant width) for fully connected linear networks
# note: critical lr for a linear network is much smaller!
###

config = {}
# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()
config['compute_norms'], config['batchsize'], config['num_epochs'] = False, batchsize, num_epochs

# folder to log experiment results
path = "explogs/scaling/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

num = 25 # number of learning rates

rows = np.logspace(-6, -3, num, endpoint=True, base=10, dtype=np.float32)
ROW_DATA = 'learning_rate'
row_id = jobid % len(rows)
lr = rows[row_id]

# build our network
layer_sizes = [data.num_pixels]
for i in range(n_hl):
    layer_sizes.append(hl_size)
layer_sizes.append(data.num_classes)

randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = fc.batchlinforward
if(update_rule == 'np'):
    gradfunc = optim.npupdate
elif(update_rule == 'sgd'):
    gradfunc = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

# Note: The way of creating a linear fwd pass is currently a hack! - value of 'linear' in the optimstate dictionary has no use (dummy val). 
# The optim.npupdate function checks if 'linear' is present as a key in this dictionary and if so, calls the fc.batchlinforward. This hacky 
# method is used as the npupdate function is jitted and branching can't be made by evaluating a value passed to the function.

optimstate = { 'lr' : lr, 't' : 0, 'linear': 1}

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            gradfunc,
                                            optimstate,
                                            randkey,
                                            verbose = False)

df = pd.DataFrame.from_dict(expdata)

pd.set_option('display.max_columns', None)
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'depth_linear.csv')):
        df.to_csv(path + 'depth_linear.csv', mode='a', header=True)
    else:
        df.to_csv(path + 'depth_linear.csv', mode='a', header=False)
    