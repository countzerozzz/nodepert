import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code is for finding the scalability of node perturbation with width for a single hidden layer, fully connected non-linear networks
###

config = {}
# parse arguments
network, update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], log_expdata, jobid = utils.parse_args()
config['compute_norms'] = False

# folder to log experiment results
path = "explogs/scaling/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

ROW_DATA = 'network width' 
rows = [100, 500, 1000, 5000, 10000]
row_id = jobid % len(rows)
hl_size = rows[row_id]
n_hl = 1

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

optimstate = { 'lr' : lr, 't' : 0}

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            gradfunc,
                                            optimstate,
                                            randkey,
                                            verbose = False)

train_df = pd.DataFrame.from_dict(expdata)

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'width.csv')):
        train_df.to_csv(path + 'width.csv', mode='a', header=True)
    else:
        train_df.to_csv(path + 'width.csv', mode='a', header=False)
    