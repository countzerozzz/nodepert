import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# simple experiments with weight decay to see if this helps NP since the norm of the weights expands much more as compared to SGD. However, we see that 
# it doesn't really help much, which adds fuel to the already huge fire that is node perturbation is beyond saving!!
###

config = {}
# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()
config['compute_norms'], config['batchsize'], config['num_epochs'] = False, batchsize, num_epochs

# folder to log experiment results
path = "explogs/"

# start all the runs with the same weight initializations to get an accurate comparison of the effects of weight decay
randkey = random.PRNGKey(0)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

num = 25 # number of learning rates

rows = np.logspace(-5, -1, num, endpoint=True, base=10, dtype=np.float32)
ROW_DATA = 'learning_rate'
row_id = jobid % len(rows)
lr = rows[row_id]
# weight decay : [0.0, 1e-4, 1e-3, 1e-2]
wd = 0.0

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
    gradfunc = optim.npwdupdate
elif(update_rule == 'sgd'):
    gradfunc = optim.sgdwdupdate

params = fc.init(layer_sizes, randkey)

optimstate = { 'lr' : lr, 't' : 0, 'wd': wd}

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
df['wd'] = wd
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'weight_decay.csv')):
        df.to_csv(path + 'weight_decay.csv', mode='a', header=True)
    else:
        df.to_csv(path + 'weight_decay.csv', mode='a', header=False)