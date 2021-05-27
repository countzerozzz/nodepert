import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import grad_dynamics
from linesearch_utils import lossfunc

### FUNCTIONALITY ###
# this code is for getting the epoch at which a crash is detected. When the accuracy falls more than 25% from the 
# maximum achieved accuracy, we define a crash.
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# folder to log experiment results
path = "explogs/crash_timing/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = 'learning_rate' 
# for NP
# rows = np.logspace(start=-3, stop=-1, num=50, endpoint=True, base=10, dtype=np.float32)
# for SGD
rows = np.logspace(start=0, stop=2, num=20, endpoint=True, base=10, dtype=np.float32)

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
forward = fc.batchforward

if(update_rule == 'np'):
    gradfunc = optim.npupdate
elif(update_rule == 'sgd'):
    gradfunc = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

optimstate = { 'lr' : lr, 't' : 0}

test_acc = []

# define metrics for measuring a crash
is_training = False; crash = False
high = -1; crash_epoch = -1

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    
    if(high > 25):
        is_training = True
    
    if(high - test_acc[-1] > 25):
        print('crash detected! Resetting params')
        crash = True
        crash_epoch = epoch
        break

    for x, y in data.get_data_batches(batchsize=batchsize):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
    
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

    if(is_training == False and epoch > 50):
        break

df = pd.DataFrame()
df['crash'], df['crash_epoch'], df['is_training'], df['highest_acc'] = [crash], [crash_epoch], [is_training], [high]
df['dataset'] = npimports.dataset
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid

if(crash):
    print("crash detected at epoch: {}".format(crash_epoch))
            
else:
    print("no crash detected ...")

pd.set_option('display.max_columns', None)
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'expdata.csv')):
        use_header = True
        
    df.to_csv(path + 'expdata.csv', mode='a', header=use_header)
