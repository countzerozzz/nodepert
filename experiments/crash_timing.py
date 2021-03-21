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
path = "explogs/crash_dynamics/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = 'learning_rate' 
# rows = np.logspace(start=-3, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
rows = np.linspace(0.005, 0.025, num=20)
# rows = [0.009, 0.01, 0.015, 0.02, 0.025]

row_id = jobid % len(rows)
lr = rows[row_id]
print('learning rate', lr)

split_percent = '[:10%]'
num_batches = int(6000 / batchsize)

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
high = -1; crash = False
stored_epoch = -1; interval = 3  # interval of batches to computer grad dynamics over after a crash.

# log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data, split_percent = split_percent)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    if(high - test_acc[-1] > 25):
        print('crash detected! Resetting params')
        crash = True
        break

    # every 5 epochs checkpoint the network params
    if((epoch-1) % 5 == 0):
        Path(path + "model_params/").mkdir(exist_ok=True)
        pickle.dump(params, open(path + "model_params/" + str(jobid) + ".pkl", "wb"))
        stored_epoch = epoch
    
    for x, y in data.get_data_batches(batchsize=batchsize, split='train'+split_percent):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
    
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

params = pickle.load(open(path + "model_params/" + str(jobid) + ".pkl", "rb"))
train_df = pd.DataFrame()
train_df['test_acc'] = test_acc

if(crash):

            
else:
    print("no crash detected, exiting...")
    exit()

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'w_norms.csv')):
        use_header = True
        
    deltal_df.to_csv(path + 'deltal.csv', mode='a', header=use_header)
