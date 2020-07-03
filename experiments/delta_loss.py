import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code calculates the change in the loss function with every update as the training progresses
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# folder to log experiment results
path = "explogs/delta_loss/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

ROW_DATA = 'number of hidden layers' 
rows = [1, 2, 3, 4]
row_id = jobid % len(rows)
n_hl = rows[row_id]

# save the average delta_loss these many number of times in an epoch
log_frequency = 10

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

test_acc = []
del_loss = []

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params_new, forward, data)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    epochdel_loss = []
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params_new, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
        
        epochdel_loss.append(optim.loss(x, y, new_params) - optim.loss(x, y, params))
        params = params_new
    
    del_loss.extend(np.mean(epochdel_loss.reshape(-1, log_frequency), axis=1))
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

train_df = pd.DataFrame()
train_df['del_loss'] = del_loss
train_df['test_acc'] = np.repeat(test_acc, log_frequency)
epochs = np.arange(start=1, stop=len(test_acc)+1, dtype=int) 
train_df['epoch'] = np.repeat(epochs, log_frequency)
train_df['network'], train_df['update_rule'], train_df['n_hl'], train_df['lr'], train_df['batchsize'], train_df['hl_size'], train_df['total_epochs'], train_df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(file_path)):
        train_df.to_csv(path + 'train_df.csv', mode='a', header=True)
        delta_loss_df.to_csv(path + 'delta_loss.csv', mode='a', header=True)
    else:
        train_df.to_csv(path + 'train_df.csv', mode='a', header=False)
        delta_loss_df.to_csv(path + 'delta_loss.csv', mode='a', header=False)
    