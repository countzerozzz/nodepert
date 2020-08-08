import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code tracks the change in the loss with every update as the training progresses. Test accuracy is measured after every epoch whereas the 
# average change in the loss is evaluated 'log_frequency' number of times in an epoch.
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# folder to log experiment results
path = "explogs/delta_loss/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

ROW_DATA = 'network depth' 
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
    test_acc.append(train.compute_metrics(params, forward, data)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    batchdel_loss = []
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params_new, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
        
        batchdel_loss.append(optim.loss(x, y, params_new) - optim.loss(x, y, params))
        params = params_new
    
    del_loss.extend(np.mean(np.array(batchdel_loss).reshape(-1, log_frequency), axis=0))
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

df = pd.DataFrame()
df['del_loss'] = del_loss
df['test_acc'] = np.repeat(test_acc, log_frequency)
epochs = np.arange(start=1, stop=len(test_acc)+1, dtype=int) 
df['epoch'] = np.repeat(epochs, log_frequency)
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid

pd.set_option('display.max_columns', None)
print(df.head(15))

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'expdata.csv')):
        df.to_csv(path + 'expdata.csv', mode='a', header=True)
    else:
        df.to_csv(path + 'expdata.csv', mode='a', header=False)
    