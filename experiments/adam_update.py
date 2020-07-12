import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code updates network weights with an Adam-like update rule using the NP gradients. Storing the final accuracy reached by the network after some 
# number of epochs. Paper: https://arxiv.org/pdf/1412.6980.pdf
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# folder to log experiment results
path = "explogs/"
randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

rows = np.logspace(start=-5, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
# adam usually requires a smaller learning rate
if(re.search('adam', update_rule)):
    rows = np.logspace(start=-6, stop=-2, num=25, endpoint=True, base=10, dtype=np.float32)

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
forward = fc.batchforward

if(re.search('np', update_rule)):
    gradfunc = optim.npupdate
elif(re.search('sgd', update_rule)):
    gradfunc = optim.sgdupdate

params = fc.init(layer_sizes, randkey)
itercount = itertools.count()

@jit
def update(i, grads, opt_state):
    return opt_update(i, grads, opt_state)

# initialize the built-in JAX adam optimizer
opt_init, opt_update, get_params = optimizers.adam(lr)

opt_state = opt_init(params)
itercount = itertools.count()

# during adam update, no use of this optimstate here and dummy value passed as the lr. This is just to get the np gradients.
optimstate = { 'lr' : lr, 't' : 0 }
test_acc = []

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    
    test_acc.append(train.compute_metrics(params, forward, data)[1])

    print('EPOCH ', epoch)
    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        # get the gradients, throw away the traditional weight updates
        new_params, grads, _ = gradfunc(x, y, params, randkey, optimstate)
        
        if(re.search('adam', update_rule)):
            # pass the gradients to the JAX adam function
            opt_state = update(next(itercount), grads, opt_state)
            new_params = get_params(opt_state)

        params = new_params

    epoch_time = time.time() - start_time
    print('epoch training time: {}\n test acc: {}\n'.format(round(epoch_time, 2), round(test_acc[-1], 3)))

df = pd.DataFrame()
pd.set_option('display.max_columns', None)
# take the mean of the last 5 epochs as the final accuracy
df['final_acc'] = [np.mean(test_acc[-5:])]
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['total_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'adam_update.csv')):
        use_header = True
    
    df.to_csv(path + 'adam_update.csv', mode='a', header=use_header)
    