import npimports
import importlib
importlib.reload(npimports)
from npimports import *

#parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

path = os.path.join("explogs/train", update_rule)
file_path = os.path.join(path, 'my_csv.csv')

randkey = random.PRNGKey(jobid)

rows = np.logspace(start=-4, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
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

optimstate = { 'lr' : lr, 't' : 0 , 'linear' : 0}

test_acc = []

for epoch in range(1, num_epochs+1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data)[1])
    
    print('EPOCH ', epoch)

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)

    epoch_time = time.time() - start_time
    print('epoch training time: {}\n test acc: {}\n'.format(round(epoch_time,2), round(test_acc[-1], 3)))

df = pd.DataFrame()
df['test_acc'] = test_acc
df['epoch'] = np.arange(start=1, stop=num_epochs+1, dtype=int) 
df['network'], df['update_rule'], df['n_hl'], df['lr'], df['batchsize'], df['hl_size'], df['num_epochs'], df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
pd.set_option('display.max_columns', None)
print(df.head(10))

if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(file_path)):
        df.to_csv(file_path, mode='a', header=True)
    else:
        df.to_csv(file_path, mode='a', header=False)
