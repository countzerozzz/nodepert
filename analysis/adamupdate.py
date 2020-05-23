import npimports
import importlib
importlib.reload(npimports)
from npimports import *

#parse arguments
config = {}
update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], log_expdata = utils.parse_args()
path = 'explogs/analysis/adamupdate/'
seed=int(time.time())
randkey = random.PRNGKey(seed)

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
forward = fc.batchforward
itercount = itertools.count()

@jit
def update(i, grads, opt_state):
    return opt_update(i, grads, opt_state)

opt_init, opt_update, get_params = optimizers.adam(lr)

opt_state = opt_init(params)
itercount = itertools.count()

expdata = {}

#no use of this optimstate here
optimstate = { 'lr' : 0, 't' : 0 }

for epoch in range(1, config['num_epochs']+1):
    start_time=time.time()
    
    train_acc, test_acc = train.compute_metrics(params, forward, data)

    print('EPOCH ', epoch)
    for x, y in data.get_data_batches(batchsize=config['batchsize'], split=data.trainsplit):
        randkey, _ = random.split(randkey)
          # update_step passing [jnp.zeros(1) = SGD, np.zeros(2) =np]
        _, grads, _ = gradfunc(x, y, params, randkey, optimstate)
        opt_state = update(next(itercount), grads, opt_state)
        params = get_params(opt_state)

    epoch_time = time.time() - start_time
    print('epoch training time: {}\n train acc: {}  test acc: {}\n'.format(round(epoch_time,2), round(train_acc, 3), round(test_acc, 3)))

# save out results of experiment
if(log_expdata):
    elapsed_time = np.sum(expdata['epoch_time'])
    meta_data=update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], elapsed_time
    utils.file_writer(path+'expdata.pkl', expdata, meta_data)
