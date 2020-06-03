import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import data_loaders.mnistloader as data

#parse arguments
config = {}
update_rule, n_hl, lrtrain, config['batchsize'], hl_size, config['num_epochs'], log_expdata = utils.parse_args()
path = 'explogs/analysis/linesearch/'
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
    optimizer = optim.npupdate
elif(update_rule == 'sgd'):
    optimizer = optim.sgdupdate

forward = fc.batchforward
batchmseloss = losses.batchmseloss

@jit
def lossfunc(x, y, params):
    h, a = forward(x, params)
    loss = batchmseloss(h[-1], y).sum()
    return loss


def linesearchfunc(randkey, start, stop, num):
    learning_rates=np.logspace(start, stop, num, endpoint=True, base=10, dtype=np.float32)
    df = pd.DataFrame()
    df['lr'] = learning_rates
    npdelta_l = []
    sgddelta_l = []

    for lr in learning_rates:
        nptmp = []
        sgdtmp = []
        optimstate = { 'lr' : lr, 't' : 0 }
        
        #set the percentage of test samples which should be averaged over, while calculating delta_L 
        for x, y in data.get_data_batches(batchsize=100, split='test[:25%]'):
            randkey, _ = random.split(randkey)
            params_new, _, _ = optim.npupdate(x, y, params, randkey, optimstate)
            nptmp.append(lossfunc(x,y, params_new) - lossfunc(x,y, params))
            params_new, _, _ = optim.sgdupdate(x, y, params, randkey, optimstate)
            sgdtmp.append(lossfunc(x,y, params_new) - lossfunc(x,y, params))

        npdelta_l.append(jnp.mean(jnp.array(nptmp)))
        sgddelta_l.append(jnp.mean(jnp.array(sgdtmp)))

        print('NP - learning rate {} , average Delta_L {}'.format(lr, npdelta_l[-1]))
        print('SGD - learning rate {} , average Delta_L {}\n'.format(lr, sgddelta_l[-1]))
    
    df['np'] = npdelta_l
    df['sgd'] = sgddelta_l

    return df

# the points along the training trajectory which we want to compute linesearch
linesearch_points=[0, 95]

params = fc.init(layer_sizes, randkey)
optimstate = { 'lr' : lrtrain, 't' : 0 }
expdata = {}
ptr=0

for epoch in range(1, config['num_epochs']+1):
    train_acc, test_acc = train.compute_metrics(params, forward, data)
    
    if(train_acc > linesearch_points[ptr]):
        print('\ncomputing linesearch, network snapshot @ {} train acc'.format(round(train_acc,3)))
        start_time = time.time()
        #between base^(start) and base^(stop), will perform linesearch for 'num' values in a loginterval
        expdata.update({linesearch_points[ptr] : linesearchfunc(randkey, start=0, stop=-4, num=25)})
        print('time to perform linesearch : ', round((time.time()-start_time), 2), ' s')
        ptr+=1
        if(ptr==len(linesearch_points)):
            break

    print('EPOCH ', epoch)
    start_time = time.time()

    for x, y in data.get_data_batches(batchsize=config['batchsize'], split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    epoch_time = time.time() - start_time
    print('epoch training time: {} train acc: {}'.format(round(epoch_time,2), round(train_acc, 3)))

# save out results of experiment
if(log_expdata):
    elapsed_time = 0
    meta_data=update_rule, n_hl, lrtrain, config['batchsize'], hl_size, config['num_epochs'], elapsed_time
    utils.file_writer(path+'expdata.pkl', expdata, meta_data)
