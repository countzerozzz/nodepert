import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(5)

num_epochs = 50
batchsize = 100
train_lr = 5e-3
# the points along the training trajectory which we want to compute linesearch
linesearch_points=[0, 0.90, 0.93, 0.97]

def linesearch_fn(randkey, start, stop, num):
    learning_rates=np.logspace(start, stop, num, endpoint=True, base=10, dtype=np.float32)
    npdelta_l=[]
    sgddelta_l=[]    
    
    for lr in learning_rates:
        nptmp = []
        sgdtmp = []
        optimstate = { 'lr' : lr, 't' : 0 }
        
        #set the percentage of test samples which should be averaged over, while calculating delta_L 
        for x, y in data.get_data_batches(batchsize=100, split='test[:20%]'):
            # update_step passing [jnp.zeros(1) = SGD, np.zeros(2) =np]
            nptmp.append(compute_linesearch.get_delta_l(x, y, params, randkey, optimstate, update_step=jnp.zeros(1)))
            sgdtmp.append(compute_linesearch.get_delta_l(x, y, params, randkey, optimstate, update_step=jnp.zeros(2)))

        npdelta_l.append(jnp.mean(jnp.array(nptmp)))
        sgddelta_l.append(jnp.mean(jnp.array(sgdtmp)))

        print('NP - learning rate {} , average Delta_L {}'.format(lr, npdelta_l[-1]))
        print('SGD - learning rate {} , average Delta_L {}\n'.format(lr, sgddelta_l[-1]))
    return (lr, npdelta_l, sgddelta_l)

# build our network
layer_sizes = [data.num_pixels, 500, data.num_classes]
randkey, _ = random.split(randkey)

print("Network structure: {}".format(layer_sizes))

forward = fc.batchforward
#define the training trajectory to be followed
optimizer = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

optimstate = { 'lr' : train_lr, 't' : 0 }
ptr=0

for epoch in range(1, num_epochs+1):
    train_acc, test_acc = train.compute_metrics(params, forward, data)
    
    if(train_acc > linesearch_points[ptr]):
        print('\ncomputing linesearch, network snapshot @ {} train acc'.format(round(train_acc,3)))
        start_time = time.time()
        #between base^(start) and base^(stop), will perform linesearch for these many values in a loginterval
        print(linesearch_fn(randkey, start=-1, stop=-7, num=10))
        print('time to perform linesearch : ', round((time.time()-start_time), 2), ' s')
        ptr+=1
        if(ptr==len(linesearch_points)):
            break

    print('EPOCH ', epoch)
    start_time = time.time()

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    epoch_time = time.time() - start_time
    print('epoch training time: {} train acc: {}'.format(round(epoch_time,2), round(train_acc, 3)))

print('training complete!\n')

