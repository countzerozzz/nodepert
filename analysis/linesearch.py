import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

num_epochs = 50
batchsize = 100
lr = 1e-3
# define the accuracy to train the network (sgd trajectory) after which we will perform the linesearch
end_acc=0.75


# build our network
layer_sizes = [data.num_pixels, 500, data.num_classes]
randkey, _ = random.split(randkey)

print("Network structure: {}".format(layer_sizes))

forward = fc.batchforward
#define the training trajectory to be followed
optimizer = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

print('starting training ...')
optimstate = { 'lr' : lr, 't' : 0 }

# for epoch in range(1, num_epochs+1):
#     print('EPOCH ', epoch)
#     start_time = time.time()

#     for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
#         randkey, _ = random.split(randkey)
#         params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

#     train_acc, test_acc = train.compute_metrics(params, forward, data)
#     epoch_time = time.time() - start_time
#     print('epoch training time: {} train acc: {}'.format(round(epoch_time,2), round(train_acc, 3)))
#     if(train_acc > end_acc):
#         break
  
# print('total training time : {} s'.format(round((time.time()-start_time), 2)))
print('training complete!\n')

start_time=time.time()
#between base^(start) and base^(stop), np.logspace gives 20 evenly spaced values in log interval
learning_rates=np.logspace(start=-1, stop=-7, num=10, endpoint=True, base=10, dtype=np.float32)
print('computing linesearch ...')

for lr in learning_rates:
    npdelta_l = []
    sgddelta_l = []
    optimstate = { 'lr' : lr, 't' : 0 }

    #set the percentage of test samples which should be averaged over, while calculating delta_L 
    for x, y in data.get_data_batches(batchsize=100, split='test[:20%]'):
        #update_step passing [jnp.zeros(1) = SGD, np.zeros(2) =np]
        sgddelta_l.append(compute_linesearch.get_delta_l(x, y, params, randkey, optimstate, update_step=jnp.zeros(1)))
        
    print('SGD - learning rate {} , average Delta_L {}'.format(lr, jnp.mean(jnp.array(sgddelta_l))))
    
    for x, y in data.get_data_batches(batchsize=100, split='test[:20%]'):
        npdelta_l.append(compute_linesearch.get_delta_l(x, y, params, randkey, optimstate, update_step=jnp.zeros(2)))
    
    print('NP - learning rate {} , average Delta_L {}'.format(lr, jnp.mean(jnp.array(npdelta_l))))

print('time to perform linesearch : ', round((time.time()-start_time), 2), ' s')