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
end_acc=0.92


# build our network
layer_sizes = [data.num_pixels, 500, data.num_classes]
randkey, _ = random.split(randkey)

print("Network structure: {}".format(layer_sizes))

batchmseloss = losses.batchmseloss
forward = fc.batchforward
noisyforward = fc.batchnoisyforward
#define the training trajectory to be followed
optimizer = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

print('starting training ...')
optimstate = { 'lr' : lr, 't' : 0 }

for epoch in range(1, num_epochs+1):
    print('EPOCH ', epoch)
    start_time = time.time()

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)

    train_acc, test_acc = train.compute_metrics(params, forward, data)
    epoch_time = time.time() - start_time
    print('epoch training time: {} train acc: {}'.format(round(epoch_time,2), round(train_acc, 3)))
    if(train_acc > end_acc):
        break
  
print('total training time : {} s'.format(round((time.time()-start_time), 2)))
print('training complete!\n')

@jit
def loss(x, y, params):
    h, a = forward(x, params)
    loss = batchmseloss(h[-1], y).sum()
    return loss

# main func for linesearch, computes the 'new' parameters of the network (w - lr*dw)
@jit
def get_new_params(x, y, params, randkey, optimstate):
  lr = optimstate['lr']
  sigma = fc.nodepert_noisescale
  randkey, _ = random.split(randkey)

  h, a, xi = noisyforward(x, params, randkey)
  noisypred = h[-1]

  h, a = forward(x, params)
  pred = h[-1]

  loss = jnp.mean(jnp.square(pred - y),1)
  noisyloss = jnp.mean(jnp.square(noisypred - y),1)
  lossdiff = (noisyloss - loss)/(sigma**2)

  grads=[]
  for ii in range(len(params)):
    dh = jnp.einsum('ij,i->ij', xi[ii], lossdiff)
    dw = jnp.einsum('ij,ik->kj', h[ii], dh)
    db = jnp.mean(dh, 0)
    grads.append((dw,db))

  return [(w - lr * dw, b - lr * db)
          for (w, b), (dw, db) in zip(params, grads)], grads, optimstate


start_time=time.time()
#between base^(start) and base^(stop), np.logspace gives 20 evenly spaced values in log interval
learning_rates=np.logspace(start=-1, stop=-7, num=20, endpoint=True, base=10, dtype=np.float32)
print('computing linesearch ...')

for lr in learning_rates:
    delta_l=[]
    optimstate = { 'lr' : lr, 't' : 0 }

    #set the percentage of test samples which should be averaged over, while calculating delta_L 
    for x, y in data.get_data_batches(batchsize=100, split='test[:20%]'):
        params_new, grads, optimstate = get_new_params(x, y, params, randkey, optimstate)
        delta_l.append(loss(x,y, params_new) - loss(x,y, params))

    print('learning rate {} , average Delta_L {}'.format(lr, jnp.mean(jnp.array(delta_l))))

print('time to perform linesearch : ', round((time.time()-start_time), 2), ' s')