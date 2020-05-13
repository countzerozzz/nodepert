import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

num_epochs = 3
batchsize = 100
log_expdata = True

lr = 1e-4

#define all networks
networks = [
# [data.num_pixels, 500, data.num_classes]
# [data.num_pixels, 500, 500, data.num_classes]
[data.num_pixels, 500, 500, 500, data.num_classes],
# [data.num_pixels, 500, 500, 500, 500, data.num_classes]
]

randkey, _ = random.split(randkey)

forward = fc.batchforward
optimizer = optim.npupdate

optimstate = { 'lr' : lr, 't' : 0 }

expdata={}
nn_metrics={}
nn_metrics['delta_mse']=[]
nn_metrics['test_acc']=[]

for layer_sizes in networks:
    n_hl = len(layer_sizes) - 2
    print("Network structure: {}".format(layer_sizes))
    params = fc.init(layer_sizes, randkey)
    for epoch in range(1, num_epochs+1):
        print('EPOCH ', epoch)
        start_time = time.time()
        
        nn_metrics['test_acc'].append(train.compute_metrics(params, forward, data)[1])
        cnt=0
        for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
            randkey, _ = random.split(randkey)
            params, grads, optimstate = optimizer(x, y, params, randkey, optimstate)
            # to get values 2 times an epoch!
            if(cnt%400==0):
                nptmp=[]
                for x_ii, y_ii in data.get_data_batches(batchsize=batchsize, split='test[:10%]'):
                    nptmp.append(compute_linesearch.get_delta_l(x_ii, y_ii, params, randkey, optimstate, update_step=jnp.zeros(1)))
                nn_metrics['delta_mse'].append(jnp.mean(jnp.array(nptmp)))

            cnt+=1
        epoch_time = time.time() - start_time
        print('train time: ', epoch_time)
        # print('epoch training time: {} test acc: {}'.format(round(epoch_time,2), round(test_acc[-1], 3)))

    expdata.update({n_hl : nn_metrics})

print('training complete!\n')
print(expdata)

path = "explogs/avg_mseexp/"

# save out results of experiment
if(log_expdata):
    pickle.dump(expdata, open(path + "delta_mse2e_5.pickle", "wb"))