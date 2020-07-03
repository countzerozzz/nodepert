import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import grad_dynamics

### FUNCTIONALITY ###
# this code measures the grad_norm, (noise of gradient)_norm, sign symmetry and angle between 'true' gradient and gradient estimates when the network crashes.
# there is a crash if the current accuracy is less than 'x%' from the max accuracy. When a crash is detected, the network restarts from the last checkpoint
# and then performs the above mentioned calculations wrt the gradients.
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# folder to log experiment results
path = "explogs/crash_dynamics/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = 'learning_rate' 
rows = np.logspace(start=-3, stop=-1, num=10, endpoint=True, base=10, dtype=np.float32)
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
params_new = params

optimstate = { 'lr' : lr, 't' : 0}

test_acc = []
# num_batches = int(len(data.train_images) / batchsize)
num_batches = 10

# define metrics for measuring a crash
high = -1
crash = False
stored_epoch = -1

# log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(1, num_epochs+1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params_new, forward, data)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    if(high - test_acc[-1] > 25):
        print('crash detected! Resetting params')
        crash = True
        break

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params_new, grads, optimstate = gradfunc(x, y, params_new, randkey, optimstate)

    # every 5 epochs checkpoint the network params
    if(epoch % 5 == 0):
        Path(path).mkdir(exist_ok=True)
        pickle.dump(params, open(path + "model_params.pkl", "wb"))
        stored_epoch = epoch
    
    params = params_new
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

params = pickle.load(open(path + "model_params.pkl", "rb"))

if(crash):
    # dataframe to store the norm of the gradients: (np, sgd and true gradient)
    grad_norms_df = pd.DataFrame(columns = ['gnorm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    grad_norms_df['update_rule'] = ""
    # dataframe to store the norm of noise in the np and sgd gradients, e.g. norm(gradtrue - gradnp)
    graddiff_norms_df = pd.DataFrame(columns = ['gdiff_norm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    graddiff_norms_df['update_rule'] = ""
    # dataframe to store the sign sign_symmetry in the np and sgd gradient, e.g. ss(gradtrue, gradnp)
    sign_symmetry_df = pd.DataFrame(columns = ['ss_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    sign_symmetry_df['update_rule'] = ""
    # dataframe to store the angles of np and sgd gradient with the true gradient, e.g. angle(gradtrue, gradnp)
    grad_angles_df = pd.DataFrame(columns = ['gangle_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    grad_angles_df['update_rule'] = ""
    
    for epoch in range(stored_epoch, stored_epoch + 5):
        print('calculating dynamics for epoch {}.'.format(epoch))
        if(train.compute_metrics(params, forward, data)[1] < 15):
            break

        for ii in range(num_batches):
            #here will have to check that a new batch (not the same x,y) is being called everytime!
            x, y = next(data.get_data_batches(batchsize=batchsize, split=data.trainsplit))
            randkey, _ = random.split(randkey)
            params_new, npgrad, _ = optim.npupdate(x, y, params, randkey, optimstate)
            _, sgdgrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)

            x, y = next(data.get_data_batches(batchsize=5000, split=data.trainsplit))
            _, truegrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)
            
            grad_norms_df = grad_norms_df.append(grad_dynamics.grad_norms(npgrad, sgdgrad, truegrad, layer_sizes))
            graddiff_norms_df = graddiff_norms_df.append(grad_dynamics.graddiff_norms(npgrad, sgdgrad, truegrad, layer_sizes))
            sign_symmetry_df = sign_symmetry_df.append(grad_dynamics.sign_symmetry(npgrad, sgdgrad, truegrad, layer_sizes))
            grad_angles_df = grad_angles_df.append(grad_dynamics.grad_angles(npgrad, sgdgrad, truegrad, layer_sizes))
        
            params = params_new
            
else:
    print("no crash detected, exiting...")
    exit()

pd.set_option('display.max_columns', None)
print(grad_norms_df.head(10))
print(graddiff_norms_df.head(10))
print(sign_symmetry_df.head(10))
print(grad_angles_df.head(10))

train_df = pd.DataFrame()
train_df['test_acc'] = test_acc
train_df['epoch'] = np.arange(start=1, stop=len(test_acc)+1, dtype=int) 
train_df['network'], train_df['update_rule'], train_df['n_hl'], train_df['lr'], train_df['batchsize'], train_df['hl_size'], train_df['total_epochs'], train_df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
print(train_df.head(10))

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    grad_norms_df.to_csv(path + 'grad_norms.csv', mode='a', header=True)
    graddiff_norms_df.to_csv(path + 'graddiff_norms.csv', mode='a', header=True)
    sign_symmetry_df.to_csv(path + 'sign_symmetry.csv', mode='a', header=True)
    grad_angles_df.to_csv(path + 'grad_angles.csv', mode='a', header=True)
    train_df.to_csv(path + 'train_df.csv', mode='a', header=True)
    