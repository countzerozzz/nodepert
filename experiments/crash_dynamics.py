import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import grad_dynamics
from linesearch_utils import lossfunc

### FUNCTIONALITY ###
# this code measures the grad_norm, (noise of gradient)_norm, sign symmetry and angle between 'true' gradient and gradient estimates when the network crashes.
# there is a crash if the current accuracy is less than 'x%' from the max accuracy. When a crash is detected, the network restarts from the last checkpoint
# and then performs the above mentioned calculations wrt the gradients.
###

# parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

# set flags for which of the metrics to compute during the crash.
# gradient norms, difference in gradient norms, sign symmetry, gradient angles, weight norms, variance of weight
flags = [1, 1, 1, 1, 1, 1]

# folder to log experiment results
path = "explogs/crash_dynamics/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine, 
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = 'learning_rate' 
# rows = np.logspace(start=-3, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
rows = np.linspace(0.005, 0.025, num=20)
# rows = [0.009, 0.01, 0.015, 0.02, 0.025]

row_id = jobid % len(rows)
lr = rows[row_id]
print('learning rate', lr)

split_percent = '[:10%]'
num_batches = int(6000 / batchsize)

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

optimstate = { 'lr' : lr, 't' : 0}

test_acc = []

# define metrics for measuring a crash
high = -1; crash = False
stored_epoch = -1; interval = 3  # interval of batches to computer grad dynamics over after a crash.

# log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data, split_percent = split_percent)[1])
    print('EPOCH {}\ntest acc: {}%'.format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    if(high - test_acc[-1] > 25):
        print('crash detected! Resetting params')
        crash = True
        break

    # every 5 epochs checkpoint the network params
    if((epoch-1) % 5 == 0):
        Path(path + "model_params/").mkdir(exist_ok=True)
        pickle.dump(params, open(path + "model_params/" + str(jobid) + ".pkl", "wb"))
        stored_epoch = epoch
    
    for x, y in data.get_data_batches(batchsize=batchsize, split='train'+split_percent):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)
    
    epoch_time = time.time() - start_time
    print('epoch training time: {}s\n'.format(round(epoch_time,2)))

params = pickle.load(open(path + "model_params/" + str(jobid) + ".pkl", "rb"))
train_df = pd.DataFrame()
train_df['test_acc'] = test_acc

if(crash):
    test_acc = []
    # dataframe to store the norm of the gradients: (np, sgd and true gradient)
    grad_norms_df = pd.DataFrame(columns = ['gnorm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    grad_norms_df['update_rule'], grad_norms_df['epoch'] = "", ""
    # dataframe to store the norm of noise in the np and sgd gradients, e.g. norm(gradtrue - gradnp)
    graddiff_norms_df = pd.DataFrame(columns = ['gdiff_norm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    graddiff_norms_df['update_rule'], graddiff_norms_df['epoch'] = "", ""
    # dataframe to store the sign sign_symmetry in the np and sgd gradient, e.g. ss(gradtrue, gradnp)
    sign_symmetry_df = pd.DataFrame(columns = ['ss_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    sign_symmetry_df['update_rule'], sign_symmetry_df['epoch'] = "", ""
    # dataframe to store the angles of np and sgd gradient with the true gradient, e.g. angle(gradtrue, gradnp)
    grad_angles_df = pd.DataFrame(columns = ['gangle_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    grad_angles_df['update_rule'], grad_angles_df['epoch'] = "", ""
    # dataframe to store the norm of the neural network weights
    w_norms_df = pd.DataFrame(columns = [ 'norm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    w_norms_df['update_rule'], w_norms_df['epoch'] = "", ""
    # dataframe to store the variance of the neural network weights
    w_var_df = pd.DataFrame(columns = ['var_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    w_var_df['update_rule'], w_var_df['epoch'] = "", ""
    
    # dataframe to store the average change in MSE
    deltal_df = pd.DataFrame(columns = ['epoch', 'del-MSE'])    
    deltal, epochs = [], []

    xl, yl = next(data.get_data_batches(batchsize=6000, split=data.trainsplit))
    
    for ii in range(stored_epoch, stored_epoch + 5):
        print('calculating dynamics for epoch {}.'.format(ii))
        for batch_id in range(num_batches):
            x, y = xl[batch_id * batchsize:(batch_id + 1) * batchsize], yl[batch_id * batchsize:(batch_id + 1) * batchsize]
            randkey, _ = random.split(randkey)
            params_new, npgrad, _ = optim.npupdate(x, y, params, randkey, optimstate)
            _, sgdgrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)

            if(batch_id % interval == 0):
                test_acc.append(train.compute_metrics(params_new, forward, data, split_percent)[1])
                _, truegrad, _ = optim.sgdupdate(xl, yl, params_new, randkey, optimstate)

                epoch = round(ii + (batch_id + 1)/num_batches, 3)
                
                if(flags[0]):
                    grad_norms_df = grad_norms_df.append(grad_dynamics.grad_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch))
                
                if(flags[1]):
                    graddiff_norms_df = graddiff_norms_df.append(grad_dynamics.graddiff_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch))
                
                if(flags[2]):
                    sign_symmetry_df = sign_symmetry_df.append(grad_dynamics.sign_symmetry(npgrad, sgdgrad, truegrad, layer_sizes, epoch))
                
                if(flags[3]):
                    grad_angles_df = grad_angles_df.append(grad_dynamics.grad_angles(npgrad, sgdgrad, truegrad, layer_sizes, epoch))
                
                if(flags[4]):
                    w_norms_df = w_norms_df.append(grad_dynamics.w_norms(params_new, layer_sizes, epoch))
                
                if(flags[5]):
                    w_var_df = w_var_df.append(grad_dynamics.w_vars(params_new, layer_sizes, epoch))
                
                deltal.append(lossfunc(x,y, params_new) - lossfunc(x,y, params))
                epochs.append(epoch)

            params = params_new
            
else:
    print("no crash detected, exiting...")
    exit()

#need to use np.repeat as different dataframes have different number of rows depending on the value being calculated.
grad_norms_df['test_acc'], grad_norms_df['jobid'] = np.repeat(test_acc, 3), jobid
graddiff_norms_df['test_acc'], graddiff_norms_df['jobid'] = np.repeat(test_acc, 2), jobid
sign_symmetry_df['test_acc'], sign_symmetry_df['jobid'] = np.repeat(test_acc, 2), jobid
grad_angles_df['test_acc'], grad_angles_df['jobid'] = np.repeat(test_acc, 2), jobid
w_norms_df['test_acc'], w_norms_df['jobid'] = test_acc, jobid
w_var_df['test_acc'], w_var_df['jobid'] = test_acc, jobid
deltal_df['del-MSE'], deltal_df['epoch'] = deltal, epochs
deltal_df['test_acc'], deltal_df['jobid'] = test_acc, jobid

pd.set_option('display.max_columns', None)
print(grad_norms_df.head(5))
print(graddiff_norms_df.head(5))
print(sign_symmetry_df.head(5))
print(grad_angles_df.head(5))
print(w_norms_df.head(5))
print(w_var_df.head(5))

train_df['epoch'] = np.arange(start=1, stop=train_df['test_acc'].size+1, dtype=int)
train_df['network'], train_df['update_rule'], train_df['n_hl'], train_df['lr'], train_df['batchsize'], train_df['hl_size'], train_df['total_epochs'], train_df['jobid'] = network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid
# print(train_df.head(10))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'w_norms.csv')):
        use_header = True
    
    if(flags[0]):
        grad_norms_df.to_csv(path + 'grad_norms.csv', mode='a', header=use_header)
    
    if(flags[1]):
        graddiff_norms_df.to_csv(path + 'graddiff_norms.csv', mode='a', header=use_header)
    
    if(flags[2]):
        sign_symmetry_df.to_csv(path + 'sign_symmetry.csv', mode='a', header=use_header)
    
    if(flags[3]):
        grad_angles_df.to_csv(path + 'grad_angles.csv', mode='a', header=use_header)
    
    if(flags[4]):
        w_norms_df.to_csv(path + 'w_norms.csv', mode='a', header=use_header)

    if(flags[5]):
        w_var_df.to_csv(path + 'w_vars.csv', mode='a', header=use_header)
    
    deltal_df.to_csv(path + 'deltal.csv', mode='a', header=use_header)
    train_df.to_csv(path + 'train_df.csv', mode='a', header=use_header)
    