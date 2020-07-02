import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import grad_dynamics

#parse arguments
network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, log_expdata, jobid = utils.parse_args()

path = "explogs/crash_dynamics/"
file_path = os.path.join(path, 'explogs.csv')

randkey = random.PRNGKey(jobid)

rows = np.logspace(start=-3, stop=-1, num=10, endpoint=True, base=10, dtype=np.float32)
# rows = [0.03]
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
params_new = params

optimstate = { 'lr' : lr, 't' : 0}

test_acc = []
num_batches = int(len(data.train_images) / batchsize)
high = -1
crash = False
crash_epoch = -1

#log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(1, num_epochs+1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params_new, forward, data)[1])
    
    high = max(test_acc)
    if(high - test_acc[-1] > 10):
        print('crash detected! Resetting params')
        crash = True
        crash_epoch = epoch
        break
    
    print('EPOCH ', epoch)

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params_new, grads, optimstate = gradfunc(x, y, params_new, randkey, optimstate)

    if(epoch % 5 == 0):
        Path(path).mkdir(exist_ok=True)
        pickle.dump(params, open(path + "model_params.pkl", "wb"))    
    
    params = params_new
    epoch_time = time.time() - start_time
    print('epoch training time: {}\n test acc: {}\n'.format(round(epoch_time,2), round(test_acc[-1], 3)))

params = pickle.load(open(path + "model_params.pkl", "rb"))

if(crash):
    sign_symmetry_df = pd.DataFrame(columns = ['ss_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    sign_symmetry_df['update_rule'] = ""
    # grad_norms_df = pd.DataFrame(columns = ['gnorm_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    # grad_norms_df['update_rule'] = ""
    # grad_angles_df = pd.DataFrame(columns = ['gangle_w'+ str(i) for i in np.arange(1,len(layer_sizes))])
    # grad_angles_df['update_rule'] = ""
    
    for epoch in range(crash_epoch, crash_epoch + 5):
        print('calculating dynamics for epoch {}.'.format(epoch))

        for ii in range(num_batches):
            for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
                randkey, _ = random.split(randkey)
                params_new, npgrad, _ = optim.npupdate(x, y, params, randkey, optimstate)
                _, sgdgrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)
                break

            for x, y in data.get_data_batches(batchsize=5000, split=data.trainsplit):
                _, truegrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)
                break
            
            sign_symmetry_df.append(grad_dynamics.sign_symmetry(npgrad, sgdgrad, truegrad))
            # grad_norms_df.append(grad_dynamics.grad_norms(npgrad, sgdgrad, truegrad))
            # grad_angles_df.append(grad_dynamics.grad_angles(npgrad, sgdgrad, truegrad))
        
        params = params_new
            
else:
    print("no crash detected, exiting...")
    exit()

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
