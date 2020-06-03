import npimports
import importlib
importlib.reload(npimports)
from npimports import *

import data_loaders.mnistloader as data

#parse arguments
config = {}

network, update_rule, n_hl, lr, config['batchsize'], hl_size, config['num_epochs'], log_expdata, jobid = utils.parse_args()
path = 'explogs/adamupdate/sgd/'
seed=np.random.randint(0,1000)
randkey = random.PRNGKey(seed)

rows = np.logspace(start=-5, stop=0, num=100, endpoint=True, base=10, dtype=np.float32)
print(rows)
ROW_DATA = 'learning_rate'
# cols = [32] 
# COL_DATA = 'convchannels'

row_id = jobid % len(rows)
# col_id = (jobid//len(rows)) % len(cols)

lr = rows[row_id]
# convchannels = cols[col_id]

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

optimstate = { 'lr' : lr, 't' : 0 }
test_acc = []

for epoch in range(1, config['num_epochs']+1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data)[1])
    
    print('EPOCH ', epoch)

    for x, y in data.get_data_batches(batchsize=config['batchsize'], split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)

    epoch_time = time.time() - start_time
    print('epoch training time: {}\n test acc: {}\n'.format(round(epoch_time,2), round(test_acc[-1], 3)))

final_acc = np.mean(test_acc[-5:])

def file_writer(path):
    with open(path+'vanilla.csv', 'a') as csvFile:
        writer = csv.writer(csvFile, lineterminator=' , ')
        writer.writerow([str(final_acc)])
        csvFile.flush()

    csvFile.close()
    return

if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    file_writer(path)
