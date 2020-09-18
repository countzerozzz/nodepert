import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# A much larger, all convolutional network, with good performance with SGD. Implementation details as mentioned in the paper: 
# https://arxiv.org/pdf/1412.6806.pdf
# difference: dropout, SGD + momentum, decaying LR
###

config = {}
# parse arguments:
update_rule, conv_depth, num_channels, lr, batchsize, num_epochs, log_expdata, jobid = utils.parse_conv_args()
network = 'conv'
config['compute_norms'], config['batchsize'], config['num_epochs'], config['num_classes'] = False, batchsize, num_epochs, data.num_classes

# folder to log experiment results
path = "explogs/"

# num = 7 # number of learning rates
# rows = np.logspace(-5, -2, num, endpoint=True, dtype=np.float32)

# ROW_DATA = 'learning_rate'
# row_id = jobid % len(rows)
# lr = rows[row_id]

#len(convout_channels) has to be same as convlayer_sizes!
# convout_channels = [num_channels] * conv_depth
convout_channels = [96, 96, 96, 192, 192, 10]

#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, data.channels, convout_channels[0]),
                   (3, 3, convout_channels[0], convout_channels[1]),
                   (3, 3, convout_channels[1], convout_channels[2]),
                   (3, 3, convout_channels[2], convout_channels[3]),
                   (3, 3, convout_channels[3], convout_channels[4]),
                   (1, 1, convout_channels[4], convout_channels[5])]

down_factor = 2 * ((len(convout_channels) + 1) // 3)
down_factor = 1
fclayer_sizes = [int((data.height / down_factor) * (data.width / down_factor) * convlayer_sizes[-1][-1]), data.num_classes]

randkey = random.PRNGKey(jobid)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

print("conv architecture {}, fc layer {}".format(convlayer_sizes, fclayer_sizes))

# get forward pass, optimizer, and optimizer state + params
forward = conv.batchforward
optim.forward = conv.batchforward
optim.noisyforward = conv.batchnoisyforward

if(update_rule == 'np'):
    optimizer = optim.npupdate
elif(update_rule == 'sgd'):
    optimizer = optim.sgdupdate

optimstate = { 'lr' : lr, 't' : 0 }

# now train
params, optimstate, expdata = train.train(  params,
                                            forward,
                                            data,
                                            config,
                                            optimizer,
                                            optimstate,
                                            randkey,
                                            verbose = True)

df = pd.DataFrame.from_dict(expdata)
df['dataset'] = npimports.dataset
pd.set_option('display.max_columns', None)
df['network'], df['update_rule'], df['conv_depth'], df['num_channels'], df['lr'], df['batchsize'], df['total_epochs'], df['jobid'] = network, update_rule, conv_depth, num_channels, lr, batchsize, num_epochs, jobid
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'conv.csv')):
        df.to_csv(path + 'conv.csv', mode='a', header=True)
    else:
        df.to_csv(path + 'conv.csv', mode='a', header=False)
