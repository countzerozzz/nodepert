import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# set the 'seed' for our experiment
# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

log_expdata = False
path = 'explogs/'

# parse conv network arguments
update_rule, lr, batchsize, num_epochs, log_expdata, jobid = utils.parse_conv_args()

# define training configs
config = {}
config['num_epochs'] = num_epochs = 10
config['batchsize'] = batchsize = 100
config['num_classes'] = data.num_classes
config['compute_norms'] = False

#length of convout_channels has to be same as convlayer_sizes!
convout_channels = [32, 32, 32]

#format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [(3, 3, data.channels, convout_channels[0]),
                   (3, 3, convout_channels[0], convout_channels[1]),
                   (3, 3, convout_channels[1], convout_channels[2])]

fclayer_sizes = [data.height * data.width * convlayer_sizes[-1][-1], data.num_classes]

randkey, _ = random.split(randkey)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

# get forward pass, optimizer, and optimizer state + params
forward = conv.batchforward
optim.forward = conv.batchforward
optim.noisyforward = conv.batchnoisyforward

if(update_rule == 'np'):
    optimizer = optim.npupdate
elif(update_rule == 'sgd'):
    optimizer = optim.sgdupdate

optimstate = { 'lr' : lr}

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
df['network'], df['update_rule'], df['lr'], df['batchsize'], df['total_epochs'], df['num_conv_layers'], df['jobid'] = network, update_rule, lr, batchsize, num_epochs, num_conv_layers, jobid
print(df.head(5))

# save the results of our experiment
if(log_expdata):
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if(not os.path.exists(path + 'conv-test.csv')):
        use_header = True
    
    df.to_csv(path + 'conv-test.csv', mode='a', header=use_header)
