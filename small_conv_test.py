import npimports
import importlib

importlib.reload(npimports)
from npimports import *
# import silence_tensorflow.auto

# set the 'seed' for our experiment
# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(0)

path = "explogs/"

# parse conv network arguments
update_rule, lr, batchsize, num_epochs, log_expdata, wd, jobid = utils.parse_conv_args()
network = "conv"
# define training configs
config = {'num_epochs': num_epochs, 'batchsize': batchsize, 'compute_norms': False, 'save_trajectory':False, 'num_classes': data.num_classes}

# length of convout_channels has to be same as convlayer_sizes!
convout_channels = [32, 32, 32]

# format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [
    (3, 3, data.channels, convout_channels[0]),
    (3, 3, convout_channels[0], convout_channels[1]),
    (3, 3, convout_channels[1], convout_channels[2]),
]

num_conv_layers = len(convlayer_sizes)
down_factor = 2
fclayer_sizes = [
    int(
        (data.height / down_factor)
        * (data.width / down_factor)
        * convlayer_sizes[-1][-1]
    ),
    data.num_classes,
]

print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?
print("conv architecture {}, fc layer {}".format(convlayer_sizes, fclayer_sizes))

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

if update_rule == "np":
    optimizer = optim.npwdupdate
elif update_rule == "sgd":
    optimizer = optim.sgdwdupdate
        
optimstate = {"lr": lr, "wd": wd}

# now train
params, optimstate, expdata = train.train(
    params, forward, data, config, optimizer, optimstate, randkey, verbose=True
)

df = pd.DataFrame.from_dict(expdata)
df["dataset"] = npimports.dataset
pd.set_option("display.max_columns", None)
(
    df["network"],
    df["update_rule"],
    df["lr"],
    df["batchsize"],
    df["total_epochs"],
    df["num_conv_layers"],
    df["jobid"],
) = (network, update_rule, lr, batchsize, num_epochs, num_conv_layers, jobid)
print(df.head(5))

# save the results of our experiment
if log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "conv-test.csv"):
        use_header = True

    df.to_csv(path + "conv-test.csv", mode="a", header=use_header)
