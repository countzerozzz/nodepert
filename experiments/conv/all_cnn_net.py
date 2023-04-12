import npimports
import importlib

importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# A much larger, all convolutional network, with good performance with SGD. Implementation details as mentioned in the paper:
# https://arxiv.org/pdf/1412.6806.pdf
# We use the same network architecture, however, exclude the following while training: dropout, SGD + momentum, decaying LR
###

# tf.config.experimental.set_visible_devices([], "GPU")
config = {}
# parse arguments:
update_rule, lr, batchsize, num_epochs, log_expdata, wd, jobid = utils.parse_conv_args()
wd=float(wd)
network = "All-CNN-A"
(
    config["compute_norms"],
    config["batchsize"],
    config["num_epochs"],
    config["num_classes"],
) = (False, batchsize, num_epochs, data.num_classes)

# folder to log experiment results
path = "explogs/conv/wd/"

# num = 7 # number of learning rates
# rows = np.logspace(-6, -3, num, endpoint=True, dtype=np.float32)

if update_rule == "sgd":
    # rows = np.concatenate(([0.05, 0.07], np.arange(1e-1, 1, 2e-1)))
    rows = np.array([0.005, 0.01, 0.05, 0.1, 0.5])

elif update_rule == "np":
    rows = np.array([4e-5, 7e-5])
    # rows = np.linspace(0.01, 0.5, num)
else:
    raise ValueError("Unknown update rule")

ROW_DATA = "learning_rate"
row_id = jobid % len(rows)
lr = rows[row_id]

# len(convout_channels) has to be same as convlayer_sizes!
# convout_channels = [num_channels] * conv_depth
convout_channels = [96, 96, 192, 192, 192, 192, 10]

# format (kernel height, kernel width, input channels, output channels)
convlayer_sizes = [
    (5, 5, data.channels, convout_channels[0]),
    (3, 3, convout_channels[0], convout_channels[1]),  # stride = 2
    (5, 5, convout_channels[1], convout_channels[2]),
    (3, 3, convout_channels[2], convout_channels[3]),  # stride = 2
    (3, 3, convout_channels[3], convout_channels[4]),
    (1, 1, convout_channels[4], convout_channels[5]),
    (1, 1, convout_channels[5], convout_channels[6]),
]

down_factor = 4
fclayer_sizes = [
    int(
        (data.height / down_factor)
        * (data.width / down_factor)
        * convlayer_sizes[-1][-1]
    ),
    data.num_classes,
]

randkey = random.PRNGKey(jobid)
convparams = conv.init_convlayers(convlayer_sizes, randkey)
randkey, _ = random.split(randkey)
fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], randkey)

params = convparams
params.append(fcparams)

print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?
print("conv architecture {}, fc layer {}".format(convlayer_sizes, fclayer_sizes))
print("model params: ", utils.get_params_count(params))

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
    params, forward, data, config, optimizer, optimstate, randkey, verbose=False
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
    df["wd"],
    df["jobid"],
) = (network, update_rule, lr, batchsize, num_epochs, wd, jobid)
print(df.head(5))

# save the results of our experiment
if log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "all-cnn-a.csv"):
        use_header = True

    df.to_csv(path + "all-cnn-a.csv", mode="a", header=use_header)