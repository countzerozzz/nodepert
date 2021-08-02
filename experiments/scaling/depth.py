import npimports
import importlib

importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code is for finding the scalability of node perturbation with depth (constant width) for fully connected non-linear networks
###

config = {}
# parse arguments
(
    network,
    update_rule,
    n_hl,
    lr,
    batchsize,
    hl_size,
    num_epochs,
    log_expdata,
    jobid,
) = utils.parse_args()
config["compute_norms"], config["batchsize"], config["num_epochs"] = (
    False,
    batchsize,
    num_epochs,
)

# folder to log experiment results
path = "explogs/scaling/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.
num = 50  # number of learning rates

rows = np.logspace(-4, 0, num, endpoint=True, dtype=np.float32)
ROW_DATA = "learning_rate"
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
forward = fc.batchforward
if update_rule == "np":
    gradfunc = optim.npupdate
elif update_rule == "sgd":
    gradfunc = optim.sgdupdate

params = fc.init(layer_sizes, randkey)

optimstate = {"lr": lr, "t": 0}

# now train
params, optimstate, expdata = train.train(
    params, forward, data, config, gradfunc, optimstate, randkey, verbose=False
)

df = pd.DataFrame.from_dict(expdata)

pd.set_option("display.max_columns", None)
(
    df["network"],
    df["update_rule"],
    df["n_hl"],
    df["lr"],
    df["batchsize"],
    df["hl_size"],
    df["total_epochs"],
    df["jobid"],
) = (network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid)
print(df.head(5))

# save the results of our experiment
if log_expdata:
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "depth.csv"):
        df.to_csv(path + "depth.csv", mode="a", header=True)
    else:
        df.to_csv(path + "depth.csv", mode="a", header=False)
