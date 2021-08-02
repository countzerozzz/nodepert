import npimports
import importlib

importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code for comparing the training trajectories of NP and SGD on various lr. We see that on smaller lr, NP traces the learning trajectory of SGD.
# the curves diverge on increasing the lr a little and further increase results in crashes in the network.
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
path = "explogs/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

# rows = [1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1]
rows = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
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
    if not os.path.exists(path + "vary_lr.csv"):
        df.to_csv(path + "vary_lr.csv", mode="a", header=True)
    else:
        df.to_csv(path + "vary_lr.csv", mode="a", header=False)
