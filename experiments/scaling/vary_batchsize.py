import npimports
import importlib

importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# this code is for exploring the effect of minibatch size on the node perturbation algorithm. With larger batches, the variance of the NP gradient
# estimate decreases.
###

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

# folder to log experiment results
path = "explogs/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different configs.

num = 25  # number of learning rates

rows = np.logspace(-5, -1, num, endpoint=True, base=10, dtype=np.float32)
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
test_acc, batch_id = [], []
# counter for the batches and interval to measure training metrics
ctr = 0
interval = 100

for epoch in range(1, num_epochs + 1):
    start_time = time.time()

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        if ctr % interval == 0:
            batch_id.append(ctr)
            test_acc.append(train.compute_metrics(params, forward, data)[1])

        randkey, _ = random.split(randkey)
        # get the gradients, throw away the traditional weight updates
        params, grads, _ = gradfunc(x, y, params, randkey, optimstate)
        ctr += 1

    epoch_time = time.time() - start_time
    print(
        "epoch training time: {}\n test acc: {}\n".format(
            round(epoch_time, 2), round(test_acc[-1], 3)
        )
    )

df = pd.DataFrame()
pd.set_option("display.max_columns", None)
df["batch_id"], df["test_acc"] = batch_id, test_acc
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
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "vary_batchsize.csv"):
        use_header = True

    df.to_csv(path + "vary_batchsize.csv", mode="a", header=use_header)
