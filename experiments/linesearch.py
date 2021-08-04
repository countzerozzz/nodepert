import npimports
import importlib

importlib.reload(npimports)
from npimports import *

from linesearch_utils import linesearchfunc

### FUNCTIONALITY ###
# the aim is to see how the critical learning rate (max lr above which the network doesn't learn) and optimal lr change as training progresses.
# we would like to know what this curve looks like as the training progresses. The next thing we want to compare are the curves when
# we are looking at a single update vs t-step update. Do they follow a similar pattern?
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

step_type = "single-step"  # 't-step'

rows = [100, 500, 1000, 5000, 10000]
ROW_DATA = "network width"
row_id = jobid % len(rows)
hl_size = rows[row_id]
n_hl = 1
# linesearch parameters: pick 'num' number of different lr values at regular log intervals in belween 10e(start) and 10e(end).
start, stop, num = -4, 1, 100
network_acc = 95

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
test_acc = []
df = pd.DataFrame()

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    test_acc.append(train.compute_metrics(params, forward, data)[1])
    print("EPOCH {}\ntest acc: {}%".format(epoch, round(test_acc[-1], 3)))

    for x, y in data.get_data_batches(batchsize=batchsize, split=data.trainsplit):
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)

    # condition for performing linesearch: every x epochs or at some particular test accuracy
    if test_acc[-1] > network_acc:
        # if(epoch % 5 == 0):
        meta_data = (forward, step_type, n_hl, update_rule, hl_size, network_acc)
        df = df.append(linesearchfunc(randkey, params, start, stop, num, meta_data))
        break

    epoch_time = time.time() - start_time
    print("epoch training time: {}s\n".format(round(epoch_time, 2)))

pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment
if log_expdata:
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "linesearch.csv"):
        df.to_csv(path + "linesearch.csv", mode="a", header=True)
    else:
        df.to_csv(path + "linesearch.csv", mode="a", header=False)
