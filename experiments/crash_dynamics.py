import npimports
import importlib

importlib.reload(npimports)
from npimports import *

import grad_dynamics
from linesearch_utils import lossfunc

### FUNCTIONALITY ###
# this code measures the grad_norm, (noise of gradient)_norm, sign symmetry and angle between 'true' gradient and gradient estimates when the network crashes.
# there is a crash if the current accuracy is less than 'x%' from the max accuracy. When a crash is detected, the network restarts from the last checkpoint
# and then performs the above mentioned calculations wrt the gradients.
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

# set flags for which of the metrics to compute during the crash.
# gradient norms, difference in gradient norms, sign symmetry, gradient angles, weight norms, variance of weight, layer activations of the network
# flags = [1, 1, 1, 1, 1, 1, 1]
flags = [0, 1, 0, 0, 0, 0, 1]
flag_funcs = {
    0: grad_dynamics.grad_norms,
    1: grad_dynamics.graddiff_norms,
    2: grad_dynamics.sign_symmetry,
    3: grad_dynamics.grad_angles,
    4: grad_dynamics.w_norms,
    5: grad_dynamics.w_vars,
    6: grad_dynamics.compute_layer_activity,
}
rows_populated = [0] * len(flags)

# folder to log experiment results
path = "explogs/crash_dynamics/"

randkey = random.PRNGKey(jobid)

# a list for running parallel jobs in slurm. Each job will correspond to a particular value in 'rows'. If running on a single machine,
# the config used will be the first value of 'rows' list. Here 'rows' will hold the values for different learning rates.

ROW_DATA = "learning_rate"
# rows = np.logspace(start=-3, stop=-1, num=25, endpoint=True, base=10, dtype=np.float32)
# rows = np.linspace(0.005, 0.025, num=20)
rows = [0.01, 0.015, 0.02, 0.025]

row_id = jobid % len(rows)
lr = rows[row_id]
print("learning rate", lr)

# if split percent is changed, the num_batches will also have to be adjusted accordingly.
split_percent = "[:10%]"
data_points = 600
num_batches = int(data_points / batchsize)

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

# define metrics for measuring a crash
high = -1
crash = False
stored_epoch = -1
interval = 1  # interval of batches to computer grad dynamics over after a crash.

# log parameters at intervals of 5 epochs and when the crash happens, reset training from this checkpoint.
for epoch in range(0, num_epochs):
    # crash=True
    # break
    start_time = time.time()
    test_acc.append(
        train.compute_metrics(params, forward, data, split_percent=split_percent)[1]
    )
    print("EPOCH {}\ntest acc: {}%".format(epoch, round(test_acc[-1], 3)))

    # if the current accuracy is lesser than the max by 'x' amount, a crash is detected. break training and reset params
    high = max(test_acc)
    if high - test_acc[-1] > 25:
        print("crash detected! Resetting params")
        crash = True
        break

    # every 5 epochs checkpoint the network params
    if epoch % 5 == 0:
        Path(path + "model_params/").mkdir(parents=True, exist_ok=True)
        pickle.dump(params, open(path + "model_params/" + str(jobid) + ".pkl", "wb"))
        stored_epoch = epoch

    for x, y in data.get_rawdata_batches(
        batchsize=batchsize, split="train" + split_percent
    ):
        x, y = data.prepare_data(x, y)
        randkey, _ = random.split(randkey)
        params, grads, optimstate = gradfunc(x, y, params, randkey, optimstate)

    epoch_time = time.time() - start_time
    print("epoch training time: {}s\n".format(round(epoch_time, 2)))

train_df = pd.DataFrame()
train_df["test_acc"] = test_acc

if crash:
    params = pickle.load(open(path + "model_params/" + str(jobid) + ".pkl", "rb"))
    test_acc = []

    df_names = [
        "grad_norms", # dataframe to store the norm of the gradients: (np, sgd and true gradient)
        "graddiff_norms",  # dataframe to store the norm of noise in the np and sgd gradients, e.g. norm(gradtrue - gradnp)
        "sign_symmetry",  # dataframe to store the sign sign_symmetry in the np and sgd gradient, e.g. ss(gradtrue, gradnp)
        "grad_angles",  # dataframe to store the angles of np and sgd gradient with the true gradient, e.g. angle(gradtrue, gradnp)
        "w_norms", # dataframe to store the norm of the neural network weights
        "w_vars", # dataframe to store the variance of the neural network weights
        "n_activity", # dataframe to store the activity of neurons at every layer
        "deltal", # dataframe to store the average change in MSE
    ]
    column_prefixes = [
        "gnorm_w",
        "gdiff_norm_w",
        "ss_w",
        "gangle_w",
        "norm_w",
        "var_w",
        "activity_l",
    ]
    dataframes = {
        name: pd.DataFrame(
            columns=[prefix + str(i) for i in np.arange(1, len(layer_sizes))]
        )
        for name, prefix in zip(df_names[:-1], column_prefixes)
    }
    dataframes["deltal"] = pd.DataFrame(columns=["epoch", "del-MSE"])
    deltal, epochs = [], []

    for df in dataframes.values():
        df["update_rule"], df["epoch"], df["test_acc"], df["jobid"] = "", "", "", ""

    xl, yl = data.prepare_data(*next(iter(data.get_rawdata_batches(batchsize=data_points, split=data.trainsplit))))

    for ii in range(stored_epoch, stored_epoch + 5):
        print("calculating dynamics for epoch {}.".format(ii))
        for batch_id in range(num_batches):
            x, y = xl[batch_id * batchsize: (batch_id + 1) * batchsize], yl[batch_id * batchsize: (batch_id + 1) * batchsize]
            print(x.shape)
            randkey, _ = random.split(randkey)
            params_new, npgrad, _ = optim.npupdate(x, y, params, randkey, optimstate)
            _, sgdgrad, _ = optim.sgdupdate(x, y, params, randkey, optimstate)

            if batch_id % interval == 0:
                test_acc.append(train.compute_metrics(params_new, forward, data, split_percent)[1])
                _, truegrad, _ = optim.sgdupdate(xl, yl, params_new, randkey, optimstate)

                epoch = round(ii + (batch_id + 1) / num_batches, 3)

                for idx, func in flag_funcs.items():
                    if flags[idx]:
                        result, rows_populated[idx] = func(npgrad, sgdgrad, truegrad, layer_sizes, epoch)
                        dataframes[df_names[idx]] = pd.concat([dataframes[df_names[idx]], result], ignore_index=True)

                deltal.append(lossfunc(x, y, params_new) - lossfunc(x, y, params))
                epochs.append(epoch)

            params = params_new

else:
    print("no crash detected, exiting...")
    exit()

# need to use np.repeat as different dataframes have different number of rows depending on the value being calculated.

for idx, _ in flag_funcs.items():
    if flags[idx]:
        dataframes[df_names[idx]]["test_acc"] = np.repeat(test_acc, rows_populated[idx])
        dataframes[df_names[idx]]["jobid"] = jobid

pd.set_option("display.max_columns", None)

for df_name in df_names + ["deltal"]:
    print(f"{df_name}:\n{dataframes[df_name].head(5)}\n")

train_df["epoch"] = np.arange(start=1, stop=train_df["test_acc"].size + 1, dtype=int)
(
    train_df["network"],
    train_df["update_rule"],
    train_df["n_hl"],
    train_df["lr"],
    train_df["batchsize"],
    train_df["hl_size"],
    train_df["total_epochs"],
    train_df["jobid"],
) = (network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid)
# print(train_df.head(10))

# save the results of our experiment
if log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)

    for name, df in dataframes.items():
        if not os.path.exists(path + name + ".csv"):
            use_header = True
        df.to_csv(path + name + ".csv", mode="a", header=use_header)

train_df["epoch"] = np.arange(start=1, stop=train_df["test_acc"].size + 1, dtype=int)
(train_df["network"], train_df["update_rule"], train_df["n_hl"],
 train_df["lr"], train_df["batchsize"], train_df["hl_size"],
 train_df["total_epochs"], train_df["jobid"]) = (network, update_rule, n_hl, lr, batchsize, hl_size, num_epochs, jobid)

if log_expdata:
    train_df.to_csv(path + "train_df.csv", mode="a", header=use_header)