import subprocess
import itertools

exec_file = "main_exps.py"
# gpu_queue, cores = "gpu_rtx", 5
gpu_queue, cores = "gpu_tesla", 12
seeds = 3

configs = {
    "exp_name": ["weight_decay"],
    "update_rule": ["np", "sgd"],
    "num_epochs": [1000],
    "batchsize": [100],
    "dataset": ["cifar10"],
    "wd": [0, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
    "lr": [5e-5, 1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    "network": ["conv", "conv-large"],
    # "network": ["fc"],
    # "n_hl": [2],
    # "hl_size": [500],
    "log_expdata": [True]
}
# function to iterate through all values of dictionary:
combinations = list(itertools.product(*configs.values()))

# generate config string to pass to bash script
for combination in combinations:
    execstr = "python experiments/" + f"{exec_file}"
    for idx, key in enumerate(configs.keys()):
        execstr += " -" + key + " " + str(combination[idx])
    # print(execstr)
    cmd = ["experiments/cluster_scripts/submit_job.sh", str(cores), str(seeds), gpu_queue, execstr]

    # Run the command and capture the output
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    print(output)