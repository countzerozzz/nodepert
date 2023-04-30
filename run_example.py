import pandas as pd


run_file = "main.py"

code = f"import sys; sys.argv = ['{run_file}', '-update_rule', 'np', '-log_expdata', 'True']; exec(open('{run_file}').read())"
exec(code)

code = f"import sys; sys.argv = ['{run_file}', '-update_rule', 'sgd', '-log_expdata', 'True']; exec(open('{run_file}').read())"
exec(code)

# now load the logs as a dataframe from the csv files, and plot the results
df = pd.read_csv("explogsa/fc-expdata.csv")

import matplotlib.pyplot as plt
# TODO: fix this
plt.plot(df.epoch, df.test_acc, label="test")

