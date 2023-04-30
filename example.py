import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

run_file = "main.py"

print("training fc network on mnist with NP")
code = f"import sys; sys.argv = ['{run_file}', '-update_rule', 'np', '-log_expdata', 'True', '-lr', '0.01', '-num_epochs', '10']; exec(open('{run_file}').read())"
exec(code)

print("training fc network on mnist with SGD")
code = f"import sys; sys.argv = ['{run_file}', '-update_rule', 'sgd', '-log_expdata', 'True', '-lr', '0.01', '-num_epochs', '10']; exec(open('{run_file}').read())"
exec(code)

print("generating plots...")
# now load the logs as a dataframe from the csv files, and plot the results
df = pd.read_csv("explogs/fc-expdata.csv")

sns.set(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the data with seaborn
sns.lineplot(data=df, x="epoch", y="test_acc", hue="update_rule", linewidth=2, ax=ax)

# Customize the plot appearance
ax.set_title("training on mnist", fontsize=20)
ax.set_xlabel("epochs", fontsize=16)
ax.set_ylabel("test accuracy", fontsize=16)
ax.legend(title="Legend", title_fontsize=14, fontsize=12)

plt.savefig("sample_training.png", dpi=300, bbox_inches="tight")
print("saved sample_training.png!")