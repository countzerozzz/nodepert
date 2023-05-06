import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    config = {
        "update_rule": "np",
        "log_expdata": True,
        "lr": 0.01,
        "num_epochs": 10,
    }

    run_file = "nodepert/main.py"
    print("training fc network on mnist with NP")
    code = f"import sys; sys.argv = ['{run_file}'," + convert_config_to_string(config) + f"]; exec(open('{run_file}').read())"
    exec(code)

    config["update_rule"] = "sgd"

    print("training fc network on mnist with SGD")
    code = f"import sys; sys.argv = ['{run_file}'," + convert_config_to_string(config) + f"]; exec(open('{run_file}').read())"
    exec(code)

    print("generating plots...")
    generate_plot("explogs/fc-expdata.csv")


# loads the experiment logs into a dataframe from the csv files, and plots the results
def generate_plot(datapath):
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

    plt.savefig("example_training.png", dpi=300, bbox_inches="tight")
    print("saved example_training.png!")


def convert_config_to_string(config):
    config_str = ""
    for k, v in config.items():
        config_str += f"'-{k}', '{v}', "
    return config_str[:-2]


if __name__ == "__main__":
    main()