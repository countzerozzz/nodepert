import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    run_file = "nodepert/main.py"
    config_str = "'-update_rule', 'np', '-lr', '0.01', '-num_epochs', '10', '-log_expdata', 'True'"

    print("training fc network on mnist with NP")
    code = f"import sys; sys.argv = ['{run_file}'," + config_str + f"]; exec(open('{run_file}').read())"
    exec(code)

    config_str = "'-update_rule', 'sgd', '-lr', '0.01', '-num_epochs', '10'"

    print("training fc network on mnist with SGD")
    code = f"import sys; sys.argv = ['{run_file}'," + config_str + f"]; exec(open('{run_file}').read())"
    exec(code)

    print("generating plots...")
    # load the experiment logs into a dataframe
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


if __name__ == "__main__":
    main()