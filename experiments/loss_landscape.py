import npimports
import importlib

from utils import npvec_to_params, params_to_npvec

importlib.reload(npimports)
from npimports import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib as mpl


def calculate_loss(params):
    loss = 0
    for x, y in data.get_rawdata_batches(batchsize=100, split="train[:20%]"):
        x, y = data.prepare_data(x, y)
        h, a = forward(x, params)
        loss += losses.batchmseloss(h[-1], y).sum()
    return loss / 100


def normalize_params(params, p_origin):
    normalized_params = []
    for (w, b), (w_origin, b_origin) in zip(params, p_origin):
        w = w * (np.linalg.norm(w_origin) / np.linalg.norm(w))
        b = b * (np.linalg.norm(b_origin) / np.linalg.norm(b))
        normalized_params.append((w, b))
    return normalized_params


# randkey = random.PRNGKey(int(time.time()))
randkey = random.PRNGKey(1)

log_expdata = False
path = "explogs/fc/"

# parse FC network arguments
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

# define training configs
config = {}
config["num_epochs"] = num_epochs
config["batchsize"] = batchsize
config["compute_norms"] = False
config["save_trajectory"] = True

# build our network
layer_sizes = [data.num_pixels, 50, 50, data.num_classes]
randkey, _ = random.split(randkey)
params = fc.init(layer_sizes, randkey)
print("Network structure: {}".format(layer_sizes))
print(xla_bridge.get_backend().platform)  # are we running on CPU or GPU?

# get forward pass, optimizer, and optimizer state + params

forward = optim.forward = fc.batchforward
optim.forward = fc.batchforward
optim.noisyforward = fc.batchnoisyforward

if update_rule == "np":
    optimizer = optim.npupdate
elif update_rule == "sgd":
    optimizer = optim.sgdupdate

optimstate = {"lr": lr}

start = time.time()
# now train
params, optimstate, expdata = train.train(
    params, forward, data, config, optimizer, optimstate, randkey, verbose=False,
)
mins, secs = divmod(time.time() - start, 60)
print("Finished training in {}m {}s".format(round(mins), round(secs)))

trajectory = expdata.pop("trajectory", None)
# need to get the weights as colums
trajectory = np.swapaxes(trajectory, 0, 1)
p_origin = npvec_to_params(trajectory[:, -1], layer_sizes)

start = time.time()
pca = PCA(n_components=2, whiten=True)
components = pca.fit_transform(trajectory)
print("calculated PCA in {}s".format(round(time.time() - start, 2)))
start = time.time()
p_0 = normalize_params(npvec_to_params(components[:, 0], layer_sizes), p_origin)
p_1 = normalize_params(npvec_to_params(components[:, 1], layer_sizes), p_origin)

fig_range = 0.5 
points = 30
a_grid = np.linspace(-1, 1, num=points) ** 3 * fig_range
b_grid = np.linspace(-1, 1, num=points) ** 3 * fig_range
loss_grid = np.zeros((points, points))

for i, a in enumerate(a_grid):
    for j, b in enumerate(b_grid):
        params = []
        for (wc, bc), (w_0, b_0), (w_1, b_1) in zip(p_origin, p_0, p_1):
            weights = wc + a * w_0 + b * w_1
            bias = bc + a * b_0 + b * b_1
            params.append((weights, bias))
        loss_grid[i][j] = calculate_loss(params)

mins, secs = divmod(time.time() - start, 60)
print("calculated loss grid {}m {}s".format(round(mins), round(secs)))
# the pseudoinverse
components_i = np.linalg.pinv(
    np.column_stack((params_to_npvec(p_0), params_to_npvec(p_1)))
)

# center the weights on the training path and project onto components
coord_path = np.array(
    [components_i @ (weights - trajectory[:, -1]) for weights in trajectory.T]
)
print("calculated path coordinates")


def plot(levels=15, ax=None, **kwargs):
    xs = a_grid
    ys = b_grid
    zs = loss_grid
    if ax is None:
        _, ax = plt.subplots(**kwargs)
        ax.set_title("The Loss Surface")
        ax.set_aspect("equal")
    # Set Levels
    min_loss = zs.min()
    max_loss = zs.max()
    levels = np.exp(
        np.linspace(np.math.log(min_loss), np.math.log(max_loss), num=levels)
    )
    # Create Contour Plot
    CS = ax.contour(
        xs,
        ys,
        zs,
        levels=levels,
        cmap="magma",
        linewidths=0.75,
        norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
    )
    ax.clabel(
        CS, inline=True, inline_spacing=10, fontsize=8, fmt="%1.1f", rightside_up=False
    )
    return ax


def plot_training_path(path, ax=None, end=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0], path[:, 1], s=4, c=colors, cmap="cividis", norm=norm,
    )
    return ax


print("plotting...")
ax = plot(dpi=200)
ax = plot_training_path(coord_path, ax)
ax.figure.savefig("experiments/loss-landscape-{}.png".format(update_rule))

df = pd.DataFrame.from_dict(expdata)
df["dataset"] = npimports.dataset
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

pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment
if log_expdata:
    use_header = False
    Path(path).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path + "fc-test.csv"):
        use_header = True

    df.to_csv(path + "fc-test.csv", mode="a", header=use_header)
    print("wrote training logs to disk")
