import npimports
import importlib
importlib.reload(npimports)
from npimports import *

args = utils.parse_args()

network = args.network
update_rule = args.update_rule
randkey = random.PRNGKey(args.jobid)

path = "explogs/" + network + "/" 
# define training configs
train_config = {'num_epochs': args.num_epochs, 'batchsize': args.batchsize, 'compute_norms': False, 'save_trajectory': False}

print(f"Dataset: {npimports.dataset}")

if network == "fc":
    # build our network
    n_hl = args.n_hl
    hl_size = args.hl_size
    layer_sizes = [data.num_pixels] + [hl_size] * n_hl + [data.num_classes]

    randkey, _ = random.split(randkey)
    params = fc.init(layer_sizes, randkey)
    print(f"fully connected network structure: {layer_sizes}\n")

    # get forward pass, optimizer, and optimizer state + params
    forward = fc.batchforward
    optim.forward = fc.batchforward
    optim.noisyforward = fc.batchnoisyforward

elif network == "conv":
    # build our network
    convout_channels = [32, 32, 32]
    # format (kernel height, kernel width, input channels, output channels)
    convlayer_sizes = [(3, 3, prev_ch, curr_ch) for prev_ch, curr_ch in zip([data.channels] + convout_channels[:-1], convout_channels)]
    # because of striding, we need to downsample the final fc layer
    down_factor = 2
    fclayer_sizes = [
        int(
            (data.height / down_factor)
            * (data.width / down_factor)
            * convlayer_sizes[-1][-1]
        ),
        data.num_classes,
    ]

    key, key2 = random.split(randkey)
    convparams = conv.init_convlayers(convlayer_sizes, key)
    fcparams = fc.init_layer(fclayer_sizes[0], fclayer_sizes[1], key2)
    params = convparams + [fcparams]
    print(f"conv architecture {convlayer_sizes}, FC layer {fclayer_sizes}\n")

    # get forward pass, optimizer, and optimizer state + params
    forward = conv.batchforward
    optim.forward = conv.batchforward
    optim.noisyforward = conv.batchnoisyforward

# Display backend
print(f"running on: {xla_bridge.get_backend().platform}")

if update_rule == "np":
    optimizer = optim.npupdate
elif update_rule == "sgd":
    optimizer = optim.sgdupdate

optimstate = {"lr": args.lr, "wd": args.wd}

# now train
params, optimstate, expdata = trainer.train(
    params, forward, data, train_config, optimizer, optimstate, randkey, verbose=False
)

df = pd.DataFrame.from_dict(expdata)

# store meta data about the experiment
df["dataset"] = npimports.dataset
for arg in vars(args):
    df[f"{arg}"] = getattr(args, arg)

pd.set_option("display.max_columns", None)
print(df.head(5))

# save the results of our experiment
if args.log_expdata:
    logdata_path = Path(path)
    logdata_path.mkdir(parents=True, exist_ok=True)

    csv_file = logdata_path / "fc-test.csv"
    write_header = not csv_file.exists()

    df.to_csv(csv_file, mode="a", header=write_header)