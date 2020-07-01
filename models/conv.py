import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# define element-wise relu:
def relu(x):
  return jnp.maximum(0, x)

#helper function to init conv layer:
def init_single_convlayer(kernel_height, kernel_width, input_channels, output_channels, w_key):
  # use he style scaling (not glorot) https://arxiv.org/pdf/1502.01852.pdf:
  std = np.sqrt(2.0 / (kernel_height * kernel_width * input_channels))

  #NOTICE: ordering changes from function argument ordering!!
  kernel = std * random.normal(w_key, (kernel_height, kernel_width, output_channels, input_channels))

  #for conv layers there are typically only 1 bias per channel
  biases = jnp.zeros((output_channels,))
  return kernel, biases

#init all the conv layers in a network of given size:
def init_convlayers(sizes, key):
  keys = random.split(key, len(sizes))
  # pdb.set_trace()
  params = []
  for ii in range(len(sizes)):
    params.append(init_single_convlayer(*sizes[ii], keys[ii]))
  return params

nodepert_noisescale = 1e-4

# build the conv forward pass for a single image:
def forward(x, params):
  x = x.reshape(1, data.height, data.width, data.channels).astype(np.float32) # NHWC
  x = jnp.transpose(x, [0,3,1,2])
  h = []; a = []
  h.append(x)

  for (kernel, biases) in params[:-1]:
    convout_channels = kernel.shape[-2]
    #output (lhs) will be in the form NCHW
    act = jax.lax.conv(h[-1],                       # lhs = NCHW image tensor
                       kernel.transpose([2,3,0,1]), # rhs = IOHW conv kernel tensor [according to JAX page]
                       (1, 1),  # window strides
                       'SAME')  # padding mode

    act = act + jnp.repeat(biases, [data.height * data.width]).reshape(1, convout_channels, data.height, data.width)

    a.append(act)
    h.append(relu(a[-1]))

  w, b = params[-1]
  act = jnp.dot(w, h[-1].flatten()) + b
  a.append(act)
  # logsoftmax = a[-1] - logsumexp(a[-1])
  # h.append(logsoftmax)
  output = sigmoid(a[-1])
  h.append(output)
  return h, a

# upgrade to handle batches using 'vmap'
batchforward = jit(vmap(forward, in_axes=(0, None), out_axes=(0, 0)))

# new noisy forward pass:
def noisyforward(x, params, randkey):
  x = x.reshape(1, data.height, data.width, 1).astype(np.float32) # NHWC
  x = jnp.transpose(x, [0,3,1,2])

  h = []; a = []; xi = []; aux = []
  h.append(x)

  for (kernel, biases) in params[:-1]:
    convout_channels = kernel.shape[-2]
    h[-1] = jax.lax.stop_gradient(h[-1])

    # act = jnp.dot(w, h[-1]) + b
    #output (lhs) will be in the form NCHW
    act = jax.lax.conv(h[-1],                           # lhs = NCHW image tensor
                       kernel.transpose([2,3,0,1]), # rhs = IOHW conv kernel tensor [according to JAX page]
                       (1, 1),  # window strides
                       'SAME')  # padding mode

    act = act + jnp.repeat(biases, [data.height * data.width]).reshape(1, convout_channels, data.height, data.width)

    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale * random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    aux.append(jnp.sum(a[-1] * noise * (1/(nodepert_noisescale ** 2))))
    h.append(relu(a[-1]))

  h[-1] = jax.lax.stop_gradient(h[-1])
  w, b = params[-1]
  act = jnp.dot(w, h[-1].flatten()) + b
  randkey, _ = random.split(randkey)
  noise = nodepert_noisescale * random.normal(randkey, act.shape)
  xi.append(noise)
  a.append(act + noise)
  aux.append(jnp.sum(a[-1] * noise * (1/(nodepert_noisescale ** 2))))

  # logsoftmax = a[-1] - logsumexp(a[-1])
  # h.append(logsoftmax)
  output = sigmoid(a[-1])
  h.append(output)
  return h, a, xi, aux

# upgrade to handle batches using 'vmap'
batchnoisyforward = jit(vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0)))
