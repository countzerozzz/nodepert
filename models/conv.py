import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import sigmoid


# define element-wise relu:
def relu(x):
  return jnp.maximum(0,x)

#helper function to init conv layer:
def init_convlayer(kernel_height, kernel_width, input_channels, output_channels, key):

  # use he style scaling (not glorot):
  std = np.sqrt(2.0 / kernel_height*kernel_width*input_channels)
  bound = np.sqrt(3.0) * std

  #make HWIO kernel -- note this seems wrong!!! HWOI??? bad naming???
  #NOTICE: ordering changes from argument ordering!!!
  w_key, b_key = random.split(key)
  kernel = random.uniform(w_key,
                          (kernel_height, kernel_width, output_channels, input_channels),
                          minval=-bound, maxval=bound)

  #for conv layers there are typically only 1 bias per channel
  biases = jnp.zeros((output_channels,))
  return kernel, biases

#init all the conv layers in a network of given size:
def init_convlayers(sizes, key):
  keys = random.split(key, len(sizes))
  # pdb.set_trace()
  params = []
  for ss, ii in zip(sizes,range(len(sizes))):
    params.append(init_convlayer(*sizes[ii], keys[ii]))
  return params

n_targets = 10
imgheight = 28
imgwidth = 28
convout_channels = 32
nodepert_noisescale = 1e-6


# build the conv forward pass for a single image:
def forward(x, params):
  x = x.reshape(1,imgheight,imgwidth,1).astype(np.float32) # NHWC
  x = jnp.transpose(x, [0,3,1,2])
  h = []; a = [];
  h.append(x)

  for (kernel, biases) in params[:-1]:

    #output (lhs) will be in the form NCHW
    act = jax.lax.conv(h[-1],                           # lhs = NCHW image tensor
                       kernel.transpose([2,3,0,1]), # rhs = IOHW conv kernel tensor [according to JAX page]
                       (1, 1),  # window strides
                       'SAME')  # padding mode

    act = act + jnp.repeat(biases,[imgheight*imgwidth]).reshape(1, convout_channels, imgheight, imgwidth)

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
  x = x.reshape(1,imgheight,imgwidth,1).astype(np.float32) # NHWC
  x = jnp.transpose(x, [0,3,1,2])

  h = []; a = []; xi = []; aux = []
  h.append(x)

  for (kernel, biases) in params[:-1]:
    # h[-1] = jax.lax.stop_gradient(h[-1])

    # act = jnp.dot(w, h[-1]) + b
    #output (lhs) will be in the form NCHW
    act = jax.lax.conv(h[-1],                           # lhs = NCHW image tensor
                       kernel.transpose([2,3,0,1]), # rhs = IOHW conv kernel tensor [according to JAX page]
                       (1, 1),  # window strides
                       'SAME')  # padding mode

    act = act + jnp.repeat(biases,[imgheight*imgwidth]).reshape(1, convout_channels, imgheight, imgwidth)

    randkey, _ = random.split(randkey)
    noise = nodepert_noisescale*random.normal(randkey, act.shape)
    xi.append(noise)
    a.append(act + noise)
    aux.append(jnp.sum(a[-1]*noise*(1/(nodepert_noisescale**2))))
    h.append(relu(a[-1]))

  # h[-1] = jax.lax.stop_gradient(h[-1])
  w, b = params[-1]
  act = jnp.dot(w, h[-1].flatten()) + b
  randkey, _ = random.split(randkey)
  noise = nodepert_noisescale*random.normal(randkey, act.shape)
  xi.append(noise)
  a.append(act + noise)
  aux.append(jnp.sum(a[-1]*noise*(1/(nodepert_noisescale**2))))

  # logsoftmax = a[-1] - logsumexp(a[-1])
  # h.append(logsoftmax)
  output = sigmoid(a[-1])
  h.append(output)
  return h, a, xi, aux

# upgrade to handle batches using 'vmap'
batchnoisyforward = jit(vmap(noisyforward, in_axes=(0, None, None), out_axes=(0, 0, 0, 0)))
