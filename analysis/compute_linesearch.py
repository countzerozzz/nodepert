import npimports
import importlib
importlib.reload(npimports)
from npimports import *

# main func for linesearch, computes the 'new' parameters of the network (w - lr*dw)
forward = fc.batchforward
noisyforward = fc.batchnoisyforward
batchmseloss = losses.batchmseloss

@jit
def loss_fn(x, y, params):
    h, a = forward(x, params)
    loss = batchmseloss(h[-1], y).sum()
    return loss

@jit
def new_params(x, y, params, randkey, optimstate, update_step):
    lr = optimstate['lr']
    #cannot branch on concrete values in a jit function
    if(update_step.shape[0] == 2):
        grads = grad(loss_fn, argnums = (2))(x, y, params)

    elif (update_step.shape[0] == 1):    
        sigma = fc.nodepert_noisescale
        randkey, _ = random.split(randkey)

        h, a, xi = noisyforward(x, params, randkey)
        noisypred = h[-1]

        h, a = forward(x, params)
        pred = h[-1]

        loss = jnp.mean(jnp.square(pred - y),1)
        noisyloss = jnp.mean(jnp.square(noisypred - y),1)
        lossdiff = (noisyloss - loss)/(sigma**2)

        grads=[]
        for ii in range(len(params)):
            dh = jnp.einsum('ij,i->ij', xi[ii], lossdiff)
            dw = jnp.einsum('ij,ik->kj', h[ii], dh) / x.shape[0]
            db = jnp.mean(dh, 0)
            grads.append((dw,db))
    else:
        print('wrong update step passed!')

    return [(w - lr * dw, b - lr * db)
            for (w, b), (dw, db) in zip(params, grads)], grads, optimstate

def get_delta_l(x, y, params, randkey, optimstate, update_step):
    params_new, grads, optimstate = new_params(x, y, params, randkey, optimstate, update_step)
    return loss_fn(x,y, params_new) - loss_fn(x,y, params)
