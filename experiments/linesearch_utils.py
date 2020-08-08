import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ### 
# this is just the utils for performing the linesearch functionality of the code (linesearch.py)
### 

@jit
def lossfunc(x, y, params):
    h, a = forward(x, params)
    loss = losses.batchmseloss(h[-1], y).sum()
    return loss

def linesearchfunc(randkey, params, start, stop, num, meta_data):
    global forward
    
    learning_rates = np.logspace(start, stop, num, endpoint=True, base=10, dtype=np.float32)
    forward, step_type, n_hl, update_rule, hl_size = meta_data
    df = pd.DataFrame()
    df['lr'] = learning_rates
    npdelta_l = []
    sgddelta_l = []

    for lr in learning_rates:
        nptmp = []
        sgdtmp = []
        optimstate = { 'lr' : lr, 't' : 0 }
        
        if(step_type == 'single-step'):
            # set the percentage of test samples which should be averaged over, while calculating delta_L 
            for x, y in data.get_data_batches(batchsize=100, split='test[:25%]'):
                randkey, _ = random.split(randkey)
                params_new, _, _ = optim.npupdate(x, y, params, randkey, optimstate)
                nptmp.append(lossfunc(x,y, params_new) - lossfunc(x,y, params))
                params_new, _, _ = optim.sgdupdate(x, y, params, randkey, optimstate)
                sgdtmp.append(lossfunc(x,y, params_new) - lossfunc(x,y, params))

            npdelta_l.append(np.array(nptmp))
            # npdelta_l.append(np.array(nptmp).mean)
            sgddelta_l.append(np.array(sgdtmp).mean)

    print([x for x in npdelta_l[i].mean])
    df['npdelta_l'] = npdelta_l
    df['sgddelta_l'] = sgddelta_l
    print(df['npdelta_l'])
    print(df['npdelta_l'].dtype)
    df['npdelta_l'] = df['npdelta_l'].astype('float32')
    npopt_lr = df['lr'][df['npdelta_l'].idxmin()]
    # sgdopt_lr = df['lr'][df['sgddelta_l'].astype('float32').idxmin()]
    
    df['npopt_lr'], df['sgdopt_lr'] = npopt_lr, sgdopt_lr
    df['step_type'], df['n_hl'], df['trajectory'], df['hl_size'] = step_type, n_hl, update_rule, hl_size
    return df