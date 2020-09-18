import npimports
import importlib
importlib.reload(npimports)
from npimports import *

### FUNCTIONALITY ###
# utility functions for crash_dynamics for measuring the grad_norm, (noise of gradient)_norm, sign symmetry and angle between 'true' gradient and gradient estimates.
###

NEG_INF = -10e6

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

# calculate the cosine of the angle between 2 vectors
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

# ratio of the number of values which are of the same sign between 2 matrices
def sign_symmetry(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ['ss_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    sign_symmetry_df = pd.DataFrame(columns = col_names)
    sign_symmetry_df['update_rule'] = ["np", "sgd"]
    sign_symmetry_df['epoch'] = np.repeat(epoch, 2)
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):
        # do a elementwise product and then calculate the number of entries which are positive
        tmp = np.array(jnp.multiply(dwnp, dwtrue))
        ss_np = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
        tmp = np.array(jnp.multiply(dwsgd, dwtrue))
        ss_sgd = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
        
        sign_symmetry_df[column] = [ss_np, ss_sgd]

    return sign_symmetry_df

# calculate the norm of the gradient estimates, true gradient
def grad_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ['gnorm_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    grad_norms_df = pd.DataFrame(columns = col_names)
    grad_norms_df['update_rule'] = ["np", "sgd", "true"]
    grad_norms_df['epoch'] = np.repeat(epoch, 3)
    
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):         
        grad_norms_df[column] = [jnp.linalg.norm(dwnp), jnp.linalg.norm(dwsgd), jnp.linalg.norm(dwtrue)]
        
    return grad_norms_df

# calculate the norm of the weights during the crash
def w_norms(params, layer_sizes, epoch):
    col_names = ['norm_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    w_norm_df = pd.DataFrame(columns = col_names)
    w_norm_df['update_rule'] = ["np"]
    w_norm_df['epoch'] = [epoch]
    w_all = 0
    
    for column, (ww, _) in zip(col_names, params):         
        tmp = jnp.linalg.norm(ww)
        w_norm_df[column] = [tmp]
        # w_all.extend(jax.numpy.ravel(ww, order='C'))
        w_all += tmp
    
    w_norm_df['all'] = [w_all]
    return w_norm_df

# calculate the norm of the 'noise' in the gradient estimates
def graddiff_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ['gdiff_norm_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    graddiff_norms_df = pd.DataFrame(columns = col_names)
    graddiff_norms_df['update_rule'] = ["np", "sgd"]
    graddiff_norms_df['epoch'] = np.repeat(epoch, 2)
    
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):         
        graddiff_norms_df[column] = [jnp.linalg.norm(dwtrue - dwnp), jnp.linalg.norm(dwtrue - dwsgd)]
        
    return graddiff_norms_df

# calculate the angle between the gradient estimates and true gradient
def grad_angles(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ['gangle_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    grad_angles_df = pd.DataFrame(columns = col_names)
    grad_angles_df['update_rule'] = ["np", "sgd"]
    grad_angles_df['epoch'] = np.repeat(epoch, 2)
    
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):            
        dwnp = np.ndarray.flatten(np.asarray(dwnp))
        dwsgd = np.ndarray.flatten(np.asarray(dwsgd))
        dwtrue = np.ndarray.flatten(np.asarray(dwtrue))
        grad_angles_df[column] = [angle_between(dwnp, dwtrue), angle_between(dwsgd, dwtrue)]
    
    return grad_angles_df