import npimports
import importlib
importlib.reload(npimports)
from npimports import *

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def sign_symmetry(npgrad, sgdgrad, truegrad, layer_sizes):
    sign_symmetry_df = pd.DataFrame(columns = ['ss_w' + str(i) for i in np.arange(1,len(layer_sizes))])
    sign_symmetry_df['update_rule'] = ""

    return sign_symmetry_df

def graddiff_norms(npgrad, sgdgrad, truegrad, layer_sizes):
    col_names = ['gnorm_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    graddiff_norms_df = pd.DataFrame(columns = col_names)
    graddiff_norms_df['update_rule'] = ["np", "sgd"]
    
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):         
        graddiff_norms_df[column] = [jnp.linalg.norm(dwtrue - dwnp), jnp.linalg.norm(dwtrue - dwsgd)]
        
    return graddiff_norms_df

def grad_angles(npgrad, sgdgrad, truegrad, layer_sizes):
    col_names = ['gangle_w' + str(i) for i in np.arange(1,len(layer_sizes))]
    grad_angles_df = pd.DataFrame(columns = col_names)
    grad_angles_df['update_rule'] = ["np", "sgd"]
    
    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(col_names, npgrad, sgdgrad, truegrad):            
        dwnp = np.ndarray.flatten(np.asarray(dwnp))
        dwsgd = np.ndarray.flatten(np.asarray(dwsgd))
        dwtrue = np.ndarray.flatten(np.asarray(dwtrue))
        grad_angles_df[column] = [angle_between(dwnp, dwtrue), angle_between(dwsgd, dwtrue)]
    
    return grad_angles_df