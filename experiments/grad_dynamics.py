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
    col_names = ["ss_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    sign_symmetry_df = pd.DataFrame(columns=col_names)
    sign_symmetry_df["update_rule"] = ["np", "sgd"]
    sign_symmetry_df["epoch"] = np.repeat(epoch, 2)
    full_dwnp, full_dwsgd, full_dwtrue = [], [], []

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        col_names, npgrad, sgdgrad, truegrad
    ):
        # do a elementwise product and then calculate the number of entries which are positive
        full_dwnp.extend(dwnp.flatten())
        full_dwsgd.extend(dwsgd.flatten())
        full_dwtrue.extend(dwtrue.flatten())
        tmp = np.array(jnp.multiply(dwnp, dwtrue))
        ss_np = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
        tmp = np.array(jnp.multiply(dwsgd, dwtrue))
        ss_sgd = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)

        sign_symmetry_df[column] = [ss_np, ss_sgd]

    tmp = np.array(jnp.multiply(jnp.asarray(full_dwnp), jnp.asarray(full_dwtrue)))
    ss_np_all = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
    tmp = np.array(jnp.multiply(jnp.asarray(full_dwsgd), jnp.asarray(full_dwtrue)))
    ss_sgd_all = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)

    sign_symmetry_df["ss_all"] = [ss_np_all, ss_sgd_all]

    return sign_symmetry_df


# calculate the norm of the gradient estimates, true gradient
def grad_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ["gnorm_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    grad_norms_df = pd.DataFrame(columns=col_names)
    grad_norms_df["update_rule"] = ["np", "sgd", "true"]
    grad_norms_df["epoch"] = np.repeat(epoch, 3)
    full_dwnp, full_dwsgd, full_dwtrue = [], [], []

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        col_names, npgrad, sgdgrad, truegrad
    ):
        grad_norms_df[column] = [
            jnp.linalg.norm(dwnp),
            jnp.linalg.norm(dwsgd),
            jnp.linalg.norm(dwtrue),
        ]
        full_dwnp.extend(dwnp.flatten())
        full_dwsgd.extend(dwsgd.flatten())
        full_dwtrue.extend(dwtrue.flatten())

    tmpnp = float(np.array(jnp.linalg.norm(jnp.asarray(full_dwnp))))
    tmpsgd = float(np.array(jnp.linalg.norm(jnp.asarray(full_dwsgd))))
    tmptrue = float(np.array(jnp.linalg.norm(jnp.asarray(full_dwtrue))))

    grad_norms_df["gnorm_all"] = [tmpnp, tmpsgd, tmptrue]
    ratio_np = tmpnp / tmptrue
    ratio_sgd = tmpsgd / tmptrue
    grad_norms_df["ratios"] = [ratio_np, ratio_sgd, -1]

    return grad_norms_df


# calculate the norm of the 'noise' in the gradient estimates
def graddiff_norms(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ["gdiff_norm_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    graddiff_norms_df = pd.DataFrame(columns=col_names)
    graddiff_norms_df["update_rule"] = ["np", "sgd"]
    graddiff_norms_df["epoch"] = np.repeat(epoch, 2)

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        col_names, npgrad, sgdgrad, truegrad
    ):
        graddiff_norms_df[column] = [
            jnp.linalg.norm(dwtrue - dwnp),
            jnp.linalg.norm(dwtrue - dwsgd),
        ]

    return graddiff_norms_df


# calculate the angle between the gradient estimates and true gradient
def grad_angles(npgrad, sgdgrad, truegrad, layer_sizes, epoch):
    col_names = ["gangle_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    grad_angles_df = pd.DataFrame(columns=col_names)
    grad_angles_df["update_rule"] = ["np", "sgd"]
    grad_angles_df["epoch"] = np.repeat(epoch, 2)

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        col_names, npgrad, sgdgrad, truegrad
    ):
        dwnp = np.ndarray.flatten(np.asarray(dwnp))
        dwsgd = np.ndarray.flatten(np.asarray(dwsgd))
        dwtrue = np.ndarray.flatten(np.asarray(dwtrue))
        grad_angles_df[column] = [
            angle_between(dwnp, dwtrue),
            angle_between(dwsgd, dwtrue),
        ]

    return grad_angles_df


# calculate the norm of the weights during the crash
def w_norms(params, layer_sizes, epoch):
    col_names = ["norm_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    w_norm_df = pd.DataFrame(columns=col_names)
    w_norm_df["update_rule"] = ["np"]
    w_norm_df["epoch"] = [epoch]
    w_all = []

    for column, (ww, _) in zip(col_names, params):
        flat_ww = ww.flatten()
        tmp = jnp.linalg.norm(flat_ww)
        w_norm_df[column] = [tmp]
        w_all.extend(flat_ww)

    tmp = float(np.array(jnp.linalg.norm(jnp.asarray(w_all))))
    w_norm_df["norm_w_all"] = [tmp]
    return w_norm_df


# calculate the variance of the weights during the crash
def w_vars(params, layer_sizes, epoch):
    col_names = ["var_w" + str(i) for i in np.arange(1, len(layer_sizes))]
    w_var_df = pd.DataFrame(columns=col_names)
    w_var_df["update_rule"] = ["np"]
    w_var_df["epoch"] = [epoch]
    w_all = []

    for column, (ww, _) in zip(col_names, params):
        flat_ww = ww.flatten()
        tmp = jnp.var(flat_ww)
        w_var_df[column] = [tmp]
        w_all.extend(flat_ww)

    tmp = float(np.array(jnp.var(jnp.asarray(w_all))))
    w_var_df["var_w_all"] = [tmp]
    return w_var_df


# get the activations of all layers of the network
def compute_layer_activity(x, y, params, layer_sizes, epoch):
    col_names = ["activity_l" + str(i) for i in np.arange(1, len(layer_sizes))]
    n_activity_df = pd.DataFrame(columns=col_names)
    n_activity_df["update_rule"] = ["np"]
    n_activity_df["epoch"] = [epoch]

    h, a = fc.batchforward(x, params)

    for i in range(1, len(layer_sizes)):
        n_activity_df["activity_l" + str(i)] = jnp.linalg.norm(h[i])

    return n_activity_df
