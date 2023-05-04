import jax.numpy as jnp
import pdb as pdb
import numpy as np
import pandas as pd
import nodepert.model.fc as fc

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


def compute_grad_stats(npgrad, sgdgrad, truegrad):
    dataframes = {}
    num_layers = len(truegrad)
    # dataframes["sign_symmetry"] = compute_sign_symmetry(num_layers, npgrad, sgdgrad, truegrad)
    # print("sign_symmetry done")
    dataframes["gradient_norm"] = compute_gradient_norm(num_layers, npgrad, sgdgrad, truegrad)
    print("gradient_norm done")
    dataframes["graddiff_norm"] = compute_graddiff_norm(num_layers, npgrad, sgdgrad, truegrad)
    print("graddiff_norm done")
    dataframes["angle_with_true_gradient"] = compute_angle_with_true_gradient(npgrad, sgdgrad, truegrad)
    print("angle_with_true_gradient done")

    return dataframes

def compute_params_stats(params):
    dataframes = {}
    num_layers = len(params)
    dataframes["weight_norm"] = compute_weight_norm(num_layers, params)
    dataframes["weight_variance"] = compute_weight_variance(num_layers, params)

    return dataframes

# get the activations of all layers of the network
def compute_activity_stats(params, x):
    dataframes = {}
    num_layers = len(params)
    forward = fc.build_batchforward()
    h, a = forward(x, params)

    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    for i in range(1, num_layers):
        df["activity_l" + str(i)] = jnp.linalg.norm(h[i])

    dataframes["activity"] = df
    return dataframes

# ratio of the number of values which are of the same sign between 2 matrices
# this is super inefficient!! see if we can jit all these functions
def compute_sign_symmetry(num_layers, npgrad, sgdgrad, truegrad):
    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np", "sgd"]
    full_dwnp, full_dwsgd, full_dwtrue = [], [], []

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        columns, npgrad, sgdgrad, truegrad
    ):
        # do a elementwise product and then calculate the number of entries which are positive
        full_dwnp.extend(dwnp.flatten())
        full_dwsgd.extend(dwsgd.flatten())
        full_dwtrue.extend(dwtrue.flatten())
        tmp = np.array(jnp.multiply(dwnp, dwtrue))
        ss_np = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
        tmp = np.array(jnp.multiply(dwsgd, dwtrue))
        ss_sgd = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)

        df[column] = [ss_np, ss_sgd]

    tmp = np.array(jnp.multiply(jnp.asarray(full_dwnp), jnp.asarray(full_dwtrue)))
    ss_np_all = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)
    tmp = np.array(jnp.multiply(jnp.asarray(full_dwsgd), jnp.asarray(full_dwtrue)))
    ss_sgd_all = np.sum(tmp >= 0) / np.sum(tmp > NEG_INF)

    df["all_layers"] = [ss_np_all, ss_sgd_all]

    return df


# calculate the norm of the gradient estimates, true gradient
def compute_gradient_norm(num_layers, npgrad, sgdgrad, truegrad):
    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np", "sgd", "true"]
    full_dwnp, full_dwsgd, full_dwtrue = [], [], []

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        columns, npgrad, sgdgrad, truegrad
    ):
        df[column] = [
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

    df["all_layers"] = [tmpnp, tmpsgd, tmptrue]

    return df


# calculate the norm of the 'noise' in the gradient estimates
def compute_graddiff_norm(num_layers, npgrad, sgdgrad, truegrad):
    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np", "sgd"]

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        columns, npgrad, sgdgrad, truegrad
    ):
        df[column] = [
            jnp.linalg.norm(dwtrue - dwnp),
            jnp.linalg.norm(dwtrue - dwsgd),
        ]

    return df


# calculate the angle between the gradient estimates and true gradient
def compute_angle_with_true_gradient(num_layers, npgrad, sgdgrad, truegrad):
    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np", "sgd"]

    for column, (dwnp, _), (dwsgd, _), (dwtrue, _) in zip(
        columns, npgrad, sgdgrad, truegrad
    ):
        dwnp = np.ndarray.flatten(np.asarray(dwnp))
        dwsgd = np.ndarray.flatten(np.asarray(dwsgd))
        dwtrue = np.ndarray.flatten(np.asarray(dwtrue))
        df[column] = [
            angle_between(dwnp, dwtrue),
            angle_between(dwsgd, dwtrue),
        ]

    return df


# calculate the norm of the weights during the crash
def compute_weight_norm(num_layers, params):
    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np"]
    w_all = []

    for column, (ww, _) in zip(columns, params):
        flat_ww = ww.flatten()
        tmp = jnp.linalg.norm(flat_ww)
        df[column] = [tmp]
        w_all.extend(flat_ww)

    tmp = float(np.array(jnp.linalg.norm(jnp.asarray(w_all))))
    df["all_layers"] = [tmp]
    return df


# calculate the variance of the weights during the crash
def compute_weight_variance(num_layers, params):

    columns = ["layer_" + str(i) for i in np.arange(1, num_layers)]
    df = pd.DataFrame(columns=columns)
    df["update_rule"] = ["np"]
    w_all = []

    for column, (ww, _) in zip(columns, params):
        flat_ww = ww.flatten()
        tmp = jnp.var(flat_ww)
        df[column] = [tmp]
        w_all.extend(flat_ww)

    tmp = float(np.array(jnp.var(jnp.asarray(w_all))))
    df["var_w_all"] = [tmp]
    return df

