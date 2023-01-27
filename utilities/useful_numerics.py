# License: BSD 3 clause

import copy

import numpy as np
from tqdm import tqdm

from aslsd.utilities import useful_functions as uf


# =============================================================================
# Error functions
# =============================================================================
def abs_error_func(x, y):
    return np.abs(x-y)


def inversion_error(y, x_estim, func, p=1, **kwargs):
    return np.abs(y-func(x_estim, **kwargs))**p


# =============================================================================
# Initialisation
# =============================================================================
def refine_interval(theta_l, theta_r, func, f_args=None, n_res=10**2):
    vec_theta = np.linspace(theta_l, theta_r, n_res)
    if f_args is None:
        f_args = {}
    vec_y = func(vec_theta, **f_args)
    return vec_theta, vec_y


def refine_inversion_refs(ref_theta, ref_y, func, f_args=None, n_res=10**2):
    L_theta = []
    L_y = []
    for ix in range(len(ref_theta)-1):
        theta_l = ref_theta[ix]
        theta_r = ref_theta[ix+1]
        vec_theta, vec_y = refine_interval(theta_l, theta_r, func,
                                           f_args=f_args, n_res=n_res)
        L_theta.extend(list(vec_theta[:-1]))
        L_y.extend(list(vec_y[:-1]))
    L_theta.append(ref_theta[-1])
    L_y.append(ref_y[-1])
    return np.array(L_theta), np.array(L_y)


def initialise_grid(y, L_theta, L_y, start_type='middle', rng=None, seed=1234):
    # Get the ixs such that ref_x[ix-1] <= x < ref_x[ix]
    ixs_grid = np.searchsorted(L_y, y, side='right')
    # Deal with the right bound case
    ixs_grid[ixs_grid >= len(L_theta)] = len(L_theta)-1
    if start_type == 'middle':
        return 0.5*(L_theta[ixs_grid-1]+L_theta[ixs_grid])
    elif start_type == 'left':
        return copy.deepcopy(L_theta[ixs_grid-1])
    elif start_type == 'right':
        return copy.deepcopy(L_theta[ixs_grid])
    elif start_type == 'random':
        if rng is None:
            rng = np.random.default_rng(seed)
        if uf.is_array(y):
            theta_0 = np.zeros(len(y))
            for ix in range(len(y)):
                theta_0[ix] = rng.uniform(L_theta[ixs_grid[ix]-1],
                                          L_theta[ixs_grid[ix]])
        else:
            theta_0 = rng.uniform(L_theta[ixs_grid[0]-1],
                                  L_theta[ixs_grid[0]])
        return theta_0


# =============================================================================
# Numerical Schemes
# =============================================================================
def get_newton(y, ratio_func=None, error_func=None, theta_0=None,
               initialiser=None, init_args=None, rng=None,
               seed=1234, lower_bound=None, upper_bound=None, n_iter=10**3,
               tol=10**-5, verbose=False, **kwargs):
    # Solve f(\theta) = y
    if rng is None:
        rng = np.random.default_rng(seed)
    if theta_0 is None:
        if initialiser is None:
            theta_0 = rng.uniform(lower_bound, upper_bound)
        else:
            if init_args is None:
                init_args = {}
            theta_0 = initialiser(y, **init_args)

    if error_func is None:
        error_func = abs_error_func
    return_float = False
    if not uf.is_array(y):
        y = np.array([y])
        return_float = True
    len_y = len(y)
    if not uf.is_array(theta_0):
        theta_0 = theta_0*np.ones(len_y)

    theta = theta_0
    active_ixs = np.arange(len_y)
    for n in tqdm(range(n_iter), disable=not verbose):
        theta_prev = copy.deepcopy(theta)
        theta[active_ixs] = theta[active_ixs]-ratio_func(y[active_ixs],
                                                         theta[active_ixs],
                                                         **kwargs)
        active_ixs = np.where((error_func(theta, theta_prev) >= tol))[0]
        if uf.is_empty(active_ixs):
            if verbose:
                print('Method converged after ', n, ' iterations.')
            break

    if return_float:
        return theta[0]
    else:
        return theta
