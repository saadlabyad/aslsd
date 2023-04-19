# License: BSD 3 clause

import numpy as np

import aslsd.utilities.useful_functions as uf


dict_ker = {}
dict_ker['n_vars'] = 2
dict_ker['var_names'] = ['$\u03C9$', '$\u03B2$', '$\u03B4$']
dict_ker['var_bounds'] = np.array([10**-10, 10**-10, 10**-10])


def tphi_func(t, vars_):
    beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) > 0:
            res[mask_pos] = beta*np.exp(-beta*t[mask_pos])
        return res
    # float implementation
    else:
        if t >= delta:
            return beta*np.exp(-beta*(t-delta))
        else:
            return 0.


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) == 0:
            return res
        beta_t = beta*(t[mask_pos]-delta)
        exp_beta = np.exp(-beta_t)
        if ix_diff == 0:
            # Derivative wrt \beta
            res[mask_pos] = (1.-beta_t)*exp_beta
            return res
        if ix_diff == 1:
            # Derivative wrt \delta
            res[mask_pos] = (beta**2)*exp_beta
            return res
    # Float implementation
    else:
        if t < delta:
            return 0.
        if ix_diff == 0:
            # Derivative wrt \beta
            return (1.-beta*(t[mask_pos]-delta))*np.exp(-beta*(t-delta))
        if ix_diff == 1:
            # Derivative wrt \delta
            return (beta**2)*np.exp(-beta*(t-delta))


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) == 0:
            return res
        if ix_diff == 0:
            # Derivative wrt \beta
            res[mask_pos] = (1./beta-(t[mask_pos]-delta))
            return res
        if ix_diff == 1:
            # Derivative wrt \delta
            res[mask_pos] = beta
            return res
    # Float implementation
    else:
        if t < delta:
            return 0.
        if ix_diff == 0:
            # Derivative wrt \beta
            return (1./beta-(t-delta))
        if ix_diff == 1:
            # Derivative wrt \delta
            return beta


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def src_simu_func(rng, vars_, size=1):
    beta, delta = vars_
    return delta+rng.exponential(scale=1./beta, size=size)


dict_ker['src_simu_func'] = src_simu_func