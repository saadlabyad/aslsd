# License: BSD 3 clause

import numpy as np

dict_ker = {}
dict_ker['n_vars'] = 1
dict_ker['var_names'] = ['$\u03C9$', '$\u03B2$']
dict_ker['var_lower_bounds'] = np.array([10**-10, 10**-10])
dict_ker['var_upper_bounds'] = np.array([np.inf, np.inf])


def tphi_func(t, vars_):
    return vars_[0]*np.exp(-vars_[0]*t)


dict_ker['tphi_func'] = tphi_func


def tpsi_func(t, vars_):
    return 1.-np.exp(-vars_[0]*t)


dict_ker['tpsi_func'] = tpsi_func


def diff_tpsi_func(t, ix_diff, vars_):
    return t*np.exp(-vars_[0]*t)


dict_ker['diff_tpsi_func'] = diff_tpsi_func


def diff_tphi_func(t, ix_diff, vars_):
    if ix_diff == 0:
        # Derivative wrt \beta
        return np.exp(-vars_[0]*t)*(1.-vars_[0]*t)


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    beta = vars_[0]
    if ix_diff == 0:
        # Derivative wrt \beta
        return (1./beta)-t


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def src_simu_func(rng, vars_, size=1):
    beta = vars_[0]
    return rng.exponential(scale=1./beta, size=size)


dict_ker['src_simu_func'] = src_simu_func
