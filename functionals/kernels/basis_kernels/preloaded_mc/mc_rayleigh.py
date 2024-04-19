# License: BSD 3 clause

import numpy as np

dict_ker = {}
dict_ker['n_vars'] = 1
dict_ker['var_names'] = ['$\u03C9$', '$\u03B2$']
dict_ker['var_lower_bounds'] = (10**-10)*np.ones(2)
dict_ker['var_upper_bounds'] = (np.inf)*np.ones(2)


def tphi_func(t, vars_):
    beta = vars_[0]
    return (t/beta**2)*np.exp(-0.5*(t/beta)**2)


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    beta = vars_[0]
    if ix_diff == 0:
        # Derivative wrt \beta
        pass


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    beta = vars_[0]
    if ix_diff == 0:
        # Derivative wrt \beta
        pass


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def tpsi_func(t, vars_):
    beta = vars_[0]
    res = 1.-np.exp(-0.5*(t/beta)**2)
    return res


dict_ker['tpsi_func'] = tpsi_func


def src_simu_func(rng, vars_, size=1):
    beta = vars_[0]
    return rng.rayleigh(scale=beta, size=size)


dict_ker['src_simu_func'] = src_simu_func
