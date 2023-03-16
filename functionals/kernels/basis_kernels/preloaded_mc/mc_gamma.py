# License: BSD 3 clause

import numpy as np
from scipy.special import gamma, digamma

dict_ker = {}
dict_ker['n_vars'] = 2
dict_ker['var_names'] = ['$\u03C9$', '$\u03B1$', '$\u03B2$']
dict_ker['var_bounds'] = 10**-10*np.ones(3)


def tphi_func(t, vars_):
    alpha = vars_[0]
    beta = vars_[1]
    coeff = (beta**alpha)/gamma(alpha)
    return coeff*(t**(alpha-1))*np.exp(-beta*t)


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    alpha = vars_[0]
    beta = vars_[1]
    if ix_diff == 0:
        # Derivative wrt \alpha
        phi_res = tphi_func(t, vars_)
        v = np.log(t)+(np.log(beta)-digamma(alpha))
        return phi_res*v
    elif ix_diff == 1:
        # Derivative wrt \beta
        phi_res = tphi_func(t, vars_)
        v = (alpha/beta)-t
        return phi_res*v


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    alpha = vars_[0]
    beta = vars_[1]
    if ix_diff == 0:
        # Derivative wrt \alpha
        return np.log(t)+(np.log(beta)-digamma(alpha))
    elif ix_diff == 1:
        # Derivative wrt \beta
        return (alpha/beta)-t


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def src_simu_func(rng, vars_, size=1):
    alpha = vars_[0]
    beta = vars_[1]
    return rng.gamma(alpha, scale=1.0/beta, size=size)


dict_ker['src_simu_func'] = src_simu_func
