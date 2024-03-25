# License: BSD 3 clause

import numpy as np
from aslsd.utilities import useful_functions as uf

dict_ker = {}
dict_ker['n_vars'] = 3
dict_ker['var_names'] = ['$\u03C9$', '$\u03B1$', '$\u03B2$', '$\u03B4']
dict_ker['var_lower_bounds'] = (10**-10)*np.ones(4)
dict_ker['var_upper_bounds'] = (np.inf)*np.ones(4)


def make_tri_corners(alpha, beta, delta):
    c_1 = alpha
    c_2 = c_1+beta
    c_3 = c_2+delta
    return c_1, c_2, c_3


def tphi_func(t, vars_):
    alpha, beta, delta = vars_
    b_div_d = beta/delta
    mul_coeff = 2./(beta+delta)
    return_float = False
    if not uf.is_array(t):
        t = np.array([t])
        return_float = True
    tau_scaled = (t-alpha)/beta
    ixs_12 = np.where((tau_scaled >= 0) & (tau_scaled < 1))
    ixs_23 = np.where((tau_scaled >= 1) & (tau_scaled < (1+delta/beta)))
    res = np.zeros(t.shape)
    if len(ixs_12) != 0:
        res[ixs_12] += mul_coeff*tau_scaled[ixs_12]
    if len(ixs_23) != 0:
        res[ixs_23] += mul_coeff*((1+b_div_d)-b_div_d*tau_scaled[ixs_23])
    if return_float:
        res = res[0]
    return res


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    pass


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    pass


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def src_simu_func(rng, vars_, size=1):
    alpha, beta, delta = vars_
    c_1, c_2, c_3 = make_tri_corners(alpha, beta, delta)
    return rng.triangular(c_1, c_2, c_3, size=size)


dict_ker['src_simu_func'] = src_simu_func
