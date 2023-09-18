# License: BSD 3 clause

import numpy as np

from aslsd.utilities import useful_functions as uf

dict_ker = {}
dict_ker['n_vars'] = 2
dict_ker['var_names'] = ['$\u03C9$', '$\u03B2$', '$\u03B4$']
dict_ker['var_lower_bounds'] = np.array([10**-10, 10**-10, 10**-10])
dict_ker['var_upper_bounds'] = np.array([np.inf, np.inf, np.inf])


def tphi_func(t, vars_):
    beta, delta = vars_
    return uf.gaussian_pdf(t, delta, beta)/uf.normal_cdf(delta/beta)


dict_ker['tphi_func'] = tphi_func


def tpsi_func(t, vars_):
    beta, delta = vars_
    F_t = uf.normal_cdf((t-delta)/beta)
    F_par = uf.normal_cdf(delta/beta)
    return 1.+(F_t-1.)/F_par


dict_ker['tpsi_func'] = tpsi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    beta, delta = vars_
    d_b_ratio = delta/beta
    gauss_ratio = uf.gaussian_pdf(d_b_ratio)/uf.normal_cdf(d_b_ratio)
    scaled_t = (t-delta)/beta
    if ix_diff == 0:
        # Derivative wrt \beta
        return (1./beta)*(scaled_t**2+d_b_ratio*gauss_ratio-1.)
    elif ix_diff == 1:
        # Derivative wrt \delta
        return (1./beta)*(scaled_t-gauss_ratio)


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    beta, delta = vars_
    if ix_diff == 0:
        # Derivative wrt \beta
        return diff_log_tphi_func(t, ix_diff, vars_)*tphi_func(t, vars_)
    elif ix_diff == 1:
        # Derivative wrt \delta
        return diff_log_tphi_func(t, ix_diff, vars_)*tphi_func(t, vars_)


dict_ker['diff_tphi_func'] = diff_tphi_func


def src_simu_func(rng, vars_, size=1):
    beta, delta = vars_
    U = rng.uniform(size=size)
    f_ratio = uf.normal_cdf(delta/beta)
    return delta+beta*uf.inverse_normal_cdf(1.-f_ratio*U)


dict_ker['src_simu_func'] = src_simu_func
