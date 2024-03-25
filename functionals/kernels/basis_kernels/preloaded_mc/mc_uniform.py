# License: BSD 3 clause

import numpy as np

dict_ker = {}
dict_ker['n_vars'] = 2
dict_ker['var_names'] = ['$\u03C9$', 'a', '$\u03B4$']
dict_ker['var_lower_bounds'] = (10**-10)*np.ones(3)
dict_ker['var_upper_bounds'] = (np.inf)*np.ones(3)


def tphi_func(t, vars_):
    a, delta = vars_
    # print('delta: ', delta)
    # print('a+delta: ', a+delta)
    # tau_scaled = (t-delta)/a
    # print('tau_scaled: ', tau_scaled == 1.)
    # ind = np.where(((tau_scaled >= 0.) & (tau_scaled < 1.)), 1., 0.)
    ind = np.where(((t >= delta) & (t < a+delta)), 1., 0.)
    res = ind/a
    return res


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    pass


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    pass


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def tpsi_func(t, vars_):
    a, delta = vars_
    tau_scaled = (t-delta)/a
    res = np.clip(tau_scaled, 0., 1.)
    return res


dict_ker['tpsi_func'] = tpsi_func


def diff_tpsi_func(t, ix_diff, vars_):
    a, delta = vars_
    tau_scaled = (t-delta)/a
    ind = np.where(((tau_scaled >= 0.) & (tau_scaled <= 1.)), 1, 0)
    ind_scaled = (-1./a)*ind
    if ix_diff == 0:
        # Derivative wrt a
        res = tau_scaled*ind_scaled
    elif ix_diff == 1:
        # Derivative wrt delta
        res = ind_scaled
    return res


dict_ker['diff_tpsi_func'] = diff_tpsi_func


def src_simu_func(rng, vars_, size=1):
    a, delta = vars_
    tau = rng.uniform(low=delta, high=delta+a, size=size)
    return tau


dict_ker['src_simu_func'] = src_simu_func
