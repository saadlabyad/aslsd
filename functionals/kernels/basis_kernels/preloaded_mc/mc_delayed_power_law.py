# License: BSD 3 clause

import numpy as np

import aslsd.utilities.useful_functions as uf


dict_ker = {}
dict_ker['n_vars'] = 3
dict_ker['var_names'] = ['$\u03C9$', '$\u03B1$', '$\u03B2$', '$\u03B4$']
dict_ker['var_lower_bounds'] = (10**-10)*np.ones(4)
dict_ker['var_upper_bounds'] = (np.inf)*np.ones(4)


def tphi_func(t, vars_):
    alpha, beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) > 0:
            res[mask_pos] = alpha*beta/(1+beta*(t[mask_pos]-delta))**(1+alpha)
        return res
    # float implementation
    else:
        if t >= delta:
            return alpha*beta/(1+beta*(t-delta))**(1+alpha)
        else:
            return 0.


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) == 0:
            return res
        if ix_diff == 0:
            # Derivative wrt \alpha
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = tphi_func(t[mask_pos], vars_)*(1./alpha-np.log(v))
            return res
        elif ix_diff == 1:
            # Derivative wrt \beta
            u = 1.-(alpha*beta)*(t[mask_pos]-delta)
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = tphi_func(t[mask_pos], vars_)*((1./beta)*(u/v))
            return res
        elif ix_diff == 2:
            # Derivative wrt \delta
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = tphi_func(t[mask_pos], vars_)*((beta*(1.+alpha))/v)
            return res
    # float implementation
    else:
        if t < delta:
            return 0.
        if ix_diff == 0:
            # Derivative wrt \alpha
            v = 1.+beta*(t-delta)
            return tphi_func(t, vars_)*(1./alpha-np.log(v))
        elif ix_diff == 1:
            # Derivative wrt \beta
            u = 1.-(alpha*beta)*(t-delta)
            v = 1.+beta*(t-delta)
            return tphi_func(t, vars_)*((1./beta)*(u/v))
        elif ix_diff == 2:
            # Derivative wrt \delta
            v = 1.+beta*(t-delta)
            return tphi_func(t, vars_)*((beta*(1.+alpha))/v)


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    # Vector implementation
    if uf.is_array(t):
        mask_pos = np.where((t >= delta))
        res = np.zeros(t.shape)
        if len(t[mask_pos]) == 0:
            return res
        if ix_diff == 0:
            # Derivative wrt \alpha
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = (1./alpha-np.log(v))
            return res
        elif ix_diff == 1:
            # Derivative wrt \beta
            u = 1.-(alpha*beta)*(t[mask_pos]-delta)
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = ((1./beta)*(u/v))
            return res
        elif ix_diff == 2:
            # Derivative wrt \delta
            v = 1.+beta*(t[mask_pos]-delta)
            res[mask_pos] = (beta*(1.+alpha))/v
            return res
    # float implementation
    else:
        if t < delta:
            return 0.
        if ix_diff == 0:
            # Derivative wrt \alpha
            v = 1.+beta*(t-delta)
            return (1./alpha-np.log(v))
        elif ix_diff == 1:
            # Derivative wrt \beta
            u = 1.-(alpha*beta)*(t-delta)
            v = 1.+beta*(t-delta)
            return ((1./beta)*(u/v))
        elif ix_diff == 2:
            # Derivative wrt \delta
            v = 1.+beta*(t-delta)
            return ((beta*(1.+alpha))/v)


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def src_simu_func(rng, vars_, size=1):
    alpha, beta, delta = vars_
    U = rng.uniform(size=size)
    return delta+(1./beta)*(U**(-1./alpha)-1.)


dict_ker['src_simu_func'] = src_simu_func
