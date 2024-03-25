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
    # Define return type for vectorization
    return_float = not uf.is_array(t)
    if return_float:
        t = np.array([t])
    # Computations
    t_del = t-delta
    mask_pos = np.where((t_del >= 0))
    res = np.zeros(t.shape)
    if len(t[mask_pos]) > 0:
        res[mask_pos] = alpha*beta*(1+beta*t_del[mask_pos])**(-1-alpha)
    # Adjust return type for vectorization
    if return_float:
        return res[0]
    else:
        return res


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    # Define return type for vectorization
    return_float = not uf.is_array(t)
    if return_float:
        t = np.array([t])
    # Computations
    t_del = t-delta
    mask_pos = np.where((t_del >= 0))
    res = np.zeros(t.shape)
    if len(t[mask_pos]) > 0:
        v = 1.+beta*t_del[mask_pos]
        if ix_diff == 0:
            # Derivative wrt \alpha
            res[mask_pos] = beta*(1.-alpha*np.log(v))*v**(-1.-alpha)
        elif ix_diff == 1:
            # Derivative wrt \beta
            res[mask_pos] = alpha*(1.-(alpha*beta)*t_del[mask_pos])*v**(-2.-alpha)
        elif ix_diff == 2:
            # Derivative wrt \delta
            coeff = alpha*(1.+alpha)*beta**2
            res[mask_pos] = coeff*v**(-2.-alpha)
    # Adjust return type for vectorization
    if return_float:
        return res[0]
    else:
        return res


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    # Define return type for vectorization
    return_float = not uf.is_array(t)
    if return_float:
        t = np.array([t])
    # Computations
    t_del = t-delta
    mask_pos = np.where((t_del >= 0))
    res = np.zeros(t.shape)
    if len(t[mask_pos]) > 0:
        v = 1.+beta*t_del[mask_pos]
        if ix_diff == 0:
            # Derivative wrt \alpha
            res[mask_pos] = (1./alpha-np.log(v))
        elif ix_diff == 1:
            # Derivative wrt \beta
            u = 1.-(alpha*beta)*t_del[mask_pos]
            res[mask_pos] = (1./beta)*(u/v)
        elif ix_diff == 2:
            # Derivative wrt \delta
            res[mask_pos] = (beta*(1.+alpha))/v
    # Adjust return type for vectorization
    if return_float:
        return res[0]
    else:
        return res


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def tpsi_func(t, vars_):
    alpha, beta, delta = vars_
    # Define return type for vectorization
    return_float = not uf.is_array(t)
    if return_float:
        t = np.array([t])
    # Computations
    t_del = t-delta
    mask_pos = np.where((t_del >= 0))
    res = np.zeros(t.shape)
    if len(t[mask_pos]) > 0:
        res[mask_pos] = 1.-(1+beta*t_del[mask_pos])**-alpha
    # Adjust return type for vectorization
    if return_float:
        return res[0]
    else:
        return res


dict_ker['tpsi_func'] = tpsi_func


def diff_tpsi_func(t, ix_diff, vars_):
    alpha, beta, delta = vars_
    # Define return type for vectorization
    return_float = not uf.is_array(t)
    if return_float:
        t = np.array([t])
    # Computations
    t_del = t-delta
    mask_pos = np.where((t_del >= 0))
    res = np.zeros(t.shape)
    if len(t[mask_pos]) > 0:
        v = 1.+beta*t_del[mask_pos]
        if ix_diff == 0:
            # Derivative wrt \alpha
            res[mask_pos] = np.log(v)*v**-alpha
        elif ix_diff == 1:
            # Derivative wrt \beta
            res[mask_pos] = alpha*t_del[mask_pos]*v**(-1.-alpha)
        elif ix_diff == 2:
            # Derivative wrt \delta
            coeff = -alpha*beta
            res[mask_pos] = coeff*v**(-1.-alpha)
    # Adjust return type for vectorization
    if return_float:
        return res[0]
    else:
        return res


dict_ker['diff_tpsi_func'] = diff_tpsi_func


def src_simu_func(rng, vars_, size=1):
    alpha, beta, delta = vars_
    U = rng.uniform(size=size)
    return delta+(1./beta)*(U**(-1./alpha)-1.)


dict_ker['src_simu_func'] = src_simu_func
