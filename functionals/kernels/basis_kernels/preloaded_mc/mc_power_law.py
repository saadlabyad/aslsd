# License: BSD 3 clause

import numpy as np

dict_ker = {}
dict_ker['n_vars'] = 2
dict_ker['var_names'] = ['$\u03C9$', '$\u03B1$', '$\u03B2$']
dict_ker['var_lower_bounds'] = (10**-10)*np.ones(3)
dict_ker['var_upper_bounds'] = np.inf*np.ones(3)


def tphi_func(t, vars_):
    alpha = vars_[0]
    beta = vars_[1]
    return alpha*beta/(1+beta*t)**(1+alpha)


dict_ker['tphi_func'] = tphi_func


def diff_tphi_func(t, ix_diff, vars_):
    alpha, beta = vars_
    v = 1.+beta*t
    if ix_diff == 0:
        # Derivative wrt \alpha
        return beta*(1.-alpha*np.log(v))*v**(-1.-alpha)
    elif ix_diff == 1:
        # Derivative wrt \beta
        return alpha*(1.-(alpha*beta)*t)*v**(-2.-alpha)


dict_ker['diff_tphi_func'] = diff_tphi_func


def diff_log_tphi_func(t, ix_diff, vars_):
    alpha, beta = vars_
    v = 1.+beta*t
    if ix_diff == 0:
        # Derivative wrt \alpha
        return 1./alpha-np.log(v)
    elif ix_diff == 1:
        # Derivative wrt \beta
        u = 1.-(alpha*beta)*t
        return (1./beta)*(u/v)


dict_ker['diff_log_tphi_func'] = diff_log_tphi_func


def tpsi_func(t, vars_):
    alpha, beta = vars_
    return 1.-(1+beta*t)**-alpha


dict_ker['tpsi_func'] = tpsi_func


def diff_tpsi_func(t, ix_diff, vars_):
    alpha, beta = vars_
    v = 1.+beta*t
    if ix_diff == 0:
        # Derivative wrt \alpha
        return np.log(v)*v**-alpha
    elif ix_diff == 1:
        # Derivative wrt \beta
        return alpha*t*v**(-1.-alpha)


dict_ker['diff_tpsi_func'] = diff_tpsi_func


def src_simu_func(rng, vars_, size=1):
    alpha, beta = vars_
    U = rng.uniform(size=size)
    tau = (1./beta)*(U**(-1./alpha)-1.)
    return tau


dict_ker['src_simu_func'] = src_simu_func
