# License: BSD 3 clause
import copy

import numpy as np

from aslsd.utilities import useful_functions as uf


def make_variables(vars_, pos_min_f=0., intercept=False):
    # alpha, a, b, c
    # alpha, a, b, delta=c+alpha*pos_min(f)
    alpha = vars_[0]
    a = vars_[1]
    b = vars_[2]
    if intercept:
        c = vars_[3]+alpha*pos_min_f
    else:
        c = 0.
    return alpha, a, b, c


def get_mu(t, f, vars_=None, pos_min_f=0., intercept=False, alpha=None, a=None,
           b=None, c=0.):
    if alpha is None:
        alpha, a, b, c = make_variables(vars_, pos_min_f=pos_min_f,
                                        intercept=intercept)
    return c+alpha*f(a*t+b)


def precompute_factors(F, T_p, alpha, a, b, c=0):
    factor_1 = a/(c*T_p+alpha*F(T_p))
    inv_factor_1 = 1./factor_1
    factor_2 = (alpha*F(b)+c*b)/a
    if c == 0:
        epsilon = None
        factor_3 = None
    else:
        epsilon = alpha/c
        factor_3 = a/c
    factor_4 = a/alpha
    return factor_1, inv_factor_1, factor_2, factor_3, factor_4, epsilon


def get_ktheta_intercept(y, T_p, alpha, a, b, c, F, inverse_h, factor_1=None,
                         inv_factor_1=None,
                         factor_2=None, factor_3=None, epsilon=None):
    if factor_1 is None:
        factor_1 = a/(c*T_p+alpha*F(T_p))
    if inv_factor_1 is None:
        inv_factor_1 = 1./factor_1
    if factor_2 is None:
        factor_2 = (alpha*F(b)+c*b)/a
    k_float = factor_1*(y+factor_2)
    k = k_float.astype(int)
    if epsilon is None:
        epsilon = alpha/c
    if factor_3 is None:
        factor_3 = a/c
    z = factor_3*(y-inv_factor_1*k+factor_2)
    theta = inverse_h(z, epsilon=epsilon)
    return k, theta


def get_ktheta_no_intercept(y, T_p, alpha, a, b, F, inverse_F, factor_1=None,
                            inv_factor_1=None, factor_2=None, factor_4=None):
    if factor_1 is None:
        factor_1 = a/(alpha*F(T_p))
    if inv_factor_1 is None:
        inv_factor_1 = 1./factor_1
    if factor_2 is None:
        factor_2 = (alpha*F(b))/a
    k_float = factor_1*(y+factor_2)
    k = k_float.astype(int)
    if factor_4 is None:
        factor_4 = a/alpha
    z = factor_4*(y-inv_factor_1*k+factor_2)
    theta = inverse_F(z)
    return k, theta


def get_ktheta(y, T_p, F, alpha, a, b, c=0, inverse_F=None, inverse_h=None,
               factor_1=None,
               inv_factor_1=None, factor_2=None, factor_3=None, factor_4=None,
               epsilon=None):
    if c == 0:
        return get_ktheta_no_intercept(y, T_p, alpha, a, b, F, inverse_F,
                                       factor_1=factor_1,
                                       inv_factor_1=inv_factor_1,
                                       factor_2=factor_2, factor_4=factor_4)
    else:
        return get_ktheta_intercept(y, T_p, alpha, a, b, c, F, inverse_h,
                                    factor_1=factor_1,
                                    inv_factor_1=inv_factor_1,
                                    factor_2=factor_2, factor_3=factor_3,
                                    epsilon=epsilon)


def ktheta2inv(k, theta, T_p, a, b):
    return (k*T_p+theta-b)/a


def get_t(y, T_p, F, alpha, a, b, c=0, inverse_F=None, inverse_h=None,
          factor_1=None, inv_factor_1=None, factor_2=None, factor_3=None,
          factor_4=None, epsilon=None):
    k, theta = get_ktheta(y, T_p, F, alpha, a, b, c=c, inverse_F=inverse_F,
                          inverse_h=inverse_h, factor_1=factor_1,
                          inv_factor_1=inv_factor_1, factor_2=factor_2,
                          factor_3=factor_3, epsilon=epsilon)
    t = ktheta2inv(k, theta, T_p, a, b)
    return t


def t2ktheta(t, T_p, a, b):
    k_float = (a*t+b)/T_p
    if uf.is_array(t):
        k = k_float.astype(int)
    else:
        k = int(k_float)
    theta = a*t+b-k*T_p
    return k, theta


def get_M(t, T_p, F_q, F=None, vars_=None, pos_min_f=0., intercept=False,
          alpha=None, a=None, b=None, c=0.):
    if alpha is None:
        alpha, a, b, c = make_variables(vars_, pos_min_f=pos_min_f,
                                        intercept=intercept)
    k, theta = t2ktheta(t, T_p, a, b)
    div_coeff = 1./(a*t)
    res = (alpha**2)*(k*F_q(T_p)+F_q(theta)-F_q(b))
    if c > 0.:
        res += 2.*alpha*c*(k*F(T_p)+F(theta)-F(b))
    res = res*div_coeff
    res += c**2
    return res


def get_diff_M(t, ix_diff, T_p, f, F_q, F=None, vars_=None, pos_min_f=0.,
               intercept=False,
               alpha=None, a=None, b=None, c=0.):
    if alpha is None:
        alpha, a, b, c = make_variables(vars_, pos_min_f=pos_min_f,
                                        intercept=intercept)
    k, theta = t2ktheta(t, T_p, a, b)
    div_coeff = 1./(a*t)

    if ix_diff == 0:
        # Derivative wrt \alpha
        res = 2.*alpha*div_coeff*(k*F_q(T_p)+F_q(theta)-F_q(b))
        if intercept:
            delta = vars_[3]
            res += 2*pos_min_f*c+2.*div_coeff*((delta+2.*alpha*pos_min_f)
                                               * (k*F(T_p)+F(theta)-F(b)))
    elif ix_diff == 1:
        # Derivative wrt a
        mul_coeff = 1./div_coeff
        scaled_t = a*t+b
        res = (alpha**2)*(mul_coeff*(f(scaled_t))**2-F_q(scaled_t)+F_q(b))
        if c > 0.:
            res += 2.*alpha*c*(mul_coeff*f(scaled_t)-F(scaled_t)+F(b))
        res = (1./a)*div_coeff*res
    elif ix_diff == 2:
        # Derivative wrt b
        scaled_t = a*t+b
        res = (alpha**2)*((f(scaled_t))**2-(f(b))**2)
        if c > 0.:
            res += 2.*alpha*c*(f(scaled_t)-f(b))
        res = div_coeff*res
    elif ix_diff == 3:
        # Derivative wrt \delta
        delta = vars_[3]
        res = 2.*c+2.*alpha*div_coeff*(k*F(T_p)+F(theta)-F(b))
    return res
