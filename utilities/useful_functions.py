# License: BSD 3 clause

import string

import numpy as np
import pandas as pd
import scipy.special

# Delete ambiguous variable names 'l' and 'o' following pycodestyle E741
lower_case_varnames = (list(string.ascii_lowercase)[:11]
                       + list(string.ascii_lowercase)[12:14]
                       + list(string.ascii_lowercase)[15:])


def get_alphabet_range(n):
    """
    Get the list of the first n letters of the Roman alphabet, excluding the
    letters 'l' and 'o' following pycodestyle E741.

    Parameters
    ----------
    n : `int`
        Number of letters to generate.

    Raises
    ------
    ValueError
        If the number of letters to generate is higher than the number of
        letters in the alphabet after discarding the two letters 'l' and 'o'.

    Returns
    -------
    `list`
        List of the first n letters of the alphabet

    """
    if n > 24:
        raise ValueError("Alphabet range larger than 24.")
    else:
        return lower_case_varnames[:n]


def prod_ratio(x, y):
    """
    Given floats :math:`x, y \\in \\mathbb{R}`, return the quantity

    .. math::
        \\frac{xy}{x+y}.

    Parameters
    ----------
    x : `float`
    y : `float`

    Returns
    -------
    `float`

    """
    return (x*y)/(x+y)


def share_ratio(x, y):
    """
    Given floats :math:`x, y \\in \\mathbb{R}`, return the quantity

    .. math::
        \\frac{x}{x+y}.

    Parameters
    ----------
    x : `float`
    y : `float`

    Returns
    -------
    `float`

    """
    return (x)/(x+y)


# =============================================================================
# Triangles
# =============================================================================
def tri(t, alpha, beta, delta):
    """
    Standard triangle function.

    This function corresponds to a triangle with left corner
    :math:`\\alpha \\geq 0`, with distance to the altitude foot
    :math:`\\beta \\geq 0`, and with distance between the
    altitude foot and the right corner of the triangle :math:`\\delta \\geq 0`.
    By standard triangle, we mean that the altitude of this triangle has `y`
    coordinate `1`.

    Formally, this function is defined for all :math:`t \\in \\mathbb{R}` by

    .. math::
        f(t) := \\left(\\frac{t-\\alpha}{\\beta} \\mathbb{1}_{\\left\\{0 \\leq t-\\alpha \\leq \\beta\\right\\}}-\\frac{t-\\alpha-\\beta-\\delta}{\\delta} \\mathbb{1}_{\\left\\{0 \\leq t-\\alpha-\\beta \\leq \\delta\\right\\}}\\right).


    Parameters
    ----------
    t : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the triangle function.
    alpha : `float`
        Left corner of the triangle.
    beta : `float`
        Distance to the altitude foot.
    delta : `float`
        Distance between the altitude foot and the right corner of the triangle.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the triangle function.

    """
    return np.piecewise(t, [(t < alpha), (t >= alpha) & (t < alpha+beta),
                            (t >= alpha+beta) & (t < alpha+beta+delta),
                            (t > alpha+beta+delta)],
                        [0, lambda x:(x-alpha)/beta,
                         lambda x:-(x-alpha-beta-delta)/delta, 0])


# =============================================================================
# Gaussian density
# =============================================================================
def gaussian_pdf(t, mean=0., std=1.):
    """
    Probability density function (pdf) of a Gaussian random variable.

    | The pdf :math:`f_{\\mathcal{N}}` of a Gaussian random variable with mean :math:`\\delta` and variance :math:`\\beta^2` is defined for all :math:`t \\in \\mathbb{R}` by

    .. math::
        f_{\\mathcal{N}}(t) := \\frac{1}{\\beta\\sqrt{2\\pi}}\\exp\\bigg(-\\frac{(t-\\delta)^2}{2\\beta^2}\\bigg).


    Parameters
    ----------
    t : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the Gaussian pdf.
    mean : `float`, optional
        Mean of the Gaussian. The default is `0.`.
    std : `float`, optional
        Standard deviation of the Gaussian. The default is `1.`.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the pdf.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in numpy, before potentially moving to our own
    implementation.

    """
    return 1./(std*np.sqrt(2*np.pi))*np.exp(-0.5*((t-mean)/std)**2)


def normal_cdf(t):
    """
    Cumulative distribution function (cdf) of a normal random variable.

    | The cdf :math:`F_{\\mathcal{N}}` of a normal random variable is defined for all :math:`t \\in \\mathbb{R}` by

    .. math::
        F_{\\mathcal{N}}(t) := \\frac{1}{\\sqrt{2\\pi}}\\int_{(-\\infty, t]}\\exp\\bigg(-\\frac{u^2}{2}\\bigg) du.


    Parameters
    ----------
    t : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the cdf.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the cdf.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in scipy, before potentially moving to our own
    implementation.

    """
    return scipy.special.ndtr(t)


def gaussian_int(start, end):
    """
    Integral :math:`I` of the normal pdf on an interval :math:`[t_0, t_1]`.

    | The integral :math:`I` of a normal pdf is defined for all :math:`t_0, t_1 \\in \\mathbb{R}` by

    .. math::
        I(t_0,t_1) := \\frac{1}{\\sqrt{2\\pi}}\\int_{[t_0, t_1]}\\exp\\bigg(-\\frac{u^2}{2}\\bigg) du.


    Parameters
    ----------
    start : `float` or `numpy.ndarray`
        Lower bound or vector of lower bounds of the integration interval.
    end : `float` or `numpy.ndarray`
        Upper bound or vector of lower bounds of the integration interval.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the integral.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in scipy, before potentially moving to our own
    implementation.

    """
    return scipy.special.ndtr(end)-scipy.special.ndtr(start)


# =============================================================================
# Gamma density
# =============================================================================
def gammaf(alpha):
    """
    Gamma function.

    | The Gamma function is defined for all :math:`\\alpha \\in \\mathbb{C}` with positive real part by

    .. math::
        \\Gamma(\\alpha) := \\int_{[0, +\\infty)} t^{\\alpha-1}\\exp^{-t} dt.

    Parameters
    ----------
    alpha : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the Gamma function.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the Gamma function.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in scipy, before potentially moving to our own
    implementation.

    """
    return scipy.special.gamma(alpha)


def linc_gammaf(alpha, x):
    """
    Lower incomplete Gamma function.

    | The lower incomplete Gamma function is defined for all :math:`\\alpha \\in \\mathbb{C}` with positive real part and for all :math:`\\alpha \\in \\mathbb{C}` by

    .. math::
        \\gamma(\\alpha, x) := \\int_{[0, x]} t^{\\alpha-1}\\exp^{-t} dt.

    Parameters
    ----------
    alpha : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the function.
    x : `float` or `numpy.ndarray`
        Upper bound or vector of upper bounds at which to evaluate the integral.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the Gamma function.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in scipy, before potentially moving to our own
    implementation.

    """
    return scipy.special.gamma(alpha)*scipy.special.gammainc(alpha, x)


def std_gamma_cdf(alpha, t):
    """
    Cumulative distribution function (cdf) of a standard Gamma random variable.

    | The cdf :math:`F_{\\Gamma}` of a Gamma random variable with shape parameter :math:`\\alpha \\in \\mathbb{C}` with positive real part is defined for all :math:`t \\in \\mathbb{R}` by

    .. math::
        F_{\\Gamma}(t) := \\frac{1}{\\Gamma(\\alpha)}\\gamma(\\alpha, t).

    Parameters
    ----------
    alpha : `float` or `numpy.ndarray`
        Shape parameter or vector of shape parameters.
    t : `float` or `numpy.ndarray`
        Float or vector of floats at which to evaluate the cdf.

    Returns
    -------
    `float` or `numpy.ndarray`
        Value or vector of values taken by the cdf.

    Notes
    ------
    In the current version of the aslsd package, we use the implementation
    of this function in scipy, before potentially moving to our own
    implementation.

    """
    return scipy.special.gammainc(alpha, t)


# =============================================================================
# Exponential Sums
# =============================================================================
def update_sum_exp_1D(beta, S_n, s):
    """
    Induction step for the computation of sums of expontential decays.

    Consider a sequence :math:`(t_1 < t_2 < \\dots < t_n < \\dots)`. Let
    :math:`\\beta \\in \\mathbb{R}`. For all integers :math:`n \\geq 2`, define
    the cumulative sum

    .. math::
        S_{n} := \\sum_{k=1}^{n-1} \\exp\\Big(-\\beta(t_n-t_k)\\Big).

    It is clear that for all :math:`n`, we have the induction formula

    .. math::
        S_{n+1} = \\exp\\Big(-\\beta(t_{n+1}-t_n)\\Big)(1+S_n).

    This function performs this induction step by returning the quantity

    .. math::
        \\exp\\big(-\\beta s\\big)(1+S_n).

    Parameters
    ----------
    beta : `float`
        Decay rate.
    S_n : `float`
        Previous value of the cumulative sum.
    s : `float`
        Inter-arrival time.

    Returns
    -------
    `float`
        Updated value of the cumulative sum.

    """
    return np.exp(-beta*s)*(1+S_n)


def double_sum_exp_1D(beta, times):
    """
    Inductive computation of double sums of exponential decays.

    Consider a sequence :math:`(t_1 < t_2 < \\dots < t_N)`. Let
    :math:`\\beta \\in \\mathbb{R}`. For all integers :math:`m \\geq 2`, define
    the cumulative sum

    .. math::
        S_{m} := \\sum_{k=1}^{m-1} \\exp\\Big(-\\beta(t_m-t_k)\\Big).

    For all integers :math:`N \\geq 2`, define the double sum

    .. math::
        Z_{N} := \\sum_{m=2}^{N} S_{m}.

    This function computes :math:`Z_n` in linear time using the induction formula

    .. math::
        S_{m+1} = \\exp\\Big(-\\beta(t_{m+1}-t_m)\\Big)(1+S_m).

    Parameters
    ----------
    beta : `float`
        Decay rate.
    times : `numpy.ndarray`
        Array of times.

    Returns
    -------
    res : `float`
        Double sum of exponential decays.

    """
    res = 0.
    u_m = 0.
    for m in range(1, len(times)):
        u_m = np.exp(-beta*(times[m]-times[m-1]))*(1.+u_m)
        res += u_m
    return res


def double_sum_exp_1D_midpoint(beta, times, T_f):
    """
    Inductive computation of double sums of exponential decays with midpoint dependence.

    Consider a sequence :math:`(t_1 < t_2 < \\dots < t_N < T_f)`. Let
    :math:`\\beta \\in \\mathbb{R}`. For all integers :math:`m \\geq 2`, define
    the cumulative sum

    .. math::
        S_{m} := \\sum_{n=1}^{m-1}\\exp\\Big(-2\\beta\\big(T_f-(t_m +t_n)/2\\big)\\Big).

    For all integers :math:`N \\geq 2`, define the double sum

    .. math::
        Z_{N} := \\sum_{m=2}^{N}S_m.

    This function computes :math:`Z_n` in linear time using the induction formula

    .. math::
        S_{m+1} = \\exp\\Big(-2\\beta\\big(T_f-(t_{m+1} +t_m)/2\\big)\\Big)+\\exp\\Big(\\beta(t_{m+1}-t_m)\\Big)S_m.    

    Parameters
    ----------
    beta : `float`
        Decay rate.
    times : `numpy.ndarray`
        Array of times.
    T_f : `float`
        Terminal time.

    Returns
    -------
    res : `float`
        Double sum of exponential decays.

    """
    res = 0.
    u_m = 0.
    for m in range(1, len(times)):
        u_m = (np.exp(beta*(times[m]-times[m-1]))*u_m
               + np.exp(-2*beta*(T_f-(times[m]+times[m-1])/2.)))
        res += u_m
    return res


# =============================================================================
# List operations
# =============================================================================
def is_array(L):
    return isinstance(L, (list, pd.core.series.Series, np.ndarray))

def is_empty(L):
    if L is None:
        return True
    elif is_array(L) or type(L) == dict:
        return len(L) == 0


def is_sorted(x, reverse=False, strict=False):
    """Check if x is sorted"""
    if not reverse:
        if strict:
            return (np.diff(x) > 0).all()
        else:
            return (np.diff(x) >= 0).all()
    else:
        if strict:
            return (np.diff(x) < 0).all()
        else:
            return (np.diff(x) <= 0).all()


def discretize_space(x_min, x_max, res, disc_type):
    """
    Return evenly spaced values in an interval or on a log scale.

    Let :math:`x_{\\textrm{min}},x_{\\textrm{max}} \\in \\mathbb{R}`. If a
    linear discretization is chosen, return `res` evenly spaced numbers in
    :math:`[x_{\\textrm{min}},x_{\\textrm{max}}]`. If a logarithmic
    discretization is chosen, return `res` evenly spaced numbers in
    :math:`[10^{x_{\\textrm{min}}},10^{x_{\\textrm{max}}}]`.

    Parameters
    ----------
    x_min : `float`
        Start value or logarithm of the start value.
    x_max : `float`
        End value or logarithm of the end value.
    res : `int`
        Number of elements to return.
    disc_type : `str`
        Discretization type.

    Returns
    -------
    `numpy.ndarray`
        Discretized set.

    """
    if disc_type == 'log':
        return np.logspace(x_min, x_max, res)
    elif disc_type == 'linear':
        return np.linspace(x_min, x_max, res)


def partition_space(l_bound, r_bound, n_strata, disc_type):
    disc = discretize_space(l_bound, r_bound, n_strata+1, disc_type)
    disc = disc.astype(int)
    strata = [None]*n_strata
    strata[0] = [l_bound, disc[1]-1]
    strata[-1] = [disc[-2], r_bound]
    for ix in range(1, len(disc)-2):
        strata[ix] = [disc[ix], disc[ix+1]-1]
    return strata


def concatenate_linspace(vec_T, vec_n_res):
    n_total = sum(vec_n_res)-(len(vec_n_res)-1)
    X = np.zeros(n_total)
    Q = np.linspace(0, vec_T[0], vec_n_res[0])
    X[:vec_n_res[0]] = Q
    start_index = vec_n_res[0]
    for i in range(1, len(vec_n_res)):
        W = np.linspace(0, vec_T[i], vec_n_res[i])
        X[start_index:start_index+vec_n_res[i]-1] = X[start_index-1]+W[1:]
        start_index += vec_n_res[i]-1
    return X


def get_ix_positions(L, Q):
    return [np.where(Q == x)[0][0] for x in L]


def get_grid_position(x, ref_x):
    # Get the ixs such that ref_x[ix-1] <= x < ref_x[ix]
    ixs_grid = np.searchsorted(ref_x, x, side='right')
    return ixs_grid
# =============================================================================
# Dictionary operations
# =============================================================================
def dict_keys2str(d):
    s = ''
    L = [key for key in d.keys()]
    for ix in range(len(L)-1):
        s += L[ix]
        s += ', '
    s += L[-1]+'.'
    return s


# =============================================================================
# Derivatives
# =============================================================================
def finite_diff(func, x, epsilon=10**-3, diff_type='central difference',
                diff_index=0):
    """
    Finite differences estimate of the derivative of `func` at `x`.

    Let :math:`f` be a function that is differentiable at :math:`x \\in \\mathbb{R}`.
    Let :math:`\\epsilon >0`. Define finite differences estimates of :math:`f^\\prime(x)`.

    The central difference estimate is define as

    .. math::
        \\frac{f(x+\\epsilon /2)-f(x-\\epsilon /2)}{\\epsilon}.

    The forward difference estimate is define as

    .. math::
        \\frac{f(x+\\epsilon)-f(x)}{\\epsilon}.

    The backward difference estimate is define as

    .. math::
        \\frac{f(x)-f(x-\\epsilon )}{\\epsilon}.

    Parameters
    ----------
    func : `function`
        Function to differentiate.
    x : `float` or `numpy.ndarray`
        Value at which to estimate the derivative.
    epsilon : `float`, optional
        Step parameter of the approximation. The default is 10**-3.
    diff_type : `str`, optional
        Type of finite differences estimator. The default is 'central difference'.
    diff_index : `int`, optional
        Index of the derivation variable. The default is 0.

    Returns
    -------
    `float`
        Finite differences estimate of the derivative.

    """
    if (type(x) == np.ndarray) and (x.size >= 1):
        delta = np.zeros(x.size)
        delta[diff_index] = epsilon
    else:
        delta = epsilon

    if diff_type == 'central difference':
        return (func(x+delta/2.)-func(x-delta/2.))/epsilon
    elif diff_type == 'forward difference':
        return (func(x+delta)-func(x))/epsilon
    if diff_type == 'backward difference':
        return (func(x)-func(x-delta))/epsilon


def get_diff_paramfunc(param_func, n_vars, epsilon=10**-3,
                       diff_type='central difference'):
    def diff_func(t, ix_diff, vars_):
        def func(y):
            return param_func(t, y)
        return finite_diff(func, vars_, epsilon=10**-3,
                           diff_type='central difference',
                           diff_index=ix_diff)
    return diff_func
