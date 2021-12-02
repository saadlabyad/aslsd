# License: BSD 3 clause

import numpy as np
import scipy.special


def prod_ratio(x, y):
    return (x*y)/(x+y)


def share_ratio(x, y):
    return (x)/(x+y)


# =============================================================================
# Triangles
# =============================================================================
def tri(t, alpha, beta, delta):
    return np.piecewise(t, [(t < alpha), (t >= alpha) & (t < alpha+beta),
                            (t >= alpha+beta) & (t < alpha+beta+delta),
                            (t > alpha+beta+delta)],
                        [0, lambda x:(x-alpha)/beta,
                         lambda x:-(x-alpha-beta-delta)/delta, 0])


# =============================================================================
# Gaussian density
# =============================================================================
def gaussian_pdf(x, mean=0., std=1.):
    return 1./(std*np.sqrt(2*np.pi))*np.exp(-0.5*((x-mean)/std)**2)


def normal_cdf(x):
    # Normal CDF evaluated in x
    # i.e. probability of a N(0,1) takes a value in [-\infty,x]
    #   scipy.special.ndtr is faster than norm.cdf
    return scipy.special.ndtr(x)


def gaussian_int(start, end):
    # Integral of the normal gaussian on [start,end]
    # i.e. probability of a N(0,1) takes a value in [start,end]
    return scipy.special.ndtr(end)-scipy.special.ndtr(start)


# =============================================================================
# Gamma density
# =============================================================================
def gammaf(alpha):
    #   Gamma function
    return scipy.special.gamma(alpha)


def linc_gammaf(alpha, x):
    #   Lower incomplete Gamma function
    return scipy.special.gamma(alpha)*scipy.special.gammainc(alpha, x)


def std_gamma_cdf(alpha, x):
    #   CDF of a standard Gamma RV
    return scipy.special.gammainc(alpha, x)


# =============================================================================
# Exponential Sums
# =============================================================================
def update_sum_exp_1D(beta, E_n, s):
    return np.exp(-beta*s)*(1+E_n)


def double_sum_exp_1D(beta, times):
    # Compute inductively
    # \sum_{m=2}^{N} \sum_{n=1}^{m-1} \exp ( -\beta (t_m -t_n) )
    res = 0.
    u_m = 0.
    for m in range(1, len(times)):
        u_m = np.exp(-beta*(times[m]-times[m-1]))*(1.+u_m)
        res += u_m
    return res


def double_sum_exp_1D_midpoint(beta, times, T_f):
    # Compute inductively
    # \sum_{m=2}^{N} \sum_{n=1}^{m-1} \exp ( -2\beta (T_f - (t_m +t_n)/2 ) )
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
def make_uniform_partition(n_buckets, index_min, index_max, Q_alloc,
                           alloc_type='Uniform'):
    list_m_p = [None]*n_buckets
    avg_step = int((index_max-index_min+1)/n_buckets)
    m_0 = index_min
    for p in range(n_buckets):
        m_1 = min(m_0+avg_step-1, index_max)
        list_m_p[p] = [m_0, m_1]
        m_0 = m_1+1
    list_m_p[-1][1] = index_max
    if alloc_type == 'Uniform':
        list_B_p = [int(Q_alloc/n_buckets) for p in range(n_buckets)]
    elif alloc_type == 'Proportional':
        list_B_p = [int(Q_alloc/n_buckets) for p in range(n_buckets)]
    return list_B_p, list_m_p


def make_semi_uniform_partition(n_buckets, index_min, index_max, Q_alloc,
                                std_offsets=[30, 50, 100, 200, 300, 10**3],
                                alloc_type='Uniform', strata_sizes=[]):
    if n_buckets == 0:
        return [], []
    else:
        list_m_p = [None]*n_buckets
        m_0 = index_min
        for p in range(len(std_offsets)):
            m_1 = m_0+std_offsets[p]
            list_m_p[p] = [m_0, m_1]
            m_0 = m_1+1
        avg_step = int((index_max-m_0+1)/(n_buckets-len(std_offsets)))
        for p in range(len(std_offsets), n_buckets):
            m_1 = min(m_0+avg_step-1, index_max)
            list_m_p[p] = [m_0, m_1]
            m_0 = m_1+1
        list_m_p[-1][1] = index_max
        if alloc_type == 'Uniform':
            list_B_p = [int(Q_alloc/n_buckets) for p in range(n_buckets)]
        elif alloc_type == 'Proportional':
            list_B_p = [int(Q_alloc/n_buckets) for p in range(n_buckets)]
        return list_B_p, list_m_p


def discretize_space(x_min, x_max, res, disc_type):
    if disc_type == 'log':
        return np.logspace(x_min, x_max, res)
    elif disc_type == 'linear':
        return np.linspace(x_min, x_max, res)


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


def get_index_in_basis(list_r, basisfunc_index):
    #   list_r is a list of r values in a mixed model
    #   basisfunc_index is the index of a basis function in the model, where all basis functions
    #   are enumerated continuously
    #   This function outputs the corresponding submodel for basisfunc_index, as well as the index of the basis function among functions of that model
    mixed_index = 0
    running_sum = list_r[0]
    old_sum = 0
    while basisfunc_index >= running_sum:
        old_sum = running_sum
        mixed_index += 1
        running_sum += list_r[mixed_index]
    index_in_basis = basisfunc_index-old_sum
    return mixed_index, index_in_basis

# list_r=[4]
# basisfunc_index=3
# mixed_index, index_in_basis=get_index_in_basis(list_r,basisfunc_index)
# print('The function of basis index ',basisfunc_index,' belongs to the group ',mixed_index,'and has index ',index_in_basis,' in this group')


# =============================================================================
# Learning rate decays
# =============================================================================
def func_rate_decay_exp(t, learning_rate, divider=2, period=200):
    q = t//period
    return learning_rate/float(divider**q)

#   Test
# X=np.linspace(0,1000,10*5)
# Y=[func_rate_decay_exp(t, learning_rate=200) for t in X]

# import matplotlib.pyplot as plt
# plt.plot(X,Y)


def func_rate_decay_lin(t, learning_rate, divider=1.):
    return learning_rate/float(divider+t)

#   Test
# X=np.linspace(0,1000,10*5)
# Y=[func_rate_decay_lin(t,learning_rate=10**-1,divider=1.) for t in X]

# import matplotlib.pyplot as plt
# plt.plot(X,Y)


# =============================================================================
# Derivatives
# =============================================================================
def finite_diff(func, x, epsilon=10**-3, diff_type='central difference',
                diff_index=0):
    """
    Finite differences estimate of the derivative of func in x.

    Parameters
    ----------
    func : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 10**-3.
    diff_type : TYPE, optional
        DESCRIPTION. The default is 'central difference'.
    diff_index : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    TYPE
        DESCRIPTION.

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


#   Test
# def func(t):
#     return np.exp(-t**2)*np.cos(-2.*t)

# def true_der_func(t):
#     return np.exp(-t**2)*(-2.*t*np.cos(-2.*t)+2*np.sin(-2.*t))

# delta=10**-10


# X=np.linspace(0,10,1000)
# Y_true=[true_der_func(x) for x in X]
# Y_finite_diff=[finite_diff(func,x,delta=delta) for x in X]

# import matplotlib.pyplot as plt
# fig=plt.figure(dpi=300)
# plt.plot(X,Y_finite_diff,label='Finite diff',color='steelblue')
# plt.plot(X,Y_true,label='True',linestyle='dashed',color='darkorange')
# plt.legend()
# fig.show()
