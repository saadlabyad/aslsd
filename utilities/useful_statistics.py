# License: BSD 3 clause

from collections import Counter
from random import random

import numpy as np
from numpy.random import default_rng
import scipy


# =============================================================================
# RNG
# =============================================================================
def make_rng(rng=None, seed=1234):
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng


# =============================================================================
# Sampling mixture weights
# =============================================================================
def get_random_mixture_weights(n, coeff=1., rng=None, seed=1234):
    # Get random mixture weights for one function with n basis functions
    rng = make_rng(rng=rng, seed=seed)
    weights = rng.uniform(size=n)
    # Rescale as probabilities
    weights /= np.sum(weights)
    weights *= coeff
    # Enforce weights to sum to 1 in case of numerical error
    weights[-1] = coeff-np.sum(weights[:-1])
    return weights


# =============================================================================
# Sampling single indices
# =============================================================================
def sample_single_ix_stratum(m_inf, m_sup, n_samples, rng=None, seed=1234):
    # Sample without replacement n_samples indices in the interval [m_inf,m_sup]
    if rng is None:
        rng = default_rng(seed)

    if m_inf > m_sup:
        raise ValueError("m_inf should be smaller than m_sup")
    else:
        return m_inf+rng.choice(m_sup-m_inf+1, size=n_samples, replace=False)


def sample_single_ix_stratified(strata, n_samples, rng=None, seed=None):
    if rng is None:
        rng = default_rng(seed)
    n_strata = len(strata)
    samples = [None]*n_strata
    for ix_strata in range(n_strata):
        m_inf = strata[ix_strata][0]
        m_sup = strata[ix_strata][1]
        samples[ix_strata] = sample_single_ix_stratum(m_inf, m_sup,
                                                      n_samples[ix_strata],
                                                      rng=rng)
    return samples


def stratified_single_ixs2times(i, list_times2end, samples):
    n_strata = len(samples)
    timediff = [list_times2end[i][samples[ix_strata]]
                for ix_strata in range(n_strata)]
    return timediff


# =============================================================================
# Sampling double indices
# =============================================================================
def sample_double_ix_stratum(i, j, h_inf, h_sup, n_samples, kappa, varpi,
                             n_events,
                             rng=None, seed=1234):
    # Sample without replacement n_samples pairs of indices
    # with lags in the interval [h_inf,h_sup]
    if rng is None:
        rng = default_rng(seed)
    if h_inf == h_sup:
        h = h_inf
        varpi_ijh = varpi[i][j][h]
        list_m = varpi_ijh+rng.choice(n_events[i]-varpi_ijh,
                                      n_samples, replace=False)
        list_n = kappa[j][i][list_m]-(h-1)
        return {'m': list_m, 'n': list_n}
    else:
        list_h_vals = h_inf+rng.choice(h_sup+1-h_inf, size=n_samples)
        counted_h_vals = Counter(list_h_vals)
        # Sample m,n for each h
        samples_m = [[] for k in range(len(counted_h_vals))]
        samples_n = [[] for k in range(len(counted_h_vals))]
        k = 0
        for h in counted_h_vals.keys():
            n_samples_h = counted_h_vals[h]
            varpi_ijh = varpi[i][j][h]
            n_h = n_events[i]-varpi_ijh
            if n_samples_h <= n_h:
                list_m = varpi_ijh+rng.choice(n_h, n_samples_h, replace=False)
            else:
                list_m = varpi_ijh+rng.choice(n_h, n_samples_h, replace=True)
            samples_m[k] = list_m
            list_n = kappa[j][i][list_m]-h+1
            samples_n[k] = list_n
            k += 1
        samples_m = np.concatenate(samples_m)
        samples_n = np.concatenate(samples_n)
        samples = {'m': samples_m, 'n': samples_n}
        return samples


def sample_double_ix_stratified(i, j, strata, n_samples, kappa, varpi,
                                n_events, rng=None, seed=1234):
    if rng is None:
        rng = default_rng(seed)
    n_strata = len(strata)
    samples = [None]*n_strata
    for ix_strata in range(n_strata):
        h_inf = strata[ix_strata][0]
        h_sup = strata[ix_strata][1]
        stratum_samples = sample_double_ix_stratum(i, j, h_inf, h_sup,
                                                   n_samples[ix_strata], kappa,
                                                   varpi, n_events, rng=rng)
        samples[ix_strata] = stratum_samples
    return samples


def stratified_double_ixs2phi_times(i, j, list_times, samples):
    n_strata = len(samples)
    timediff = [list_times[i][samples[ix_strata]['m']]
                - list_times[j][samples[ix_strata]['n']]
                for ix_strata in range(n_strata)]
    return timediff


def stratified_double_ixs2upsilon_times(i, j, list_times, list_times2end,
                                        samples):
    n_strata = len(samples)
    timediff = [{'tf': list_times2end[i][samples[ix_strata]['m']],
                 'mn': list_times[i][samples[ix_strata]['m']]
                 - list_times[j][samples[ix_strata]['n']]}
                for ix_strata in range(n_strata)]
    return timediff


# =============================================================================
# Compute stats
# =============================================================================
def estimate_mean(q, list_samples):
    return np.sum(list_samples)/float(q)


def estimate_std_wor_unbiased(q, list_samples, mean, N):
    if q == 1:
        return 0.
    else:
        finite_pop_corr = float(N-1.)/float(N)
        sum_sq = np.sum(np.square(list_samples))
        return finite_pop_corr*((1./(q-1.))*sum_sq-(q/(q-1.))*mean**2)


# =============================================================================
# Batch updates
# =============================================================================
def batch_mean(mean_1, q_1, mean_2, q_2):
    # Given an unbiased estimate of the mean wor of sample size q_1 > 1
    # And one of sample size q_2 > 1
    # Output one of size q_1+q_2
    ratio_1 = float(q_1)/float(q_1+q_2)
    return ratio_1*mean_1+(1.-ratio_1)*mean_2


def batch_std(std_1, q_1, std_2, q_2, N, mean_1, mean_2,
              replacement=False, style='unbiased'):
    # Given an unbiased estimate of the std wor of sample size q_1 > 1
    # And one of sample size q_2 > 1
    # Output one of size q_1+q_2
    # N is the size of the set
    if replacement:
        finite_pop = 1.
    else:
        finite_pop = (N-1.)/N
    if style == 'unbiased':
        ratio_1 = (q_1-1.)/(q_1+q_2-1.)
        std = (ratio_1*std_1**2+(1.-ratio_1)*std_2**2
               + (finite_pop
                  * ((std_1*std_2)/((q_1+q_2-1.)*(q_1+q_2)))
                  * (mean_1-mean_2)**2))
    return np.sqrt(std)


# =============================================================================
# KS Test
# =============================================================================
def get_ks_alpha(c, N_r):
    S = 0.
    coeff = -2*c**2
    for r in range(1, N_r+1):
        u = (-1)**(r-1)*np.exp(coeff*r**2)
        S += u
    return 2*S


# =============================================================================
# Wald-Wolofwitz test
# =============================================================================
def get_runs(L):
    n = len(L)
    runs = np.zeros(n, dtype=int)
    median = np.median(L)
    ixs_pos = np.where(L > median)[0]
    runs[ixs_pos] = 1
    return runs


def get_counts(runs):
    n = len(runs)
    n_pos = np.sum(runs)
    n_neg = n-n_pos
    diff = np.abs(runs[1:]-runs[:-1])
    n_runs = 1+np.sum(diff)
    return n_runs, n_pos, n_neg


def compute_wald_wolfowitz_stat(n_runs, n_pos, n_neg):
    n = n_pos+n_neg
    # n_prod = n_pos*n_neg
    # r_mean = 1.+2.*n_prod/n
    # r_var = 2.*(n_prod/n**2)*((2.*n_prod-n)/(n-1))
    root_n = np.sqrt(n)
    n_prod_div_n = (n_pos/root_n)*(n_neg/root_n)
    r_mean = 1.+2.*n_prod_div_n
    r_var = 2.*n_prod_div_n*((2.*n_prod_div_n-1.)/(n-1))
    r_std = np.sqrt(r_var)
    z = (n_runs-r_mean)/r_std
    return z


def compute_wald_wolfowitz_pvalue(z):
    pvalue = 2.*scipy.stats.norm.sf(np.abs(z))
    return pvalue


def wald_wolfowitz_runstest(L):
    runs = get_runs(L)
    n_runs, n_pos, n_neg = get_counts(runs)
    z = compute_wald_wolfowitz_stat(n_runs, n_pos, n_neg)
    pvalue = compute_wald_wolfowitz_pvalue(z)
    res = {'pvalue': pvalue, 'statistic': z}
    return res
