# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# =============================================================================
# Tests and plots
# =============================================================================
def ks_test_residuals(residuals):
    return [stats.kstest(res, 'expon') for res in residuals]


def qq_plot(residuals, n_models=1, labels=None, style='exponential',
            substract_yx=False, normalize=False, max_points=None, margin=0.01,
            display_line45=True, log_scale=False, ax=None,
            save=False, filename='image.png', show=False, **kwargs):
    #   Draw Q-Q plot of the residuals of each model.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    # if labels is not None:
    #     labelling = False
    # else:
    #     labels = [' ']*n_models
    #     labelling = True

    if style == 'exponential':
        (osm, osr) = stats.probplot(residuals, dist="expon",
                                    plot=None, fit=False)
        renorm_factor = np.sqrt(len(osr))
        if max_points is not None:
            max_points = min(max_points, len(osm))
            
            # indices_subsample = np.concatenate(([m for m in range(int(margin*len(osm)))], np.linspace(int(margin*len(osm)), int((1-margin)*len(osm)), max_points), [m for m in range(int((1-margin)*len(osm)), len(osm))]))
            indices_subsample = np.concatenate(([m for m in range(int(margin*len(osm)))], np.linspace(int(margin*len(osm)), int((1-margin)*len(osm)), max_points), [m for m in range(int((1-margin)*len(osm)), len(osm))]))
            indices_subsample = [int(x) for x in indices_subsample]
            osm = osm[indices_subsample]
            osr = osr[indices_subsample]
        if substract_yx:
            ax.plot(osm, osr-osm, marker="o", linestyle="None", **kwargs)
        else:
            ax.plot(osm, osr, marker="o", linestyle="None", **kwargs)
            x_45 = np.linspace(*ax.get_xlim())
            ax.plot(x_45, x_45, linestyle='dashed', color='grey')
    elif style == 'uniform':
        (osm, osr) = stats.probplot(1-np.exp(-residuals),
                                    dist=stats.uniform, plot=None,
                                    fit=False)
        renorm_factor = np.sqrt(len(osr))
        if max_points is not None:
            max_points = min(max_points, len(osm))
            margin = 0.01
            indices_subsample = np.concatenate(([m for m in range(int(margin*len(osm)))], np.linspace(int(margin*len(osm)), int((1-margin)*len(osm)), max_points), [m for m in range(int((1-margin)*len(osm)), len(osm))]))
            indices_subsample = [int(x) for x in indices_subsample]
            osm = osm[indices_subsample]
            osr = osr[indices_subsample]
        if substract_yx:
            if normalize:
                ax.plot(osm, renorm_factor*(osr-osm), **kwargs)
            else:
                ax.plot(osm, osr-osm, **kwargs)
            ax.axhline(y=0., linestyle='dashed', color='grey')
        else:
            ax.plot(osm, osr, **kwargs)
    # if labelling:
    #     ax.legend()
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if save:
        plt.savefig(filename)
    if show:
        plt.show()
    return ax


# =============================================================================
# Runs test
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
    n_prod = n_pos*n_neg
    r_mean = 1.+2.*n_prod/n
    r_var = 2.*(n_prod/n**2)*((2.*n_prod-n)/(n-1))
    r_std = np.sqrt(r_var)
    z = (n_runs-r_mean)/r_std
    return z


def compute_wald_wolfowitz_pvalue(z):
    pvalue = 2.*stats.norm.sf(np.abs(z))
    return pvalue


def wald_wolfowitz_runstest(L):
    runs = get_runs(L)
    n_runs, n_pos, n_neg = get_counts(runs)
    z = compute_wald_wolfowitz_stat(n_runs, n_pos, n_neg)
    pvalue = compute_wald_wolfowitz_pvalue(z)
    res = {'pvalue': pvalue, 'statistic': z}
    return res

# =============================================================================
# Homogeneous Poisson Process
# =============================================================================
def get_residuals_k_hompoisson(k, process_path, mu):
    # Unwrap data
    list_times = process_path.list_times
    mu_k = mu[k]
    # Compute
    residuals_k = mu_k*(list_times[k][1:]-list_times[k][:-1])
    return residuals_k


def get_residuals_hompoisson(process_path, mu):
    residuals = [get_residuals_k_hompoisson(k, process_path, mu)
                 for k in range(process_path.d)]
    return residuals


# =============================================================================
# Non-homogeneous Poisson Process
# =============================================================================
def get_residuals_k_nonhompoisson(k, process_path, mu_primitive, mu_param):
    # Compute
    times_comp = mu_primitive[k](process_path.list_times[k], mu_param[k])
    residuals_k = times_comp[1:]-times_comp[:-1]
    return residuals_k


def get_residuals_nonhompoisson(process_path, mu_primitive, mu_param):
    residuals = [get_residuals_k_nonhompoisson(k, process_path, mu_primitive,
                                               mu_param)
                 for k in range(process_path.d)]
    return residuals


# =============================================================================
# MHP
# =============================================================================
def get_residuals_k_mhp(k, process_path, psi, mu, kernel_param,
                        cutoff=False, cutoff_ixlag=200,
                        sampling=False, sample_size=10**3, rng=None, seed=1234,
                        verbose=False):
    # Unwrap data
    d = process_path.d
    list_times = process_path.list_times
    kappa = process_path.kappa
    n_events = process_path.n_events
    N_k = n_events[k]
    varpi_ki1 = np.array([process_path.varpi[k][i][1] for i in range(d)],
                         dtype=int)

    # Prepare output
    kernel_part = np.zeros((d, N_k))  # Kernel part of compensator times

    # I. Computations for Poisson part
    poisson_residuals = get_residuals_k_hompoisson(k, process_path, mu)

    # II. Computations for Kernel part
    for i in range(d):
        # II.1. Compute transofrmed time at varpi_ki1 exactly
        if varpi_ki1[i] > 0:
            # If \varpi_{ki,1} > 0, then all residuals r_m with
            # m < \varpi_{ki,1}-1 do not have a kernel contribution part
            # (because all such t^k_m and t^k_{m+1} have no preceding
            # event of type i).
            # For m = \varpi_{ki,1}-1, t^k_m has no predecessor of type i,
            # but t^k_{m+1} does: so the kernel contribution term is non-null.
            t_m = list_times[k][varpi_ki1[i]]
            kernel_part[i][varpi_ki1[i]] = np.sum(
                psi[k][i](t_m-list_times[i][:kappa[i][k][varpi_ki1[i]]+1],
                          kernel_param[k][i]))
        # II.2. Compute residuals for all m in [varpi_ki1, N_k-1],
        # that is for all ixs m such that both t^k_{m+1} and t^k_{m}
        # have antecedents of type i
        for m in tqdm(range(varpi_ki1[i]+1, N_k), disable=not verbose):
            t_m = list_times[k][m]
            kappa_old = kappa[i][k][m-1]
            kappa_new = kappa[i][k][m]
            # Compute exactly the contribution of new events in the interval
            # t^k_{m-1}, t^k_m
            new_events = list_times[i][kappa_old+1:kappa_new+1]
            if len(new_events) > 0:
                contrib_new = np.sum(psi[k][i](t_m-new_events,
                                               kernel_param[k][i]))
            else:
                contrib_new = 0.
            # Compute the contribution of old events
            contrib_old = np.sum(psi[k][i](t_m-list_times[i][:kappa_old+1],
                                           kernel_param[k][i]))
            # Merge results
            kernel_part[i][m] = contrib_new+contrib_old

    # Group results
    kernel_residuals = (kernel_part[:, 1:]-kernel_part[:, :-1]).sum(axis=0)
    residuals_k = poisson_residuals+kernel_residuals
    return residuals_k


def get_residuals_mhp(process_path, psi, mu, kernel_param, cutoff=False,
                      cutoff_ixlag=200, sampling=False, sample_size=10**3,
                      rng=None, seed=1234, verbose=False):
    d = process_path.d
    return [get_residuals_k_mhp(k, process_path, psi, mu, kernel_param,
                                cutoff=cutoff, cutoff_ixlag=cutoff_ixlag,
                                sampling=sampling, sample_size=sample_size,
                                rng=rng, seed=seed, verbose=verbose)
            for k in range(d)]


# =============================================================================
# MTLH
# =============================================================================
def get_residuals_k_mtlh(k, process_path, mu_primitive, psi,
                         expected_impact_matrix,
                         mu_param, kernel_param, impact_param,
                         cutoff=False, cutoff_ixlag=200,
                         sampling=False, sample_size=10**3, rng=None, seed=1234,
                         verbose=False):
    # Unwrap data
    d = process_path.d
    list_times = process_path.list_times
    kappa = process_path.kappa
    N_k = process_path.n_events[k]
    varpi_ki1 = np.array([process_path.varpi[k][i][1] for i in range(d)],
                         dtype=int)

    # Prepare output
    kernel_part = np.zeros((d, N_k))  # Kernel part of compensator times

    # I. Computations for Poisson part
    poisson_residuals = get_residuals_k_nonhompoisson(k, process_path,
                                                      mu_primitive, mu_param)

    # II. Computations for Kernel part
    for i in range(d):
        e_ki = expected_impact_matrix[k][i]
        # II.1. Compute transofrmed time at varpi_ki1 exactly
        t_m = list_times[k][varpi_ki1[i]]
        kernel_part[i][varpi_ki1[i]] = np.sum(
            psi[k][i](t_m-list_times[i][:kappa[i][k][varpi_ki1[i]]+1],
                      kernel_param[k][i]))*e_ki
        # II.2. Compute residuals for all m in [varpi_ki1, N_k-1],
        # that is for all ixs m such that both t^k_{m+1} and t^k_{m}
        # have antecedents of type i
        for m in tqdm(range(varpi_ki1[i]+1, N_k), disable=not verbose):
            t_m = list_times[k][m]
            kappa_old = kappa[i][k][m-1]
            kappa_new = kappa[i][k][m]
            # Compute exactly the contribution of new events in the interval
            # t^k_{m-1}, t^k_m
            new_events = list_times[i][kappa_old+1:kappa_new+1]
            if len(new_events) > 0:
                contrib_new = np.sum(psi[k][i](t_m-new_events,
                                               kernel_param[k][i]))
            else:
                contrib_new = 0.
            # Compute the contribution of old events
            contrib_old = np.sum(psi[k][i](t_m-list_times[i][:kappa_old+1],
                                           kernel_param[k][i]))
            # Merge results
            kernel_part[i][m] = (contrib_new+contrib_old)*e_ki

    # Group results
    # ker_diff = kernel_part[:, 1:]-kernel_part[:, :-1]
    kernel_residuals = (kernel_part[:, 1:]-kernel_part[:, :-1]).sum(axis=0)
    residuals_k = poisson_residuals+kernel_residuals
    return residuals_k


def get_residuals_mtlh(process_path, mu_primitive, psi, expected_impact_matrix,
                       mu_param, kernel_param, impact_param, cutoff=False,
                       cutoff_ixlag=200, sampling=False, sample_size=10**3,
                       rng=None, seed=1234, verbose=False):
    d = process_path.d
    residuals = [None]*d
    for k in range(d):
        residuals[k] = get_residuals_k_mtlh(k, process_path, mu_primitive, psi,
                                            expected_impact_matrix, mu_param,
                                            kernel_param, impact_param,
                                            cutoff=cutoff,
                                            cutoff_ixlag=cutoff_ixlag,
                                            sampling=sampling,
                                            sample_size=sample_size,
                                            rng=rng, seed=seed,
                                            verbose=verbose)
    return residuals
