# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
from tqdm import tqdm


from scipy import stats
import statsmodels.api as sm


#   Homogeneous Poisson Process
def get_residuals_k_hpoisson(d, k, mu_k, list_times):
    residuals_k = mu_k*(list_times[k][1:]-list_times[k][:-1])
    return residuals_k


def get_residuals_hpoisson(d, mu, list_times):
    return [get_residuals_k_hpoisson(d, k, mu[k], list_times)
            for k in range(d)]


#   MHP
def get_residuals_k(k, process_path, psi, mu, kernel_param,
                    sampling=False, sample_size=10**3, seed=1234,
                    verbose=False):
    d = process_path.d
    list_times = process_path.list_times
    kappa = process_path.kappa
    varpi = process_path.varpi
    n_events = process_path.n_events
    N_k = n_events[k]

    mu_k = mu[k]

    if sampling:
        sample_size = max(min(sample_size, N_k-2), 1)
        residuals_k = np.zeros(sample_size)
        rng = np.random.default_rng(seed)
        res_indices = rng.choice(N_k-1, size=sample_size, replace=False,
                                 p=None)
        res_indices = list(res_indices)
    else:
        residuals_k = np.zeros(N_k-1)
        res_indices = [ind for ind in range(N_k-1)]

    active_ind = 0
    for m_index in tqdm(range(len(res_indices)), disable=not verbose):
        m = res_indices[m_index]
        t_m = list_times[k][m]
        t_m_1 = list_times[k][m+1]
        residuals_k[active_ind] = mu_k*(t_m_1-t_m)
        for i in range(d):
            # ind_prev is the index of the last event of type i happening
            # strictly before [t^k_m,t^k_{m+1}] (if any)

            # ind_first is the index of the first event of type i happening
            # in [t^k_m,t^k_{m+1}] (if any)

            # ind_last is the index of the last event of type i happening
            # in [t^k_m,t^k_{m+1}] (if any)
            if k == i:
                ind_prev = m-1
                ind_first = m
                ind_last = m
            else:
                ind_prev = kappa[i][k][m]
                if m <= kappa[k][i][-2]+1:
                    ind_first = varpi[i][k][m]
                    ind_last = kappa[i][k][m+1]
                else:
                    ind_first = -1

            if ind_first > -1:
                # Events of type i happening in [t^k_m,t^k_{m+1}]
                t_ind_first = list_times[i][ind_first]
                if t_ind_first < t_m_1:
                    loc_times_1 = t_m_1-list_times[i][np.arange(ind_first,
                                                                ind_last+1)]
                    residuals_k[active_ind] += np.sum(psi[k][i](loc_times_1,
                                                                kernel_param[k][i]))

            # Events of type i happening strictly before [t^k_m,t^k_{m+1}]
            if ind_prev >= 0:
                loc_times_2 = list_times[i][np.arange(ind_prev+1)]
                residuals_k[active_ind] += (np.sum(psi[k][i](t_m_1-loc_times_2,
                                                             kernel_param[k][i]))
                                            - np.sum(psi[k][i](t_m-loc_times_2,
                                                               kernel_param[k][i])))
        active_ind += 1
    return residuals_k


def get_residuals(process_path, psi, mu, kernel_param, sampling=False,
                  sample_size=10**3, seed=1234, verbose=False):
    d = process_path.d
    return [get_residuals_k(k, process_path, psi, mu, kernel_param,
                            sampling=sampling, sample_size=sample_size,
                            seed=seed, verbose=verbose) for k in range(d)]


def ks_test_residuals(residuals):
    return [stats.kstest(res, 'expon') for res in residuals]


def qq_plot(residuals, n_models=1, labels=None, style='exponential',
            substract_yx=False, normalize=False, max_points=None,
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
            margin = 0.01
            indices_subsample = np.concatenate(([m for m in range(int(margin*len(osm)))], np.linspace(int(margin*len(osm)), int((1-margin)*len(osm)), max_points), [m for m in range(int((1-margin)*len(osm)), len(osm))]))
            indices_subsample = [int(x) for x in indices_subsample]
            osm = osm[indices_subsample]
            osr = osr[indices_subsample]
        if substract_yx:
            ax.plot(osm, osr-osm, marker="o", linestyle="None", **kwargs)
        else:
            ax.plot(osm, osr, marker="o", linestyle="None", **kwargs)
            x_45 = np.linspace(*ax.get_xlim())
            ax.plot(x_45, x_45, c='black')
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
