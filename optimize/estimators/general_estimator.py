# License: BSD 3 clause

import copy
import itertools

import numpy as np

from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification
from aslsd.utilities import useful_statistics as us


class GeneralEstimator():
    def __init__(self, **kwargs):
        self.t = 0
        self.is_done = False
        # Logging
        self.is_grad_target = kwargs.get('is_grad_target', False)

        # Logging
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
        self.is_log_allocs = kwargs.get('is_log_allocs', False)
        self.is_log_sum_f = kwargs.get('is_log_sum_f', False)
        self.is_log_sum_diff_f = kwargs.get('is_log_sum_diff_f', False)

    # Stratification parameters
    def set_stratification(self, data, **kwargs):
        stratif = kwargs.get('stratification', None)
        if stratif is None:
            stratif = GeneralStratification(data, **kwargs)
        self.stratif = stratif

    def clear_estimates(self, n_param=1):
        # Sum f
        # Total estimates
        self.sum_f = 0.
        # Local estimates
        self.vecsum_f_adaptive = np.zeros(self.stratif.n_adaptive_strata)
        self.sum_f_adaptive = 0.
        self.vecsum_f_nonadaptive = np.zeros(self.stratif.n_nonadaptive_strata)
        self.sum_f_nonadaptive = 0.
        self.sum_f_exact = 0.

        # Gradient
        # Total estimates
        self.sum_diff_f = np.zeros(n_param)
        # Local estimates
        self.vecsum_diff_f_adaptive = np.zeros((n_param,
                                                self.stratif.n_adaptive_strata))
        self.sum_diff_f_adaptive = np.zeros(n_param)
        self.vecsum_diff_f_nonadaptive = np.zeros((n_param,
                                                   self.stratif.n_nonadaptive_strata))
        self.sum_diff_f_nonadaptive = np.zeros(n_param)
        self.sum_diff_f_exact = np.zeros(n_param)
        self.sum_diff_f = np.zeros(n_param)

    # Logging
    def intialize_logs(self, n_param=1):
        n_iter = self.n_iter
        # Sampled indices
        if self.is_log_ixs:
            self.logged_ixs = {'adaptive': [None for t in range(n_iter)],
                               'non_adaptive': [None for t in range(n_iter)]}
        else:
            self.logged_ixs = None
        # Allocations
        if self.is_log_allocs:
            self.logged_allocs = [None for t in range(n_iter)]
        else:
            self.logged_allocs = None

        # Sum estimates
        if self.is_log_sum_f:
            self.logged_sum_f = np.zeros(n_iter)
        if self.is_log_sum_diff_f:
            self.logged_sum_diff_f = np.zeros((n_iter, n_param))

    def get_log(self):
        logs = {'samples': self.logged_ixs, 'allocs': self.logged_allocs}
        if self.is_log_sum_f:
            logs['sum_f'] = self.logged_sum_f
        if self.is_log_sum_diff_f:
            logs['sum_diff_f'] = self.logged_sum_diff_f
        return copy.deepcopy(logs)

    # Estimates
    def estimate_sum_adaptivedom(self, data, f=None, diff_f=None, n_param=1,
                                 f_args=None, diff_f_args=None,
                                 compute_f_sum=False, compute_diff_f_sum=True,
                                 grad_alloc=True, rng=None, seed=None,
                                 coeff=1.):
        strf = self.stratif
        if strf.n_adaptive_strata == 0:
            return
        if f_args is None:
            f_args = {}
        if diff_f_args is None:
            diff_f_args = {}
        if grad_alloc:
            compute_diff_f_sum = True
        else:
            compute_f_sum = True

        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
            

        adaptive_strata = strf.adaptive_strata

        n_adaptive_strata = strf.n_adaptive_strata
        n_adaptive_iters = strf.n_adaptive_iters
        rel_alloc = (strf.ema_weight*strf.adaptive_rel_alloc
                     + (1.-strf.ema_weight)*strf.guess_rel_alloc)
        default_rel_alloc = copy.deepcopy(rel_alloc)
        # Save initial guess
        strf.guess_rel_alloc = copy.deepcopy(rel_alloc)

        mean_f = np.zeros(n_adaptive_strata)
        mean_diff_f = np.zeros((n_param, n_adaptive_strata))

        mean = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        n_samples = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[0]))))) for ix_strata in range(n_adaptive_strata)]
        samples = [None for ix_iter in range(n_adaptive_iters)]

        # Iteration 1
        # This step is treated apart because of the batch formula for the std
        samples[0] = us.sample_single_ix_stratified(adaptive_strata, n_samples,
                                                    rng=rng)

        # Convert indices to times
        data_samples = [data[samples[0][ix]] for ix in range(n_adaptive_strata)]

        for ix_strata in range(n_adaptive_strata):
            if compute_f_sum:
                f_vals = coeff*f(data_samples[ix_strata], **f_args)
                # Initialize satistics
                mean_f[ix_strata] = np.mean(f_vals)
                self.vecsum_f_adaptive[ix_strata] = mean_f[ix_strata]
            if compute_diff_f_sum:
                diff_f_vals = [None]*n_param
                if grad_alloc:
                    grad_norm_vals = np.zeros(len(data_samples[ix_strata]))
                for ix_param in range(n_param):
                    diff_f_vals[ix_param] = coeff*diff_f(data_samples[ix_strata],
                                                         ix_param, **diff_f_args)
                    mean_diff_f[ix_param][ix_strata] = np.mean(diff_f_vals[ix_param])
                    self.vecsum_diff_f_adaptive[ix_param][ix_strata] = mean_diff_f[ix_param][ix_strata]
                    if grad_alloc:
                        grad_norm_vals += diff_f_vals[ix_param]**2
            if grad_alloc:
                grad_norm_vals = np.sqrt(grad_norm_vals)
                std[ix_strata] = np.std(grad_norm_vals)
                mean[ix_strata] = np.mean(grad_norm_vals)
            else:
                std[ix_strata] = np.std(f_vals)
                mean[ix_strata] = np.mean(f_vals)
            # Adapt allocation
            strata_size = strf.adaptive_strata_sizes[ix_strata]
            rel_alloc[ix_strata] = (strata_size
                                    * np.sqrt(strata_size/(strata_size-1.))
                                    * std[ix_strata])
        sum_rel_alloc = np.sum(rel_alloc)
        if sum_rel_alloc > 0.:
            rel_alloc = rel_alloc/sum_rel_alloc
        else:
            rel_alloc = default_rel_alloc

        # General iterations
        for ix in range(1, n_adaptive_iters):
            n_samples_step = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[ix]))))) for ix_strata in range(n_adaptive_strata)]
            samples[ix] = us.sample_single_ix_stratified(adaptive_strata,
                                                         n_samples_step,
                                                         rng=rng)
            data_samples = [data[samples[ix][ix_2]] for ix_2 in range(n_adaptive_strata)]

            for ix_strata in range(n_adaptive_strata):
                strata_size = strf.adaptive_strata_sizes[ix_strata]
                if compute_f_sum:
                    f_vals = coeff*f(data_samples[ix_strata], **f_args)
                    # Compute statistics
                    mean_f_delta = np.mean(f_vals)
                    mean_f[ix_strata] = us.batch_mean(mean_f[ix_strata],
                                                      n_samples[ix_strata],
                                                      mean_f_delta,
                                                      n_samples_step[ix_strata])
                    self.vecsum_f_adaptive[ix_strata] = mean_f[ix_strata]

                    if not grad_alloc:
                        std_delta = np.std(f_vals)
                        std[ix_strata] = us.batch_std(std[ix_strata],
                                                      n_samples[ix_strata],
                                                      std_delta,
                                                      n_samples_step[ix_strata],
                                                      strata_size[ix_strata],
                                                      mean[ix_strata],
                                                      mean_f_delta)
                        mean[ix_strata] = mean_f[ix_strata]

                if compute_diff_f_sum:
                    grad_norm_vals = np.zeros(len(data_samples[ix_strata]))
                    for ix_param in range(n_param):
                        diff_f_vals = coeff*diff_f(data_samples[ix_strata],
                                                   ix_param,
                                                   **diff_f_args)
                        mean_diff_delta = np.mean(diff_f_vals)
                        mean_diff_f[ix_param][ix_strata] = us.batch_mean(mean_diff_f[ix_param][ix_strata],
                                                                         n_samples[ix_strata],
                                                                         mean_diff_delta,
                                                                         n_samples_step[ix_strata])
                        self.vecsum_diff_f_adaptive[ix_param][ix_strata] = mean_diff_f[ix_param][ix_strata]
                        if grad_alloc:
                            grad_norm_vals += diff_f_vals[ix_param]**2
                    if grad_alloc:
                        grad_norm_vals = np.sqrt(grad_norm_vals)
                        mean_grad_norm_delta = np.mean(grad_norm_vals)
                        std_delta = np.std(grad_norm_vals)
                        std[ix_strata] = us.batch_std(std[ix_strata],
                                                      n_samples[ix_strata],
                                                      std_delta,
                                                      n_samples_step[ix_strata],
                                                      strata_size[ix_strata],
                                                      mean[ix_strata],
                                                      mean_grad_norm_delta)
                        mean[ix_strata] = us.batch_mean(mean[ix_strata],
                                                        n_samples[ix_strata],
                                                        mean_grad_norm_delta,
                                                        n_samples_step[ix_strata])

                # Adapt allocation
                rel_alloc[ix_strata] = (strata_size[ix_strata]
                                        * strf.adaptive_strata_fpop[ix_strata]
                                        * std[ix_strata])
                n_samples[ix_strata] += n_samples_step[ix_strata]
            sum_rel_alloc = np.sum(rel_alloc)
            if sum_rel_alloc > 0.:
                rel_alloc = rel_alloc/sum_rel_alloc
            else:
                rel_alloc = default_rel_alloc

        # Save relative allocation
        strf.adaptive_rel_alloc = copy.deepcopy(rel_alloc)
        if self.is_log_allocs:
            self.logged_allocs[self.t] = copy.deepcopy(rel_alloc)

        # Compute all sums of derivatives
        if compute_diff_f_sum:
            for ix_param in range(n_param):
                # Rescale means
                for ix_strata in range(n_adaptive_strata):
                    self.vecsum_diff_f_adaptive[ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                # Sum over strata
                self.sum_diff_f_adaptive[ix_param] = sum(self.vecsum_diff_f_adaptive[ix_param])

        # Compute sum of f
        if compute_f_sum:
            # Rescale means
            for ix_strata in range(n_adaptive_strata):
                self.vecsum_f_adaptive[ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
            # Sum over strata
            self.sum_f_adaptive = sum(self.vecsum_f_adaptive)

        # Save sampled indices if needed
        if self.is_log_ixs:
            samples = [[samples[ix_iter][ix_strata]
                        for ix_iter in range(n_adaptive_iters)]
                       for ix_strata in range(n_adaptive_strata)]
            for ix_strata in range(n_adaptive_strata):
                samples[ix_strata] = np.concatenate(samples[ix_strata])
            self.logged_ixs['adaptive'][self.t] = samples

    def estimate_sum_nonadaptivedom(self, data, f=None, diff_f=None, n_param=1,
                                    f_args=None, diff_f_args=None,
                                    compute_f_sum=False,
                                    compute_diff_f_sum=True, rng=None,
                                    seed=None, coeff=1.):
        strf = self.stratif
        if strf.n_nonadaptive_strata == 0:
            return
        if f_args is None:
            f_args = {}
        if diff_f_args is None:
            diff_f_args = {}
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        nonadaptive_strata = strf.nonadaptive_strata
        nonadaptive_strata_sizes = strf.nonadaptive_strata_sizes
        n_nonadaptive_strata = strf.n_nonadaptive_strata
        nonadaptive_abs_alloc = strf.nonadaptive_abs_alloc
        # Sample indices
        samples = us.sample_single_ix_stratified(nonadaptive_strata,
                                                 nonadaptive_abs_alloc,
                                                 rng=rng)
        if self.is_log_ixs:
            self.logged_ixs['nonadaptive'][self.t] = samples
        data_samples = [data[samples[ix]] for ix in range(n_nonadaptive_strata)]

        # Compute estimates of f derivatives
        if compute_diff_f_sum:
            for ix_param in range(n_param):
                for ix_strata in range(n_nonadaptive_strata):
                    self.vecsum_diff_f_nonadaptive[ix_param][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(coeff*diff_f(data_samples[ix_strata], ix_param, **diff_f_args))
                # Sum over strata
                self.sum_diff_f_nonadaptive[ix_param] = sum(self.vecsum_diff_f_nonadaptive[ix_param])

        # Compute estimate of sum f
        if compute_f_sum:
            for ix_strata in range(n_nonadaptive_strata):
                self.vecsum_f_nonadaptive[ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(coeff*f(data_samples[ix_strata], **f_args))
            # Sum over strata
            self.sum_f_nonadaptive = sum(self.vecsum_f_nonadaptive)

    def estimate_sum_exactdom(self, data, f=None, diff_f=None, n_param=1,
                              f_args=None, diff_f_args=None,
                              compute_f_sum=False, compute_diff_f_sum=True,
                              coeff=1.):
        strf = self.stratif
        if strf.n_exact == 0:
            return
        if f_args is None:
            f_args = {}
        if diff_f_args is None:
            diff_f_args = {}
        exact_ixs = strf.exact_domain
        # Compute estimates of f derivatives
        if compute_diff_f_sum:
            for ix_param in range(n_param):
                self.sum_diff_f_exact[ix_param] = np.sum(coeff*diff_f(data[exact_ixs], ix_param, **diff_f_args))

        # Compute estimate of sum f
        if compute_f_sum:
            self.sum_f_exact = np.sum(coeff*f(data[exact_ixs], **f_args))

    def estimate_sum(self, data, f=None, diff_f=None, n_param=1, f_args=None,
                     diff_f_args=None, compute_f_sum=False,
                     compute_diff_f_sum=True, grad_alloc=True, rng=None,
                     seed=None, coeff=1., count_iter=True):
        self.clear_estimates(n_param=n_param)
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        # Estimates on the adaptive domain
        self.estimate_sum_adaptivedom(data, f=f, diff_f=diff_f, n_param=n_param,
                                      f_args=f_args, diff_f_args=diff_f_args,
                                      compute_f_sum=compute_f_sum,
                                      compute_diff_f_sum=compute_diff_f_sum,
                                      grad_alloc=grad_alloc, rng=rng,
                                      seed=seed, coeff=coeff)

        # Estimates on the non-adaptive domain
        self.estimate_sum_nonadaptivedom(data, f=f, diff_f=diff_f, n_param=n_param,
                                         f_args=f_args,
                                         diff_f_args=diff_f_args,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         rng=rng, seed=seed, coeff=coeff)
        # Exact domain
        self.estimate_sum_exactdom(data, f=f, diff_f=diff_f, n_param=n_param,
                                   f_args=f_args, diff_f_args=diff_f_args,
                                   compute_f_sum=compute_f_sum,
                                   compute_diff_f_sum=compute_diff_f_sum,
                                   coeff=coeff)

        # Compute estimates of f derivatives
        if compute_diff_f_sum:
            for ix_param in range(n_param):
                S = self.sum_diff_f_adaptive[ix_param]
                S += self.sum_diff_f_nonadaptive[ix_param]
                S += self.sum_diff_f_exact[ix_param]
                self.sum_diff_f[ix_param] = S

        # Compute estimate of sum of f
        if compute_f_sum:
            self.sum_f = self.sum_f_adaptive+self.sum_f_nonadaptive+self.sum_f_exact

        if count_iter:
            self.t += 1

    # True Optimal allocations
    def get_true_allocation_f(self, f=None, diff=None, n_param=1, f_args=None,
                              diff_f_args=None):
        strf = self.stratif
        adaptive_strata = strf.adaptive_strata
        n_adaptive_strata = strf.n_adaptive_strata
        strata_sizes = strf.adaptive_strata_sizes

        alloc = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        for ix_strata in range(n_adaptive_strata):
            h_inf = adaptive_strata[ix_strata][0]
            h_sup = adaptive_strata[ix_strata][1]
            f_vals = np.zeros(strata_sizes[ix_strata])
            ix_start = 0
            ix_end = 0
            for h in range(h_inf, h_sup+1):
                ix_start = ix_end
                times_m = self.list_times[k][self.varpi[k][i][h]:self.event_counts[k]]
                times_n = self.list_times[i][[self.kappa[i][k][m]+1-h for m in range(self.varpi[k][i][h], self.event_counts[k])]]
                ix_end = ix_start+self.n_events[k]-self.varpi[k][i][h]
                f_vals[ix_start:ix_end] = f(times_m-times_n, **f_args)

            # Initialize satistics
            std[ix_strata] = np.std(f_vals)
            # Adapt allocation
            alloc[ix_strata] = (strata_sizes[ix_strata]
                                * strf.adaptive_strata_fpop[ix_strata]
                                * std[ix_strata])
        alloc = alloc/np.sum(alloc)
        return alloc
