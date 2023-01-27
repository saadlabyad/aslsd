# License: BSD 3 clause

import copy
import itertools

import numpy as np

from aslsd.estimators.estimator import Estimator
from aslsd.estimators.single_ix_stratification import SingleIxStratification
from aslsd.estimators.double_ix_stratification import DoubleIxStratification
from aslsd.estimators.general_estimator import GeneralEstimator
from aslsd.estimators.general_stratification import GeneralStratification
from aslsd.utilities import useful_statistics as us


class MtlhStratified(Estimator):
    def __init__(self, **kwargs):
        self.t = 0
        self.is_done = False
        # Logging
        self.is_grad_target = kwargs.get('is_grad_target', False)

        # Logging
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
        self.is_log_allocs = kwargs.get('is_log_allocs', False)
        self.is_log_lse = kwargs.get('is_log_lse', False)

        # Additive terms appearing in LSE gradient estimate
        # Single sums in mu derivative
        self.sum_psi = None
        # Single sums in kernel derivative
        self.sum_diff_psi = None
        self.sum_diff_upsilonzero = None
        # Double sums in kernel derivative
        self.vecsum_diff_phi = None
        self.sum_diff_phi = None
        self.vecsum_diff_sim_upsilon = None
        self.sum_diff_sim_upsilon = None
        self.vecsum_diff_cross_upsilon = None
        self.sum_diff_cross_upsilon = None

        # Additive terms appearing in LSE estimate only
        # Single sums
        self.sum_upsilonzero = None
        # Double sums
        self.vecsum_phi = None
        self.sum_phi = None
        self.vecsum_upsilon = None
        self.sum_upsilon = None

    # Stratification parameters
    def set_singlesum_estimation(self, **kwargs):
        d = self.d
        # psi
        psi_stratif = kwargs.get('psi_stratification',
                                 [None for i in range(d)])
        for i in range(d):
            if psi_stratif[i] is None:
                n_events_i = self.n_events[i]
                psi_stratif[i] = SingleIxStratification(n_events_i, **kwargs)
        self.psi_stratif = psi_stratif
        # upsilonzero
        upsilonzero_stratif = kwargs.get('upsilonzero_stratification',
                                         [None for i in range(d)])
        for i in range(d):
            if upsilonzero_stratif[i] is None:
                n_events_i = self.n_events[i]
                upsilonzero_stratif[i] = SingleIxStratification(n_events_i,
                                                                **kwargs)
        self.upsilonzero_stratif = upsilonzero_stratif

    def set_doublesum_estimation(self, **kwargs):
        d = self.d
        k = self.k

        # upsilon
        upsilon_stratif = kwargs.get('upsilon_stratification',
                                     [[None for j in range(d)]
                                      for i in range(d)])
        for i, j in itertools.product(range(d), range(d)):
            if upsilon_stratif[i][j] is None:
                h_max = self.kappa[j][i][self.n_events[i]-1]+1
                upsilon_stratif[i][j] = DoubleIxStratification(h_max, **kwargs)
                upsilon_stratif[i][j].get_stratification_size(self.lag_sizes, i, j)
        self.upsilon_stratif = upsilon_stratif
        # phi
        phi_stratif = kwargs.get('phi_stratification', [None]*d)
        for i in range(d):
            if phi_stratif[i] is None:
                h_max = self.kappa[i][k][self.n_events[k]-1]+1
                phi_stratif[i] = DoubleIxStratification(h_max, **kwargs)
                phi_stratif[i].get_stratification_size(self.lag_sizes, k, i)
        self.phi_stratif = phi_stratif

    def set_stratification(self, **kwargs):
        self.set_singlesum_estimation(**kwargs)
        self.set_doublesum_estimation(**kwargs)
        d = self.d
        k = self.k
        # Additive terms appearing in LSE gradient estimate
        # Single sums in mu derivative
        self.sum_psi = np.zeros(d)
        # Single sums in kernel derivative
        self.sum_diff_psi = [np.zeros(self.matrix_n_param[k][i]) for i in range(d)]
        self.sum_diff_upsilonzero = [np.zeros(self.matrix_n_param[k][i]) for i in range(d)]
        # Double sums in kernel derivative
        self.vecsum_diff_phi = [[np.zeros(self.phi_stratif[i].n_adaptive_strata) for x in range(self.matrix_n_param[k][i])] for i in range(d)]
        self.sum_diff_phi = [[0. for x in range(self.matrix_n_param[k][i])] for i in range(d)]
        self.vecsum_diff_sim_upsilon = [[np.zeros(self.upsilon_stratif[i][i].n_adaptive_strata) for x in range(self.matrix_n_param[k][i])] for i in range(d)]
        self.sum_diff_sim_upsilon = [[0. for x in range(self.matrix_n_param[k][i])] for i in range(d)]
        vecsum_diff_cross_upsilon = [[[None for ix_func in range(3)] for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            for ix_func in [1,2]:
                z = [i, j][ix_func-1]
                vecsum_diff_cross_upsilon[i][j][ix_func] = [np.zeros(self.upsilon_stratif[i][j].n_adaptive_strata) for x in range(self.matrix_n_param[k][z])]
        self.vecsum_diff_cross_upsilon = vecsum_diff_cross_upsilon
        sum_diff_cross_upsilon = [[[None for ix_func in range(3)] for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            for ix_func in [1, 2]:
                z = [i, j][ix_func-1]
                sum_diff_cross_upsilon[i][j][ix_func] = [0. for x in range(self.matrix_n_param[k][z])]
        self.sum_diff_cross_upsilon = sum_diff_cross_upsilon
        # Additive terms appearing in LSE estimate only
        # Single sums
        self.sum_upsilonzero = np.zeros(d)
        # Double sums
        self.vecsum_phi = [np.zeros(self.phi_stratif[i].n_adaptive_strata) for i in range(d)]
        self.sum_phi = np.zeros(d)
        self.vecsum_upsilon = [[np.zeros(self.upsilon_stratif[i][j].n_adaptive_strata) for j in range(d)] for i in range(d)]
        self.sum_upsilon = [[0. for j in range(d)] for i in range(d)]

    # Logging
    def intialize_logs(self):
        d = self.d
        n_iter = self.n_iter
        # Sampled indices
        if self.is_log_ixs:
            self.logged_ixs_phi = [[None for t in range(n_iter)]
                                   for i in range(d)]
            self.logged_ixs_upsilon = [[[None for t in range(n_iter)]
                                        for j in range(d)] for i in range(d)]
            self.logged_ixs_psi = [[None for t in range(n_iter)]
                                   for i in range(d)]
            self.logged_ixs_upsilonzero = [[None for t in range(n_iter)]
                                           for i in range(d)]
        else:
            self.logged_ixs_phi = None
            self.logged_ixs_upsilon = None
            self.logged_ixs_psi = None
            self.logged_ixs_upsilonzero = None
        # Allocations
        if self.is_log_allocs:
            self.logged_allocs_phi = [[None for t in range(n_iter)]
                                      for i in range(d)]
            self.logged_allocs_upsilon = [[[None for t in range(n_iter)]
                                           for j in range(d)]
                                          for i in range(d)]
        else:
            self.logged_allocs_phi = None
            self.logged_allocs_upsilon = None

        # LSE
        if self.is_log_lse:
            self.logged_lse = np.zeros(n_iter)

    def get_log(self):
        logs = {'samples': {}, 'allocs': {}}
        if self.is_log_lse:
            logs['lse'] = self.logged_lse

        if self.is_log_ixs:
            logs['samples']['psi'] = self.logged_ixs_psi
            logs['samples']['upsilonzero'] = self.logged_ixs_upsilonzero
            logs['samples']['phi'] = self.logged_ixs_phi
            logs['samples']['upsilon'] = self.logged_ixs_upsilon

        if self.is_log_allocs:
            logs['allocs']['phi'] = self.logged_allocs_phi
            logs['allocs']['upsilon'] = self.logged_allocs_upsilon

        return logs

    # Estimates
    # Single sums
    def estimate_dpsi(self, i, x_ki, grad=True, rng=None,
                      seed=None):
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        strf = self.psi_stratif[i]
        strata = strf.strata
        strata_sizes = strf.strata_sizes
        n_strata = strf.n_strata
        abs_alloc = strf.abs_alloc
        m_spec = strf.m_spec
        # Sample indices
        samples = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
        if self.is_log_ixs:
            self.logged_ixs_psi[i][self.t] = samples
        # Convert indices to time differences
        times = us.stratified_single_ixs2times(i, self.list_times2end, samples)

        S = 0.
        for ix_strata in range(n_strata):
            S += strata_sizes[ix_strata]*np.mean(self.psi[k][i](times[ix_strata], x_ki))
        # print('S without spec', S)
        S += np.sum(self.psi[k][i](self.list_times2end[i][m_spec:], x_ki))
        # print('S with spec', S)
        self.sum_psi[i] = S
        if grad:
            for ix_param in range(n_param_ki):
                S = 0.
                for ix_strata in range(n_strata):
                    S += strata_sizes[ix_strata]*np.mean(self.diff_psi[k][i](times[ix_strata], ix_param, x_ki))
                S += np.sum(self.diff_psi[k][i](self.list_times2end[i][m_spec:], ix_param, x_ki))
                self.sum_diff_psi[i][ix_param] = S

    def estimate_dupsilonzero(self, i, x_ki, func=False, grad=True, rng=None,
                              seed=None):
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        strf = self.upsilonzero_stratif[i]
        strata = strf.strata
        strata_sizes = strf.strata_sizes
        n_strata = strf.n_strata
        abs_alloc = strf.abs_alloc
        m_spec = strf.m_spec
        # Sample indices
        samples = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
        if self.is_log_ixs:
            self.logged_ixs_upsilonzero[i][self.t] = samples
        # Convert indices to times
        times = us.stratified_single_ixs2times(i, self.list_times2end, samples)
        if func:
            S = 0.
            for ix_strata in range(n_strata):
                S += strata_sizes[ix_strata]*np.mean(self.upsilon[i][i][k](times[ix_strata], 0., x_ki,
                                                  x_ki))
            S += np.sum(self.upsilon[i][i][k](self.list_times2end[i][m_spec:], 0.,
                                              x_ki, x_ki))
            self.sum_upsilonzero[i] = S
        if grad:
            for ix_param in range(n_param_ki):
                S = 0.
                for ix_strata in range(n_strata):
                    S += strata_sizes[ix_strata]*np.mean(self.diff_sim_upsilon[k][i](times[ix_strata], 0., ix_param, x_ki))
                S += np.sum(self.diff_sim_upsilon[k][i](self.list_times2end[i][m_spec:], 0., ix_param, x_ki))
                self.sum_diff_upsilonzero[i][ix_param] = S

    # Double sums
    # Phi
    def estimate_dphi_adaptivedom(self, i, x_ki, rng=None, seed=None):
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]

        strf = self.phi_stratif[i]
        n_adaptive_strata = strf.n_adaptive_strata
        n_adaptive_iters = strf.n_adaptive_iters
        rel_alloc = (strf.ema_weight*strf.adaptive_rel_alloc
                     + (1.-strf.ema_weight)*strf.guess_rel_alloc)
        default_rel_alloc = copy.deepcopy(rel_alloc)
        # Save initial guess
        strf.guess_rel_alloc = copy.deepcopy(rel_alloc)

        mean = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        n_samples = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[0]))))) for ix_strata in range(n_adaptive_strata)]
        samples = [None for ix_iter in range(n_adaptive_iters)]

        # Iteration 1
        # This step is treated apart because of the batch formula for the std
        samples[0] = us.sample_double_ix_stratified(k, i,
                                                    strf.adaptive_strata,
                                                    n_samples,
                                                    self.kappa, self.varpi,
                                                    self.n_events, rng=rng)
        times = us.stratified_double_ixs2phi_times(k, i, self.list_times,
                                                   samples[0])
        for ix_strata in range(n_adaptive_strata):
            phi_vals = self.phi[k][i](times[ix_strata], x_ki)
            # Initialize satistics
            mean[ix_strata] = np.mean(phi_vals)
            for ix_param in range(n_param_ki):
                diff_phi_vals = self.diff_phi[k][i](times[ix_strata],
                                                    ix_param, x_ki)
                mean_diff = np.mean(diff_phi_vals)
                self.vecsum_diff_phi[i][ix_param][ix_strata] = mean_diff
            std[ix_strata] = np.std(phi_vals)
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
        # print('Sum rel_alloc phi', np.sum(rel_alloc))
        # General iterations
        for ix in range(1, n_adaptive_iters):
            nsamples_step = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[ix]))))) for ix_strata in range(n_adaptive_strata)]
            samples[ix] = us.sample_double_ix_stratified(k, i,
                                                         strf.adaptive_strata,
                                                         nsamples_step,
                                                         self.kappa,
                                                         self.varpi,
                                                         self.n_events,
                                                         rng=rng)
            times = us.stratified_double_ixs2phi_times(k, i,
                                                       self.list_times,
                                                       samples[ix])
            for ix_strata in range(n_adaptive_strata):
                phi_vals = self.phi[k][i](times[ix_strata], x_ki)
                # Compute statistics
                mean_delta = np.mean(phi_vals)
                std_delta = np.std(phi_vals)
                # Update satistics
                strata_size = strf.adaptive_strata_sizes[ix_strata]
                std[ix_strata] = us.batch_std(std[ix_strata],
                                              n_samples[ix_strata],
                                              std_delta,
                                              nsamples_step[ix_strata],
                                              strata_size[ix_strata],
                                              mean[ix_strata],
                                              mean_delta)
                mean[ix_strata] = us.batch_mean(mean[ix_strata],
                                                n_samples[ix_strata],
                                                mean_delta,
                                                nsamples_step[ix_strata])
                for ix_param in range(n_param_ki):
                    diff_phi_vals = self.diff_phi[k][i](times[ix_strata],
                                                        ix_param, x_ki)
                    mean_diff_delta = np.mean(diff_phi_vals)
                    self.vecsum_diff_phi[i][ix_param][ix_strata] = us.batch_mean(self.sum_diff_phi[i][ix_param][ix_strata], n_samples[ix_strata], mean_diff_delta, nsamples_step[ix_strata])
                # Adapt allocation
                rel_alloc[ix_strata] = (strata_size[ix_strata]
                                        * strf.adaptive_strata_fpop[ix_strata]
                                        * std[ix_strata])
                n_samples[ix_strata] += nsamples_step[ix_strata]
            sum_rel_alloc = np.sum(rel_alloc)
            if sum_rel_alloc > 0.:
                rel_alloc = rel_alloc/sum_rel_alloc
            else:
                rel_alloc = default_rel_alloc
        # Save relative allocation
        strf.adaptive_rel_alloc = copy.deepcopy(rel_alloc)
        if self.is_log_allocs:
            self.logged_allocs_phi[i][self.t] = copy.deepcopy(rel_alloc)

        # Compute all sums of derivatives
        for ix_param in range(n_param_ki):
            # Rescale means
            for ix_strata in range(n_adaptive_strata):
                self.vecsum_diff_phi[i][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
            # Sum over strata
            self.sum_diff_phi[i][ix_param] = sum(self.vecsum_diff_phi[i][ix_param])
        # Compute sum of phi; useful to log the LSE estimates
        if self.is_log_lse:
            for ix_strata in range(n_adaptive_strata):
                self.vecsum_phi[i][ix_strata] = (mean[ix_strata]
                                                 * strf.adaptive_strata_sizes[ix_strata])
            self.sum_phi[i] = sum(self.vecsum_phi[i])
        # Save sampled indices if needed
        if self.is_log_ixs:
            samples = [[samples[ix_iter][ix_strata]
                        for ix_iter in range(n_adaptive_iters)]
                       for ix_strata in range(n_adaptive_strata)]
            for ix_strata in range(n_adaptive_strata):
                samples[ix_strata] = np.concatenate(samples[ix_strata])
            self.logged_ixs_phi[i][self.t]['adaptive'] = samples

    def estimate_dphi(self, i, x_ki, rng=None, seed=None):
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        # Estimates on the adaptive on the domain
        self.estimate_dphi_adaptivedom(i, x_ki, rng=rng)

        # Estimates on the non-adaptive on the domain
        strf = self.phi_stratif[i]
        nonadaptive_strata = strf.nonadaptive_strata
        nonadaptive_strata_sizes = strf.nonadaptive_strata_sizes
        n_nonadaptive_strata = strf.n_nonadaptive_strata
        nonadaptive_abs_alloc = strf.nonadaptive_abs_alloc
        # Sample index pairs
        samples = us.sample_double_ix_stratified(k, i, nonadaptive_strata,
                                                 nonadaptive_abs_alloc,
                                                 self.kappa, self.varpi,
                                                 self.n_events, rng=rng)
        if self.is_log_ixs:
            self.logged_ixs_phi[i][self.t]['nonadaptive'] = samples
        # Convert index pairs to time differences
        times = us.stratified_double_ixs2phi_times(k, i, self.list_times,
                                                   samples)
        # Compute estimates of phi derivatives
        for ix_param in range(n_param_ki):
            S = self.sum_diff_phi[i][ix_param]
            for ix_strata in range(n_nonadaptive_strata):
                S += nonadaptive_strata_sizes[ix_strata]*np.mean(self.diff_phi[k][i](times[ix_strata], ix_param, x_ki))
            self.sum_diff_phi[i][ix_param] = S

        # Compute estimate of sum of phi; needed to compute the LSE estimate.
        if self.is_log_lse:
            S = self.sum_phi[i]
            for ix_strata in range(n_nonadaptive_strata):
                S += nonadaptive_strata_sizes[ix_strata]*np.mean(self.phi[k][i](times[ix_strata], x_ki))
            self.sum_phi[i] = S

    def estimate_dphi_adaptivedom_gtarget(self, i, x_ki, rng=None, seed=None):
        pass

    def estimate_dphi_gtarget(self, i, x_ki, rng=None, seed=None):
        pass

    # Upsilon
    def estimate_dupsilon_adaptivedom(self, i, j, x_ki, x_kj, rng=None,
                                      seed=None):
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        k = self.k
        n_param_ki = self.matrix_n_param[k][i]
        n_param_kj = self.matrix_n_param[k][j]
        list_n_param = [None, n_param_ki, n_param_kj]

        strf = self.upsilon_stratif[i][j]
        n_adaptive_strata = strf.n_adaptive_strata
        n_adaptive_iters = strf.n_adaptive_iters
        rel_alloc = (strf.ema_weight*strf.adaptive_rel_alloc
                     + (1.-strf.ema_weight)*strf.guess_rel_alloc)
        default_rel_alloc = copy.deepcopy(rel_alloc)
        # Save initial guess
        strf.guess_rel_alloc = copy.deepcopy(rel_alloc)

        mean = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        n_samples = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[0]))))) for ix_strata in range(n_adaptive_strata)]
        samples = [None for ix_iter in range(n_adaptive_iters)]

        # Iteration 1
        # This step is treated apart because of the batch formula for the std
        samples[0] = us.sample_double_ix_stratified(i, j,
                                                    strf.adaptive_strata,
                                                    n_samples,
                                                    self.kappa, self.varpi,
                                                    self.n_events, rng=rng)
        times = us.stratified_double_ixs2upsilon_times(i, j,
                                                       self.list_times,
                                                       self.list_times2end,
                                                       samples[0])
        for ix_strata in range(n_adaptive_strata):
            
            upsilon_vals = self.upsilon[i][j][k](times[ix_strata]['tf'],
                                              times[ix_strata]['mn'], x_ki,
                                              x_kj)
            # Initialize satistics
            mean[ix_strata] = np.mean(upsilon_vals)
            if i == j:
                for ix_param in range(n_param_ki):
                    diff_upsilon_vals = self.diff_sim_upsilon[k][i](times[ix_strata]['tf'],
                                                                    times[ix_strata]['mn'],
                                                                    ix_param, x_ki)
                    mean_diff = np.mean(diff_upsilon_vals)
                    self.vecsum_diff_sim_upsilon[i][ix_param][ix_strata] = np.mean(mean_diff)

            else:
                for ix_func in [1, 2]:
                    for ix_param in range(list_n_param[ix_func]):
                        diff_upsilon_vals = self.diff_cross_upsilon[i][j][k](times[ix_strata]['tf'], times[ix_strata]['mn'], ix_func, ix_param, x_ki, x_kj)
                        mean_diff = np.mean(diff_upsilon_vals)
                        self.vecsum_diff_cross_upsilon[i][j][ix_func][ix_param][ix_strata] = np.mean(mean_diff)
            std[ix_strata] = np.std(upsilon_vals)
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
        # print('rel_alloc', rel_alloc)
        # print('Sum rel alloc upsilon', np.sum(rel_alloc))
        # print('x_ki', x_ki)
        # print('x_kj', x_kj)
        # General iterations
        for ix in range(1, n_adaptive_iters):
            nsamples_step = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[ix]))))) for ix_strata in range(n_adaptive_strata)]
            samples[ix] = us.sample_double_ix_stratified(i, j,
                                                         strf.adaptive_strata,
                                                         nsamples_step,
                                                         self.kappa,
                                                         self.varpi,
                                                         self.n_events,
                                                         rng=rng)
            times = us.stratified_double_ixs2upsilon_times(i, j,
                                                           self.list_times,
                                                           self.list_times2end,
                                                           samples[ix])
            for ix_strata in range(n_adaptive_strata):
                upsilon_vals = self.upsilon[i][j][k](times[ix_strata]['tf'],
                                                     times[ix_strata]['mn'],
                                                     x_ki, x_kj)
                # Compute statistics
                mean_delta = np.mean(upsilon_vals)
                std_delta = np.std(upsilon_vals)
                # Update satistics
                strata_size = strf.adaptive_strata_sizes[ix_strata]
                std[ix_strata] = us.batch_std(std[ix_strata],
                                              n_samples[ix_strata],
                                              std_delta,
                                              nsamples_step[ix_strata],
                                              strata_size[ix_strata],
                                              mean[ix_strata],
                                              mean_delta)
                mean[ix_strata] = us.batch_mean(mean[ix_strata],
                                                n_samples[ix_strata],
                                                mean_delta,
                                                nsamples_step[ix_strata])
                if i == j:
                    for ix_param in range(n_param_ki):
                        diff_upsilon_vals = self.diff_sim_upsilon[k][i](times[ix_strata]['tf'], times[ix_strata]['mn'], ix_param, x_ki)
                        mean_diff_delta = np.mean(diff_upsilon_vals)
                        self.vecsum_diff_sim_upsilon[i][ix_param][ix_strata] = us.batch_mean(self.sum_diff_sim_upsilon[i][ix_param][ix_strata], n_samples[ix_strata], mean_diff_delta, nsamples_step[ix_strata])
                else:
                    for ix_func in [1, 2]:
                        for ix_param in range(list_n_param[ix_func]):
                            diff_upsilon_vals = self.diff_cross_upsilon[i][j][k](times[ix_strata]['tf'], times[ix_strata]['mn'], ix_func,
                                                                                 ix_param, x_ki, x_kj)
                            mean_diff_delta = np.mean(diff_upsilon_vals)
                            self.vecsum_diff_cross_upsilon[i][j][ix_func][ix_param][ix_strata] = us.batch_mean(self.sum_diff_cross_upsilon[i][j][ix_func][ix_param][ix_strata], n_samples[ix_strata], mean_diff_delta, nsamples_step[ix_strata])
                # Adapt allocation
                rel_alloc[ix_strata] = (strata_size[ix_strata]
                                        * strf.adaptive_strata_fpop[ix_strata]
                                        * std[ix_strata])
                n_samples[ix_strata] += nsamples_step[ix_strata]
            sum_rel_alloc = np.sum(rel_alloc)
            if sum_rel_alloc > 0.:
                rel_alloc = rel_alloc/sum_rel_alloc
            else:
                rel_alloc = default_rel_alloc
        # Save relative allocation
        strf.adaptive_rel_alloc = copy.deepcopy(rel_alloc)
        if self.is_log_allocs:
            self.logged_allocs_upsilon[i][j][self.t] = copy.deepcopy(rel_alloc)

        # Compute all sums of derivatives
        if i == j:
            for ix_param in range(n_param_ki):
                # Rescale means
                for ix_strata in range(n_adaptive_strata):
                    self.vecsum_diff_sim_upsilon[i][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                # Sum over strata
                self.sum_diff_sim_upsilon[i][ix_param] = sum(self.vecsum_diff_sim_upsilon[i][ix_param])
        else:
            for ix_func in [1, 2]:
                for ix_param in range(list_n_param[ix_func]):
                    # Rescale means
                    for ix_strata in range(n_adaptive_strata):
                        self.vecsum_diff_cross_upsilon[i][j][ix_func][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                    # Sum over strata
                    self.sum_diff_cross_upsilon[i][j][ix_func][ix_param] = sum(self.vecsum_diff_cross_upsilon[i][j][ix_func][ix_param])
        # Compute sum of upsilon; useful to log the LSE estimates
        if self.is_log_lse:
            for ix_strata in range(n_adaptive_strata):
                self.vecsum_upsilon[i][j][ix_strata] = (mean[ix_strata]*strf.adaptive_strata_sizes[ix_strata])
            self.sum_upsilon[i][j] = sum(self.vecsum_upsilon[i][j])
        # Save sampled indices if needed
        if self.is_log_ixs:
            samples = [[samples[ix_iter][ix_strata]
                        for ix_iter in range(n_adaptive_iters)]
                       for ix_strata in range(n_adaptive_strata)]
            for ix_strata in range(n_adaptive_strata):
                samples[ix_strata] = np.concatenate(samples[ix_strata])
            self.logged_ixs_upsilon[i][j][self.t]['adaptive'] = samples

    def estimate_dupsilon(self, i, j, x_ki, x_kj, rng=None, seed=None):
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]
        n_param_kj = self.matrix_n_param[k][j]
        list_n_param = [None, n_param_ki, n_param_kj]
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        # Estimates on the adaptive on the domain
        self.estimate_dupsilon_adaptivedom(i, j, x_ki, x_kj, rng=rng)

        # Estimates on the non-adaptive on the domain
        strf = self.upsilon_stratif[i][j]
        nonadaptive_strata = strf.nonadaptive_strata
        nonadaptive_strata_sizes = strf.nonadaptive_strata_sizes
        n_nonadaptive_strata = strf.n_nonadaptive_strata
        nonadaptive_abs_alloc = strf.nonadaptive_abs_alloc
        # Sample index pairs
        samples = us.sample_double_ix_stratified(i, j, nonadaptive_strata,
                                                 nonadaptive_abs_alloc,
                                                 self.kappa, self.varpi,
                                                 self.n_events, rng=rng)
        if self.is_log_ixs:
            self.logged_ixs_upsilon[i][j][self.t]['nonadaptive'] = samples
        # Convert index pairs to time differences
        times = us.stratified_double_ixs2upsilon_times(i, j,
                                                       self.list_times,
                                                       self.list_times2end,
                                                       samples)
        # Compute estimates of upsilon derivatives
        if i == j:
            for ix_param in range(n_param_ki):
                S = self.sum_diff_sim_upsilon[i][ix_param]
                for ix_strata in range(n_nonadaptive_strata):
                    S += nonadaptive_strata_sizes[ix_strata]*np.mean(self.diff_sim_upsilon[k][i](times[ix_strata]['tf'], times[ix_strata]['mn'], ix_param, x_ki))
                self.sum_diff_sim_upsilon[i][ix_param] = S
        else:
            for ix_func in [1, 2]:
                for ix_param in range(list_n_param[ix_func]):
                    S = self.sum_diff_cross_upsilon[i][j][ix_func][ix_param]
                    for ix_strata in range(n_nonadaptive_strata):
                        S += nonadaptive_strata_sizes[ix_strata]*np.mean(self.diff_cross_upsilon[i][j][k](times[ix_strata]['tf'], times[ix_strata]['mn'], ix_func, ix_param, x_ki, x_kj))
                    self.sum_diff_cross_upsilon[i][j][ix_func][ix_param] = S

        # Compute estimate of sum of upsilon; needed to compute the LSE estimate.
        if self.is_log_lse:
            S = self.sum_upsilon[i][j]
            for ix_strata in range(n_nonadaptive_strata):
                S += nonadaptive_strata_sizes[ix_strata]*np.mean(self.upsilon[i][j][k](times[ix_strata]['tf'], times[ix_strata]['mn'], x_ki, x_kj))
            self.sum_upsilon[i][j] = S

    def estimate_dupsilon_adaptivedom_gtarget(self, i, j, x_ki, x_kj, rng=None,
                                              seed=None):
        pass

    def estimate_dupsilon_gtarget(self, i, j, x_ki, x_kj, rng=None, seed=None):
        pass

    # Final
    def diff_lse_mu_k(self, mu_k):
        d = self.d
        k = self.k
        T_f = self.T_f
        eta_k = self.eta[k]

        res = 2*(mu_k-eta_k)
        for i in range(d):
            res += 2.*(self.sum_psi[i]/T_f)
        return res

    def diff_lse_theta_kpr(self, p, r, mu_k):
        # Derivative of Z with respect to the rth parameter of phi_kp
        d = self.d
        T_f = self.T_f

        res = 0.

        # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
        for i in itertools.chain(range(p), range(p+1, d)):
            res += 2.*(self.sum_diff_cross_upsilon[i][p][2][r]/T_f)

        # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
        for j in itertools.chain(range(p), range(p+1, d)):
            res += 2.*(self.sum_diff_cross_upsilon[p][j][1][r]/T_f)

        # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
        res += 2.*(self.sum_diff_sim_upsilon[p][r]/T_f)

        # Term 4 : Phi_{kp}
        res -= 2.*(self.sum_diff_phi[p][r]/T_f)

        # Term 5 : Psi_{kp}
        res += 2.*mu_k*(self.sum_diff_psi[p][r]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        res += self.sum_diff_upsilonzero[p][r]/T_f

        return res

    def lse_k_estimate(self, x_k, rng=None, seed=None):
        # Compute the LSE estimate
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        d = self.d
        k = self.k
        T_f = self.T_f
        eta_k = self.eta[k]
        mu_k = x_k[0]

        res = 0.
        res += mu_k**2-2*eta_k*mu_k

        # If a gradient estimate has already been produced for x_k.
        # This is useful to produce an LSE estimate while leveraging
        # computations already made for a gradient estimate
        if self.is_done:
            # Single sums
            for i in range(d):
                #   Psi_ki(T-t^i_m)
                res += 2.*mu_k*(self.sum_psi[i]/T_f)
                #   Upsilon_iik(T-t^i_m,0)
                res += self.sum_upsilonzero[i]/T_f

            # Double sums
            # Upsilon
            for i, j in itertools.product(range(d), range(d)):
                res += 2.*(self.sum_upsilon[i][j]/T_f)
            # Phi
            for j in range(d):
                res -= 2.*(self.sum_phi[j]/T_f)
            return res
        # If a gradient estimate has not been produced yet for x_k.
        # In this case, we need to sample data and compute the LSE
        # estimate from scratch
        else:
            pass

    def lse_k_grad_estimate(self, x_k, rng=None, seed=None):
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        k = self.k
        d = self.d
        mu_k = x_k[0]
        grad = np.zeros(self.n_param_k)
        # Single sums
        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            self.estimate_dpsi(i, x_ki, grad=True, rng=rng)
            self.estimate_dupsilonzero(i, x_ki, func=self.is_log_lse,
                                       grad=True, rng=rng)

        # Double Sums
        # Phi
        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            if self.is_grad_target:
                self.estimate_dphi_gtarget(i, x_ki, rng=rng)
            else:
                self.estimate_dphi(i, x_ki, rng=rng)

        # Upsilon
        for i, j in itertools.product(range(d), range(d)):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            if self.is_grad_target:
                self.estimate_dupsilon_gtarget(i, j, x_ki, x_kj, rng=rng)
            else:
                self.estimate_dupsilon(i, j, x_ki, x_kj, rng=rng)

        # Derivative with respect to \mu_k
        grad[0] = self.diff_lse_mu_k(mu_k)

        # Derivative with respect to kernel parameters
        for ix_param in range(1, self.n_param_k):
            p = self.ix_map[k][ix_param]['ker']
            r = self.ix_map[k][ix_param]['par']
            grad[ix_param] = self.diff_lse_theta_kpr(p, r, mu_k)

        self.is_done = True
        # Log LSE
        if self.is_log_lse:
            lse = self.lse_k_estimate(x_k)
            self.logged_lse[self.t] = lse
        self.t += 1
        self.is_done = False
        return grad

    # True Optimal allocations
    def get_true_allocation_phi(self, i, x_ki):
        k = self.k
        n_param_ki = self.matrix_n_param[k][i]

        strf = self.phi_stratif[i]
        adaptive_strata = strf.adaptive_strata
        n_adaptive_strata = strf.n_adaptive_strata
        strata_sizes = strf.adaptive_strata_sizes

        alloc = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        for ix_strata in range(n_adaptive_strata):
            h_inf = adaptive_strata[ix_strata][0]
            h_sup = adaptive_strata[ix_strata][1]
            phi_vals = np.zeros(strata_sizes[ix_strata])
            ix_start = 0
            ix_end = 0
            for h in range(h_inf, h_sup+1):
                ix_start = ix_end
                times_m = self.list_times[k][self.varpi[k][i][h]:self.event_counts[k]]
                times_n = self.list_times[i][[self.kappa[i][k][m]+1-h for m in range(self.varpi[k][i][h], self.event_counts[k])]]
                ix_end = ix_start+self.n_events[k]-self.varpi[k][i][h]
                phi_vals[ix_start:ix_end] = self.phi[k][i](times_m-times_n, x_ki)

            # Initialize satistics
            std[ix_strata] = np.std(phi_vals)
            # Adapt allocation
            alloc[ix_strata] = (strata_sizes[ix_strata]
                                * strf.adaptive_strata_fpop[ix_strata]
                                * std[ix_strata])
        alloc = alloc/np.sum(alloc)
        return alloc

    def get_true_allocation_upsilon(self, i, j, x_ki, x_kj):
        k = self.k
        T_f = self.TT_f
        n_param_ki = self.matrix_n_param[k][i]
        n_param_kj = self.matrix_n_param[k][j]
        list_n_param = [None, 1, 2]

        strf = self.phi_stratif[i]
        adaptive_strata = strf.adaptive_strata
        n_adaptive_strata = strf.n_adaptive_strata
        strata_sizes = strf.adaptive_strata_sizes

        alloc = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        for ix_strata in range(n_adaptive_strata):
            h_inf = adaptive_strata[ix_strata][0]
            h_sup = adaptive_strata[ix_strata][1]
            upsilon_vals = np.zeros(strata_sizes[ix_strata])
            ix_start = 0
            ix_end = 0
            for h in range(h_inf, h_sup+1):
                ix_start = ix_end
                times_m = self.list_times[k][self.varpi[k][i][h]:
                                             self.event_counts[k]]
                times_n = self.list_times[i][[self.kappa[i][k][m]+1-h
                                              for m in range(self.varpi[k][i][h], self.event_counts[k])]]
                ix_end = ix_start+self.n_events[k]-self.varpi[k][i][h]
                if self.is_grad_target:
                    pass
                else:
                    upsilon_vals[ix_start:ix_end] = self.upsilon[i][j][k](T_f-times_m, times_m-times_n, x_ki, x_kj)

            # Initialize satistics
            std[ix_strata] = np.std(upsilon_vals)
            # Adapt allocation
            alloc[ix_strata] = (strata_sizes[ix_strata]
                                * strf.adaptive_strata_fpop[ix_strata]
                                * std[ix_strata])
        alloc = alloc/np.sum(alloc)
        return alloc
