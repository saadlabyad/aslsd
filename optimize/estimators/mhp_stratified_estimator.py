# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification
from aslsd.optimize.stratifications.\
    double_ix_stratification import DoubleIxStratification
from aslsd.utilities import useful_statistics as us


class MHPStratEstim(Estimator):
    def __init__(self, is_grad_target=False, is_log_ixs=False,
                 is_log_allocs=False, is_log_total_estimates=False,
                 is_log_strata_estimates=False, is_log_lse=False):
        self.t = 0

        # Gradient target
        self.is_grad_target = is_grad_target

        # Logging
        self.is_log_ixs = is_log_ixs
        self.is_log_allocs = is_log_allocs
        self.is_log_total_estimates = is_log_total_estimates
        self.is_log_strata_estimates = is_log_strata_estimates
        self.is_log_lse = is_log_lse

        # Estimation
        # Sum f
        # Total estimates
        self.sum_f = {}
        # Local estimates
        self.vecsum_f_adaptive = {}
        self.sum_f_adaptive = {}
        self.vecsum_f_nonadaptive = {}
        self.sum_f_nonadaptive = {}
        self.sum_f_exact = {}

        # Gradient
        # Total estimates
        self.sum_diff_f = {}
        # Local estimates
        self.vecsum_diff_f_adaptive = {}
        self.sum_diff_f_adaptive = {}
        self.vecsum_diff_f_nonadaptive = {}
        self.sum_diff_f_nonadaptive = {}
        self.sum_diff_f_exact = {}

    # Stratification parameters
    def set_singlesum_estimation(self, n_exact=None, n_samples_adaptive=None,
                                 nonadaptive_sample_size=None, **kwargs):
        d = self.d
        local_kwargs = {}
        if n_exact is not None:
            local_kwargs['n_exact'] = n_exact
        if n_samples_adaptive is not None:
            local_kwargs['n_samples_adaptive'] = n_samples_adaptive
        if nonadaptive_sample_size is not None:
            local_kwargs['nonadaptive_sample_size'] = nonadaptive_sample_size
        # psi
        psi_stratif = kwargs.get('psi_stratification',
                                 [None for i in range(d)])
        for i in range(d):
            log_name = 'psi_'+str(i)
            if psi_stratif[i] is None:
                n_events_i = self.n_events[i]
                psi_stratif[i] = GeneralStratification(n=n_events_i,
                                                       **local_kwargs)
            self.stratif[log_name] = psi_stratif[i]
        # upsilonzero
        upsilonzero_stratif = kwargs.get('upsilonzero_stratification',
                                         [None for i in range(d)])
        for i in range(d):
            log_name = 'upsilonzero_'+str(i)
            if upsilonzero_stratif[i] is None:
                n_events_i = self.n_events[i]
                upsilonzero_stratif[i] = GeneralStratification(n=n_events_i,
                                                               **local_kwargs)
            self.stratif[log_name] = upsilonzero_stratif[i]

    def set_doublesum_estimation(self, n_samples_adaptive=None,
                                 nonadaptive_sample_size=None, **kwargs):
        d = self.d
        k = self.k
        local_kwargs = {}
        if n_samples_adaptive is not None:
            local_kwargs['n_samples_adaptive'] = n_samples_adaptive
        if nonadaptive_sample_size is not None:
            local_kwargs['nonadaptive_sample_size'] = nonadaptive_sample_size
        # upsilon
        upsilon_stratif = kwargs.get('upsilon_stratification',
                                     [[None for j in range(d)]
                                      for i in range(d)])
        for i, j in itertools.product(range(d), range(d)):
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            if upsilon_stratif[i][j] is None:
                h_max = self.kappa[j][i][self.n_events[i]-1]+1
                upsilon_stratif[i][j] = DoubleIxStratification(h_max=h_max,
                                                               **local_kwargs)
                upsilon_stratif[i][j].get_stratification_size(self.lag_sizes,
                                                              i, j)
            self.stratif[log_name] = upsilon_stratif[i][j]
        # phi
        phi_stratif = kwargs.get('phi_stratification', [None]*d)
        for i in range(d):
            log_name = 'phi_'+str(i)
            if phi_stratif[i] is None:
                h_max = self.kappa[i][k][self.n_events[k]-1]+1
                phi_stratif[i] = DoubleIxStratification(h_max=h_max,
                                                        **local_kwargs)
                phi_stratif[i].get_stratification_size(self.lag_sizes, k, i)
        self.stratif[log_name] = phi_stratif[i]

    def get_list_log_names(self):
        d = self.d
        list_log_names = []
        for i in range(d):
            list_log_names.append('phi_'+str(i))
            list_log_names.append('upsilonzero_'+str(i))
            list_log_names.append('psi_'+str(i))
        for i, j in itertools.product(range(d), range(d)):
            list_log_names.append('upsilon_'+str(i)+'_'+str(j))
        return list_log_names

    def get_singlesum_lognames(self):
        d = self.d
        single_ix_lognames = []
        # psi
        for i in range(d):
            log_name = 'psi_'+str(i)
            single_ix_lognames.append(log_name)
        # upsilonzero
        for i in range(d):
            log_name = 'upsilonzero_'+str(i)
            single_ix_lognames.append(log_name)
        return single_ix_lognames

    def get_doublesum_lognames(self):
        d = self.d
        double_ix_lognames = []
        # upsilon
        for i, j in itertools.product(range(d), range(d)):
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            double_ix_lognames.append(log_name)
        # phi
        for i in range(d):
            log_name = 'phi_'+str(i)
            double_ix_lognames.append(log_name)
        return double_ix_lognames

    def make_param_indexing(self):
        d = self.d
        k = self.k
        # parameters indexing
        self.n_func = {}
        self.n_param = {}
        # mu
        for i in range(d):
            # phi
            log_name = 'phi_'+str(i)
            self.n_func[log_name] = 1
            self.n_param[log_name] = self.matrix_n_param[k][i]
            # upsilonzero
            log_name = 'upsilonzero_'+str(i)
            self.n_func[log_name] = 1
            self.n_param[log_name] = self.matrix_n_param[k][i]
            # Psi
            log_name = 'psi_'+str(i)
            self.n_func[log_name] = 2
            self.n_param[log_name] = [None, self.matrix_n_param[k][i], 1]
        for i, j in itertools.product(range(d), range(d)):
            # upsilon
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            if i == j:
                self.n_func[log_name] = 1
                self.n_param[log_name] = self.matrix_n_param[k][i]
            else:
                self.n_func[log_name] = 2
                self.n_param[log_name] = [None, self.matrix_n_param[k][i],
                                          self.matrix_n_param[k][j]]

    def set_stratification(self, n_exact_single=None,
                           n_samples_adaptive_single=None,
                           nonadaptive_sample_size_single=None,
                           single_strfs=None, n_samples_adaptive_double=None,
                           nonadaptive_sample_size_double=None,
                           double_strfs=None):
        self.stratif = {}
        if single_strfs is None:
            single_strfs = {}
        self.set_singlesum_estimation(n_exact=n_exact_single,
                                      n_samples_adaptive=n_samples_adaptive_single,
                                      nonadaptive_sample_size=nonadaptive_sample_size_single,
                                      **single_strfs)
        if double_strfs is None:
            double_strfs = {}
        self.set_doublesum_estimation(n_samples_adaptive=n_samples_adaptive_double,
                                      nonadaptive_sample_size=nonadaptive_sample_size_double,
                                      **double_strfs)

    def initialize(self, k, n_iter, model, data, n_exact_single=None,
                   n_samples_adaptive_single=None,
                   nonadaptive_sample_size_single=None,
                   single_strfs=None, n_samples_adaptive_double=None,
                   nonadaptive_sample_size_double=None, double_strfs=None):
        self.k = k
        self.n_iter = n_iter
        self.initialize_model_data(model, data)
        self.set_stratification(n_exact_single=n_exact_single,
                                n_samples_adaptive_single=n_samples_adaptive_single,
                                nonadaptive_sample_size_single=nonadaptive_sample_size_single,
                                single_strfs=single_strfs,
                                n_samples_adaptive_double=n_samples_adaptive_double,
                                nonadaptive_sample_size_double=nonadaptive_sample_size_double,
                                double_strfs=double_strfs)
        self.list_log_names = self.get_list_log_names()
        self.make_param_indexing()
        self.initialize_logs()

    def make_dict_n_param(self):
        d = self.d
        k = self.k
        dict_n_param = {}
        # mu term
        dict_n_param['mu_term'] = 1
        # Sum Phi
        dict_n_param['sum_phi'] = [None for j in range(d)]
        for j in range(d):
            dict_n_param['sum_phi'][j] = self.matrix_n_param[k][j]
        # Sum Psi
        dict_n_param['sum_psi'] = [None for i in range(d)]
        for i in range(d):
            dict_n_param['sum_psi'][i] = [None, self.matrix_n_param[k][i], 1]
        # Sum Upsilon
        dict_n_param['sum_upsilon'] = [[None for j in range(d)]
                                       for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            if i == j:
                dict_n_param['sum_upsilon'][i][i] = self.matrix_n_param[k][i]
            else:
                dict_n_param['sum_upsilon'][i][j] = [None,
                                                     self.matrix_n_param[k][i],
                                                     self.matrix_n_param[k][j]]
        # Sum Upsilonzero
        dict_n_param['sum_upsilonzero'] = [None for i in range(d)]
        for i in range(d):
            dict_n_param['sum_upsilonzero'][i] = self.matrix_n_param[k][i]

        # Result
        self.dict_n_param = dict_n_param

    def make_null_sumarray(self, n_param, n_func=1, n_strata=None):
        if n_func == 1:
            if n_strata is None:
                L = np.zeros(n_param)
            else:
                L = np.zeros((n_param, n_strata))
        else:
            L = [None]*3
            for ix_func in [1, 2]:
                if n_strata is None:
                    L[ix_func] = np.zeros(n_param[ix_func])
                else:
                    L[ix_func] = np.zeros((n_param[ix_func], n_strata))
        return L

    def clear_estimates(self):
        # mu term
        self.mu_term = 0.
        self.diff_mu_term = 0.

        # Sum terms
        for log_name in self.list_log_names:
            strf = self.stratif[log_name]

            # Sum f
            # Total estimates
            self.sum_f[log_name] = 0.
            # Local estimates
            self.vecsum_f_adaptive[log_name] = np.zeros(strf.n_adaptive_strata)
            self.sum_f_adaptive[log_name] = 0.
            self.vecsum_f_nonadaptive[log_name] = np.zeros(strf.n_nonadaptive_strata)
            self.sum_f_nonadaptive[log_name] = 0.
            self.sum_f_exact[log_name] = 0.

            # Gradient
            n_func = self.n_func[log_name]
            n_param = self.n_param[log_name]
            # Total estimates
            self.sum_diff_f[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=None)
            # Local estimates
            self.vecsum_diff_f_adaptive[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=strf.n_adaptive_strata)
            self.sum_diff_f_adaptive[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=None)
            self.vecsum_diff_f_nonadaptive[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=strf.n_nonadaptive_strata)
            self.sum_diff_f_nonadaptive[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=None)
            self.sum_diff_f_exact[log_name] = self.make_null_sumarray(n_param, n_func=n_func, n_strata=None)

# =============================================================================
# Logging
# =============================================================================
    def make_null_sumarray_logs(self, n_param, n_func=1, n_strata=None,
                                n_iter=1):
        # Make null arrays to store logs
        if n_func == 1:
            if n_strata is None:
                L = np.zeros((n_param, n_iter))
            else:
                L = np.zeros((n_param, n_strata, n_iter))
        else:
            L = [None]*3
            for ix_func in [1, 2]:
                if n_strata is None:
                    L[ix_func] = np.zeros((n_param[ix_func], n_iter))
                else:
                    L[ix_func] = np.zeros((n_param[ix_func], n_strata,
                                           n_iter))
        return L

    def update_sumarray_logs(self, log_type, log_name, ix_iter, n_param,
                             n_func=1, n_strata=None):
        logs = getattr(self, log_type+'_log')[log_name]
        val = getattr(self, log_type)[log_name]
        if n_func == 1:
            if n_strata is None:
                logs[:, ix_iter] = val+0.
            else:
                logs[:, :, ix_iter] = val+0.
        else:
            for ix_func in [1, 2]:
                if n_strata is None:
                    logs[ix_func][:, ix_iter] = val[ix_func]+0.
                else:
                    logs[ix_func][:, :, ix_iter] = val[ix_func]+0.

    def initialize_logs(self):
        d = self.d
        n_iter = self.n_iter
        # Sampled indices
        self.logged_ixs = {}
        if self.is_log_ixs:
            for log_name in self.list_log_names:
                self.logged_ixs[log_name] = {}
                self.logged_ixs[log_name]['adaptive'] = [None]*n_iter
                self.logged_ixs[log_name]['nonadaptive'] = [None]*n_iter

        else:
            for log_name in self.list_log_names:
                self.logged_ixs[log_name] = {}
                self.logged_ixs[log_name]['adaptive'] = None
                self.logged_ixs[log_name]['nonadaptive'] = None
        # Allocations
        self.logged_allocs = {}
        if self.is_log_allocs:
            for log_name in self.list_log_names:
                self.logged_allocs[log_name] = [None]*n_iter
        else:
            for log_name in self.list_log_names:
                self.logged_allocs[log_name] = None

        # LSE
        if self.is_log_lse:
            self.logged_lse = np.zeros(n_iter+1)

        # Estimates
        if self.is_log_total_estimates:
            # mu term
            self.mu_term_log = np.zeros(n_iter)
            self.diff_mu_term_log = np.zeros(n_iter)

            # Sum terms
            self.sum_f_log = {}
            self.sum_diff_f_log = {}
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]

                # Sum f
                # Total estimates
                self.sum_f_log[log_name] = np.zeros(n_iter)

                # Gradient
                n_func = self.n_func[log_name]
                n_param = self.n_param[log_name]
                # Total estimates
                self.sum_diff_f_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=None, n_iter=n_iter)

        if self.is_log_strata_estimates:
            self.vecsum_f_adaptive_log = {}
            self.sum_f_adaptive_log = {}
            self.vecsum_f_nonadaptive_log = {}
            self.sum_f_nonadaptive_log = {}
            self.sum_f_exact_log = {}
            self.vecsum_diff_f_adaptive_log = {}
            self.sum_diff_f_adaptive_log = {}
            self.vecsum_diff_f_nonadaptive_log = {}
            self.sum_diff_f_nonadaptive_log = {}
            self.sum_diff_f_exact_log = {}
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]
                # Sum f
                self.vecsum_f_adaptive_log[log_name] = np.zeros((strf.n_adaptive_strata,
                                                                 n_iter))
                self.sum_f_adaptive_log[log_name] = np.zeros(n_iter)
                self.vecsum_f_nonadaptive_log[log_name] = np.zeros((strf.n_nonadaptive_strata,
                                                                    n_iter))
                self.sum_f_nonadaptive_log[log_name] = np.zeros(n_iter)
                self.sum_f_exact_log[log_name] = np.zeros(n_iter)
                # Gradient
                n_func = self.n_func[log_name]
                n_param = self.n_param[log_name]
                # Local estimates
                self.vecsum_diff_f_adaptive_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=strf.n_adaptive_strata, n_iter=n_iter)
                self.sum_diff_f_adaptive_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=None, n_iter=n_iter)
                self.vecsum_diff_f_nonadaptive_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=strf.n_nonadaptive_strata, n_iter=n_iter)
                self.sum_diff_f_nonadaptive_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=None, n_iter=n_iter)
                self.sum_diff_f_exact_log[log_name] = self.make_null_sumarray_logs(n_param, n_func=n_func, n_strata=None, n_iter=n_iter)

    def log_estimates(self):
        ix_iter = self.t
        if self.is_log_total_estimates:
            # mu term
            self.mu_term_log[ix_iter] = self.mu_term
            self.diff_mu_term_log[ix_iter] = self.diff_mu_term

            # Sum terms
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]

                # Sum f
                # Total estimates
                self.sum_f_log[log_name][ix_iter] = self.sum_f[log_name]

                # Gradient
                n_func = self.n_func[log_name]
                n_param = self.n_param[log_name]
                # Total estimates
                self.update_sumarray_logs('sum_diff_f', log_name, ix_iter,
                                          n_param, n_func=n_func,
                                          n_strata=None)
        if self.is_log_strata_estimates:
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]
                # Sum f
                self.vecsum_f_adaptive_log[log_name][:, ix_iter] = self.vecsum_f_adaptive[log_name]+0.
                self.sum_f_adaptive_log[log_name][ix_iter] = self.sum_f_adaptive[log_name]
                self.vecsum_f_nonadaptive_log[log_name][:, ix_iter] = self.vecsum_f_nonadaptive[log_name]+0.
                self.sum_f_nonadaptive_log[log_name][ix_iter] = self.sum_f_nonadaptive[log_name]
                self.sum_f_exact_log[log_name][ix_iter] = self.sum_f_exact[log_name]
                # Gradient
                n_func = self.n_func[log_name]
                n_param = self.n_param[log_name]
                # Local estimates
                self.update_sumarray_logs('vecsum_diff_f_adaptive', log_name,
                                          ix_iter, n_param, n_func=n_func,
                                          n_strata=strf.n_adaptive_strata)
                self.update_sumarray_logs('sum_diff_f_adaptive', log_name,
                                          ix_iter, n_param, n_func=n_func,
                                          n_strata=None)
                self.update_sumarray_logs('vecsum_diff_f_nonadaptive',
                                          log_name, ix_iter, n_param,
                                          n_func=n_func,
                                          n_strata=strf.n_nonadaptive_strata)
                self.update_sumarray_logs('sum_diff_f_nonadaptive',
                                          log_name, ix_iter, n_param,
                                          n_func=n_func,
                                          n_strata=None)
                self.update_sumarray_logs('sum_diff_f_exact',
                                          log_name, ix_iter, n_param,
                                          n_func=n_func,
                                          n_strata=None)

    def get_lse_k_components_log(self):
        d = self.d
        T_f = self.T_f
        n_lse_k_components = 1+len(self.list_log_names)
        # Initialize
        lse_k_components = np.zeros((self.n_iter, n_lse_k_components))
        lse_k_component_names = ['mu_term']

        # Get components
        ix_component = 0
        # Baseline terms
        lse_k_components[:, ix_component] = self.mu_term_log+0.
        ix_component += 1

        # Baseline-kernel correlation
        for i in range(d):
            log_name = 'psi_'+str(i)
            lse_k_components[:, ix_component] = 2.*(self.sum_f_log[log_name]/T_f)+0.
            lse_k_component_names.append(log_name)
            ix_component += 1

        # kernel-kernel correlation
        for i, j in itertools.product(range(d), range(d)):
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            lse_k_components[:, ix_component] = 2.*(self.sum_f_log[log_name]/T_f)+0.
            lse_k_component_names.append(log_name)
            ix_component += 1

        # Term 4 : Kernel terms
        for j in range(d):
            log_name = 'phi_'+str(j)
            lse_k_components[:, ix_component] = -2.*(self.sum_f_log[log_name]/T_f)+0.
            lse_k_component_names.append(log_name)
            ix_component += 1
        for i in range(d):
            log_name = 'upsilonzero_'+str(i)
            lse_k_components[:, ix_component] = self.sum_f_log[log_name]/T_f+0.
            lse_k_component_names.append(log_name)
            ix_component += 1

        return lse_k_components, lse_k_component_names

    def get_log(self):
        logs = {}
        if self.is_log_lse:
            logs['lse'] = self.logged_lse

        if self.is_log_ixs:
            logs['samples'] = copy.deepcopy(self.logged_ixs)

        if self.is_log_allocs:
            logs['allocs'] = copy.deepcopy(self.logged_allocs)

        return logs

# =============================================================================
# Sum estimation
# =============================================================================
    def estimate_sum_adaptivedom(self, strf, sampling_func, ixs_transform,
                                 get_n_samples, log_name, f=None,
                                 diff_f=None, n_func=1,
                                 n_param=None,
                                 f_args=None, diff_f_args=None,
                                 compute_f_sum=False, compute_diff_f_sum=True,
                                 grad_alloc=True, rng=None, seed=None,
                                 coeff=1.):
        if strf.n_adaptive_strata == 0:
            return
        if f_args is None:
            f_args_ = [{} for ix in range(n_func)]
        if diff_f_args is None:
            diff_f_args_ = [{} for ix in range(n_func)]

        if grad_alloc:
            compute_diff_f_sum = True
        else:
            compute_f_sum = True
        if compute_diff_f_sum and n_func > 1:
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

        # Estimation statistics
        mean_f = np.zeros(n_adaptive_strata)
        mean_diff_f = [None]*n_func
        if n_func == 1:
            mean_diff_f = [[None for ix_strata in range(n_adaptive_strata)]
                           for ix_param in range(n_param)]
        else:
            mean_diff_f = [None]*3
            for ix_func in [1, 2]:
                mean_diff_f[ix_func] = [[None for ix_strata in range(n_adaptive_strata)]
                                        for ix_param in range(n_param[ix_func])]

        # Allocation decision statistics
        mean = np.zeros(n_adaptive_strata)
        std = np.zeros(n_adaptive_strata)
        n_samples = [min(strf.adaptive_strata_sizes[ix_strata], max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[0]))))) for ix_strata in range(n_adaptive_strata)]
        samples = [None for ix_iter in range(n_adaptive_iters)]

        # Iteration 1
        # This step is treated apart because of the batch formula for the std
        samples[0] = sampling_func(adaptive_strata, n_samples, rng=rng)

        for ix_strata in range(n_adaptive_strata):
            n_samples_iter = get_n_samples(samples[0][ix_strata])
            data_samples = ixs_transform(samples[0][ix_strata])
            if compute_f_sum:
                f_args_ = dict(f_args, **data_samples)
                f_vals = f(**f_args_)
                # Initialize satistics
                f_vals = coeff*f_vals
                mean_f[ix_strata] = np.mean(f_vals)
                self.vecsum_f_adaptive[log_name][ix_strata] = mean_f[ix_strata]
            if compute_diff_f_sum:
                if grad_alloc:
                    grad_norm_vals = np.zeros(n_samples_iter)
                    if n_func == 1:
                        for ix_param in range(n_param):
                            diff_f_args_ = dict(diff_f_args, **data_samples)
                            diff_f_args_['ix_diff'] = ix_param
                            diff_f_vals = coeff*diff_f(**diff_f_args_)
                            mean_diff_f[ix_param][ix_strata] = np.mean(diff_f_vals)
                            self.vecsum_diff_f_adaptive[log_name][ix_param][ix_strata] = mean_diff_f[ix_param][ix_strata]
                            if grad_alloc:
                                grad_norm_vals += diff_f_vals**2
                    else:
                        for ix_func in [1, 2]:
                            for ix_param in range(n_param[ix_func]):
                                diff_f_args_ = dict(diff_f_args,
                                                    **data_samples)
                                diff_f_args_['ix_diff'] = ix_param
                                diff_f_args_['ix_func'] = ix_func
                                diff_f_vals = coeff*diff_f(**diff_f_args_)
                                mean_diff_f[ix_func][ix_param][ix_strata] = np.mean(diff_f_vals)
                                self.vecsum_diff_f_adaptive[log_name][ix_func][ix_param][ix_strata] = mean_diff_f[ix_func][ix_param][ix_strata]
                                if grad_alloc:
                                    grad_norm_vals += diff_f_vals**2

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
            n_samples_step = [min(strf.adaptive_strata_sizes[ix_strata],
                              max(1, (int(rel_alloc[ix_strata]*(n_adaptive_strata+strf.n_samples_adaptive[ix]))))) for ix_strata in range(n_adaptive_strata)]
            samples[ix] = sampling_func(adaptive_strata, n_samples_step,
                                        rng=rng)
            for ix_strata in range(n_adaptive_strata):
                data_samples = ixs_transform(samples[ix][ix_strata])
                n_samples_iter = get_n_samples(samples[ix][ix_strata])
                strata_size = strf.adaptive_strata_sizes[ix_strata]
                if compute_f_sum:
                    f_args_ = dict(f_args, **data_samples)
                    f_vals = coeff*f(**f_args_)
                    # Compute statistics
                    mean_f_delta = np.mean(f_vals)
                    mean_f[ix_strata] = us.batch_mean(mean_f[ix_strata],
                                                      n_samples[ix_strata],
                                                      mean_f_delta,
                                                      n_samples_step[ix_strata])
                    self.vecsum_f_adaptive[log_name][ix_strata] = mean_f[ix_strata]

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
                    if grad_alloc:
                        grad_norm_vals = np.zeros(n_samples_iter)
                    if n_func == 1:
                        for ix_param in range(n_param):
                            diff_f_args_ = dict(diff_f_args, **data_samples)
                            diff_f_args_['ix_diff'] = ix_param
                            diff_f_vals = coeff*diff_f(**diff_f_args_)
                            mean_diff_delta = np.mean(diff_f_vals)
                            mean_diff_f[ix_param][ix_strata] = us.batch_mean(mean_diff_f[ix_param][ix_strata],
                                                                             n_samples[ix_strata],
                                                                             mean_diff_delta,
                                                                             n_samples_step[ix_strata])
                            self.vecsum_diff_f_adaptive[log_name][ix_param][ix_strata] = mean_diff_f[ix_param][ix_strata]
                            if grad_alloc:
                                grad_norm_vals += diff_f_vals**2
                    else:
                        for ix_func in [1, 2]:
                            for ix_param in range(n_param[ix_func]):
                                diff_f_args_ = dict(diff_f_args,
                                                    **data_samples)
                                diff_f_args_['ix_diff'] = ix_param
                                diff_f_args_['ix_func'] = ix_func
                                diff_f_vals = coeff*diff_f(**diff_f_args_)
                                mean_diff_delta = np.mean(diff_f_vals)
                                mean_diff_f[ix_func][ix_param][ix_strata] = us.batch_mean(mean_diff_f[ix_func][ix_param][ix_strata],
                                                                                          n_samples[ix_strata],
                                                                                          mean_diff_delta,
                                                                                          n_samples_step[ix_strata])
                                self.vecsum_diff_f_adaptive[log_name][ix_func][ix_param][ix_strata] = mean_diff_f[ix_func][ix_param][ix_strata]
                                if grad_alloc:
                                    grad_norm_vals += diff_f_vals**2

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
            self.logged_allocs[log_name][self.t] = copy.deepcopy(rel_alloc)

        # Compute all sums of derivatives
        if compute_diff_f_sum:
                if n_func == 1:
                    for ix_param in range(n_param):
                        # Rescale means
                        for ix_strata in range(n_adaptive_strata):
                            self.vecsum_diff_f_adaptive[log_name][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                        # Sum over strata
                        self.sum_diff_f_adaptive[log_name][ix_param] = sum(self.vecsum_diff_f_adaptive[log_name][ix_param])
                else:
                    for ix_func in [1, 2]:
                        for ix_param in range(n_param[ix_func]):
                            # Rescale means
                            for ix_strata in range(n_adaptive_strata):
                                self.vecsum_diff_f_adaptive[log_name][ix_func][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                            # Sum over strata
                            self.sum_diff_f_adaptive[log_name][ix_func][ix_param] = sum(self.vecsum_diff_f_adaptive[log_name][ix_func][ix_param])

        # Compute sum of f
        if compute_f_sum:
            # Rescale means
            for ix_strata in range(n_adaptive_strata):
                self.vecsum_f_adaptive[log_name][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
            # Sum over strata
            self.sum_f_adaptive[log_name] = sum(self.vecsum_f_adaptive[log_name])

        # Save sampled indices if needed
        if self.is_log_ixs:
            samples = [[samples[ix_iter][ix_strata]
                        for ix_iter in range(n_adaptive_iters)]
                       for ix_strata in range(n_adaptive_strata)]
            self.logged_ixs[log_name]['adaptive'][self.t] = samples

    def estimate_sum_nonadaptivedom(self, strf, sampling_func, ixs_transform,
                                    get_n_samples, log_name, f=None,
                                    diff_f=None, n_func=1, n_param=None,
                                    f_args=None, diff_f_args=None,
                                    compute_f_sum=False,
                                    compute_diff_f_sum=True, rng=None,
                                    seed=None, coeff=1.):
        # If the stratification is empty, stop.
        if strf.n_nonadaptive_strata == 0:
            return

        # Arguments for f and diff_f
        if f_args is None:
            f_args_ = [{} for ix in range(n_func)]
        if diff_f_args is None:
            diff_f_args_ = [{} for ix in range(n_func)]

        # Computations to be made
        if compute_diff_f_sum and n_func > 1:
            compute_f_sum = True

        # rng initialisaiton
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
        samples = sampling_func(nonadaptive_strata, nonadaptive_abs_alloc,
                                rng=rng)
        if self.is_log_ixs:
            self.logged_ixs[log_name]['nonadaptive'][self.t] = samples

        for ix_strata in range(n_nonadaptive_strata):
            data_samples = ixs_transform(samples[ix_strata])
            # Compute estimate of sum f
            if compute_f_sum:
                n_samples_iter = get_n_samples(samples[ix_strata])
                f_args_ = dict(f_args, **data_samples)
                f_vals = coeff*f(**f_args_)
                self.vecsum_f_nonadaptive[log_name][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(f_vals)

            # Compute estimates of f derivatives
            if compute_diff_f_sum:
                    if n_func == 1:
                        for ix_param in range(n_param):
                            diff_f_args_ = dict(diff_f_args, **data_samples)
                            diff_f_args_['ix_diff'] = ix_param
                            diff_f_vals = coeff*diff_f(**diff_f_args_)
                            self.vecsum_diff_f_nonadaptive[log_name][ix_param][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(diff_f_vals)
                    else:
                        for ix_func in [1, 2]:
                            for ix_param in range(n_param[ix_func]):
                                diff_f_args_ = dict(diff_f_args,
                                                    **data_samples)
                                diff_f_args_['ix_diff'] = ix_param
                                diff_f_args_['ix_func'] = ix_func
                                diff_f_vals = coeff*diff_f(**diff_f_args_)
                                self.vecsum_diff_f_nonadaptive[log_name][ix_func][ix_param][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(diff_f_vals)


        # Sum over strata
        if compute_f_sum:
            self.sum_f_nonadaptive[log_name] = sum(self.vecsum_f_nonadaptive[log_name])
        if compute_diff_f_sum:
                if n_func == 1:
                    for ix_param in range(n_param):
                        self.sum_diff_f_nonadaptive[log_name][ix_param] = sum(self.vecsum_diff_f_nonadaptive[log_name][ix_param])
                else:
                    for ix_func in [1, 2]:
                        for ix_param in range(n_param[ix_func]):
                            self.sum_diff_f_nonadaptive[log_name][ix_func][ix_param] = sum(self.vecsum_diff_f_nonadaptive[log_name][ix_func][ix_param])


    def estimate_sum_exactdom(self, strf, ixs_transform, get_n_samples,
                              log_name, f=None, diff_f=None, n_func=1,
                              n_param=None, f_args=None, diff_f_args=None,
                              compute_f_sum=False, compute_diff_f_sum=True,
                              coeff=1.):
        # If the stratification is empty, stop.
        if strf.n_exact == 0:
            return

        # Arguments for f and diff_f
        if f_args is None:
            f_args = {}
        if diff_f_args is None:
            diff_f_args = {}

        # Computations to be made
        if compute_diff_f_sum and n_func > 1:
            compute_f_sum = True

        exact_ixs = strf.exact_domain
        data_samples = ixs_transform(exact_ixs)
        n_samples_iter = get_n_samples(exact_ixs)

        # Compute sum f
        if compute_f_sum:
            f_args_ = dict(f_args, **data_samples)
            f_vals = coeff*f(**f_args_)
            self.sum_f_exact[log_name] = np.sum(f_vals)

        # Compute sum f derivatives
        if compute_diff_f_sum:
            if n_func == 1:
                for ix_param in range(n_param):
                    diff_f_args_ = dict(diff_f_args, **data_samples)
                    diff_f_args_['ix_diff'] = ix_param
                    diff_f_vals = coeff*diff_f(**diff_f_args_)
                    self.sum_diff_f_exact[log_name][ix_param] = np.sum(diff_f_vals)
            else:
                for ix_func in [1, 2]:
                    for ix_param in range(n_param[ix_func]):
                        diff_f_args_ = dict(diff_f_args, **data_samples)
                        diff_f_args_['ix_diff'] = ix_param
                        diff_f_args_['ix_func'] = ix_func
                        diff_f_vals = coeff*diff_f(**diff_f_args_)
                        self.sum_diff_f_exact[log_name][ix_func][ix_param] = np.sum(diff_f_vals)

    def estimate_sum(self, strf, sampling_func, ixs_transform, get_n_samples,
                     log_name, f=None, diff_f=None, n_func=1, n_param=None,
                     f_args=None, diff_f_args=None, compute_f_sum=False,
                     compute_diff_f_sum=True, grad_alloc=True, rng=None,
                     seed=None, coeff=1., count_iter=False):
        # rng
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        # Estimates on the adaptive domain
        self.estimate_sum_adaptivedom(strf, sampling_func, ixs_transform,
                                      get_n_samples, log_name, f=f,
                                      diff_f=diff_f, n_func=n_func,
                                      n_param=n_param,
                                      f_args=f_args,
                                      diff_f_args=diff_f_args,
                                      compute_f_sum=compute_f_sum,
                                      compute_diff_f_sum=compute_diff_f_sum,
                                      grad_alloc=grad_alloc, rng=rng,
                                      seed=seed, coeff=coeff)

        # Estimates on the non-adaptive domain
        self.estimate_sum_nonadaptivedom(strf, sampling_func, ixs_transform,
                                         get_n_samples, log_name,
                                         f=f, diff_f=diff_f,
                                         n_func=n_func,
                                         n_param=n_param,
                                         f_args=f_args,
                                         diff_f_args=diff_f_args,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         rng=rng,
                                         seed=seed, coeff=coeff)
        # Exact domain
        self.estimate_sum_exactdom(strf, ixs_transform, get_n_samples,
                                   log_name, f=f, diff_f=diff_f, n_func=n_func,
                                   n_param=n_param, f_args=f_args,
                                   diff_f_args=diff_f_args,
                                   compute_f_sum=compute_f_sum,
                                   compute_diff_f_sum=compute_diff_f_sum,
                                   coeff=coeff)

        # Compute estimate of sum of f
        if compute_f_sum:
            self.sum_f[log_name] = (self.sum_f_adaptive[log_name]
                                    + self.sum_f_nonadaptive[log_name]
                                    + self.sum_f_exact[log_name])

        # Compute estimates of f derivatives
        if compute_diff_f_sum:
            if n_func == 1:
                for ix_param in range(n_param):
                    S = self.sum_diff_f_adaptive[log_name][ix_param]
                    S += self.sum_diff_f_nonadaptive[log_name][ix_param]
                    S += self.sum_diff_f_exact[log_name][ix_param]
                    self.sum_diff_f[log_name][ix_param] = S
            else:
                for ix_func in [1, 2]:
                    for ix_param in range(n_param[ix_func]):
                        S = self.sum_diff_f_adaptive[log_name][ix_func][ix_param]
                        S += self.sum_diff_f_nonadaptive[log_name][ix_func][ix_param]
                        S += self.sum_diff_f_exact[log_name][ix_func][ix_param]
                        self.sum_diff_f[log_name][ix_func][ix_param] = S

        # Iteration
        if count_iter:
            self.t += 1

# =============================================================================
# Recombine functionals
# =============================================================================
    def recombine_lse_k(self):
        d = self.d
        T_f = self.T_f

        res = 0.

        # Baseline terms
        res += self.mu_term

        # Baseline-kernel correlation
        for i in range(d):
            res += 2.*(self.sum_f['psi_'+str(i)]/T_f)

        # kernel-kernel correlation
        for i, j in itertools.product(range(d), range(d)):
            res += 2.*(self.sum_f['upsilon_'+str(i)+'_'+str(j)]/T_f)

        # Term 4 : Kernel terms
        for j in range(d):
            res -= 2.*(self.sum_f['phi_'+str(j)]/T_f)
        for i in range(d):
            res += self.sum_f['upsilonzero_'+str(i)]/T_f

        return res

    def recombine_diff_lse_mu_k(self):
        # Derivative of Z with respect to the rth parameter of mu_k
        d = self.d
        T_f = self.T_f

        # Mu term
        res = self.diff_mu_term

        # Psi term
        ix_func = 2
        r = 0
        for i in range(d):
            log_name = 'psi_'+str(i)
            res += 2.*(self.sum_diff_f[log_name][ix_func][r]/T_f)
        return res

    def recombine_diff_lse_theta_kpr(self, p, r):
        # Derivative of Z with respect to the rth parameter of phi_kp
        d = self.d
        T_f = self.T_f

        res = 0.

        # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
        for i in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(i)+'_'+str(p)
            ix_func = 2
            res += 2.*(self.sum_diff_f[log_name][ix_func][r]/T_f)

        # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
        for j in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(p)+'_'+str(j)
            ix_func = 1
            res += 2.*(self.sum_diff_f[log_name][ix_func][r]/T_f)

        # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilon_'+str(p)+'_'+str(p)
        res += 2.*(self.sum_diff_f[log_name][r]/T_f)

        # Term 4 : Phi_{kp}
        log_name = 'phi_'+str(p)
        res -= 2.*(self.sum_diff_f[log_name][r]/T_f)

        # Term 5 : psi_{kp}
        log_name = 'psi_'+str(p)
        ix_func = 1
        res += 2.*(self.sum_diff_f[log_name][ix_func][r]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilonzero_'+str(p)
        res += self.sum_diff_f[log_name][r]/T_f

        return res

    def recombine_grad_lse_k(self):
        d = self.d
        k = self.k
        grad = np.zeros(self.n_param_k)
        # Derivative with respect to baseline parameters
        grad[0] = self.recombine_diff_lse_mu_k()

        # Derivative with respect to kernel parameters
        for p in range(d):
            for ix in range(self.matrix_n_param[k][p]):
                ix_param = self.interval_map[k][p][0]+ix
                r = self.ix_map[k][ix_param]['par']
                grad[ix_param] = self.recombine_diff_lse_theta_kpr(p, r)
        return grad

# =============================================================================
# Compute derivatives
# =============================================================================
    def compute_mu_term(self, mu_k, compute_f_sum=False,
                        compute_diff_f_sum=True):
        k = self.k
        if compute_f_sum:
            self.mu_term = mu_k**2-2.*self.eta[k]*mu_k
        if compute_diff_f_sum:
            self.diff_mu_term = 2.*(mu_k-self.eta[k])

    def compute_psi_sum(self, i, x_ki, mu_k, compute_f_sum=False,
                        compute_diff_f_sum=True, grad_alloc=True, rng=None,
                        seed=None):
        k = self.k
        log_name = 'psi_'+str(i)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            # phi samples, ix_f = 0
            data_samples = {'t': self.list_times2end[i][ix_samples_stratum]}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            n_samples = len(ix_samples_stratum)
            return n_samples

        def K(t, params_ker, param_mu):
            res = param_mu*self.psi[k][i](t, params_ker)
            return res

        def diff_K(t, ix_func, ix_diff, params_ker, param_mu):
            if ix_func == 1:
                res = param_mu*self.diff_psi[k][i](t, ix_diff, params_ker)
            elif ix_func == 2:
                res = self.psi[k][i](t, params_ker)
            return res

        f = K
        diff_f = diff_K
        n_func = 2
        n_param = self.n_param[log_name]
        f_args = {'params_ker': x_ki, 'param_mu': mu_k}
        diff_f_args = {'params_ker': x_ki, 'param_mu': mu_k}
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, f=f, diff_f=diff_f,
                          n_func=n_func, n_param=n_param,
                          f_args=f_args,
                          diff_f_args=diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_upsilonzero_sum(self, i, x_ki, compute_f_sum=False,
                                compute_diff_f_sum=True, grad_alloc=True,
                                rng=None, seed=None):
        k = self.k
        log_name = 'upsilonzero_'+str(i)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)
            # Sampling
            ixs = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            data_samples = {'t': self.list_times2end[i][ix_samples_stratum],
                            's': 0.}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            n_samples = len(ix_samples_stratum)
            return n_samples

        log_name = 'upsilonzero_'+str(i)
        f = self.upsilon[i][i][k]
        diff_f = self.diff_sim_upsilon[k][i]
        n_func = 1

        # n_param
        n_param = self.n_param[log_name]

        # function args
        f_args = {'params_1': x_ki, 'params_2': x_ki}
        diff_f_args = {'params': x_ki}
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, f=f, diff_f=diff_f, n_func=n_func,
                          n_param=n_param, f_args=f_args,
                          diff_f_args=diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_phi_sum(self, i, x_ki, compute_f_sum=False,
                        compute_diff_f_sum=True, grad_alloc=True, rng=None,
                        seed=None):
        k = self.k
        log_name = 'phi_'+str(i)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_double_ix_stratified(k, i, strata,
                                                 abs_alloc, self.kappa,
                                                 self.varpi, self.n_events,
                                                 rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            ixs_n = ix_samples_stratum['n']
            # phi samples, ix_f = 0
            t_m = self.list_times[k][ixs_m]
            t_n = self.list_times[i][ixs_n]
            data_samples = {'t': t_m-t_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        log_name = 'phi_'+str(i)
        f = self.phi[k][i]
        diff_f = self.diff_phi[k][i]
        n_func = 1
        n_param = self.n_param[log_name]
        f_args = {'params': x_ki}
        diff_f_args = {'params': x_ki}
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, f=f, diff_f=diff_f, n_func=n_func,
                          n_param=n_param, f_args=f_args,
                          diff_f_args=diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_cross_upsilon_sum(self, i, j, x_ki, x_kj, compute_f_sum=False,
                                  compute_diff_f_sum=True, grad_alloc=True,
                                  rng=None, seed=None):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(j)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_double_ix_stratified(i, j, strata,
                                                 abs_alloc, self.kappa,
                                                 self.varpi, self.n_events,
                                                 rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            ixs_n = ix_samples_stratum['n']
            # phi samples, ix_f = 0
            t_m = self.list_times[i][ixs_m]
            t_n = self.list_times[j][ixs_n]
            data_samples = {'t': self.list_times2end[i][ixs_m], 's': t_m-t_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        log_name = 'upsilon_'+str(i)+'_'+str(j)
        f = self.upsilon[i][j][k]
        diff_f = self.diff_cross_upsilon[i][j][k]
        n_func = 2

        # n_param
        n_param = self.n_param[log_name]

        # function args
        f_args = {'params_1': x_ki, 'params_2': x_kj}
        diff_f_args = {'params_1': x_ki, 'params_2': x_kj}
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, f=f, diff_f=diff_f,
                          n_func=n_func, n_param=n_param,
                          f_args=f_args,
                          diff_f_args=diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_sim_upsilon_sum(self, i, x_ki,
                                compute_f_sum=False,
                                compute_diff_f_sum=True, grad_alloc=True,
                                rng=None, seed=None):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(i)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_double_ix_stratified(i, i, strata,
                                                 abs_alloc, self.kappa,
                                                 self.varpi, self.n_events,
                                                 rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            ixs_n = ix_samples_stratum['n']
            # phi samples, ix_f = 0
            t_m = self.list_times[i][ixs_m]
            t_n = self.list_times[i][ixs_n]
            data_samples = {'t': self.list_times2end[i][ixs_m], 's': t_m-t_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        f = self.upsilon[i][i][k]
        diff_f = self.diff_sim_upsilon[k][i]
        n_func = 1

        # n_param
        n_param = self.n_param[log_name]

        # function args
        f_args = {'params_1': x_ki, 'params_2': x_ki}
        diff_f_args = {'params': x_ki}
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, f=f, diff_f=diff_f, n_func=n_func,
                          n_param=n_param, f_args=f_args,
                          diff_f_args=diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_objective(self, x_k, compute_f_sum=False,
                          compute_diff_f_sum=True, grad_alloc=True, rng=None,
                          seed=None, count_iter=True):
        self.clear_estimates()
        # rng initialisation
        if rng is None:
            rng = np.random.default_rng(seed)

        d = self.d
        k = self.k
        mu_k = x_k[0]
        # mu term
        self.compute_mu_term(mu_k, compute_f_sum=compute_f_sum,
                             compute_diff_f_sum=compute_diff_f_sum)

        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            # K term
            self.compute_psi_sum(i, x_ki, mu_k,
                                 compute_f_sum=compute_f_sum,
                                 compute_diff_f_sum=compute_diff_f_sum,
                                 grad_alloc=grad_alloc, rng=rng)

            # Upsilonzero term
            self.compute_upsilonzero_sum(i, x_ki,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         grad_alloc=grad_alloc, rng=rng)

            # Phi term
            self.compute_phi_sum(i, x_ki,
                                 compute_f_sum=compute_f_sum,
                                 compute_diff_f_sum=compute_diff_f_sum,
                                 grad_alloc=grad_alloc, rng=rng)

            # Sim upsilon
            self.compute_sim_upsilon_sum(i, x_ki,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         grad_alloc=grad_alloc, rng=rng)
        for i, j in itertools.product(range(d), range(d)):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            if i != j:
                self.compute_cross_upsilon_sum(i, j, x_ki, x_kj,
                                               compute_f_sum=compute_f_sum,
                                               compute_diff_f_sum=compute_diff_f_sum,
                                               grad_alloc=grad_alloc, rng=rng)
        # Log sum estimates
        self.log_estimates()

        # Iteration
        if count_iter:
            self.t += 1

# =============================================================================
# LSE and LSE gradient
# =============================================================================
    def lse_k_estimate(self, x_k, compute_diff_f_sum=False, grad_alloc=True,
                       rng=None, seed=None, count_iter=True):
        # Compute the LSE estimate
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        self.compute_objective(x_k, compute_f_sum=True,
                               compute_diff_f_sum=compute_diff_f_sum,
                               grad_alloc=grad_alloc,
                               rng=rng, count_iter=count_iter)
        lse_k = self.recombine_lse_k()
        self.lse_k = lse_k
        return lse_k

    def lse_k_grad_estimate(self, x_k, grad_alloc=True, rng=None, seed=None,
                            count_iter=True):
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)
        self.compute_objective(x_k, compute_f_sum=self.is_log_lse,
                               compute_diff_f_sum=True, grad_alloc=grad_alloc,
                               rng=rng, count_iter=count_iter)

        grad = self.recombine_grad_lse_k()
        self.grad_lse_k = grad

        # Log LSE
        if self.is_log_lse:
            lse_k = self.recombine_lse_k()
            self.logged_lse[self.t] = lse_k
            self.lse_k = lse_k

        return grad
# =============================================================================
# Compute true optimal allocations
# =============================================================================
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

# =============================================================================
# Serialization
# =============================================================================
    def save(self, file, **kwargs):
        # Basic attributes
        dict_basic_attr = {}
        dict_basic_attr['is_grad_target'] = self.is_grad_target
        # Logging
        for attr_name in ['is_log_ixs', 'is_log_allocs',
                          'is_log_total_estimates', 'is_log_strata_estimates',
                          'is_log_lse']:
            dict_basic_attr[attr_name] = getattr(self, attr_name)
        # Params
        for attr_name in ['k', 'n_iter']:
            dict_basic_attr[attr_name] = getattr(self, attr_name)
        # Save file
        if file.endswith('.pickle'):
            file_basic_attr = file[:-7]+'_basic_attr.pickle'
        else:
            file_basic_attr = file+'_basic_attr'
        pickle_out = open(file_basic_attr, "wb", **kwargs)
        pickle.dump(dict_basic_attr, pickle_out)
        pickle_out.close()

        # Stratifications
        for log_name in self.list_log_names:
            if file.endswith('.pickle'):
                file_strfs = file[:-7]+'_strf_'+log_name+'.pickle'
            else:
                file_strfs = file+'_strf_'+log_name
            self.stratif[log_name].save(file_strfs)

        # Current state
        dict_current_state = {}
        for attr_name in ['t', 'mu_term', 'diff_mu_term', 'sum_f',
                          'vecsum_f_adaptive', 'sum_f_adaptive',
                          'vecsum_f_nonadaptive', 'sum_f_nonadaptive',
                          'sum_f_exact', 'sum_diff_f',
                          'vecsum_diff_f_adaptive', 'sum_diff_f_adaptive',
                          'vecsum_diff_f_nonadaptive',
                          'sum_diff_f_nonadaptive', 'sum_diff_f_exact']:
            dict_current_state[attr_name] = getattr(self, attr_name)
        # Save file
        if file.endswith('.pickle'):
            file_current_state = file[:-7]+'_current_state.pickle'
        else:
            file_current_state = file+'_current_state'
        pickle_out = open(file_current_state, "wb", **kwargs)
        pickle.dump(dict_current_state, pickle_out)
        pickle_out.close()

        # General logs
        dict_general_logs = {}
        if self.is_log_ixs:
            dict_general_logs['logged_ixs'] = self.logged_ixs
        if self.is_log_allocs:
            dict_general_logs['logged_allocs'] = self.logged_allocs
        if self.is_log_lse:
            dict_general_logs['logged_lse'] = self.logged_lse
        # Save file (if non-empty)
        if len(dict_general_logs) > 0:
            if file.endswith('.pickle'):
                file_general_logs = file[:-7]+'_general_logs.pickle'
            else:
                file_general_logs = file+'_general_logs'
            pickle_out = open(file_general_logs, "wb", **kwargs)
            pickle.dump(dict_general_logs, pickle_out)
            pickle_out.close()

        # Estimates logs
        dict_estimates_logs = {}
        if self.is_log_total_estimates:
            for attr_name in ['mu_term_log', 'diff_mu_term_log',
                              'sum_f_log', 'sum_diff_f_log']:
                dict_estimates_logs[attr_name] = getattr(self, attr_name)
        if self.is_log_strata_estimates:
            for attr_name in ['vecsum_f_adaptive_log', 'sum_f_adaptive_log',
                              'vecsum_f_nonadaptive_log',
                              'sum_f_nonadaptive_log',
                              'sum_f_exact_log', 'vecsum_diff_f_adaptive_log',
                              'sum_diff_f_adaptive_log',
                              'vecsum_diff_f_nonadaptive_log',
                              'sum_diff_f_nonadaptive_log',
                              'sum_diff_f_exact_log']:
                dict_estimates_logs[attr_name] = getattr(self, attr_name)
        # Save file (if non-empty)
        if len(dict_estimates_logs) > 0:
            if file.endswith('.pickle'):
                file_estimates_logs = file[:-7]+'_estimates_logs.pickle'
            else:
                file_estimates_logs = file+'_estimates_logs'
            pickle_out = open(file_estimates_logs, "wb", **kwargs)
            pickle.dump(dict_estimates_logs, pickle_out)
            pickle_out.close()

    def load(self, file, model, process_path, **kwargs):
        # Basic attributes
        # Load file
        if file.endswith('.pickle'):
            file_basic_attr = file[:-7]+'_basic_attr.pickle'
        else:
            file_basic_attr = file+'_basic_attr'
        pickle_in = open(file_basic_attr, "rb")
        dict_basic_attr = pickle.load(pickle_in)
        for attr_name in dict_basic_attr.keys():
            setattr(self, attr_name, dict_basic_attr[attr_name])

        # Initialization functions requiring model and data
        self.initialize_model_data(model, process_path)
        # Load stratifications
        self.stratif = {}
        for log_name in self.get_singlesum_lognames():
            if file.endswith('.pickle'):
                file_strfs = file[:-7]+'_strf_'+log_name+'.pickle'
            else:
                file_strfs = file+'_strf_'+log_name
            strf = GeneralStratification()
            strf.load(file_strfs)
            self.stratif[log_name] = strf
        for log_name in self.get_doublesum_lognames():
            if file.endswith('.pickle'):
                file_strfs = file[:-7]+'_strf_'+log_name+'.pickle'
            else:
                file_strfs = file+'_strf_'+log_name
            strf = DoubleIxStratification()
            strf.load(file_strfs)
            self.stratif[log_name] = strf
        # Log names
        self.list_log_names = self.get_list_log_names()
        # Param indexing
        self.make_param_indexing()

        # Current state
        # Load file
        if file.endswith('.pickle'):
            file_current_state = file[:-7]+'_current_state.pickle'
        else:
            file_current_state = file+'_current_state'
        pickle_in = open(file_current_state, "rb")
        dict_current_state = pickle.load(pickle_in)
        for attr_name in dict_current_state.keys():
            setattr(self, attr_name,  dict_current_state[attr_name])

        # General logs
        # Load file (if it exists)
        try:
            if file.endswith('.pickle'):
                file_general_logs = file[:-7]+'_general_logs.pickle'
            else:
                file_general_logs = file+'_general_logs'
            pickle_in = open(file_general_logs, "rb")
            dict_general_logs = pickle.load(pickle_in)
        except:
            pass
        # Indices
        if self.is_log_ixs:
            self.logged_ixs = dict_general_logs['logged_ixs']
        else:
            self.logged_ixs = {}
            for log_name in self.list_log_names:
                self.logged_ixs[log_name] = {}
                self.logged_ixs[log_name]['adaptive'] = None
                self.logged_ixs[log_name]['nonadaptive'] = None
        # Allocations
        if self.is_log_allocs:
            self.logged_allocs = dict_general_logs['logged_allocs']
        else:
            self.logged_allocs = {}
            for log_name in self.list_log_names:
                self.logged_allocs[log_name] = None
        # LSE
        if self.is_log_lse:
            self.logged_lse = dict_general_logs['logged_lse']

        # Estimates logs
        # Load file
        if file.endswith('.pickle'):
            file_estimates_logs = file[:-7]+'_estimates_logs.pickle'
        else:
            file_estimates_logs = file+'_estimates_logs'
        pickle_in = open(file_estimates_logs, "rb")
        dict_estimates_logs = pickle.load(pickle_in)
        for attr_name in dict_estimates_logs.keys():
            setattr(self, attr_name,  dict_estimates_logs[attr_name])
