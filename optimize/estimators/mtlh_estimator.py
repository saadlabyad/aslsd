# License: BSD 3 clause

import copy
import itertools

import numpy as np

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.stratifications.\
    double_ix_stratification import DoubleIxStratification
from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification
from aslsd.utilities import useful_statistics as us


class MtlhStratified(Estimator):
    def __init__(self, **kwargs):
        self.t = 0

        # Logging
        self.is_grad_target = kwargs.get('is_grad_target', False)

        # Logging
        self.is_log_lse = kwargs.get('is_log_lse', False)
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
        self.is_log_allocs = kwargs.get('is_log_allocs', False)
        self.is_log_sum_f = kwargs.get('is_log_sum_f', False)
        self.is_log_sum_diff_f = kwargs.get('is_log_sum_diff_f', False)

    # Stratification parameters
    def set_singlesum_estimation(self, list_times, **kwargs):
        d = self.d
        k = self.k

        # Mu
        mu_stratif = kwargs.get('mu_stratification', None)
        if mu_stratif is None:
            mu_stratif = GeneralStratification(list_times[k], **kwargs)
        self.stratif['mu'] = mu_stratif

        # K
        K_stratif = kwargs.get('K_stratification', [None for i in range(d)])
        for i in range(d):
            log_name = 'K_'+str(i)
            if K_stratif[i] is None:
                n_events_i = self.n_events[i]
                K_stratif[i] = GeneralStratification(list_times[k], **kwargs)
            self.stratif[log_name] = K_stratif[i]

        # upsilonzero
        upsilonzero_stratif = kwargs.get('upsilonzero_stratification',
                                         [None for i in range(d)])
        for i in range(d):
            log_name = 'upsilonzero_'+str(i)
            if upsilonzero_stratif[i] is None:
                n_events_i = self.n_events[i]
                upsilonzero_stratif[i] = GeneralStratification(list_times[k], **kwargs)
            self.stratif[log_name] = upsilonzero_stratif[i]

    def set_doublesum_estimation(self, **kwargs):
        d = self.d
        k = self.k

        # upsilon
        upsilon_stratif = kwargs.get('upsilon_stratification',
                                     [[None for j in range(d)]
                                      for i in range(d)])
        for i, j in itertools.product(range(d), range(d)):
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            if upsilon_stratif[i][j] is None:
                h_max = self.kappa[j][i][self.n_events[i]-1]+1
                upsilon_stratif[i][j] = DoubleIxStratification(h_max, **kwargs)
                upsilon_stratif[i][j].get_stratification_size(self.lag_sizes,
                                                              i, j)
            self.stratif[log_name] = upsilon_stratif[i][j]

        # phi
        phi_stratif = kwargs.get('phi_stratification', [None]*d)
        for i in range(d):
            log_name = 'phi_'+str(i)
            if phi_stratif[i] is None:
                h_max = self.kappa[i][k][self.n_events[k]-1]+1
                phi_stratif[i] = DoubleIxStratification(h_max, **kwargs)
                phi_stratif[i].get_stratification_size(self.lag_sizes, k, i)
            self.stratif[log_name] = phi_stratif[i]

    def set_stratification(self, list_times, **kwargs):
        self.stratif = {}
        self.set_singlesum_estimation(list_times, **kwargs)
        self.set_doublesum_estimation(**kwargs)
        d = self.d
        k = self.k

        # log names
        self.list_log_names = []
        self.list_log_names.append('mu')
        for i in range(d):
            self.list_log_names.append('phi_'+str(i))
            self.list_log_names.append('upsilonzero_'+str(i))
            self.list_log_names.append('K_'+str(i))
        for i, j in itertools.product(range(d), range(d)):
            self.list_log_names.append('upsilon_'+str(i)+'_'+str(j))

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
        self.sum_diff_f = {}

        # parameters indexing
        self.n_f = {}
        self.vec_n_func = {}
        self.list_n_param = {}
        # mu
        log_name = 'mu'
        self.n_f[log_name] = 1
        self.vec_n_func[log_name] = [None]
        self.list_n_param[log_name] = [self.vector_n_param_mu[k]]
        for i in range(d):
            # phi
            log_name = 'phi_'+str(i)
            self.n_f[log_name] = 2
            self.vec_n_func[log_name] = [None, None]
            self.list_n_param[log_name] = [self.matrix_n_param_ker[k][i],
                                           self.matrix_n_param_imp[k][i]]
            # upsilonzero
            log_name = 'upsilonzero_'+str(i)
            self.n_f[log_name] = 2
            self.vec_n_func[log_name] = [None, None]
            self.list_n_param[log_name] = [self.matrix_n_param_ker[k][i],
                                           self.matrix_n_param_imp[k][i]]
            # K
            log_name = 'K_'+str(i)
            self.n_f[log_name] = 2
            self.vec_n_func[log_name] = [2, None]
            self.list_n_param[log_name] = [None]*2
            self.list_n_param[log_name][0] = [None,
                                              self.matrix_n_param_ker[k][i],
                                              self.vector_n_param_mu[k]]
            self.list_n_param[log_name][1] = self.matrix_n_param_imp[k][i]
        for i, j in itertools.product(range(d), range(d)):
            # upsilon
            log_name = 'upsilon_'+str(i)+'_'+str(j)
            self.n_f[log_name] = 2
            if i == j:
                self.vec_n_func[log_name] = [None, None]
                self.list_n_param[log_name] = [self.matrix_n_param_ker[k][i],
                                               self.matrix_n_param_imp[k][i]]
            else:
                self.vec_n_func[log_name] = [2, 2]
                self.list_n_param[log_name] = [None]*2
                self.list_n_param[log_name][0] = [None,
                                                  self.matrix_n_param_ker[k][i],
                                                  self.matrix_n_param_ker[k][j]]
                self.list_n_param[log_name][1] = [None,
                                                  self.matrix_n_param_imp[k][i],
                                                  self.matrix_n_param_imp[k][j]]

    def make_dict_n_param(self):
        d = self.d
        k = self.k
        dict_n_param = {}

        # M term
        dict_n_param['M_term'] = self.vector_n_param_mu[k]
        # Sum Mu
        dict_n_param['sum_mu'] = self.vector_n_param_mu[k]
        # Sum Phi
        dict_n_param['sum_phi'] = [None for j in range(d)]
        for j in range(d):
            dict_n_param['sum_phi'][j]['ker'] = self.matrix_n_param_ker[k][j]
            dict_n_param['sum_phi'][j]['imp'] = self.matrix_n_param_imp[k][j]
        # Sum K
        dict_n_param['sum_K'] = [None for i in range(d)]
        for i in range(d):
                dict_n_param['sum_K'][i]['ker'] = [None,
                                                   self.matrix_n_param_ker[k][i],
                                                   self.vector_n_param_mu[k]]
                dict_n_param['sum_K'][i]['imp'] = self.matrix_n_param_imp[k][i]
        # Sum Upsilon
        dict_n_param['sum_upsilon'] = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            if i == j:
                dict_n_param['sum_upsilon'][i][i]['ker'] = self.matrix_n_param_ker[k][i]
                dict_n_param['sum_upsilon'][i][i]['imp'] = self.matrix_n_param_imp[k][i]
            else:
                dict_n_param['sum_upsilon'][i][j]['ker'] = [None,
                                                            self.matrix_n_param_ker[k][i],
                                                            self.matrix_n_param_ker[k][j]]
                dict_n_param['sum_upsilon'][i][j]['imp'] = [None,
                                                            self.matrix_n_param_imp[k][i],
                                                            self.matrix_n_param_imp[k][j]]
        # Sum Upsilonzero
        dict_n_param['sum_upsilonzero'] = [None for i in range(d)]
        for i in range(d):
            dict_n_param['sum_upsilonzero'][i]['ker'] = self.matrix_n_param_ker[k][i]
            dict_n_param['sum_upsilonzero'][i]['imp'] = self.matrix_n_param_imp[k][i]

        # Result
        self.dict_n_param = dict_n_param

    def make_null_sumarray(self, n_f, list_n_param, vec_n_func=None, n_strata=None):
        L = [None]*n_f
        for ix_f in range(n_f):
            if vec_n_func[ix_f] is None:
                n_param = list_n_param[ix_f]
                if n_strata is None:
                    L[ix_f] = np.zeros(n_param)
                else:
                    L[ix_f] = np.zeros((n_param, n_strata))
            else:
                L[ix_f] = [None]*3
                for ix_func in [1, 2]:
                    n_param = list_n_param[ix_f][ix_func]
                    if n_strata is None:
                        L[ix_f][ix_func] = np.zeros(n_param)
                    else:
                        L[ix_f][ix_func] = np.zeros((n_param, n_strata))
        return L

    def clear_estimates(self):
        k = self.k
        # M term
        self.M_term = 0.
        self.diff_M_term = np.zeros(self.vector_n_param_mu[k])

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
            n_f = self.n_f[log_name]
            vec_n_func = self.vec_n_func[log_name]
            list_n_param = self.list_n_param[log_name]
            # Total estimates
            self.sum_diff_f[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)
            # Local estimates
            self.vecsum_diff_f_adaptive[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=strf.n_adaptive_strata)
            self.sum_diff_f_adaptive[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)
            self.vecsum_diff_f_nonadaptive[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=strf.n_nonadaptive_strata)
            self.sum_diff_f_nonadaptive[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)
            self.sum_diff_f_exact[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)
            self.sum_diff_f[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)


    # Logging
    def intialize_logs(self):
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

    def get_log(self):
        logs = {}
        if self.is_log_lse:
            logs['lse'] = self.logged_lse

        if self.is_log_ixs:
            logs['samples'] = copy.deepcopy(self.logged_ixs)

        if self.is_log_allocs:
            logs['allocs'] = copy.deepcopy(self.logged_allocs)

        return logs

    # Upsilon Estimation
    def estimate_sum_adaptivedom(self, strf, sampling_func, ixs_transform,
                                 get_n_samples, log_name, list_f=None,
                                 list_diff_f=None, is_ix_func=None, n_func=1,
                                 list_n_param=None,
                                 list_f_args=None, list_diff_f_args=None,
                                 compute_f_sum=False, compute_diff_f_sum=True,
                                 grad_alloc=True, rng=None, seed=None,
                                 coeff=1.):
        if strf.n_adaptive_strata == 0:
            return
        if list_f_args is None:
            f_args = [{} for ix in range(n_func)]
        if list_diff_f_args is None:
            diff_f_args = [{} for ix in range(n_func)]

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
        for ix_f in range(n_func):
            if is_ix_func[ix_f]:
                mean_diff_f[ix_f] = [None]*3
                for ix_func in [1, 2]:
                    mean_diff_f[ix_f][ix_func] = [None]*list_n_param[ix_f][ix_func]
                    for ix_param in range(list_n_param[ix_f][ix_func]):
                        mean_diff_f[ix_f][ix_func][ix_param] = [None]*n_adaptive_strata
            else:
                mean_diff_f[ix_f] = [None]*list_n_param[ix_f]
                for ix_param in range(list_n_param[ix_f]):
                    mean_diff_f[ix_f][ix_param] = [None]*n_adaptive_strata

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
                list_f_vals = np.zeros((n_samples_iter, n_func))
                for ix_f in range(n_func):
                    f_args = dict(list_f_args[ix_f], **data_samples[ix_f])
                    list_f_vals[:, ix_f] = list_f[ix_f](**f_args)
                # Initialize satistics
                f_vals = coeff*np.prod(list_f_vals, axis=1)
                mean_f[ix_strata] = np.mean(f_vals)
                self.vecsum_f_adaptive[log_name][ix_strata] = mean_f[ix_strata]
            if compute_diff_f_sum:
                if grad_alloc:
                    grad_norm_vals = np.zeros(n_samples_iter)
                for ix_f in range(n_func):
                    if n_func == 1:
                        local_prod = 1.
                    else:
                        local_f_vals = np.delete(list_f_vals, ix_f, axis=1)
                        local_prod = np.prod(local_f_vals, axis=1)

                    if is_ix_func[ix_f]:
                        for ix_func in [1, 2]:
                            for ix_param in range(list_n_param[ix_f][ix_func]):
                                diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                                diff_f_args['ix_diff'] = ix_param
                                diff_f_args['ix_func'] = ix_func
                                diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                                mean_diff_f[ix_f][ix_func][ix_param][ix_strata] = np.mean(diff_f_vals)
                                self.vecsum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param][ix_strata] = mean_diff_f[ix_f][ix_func][ix_param][ix_strata]
                                if grad_alloc:
                                    grad_norm_vals += diff_f_vals**2
                    else:
                        for ix_param in range(list_n_param[ix_f]):
                            diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                            diff_f_args['ix_diff'] = ix_param
                            diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                            mean_diff_f[ix_f][ix_param][ix_strata] = np.mean(diff_f_vals)
                            self.vecsum_diff_f_adaptive[log_name][ix_f][ix_param][ix_strata] = mean_diff_f[ix_f][ix_param][ix_strata]
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
                    list_f_vals = np.zeros((n_samples_iter, n_func))
                    for ix_f in range(n_func):
                        f_args = dict(list_f_args[ix_f], **data_samples[ix_f])
                        list_f_vals[:, ix_f] = list_f[ix_f](**f_args)
                    # Initialize satistics
                    f_vals = coeff*np.prod(list_f_vals, axis=1)
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
                    for ix_f in range(n_func):
                        if n_func == 1:
                            local_prod = 1.
                        else:
                            local_f_vals = np.delete(list_f_vals, ix_f, axis=1)
                            local_prod = np.prod(local_f_vals, axis=1)
    
                        if is_ix_func[ix_f]:
                            for ix_func in [1, 2]:
                                for ix_param in range(list_n_param[ix_f][ix_func]):
                                    diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                                    diff_f_args['ix_diff'] = ix_param
                                    diff_f_args['ix_func'] = ix_func
                                    diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                                    mean_diff_delta = np.mean(diff_f_vals)
                                    mean_diff_f[ix_f][ix_func][ix_param][ix_strata] = us.batch_mean(mean_diff_f[ix_f][ix_func][ix_param][ix_strata],
                                                                                                    n_samples[ix_strata],
                                                                                                    mean_diff_delta,
                                                                                                    n_samples_step[ix_strata])
                                    self.vecsum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param][ix_strata] = mean_diff_f[ix_f][ix_func][ix_param][ix_strata]
                                    if grad_alloc:
                                        grad_norm_vals += diff_f_vals**2
                        else:
                            for ix_param in range(list_n_param[ix_f]):
                                diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                                diff_f_args['ix_diff'] = ix_param
                                diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                                mean_diff_delta = np.mean(diff_f_vals)
                                mean_diff_f[ix_f][ix_param][ix_strata] = us.batch_mean(mean_diff_f[ix_f][ix_param][ix_strata],
                                                                                       n_samples[ix_strata],
                                                                                       mean_diff_delta,
                                                                                       n_samples_step[ix_strata])
                                self.vecsum_diff_f_adaptive[log_name][ix_f][ix_param][ix_strata] = mean_diff_f[ix_f][ix_param][ix_strata]
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
            for ix_f in range(n_func):
                if is_ix_func[ix_f]:
                    for ix_func in [1, 2]:
                        for ix_param in range(list_n_param[ix_f][ix_func]):
                            # Rescale means
                            for ix_strata in range(n_adaptive_strata):
                                self.vecsum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                            # Sum over strata
                            self.sum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param] = sum(self.vecsum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param])
                else:
                    for ix_param in range(list_n_param[ix_f]):
                        # Rescale means
                        for ix_strata in range(n_adaptive_strata):
                            self.vecsum_diff_f_adaptive[log_name][ix_f][ix_param][ix_strata] *= strf.adaptive_strata_sizes[ix_strata]
                        # Sum over strata
                        self.sum_diff_f_adaptive[log_name][ix_f][ix_param] = sum(self.vecsum_diff_f_adaptive[log_name][ix_f][ix_param])
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
            # for ix_strata in range(n_adaptive_strata):
            #     print('ix_strata', ix_strata)
            #     print('samples[ix_strata]', samples[ix_strata])
            #     samples[ix_strata] = np.concatenate(samples[ix_strata])
            self.logged_ixs[log_name]['adaptive'][self.t] = samples

    def estimate_sum_nonadaptivedom(self, strf, sampling_func, ixs_transform,
                                    get_n_samples, log_name,
                                    list_f=None, list_diff_f=None,
                                    is_ix_func=None, n_func=1,
                                    list_n_param=None, list_f_args=None,
                                    list_diff_f_args=None,
                                    compute_f_sum=False,
                                    compute_diff_f_sum=True, rng=None,
                                    seed=None, coeff=1.):
        # If the stratification is empty, stop.
        if strf.n_nonadaptive_strata == 0:
            return

        # Arguments for f and diff_f
        if list_f_args is None:
            f_args = [{} for ix in range(n_func)]
        if list_diff_f_args is None:
            diff_f_args = [{} for ix in range(n_func)]

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
                list_f_vals = np.zeros((n_samples_iter, n_func))
                for ix_f in range(n_func):
                    f_args = dict(list_f_args[ix_f], **data_samples[ix_f])
                    list_f_vals[:, ix_f] = list_f[ix_f](**f_args)
                # Initialize satistics
                f_vals = coeff*np.prod(list_f_vals, axis=1)
                self.vecsum_f_nonadaptive[log_name][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(f_vals)

            # Compute estimates of f derivatives
            if compute_diff_f_sum:
                for ix_f in range(n_func):
                    if n_func == 1:
                        local_prod = 1.
                    else:
                        local_f_vals = np.delete(list_f_vals, ix_f, axis=1)
                        local_prod = np.prod(local_f_vals, axis=1)

                    if is_ix_func[ix_f]:
                        for ix_func in [1, 2]:
                            for ix_param in range(list_n_param[ix_f][ix_func]):
                                diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                                diff_f_args['ix_diff'] = ix_param
                                diff_f_args['ix_func'] = ix_func
                                diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                                self.vecsum_diff_f_nonadaptive[log_name][ix_f][ix_func][ix_param][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(diff_f_vals)
                    else:
                        for ix_param in range(list_n_param[ix_f]):
                            diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                            diff_f_args['ix_diff'] = ix_param
                            diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                            self.vecsum_diff_f_nonadaptive[log_name][ix_f][ix_param][ix_strata] = nonadaptive_strata_sizes[ix_strata]*np.mean(diff_f_vals)

        # Sum over strata
        if compute_f_sum:
            self.sum_f_nonadaptive[log_name] = sum(self.vecsum_f_nonadaptive[log_name])
        if compute_diff_f_sum:
            for ix_f in range(n_func):
                if is_ix_func[ix_f]:
                    for ix_func in [1, 2]:
                        for ix_param in range(list_n_param[ix_f][ix_func]):
                            self.sum_diff_f_nonadaptive[log_name][ix_f][ix_func][ix_param] = sum(self.vecsum_diff_f_nonadaptive[log_name][ix_f][ix_func][ix_param])
                else:
                    for ix_param in range(list_n_param[ix_f]):
                        self.sum_diff_f_nonadaptive[log_name][ix_f][ix_param] = sum(self.vecsum_diff_f_nonadaptive[log_name][ix_f][ix_param])

    def estimate_sum_exactdom(self, strf, ixs_transform, get_n_samples,
                              log_name,
                              list_f=None, list_diff_f=None,
                              is_ix_func=None, n_func=1,
                              list_n_param=None, list_f_args=None,
                              list_diff_f_args=None,
                              compute_f_sum=False,
                              compute_diff_f_sum=True, coeff=1.):
        # If the stratification is empty, stop.
        if strf.n_exact == 0:
            return

        # Arguments for f and diff_f
        if list_f_args is None:
            f_args = [{} for ix in range(n_func)]
        if list_diff_f_args is None:
            diff_f_args = [{} for ix in range(n_func)]

        # Computations to be made
        if compute_diff_f_sum and n_func > 1:
            compute_f_sum = True

        exact_ixs = strf.exact_domain
        data_samples = ixs_transform(exact_ixs)
        n_samples_iter = get_n_samples(exact_ixs)

        # Compute sum f
        if compute_f_sum:
            list_f_vals = np.zeros((n_samples_iter, n_func))
            for ix_f in range(n_func):
                f_args = dict(list_f_args[ix_f], **data_samples[ix_f])
                list_f_vals[:, ix_f] = list_f[ix_f](**f_args)
            f_vals = coeff*np.prod(list_f_vals, axis=1)
            self.sum_f_exact[log_name] = np.sum(f_vals)

        # Compute sum f derivatives
        if compute_diff_f_sum:
            for ix_f in range(n_func):
                if n_func == 1:
                    local_prod = 1.
                else:
                    local_f_vals = np.delete(list_f_vals, ix_f, axis=1)
                    local_prod = np.prod(local_f_vals, axis=1)

                if is_ix_func[ix_f]:
                    for ix_func in [1, 2]:
                        for ix_param in range(list_n_param[ix_f][ix_func]):
                            diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                            diff_f_args['ix_diff'] = ix_param
                            diff_f_args['ix_func'] = ix_func
                            diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                            self.sum_diff_f_exact[log_name][ix_f][ix_func][ix_param] = np.sum(diff_f_vals)
                else:
                    for ix_param in range(list_n_param[ix_f]):
                        diff_f_args = dict(list_diff_f_args[ix_f], **data_samples[ix_f])
                        diff_f_args['ix_diff'] = ix_param
                        diff_f_vals = coeff*list_diff_f[ix_f](**diff_f_args)*local_prod
                        self.sum_diff_f_exact[log_name][ix_f][ix_param] = np.sum(diff_f_vals)

    def estimate_sum(self, strf, sampling_func, ixs_transform,
                     get_n_samples, log_name, list_f=None,
                     list_diff_f=None, is_ix_func=None, n_func=1,
                     list_n_param=None,
                     list_f_args=None, list_diff_f_args=None,
                     compute_f_sum=False, compute_diff_f_sum=True,
                     grad_alloc=True, rng=None, seed=None,
                     coeff=1., count_iter=False):
        # rng
        if rng is None:
            if seed is None:
                rng = np.random.default_rng()
            else:
                rng = np.random.default_rng(seed)

        # Estimates on the adaptive domain
        self.estimate_sum_adaptivedom(strf, sampling_func, ixs_transform,
                                      get_n_samples,
                                      log_name, list_f=list_f,
                                      list_diff_f=list_diff_f, is_ix_func=is_ix_func, n_func=n_func,
                                      list_n_param=list_n_param,
                                      list_f_args=list_f_args, list_diff_f_args=list_diff_f_args,
                                      compute_f_sum=compute_f_sum, compute_diff_f_sum=compute_diff_f_sum,
                                      grad_alloc=grad_alloc, rng=rng, seed=seed,
                                      coeff=coeff)

        # Estimates on the non-adaptive domain
        self.estimate_sum_nonadaptivedom(strf, sampling_func, ixs_transform,
                                         get_n_samples, log_name,
                                         list_f=list_f, list_diff_f=list_diff_f,
                                         is_ix_func=is_ix_func, n_func=n_func,
                                         list_n_param=list_n_param,
                                         list_f_args=list_f_args,
                                         list_diff_f_args=list_diff_f_args,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         rng=rng,
                                         seed=seed, coeff=coeff)
        # Exact domain
        self.estimate_sum_exactdom(strf, ixs_transform, get_n_samples, log_name,
                                   list_f=list_f, list_diff_f=list_diff_f,
                                   is_ix_func=is_ix_func, n_func=n_func,
                                   list_n_param=list_n_param,
                                   list_f_args=list_f_args,
                                   list_diff_f_args=list_diff_f_args,
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
            for ix_f in range(n_func):
                if is_ix_func[ix_f]:
                    for ix_func in [1, 2]:
                        for ix_param in range(list_n_param[ix_f][ix_func]):
                            S = self.sum_diff_f_adaptive[log_name][ix_f][ix_func][ix_param]
                            S += self.sum_diff_f_nonadaptive[log_name][ix_f][ix_func][ix_param]
                            S += self.sum_diff_f_exact[log_name][ix_f][ix_func][ix_param]
                            self.sum_diff_f[log_name][ix_f][ix_func][ix_param] = S
                else:
                    for ix_param in range(list_n_param[ix_f]):
                        for ix_param in range(list_n_param[ix_f]):
                            S = self.sum_diff_f_adaptive[log_name][ix_f][ix_param]
                            S += self.sum_diff_f_nonadaptive[log_name][ix_f][ix_param]
                            S += self.sum_diff_f_exact[log_name][ix_f][ix_param]
                            self.sum_diff_f[log_name][ix_f][ix_param] = S

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

        # baseline terms
        res += self.M_term-2.*(self.sum_f['mu']/T_f)
        for i in range(d):
            res += 2.*(self.sum_f['K_'+str(i)]/T_f)

        # kernel-impact terms
        for i, j in itertools.product(range(d), range(d)):
            res += 2.*(self.sum_f['upsilon_'+str(i)+'_'+str(j)]/T_f)

        # Term 4 : Phi_{kp}
        for j in range(d):
            res -= 2.*(self.sum_f['phi_'+str(j)]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        for i in range(d):
            res += self.sum_f['upsilonzero_'+str(i)]/T_f

        return res

    def recombine_diff_lse_mu_kr(self, r):
        # Derivative of Z with respect to the rth parameter of mu_k
        d = self.d
        T_f = self.T_f
        res = self.diff_M_term[r]

        # Mu term
        res -= 2.*(self.sum_diff_f['mu'][0][r]/T_f)

        # K term
        ix_f = 0
        ix_func = 2
        for i in range(d):
            log_name = 'K_'+str(i)
            res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)
        return res

    def recombine_diff_lse_theta_kpr_kernel(self, p, r):
        # Derivative of Z with respect to the rth parameter of phi_kp
        d = self.d
        T_f = self.T_f

        res = 0.

        # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
        for i in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(i)+'_'+str(p)
            ix_f = 0
            ix_func = 2
            res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)

        # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
        for j in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(p)+'_'+str(j)
            ix_f = 0
            ix_func = 1
            res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)

        # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilon_'+str(p)+'_'+str(p)
        ix_f = 0
        res += 2.*(self.sum_diff_f[log_name][ix_f][r]/T_f)

        # Term 4 : Phi_{kp}
        log_name = 'phi_'+str(p)
        ix_f = 0
        res -= 2.*(self.sum_diff_f[log_name][ix_f][r]/T_f)

        # Term 5 : K_{kp}
        log_name = 'K_'+str(p)
        ix_f = 0
        ix_func = 1
        res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilonzero_'+str(p)
        ix_f = 0
        res += self.sum_diff_f[log_name][ix_f][r]/T_f

        return res

    def recombine_diff_lse_theta_kpr_impact(self, p, r):
        # Derivative of Z with respect to the rth parameter of I_kp
        d = self.d
        T_f = self.T_f

        res = 0.

        # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
        for i in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(i)+'_'+str(p)
            ix_f = 1
            ix_func = 2
            res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)

        # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
        for j in itertools.chain(range(p), range(p+1, d)):
            log_name = 'upsilon_'+str(p)+'_'+str(j)
            ix_f = 1
            ix_func = 1
            res += 2.*(self.sum_diff_f[log_name][ix_f][ix_func][r]/T_f)

        # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilon_'+str(p)+'_'+str(p)
        ix_f = 1
        res += 2.*(self.sum_diff_f[log_name][ix_f][r]/T_f)

        # Term 4 : Phi_{kp}
        log_name = 'phi_'+str(p)
        ix_f = 1
        res -= 2.*(self.sum_diff_f[log_name][ix_f][r]/T_f)

        # Term 5 : K_{kp}
        log_name = 'K_'+str(p)
        ix_f = 1
        res += 2.*(self.sum_diff_f[log_name][ix_f][r]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        log_name = 'upsilonzero_'+str(p)
        ix_f = 1
        res += self.sum_diff_f[log_name][ix_f][r]/T_f

        return res

    def recombine_grad_lse_k(self):
        d = self.d
        k = self.k
        grad = np.zeros(self.n_param_k)
        # Derivative with respect to baseline parameters
        for ix_param in range(self.vector_n_param_mu[k]):
            grad[ix_param] = self.recombine_diff_lse_mu_kr(ix_param)

        # Derivative with respect to kernel parameters
        for p in range(d):
            for ix_param in range(self.interval_map_ker[k][p][0],
                                  self.interval_map_ker[k][p][1]):
                r = self.ix_map_ker[k][ix_param]['par']
                grad[ix_param] = self.recombine_diff_lse_theta_kpr_kernel(p, r)

        # Derivative with respect to impact parameters
        for p in range(d):
            for ix_param in range(self.interval_map_imp[k][p][0],
                                  self.interval_map_imp[k][p][1]):
                r = self.ix_map_imp[k][ix_param]['par']
                grad[ix_param] = self.recombine_diff_lse_theta_kpr_impact(p, r)

        return grad

# =============================================================================
# Compute derivatives
# =============================================================================
    def compute_M_term(self, x_mu_k, compute_f_sum=False,
                       compute_diff_f_sum=True):
        k = self.k
        T_f = self.T_f
        if compute_f_sum:
            self.M_term = self.M[k](T_f, x_mu_k)
        if compute_diff_f_sum:
            for ix_diff in range(len(x_mu_k)):
                self.diff_M_term[ix_diff] = self.diff_M[k](T_f, ix_diff,
                                                           x_mu_k)

    def compute_mu_sum(self, x_mu_k, compute_f_sum=False,
                       compute_diff_f_sum=True, grad_alloc=True, rng=None,
                       seed=None):
        k = self.k
        strf = self.stratif['mu']

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            data_samples = [None]
            # phi samples, ix_f = 0
            t_m = self.list_times[k][ix_samples_stratum]
            data_samples[0] = {'t': t_m}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            n_samples = len(ix_samples_stratum)
            return n_samples

        log_name = 'mu'
        list_f = [self.mu[k]]
        list_diff_f = [self.diff_mu[k]]
        is_ix_func = [False]
        n_func = 1
        list_n_param = self.list_n_param[log_name]
        list_f_args = [{'params': x_mu_k}]
        list_diff_f_args = [{'params': x_mu_k}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_K_sum(self, i, x_ker_ki, x_mu_k, x_imp_ki, compute_f_sum=False,
                      compute_diff_f_sum=True, grad_alloc=True, rng=None,
                      seed=None):
        k = self.k
        log_name = 'K_'+str(i)
        strf = self.stratif[log_name]

        def sampling_func(strata, abs_alloc, rng=None, seed=1234):
            # rng initialisation
            if rng is None:
                rng = np.random.default_rng(seed)

            # Sampling
            ixs = us.sample_single_ix_stratified(strata, abs_alloc, rng=rng)
            return ixs

        def ixs_transform(ix_samples_stratum):
            data_samples = [None]*2
            # phi samples, ix_f = 0
            t_m = self.list_times[i][ix_samples_stratum]
            data_samples[0] = {'t': self.list_times2end[i][ix_samples_stratum],
                               's': t_m}
            # Imp samples, ix_f = 1
            zeta_m = self.list_marks[i][ix_samples_stratum]
            data_samples[1] = {'zeta': zeta_m}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            n_samples = len(ix_samples_stratum)
            return n_samples

        list_f = [self.K[k][i], self.impact[k][i]]
        list_diff_f = [self.diff_K[k][i], self.diff_impact[k][i]]
        is_ix_func = [True, False]
        n_func = 2
        list_n_param = self.list_n_param[log_name]
        list_f_args = [{'params_ker': x_ker_ki, 'params_mu': x_mu_k},
                       {'params': x_imp_ki}]
        list_diff_f_args = [{'params_ker': x_ker_ki, 'params_mu': x_mu_k},
                            {'params': x_imp_ki}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_upsilonzero_sum(self, i, x_ker_ki, x_imp_ki,
                                compute_f_sum=False,
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
            data_samples = [None]*2
            # phi samples, ix_f = 0
            data_samples[0] = {'t': self.list_times2end[i][ix_samples_stratum],
                               's': 0.}
            # Imp samples, ix_f = 1
            zeta = self.list_marks[i][ix_samples_stratum]
            data_samples[1] = {'zeta': zeta}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            n_samples = len(ix_samples_stratum)
            return n_samples

        # Impact function
        def mixed_impact(zeta, params):
            imp = self.impact[k][i](zeta, params)
            return imp**2

        def diff_mixed_impact(zeta, ix_diff, params):
            imp = self.impact[k][i](zeta, params)
            diff_imp = self.diff_impact[k][i](zeta, ix_diff, params)
            return 2.*imp*diff_imp

        log_name = 'upsilonzero_'+str(i)
        list_f = [self.upsilon[i][i][k], mixed_impact]
        list_diff_f = [self.diff_sim_upsilon[k][i], diff_mixed_impact]
        is_ix_func = [False, False]
        n_func = 2

        # n_param
        list_n_param = self.list_n_param[log_name]

        # function args
        list_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_ki},
                       {'params': x_imp_ki}]
        list_diff_f_args = [{'params': x_ker_ki},
                            {'params': x_imp_ki}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_phi_sum(self, i, x_ker_ki, x_imp_ki, compute_f_sum=False,
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
            data_samples = [None]*2
            # phi samples, ix_f = 0
            t_m = self.list_times[k][ixs_m]
            t_n = self.list_times[i][ixs_n]
            data_samples[0] = {'t': t_m-t_n}
            # Imp samples, ix_f = 1
            zeta_n = self.list_marks[i][ixs_n]
            data_samples[1] = {'zeta': zeta_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        log_name = 'phi_'+str(i)
        list_f = [self.phi[k][i], self.impact[k][i]]
        list_diff_f = [self.diff_phi[k][i], self.diff_impact[k][i]]
        is_ix_func = [False, False]
        n_func = 2
        list_n_param = self.list_n_param[log_name]
        list_f_args = [{'params': x_ker_ki}, {'params': x_imp_ki}]
        list_diff_f_args = [{'params': x_ker_ki}, {'params': x_imp_ki}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_cross_upsilon_sum(self, i, j, x_ker_ki, x_ker_kj, x_imp_ki,
                                  x_imp_kj, compute_f_sum=False,
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
            data_samples = [None]*2
            # phi samples, ix_f = 0
            t_m = self.list_times[i][ixs_m]
            t_n = self.list_times[j][ixs_n]
            data_samples[0] = {'t': self.list_times2end[i][ixs_m],
                               's': t_m-t_n}
            # Imp samples, ix_f = 1
            zeta_m = self.list_marks[i][ixs_m]
            zeta_n = self.list_marks[j][ixs_n]
            data_samples[1] = {'zeta_1': zeta_m, 'zeta_2': zeta_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        # Impact function
        def mixed_impact(zeta_1, zeta_2, params_1, params_2):
            imp_1 = self.impact[k][i](zeta_1, params_1)
            imp_2 = self.impact[k][j](zeta_2, params_2)
            return imp_1*imp_2

        def diff_mixed_impact(zeta_1, zeta_2, ix_func, ix_diff, params_1,
                              params_2):
            if ix_func == 1:
                diff_imp_1 = self.diff_impact[k][i](zeta_1, ix_diff, params_1)
                imp_2 = self.impact[k][j](zeta_2, params_2)
                return diff_imp_1*imp_2
            elif ix_func == 1:
                imp_1 = self.impact[k][i](zeta_1, params_1)
                diff_imp_2 = self.diff_impact[k][j](zeta_2, ix_diff, params_2)
                return imp_1*diff_imp_2

        log_name = 'upsilon_'+str(i)+'_'+str(j)
        list_f = [self.upsilon[i][j][k], mixed_impact]
        list_diff_f = [self.diff_cross_upsilon[i][j][k], diff_mixed_impact]
        is_ix_func = [True, True]
        n_func = 2

        # n_param
        list_n_param = self.list_n_param[log_name]

        # function args
        list_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_kj},
                       {'params_1': x_imp_ki, 'params_2': x_imp_kj}]
        list_diff_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_kj},
                            {'params_1': x_imp_ki, 'params_2': x_imp_kj}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
                          compute_f_sum=compute_f_sum,
                          compute_diff_f_sum=compute_diff_f_sum,
                          grad_alloc=grad_alloc, rng=rng, seed=seed,
                          coeff=1., count_iter=False)

    def compute_sim_upsilon_sum(self, i, x_ker_ki, x_imp_ki,
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
            data_samples = [None]*2
            # phi samples, ix_f = 0
            t_m = self.list_times[i][ixs_m]
            t_n = self.list_times[i][ixs_n]
            data_samples[0] = {'t': self.list_times2end[i][ixs_m],
                               's': t_m-t_n}
            # Imp samples, ix_f = 1
            zeta_m = self.list_marks[i][ixs_m]
            zeta_n = self.list_marks[i][ixs_n]
            data_samples[1] = {'zeta_1': zeta_m, 'zeta_2': zeta_n}
            return data_samples

        def get_n_samples(ix_samples_stratum):
            ixs_m = ix_samples_stratum['m']
            n_samples = len(ixs_m)
            return n_samples

        # Impact function
        def mixed_impact(zeta_1, zeta_2, params):
            imp_1 = self.impact[k][i](zeta_1, params)
            imp_2 = self.impact[k][i](zeta_2, params)
            return imp_1*imp_2

        def diff_mixed_impact(zeta_1, zeta_2, ix_diff, params):
            imp_1 = self.impact[k][i](zeta_1, params)
            imp_2 = self.impact[k][i](zeta_2, params)
            diff_imp_1 = self.diff_impact[k][i](zeta_1, ix_diff, params)
            diff_imp_2 = self.diff_impact[k][i](zeta_2, ix_diff, params)
            return imp_1*diff_imp_2+diff_imp_1*imp_2

        list_f = [self.upsilon[i][i][k], mixed_impact]
        list_diff_f = [self.diff_sim_upsilon[k][i], diff_mixed_impact]
        is_ix_func = [False, False]
        n_func = 2

        # n_param
        list_n_param = self.list_n_param[log_name]

        # function args
        list_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_ki},
                       {'params': x_imp_ki}]
        list_diff_f_args = [{'params': x_ker_ki},
                            {'params': x_imp_ki}]
        self.estimate_sum(strf, sampling_func, ixs_transform, get_n_samples,
                          log_name, list_f=list_f,
                          list_diff_f=list_diff_f, is_ix_func=is_ix_func,
                          n_func=n_func, list_n_param=list_n_param,
                          list_f_args=list_f_args,
                          list_diff_f_args=list_diff_f_args,
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
        x_mu_k, x_ker_k, x_imp_k = self.xk2matrix_params(k, x_k)
        # M term
        self.compute_M_term(x_mu_k, compute_f_sum=compute_f_sum,
                            compute_diff_f_sum=compute_diff_f_sum)
        # Mu term
        self.compute_mu_sum(x_mu_k, compute_f_sum=compute_f_sum,
                            compute_diff_f_sum=compute_diff_f_sum,
                            grad_alloc=grad_alloc, rng=rng)

        for i in range(d):
            # K term
            self.compute_K_sum(i, x_ker_k[i], x_mu_k, x_imp_k[i],
                               compute_f_sum=compute_f_sum,
                               compute_diff_f_sum=compute_diff_f_sum,
                               grad_alloc=grad_alloc, rng=rng)

            # Upsilonzero term
            self.compute_upsilonzero_sum(i, x_ker_k[i], x_imp_k[i],
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         grad_alloc=grad_alloc, rng=rng)

            # Phi term
            self.compute_phi_sum(i, x_ker_k[i], x_imp_k[i],
                                 compute_f_sum=compute_f_sum,
                                 compute_diff_f_sum=compute_diff_f_sum,
                                 grad_alloc=grad_alloc, rng=rng)

            # Sim upsilon
            self.compute_sim_upsilon_sum(i, x_ker_k[i], x_imp_k[i],
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         grad_alloc=grad_alloc, rng=rng)
        for i, j in itertools.product(range(d), range(d)):
            if i != j:
                self.compute_cross_upsilon_sum(i, j, x_ker_k[i], x_ker_k[j],
                                               x_imp_k[i], x_imp_k[j],
                                               compute_f_sum=compute_f_sum,
                                               compute_diff_f_sum=compute_diff_f_sum,
                                               grad_alloc=grad_alloc, rng=rng)
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
        lse = self.recombine_lse_k()
        return lse

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

        # Log LSE
        if self.is_log_lse:
            lse = self.recombine_lse_k()
            self.logged_lse[self.t] = lse

        return grad

# =============================================================================
# True Optimal allocations
# =============================================================================
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
