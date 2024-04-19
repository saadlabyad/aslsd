# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification
from aslsd.optimize.stratifications.\
    double_ix_stratification import DoubleIxStratification


class MHPExactEstim(Estimator):
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
        self.sum_f = {}
        # Gradient
        self.sum_diff_f = {}

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
        # M term
        dict_n_param['M_term'] = 1
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
        # Mu term
        self.mu_term = 0.
        self.diff_mu_term = 0.

        # Sum terms
        for log_name in self.list_log_names:
            # Sum f
            # Total estimates
            self.sum_f[log_name] = 0.

            # Gradient
            n_func = self.n_func[log_name]
            n_param = self.n_param[log_name]
            # Total estimates
            self.sum_diff_f[log_name] = self.make_null_sumarray(
                n_param, n_func=n_func, n_strata=None)

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

    def log_estimates(self):
        ix_iter = self.t
        if self.is_log_total_estimates:
            # mu term
            self.mu_term_log[ix_iter] = self.mu_term
            self.diff_mu_term_log[ix_iter] = self.diff_mu_term

            # Sum terms
            for log_name in self.list_log_names:
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
# Recombine functionals
# =============================================================================
    def recombine_lse_k(self):
        d = self.d
        T_f = self.T_f

        res = 0.

        # baseline terms
        res += self.mu_term
        for i in range(d):
            res += 2.*(self.sum_f['psi_'+str(i)]/T_f)

        # kernel terms
        for i, j in itertools.product(range(d), range(d)):
            res += 2.*(self.sum_f['upsilon_'+str(i)+'_'+str(j)]/T_f)

        # Term 4 : Phi_{kp}
        for j in range(d):
            res -= 2.*(self.sum_f['phi_'+str(j)]/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        for i in range(d):
            res += self.sum_f['upsilonzero_'+str(i)]/T_f

        return res

    def recombine_diff_lse_mu_k(self):
        # Derivative of Z with respect to  mu_k
        d = self.d
        T_f = self.T_f
        res = self.diff_mu_term

        # psi term
        ix_func = 2
        for i in range(d):
            log_name = 'psi_'+str(i)
            res += 2.*(self.sum_diff_f[log_name][ix_func][0]/T_f)
        return res

    def recombine_diff_lse_theta_kpr_kernel(self, p, r):
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
        # Derivative with respect to the baseline parameters
        grad[0] = self.recombine_diff_lse_mu_k()

        # Derivative with respect to kernel parameters
        for p in range(d):
            for ix_param in range(1, self.n_param_k):
                r = self.ix_map[k][ix_param]['par']
                grad[ix_param] = self.recombine_diff_lse_theta_kpr_kernel(p, r)
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
                        compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'psi_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # Data
        t = self.list_times2end[i]

        n_param = self.n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Compute sum f
        if compute_f_sum:
            # Psi values
            f_vals = self.psi[k][i](t, x_ki)
            sum_f_vals = np.sum(f_vals)
            # Sum
            self.sum_f[log_name] = mu_k*sum_f_vals

        # Compute sum f derivatives
        if compute_diff_f_sum:
            # Derivatives wrt kernel parameters
            ix_func = 1
            for ix_param in range(n_param[ix_func]):
                diff_f_vals = self.diff_psi[k][i](t, ix_param, x_ki)
                self.sum_diff_f[log_name][ix_func][ix_param] = mu_k*np.sum(diff_f_vals)

            # Derivatives wrt mu
            ix_func = 2
            ix_param = 0
            self.sum_diff_f[log_name][ix_func][ix_param] = sum_f_vals

    def compute_upsilonzero_sum(self, i, x_ki,
                                compute_f_sum=False,
                                compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilonzero_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # Data
        t = self.list_times2end[i]

        # n_param
        n_param = self.n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Compute sum f
        if compute_f_sum:
            # Upsilon values
            f_vals = self.upsilon[i][i][k](t, 0., x_ki, x_ki)
            self.sum_f[log_name] = np.sum(f_vals)
        # Compute sum f derivatives
        if compute_diff_f_sum:
            # Upsilon values
            for ix_param in range(n_param):
                diff_f_vals = self.diff_sim_upsilon[k][i](t, 0., ix_param,
                                                          x_ki)
                self.sum_diff_f[log_name][ix_param] = np.sum(diff_f_vals)

    def compute_phi_sum(self, i, x_ki, compute_f_sum=False,
                        compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'phi_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        kappa = self.kappa
        varpi = self.varpi

        n_param = self.n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        for m in tqdm(range(varpi[k][i][1], len(self.list_times[k])),
                      disable=not verbose):
            t_m = self.list_times[k][m]
            t_n = self.list_times[i][:kappa[i][k][m]+1]
            diff_times = t_m-t_n
            phi_vals = self.phi[k][i](diff_times, x_ki)
            self.sum_f[log_name] += np.sum(phi_vals)
            if compute_diff_f_sum:
                for ix_param in range(n_param):
                    diff_f_vals = self.diff_phi[k][i](diff_times, ix_param,
                                                      x_ki)
                    self.sum_diff_f[log_name][ix_param] += np.sum(diff_f_vals)

    def compute_cross_upsilon_sum(self, i, j, x_ki, x_kj, compute_f_sum=False,
                                  compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(j)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # n_param
        n_param = self.n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Data
        kappa = self.kappa
        varpi = self.varpi

        # Computations
        for m in tqdm(range(varpi[i][j][1], len(self.list_times[i])),
                      disable=not verbose):
            t_m = self.list_times[i][m]
            t_n = self.list_times[j][:kappa[j][i][m]+1]
            n_vals = len(t_n)
            diff_times = t_m-t_n
            t = self.list_times2end[i][m]*np.ones(n_vals)
            ups_vals = self.upsilon[i][j][k](t, diff_times, x_ki, x_kj)
            self.sum_f[log_name] += np.sum(ups_vals)
            if compute_diff_f_sum:
                # Kernel term
                for ix_func in [1, 2]:
                    for ix_param in range(n_param[ix_func]):
                        diff_vals = self.diff_cross_upsilon[i][j][k](t, diff_times, ix_func, ix_param, x_ki, x_kj)
                        self.sum_diff_f[log_name][ix_func][ix_param] += np.sum(diff_vals)

    def compute_sim_upsilon_sum(self, i, x_ki,
                                compute_f_sum=False,
                                compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # n_param
        n_param = self.n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Data
        kappa = self.kappa
        varpi = self.varpi

        # Computations
        for m in tqdm(range(varpi[i][i][1], len(self.list_times[i])),
                      disable=not verbose):
            t_m = self.list_times[i][m]
            t_n = self.list_times[i][:kappa[i][i][m]+1]
            diff_times = t_m-t_n
            n_vals = len(t_n)
            t = self.list_times2end[i][m]*np.ones(n_vals)
            ups_vals = self.upsilon[i][i][k](t, diff_times, x_ki, x_ki)
            self.sum_f[log_name] += np.sum(ups_vals)
            if compute_diff_f_sum:
                for ix_param in range(n_param):
                    self.sum_diff_f[log_name][ix_param] += np.sum(self.diff_sim_upsilon[k][i](t, diff_times, ix_param,
                                                                                              x_ki))

    def compute_objective(self, x_k, compute_f_sum=False,
                          compute_diff_f_sum=True, count_iter=True,
                          verbose=False):
        self.clear_estimates()

        d = self.d
        k = self.k
        mu_k = x_k[0]

        # mu term
        self.compute_mu_term(mu_k, compute_f_sum=compute_f_sum,
                             compute_diff_f_sum=compute_diff_f_sum)

        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            # Psi term
            self.compute_psi_sum(i, x_ki, mu_k,
                               compute_f_sum=compute_f_sum,
                               compute_diff_f_sum=compute_diff_f_sum,
                               verbose=verbose)

            # Upsilonzero term
            self.compute_upsilonzero_sum(i, x_ki,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         verbose=verbose)

            # Phi term
            self.compute_phi_sum(i, x_ki,
                                 compute_f_sum=compute_f_sum,
                                 compute_diff_f_sum=compute_diff_f_sum,
                                 verbose=verbose)

            # Sim upsilon
            self.compute_sim_upsilon_sum(i, x_ki,
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         verbose=verbose)
        for i, j in itertools.product(range(d), range(d)):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            if i != j:
                self.compute_cross_upsilon_sum(i, j, x_ki, x_kj,
                                               compute_f_sum=compute_f_sum,
                                               compute_diff_f_sum=compute_diff_f_sum,
                                               verbose=verbose)

        # Log sum estimates
        self.log_estimates()

        # Iteration
        if count_iter:
            self.t += 1

# =============================================================================
# LSE and LSE gradient
# =============================================================================
    def lse_k_estimate(self, x_k, compute_diff_f_sum=False, count_iter=True,
                       verbose=False):
        # Compute the LSE estimate
        self.compute_objective(x_k, compute_f_sum=True,
                               compute_diff_f_sum=compute_diff_f_sum,
                               count_iter=count_iter, verbose=verbose)
        lse_k = self.recombine_lse_k()
        self.lse_k = lse_k
        return lse_k

    def lse_k_grad_estimate(self, x_k, grad_alloc=True, rng=None, seed=None,
                            count_iter=True, verbose=False):
        self.compute_objective(x_k, compute_f_sum=self.is_log_lse,
                               compute_diff_f_sum=True, verbose=verbose)

        grad = self.recombine_grad_lse_k()
        self.grad_lse_k = grad
        # Log LSE
        if self.is_log_lse:
            lse_k = self.recombine_lse_k()
            self.logged_lse[self.t] = lse_k
            self.lse_k = lse_k
        return grad

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
                          'sum_diff_f']:
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
