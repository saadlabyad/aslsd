# License: BSD 3 clause

import copy
import itertools

import numpy as np
from tqdm import tqdm

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.stratifications.\
    double_ix_stratification import DoubleIxStratification
from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification
from aslsd.utilities import useful_statistics as us


class ExactMtlhStratified(Estimator):
    def __init__(self, **kwargs):
        self.t = 0

        # Logging
        self.is_grad_target = kwargs.get('is_grad_target', False)

        # Logging
        self.is_log_lse = kwargs.get('is_log_lse', False)
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
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
                K_stratif[i] = GeneralStratification(list_times[i], **kwargs)
            self.stratif[log_name] = K_stratif[i]

        # upsilonzero
        upsilonzero_stratif = kwargs.get('upsilonzero_stratification',
                                         [None for i in range(d)])
        for i in range(d):
            log_name = 'upsilonzero_'+str(i)
            if upsilonzero_stratif[i] is None:
                n_events_i = self.n_events[i]
                upsilonzero_stratif[i] = GeneralStratification(list_times[i], **kwargs)
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

            # Gradient
            n_f = self.n_f[log_name]
            vec_n_func = self.vec_n_func[log_name]
            list_n_param = self.list_n_param[log_name]
            # Total estimates
            self.sum_diff_f[log_name] = self.make_null_sumarray(
                n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)

    # Logging
    def intialize_logs(self):
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

        # LSE
        if self.is_log_lse:
            self.logged_lse = np.zeros(n_iter+1)

    def get_log(self):
        logs = {}
        if self.is_log_lse:
            logs['lse'] = self.logged_lse

        if self.is_log_ixs:
            logs['samples'] = copy.deepcopy(self.logged_ixs)

        return logs

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
                       compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'mu'
        if verbose:
            print('Computing ', log_name, ' term ...')
        n_func = 1
        list_n_param = self.list_n_param[log_name]

        # Times
        t_m = self.list_times[k]

        # Compute sum f
        if compute_f_sum:
            self.sum_f[log_name] = np.sum(self.mu[k](t_m, x_mu_k))

        # Compute sum f derivatives
        if compute_diff_f_sum:
            for ix_f in range(n_func):
                for ix_param in range(list_n_param[ix_f]):
                    self.sum_diff_f[log_name][ix_f][ix_param] = np.sum(self.diff_mu[k](t_m, ix_param, x_mu_k))

    def compute_K_sum(self, i, x_ker_ki, x_mu_k, x_imp_ki, compute_f_sum=False,
                      compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'K_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # Data
        t = self.list_times2end[i]
        s = self.list_times[i]
        xi = self.list_marks[i]

        n_func = 2
        list_n_param = self.list_n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Compute sum f
        if compute_f_sum:
            list_f_vals = np.zeros((len(t), n_func))
            # K values
            list_f_vals[:, 0] = self.K[k][i](t, s, x_ker_ki, x_mu_k)
            # Impact values
            list_f_vals[:, 1] = self.impact[k][i](xi, x_imp_ki)
            # Sum
            f_vals = np.prod(list_f_vals, axis=1)
            self.sum_f[log_name] = np.sum(f_vals)

        # Compute sum f derivatives
        if compute_diff_f_sum:
            # K values
            ix_f = 0
            for ix_func in [1, 2]:
                for ix_param in range(list_n_param[ix_f][ix_func]):
                    diff_f_vals = (self.diff_K[k][i](t, s, ix_func, ix_param,
                                                     x_ker_ki, x_mu_k)
                                   * list_f_vals[:, 1])
                    self.sum_diff_f[log_name][ix_f][ix_func][ix_param] = np.sum(diff_f_vals)

            # Impact values
            ix_f = 1
            for ix_param in range(list_n_param[ix_f]):
                diff_f_vals = (self.diff_impact[k][i](xi, ix_param, x_imp_ki)
                               * list_f_vals[:, 0])
                self.sum_diff_f[log_name][ix_f][ix_param] = np.sum(diff_f_vals)

    def compute_upsilonzero_sum(self, i, x_ker_ki, x_imp_ki,
                                compute_f_sum=False,
                                compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilonzero_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # Data
        t = self.list_times2end[i]
        xi = self.list_marks[i]

        n_func = 2

        # n_param
        list_n_param = self.list_n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Compute sum f
        if compute_f_sum:
            list_f_vals = np.zeros((len(t), n_func))
            # Upsilon values
            list_f_vals[:, 0] = self.upsilon[i][i][k](t, 0., x_ker_ki,
                                                      x_ker_ki)
            # Impact values
            imp_vals = self.impact[k][i](xi, x_imp_ki)
            list_f_vals[:, 1] = imp_vals**2
            # Sum
            f_vals = np.prod(list_f_vals, axis=1)
            self.sum_f[log_name] = np.sum(f_vals)

        # Compute sum f derivatives
        if compute_diff_f_sum:
            # Upsilon values
            ix_f = 0
            for ix_param in range(list_n_param[ix_f]):
                diff_f_vals = (self.diff_sim_upsilon[k][i](t, 0., ix_param,
                                                           x_ker_ki)
                               * list_f_vals[:, 1])
                self.sum_diff_f[log_name][ix_f][ix_param] = np.sum(diff_f_vals)
            # Impact values
            ix_f = 1
            for ix_param in range(list_n_param[ix_f]):
                diff_f_vals = (list_f_vals[:, 0]
                               * 2.*imp_vals*self.diff_impact[k][i](xi,
                                                                    ix_param,
                                                                    x_imp_ki))
                self.sum_diff_f[log_name][ix_f][ix_param] = np.sum(diff_f_vals)

    def compute_phi_sum(self, i, x_ker_ki, x_imp_ki, compute_f_sum=False,
                        compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'phi_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        kappa = self.kappa
        varpi = self.varpi

        list_n_param = self.list_n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Data
        xi = self.list_marks[i][:kappa[i][k][len(self.list_times[k])-1]+1]
        imp_vals = self.impact[k][i](xi, x_imp_ki)
        diff_imp_vals = np.zeros((len(x_imp_ki), len(xi)))
        for ix_param in range(len(x_imp_ki)):
            diff_imp_vals[ix_param] = self.diff_impact[k][i](xi, ix_param,
                                                             x_imp_ki)

        for m in tqdm(range(varpi[k][i][1], len(self.list_times[k])),
                      disable=not verbose):
            t_m = self.list_times[k][m]
            t_n = self.list_times[i][:kappa[i][k][m]+1]
            n_vals = len(t_n)
            diff_times = t_m-t_n
            phi_vals = self.phi[k][i](diff_times, x_ker_ki)
            self.sum_f[log_name] += np.sum(phi_vals*imp_vals[:n_vals])
            if compute_diff_f_sum:
                # Kernel term
                ix_f = 0
                for ix_param in range(list_n_param[ix_f]):
                    self.sum_diff_f[log_name][ix_f][ix_param] += np.sum(self.diff_phi[k][i](diff_times, ix_param, x_ker_ki)
                                                                        * imp_vals[:n_vals])
                # Impact term
                ix_f = 1
                for ix_param in range(list_n_param[ix_f]):
                    self.sum_diff_f[log_name][ix_f][ix_param] += np.sum(phi_vals
                                                                        * diff_imp_vals[ix_param][:n_vals])

    def compute_cross_upsilon_sum(self, i, j, x_ker_ki, x_ker_kj, x_imp_ki,
                                  x_imp_kj, compute_f_sum=False,
                                  compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(j)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # n_param
        list_n_param = self.list_n_param[log_name]

        # function args
        list_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_kj},
                       {'params_1': x_imp_ki, 'params_2': x_imp_kj}]
        list_diff_f_args = [{'params_1': x_ker_ki, 'params_2': x_ker_kj},
                            {'params_1': x_imp_ki, 'params_2': x_imp_kj}]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Data
        kappa = self.kappa
        varpi = self.varpi
        xi_m = self.list_marks[i]
        xi_n = self.list_marks[j][:kappa[j][i][len(self.list_times[i])-1]+1]
        imp_vals_m = self.impact[k][i](xi_m, x_imp_ki)
        imp_vals_n = self.impact[k][j](xi_n, x_imp_kj)
        diff_imp_vals_m = np.zeros((len(x_imp_ki), len(xi_m)))
        for ix_param in range(len(x_imp_ki)):
            diff_imp_vals_m[ix_param] = self.diff_impact[k][i](xi_m, ix_param,
                                                               x_imp_ki)
        diff_imp_vals_n = np.zeros((len(x_imp_kj), len(xi_n)))
        for ix_param in range(len(x_imp_kj)):
            diff_imp_vals_n[ix_param] = self.diff_impact[k][j](xi_n, ix_param,
                                                               x_imp_kj)
        # Computations
        for m in tqdm(range(varpi[i][j][1], len(self.list_times[i])),
                      disable=not verbose):
            t_m = self.list_times[i][m]
            t_n = self.list_times[j][:kappa[j][i][m]+1]
            n_vals = len(t_n)
            diff_times = t_m-t_n
            t = self.list_times2end[i][m]*np.ones(n_vals)
            ups_vals = self.upsilon[i][j][k](t, diff_times, x_ker_ki, x_ker_kj)
            local_imp_vals = imp_vals_m[m]*imp_vals_n[:n_vals]
            self.sum_f[log_name] += np.sum(ups_vals*local_imp_vals)
            if compute_diff_f_sum:
                # Kernel term
                ix_f = 0
                for ix_func in [1, 2]:
                    for ix_param in range(list_n_param[ix_f][ix_func]):
                        self.sum_diff_f[log_name][ix_f][ix_func][ix_param] += np.sum(self.diff_cross_upsilon[i][j][k](t,
                                                                                                                      diff_times, ix_func, ix_param,
                                                                                                                      x_ker_ki, x_ker_kj)
                                                                                     * local_imp_vals)
                # Impact term
                ix_f = 1
                for ix_func in [1, 2]:
                    for ix_param in range(list_n_param[ix_f][ix_func]):
                        if ix_func == 1:
                            diff_imp_term = (diff_imp_vals_m[ix_param][m]
                                             * imp_vals_n[:n_vals])
                        elif ix_func == 2:
                            diff_imp_term = (imp_vals_m[m]
                                             * diff_imp_vals_n[ix_param][:n_vals])
                        self.sum_diff_f[log_name][ix_f][ix_func][ix_param] += np.sum(ups_vals
                                                                                     * diff_imp_term)

    def compute_sim_upsilon_sum(self, i, x_ker_ki, x_imp_ki,
                                compute_f_sum=False,
                                compute_diff_f_sum=True, verbose=False):
        k = self.k
        log_name = 'upsilon_'+str(i)+'_'+str(i)
        if verbose:
            print('Computing ', log_name, ' term ...')

        # n_param
        list_n_param = self.list_n_param[log_name]

        # Computations to be made
        if compute_diff_f_sum:
            compute_f_sum = True

        # Data
        kappa = self.kappa
        varpi = self.varpi
        xi = self.list_marks[i]
        imp_vals = self.impact[k][i](xi, x_imp_ki)
        diff_imp_vals = np.zeros((len(x_imp_ki), len(xi)))
        for ix_param in range(len(x_imp_ki)):
            diff_imp_vals[ix_param] = self.diff_impact[k][i](xi, ix_param,
                                                             x_imp_ki)
        # Computations
        for m in tqdm(range(varpi[i][i][1], len(self.list_times[i])),
                      disable=not verbose):
            t_m = self.list_times[i][m]
            t_n = self.list_times[i][:kappa[i][i][m]+1]
            diff_times = t_m-t_n
            n_vals = len(t_n)
            t = self.list_times2end[i][m]*np.ones(n_vals)
            ups_vals = self.upsilon[i][i][k](t, diff_times, x_ker_ki, x_ker_ki)
            local_imp_vals = imp_vals[m]*imp_vals[:n_vals]
            self.sum_f[log_name] += np.sum(ups_vals*local_imp_vals)
            if compute_diff_f_sum:
                # Kernel term
                ix_f = 0
                for ix_param in range(list_n_param[ix_f]):
                    self.sum_diff_f[log_name][ix_f][ix_param] += np.sum(self.diff_sim_upsilon[k][i](t,
                                                                                                    diff_times, ix_param,
                                                                                                    x_ker_ki)
                                                                        * local_imp_vals)
                # Impact term
                ix_f = 1
                for ix_param in range(list_n_param[ix_f]):
                    diff_imp_term = ((diff_imp_vals[ix_param][m]
                                      * imp_vals[:n_vals])
                                     + (imp_vals[m]
                                        * diff_imp_vals[ix_param][:n_vals]))
                    self.sum_diff_f[log_name][ix_f][ix_param] += np.sum(ups_vals
                                                                        * diff_imp_term)

    def compute_objective(self, x_k, compute_f_sum=False,
                          compute_diff_f_sum=True, count_iter=True,
                          verbose=False):
        self.clear_estimates()

        d = self.d
        k = self.k
        x_mu_k, x_ker_k, x_imp_k = self.xk2matrix_params(k, x_k)
        # M term
        self.compute_M_term(x_mu_k, compute_f_sum=compute_f_sum,
                            compute_diff_f_sum=compute_diff_f_sum)
        # Mu term
        self.compute_mu_sum(x_mu_k, compute_f_sum=compute_f_sum,
                            compute_diff_f_sum=compute_diff_f_sum,
                            verbose=verbose)

        for i in range(d):
            # K term
            self.compute_K_sum(i, x_ker_k[i], x_mu_k, x_imp_k[i],
                               compute_f_sum=compute_f_sum,
                               compute_diff_f_sum=compute_diff_f_sum,
                               verbose=verbose)

            # Upsilonzero term
            self.compute_upsilonzero_sum(i, x_ker_k[i], x_imp_k[i],
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         verbose=verbose)

            # Phi term
            self.compute_phi_sum(i, x_ker_k[i], x_imp_k[i],
                                 compute_f_sum=compute_f_sum,
                                 compute_diff_f_sum=compute_diff_f_sum,
                                 verbose=verbose)

            # Sim upsilon
            self.compute_sim_upsilon_sum(i, x_ker_k[i], x_imp_k[i],
                                         compute_f_sum=compute_f_sum,
                                         compute_diff_f_sum=compute_diff_f_sum,
                                         verbose=verbose)
        for i, j in itertools.product(range(d), range(d)):
            if i != j:
                self.compute_cross_upsilon_sum(i, j, x_ker_k[i], x_ker_k[j],
                                               x_imp_k[i], x_imp_k[j],
                                               compute_f_sum=compute_f_sum,
                                               compute_diff_f_sum=compute_diff_f_sum,
                                               verbose=verbose)
        # Iteration
        if count_iter:
            self.t += 1

# =============================================================================
# LSE and LSE gradient
# =============================================================================
    def lse_k_estimate(self, x_k, compute_diff_f_sum=False, count_iter=True):
        # Compute the LSE estimate
        self.compute_objective(x_k, compute_f_sum=True,
                               compute_diff_f_sum=compute_diff_f_sum,
                               count_iter=count_iter)
        lse = self.recombine_lse_k()
        return lse

    def lse_k_grad_estimate(self, x_k, count_iter=True, verbose=False):
        self.compute_objective(x_k, compute_f_sum=self.is_log_lse,
                               compute_diff_f_sum=True, verbose=verbose)

        grad = self.recombine_grad_lse_k()

        # Log LSE
        if self.is_log_lse:
            lse = self.recombine_lse_k()
            self.logged_lse[self.t] = lse

        return grad
