# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.stratifications.\
    general_stratification import GeneralStratification


class PoissonExactEstim(Estimator):
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

    def initialize(self, k, n_iter, model, data, n_exact_single=None,
                   n_samples_adaptive_single=None,
                   nonadaptive_sample_size_single=None,
                   single_strfs=None):
        self.k = k
        self.n_iter = n_iter
        self.initialize_model_data(model, data)
        self.set_stratification(n_exact_single=n_exact_single,
                                n_samples_adaptive_single=n_samples_adaptive_single,
                                nonadaptive_sample_size_single=nonadaptive_sample_size_single,
                                single_strfs=single_strfs)
        self.list_log_names = self.get_list_log_names()
        self.make_param_indexing()
        self.initialize_logs()

# =============================================================================
# Stratification
# =============================================================================
    def set_singlesum_estimation(self, n_exact=None, n_samples_adaptive=None,
                                 nonadaptive_sample_size=None, **kwargs):
        d = self.d
        k = self.k
        local_kwargs = {}
        if n_exact is not None:
            local_kwargs['n_exact'] = n_exact
        if n_samples_adaptive is not None:
            local_kwargs['n_samples_adaptive'] = n_samples_adaptive
        if nonadaptive_sample_size is not None:
            local_kwargs['nonadaptive_sample_size'] = nonadaptive_sample_size
        # Mu
        mu_stratif = kwargs.get('mu_stratification', None)
        if mu_stratif is None:
            n_events_k = self.n_events[k]
            mu_stratif = GeneralStratification(n=n_events_k, **local_kwargs)
        self.stratif['mu'] = mu_stratif

    def set_stratification(self, n_exact_single=None,
                           n_samples_adaptive_single=None,
                           nonadaptive_sample_size_single=None,
                           single_strfs=None):
        self.stratif = {}
        if single_strfs is None:
            single_strfs = {}
        self.set_singlesum_estimation(n_exact=n_exact_single,
                                      n_samples_adaptive=n_samples_adaptive_single,
                                      nonadaptive_sample_size=nonadaptive_sample_size_single,
                                      **single_strfs)

# =============================================================================
# Log names
# =============================================================================
    def get_list_log_names(self):
        d = self.d
        list_log_names = ['mu']
        return list_log_names

    def get_singlesum_lognames(self):
        d = self.d
        single_ix_lognames = ['mu']
        return single_ix_lognames

# =============================================================================
# Parameters book-keeping
# =============================================================================
    def make_param_indexing(self):
        d = self.d
        k = self.k
        # parameters indexing
        self.n_f = {}
        self.vec_n_func = {}
        self.list_n_param = {}
        # mu
        log_name = 'mu'
        self.n_f[log_name] = 1
        self.vec_n_func[log_name] = [None]
        self.list_n_param[log_name] = [self.n_param_k]

    def make_dict_n_param(self):
        d = self.d
        k = self.k
        dict_n_param = {}

        # M term
        dict_n_param['M_term'] = self.n_param_k
        # Sum Mu
        dict_n_param['sum_mu'] = self.n_param_k

        # Result
        self.dict_n_param = dict_n_param

# =============================================================================
# Data allocation
# =============================================================================
    def make_null_sumarray(self, n_f, list_n_param, vec_n_func=None,
                           n_strata=None):
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
        self.diff_M_term = np.zeros(self.n_param_k)

        # Sum terms
        for log_name in self.list_log_names:
            # Sum f
            # Total estimates
            self.sum_f[log_name] = 0.

            # Gradient
            n_f = self.n_f[log_name]
            vec_n_func = self.vec_n_func[log_name]
            list_n_param = self.list_n_param[log_name]
            # Total estimates
            self.sum_diff_f[log_name] = self.make_null_sumarray(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None)

# =============================================================================
# Logging
# =============================================================================
    def make_null_sumarray_logs(self, n_f, list_n_param, vec_n_func=None,
                                n_strata=None, n_iter=1):
        # Make null arrays to store logs
        L = [None]*n_f
        for ix_f in range(n_f):
            if vec_n_func[ix_f] is None:
                n_param = list_n_param[ix_f]
                if n_strata is None:
                    L[ix_f] = np.zeros(n_param, n_iter)
                else:
                    L[ix_f] = np.zeros((n_param, n_strata, n_iter))
            else:
                L[ix_f] = [None]*3
                for ix_func in [1, 2]:
                    n_param = list_n_param[ix_f][ix_func]
                    if n_strata is None:
                        L[ix_f][ix_func] = np.zeros(n_param, n_iter)
                    else:
                        L[ix_f][ix_func] = np.zeros((n_param, n_strata,
                                                     n_iter))
        return L

    def update_sumarray_logs(self, log_type, log_name, ix_iter, n_f,
                             list_n_param, vec_n_func=None, n_strata=None):
        logs = getattr(self, log_type+'_log')[log_name]
        val = getattr(self, log_type)[log_name]

        for ix_f in range(n_f):
            if vec_n_func[ix_f] is None:
                n_param = list_n_param[ix_f]
                if n_strata is None:
                    logs[ix_f][:, ix_iter] = val[ix_f]+0.
                else:
                    logs[ix_f][:, :, ix_iter] = val[ix_f]+0.
            else:
                for ix_func in [1, 2]:
                    n_param = list_n_param[ix_f][ix_func]
                    if n_strata is None:
                        logs[ix_f][ix_func][:, ix_iter] = val[ix_f][ix_func]+0.
                    else:
                        logs[ix_f][ix_func][:, :, ix_iter] = val[ix_f][ix_func]+0.

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
            # M term
            self.M_term_log = np.zeros(n_iter)
            self.diff_M_term_log = np.zeros(n_iter)

            # Sum terms
            self.sum_f_log = {}
            self.sum_diff_f_log = {}
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]

                # Sum f
                # Total estimates
                self.sum_f_log[log_name] = np.zeros(n_iter)
                # Gradient
                n_f = self.n_f[log_name]
                vec_n_func = self.vec_n_func[log_name]
                list_n_param = self.list_n_param[log_name]
                # Total estimates
                self.sum_diff_f_log[log_name] = self.make_null_sumarray_logs(n_f, list_n_param, vec_n_func=vec_n_func, n_strata=None, n_iter=n_iter)

    def log_estimates(self):
        ix_iter = self.t
        if self.is_log_total_estimates:
            # M term
            self.M_term_log[ix_iter] = self.M_term
            self.diff_M_term_log[ix_iter] = self.diff_M_term

            # Sum terms
            for log_name in self.list_log_names:
                strf = self.stratif[log_name]

                # Sum f
                # Total estimates
                self.sum_f_log[log_name][ix_iter] = self.sum_f[log_name]

                # Gradient
                n_f = self.n_f[log_name]
                vec_n_func = self.vec_n_func[log_name]
                list_n_param = self.list_n_param[log_name]
                # Total estimates
                self.update_sumarray_logs('sum_diff_f', log_name, ix_iter,
                                          n_f, list_n_param,
                                          vec_n_func=vec_n_func,
                                          n_strata=None)

    def get_lse_k_components_log(self):
        d = self.d
        T_f = self.T_f
        n_lse_k_components = 1+len(self.list_log_names)
        # Initialize
        lse_k_components = np.zeros((self.n_iter, n_lse_k_components))
        lse_k_component_names = ['M_term']

        # Get components
        ix_component = 0
        # Baseline terms: M
        lse_k_components[:, ix_component] = self.M_term_log+0.
        ix_component += 1

        # Baseline term: sum
        log_name = 'mu'
        lse_k_components[:, ix_component] = -2.*(self.sum_f_log[log_name]/T_f)+0.
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
        T_f = self.T_f
        res = self.M_term-2.*(self.sum_f['mu']/T_f)
        return res

    def recombine_diff_lse_mu_kr(self, r):
        # Derivative of Z with respect to the rth parameter of mu_k
        T_f = self.T_f
        res = self.diff_M_term[r]

        # Mu term
        res -= 2.*(self.sum_diff_f['mu'][0][r]/T_f)

        return res

    def recombine_grad_lse_k(self):
        d = self.d
        k = self.k
        grad = np.zeros(self.n_param_k)
        # Derivative with respect to baseline parameters
        for ix_param in range(self.n_param_k):
            grad[ix_param] = self.recombine_diff_lse_mu_kr(ix_param)
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

    def compute_objective(self, x_k, compute_f_sum=False,
                          compute_diff_f_sum=True, count_iter=True,
                          verbose=False):
        self.clear_estimates()

        d = self.d
        k = self.k

        # M term
        self.compute_M_term(x_k, compute_f_sum=compute_f_sum,
                            compute_diff_f_sum=compute_diff_f_sum)
        # Mu term
        self.compute_mu_sum(x_k, compute_f_sum=compute_f_sum,
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

    def lse_k_grad_estimate(self, x_k, count_iter=True, verbose=False):
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
