# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.optimize.estimators.general_estimator import GeneralEstimator
from aslsd.optimize.optim_logging.general_optim_logger import GeneralOptimLogger
from aslsd.optimize.solvers.solver import Solver
from aslsd.optimize.solvers.momentum import Momentum
from aslsd.optimize.solvers.rmsprop import RMSprop
from aslsd.optimize.solvers.adam import ADAM
from aslsd.stats.residual_analysis import goodness_of_fit as gof
import aslsd.utilities.useful_functions as uf
import aslsd.utilities.useful_statistics as us
from aslsd.stats.events.process_path import ProcessPath


class NonHomPoisson:
    """
    Class for homogeneous Poisson models of point processes.

    Let :math:`\\mathbf{N}` be a d-dimensional counting process with
    conditional intensity :math:`\\boldsymbol{\\lambda}`.
    We say that :math:`\\mathbf{N}` is a homogeneous Poisson process if there
    exists positive constants :math:`(\\mu_{i})_{i \\in[d]}` (that we call
    baselines), such that for all :math:`i \\in[d]` and for all
    :math:`t \\geq 0`, we have

    .. math::
        \\lambda_{i}(t):=\\mu_{i}.


    Attributes
    ----------
    d : `int`
        Dimension of the homogeneous Poisson process.
    index_from_one : `bool`, optional
        Start the indexing of baselines from 1 instead of 0. The default is False.
    mu_names : `list` of `str`, optional
        List of names of baseline parameters. The default is None.
    is_fitted : `bool`, optional
        True if this model has been fitted. The default is False.
    fitted_mu : `numpy.ndarray`, optional
        Array of fitted baselines. The default is None.
    fit_residuals : `list` of `numpy.ndarray`, optional
        List of fit residuals. The default is None.

    """

    def __init__(self, baselines, d=None, index_from_one=False, mu_names=None,
                 is_fitted=False, fitted_mu_param=None, fit_residuals=None):
        if uf.is_array(baselines):
            vec_baselines = copy.deepcopy(baselines)
        else:
            vec_baselines = [baselines]
        self.baselines = vec_baselines
        if d is None:
            d = len(self.baselines)
        self.d = d
        self.vec_n_param = [self.baselines[i].n_param for i in range(d)]
        if mu_names is None:
            mu_names = self.get_param_names(index_from_one=index_from_one)
        self.mu_names = mu_names
        self.is_fitted = False
        self.fitted_mu_param = fitted_mu_param
        self.fit_residuals = fit_residuals

        intensity = self.make_intensity()
        self.intensity = intensity

    # Intensity
    def make_intensity(self):
        d = self.d
        intensity = [None]*d
        for i in range(d):
            def make_f(i):
                def f(t, mu_param=None):
                    mu_param = self.load_param(mu_param=mu_param)
                    return self.baselines[i].mu(t, mu_param[i])
                return f
            f = make_f(i)
            intensity[i] = copy.deepcopy(f)
        return intensity

    # Param names
    def get_param_names(self, index_from_one=False):
        """
        Get the standard names of baseline variables.

        By default, we denote the baselines by :math:`(\\mu_{i})_{i \\in[d]}`.

        Parameters
        ----------
        index_from_one : `bool`, optional
            Start the indexing of baselines from 1 instead of 0. The default is False.

        Returns
        -------
        mu_names : `list` of `str`
            List of names of baseline parameters.

        """
        d = self.d
        mu_names = [self.baselines[i].get_vec_param_names() for i in range(d)]
        return mu_names

    # Parameters operations
    def load_param(self, mu_param=None):
        if mu_param is None:
            mu_param = self.fitted_mu_param
            if mu_param is None:
                raise ValueError("Missing value for Mu")
        return mu_param

    # Fit
    # Exact evaluation
    def get_exact_lse_k(self, k, list_times, T_f, base_param_k):
        sum_term = np.sum(self.baselines.mu[k](list_times[k], base_param_k))
        return self.baselines.M[k](T_f, base_param_k)-(2./T_f)*sum_term

    def get_exact_diff_lse_k(self, k, list_times, T_f, ix_diff, base_param_k):
        sum_term = np.sum(self.baselines.diff_mu[k](list_times[k], ix_diff,
                                                    base_param_k))
        return self.baselines.diff_M[k](T_f, ix_diff,
                                        base_param_k)-(2./T_f)*sum_term

    def get_exact_grad_lse_k(self, k, list_times, T_f, base_param_k):
        n_param_k = len(base_param_k)
        grad = np.zeros(n_param_k)
        for ix in range(n_param_k):
            grad[ix] = self.get_exact_diff_lse_k(k, list_times, T_f, ix,
                                                 base_param_k)
        return grad

    # Approximation
    def get_approx_lse_k(self, k, list_times, T_f, base_param_k):
        sum_term = np.sum(self.baselines.mu[k](list_times[k], base_param_k))
        return self.baselines.M[k](T_f, base_param_k)-(2./T_f)*sum_term

    def get_approx_diff_lse_k(self, k, list_times, T_f, ix_diff, base_param_k):
        sum_term = np.sum(self.baselines.diff_mu[k](list_times[k], ix_diff,
                                                    base_param_k))
        return self.baselines.diff_M[k](T_f, ix_diff,
                                        base_param_k)-(2./T_f)*sum_term

    def get_approx_grad_lse_k(self, k, list_times, T_f, base_param_k):
        n_param_k = len(base_param_k)
        grad = np.zeros(n_param_k)
        for ix in range(n_param_k):
            grad[ix] = self.get_exact_diff_lse_k(k, list_times, T_f, ix,
                                                 base_param_k)
        return grad

    # Fit
    def clear_fit(self):
        self.is_fitted = False
        self.fitted_mu_param = None
        self.fit_residuals = None
        self.fit_log = None

    def fit(self, process_path, x_0=None, n_iter=1000, solvers=None,
            estimators=None, rng=None, seed=None, verbose=False, clear=True,
            poisson_sol=False,
            write=True, grad_alloc=False, strf_args=None, estim_args=None,
            log_args=None,
            **kwargs):
        """
        Fit the Non-homogeneous Poisson model to some observations.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The least squares error (LSE) of this model for these observations is
        defined as

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\frac{1}{T} \\sum_{k=1}^{d} \\int_{0}^{T} \\lambda_{k}(t)^{2} \\mathrm{~d} t-\\frac{2}{T} \\sum_{k=1}^{d} \\sum_{m=1}^{N_{T}^{k}} \\lambda_{k}\\left(t_{m}^{k}\\right).

        For a Non-homogeneous Poisson model, this simplifies to

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\sum_{k=1}^{d} \\bigg( \\mu_{k}^{2} -2 \\frac{N_{T}^{k}}{T} \\bigg).

        This expression is minimized if for all :math:`k \\in [d]` we have

        .. math::
            \\mu_k^{*}:=\\frac{N_{T}^{k}}{T}.

        The optimal baseline vector :math:`\\boldsymbol{\\mu^{*}}` defined
        above is also the maximizer of the loglikelihood of these observations.

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        clear : `bool`, optional
            If true, delete all previously saved results and logs from the
            corresponding attributes of the MHP object. The default is True.
        write : `bool`, optional
            If true, save the estimation results and logs in the corresponding
            attributes of the MHP object. The default is True.

        Returns
        -------
        fitted_mu : `numpy.ndarray`
            Fitted baselines.

        """
        if rng is None:
            rng = np.random.default_rng(seed)
        d = self.d
        list_times = process_path.list_times
        T_f = process_path.T_f
        # Poisson Solution if activated
        if poisson_sol:
            x = [None]*d
            for i in range(d):
                x[i] = self.baselines[i].basis_mus[0].get_events_rate(
                    list_times[i])
            if write:
                self.is_fitted = True
                self.fitted_mu_param = x
            return x

        # Stratification arguments
        if strf_args is None:
            strf_args_ = [{} for i in range(d)]
        elif type(strf_args) == dict:
            strf_args_ = [copy.deepcopy(strf_args) for i in range(d)]
        elif uf.is_array(strf_args):
            if len(strf_args) == d:
                strf_args_ = copy.deepcopy(strf_args)

        # Estimation arguments
        if estim_args is None:
            estim_args_ = [{} for i in range(d)]
        elif type(estim_args) == dict:
            estim_args_ = [copy.deepcopy(estim_args) for i in range(d)]
        elif uf.is_array(estim_args):
            if len(estim_args) == d:
                estim_args_ = copy.deepcopy(estim_args)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Bounds
        lower_bnds = [self.baselines[i].get_param_lower_bounds()
                      for i in range(d)]
        upper_bnds = [self.baselines[i].get_param_upper_bounds()
                      for i in range(d)]

        # Initialisation
        if x_0 is None:
            x_0 = [None]*d
            for ix in range(d):
                x_0[ix] = self.baselines[ix].get_param_lower_bounds()+rng.uniform(low=0., high=1., size=self.vec_n_param[ix])

        # Initialize Estimators
        if estimators is None:
            estimators = [None]*d
            for i in range(d):
                estimators[i] = GeneralEstimator(**estim_args_[i])

        for i in range(d):
            estimators[i].n_iter = n_iter[i]
            estimators[i].initialize_logs(n_param=self.vec_n_param[i])
            estimators[i].set_stratification(**strf_args_[i])

        # Initialize Solvers
        if solvers is None:
            solvers = [ADAM(**kwargs) for k in range(d)]
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]
            elif type(solvers) == str:
                if solvers == 'Momentum':
                    solvers = [Momentum(**kwargs) for k in range(d)]
                elif solvers == 'RMSprop':
                    solvers = [RMSprop(**kwargs) for k in range(d)]
                elif solvers == 'ADAM':
                    solvers = [ADAM(**kwargs) for k in range(d)]

        # Initialize logger
        if log_args is None:
            log_args = {}
        logger = GeneralOptimLogger(d, n_iter, **log_args)

        # Scheme
        x = [None]*d
        for k in range(d):
            x_k = x_0[k]
            logger.log_param(k, 0, x_k)
            lower_bounds_k = lower_bnds[k]
            upper_bounds_k = upper_bnds[k]
            n_iter_k = n_iter[k]

            for t in tqdm(range(n_iter_k), disable=not verbose):
                # Compute LSE gradient estimate for parameters x_k
                estimators[k].estimate_sum(list_times[k],
                                           f=self.baselines[k].mu,
                                           diff_f=self.baselines[k].diff_mu,
                                           n_param=self.vec_n_param[k],
                                           f_args={'params': x_k},
                                           diff_f_args={'params': x_k},
                                           compute_f_sum=estimators[k].is_log_sum_f,
                                           compute_diff_f_sum=True,
                                           grad_alloc=grad_alloc, rng=rng,
                                           count_iter=True)
                grad_M_k = np.array([self.baselines[k].diff_M(T_f,
                                                              ix_diff, x_k) for ix_diff in range(self.vec_n_param[k])])
                g_t = grad_M_k-2.*(estimators[k].sum_diff_f/T_f)
                # g_t[1] = 0.
                # g_t[2] = 0.
                logger.log_grad(k, t, g_t)
                # Apply solver iteration
                x_k = solvers[k].iterate(t, x_k, g_t)
                # Project into space of parameters
                x_k = np.clip(x_k, lower_bounds_k, upper_bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k

        if write:
            self.is_fitted = True
            self.fitted_mu_param = x
            logger.process_logs(self)
            self.fit_log = logger
        return x

    # Simulation
    def simulate(self, T_f, mu_param=None, rng=None, seed=1234):
        """
        Simulate a path of the homogeneous Poisson model.

        Parameters
        ----------
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baselines. If None, use the fitted baselines. The default
            is None.
        seed : `int`, optional
            Seed of the random generator. The default is 1234.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of jump times per dimension.

        """
        rng = us.make_rng(rng=rng, seed=seed)
        if mu_param is None:
            mu_param = self.fitted_mu_param
            if mu_param is None:
                raise ValueError("Missing value for mu_param")
        d = self.d
        list_times = [None]*d
        rng = np.random.default_rng(seed)
        for i in range(d):
            list_times[i] = self.baselines[i].simulate(T_f, mu_param[i],
                                                           rng=rng)
        process_path = ProcessPath(list_times, T_f)
        return process_path

    # Evaluation
    def get_residuals(self,  process_path, mu_param=None, write=True):
        """
        Compute the residuals of the model.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The residuals of a homogeneous Poisson model :math:`\\boldsymbol{\\mu}`
        are defined by

        .. math::
            \\left\\{\\mu_k(t^k_{m+1}-t^k_{m}): k \\in [d], m \\in [N^k_T-1] \\right\\}.

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times per dimension.
        mu : `numpy.ndarray`, optional
            Vector of baselines. If None, use the fitted baselines. The default
            is None.
        write : `bool`, optional
            Save the results. The default is True.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        residuals : `list` of `numpy.ndarray`
            List of residuals per dimension.

        """
        d = self.d
        residuals = [None]*d
        if mu_param is None:
            if self.is_fitted:
                mu_param = self.fitted_mu_param
            else:
                raise ValueError("Mu must be specified.")

        d = self.d
        resdiuals = [None]*d
        for i in range(d):
            residuals[i] = self.baselines[i].get_residuals(process_path.list_times[i], mu_param[i])
        if self.is_fitted and write:
            self.fit_residuals = residuals
        return residuals

    def ks_test_residuals(self, residuals=None):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.ks_test_residuals(residuals)

    def qq_plot(self, i, residuals=None, labels=None, style='exponential',
                substract_yx=False, normalize=False, max_points=None,
                display_line45=True, log_scale=False,
                ax=None, save=False, filename='image.png', show=False,
                **kwargs):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.qq_plot(residuals[i], n_models=1, labels=labels,
                           style=style, substract_yx=substract_yx,
                           normalize=normalize, max_points=max_points,
                           display_line45=display_line45, log_scale=log_scale,
                           ax=ax, save=save,
                           filename=filename, show=show, **kwargs)

    # Serialization
    def save(self, file, **kwargs):
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_out = open(file_mu, "wb", **kwargs)
        pickle.dump(self.fitted_mu_param, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        file_residuals = file+'_fitted_residuals'
        pickle_out = open(file_residuals, "wb", **kwargs)
        pickle.dump(self.fit_residuals, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_in = open(file_mu, "rb")
        fitted_mu_param = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        pickle_in = open(file_residuals, "rb")
        fitted_residuals = pickle.load(pickle_in)

        self.clear_fit()

        self.is_fitted = True
        self.fitted_mu_param = fitted_mu_param
        self.fit_residuals = fitted_residuals