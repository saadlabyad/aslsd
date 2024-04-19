# License: BSD 3 clause

import copy
import itertools
import pickle

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.estimators.poisson_exact_estimator import PoissonExactEstim
from aslsd.optimize.estimators.poisson_stratified_estimator import PoissonStratEstim
from aslsd.optimize.optim_logging.optim_logger import OptimLogger
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

    def __init__(self, _baselines_vec, index_from_one=False, mu_names=None):
        self.clear_fit()
        self.d = len(_baselines_vec)
        self.is_fitted = False
        self.index_from_one = index_from_one

        self.baselines_vec = _baselines_vec

        intensity = self.make_intensity()
        self.intensity = intensity

    # Baselines vector
    @property
    def baselines_vec(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._baselines_vec

    @baselines_vec.setter
    def baselines_vec(self, L):
        if not uf.is_array(L):
            L = [L]
        # Baseline attributes only
        self._baselines_vec = copy.deepcopy(L)
        self.d = len(L)
        self.vector_n_param = self.get_vector_n_param()
        # Parameters names
        self.param_names = self.get_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.param_lower_bounds = self.get_param_lower_bounds()
        self.param_upper_bounds = self.get_param_upper_bounds()

        # General updates
        self.make_functionals()
        self.n_param_k = [n for n in self.vector_n_param]
        self.n_param = sum(self.n_param_k)

    @baselines_vec.deleter
    def baselines_vec(self):
        del self._baselines_vec

# =============================================================================
# Exact computation of loss functions (wrappers)
# =============================================================================
    def get_lse_k(self, k, process_path, mu_param=None, verbose=False,
                  initialize=False):
        if verbose:
            print('Computing partial LSE k=', k, '...')
        # Exact LSE
        mu_param = self.load_param(mu_param=mu_param)
        d = self.d
        # Load estimators config
        estimators = [PoissonExactEstim() for k in range(d)]
        # Initialize Estimators with training data
        for k in range(d):
            estimators[k].initialize(k, 10, self, process_path)
        lse_k = estimators[k].lse_k_estimate(mu_param[k], verbose=verbose)
        return lse_k

    def get_lse(self, process_path, mu_param=None, verbose=False,
                initialize=False):
        # Exact lse
        lse = 0.
        for k in range(self.d):
            lse += self.get_lse_k(k, process_path, mu_param=mu_param,
                                  verbose=verbose, initialize=initialize)
        return lse

    def get_intensity_at_jumps(self, process_path, mu_param=None,
                               verbose=False):
        mu_param = self.load_param(mu_param=mu_param)
        # Path
        d = process_path.d
        list_times = process_path.list_times
        # Precomp
        intensity = [np.zeros(process_path.n_events[i])
                     for i in range(d)]
        if verbose:
            print('Starting Computations...')
        # Compute Intensity
        for k in range(d):
            if verbose:
                print('Computing intensity, dimension k=', str(k), ' ...')
            # Parameters
            mu_param_k = mu_param[k]
            # Baseline part
            intensity[k] = self.mu[k](list_times[k], mu_param[k])
        return intensity

    # Intensity
    def make_intensity(self):
        d = self.d
        intensity = [None]*d
        for i in range(d):
            def make_f(i):
                def f(t, mu_param=None):
                    mu_param = self.load_param(mu_param=mu_param)
                    return self.baselines_vec[i].mu(t, mu_param[i])
                return f
            f = make_f(i)
            intensity[i] = copy.deepcopy(f)
        return intensity

    # Param names
    def get_param_names(self, index_from_one=False):
        """
        Get the matrix of parameter names per kernel model.
        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        mu_param_names = [None]*d
        if d == 1:
            mu = self._baselines_vec[0]
            vec_names = mu.get_vec_param_names()
            n_param = mu.n_param
            mu_param_names[0] = [None]*n_param
            for ix_param in range(n_param):
                ix_basis = mu.ix_map[ix_param]['mu']
                ix_2 = mu.ix_map[ix_param]['par']
                if mu.n_basis_mus == 1:
                    mu_param_names[0][ix_param] = vec_names[ix_basis][ix_2]
                else:
                    sub = str(ix_basis+int(index_from_one))
                    s = vec_names[ix_basis][ix_2]
                    mu_param_names[0][ix_param] = uf.add_subscript(s, sub)

        else:
            for i in range(d):
                mu = self._baselines_vec[i]
                vec_names = mu.get_vec_param_names()
                n_param = mu.n_param
                mu_param_names[i] = [None]*n_param
                if mu.n_basis_mus == 1:
                    for ix_param in range(n_param):
                        ix_basis = mu.ix_map[ix_param]['mu']
                        ix_2 = mu.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))
                        s = vec_names[ix_basis][ix_2]
                        mu_param_names[i][ix_param] = uf.add_subscript(s, sub)
                else:
                    for ix_param in range(n_param):
                        ix_basis = mu.ix_map[ix_param]['mu']
                        ix_2 = mu.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(ix_basis+int(index_from_one))+','
                        s = vec_names[ix_basis][ix_2]
                        mu_param_names[i][ix_param] = uf.add_subscript(s, sub)
        return mu_param_names

    def get_vector_n_param(self):
        d = self.d
        vector_n_param = [self._baselines_vec[i].n_param for i in range(d)]
        return vector_n_param

    def get_param_lower_bounds(self):
        d = self.d
        bnds = [self._baselines_vec[i].get_param_lower_bounds()
                for i in range(d)]
        return bnds

    def get_param_upper_bounds(self):
        d = self.d
        bnds = [self._baselines_vec[i].get_param_upper_bounds()
                for i in range(d)]
        return bnds

    # Parameters operations
    def load_param(self, mu_param=None):
        if mu_param is None:
            mu_param = self.fitted_mu_param
            if mu_param is None:
                raise ValueError("Missing value for Mu")
        return mu_param

    def init_estimator(self, estimator, k):
        # Ixs book-keeping
        estimator.n_param_k = self.vector_n_param[k]
        # Functionals
        estimator.M = self.M
        estimator.diff_M = self.diff_M
        estimator.mu = self.mu
        estimator.diff_mu = self.diff_mu

    def make_functionals(self):
        d = self.d

        # Mu
        self.mu_compensator = [self._baselines_vec[i].compensator for i in range(d)]
        self.mu = [None for i in range(d)]
        self.diff_mu = [None for i in range(d)]
        self.M = [None for i in range(d)]
        self.diff_M = [None for i in range(d)]
        for i in range(d):
            baseline = self._baselines_vec[i]
            mu = baseline.make_mu()
            self.mu[i] = mu
            diff_mu = baseline.make_diff_mu()
            self.diff_mu[i] = diff_mu
            M = baseline.make_M()
            self.M[i] = M
            diff_M = baseline.make_diff_M()
            self.diff_M[i] = diff_M

    # Fit
    def clear_fit(self):
        self.is_fitted = False
        self.fitted_mu_param = None
        self.fit_residuals = None
        self.fit_log = None
        self.fit_estim = None

    def fit(self, process_path, analytic_sol=False, x_0=None,
            init_method='fo_feasible', param_init_args=None,
            n_iter=1000,
            solvers=None,
            solver_args=None, exact_grad=False, estimators=None,
            is_log_lse=False,
            is_grad_target=False, is_log_ixs=False, is_log_allocs=False,
            is_log_total_estimates=False, is_log_strata_estimates=False,
            n_exact_single=None, n_samples_adaptive_single=None,
            nonadaptive_sample_size_single=None, single_strfs=None,
            logger=None,
            logger_args=None, rng=None, seed=1234, verbose=False, clear=True,
            write=True, **kwargs):
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
        d = self.d
        # Random number generator
        rng = us.make_rng(rng=rng, seed=seed)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        list_times = process_path.list_times
        T_f = process_path.T_f
        # Poisson Solution if activated
        if analytic_sol:
            x = [None]*d
            for i in range(d):
                x[i] = self.baselines_vec[i].basis_mus[0].get_events_rate(
                    list_times[i])
            if write:
                self.is_fitted = True
                self.fitted_mu_param = x
            return x

        # Initialize mappings
        if param_init_args is None:
            param_init_args = {}
        if solver_args is None:
            solver_args = {}
        if logger_args is None:
            logger_args = {}
        logger_args['is_log_allocs'] = is_log_allocs
        logger_args['is_log_ixs'] = is_log_ixs

        # Model Bounds
        lower_bnds = self.get_param_lower_bounds()
        upper_bnds = self.get_param_upper_bounds()

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if x_0 is None:
            x_0 = [None]*d
            for ix in range(d):
                x_0[ix] = self.baselines_vec[ix].get_param_lower_bounds()+rng.uniform(low=0., high=1., size=self.vector_n_param[ix])

        # Load estimators config
        if estimators is None:
            if exact_grad:
                estimators = [PoissonExactEstim(is_grad_target=is_grad_target,
                                                is_log_ixs=is_log_ixs,
                                                is_log_allocs=is_log_allocs,
                                                is_log_total_estimates=is_log_total_estimates,
                                                is_log_strata_estimates=is_log_strata_estimates,
                                                is_log_lse=is_log_lse)
                              for k in range(d)]
            else:
                estimators = [PoissonStratEstim(is_grad_target=is_grad_target,
                                                is_log_ixs=is_log_ixs,
                                                is_log_allocs=is_log_allocs,
                                                is_log_total_estimates=is_log_total_estimates,
                                                is_log_strata_estimates=is_log_strata_estimates,
                                                is_log_lse=is_log_lse)
                              for k in range(d)]
        else:
            if issubclass(type(estimators), Estimator):
                estimators = [copy.deepcopy(estimators) for k in range(d)]
        # Initialize Estimators with training data
        for k in range(d):
            estimators[k].initialize(k, n_iter[k], self, process_path,
                                     n_exact_single=n_exact_single,
                                     n_samples_adaptive_single=n_samples_adaptive_single,
                                     nonadaptive_sample_size_single=nonadaptive_sample_size_single,
                                     single_strfs=single_strfs)

        # Initialize Solvers
        if solvers is None:
            solvers = [ADAM(**solver_args) for k in range(d)]
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]
            elif type(solvers) == str:
                if solvers == 'Momentum':
                    solvers = [Momentum(**solver_args) for k in range(d)]
                elif solvers == 'RMSprop':
                    solvers = [RMSprop(**solver_args) for k in range(d)]
                elif solvers == 'ADAM':
                    solvers = [ADAM(**solver_args) for k in range(d)]

        # Initialize logger
        logger = OptimLogger(d, n_iter, is_log_lse=is_log_lse, **logger_args)
        self.init_logger(logger)

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
                g_t = estimators[k].lse_k_grad_estimate(x_k, rng)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration
                x_k = solvers[k].iterate(t, x_k, g_t)
                # Project into space of parameters
                x_k = np.clip(x_k, lower_bounds_k, upper_bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        # Update logger
        logger.is_logged_estimators = True
        logger.estimator_types = [type(estimators[k]) for k in range(d)]
        if write:
            self.is_fitted = True
            self.fitted_mu_param = x
            self.fit_estim = estimators
            self.process_logs(logger)
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
            list_times[i] = self.baselines_vec[i].simulate(T_f, mu_param[i],
                                                           rng=rng)
        process_path = ProcessPath(list_times, T_f)
        return process_path

    # Evaluation
    def get_residuals(self,  process_path, mu_param=None, write=True,
                      verbose=False):
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
            residuals[i] = self.baselines_vec[i].get_residuals(process_path.list_times[i], mu_param[i])
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

    def plot_solver_path_seq(self, true_param=None, min_param=None,
                             axes=None, save=False, filename='image.png',
                             show=False, **kwargs):
        d = self.d
        if not self.is_fitted:
            raise ValueError("MHP must be fitted before plotting solver path")
        fit_log = self.fit_log
        n_iter = fit_log.n_iter
        param_names = self.param_names
        for i in range(d):
            for ix_param in range(self.vector_n_param[i]):
                fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                                         sharey=False, **kwargs)
                # Parameter
                axes[0].plot([fit_log.param_logs[i][n][ix_param]
                              for n in range(n_iter[i]+1)],
                             color='steelblue')
                if true_param is not None:
                    axes[0].axhline(true_param[i][ix_param],
                                   color='steelblue',
                                   linestyle='solid')
                if min_param is not None:
                    axes[0].axhline(min_param[i][ix_param],
                                    color='steelblue',
                                    linestyle='solid')
                axes[0].set(ylabel='Parameter')

                # Derivative
                axes[1].plot([fit_log.grad_logs[i][n][ix_param]
                              for n in range(n_iter[i])],
                             color='steelblue')
                axes[1].axhline(0., color='grey',
                               linestyle='dashed')
                axes[1].set(ylabel='Derivative')

                # Legend
                axes[0].set(xlabel='Iteration')
                axes[1].set(xlabel='Iteration')
                fig.suptitle('Updates of '+param_names[i][ix_param])
                fig.tight_layout()
                fig.show()

    # Logging
    def init_logger(self, logger):
        d = self.d
        n_iter = logger.n_iter
        n_param_k = self.n_param_k
        if logger.is_log_param:
            logger.param_logs = [np.zeros((n_iter[k]+1, n_param_k[k]))
                                 for k in range(d)]
        if logger.is_log_grad:
            logger.grad_logs = [np.zeros((n_iter[k], n_param_k[k]))
                                for k in range(d)]

    def process_logs(self, logger):
        d = self.d
        if logger.is_log_lse:
            for k in range(d):
                logger.lse[k] = self.fit_estim[k].logged_lse

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

        if file.endswith('.pickle'):
            file_fit_log = file+'_fit_log.pickle'
        else:
            file_fit_log = file+'_fit_log'
        pickle_out = open(file_fit_log, "wb", **kwargs)
        pickle.dump(self.fit_log, pickle_out)
        pickle_out.close()

        # Save fit estimators
        for k in range(self.d):
            suffix = 'estimator_'+str(k)
            if file.endswith('.pickle'):
                file_fit_estim_k = file[:-7]+'_'+suffix+'.pickle'
            else:
                file_fit_estim_k = file+'_'+suffix
            try:
                self.fit_estim[k].save(file_fit_estim_k)
            except:
                pass

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

        if file.endswith('.pickle'):
            file_fit_log = file+'_fit_log.pickle'
        else:
            file_fit_log = file+'_fit_log'
        try:
            pickle_in = open(file_fit_log, "rb")
            fit_log = pickle.load(pickle_in)
            self.fit_log = fit_log
        except:
            pass

    def load_fit_estimators(self, file, process_path, **kwargs):
        # Load fit estimators
        if (self.fit_log is not None) and (self.fit_log.is_logged_estimators):
            estimators = [None]*self.d
            for k in range(self.d):
                # Create object
                estimator_type = self.fit_log.estimator_types[k]
                if estimator_type == PoissonStratEstim:
                    estimators[k] = PoissonStratEstim()
                elif estimator_type == PoissonExactEstim:
                    estimators[k] = PoissonExactEstim()
                # Load value
                suffix = 'estimator_'+str(k)
                if file.endswith('.pickle'):
                    file_fit_estim_k = file[:-7]+'_'+suffix+'.pickle'
                else:
                    file_fit_estim_k = file+'_'+suffix
                estimators[k].load(file_fit_estim_k, self, process_path)
            self.fit_estim = estimators
