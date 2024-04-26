# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from aslsd.optimize.estimators.estimator import Estimator
from aslsd.optimize.estimators.mhp_exact_estimator import MHPExactEstim
from aslsd.optimize.estimators. mhp_stratified_estimator import MHPStratEstim
from aslsd.optimize.optim_logging.optim_logger import OptimLogger
from aslsd.optimize.solvers.solver import Solver
from aslsd.optimize.solvers.momentum import Momentum
from aslsd.optimize.solvers.rmsprop import RMSprop
from aslsd.optimize.solvers.adam import ADAM
from aslsd.stats.residual_analysis import goodness_of_fit as gof
from aslsd.utilities import graphic_tools as gt
from aslsd.utilities import useful_functions as uf
from aslsd.utilities import useful_numerics as un
from aslsd.utilities import useful_statistics as us
from aslsd.functionals.baselines.baseline import BaselineModel
from aslsd.functionals.baselines.\
    basis_baselines.basis_baseline_constant import ConstantBaseline
from aslsd.stats.events.path_event import PathEvent
from aslsd.stats.events.process_path import ProcessPath
from aslsd.stats.events.episodes import Episodes


class MHP:
    """
    Class for multivariate Hawkes processes (MHP) models.

    Let :math:`\\mathbf{N}` be a d-dimensional counting process with
    conditional intensity :math:`\\boldsymbol{\\lambda}`.
    We say that :math:`\\mathbf{N}` is a (linear) MHP if, for all
    :math:`i \\in[d]` and for all :math:`t \\geq 0`, we have

    .. math::
        \\lambda_{i}(t):=\\mu_{i}+\\sum_{j=1}^{d}
        \\sum_{\\left\\{m: t_{m}^{j}<t\\right\\}} \\phi_{ij}\\left(t-t_{m}^{j}\\right),

    where

    * :math:`\\forall i,j \\in [d],  \\phi_{ij}:[0,+\\infty) \\to [0,+\\infty)` is in :math:`L_{1}`. We write :math:`\\boldsymbol{\\phi}=(\\phi_{ij})_{i,j\\in [d]}`. The functions :math:`\\phi_{i j}` are called the kernels of the MHP;

    * :math:`\\forall i \\in[d], \\mu_{i}>0`. The floats :math:`\\mu_{i}` are called baseline intensities.

    For all :math:`i,j \\in [d]`, and for all :math:`t \\geq 0`, define

    .. math::
        \\psi_{ij}(t):=\\int_0^t \\phi_{ij}(u)\\mathrm{d}u.

    We write :math:`\\boldsymbol{\\psi}=(\\psi_{ij})_{ij\\in [d]}`.

    For all :math:`i,j,k \\in [d]`, and for all :math:`t,s \\geq 0`,
    define

    .. math::
        \\Upsilon_{ijk}(t,s):=\\int_0^t \\phi_{ki}(u)\\phi_{kj}(u+s)\\mathrm{d}u.

    We write :math:`\\boldsymbol{\\Upsilon}=(\\Upsilon_{ijk})_{ijk\\in [d]}`.

    In our implementation, the class attribute

    * `phi` denotes the matrix of kernel functions :math:`\\boldsymbol{\\phi}`, such that `phi[i][j]` is the kernel function :math:`\\phi_{ij}`;

    * `psi` denotes the matrix of primitive functions :math:`\\boldsymbol{\\psi}`, such that `psi[i][j]` is the primitive function :math:`\\psi_{ij}`;

    * `upsilon` denotes the tensor of correlation functions :math:`\\boldsymbol{\\Upsilon}`, such that `upsilon[i][j][k]` is the correlation function :math:`\\Upsilon_{ijk}`.

    Attributes
    ----------
    kernel_matrix : `list` of `list` of `KernelModel`
        Matrix of kernel models.

    d : `int`
        Dimension of the MHP.

    matrix_n_param : `list`
        Matrix of numbers of parameters per kernel.

    n_ker_param : `int`
        Total number of kernel parameters.

    ix_map : `list` of `list` of `dict`
        DESCRIPTION. The default is None.

    interval_map : `list` of `list` of `int`
        Matrix of indices of first and last parameters of kernels in the flat
        vector of parameters.

    mu_names : `list` of `str`
        List of names of baseline parameters.

    ker_param_names : `list`
        Tensor of names of kernel parameters.

    param_bounds : TYPE, optional
        Tensor of lower bounds of kernel parameters.

    phi : `list` of `list` of `function`
        Matrix of kernel functions.

    diff_phi : `list` of `list` of `function`
        Matrix of derivatives of kernel functions.

    psi : `list` of `list` of `function`
        Matrix of `psi` functions.

    diff_psi : `list` of `list` of `function`
        Matrix of derivatives of `psi` functions.

    upsilon : `list` of `list` of `list` of `function`
        Matrix of `upsilon` functions.

    diff_sim_upsilon : `list` of `list` of `function`
        Matrix of derivatives of auto-correlation `upsilon` functions.

    diff_cross_upsilon : `list` of `list` of `list` of `function`
        Matrix of derivatives of cross-correlation `upsilon` functions.

    is_fitted : `bool`
        True if the MHP has been fitted.

    fitted_mu : `numpy.ndarray`
        Fitted baseline.

    fitted_ker_param : `numpy.ndarray`
        Fitted kernel parameters.

    fit_residuals : `numpy.ndarray`
        Fit residuals.

    fitted_adjacency : `numpy.ndarray`
        Adjacency matrix for fitted kernel parameters.

    fit_log : `aslsd.OptimLogger`
        Fit log.

    """

    def __init__(self, _kernel_matrix):
        """
        Constructor of objects of class MHP.

        Parameters
        ----------
        _kernel_matrix : `list` of `list` of `KernelModel`
            Matrix of kernel models.

        """
        self.clear_fit()
        self.kernel_matrix = _kernel_matrix

    # Kernel matrix
    @property
    def kernel_matrix(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._kernel_matrix

    @kernel_matrix.setter
    def kernel_matrix(self, M):
        self._kernel_matrix = M

        self.d = len(M)

        self.matrix_n_param = self.get_n_param()
        self.n_ker_param = sum([sum(self.matrix_n_param[i])
                                for i in range(self.d)])

        ix_map, interval_map = self.make_maps()
        self.ix_map = ix_map
        self.interval_map = interval_map

        param_names = self.get_param_names()
        self.mu_names = param_names['mu']
        self.ker_param_names = param_names['kernels']
        self.param_lower_bounds = self.get_param_lower_bounds()
        self.param_upper_bounds = self.get_param_upper_bounds()
        self.make_functionals()

    @kernel_matrix.deleter
    def kernel_matrix(self):
        del self._kernel_matrix

    # N params
    def get_n_param(self):
        """
        Get the matrix of number of parameters per kernel model.

        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list` of `list` of `int`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        mat_n_param = [[self._kernel_matrix[i][j].n_param for j in range(d)]
                       for i in range(d)]
        return mat_n_param

    # Parameters map
    def make_maps(self):
        """
        Compute the mapping between the MHP parameter indices and the flattened
        vectors of parameters.

        Denote by :math:`d` the dimensionality of the MHP model. For all
        :math:`i, j \\in [d]`, denote the vector of parameters of the kernel
        :math:`\\phi_{i j}` by :math:`\\vartheta_{i j}`, the dimension of
        :math:`\\vartheta_{i j}` by :math:`\\rho_{i j}`, and the total number
        of event type :math:`k` parameters of the model by

        .. math::
            n^{(k)}_{\\text {param }}:=1+\\sum_{i=1}^{d} \\rho_{k i},
        where the additive term :math:`1` in the right hand side correspond to
        the baseline parameter.

        For all :math:`k \\in[d]`, concatenate the vectors of event type
        :math:`k` kernel parameters as

        .. math::
            \\vartheta_{k}^{\\top}=\\left(\\vartheta_{k 1}^{\\top}, \\ldots, \\vartheta_{k d}^{\\top}\\right),

        and define the vectors of event type :math:`k` MHP parameters
        :math:`\\theta_{k}^{\\top}=\\left(\\mu_{k}, \\vartheta_{k}^{\\top}\\right)`.
        Finally, define the vector of parameters :math:`\\theta` of the MHP by

        .. math::
            \\theta^{\\top}=\\left(\\theta_{1}^{\\top}, \\ldots, \\theta_{d}^{\\top}\\right) .

        The vector of event type :math:`k` MHP parameters, :math:`\\theta_{k}`,
        is the vector of dimension :math:`n^{(k)}_{\\text {param }}` containing
        the baseline parameter :math:`\\mu_{k}` and the parameters of all
        kernels :math:`(\\phi_{kj})_{j \\in [d]}`.

        The list `ix_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`i \\in n_k`, `ix_map[k][i]` is a dictionary with keys `ker` and
        `par`. The value `ix_map[k][i]['ker']` is an integer representing the
        index `j` such that the `i`-th parameter of the flat vector of
        parameters is a parameter of the kernel :math:`\\phi_{kj}`. By default,
        since the `0`-th parameter of the flat vector corresponds to the
        background rate :math:`\\mu_k`, we set `ix_map[k][0]['ker']` to `-1`.
        Now let `j=ix_map[k][i]['ker']`. The value `ix_map[k][i]['par']` is an
        integer representing the
        index `p` such that the `i`-th parameter of the flat vector of
        parameters is the `p`-th parameter of the kernel :math:`\\phi_{kj}`.
        The list `interval_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`j \\in [d]`, `interval_map[k][j]` is a list of two integers
        `[p,q]` such that `p` (resp. `q-1`) is the index of the first (resp.
        last) parameter of kernel :math:`\\phi_{kj}` in the flat vector of
        parameters.

        Returns
        -------
        ix_map : `list` of `list` of `dict`
            DESCRIPTION.
        interval_map : `list` of `list` of `int`
            Matrix of indices of first and last parameters of kernels in the
            flat vector of parameters.

        """
        d = self.d
        ix_map = [[None for i in range(1+sum(self.matrix_n_param[k]))]
                  for k in range(d)]
        interval_map = [[None for i in range(d)] for k in range(d)]
        for k in range(d):
            ix_map[k][0] = -1

            ix_ker = 0
            ix_param = 0
            ix_left = 1
            x = 1
            n_param_k = 1+sum(self.matrix_n_param[k])
            while x < n_param_k:
                ix_map[k][x] = {'ker': ix_ker, 'par': ix_param}
                if ix_param == self._kernel_matrix[k][ix_ker].n_param-1:
                    interval_map[k][ix_ker] = [ix_left, x+1]
                    ix_ker += 1
                    ix_param = 0
                    ix_left = x+1
                    x += 1
                else:
                    ix_param += 1
                    x += 1
        return ix_map, interval_map

    def tensor2matrix_params(self, tensor_param):
        """
        Convert the list of flat vectors of parameters to a vector of
        background rates `mu` and a matrix of kernel parameters `kernel_param`.

        Parameters
        ----------
        tensor_param : TYPE
            DESCRIPTION.

        Returns
        -------
        mu : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        d = self.d
        mu = np.array([tensor_param[i][0] for i in range(d)])
        kernel_param = np.array([[tensor_param[i][self.interval_map[i][j][0]:
                                                  self.interval_map[i][j][1]]
                                  for j in range(d)] for i in range(d)],
                                dtype=object)
        return mu, kernel_param

    def matrix2tensor_params(self, mu, kernel_param):
        """
        Convert the vector of background rates `mu` and the matrix of kernel
        parameters `kernel_param` to the list of flat vectors of parameters.

        Parameters
        ----------
        mu : TYPE
            Vector of baselines.
        kernel_param : TYPE
            Matrix of kernel parameters.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        d = self.d
        x = [None]*d
        for k in range(d):
            x_k = []
            # mu_k
            x_k.append(mu[k])
            # kernel parameters
            for i in range(d):
                x_k.extend(copy.deepcopy(kernel_param[k][i]))
            x[k] = np.array(copy.deepcopy(x_k), dtype=object)
        return np.array(x)

    def tensor2matrix_solverpaths(self, tensor_paths):
        #   Convert the list of parameters
        #   tensor_paths is a tensor of paths of solvers per parameters
        d = self.d
        list_n = [len(tensor_paths[k]) for k in range(d)]
        mu_paths = [[tensor_paths[k][n][0] for n in range(list_n[k])]
                    for k in range(d)]
        kernel_param_paths = [[[tensor_paths[i][n][self.interval_map[i][j][0]:
                                                   self.interval_map[i][j][1]]
                                for n in range(list_n[i])] for j in range(d)]
                              for i in range(d)]
        return mu_paths, kernel_param_paths

    def make_xk(self, k, mu_k=None, mu=None, ker_param=None):
        d = self.d
        if ker_param is None:
            ker_param = self.fitted_ker_param
        if mu_k is None:
            if mu is None:
                if self.fitted_mu is None:
                    mu = np.zeros(d)
                else:
                    mu = self.fitted_mu
        else:
            mu = np.zeros(d)
            mu[k] = mu_k
        x = self.matrix2tensor_params(mu, ker_param)
        x_k = x[k]
        return x_k

    def load_param(self, mu=None, kernel_param=None):
        if mu is None:
            mu = self.fitted_mu
            if mu is None:
                raise ValueError("Missing value for Mu")
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
            if kernel_param is None:
                raise ValueError("Missing value for Kernel parameters")
        mu = np.array(mu)
        return mu, kernel_param

    # Omega
    def is_sbf(self):
        d = self.d
        for i, j in itertools.product(range(d), range(d)):
            if not self._kernel_matrix[i][j].is_sbf():
                return False
        return True

    # Bounds
    def get_param_lower_bounds(self):
        d = self.d
        bnds = [[self._kernel_matrix[i][j].get_param_lower_bounds()
                 for j in range(d)] for i in range(d)]
        return bnds

    def get_param_upper_bounds(self):
        d = self.d
        bnds = [[self._kernel_matrix[i][j].get_param_upper_bounds()
                 for j in range(d)] for i in range(d)]
        return bnds

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
        param_names = {}
        if d == 1:
            mu_names = ['$\u03BC$']
            kernel = self._kernel_matrix[0][0]
            vec_names = kernel.get_vec_param_names()
            n_param = kernel.n_param
            ker_param_names = [[None for j in range(d)] for i in range(d)]
            ker_param_names[0][0] = [None]*n_param
            for ix_param in range(n_param):
                ix_ker = kernel.ix_map[ix_param]['ker']
                ix_2 = kernel.ix_map[ix_param]['par']
                if kernel.n_basis_ker == 1:
                    ker_param_names[0][0][ix_param] = vec_names[ix_ker][ix_2]
                else:
                    ker_param_names[0][0][ix_param] = (vec_names[ix_ker][ix_2][:-1] + '_{'+str(ix_ker+int(index_from_one))+'}$')
        else:
            mu_names = ['$\u03BC_{'+str(i+int(index_from_one))+'}$'
                        for i in range(d)]
            ker_param_names = [[None for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                kernel = self._kernel_matrix[i][j]
                vec_names = kernel.get_vec_param_names()
                n_param = kernel.n_param
                ker_param_names[i][j] = [None]*n_param
                if kernel.n_basis_ker == 1:
                    for ix_param in range(n_param):
                        ix_ker = kernel.ix_map[ix_param]['ker']
                        ix_2 = kernel.ix_map[ix_param]['par']
                        ker_param_names[i][j][ix_param] = vec_names[ix_ker][ix_2][:-1]+ '_{'+str(i+int(index_from_one))+','+str(j+int(index_from_one))+'}$'
                else:
                    for ix_param in range(n_param):
                        ix_ker = kernel.ix_map[ix_param]['ker']
                        ix_2 = kernel.ix_map[ix_param]['par']
                        ker_param_names[i][j][ix_param] = vec_names[ix_ker][ix_2][:-1]+ '_{'+str(i+int(index_from_one))+','+str(j+int(index_from_one))+','+str(ix_ker+int(index_from_one))+'}$'
        param_names['mu'] = mu_names
        param_names['kernels'] = ker_param_names
        return param_names

    # Kernel functionals
    def make_functionals(self):
        d = self.d
        # Baselines
        self.baselines_vec = [BaselineModel([ConstantBaseline()])
                              for i in range(d)]
        self.mu_compensator = [self.baselines_vec[i].compensator for i in range(d)]
        self.mu = [None for i in range(d)]
        self.diff_mu = [None for i in range(d)]
        self.M = [None for i in range(d)]
        self.diff_M = [None for i in range(d)]
        for i in range(d):
            baseline = self.baselines_vec[i]
            mu = baseline.make_mu()
            self.mu[i] = mu
            diff_mu = baseline.make_diff_mu()
            self.diff_mu[i] = diff_mu
            M = baseline.make_M()
            self.M[i] = M
            diff_M = baseline.make_diff_M()
            self.diff_M[i] = diff_M

        # Kernels
        self.phi = [[None for j in range(d)] for i in range(d)]
        self.diff_phi = [[None for j in range(d)] for i in range(d)]
        self.psi = [[None for j in range(d)] for i in range(d)]
        self.diff_psi = [[None for j in range(d)] for i in range(d)]
        self.upsilon = [[[None for k in range(d)] for j in range(d)]
                        for i in range(d)]
        self.diff_sim_upsilon = [[None for j in range(d)]
                                 for i in range(d)]
        self.diff_cross_upsilon = [[[None for k in range(d)]
                                    for j in range(d)] for i in range(d)]
        # Single Kernel functionals
        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            phi = kernel.make_phi()
            self.phi[i][j] = phi
            diff_phi = kernel.make_diff_phi()
            self.diff_phi[i][j] = diff_phi
            psi = kernel.make_psi()
            self.psi[i][j] = psi
            diff_psi = kernel.make_diff_psi()
            self.diff_psi[i][j] = diff_psi
            diff_sim_upsilon = kernel.make_diff_sim_upsilon()
            self.diff_sim_upsilon[i][j] = diff_sim_upsilon
        # Kernel interaction functionals
        for i, j, k in itertools.product(range(d), range(d), range(d)):
            kernel_ki = self._kernel_matrix[k][i]
            kernel_kj = self._kernel_matrix[k][j]
            if kernel_ki.is_compatible(kernel_kj):
                func = kernel_ki.make_upsilon()

                def upsilon(t, s, params_1, params_2):
                    return func(kernel_kj, t, s, params_1, params_2)
                self.upsilon[i][j][k] = upsilon
                diff_func = kernel_ki.make_diff_cross_upsilon()

                def diff_cross_upsilon(t, s, ix_func, ix_diff, params_1,
                                       params_2):
                    return diff_func(kernel_kj, t, s, ix_func, ix_diff,
                                     params_1, params_2)
                self.diff_cross_upsilon[i][j][k] = diff_cross_upsilon
            else:
                raise NotImplementedError("No available interaction"
                                          " between kernel", k, ",", i,
                                          " and kernel ", k, ",", j)

# =============================================================================
# Branching representation
# =============================================================================
    def make_adjacency_matrix(self, kernel_param=None):
        """
        Compute the adjacency matrix of the MHP.

        The adjacency matrix :math:`A` of an MHP is the :math:`d\\times d`
        matrix of :math:`L_{1}` norms of kernels; that is, for all
        :math:`i,j \\in [d]` the corresponding entry of this matrix is given by

        .. math::
            A_{ij} := \\int_{[0,+\\infty]} |\\phi_{ij}(u)|du..


        Parameters
        ----------
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters at which to evaluate the adjacency
            matrix. The default is None, in that case fitted kernel
            parameters will be used if they are stored in the corresponding
            attribute of the MHP object.

        Raises
        ------
        ValueError
            Raise an error if the kernel parameters not specified and there
            are no kernel parameters saved as an atrribute.

        Returns
        -------
        adjacency : `numpy.ndarray`
            Adjacency matrix of the MHP.

        """
        log_fitted_adjacency = False
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
                log_fitted_adjacency = True
            else:
                raise ValueError("kernel_param must be specified.")
        d = self.d
        adjacency = np.zeros((d, d))
        for i, j in itertools.product(range(d), range(d)):
            x_ker_ij = kernel_param[i][j]
            adjacency[i, j] = self._kernel_matrix[i][j].l1_norm(x_ker_ij)
        if log_fitted_adjacency:
            self.fitted_adjacency = adjacency
        return adjacency

    def get_branching_ratio(self, adjacency=None, kernel_param=None):
        """
        Compute the branching ratio of the MHP.

        The branching ratio of an MHP is equal to the spectral radius of
        its adjacency matrix; that is, the maximum of the absolute values of
        the eigenvalues of the adjacency matrix.

        Parameters
        ----------
        adjacency : `numpy.ndarray`, optional
            Adjacency matrix of the MHP. The default is None, in that case it
            will be computed.

        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters at which to evaluate the adjacency
            matrix. The default is None, in that case fitted kernel
            parameters will be used if they are stored in the corresponding
            attribute of the MHP object.

        Raises
        ------
        ValueError
            Raise an error if the adjacency matrix is not specified, the kernel
            parameters not specified and there are no kernel parameters saved
            as an atrribute.

        Returns
        -------
         adjacency : `float`
            Branching ratio of the MHP.

        """
        if adjacency is None:
            if kernel_param is None:
                if self.is_fitted:
                    kernel_param = self.fitted_ker_param
                else:
                    raise ValueError("kernel_param must be specified.")
            adjacency = self.make_adjacency_matrix(kernel_param)
        bratio = np.max(np.absolute(np.linalg.eigvals(adjacency)))
        return bratio

# =============================================================================
# Exact computation of loss functions (wrappers)
# =============================================================================
    def get_lse_k(self, k, process_path, mu=None, kernel_param=None,
                  verbose=False, initialize=False):
        if verbose:
            print('Computing partial LSE k=', k, '...')
        # Exact LSE
        d = self.d
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # Initialize Estimators
        estimators = [MHPExactEstim() for k in range(d)]

        for k in range(d):
            estimators[k].initialize(k, 10, self, process_path)
        x = self.matrix2tensor_params(mu, kernel_param)
        lse_k = estimators[k].lse_k_estimate(x[k], verbose=verbose)
        return lse_k

    def get_lse(self, process_path, mu=None, kernel_param=None,
                verbose=False, initialize=False):
        # Exact lse
        lse = 0.
        for k in range(self.d):
            lse += self.get_lse_k(k, process_path, mu=mu,
                                  kernel_param=kernel_param, verbose=verbose,
                                  initialize=initialize)
        return lse

# =============================================================================
# First order statistics
# =============================================================================
    def get_intensity_at_jumps(self, process_path, mu=None, kernel_param=None,
                               verbose=False):
        d = self.d
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)

        # Compute
        intensity = [mu[i]+np.zeros(process_path.n_events[i])
                     for i in range(d)]
        if verbose:
            print('Starting Computations...')
        for i, j in itertools.product(range(d), range(d)):
            for m in tqdm(range(process_path.varpi[i][j][1],
                                process_path.n_events[i]),
                          disable=not verbose):
                t_m = process_path.list_times[i][m]
                ix_bnd = process_path.kappa[j][i][m]+1
                t_n = process_path.list_times[j][:ix_bnd]
                t_diff = t_m-t_n
                intensity[i][m] += np.sum(self.phi[i][j](t_diff,
                                                         kernel_param[i][j]))
        return intensity

    def get_stationary_intensity(self, mu=None, adjacency=None,
                                 kernel_param=None):
        d = self.d
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        if adjacency is None:
            adjacency = self.make_adjacency_matrix(kernel_param)
        # LLN
        power_adj = np.linalg.inv(np.eye(d)-adjacency)
        eta_star = power_adj.dot(mu)
        return eta_star

    def get_exo_ratio(self, mu=None, kernel_param=None):
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # LLN
        eta_star = self.get_stationary_intensity(mu=mu,
                                                 kernel_param=kernel_param)
        l1_eta_star = np.sum(eta_star)
        l1_mu = np.sum(mu)
        exo_ratio = l1_mu/l1_eta_star
        return exo_ratio

    def get_mu_lln(self, eta_star, adjacency):
        d = self.d
        mu = (np.eye(d)-adjacency).dot(eta_star)
        return mu

    def estimate_eta_star(self, data):
        if type(data) == ProcessPath:
            return data.eta
        elif type(data) == Episodes:
            eta = data.average_eta
            return eta
        else:
            raise ValueError('data must be a aslsd.ProcessPath or a list of'
                             ' aslsd.ProcessPath objects.')

# =============================================================================
# Simulation
# =============================================================================
    def simulate(self, T_f, T_i=0., history=None, mu=None, kernel_param=None,
                 check_stability=True, adjacency=None, rng=None, seed=1234,
                 verbose=False):
        """
        Simulate a path of the MHP.

        Parameters
        ----------
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        check_stability : `bool`, optional
            If True, only simulate from the MHP if it is stable. The default
            is True.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline or the kernel parameters are not
            specified and there is no fitted baseline or kernel parameters
            saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of simulated jump times for each dimension.

        """
        d = self.d
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # Adjacency matrix
        if adjacency is None:
            adjacency = self.make_adjacency_matrix(kernel_param)
        if check_stability:
            branching_ratio = self.get_branching_ratio(adjacency=adjacency)
            if branching_ratio >= 1:
                raise ValueError("Cannot simulate from unstable MHP: ",
                                 "The branching ratio of this MHP is ",
                                 branching_ratio, " > 1.")
        # Offset generators
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        # Start simulation
        if verbose:
            print('Simulating events...')
        # Step 0. Intialise Generations
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen
        generations = [None]*d
        for i in range(d):
            if history is not None and len(history.list_times[i]) > 0:
                generations[i] = [copy.deepcopy(np.array(history.list_times[i]))]
            else:
                generations[i] = [[]]

        # Step 1. Generate immigrants
        # Number of immigrants
        Nim = rng.poisson(mu*(T_f-T_i))

        # Location of immigrants
        immigrants = [None]*d
        for i in range(d):
            immigrants[i] = rng.uniform(low=T_i, high=T_f, size=Nim[i])
            if len(generations[i][0]) == 0:
                generations[i][0] = copy.deepcopy(immigrants[i])
            else:
                generations[i][0] = np.concatenate((generations[i][0],
                                                    immigrants[i]))

        #   Step 2. Fill via repeated generations
        def sum_generation(L, index):
            return sum([len(L[i][index]) for i in range(d)])

        ix_gen = 1
        while sum_generation(generations, ix_gen-1):
            for k in range(d):
                generations[k].append(np.array([]))
            for j in range(d):
                # Simulate the offspring of the "ix_gen-1"th generation of
                # events of type j
                if len(generations[j][ix_gen-1]) > 0:
                    for i in range(d):
                        # Set number of offspring
                        Noff = rng.poisson(adjacency[i][j],
                                           size=len(generations[j][ix_gen-1]))
                        parenttimes = generations[j][ix_gen-1].repeat(Noff)
                        offsets = offset_gens[i][j](rng, N=Noff.sum())
                        offspringtime = parenttimes + offsets
                        generations[i][ix_gen] = np.append(generations[i][ix_gen], np.array([x for x in offspringtime if (x < T_f) and (x > T_i)]))
            ix_gen += 1
        list_times = [np.array(sorted([x for sublist in generations[i]
                                       for x in sublist])) for i in range(d)]
        # Filtering
        for i in range(d):
            list_times[i] = list_times[i][list_times[i] > T_i]
        # Simulation complete
        if verbose:
            n_tot = sum([len(L) for L in list_times])
            print('Simulation Complete, ', n_tot, ' events simulated.')
        process_path = ProcessPath(list_times, T_f)
        return process_path

    def get_baseline_events(self, T_f, T_i=0., history=None, mu=None,
                            kernel_param=None,
                            check_stability=True, adjacency=None, rng=None,
                            seed=1234,
                            verbose=False):
        d = self.d
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # Adjacency matrix
        if adjacency is None:
            adjacency = self.make_adjacency_matrix(kernel_param)
        if check_stability:
            branching_ratio = self.get_branching_ratio(adjacency=adjacency)
            if branching_ratio >= 1:
                raise ValueError("Cannot simulate from unstable MHP: ",
                                 "The branching ratio of this MHP is ",
                                 branching_ratio, " > 1.")
        # Offset generators
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        # Start simulation
        if verbose:
            print('Simulating events...')
        # Step 0. Intialise Generations
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen
        generations = [None]*d
        for i in range(d):
            if history is not None and len(history.list_times[i]) > 0:
                generations[i] = [copy.deepcopy(np.array(history.list_times[i]))]
            else:
                generations[i] = [[]]

        # Step 1. Generate immigrants
        # Number of immigrants
        Nim = rng.poisson(mu*(T_f-T_i))

        # Location of immigrants
        immigrants = [None]*d
        for i in range(d):
            immigrants[i] = rng.uniform(low=T_i, high=T_f, size=Nim[i])

        # Immigrants
        immigrants = [sorted(immigrants[i]) for i in range(d)]
        immigrants = [np.array(immigrants[i]) for i in range(d)]
        return immigrants

    def simulate_episodes(self, T_f, T_i=0., history=None, n_episodes=1,
                          mu=None,
                          kernel_param=None, check_stability=True,
                          adjacency=None,
                          rng=None, seed=1234,
                          verbose=False):
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # Adjacency matrix
        if adjacency is None:
            adjacency = self.make_adjacency_matrix(kernel_param)
        if check_stability:
            branching_ratio = self.get_branching_ratio(adjacency=adjacency)
            if branching_ratio >= 1:
                raise ValueError("Cannot simulate from unstable MHP: ",
                                 "The branching ratio of this MHP is ",
                                 branching_ratio, " > 1.")
        list_paths = [None]*n_episodes
        vec_seeds = rng.choice(max(10**7, n_episodes+1), size=n_episodes,
                               replace=False)
        for ix in tqdm(range(n_episodes), disable=not verbose):
            seed = vec_seeds[ix]
            list_paths[ix] = self.simulate(T_f, T_i=T_i, history=history,
                                           mu=mu, kernel_param=kernel_param,
                                           check_stability=False,
                                           adjacency=adjacency, rng=None,
                                           seed=seed, verbose=False)
        episodes = Episodes(list_paths=list_paths, T_f=T_f)
        return episodes

    def simu_multipath(self, path_res, t_res, x_min, x_max, mu=None,
                       kernel_param=None, rng=None, base_seed=1234,
                       verbose=False,
                       disc_type='log'):
        d = self.d
        # RNG
        rng = us.make_rng(rng=rng, seed=base_seed)
        vec_seeds = rng.choice(10**5, size=path_res, replace=False)

        if disc_type == 'log':
            T_f = 10**x_max
        elif disc_type == 'linear':
            T_f = x_max
        list_Tf = uf.discretize_space(x_min, x_max, t_res, disc_type)
        list_paths = [[[] for j in range(path_res)] for i in range(t_res)]
        for ix_path in range(path_res):
            seed = vec_seeds[ix_path]
            process_path = self.simulate(T_f, mu=mu, kernel_param=kernel_param,
                                         seed=seed, verbose=verbose)
            for ix_t in range(t_res):
                T_trunc = list_Tf[ix_t]
                trunc_path = process_path.truncate(T_trunc)
                list_paths[ix_t][ix_path] = trunc_path
        return list_Tf, list_paths

# =============================================================================
# Simulate descendants
# =============================================================================
    # Simulation
    def simulate_descendants(self, event, T_f,
                             kernel_param=None, book_keeping=False,
                             rng=None, seed=1234,
                             verbose=False):
        """
        Simulate descendants of an event that happened in t_src.

        Parameters
        ----------
        event : `aslsd.PathEvent`
            Source event.
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline or the kernel parameters are not
            specified and there is no fitted baseline or kernel parameters
            saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of simulated jump times for each dimension.

        """
        d = self.d
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=None, kernel_param=kernel_param)
        # Adjacency matrix
        adjacency = self.make_adjacency_matrix(kernel_param)
        branching_ratio = self.get_branching_ratio(adjacency=adjacency)
        if branching_ratio >= 1:
            raise ValueError("Cannot simulate from unstable MHP: ",
                             "The branching ratio of this MHP is ",
                             branching_ratio, " > 1.")
        # Offset generators
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        # Source event
        dim_src = event.dim
        t_src = event.time

        # Assert parent event is admissible
        if t_src > T_f:
            raise ValueError("Parent event cannot be posterior to terminal\
                             time")

        # Start simulation
        if verbose:
            print('Simulating events...')
        # Step 0. Intialise Generations
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen
        generations = [None]*d
        for i in range(d):
            if i == dim_src:
                generations[i] = [np.array([t_src])]
            else:
                generations[i] = [[]]

        #   Step 2. Fill via repeated generations
        def sum_generation(L, index):
            return sum([len(L[i][index]) for i in range(d)])

        ix_gen = 1
        while sum_generation(generations, ix_gen-1):
            for k in range(d):
                generations[k].append(np.array([]))
            for j in range(d):
                # Simulate the offspring of the "ix_gen-1"th generation of
                # events of type j
                if len(generations[j][ix_gen-1]) > 0:
                    for i in range(d):
                        # Set number of offspring
                        Noff = rng.poisson(adjacency[i][j],
                                           size=len(generations[j][ix_gen-1]))
                        parenttimes = generations[j][ix_gen-1].repeat(Noff)
                        offsets = offset_gens[i][j](rng, N=Noff.sum())
                        offspringtime = parenttimes + offsets
                        generations[i][ix_gen] = np.append(generations[i][ix_gen], np.array([x for x in offspringtime if (x < T_f)]))
            ix_gen += 1
        list_times = [np.array(sorted([x for sublist in generations[i]
                                       for x in sublist])) for i in range(d)]
        # Filtering w.r.t t_src
        valid_ixs = np.where(list_times[dim_src] > t_src)[0]
        list_times[dim_src] = list_times[dim_src][valid_ixs]

        # Simulation complete
        if verbose:
            n_tot = sum([len(L) for L in list_times])
            print('Simulation Complete, ', n_tot, ' events simulated.')
        process_path = ProcessPath(list_times, T_f, T_i=t_src,
                                   book_keeping=book_keeping)
        return process_path

    def simulate_descendants_multi(self, n_paths, event, T_f, T_i=0.,
                                   kernel_param=None, book_keeping=False,
                                   rng=None, base_seed=1234, verbose=False):
        """
        Simulate descendants of an event that happened in t_src.

        Parameters
        ----------
        event : `aslsd.PathEvent`
            Source event.
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline or the kernel parameters are not
            specified and there is no fitted baseline or kernel parameters
            saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of simulated jump times for each dimension.

        """
        # RNG
        rng = us.make_rng(rng=rng, seed=base_seed)
        vec_seeds = rng.choice(max(10**5, 10*n_paths), size=n_paths,
                               replace=False)
        desc_multi = [None]*n_paths
        # Prepare parameters
        for ix_path in tqdm(range(n_paths), disable=not verbose):
            seed = vec_seeds[ix_path]
            process_path = self.simulate_descendants(event, T_f,
                                                     kernel_param=kernel_param,
                                                     book_keeping=book_keeping,
                                                     seed=seed, verbose=False)
            desc_multi[ix_path] = copy.deepcopy(process_path)
        return desc_multi

# =============================================================================
# Simulate one step ahead
# =============================================================================
    def simulate_one_step(self, T_f, T_i=0., history=None, mu=None,
                          kernel_param=None, rng=None, seed=1234,
                          verbose=False):
        """
        Simulate a path of the MHP up to the first jump.

        Parameters
        ----------
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline or the kernel parameters are not
            specified and there is no fitted baseline or kernel parameters
            saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of simulated jump times for each dimension.

        """
        d = self.d
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Prepare parameters
        mu, kernel_param = self.load_param(mu=mu, kernel_param=kernel_param)
        # Adjacency matrix
        adjacency = self.make_adjacency_matrix(kernel_param)
        branching_ratio = self.get_branching_ratio(adjacency=adjacency)
        if branching_ratio >= 1:
            raise ValueError("Cannot simulate from unstable MHP: ",
                             "The branching ratio of this MHP is ",
                             branching_ratio, " > 1.")
        # Offset generators
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        # Start simulation
        if verbose:
            print('Simulating events...')
        # Step 0. Intialise Generations
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen
        generations = [None]*d
        for i in range(d):
            if history is not None and len(history.list_times[i]) > 0:
                generations[i] = [copy.deepcopy(np.array(history.list_times[i]))]
            else:
                generations[i] = [[]]

        # Step 1. Generate immigrants
        # Number of immigrants
        Nim = rng.poisson(mu*(T_f-T_i))

        # Location of immigrants
        immigrants = [None]*d
        for i in range(d):
            immigrants[i] = rng.uniform(low=T_i, high=T_f, size=Nim[i])
            if len(generations[i][0]) == 0:
                generations[i][0] = copy.deepcopy(immigrants[i])
            else:
                generations[i][0] = np.concatenate((generations[i][0],
                                                    immigrants[i]))

        #   Step 2. Fill for one generation
        for k in range(d):
            generations[k].append(np.array([]))
        for j in range(d):
            # Simulate the offspring of the 0th generation of
            # events of type j
            if len(generations[j][0]) > 0:
                for i in range(d):
                    # Set number of offspring
                    Noff = rng.poisson(adjacency[i][j],
                                       size=len(generations[j][0]))
                    parenttimes = generations[j][0].repeat(Noff)
                    offsets = offset_gens[i][j](rng, N=Noff.sum())
                    offspringtime = parenttimes + offsets
                    generations[i][1] = np.append(generations[i][1],
                                                  np.array([x for x in offspringtime
                                                            if (x < T_f)
                                                            and (x > T_i)]))

        list_times = [np.array(sorted([x for sublist in generations[i]
                                       for x in sublist])) for i in range(d)]
        # Filtering w.r.t. T_i
        for i in range(d):
            list_times[i] = list_times[i][list_times[i] > T_i]
        # Pick comparison candidate
        t_next = np.inf
        dim_next = 0
        for i in range(d):
            if len(list_times[i]) > 0:
                if list_times[i][0] < t_next:
                    t_next = list_times[i][0]
                    dim_next = i
        # Wrap as PathEvent object
        next_event = PathEvent(time=t_next, dim=dim_next)
        if verbose:
            print('Simulation Complete.')
        return next_event

# =============================================================================
# Fully random sampling of parameters
# =============================================================================
    def rescale_ker_param(self, kernel_param, target_bratio=0.6):
        d = self.d
        branching_ratio = self.get_branching_ratio(kernel_param=kernel_param)
        if branching_ratio > 0.:
            scaling = target_bratio/branching_ratio
        for i, j in itertools.product(range(d), range(d)):
            kernel_param[i][j] = np.array(kernel_param[i][j])
            if branching_ratio > 0.:
                vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                kernel_param[i][j][vec_ix_omega] *= scaling
        return kernel_param

    def make_max_param_array(self, float_max, max_omega=1.):
        # Make array of max param values
        d = self.d
        max_param = [[[None for x
                       in range(self._kernel_matrix[i][j].n_param)]
                      for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            n_param = self._kernel_matrix[i][j].n_param
            vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
            lower_bnds = self.param_lower_bounds[i][j]
            upper_bnds = self.param_upper_bounds[i][j]
            for x in range(n_param):
                if x in vec_ix_omega:
                    max_param[i][j][x] = max_omega
                else:
                    # If float_max < lower_bound, set max_param to lower_bound
                    max_param[i][j][x] = max(float_max, lower_bnds[x])
                    # clip the result to ensure it is less than upper_bound
                    max_param[i][j][x] = min(max_param[i][j][x], upper_bnds[x])
        return max_param

    def get_random_param(self, min_mu=0., max_mu=None, max_omega=1.,
                         max_param=5., target_bratio=0.6, seed=1234, rng=None,
                         flatten=False):
        rng = us.make_rng(rng=rng, seed=seed)
        d = self.d

        # Mu
        if not uf.is_array(min_mu):
            min_mu = np.ones(d)*min_mu
        if max_mu is None:
            max_mu = max(max(min_mu), 1.)
        if not uf.is_array(max_mu):
            max_mu = np.ones(d)*max_mu
        mu = np.zeros(d)
        for i in range(d):
            mu[i] = rng.uniform(low=min_mu[i], high=max_mu[i], size=1)[0]

        # Kernels
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if not uf.is_array(max_param):
            float_max = max_param
            max_param = self.make_max_param_array(float_max,
                                                  max_omega=max_omega)

        for i, j in itertools.product(range(d), range(d)):
            n_param = self._kernel_matrix[i][j].n_param
            kernel_param[i][j] = np.zeros(n_param)
            lower_bnds = self.param_lower_bounds[i][j]
            for ix in range(n_param):
                val = rng.uniform(low=lower_bnds[ix],
                                  high=max_param[i][j][ix],
                                  size=1)[0]
                kernel_param[i][j][ix] = val

        # Rescaling
        kernel_param = self.rescale_ker_param(kernel_param, target_bratio)

        # Flatten
        if flatten:
            return self.matrix2tensor_params(mu, kernel_param)
        else:
            return mu, kernel_param

# =============================================================================
# Random sampling of parameters wrt to reference values
# =============================================================================
    def get_random_mu_from_ref(self, ref_mu, range_ref=0.1,
                               rng=None, seed=1234):
        rng = us.make_rng(rng=rng, seed=seed)
        d = self.d
        mu = np.zeros(d)
        for i in range(d):
            mu[i] = rng.uniform(low=max(0., (1.-range_ref)*ref_mu[i]),
                                high=(1+range_ref)*ref_mu[i], size=1)[0]
        return mu

    def get_random_ker_from_ref(self, ref_ker_param, range_ref=0.1,
                                rng=None, seed=1234):
        rng = us.make_rng(rng=rng, seed=seed)
        d = self.d
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        for i, j in itertools.product(range(d), range(d)):
            n_param = self._kernel_matrix[i][j].n_param
            kernel_param[i][j] = np.zeros(n_param)
            lower_bnds = self.param_lower_bounds[i][j]
            upper_bnds = self.param_upper_bounds[i][j]
            for ix in range(n_param):
                lo = max(lower_bnds[ix],
                         (1.-range_ref)*ref_ker_param[i][j][ix])
                hi = min(upper_bnds[ix],
                         (1.+range_ref)*ref_ker_param[i][j][ix])
                val = rng.uniform(low=lo, high=hi, size=1)[0]
                kernel_param[i][j][ix] = val
        return kernel_param

    def get_random_param_from_ref(self, ref_mu=None, ref_ker_param=None,
                                  x_ref=None, range_ref=0.1,
                                  rng=None, seed=1234):
        if ref_mu is None:
            ref_mu, ref_ker_param = self.tensor2matrix_params(x_ref)
        mu = self.get_random_mu_from_ref(ref_mu, range_ref=range_ref,
                                         rng=rng, seed=seed)
        kernel_param = self.get_random_ker_from_ref(ref_ker_param,
                                                    range_ref=range_ref,
                                                    rng=rng, seed=seed)
        return mu, kernel_param

# =============================================================================
# Random sampling of parameters with first order feasibility
# =============================================================================
    def get_random_fo_1d(self, eta_star, a=None, b=None, rng=None, seed=1234):
        rng = us.make_rng(rng=rng, seed=seed)
        # Bounds
        if a is None:
            a = 0.3
        if b is None:
            b = 0.7
        a = min(a, b)
        # Sample omega
        omega_00 = rng.uniform(low=a, high=b)
        # Get mu from LLN
        mu_0 = (1.-omega_00)*eta_star[0]
        # Wrap results
        mu = np.array([mu_0])
        omega = np.array([[omega_00]])
        return mu, omega

    def get_random_fo_2d(self, eta_star, a=None, b=None, rng=None, seed=1234):
        rng = us.make_rng(rng=rng, seed=seed)
        # Bounds
        if a is None:
            a = np.zeros(2)
        if b is None:
            b = np.ones(2)
        omega = np.zeros((2, 2))
        # Stationary regime intensity
        r = eta_star[1]/eta_star[0]
        # Sample omega values, impact on dimension 0
        omega[0, 0] = rng.uniform(low=a[0], high=b[0])
        hi_bnd_0 = (1-omega[0, 0])/r
        omega[0, 1] = rng.uniform(low=0., high=hi_bnd_0)
        # Sample omega values, impact on dimension 1
        omega[1, 1] = rng.uniform(low=a[1], high=b[1])
        hi_bnd_1 = (1-omega[1, 1])*r
        omega[1, 0] = rng.uniform(low=0., high=hi_bnd_1)
        # Get mu from LLN
        mu = self.get_mu_lln(eta_star, omega)
        return mu, omega

    def get_random_fo_bivar(self, eta_star, a=None, b=None, rng=None,
                            seed=1234):
        rng = us.make_rng(rng=rng, seed=seed)
        d = self.d
        # Bounds
        if a is None:
            a = 0.
        if b is None:
            b = 1.
        # Stationary regime intensity
        m = np.min(eta_star)/np.sum(eta_star)
        r = m/(1.-m)
        # Sample omega_S
        omega_S = rng.uniform(low=a, high=b)
        # Sample omega_C
        omega_C = rng.uniform(low=0., high=r*(1.-omega_S))
        # Wrap omega
        omega = np.zeros((d, d))
        for i, j in itertools.product(range(d), range(d)):
            if i == j:
                omega[i, i] = omega_S
            else:
                omega[i, j] = omega_C
        # Get mu from LLN
        mu = self.get_mu_lln(eta_star, omega)
        return mu, omega

    def get_random_fo(self, eta_star, a=None, b=None, rng=None, seed=1234,
                      bivariate=False):
        d = self.d
        if d > 2:
            bivariate = True
        if bivariate:
            mu, omega = self.get_random_fo_bivar(eta_star, a=a, b=b, rng=rng,
                                                 seed=seed)
        else:
            if d == 1:
                mu, omega = self.get_random_fo_1d(eta_star, a=a, b=b, rng=rng,
                                                  seed=seed)
            elif d == 2:
                mu, omega = self.get_random_fo_2d(eta_star, a=a, b=b, rng=rng,
                                                  seed=seed)
        return mu, omega

    def get_random_param_with_fo(self, eta_star, a=None, b=None, rng=None,
                                 seed=1234, bivariate=False,
                                 max_density_param=5., flatten=False):
        rng = us.make_rng(rng=rng, seed=seed)
        mu, omega = self.get_random_fo(eta_star, a=a, b=b, rng=rng,
                                       bivariate=bivariate)
        # Kernel param
        d = self.d
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if not uf.is_array(max_density_param):
            float_max = max_density_param
            max_density_param = self.make_max_param_array(float_max,
                                                          max_omega=1.)
        for i, j in itertools.product(range(d), range(d)):
            n_param = self._kernel_matrix[i][j].n_param
            kernel_param[i, j] = np.zeros(n_param)
            # L_1 weights
            n_b = self._kernel_matrix[i][j].n_basis_ker
            l1_weights = us.get_random_mixture_weights(n_b, coeff=omega[i, j],
                                                       rng=rng, seed=1234)
            ix_omegas = self._kernel_matrix[i][j].ix_omegas()
            kernel_param[i, j][ix_omegas] = l1_weights+0.
            # Kernel densities parameters
            lower_bnds = self.param_lower_bounds[i][j]
            for ix in range(n_param):
                if ix not in ix_omegas:
                    val = rng.uniform(low=lower_bnds[ix],
                                      high=max_density_param[i][j][ix],
                                      size=1)[0]
                    kernel_param[i][j][ix] = val
        # Flatten
        if flatten:
            return self.matrix2tensor_params(mu, kernel_param)
        else:
            return mu, kernel_param

# =============================================================================
# Initialize parameters
# =============================================================================
    def initialize_param(self, init_method, data=None, rng=None, seed=1234,
                         flatten=False, **kwargs):
        kwargs['flatten'] = flatten
        rng = us.make_rng(rng=rng, seed=seed)
        kwargs['rng'] = rng
        if init_method == 'full_random':
            return self.get_random_param(**kwargs)
        elif init_method == 'from_ref':
            return self.get_random_param_from_ref(**kwargs)
        elif init_method == 'fo_feasible':
            eta_star = self.estimate_eta_star(data)
            kwargs['eta_star'] = eta_star
            return self.get_random_param_with_fo(**kwargs)

# =============================================================================
# Estimation
# =============================================================================
    # Estimator functions
    def init_estimator(self, estimator, k):
        # Ixs book-keeping
        estimator.n_param_k = 1+sum(self.matrix_n_param[k])
        estimator.matrix_n_param = self.matrix_n_param
        estimator.ix_map = self.ix_map
        estimator.interval_map = self.interval_map
        # Functionals
        estimator.phi = self.phi
        estimator.diff_phi = self.diff_phi
        estimator.psi = self.psi
        estimator.diff_psi = self.diff_psi
        estimator.upsilon = self.upsilon
        estimator.diff_sim_upsilon = self.diff_sim_upsilon
        estimator.diff_cross_upsilon = self.diff_cross_upsilon

    # Fit
    def init_logger(self, logger):
        d = self.d
        n_iter = logger.n_iter
        n_param_k = [1+sum(self.matrix_n_param[k]) for k in range(d)]

        if logger.is_log_param:
            logger.param_logs = [np.zeros((n_iter[k]+1, n_param_k[k]))
                                 for k in range(d)]
            logger.mu = [np.zeros(n_iter[k]+1) for k in range(d)]
            logger.ker = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                          for i in range(d)]
        if logger.is_log_grad:
            logger.grad_logs = [np.zeros((n_iter[k], n_param_k[k]))
                                for k in range(d)]
            logger.grad_mu = [[None for x in range(n_iter[k])]
                              for k in range(d)]
            logger.grad_ker = [[[None for x in range(n_iter[i])]
                                for j in range(d)] for i in range(d)]

        logger.mu_0 = None
        logger.ker_0 = None

    def process_logs(self, logger):
        d = self.d
        if logger.is_log_param:
            for i in range(d):
                for ix in range(logger.n_iter[i]+1):
                    logger.mu[i][ix] = logger.param_logs[i][ix][0]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]+1):
                    logger.ker[i][j][ix] = logger.param_logs[i][ix][self.interval_map[i][j][0]:self.interval_map[i][j][1]]
        if logger.is_log_grad:
            for i in range(d):
                for ix in range(logger.n_iter[i]):
                    logger.grad_mu[i][ix] = logger.grad_logs[i][ix][0]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]):
                    logger.grad_ker[i][j][ix] = logger.grad_logs[i][ix][self.interval_map[i][j][0]:self.interval_map[i][j][1]]
        if logger.is_log_lse:
            logger.lse = [None]*d
            for k in range(d):
                logger.lse[k] = self.fit_estim[k].logged_lse
        if logger.is_log_ixs:
            for k in range(d):
                logger.samples[k] = {}
                logger.samples[k]['psi'] = logger.estimator_logs[k]['samples']['psi']
                logger.samples[k]['upsilonzero'] = logger.estimator_logs[k]['samples']['upsilonzero']
                logger.samples[k]['phi'] = logger.estimator_logs[k]['samples']['phi']
                logger.samples[k]['upsilon'] = logger.estimator_logs[k]['samples']['upsilon']
        if logger.is_log_allocs:
            for k in range(d):
                logger.allocs[k] = {}
                logger.allocs[k] = logger.estimator_logs[k]['allocs']

    def clear_fit(self):
        """
        Delete all previously saved results and logs from the
        corresponding attributes of the MHP object.

        """
        self.is_fitted = False
        self.fitted_mu = None
        self.fitted_ker_param = None
        self.fit_residuals = None
        self.fitted_adjacency = None
        self.fit_log = None
        self.fit_estim = None

    def fit(self, process_path, x_0=None, init_method='fo_feasible',
            param_init_args=None, n_iter=1000, solvers=None, solver_args=None,
            exact_grad=False, estimators=None, is_log_lse=False,
            is_grad_target=False, is_log_ixs=False, is_log_allocs=False,
            is_log_total_estimates=False, is_log_strata_estimates=False,
            n_exact_single=None, n_samples_adaptive_single=None,
            nonadaptive_sample_size_single=None, single_strfs=None,
            n_samples_adaptive_double=None,
            nonadaptive_sample_size_double=None, double_strfs=None,
            logger=None,
            logger_args=None, rng=None, seed=1234, verbose=False, clear=True,
            write=True):
        """
        Fit the MHP model to some observations.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The least squares error (LSE) of this model for these observations is
        defined as

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\frac{1}{T} \\sum_{k=1}^{d} \\int_{0}^{T} \\lambda_{k}(t)^{2} \\mathrm{~d} t-\\frac{2}{T} \\sum_{k=1}^{d} \\sum_{m=1}^{N_{T}^{k}} \\lambda_{k}\\left(t_{m}^{k}\\right).

        For a homogeneous Poisson model, this simplifies to

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\sum_{k=1}^{d} \\bigg( \\mu_{k}^{2} -2 \\frac{N_{T}^{k}}{T} \\bigg).

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        x_0 : `list` of `numpy.ndarray`, optional
            Initial guess for parameters. The default is None.
        n_iter : `list` or `int`, optional
            n_iter[k] is the number of iterations of the the optimisation
            algorithm for problem k. If  n_iter is of type `int`, it will be
            converted to a d-dimensional array where each entry is equal to
            that integer. The default is 1000.
        solvers : `list` of `aslsd.Solver`, optional
            solvers[k] is the optimization solver for problem k. The default
            is None.
        estimators : `list` of `aslsd.Esimtator`, optional
            estimators[k] is the gradient estimator for problem k. The default
            is None.
        logger : `aslsd.OptimLogger`, optional
            DESCRIPTION. The default is None.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.
        clear : `bool`, optional
            If true, delete all previously saved results and logs from the
            corresponding attributes of the MHP object. The default is True.
        write : `bool`, optional
            If true, save the estimation results and logs in the corresponding
            attributes of the MHP object. The default is True.
        **kwargs : `dict`
            Additional keyword arguments.

        Returns
        -------
        fitted_mu : `numpy.ndarray`
            Fitted baselines.
        fitted_ker_param : `numpy.ndarray`
            Fitted kernel parameters.

        """
        d = self.d
        # Random number generator
        rng = us.make_rng(rng=rng, seed=seed)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Initialize mappings
        if param_init_args is None:
            param_init_args = {}
        if solver_args is None:
            solver_args = {}
        if logger_args is None:
            logger_args = {}
        logger_args['is_log_allocs'] = is_log_allocs
        logger_args['is_log_ixs'] = is_log_ixs

        # Model bounds
        mu_lower_bnds = 10**-10*np.ones(d)
        mu_upper_bnds = np.array([np.inf for k in range(d)])
        lower_bnds = self.matrix2tensor_params(mu_lower_bnds,
                                               self.param_lower_bounds)
        upper_bnds = self.matrix2tensor_params(mu_upper_bnds,
                                               self.param_upper_bounds)

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if verbose:
            print('Sampling initial guess...')
        if x_0 is None:
            mu_0, ker_0 = self.initialize_param(init_method, data=process_path,
                                                rng=rng, **param_init_args)
            x_0 = self.matrix2tensor_params(mu_0, ker_0)
        else:
            mu_0, ker_0 = self.tensor2matrix_params(x_0)

        # Initialize Estimators
        if estimators is None:
            if exact_grad:
                estimators = [MHPExactEstim(is_grad_target=is_grad_target,
                                            is_log_ixs=is_log_ixs,
                                            is_log_allocs=is_log_allocs,
                                            is_log_total_estimates=is_log_total_estimates,
                                            is_log_strata_estimates=is_log_strata_estimates,
                                            is_log_lse=is_log_lse)
                              for k in range(d)]
            else:
                estimators = [MHPStratEstim(is_grad_target=is_grad_target,
                                            is_log_ixs=is_log_ixs,
                                            is_log_allocs=is_log_allocs,
                                            is_log_total_estimates=is_log_total_estimates,
                                            is_log_strata_estimates=is_log_strata_estimates,
                                            is_log_lse=is_log_lse)
                              for k in range(d)]
        else:
            if issubclass(type(estimators), Estimator):
                estimators = [copy.deepcopy(estimators) for k in range(d)]
        for k in range(d):
            estimators[k].initialize(k, n_iter[k], self, process_path,
                                     n_exact_single=n_exact_single,
                                     n_samples_adaptive_single=n_samples_adaptive_single,
                                     nonadaptive_sample_size_single=nonadaptive_sample_size_single,
                                     single_strfs=single_strfs,
                                     n_samples_adaptive_double=n_samples_adaptive_double,
                                     nonadaptive_sample_size_double=nonadaptive_sample_size_double,
                                     double_strfs=double_strfs)

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
                g_t = estimators[k].lse_k_grad_estimate(x_k, rng=rng,
                                                        grad_alloc=is_grad_target)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration
                x_k = solvers[k].iterate(t, x_k, g_t)
                # Project into space of parameters
                x_k = np.clip(x_k, lower_bounds_k, upper_bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        fitted_mu, fitted_ker_param = self.tensor2matrix_params(x)
        # Update logger
        logger.is_logged_estimators = True
        logger.estimator_types = [type(estimators[k]) for k in range(d)]
        if write:
            self.is_fitted = True
            self.fitted_mu = fitted_mu
            self.fitted_ker_param = fitted_ker_param
            self.fit_estim = estimators
            self.process_logs(logger)
            logger.mu_0 = mu_0
            logger.ker_0 = ker_0
            self.fit_log = logger
        return fitted_mu, fitted_ker_param

    def fit_episodes(self, episodes, x_0=None, init_method='fo_feasible',
                     param_init_args=None, n_iter=1000, solvers=None,
                     solver_args=None, exact_grad=False, estimators=None,
                     is_log_lse=False, is_grad_target=False, is_log_ixs=False,
                     is_log_allocs=False, is_log_total_estimates=False,
                     is_log_strata_estimates=False, n_exact_single=None,
                     n_samples_adaptive_single=None,
                     nonadaptive_sample_size_single=None, single_strfs=None,
                     n_samples_adaptive_double=None,
                     nonadaptive_sample_size_double=None, double_strfs=None,
                     logger=None, logger_args=None,
                     rng=None, seed=1234, verbose=False, clear=True,
                     write=True):
        """
        Fit the MHP model to some observations.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The least squares error (LSE) of this model for these observations is
        defined as

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\frac{1}{T} \\sum_{k=1}^{d} \\int_{0}^{T} \\lambda_{k}(t)^{2} \\mathrm{~d} t-\\frac{2}{T} \\sum_{k=1}^{d} \\sum_{m=1}^{N_{T}^{k}} \\lambda_{k}\\left(t_{m}^{k}\\right).

        For a homogeneous Poisson model, this simplifies to

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\sum_{k=1}^{d} \\bigg( \\mu_{k}^{2} -2 \\frac{N_{T}^{k}}{T} \\bigg).

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        x_0 : `list` of `numpy.ndarray`, optional
            Initial guess for parameters. The default is None.
        n_iter : `list` or `int`, optional
            n_iter[k] is the number of iterations of the the optimisation
            algorithm for problem k. If  n_iter is of type `int`, it will be
            converted to a d-dimensional array where each entry is equal to
            that integer. The default is 1000.
        solvers : `list` of `aslsd.Solver`, optional
            solvers[k] is the optimization solver for problem k. The default
            is None.
        estimators : `list` of `aslsd.Esimtator`, optional
            estimators[k] is the gradient estimator for problem k. The default
            is None.
        logger : `aslsd.OptimLogger`, optional
            DESCRIPTION. The default is None.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.
        clear : `bool`, optional
            If true, delete all previously saved results and logs from the
            corresponding attributes of the MHP object. The default is True.
        write : `bool`, optional
            If true, save the estimation results and logs in the corresponding
            attributes of the MHP object. The default is True.
        **kwargs : `dict`
            Additional keyword arguments.

        Returns
        -------
        fitted_mu : `numpy.ndarray`
            Fitted baselines.
        fitted_ker_param : `numpy.ndarray`
            Fitted kernel parameters.

        """
        d = self.d
        n_episodes = episodes.n_episodes
        # Random number generator
        rng = us.make_rng(rng=rng, seed=seed)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Initialize mappings
        if param_init_args is None:
            param_init_args = {}
        if solver_args is None:
            solver_args = {}
        if logger_args is None:
            logger_args = {}

        # Model bounds
        mu_lower_bnds = 10**-10*np.ones(d)
        mu_upper_bnds = np.array([np.inf for k in range(d)])
        lower_bnds = self.matrix2tensor_params(mu_lower_bnds,
                                               self.param_lower_bounds)
        upper_bnds = self.matrix2tensor_params(mu_upper_bnds,
                                               self.param_upper_bounds)

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if verbose:
            print('Sampling initial guess...')
        if x_0 is None:
            mu_0, ker_0 = self.initialize_param(init_method, data=episodes,
                                                rng=rng, **param_init_args)
            x_0 = self.matrix2tensor_params(mu_0, ker_0)
        else:
            mu_0, ker_0 = self.tensor2matrix_params(x_0)

        # Initialize Estimators
        if estimators is None:
            if exact_grad:
                estimators = [[MHPExactEstim(is_grad_target=is_grad_target,
                                             is_log_ixs=is_log_ixs,
                                             is_log_allocs=is_log_allocs,
                                             is_log_total_estimates=is_log_total_estimates,
                                             is_log_strata_estimates=is_log_strata_estimates,
                                             is_log_lse=is_log_lse)
                               for ix_path in range(n_episodes)]
                              for k in range(d)]
            else:
                estimators = [[MHPStratEstim(is_grad_target=is_grad_target,
                                             is_log_ixs=is_log_ixs,
                                             is_log_allocs=is_log_allocs,
                                             is_log_total_estimates=is_log_total_estimates,
                                             is_log_strata_estimates=is_log_strata_estimates,
                                             is_log_lse=is_log_lse)
                               for ix_path in range(n_episodes)]
                              for k in range(d)]
        else:
            if issubclass(type(estimators), Estimator):
                estimators = [[copy.deepcopy(estimators)
                               for ix_path in range(n_episodes)]
                              for k in range(d)]
        for k in range(d):
            for ix_path in range(n_episodes):
                process_path = episodes.list_paths[ix_path]
                estimators[k][ix_path].initialize(k, n_iter[k], self,
                                                  process_path,
                                                  n_exact_single=n_exact_single,
                                                  n_samples_adaptive_single=n_samples_adaptive_single,
                                                  nonadaptive_sample_size_single=nonadaptive_sample_size_single,
                                                  single_strfs=single_strfs,
                                                  n_samples_adaptive_double=n_samples_adaptive_double,
                                                  nonadaptive_sample_size_double=nonadaptive_sample_size_double,
                                                  double_strfs=double_strfs)

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
        logger = OptimLogger(d, n_iter, **logger_args)
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
                g_per_path = np.zeros((n_episodes, len(x_k)))
                for ix in range(n_episodes):
                    g_per_path[ix] = estimators[k][ix].lse_k_grad_estimate(x_k,
                                                                           rng)
                g_t = np.mean(g_per_path, axis=0)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration
                x_k = solvers[k].iterate(t, x_k, g_t)
                # Project into space of parameters
                x_k = np.clip(x_k, lower_bounds_k, upper_bounds_k)
                logger.log_param(k, t+1, x_k)
            # esimator_k_log = estimators[k].get_log()
            # logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        fitted_mu, fitted_ker_param = self.tensor2matrix_params(x)
        if write:
            self.is_fitted = True
            self.fitted_mu = fitted_mu
            self.fitted_ker_param = fitted_ker_param
            self.process_logs(logger)
            logger.mu_0 = mu_0
            logger.ker_0 = ker_0
            self.fit_log = logger
        return fitted_mu, fitted_ker_param
# =============================================================================
# Goodness of fit
# =============================================================================
    # Residuals
    def get_residuals(self, process_path, mu=None, kernel_param=None,
                      sampling=False, sample_size=10**3, seed=1234, write=True,
                      verbose=False):
        """
        Compute the residuals of the model.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        Let :math:`k \\in [d]`, define the compensator of :math:`N^k`
        for all :math:`t \\geq 0` by

        .. math::
            \\Lambda_{k}(t):=\\int_{[0,t]}\\lambda_k(t)\\mathrm{~d} t.

        For all :math:`m \\in \\mathbb{N}^{*}, k \\in [d]`, let

        .. math::
            s_{m}^{k}=\\Lambda_{k}\\left(t_{m}^{k}\\right).

        For each :math:`k \\in[d]`, define the point process

        .. math::
            \\mathcal{S}^{k}:=\\left\\{s_{m}^{k}: m \\in \\mathbb{N}^{*}\\right\\};

        then :math:`\\left(\mathcal{S}^{k}\\right)_{k \\in[d]}` are independent
        standard Poisson processes. The inter-arrival times of
        :math:`\\mathcal{S}^{k}` ('residuals'), for a model that fits the data
        well must therefore be close to a standard exponential distribution.        

        Parameters
        ----------
        process_path : `aslsd.ProcessPath`
            DESCRIPTION.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        sampling : `bool`, optional
            If True, subsample the residuals. The default is False.
        sample_size : `int`, optional
            Size of the subsample of residuals. The default is 10**3. Only
            used if sampling is True.
        seed : `int`, optional
            Seed of the random number generator. The default is 1234. Only
            used if sampling is true.
        write : `bool`, optional
            If true, save computed residuals in the corresponding
            attributes of the MHP object. The default is True.
        verbose : `bool`, optional
            If True, print progress bar. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        residuals : `list` of `numpy.ndarray`
            Residuals of the fitted model.

        """
        if mu is None or kernel_param is None:
            if self.is_fitted:
                mu = self.fitted_mu
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("Both mu and kernel_param must be specified.")
        residuals = gof.get_residuals_mhp(process_path, self.psi, mu,
                                      kernel_param, sampling=sampling,
                                      sample_size=sample_size, seed=seed,
                                      verbose=verbose)
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
                display_line45=True, log_scale=False, ax=None, save=False,
                filename='image.png', show=False, **kwargs):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.qq_plot(residuals[i], n_models=1, labels=labels,
                           style=style, substract_yx=substract_yx,
                           normalize=normalize, max_points=max_points,
                           display_line45=display_line45, log_scale=log_scale,
                           ax=ax, save=save, filename=filename, show=show,
                           **kwargs)

# =============================================================================
# Metrics
# =============================================================================
    # L2
    def get_l2_projection(self, mhp_2, param_2, n_iter=1000, try_sbf=True,
                          solver=None, log_error=False, rng=None,
                          seed=1234,
                          verbose=False, **kwargs):
        d = self.d
        rng = us.make_rng(rng=rng, seed=seed)
        # Model
        bnds = self.param_lower_bounds

        # Initialisation
        ref_mu = kwargs.get('ref_mu', None)
        ref_ker_param = kwargs.get('ref_ker_param', None)
        range_ref = kwargs.get('range_ref', 0.1)
        target_bratio = kwargs.get('target_bratio', 0.6)
        max_omega = kwargs.get('max_omega', 1.)
        true_omega = kwargs.get('true_omega', None)
        max_param = kwargs.get('max_param', 5.)
        min_mu = kwargs.get('min_mu', 0.)
        max_mu = kwargs.get('max_mu', None)
        mu_0, ker_0 = self.get_random_param(min_mu=min_mu, max_mu=max_mu,
                                            max_omega=max_omega,
                                            max_param=max_param,
                                            target_bratio=target_bratio,
                                            seed=1234,
                                            flatten=False, rng=rng)
        param_1 = [[None for j in range(d)] for i in range(d)]
        error = np.zeros((d, d))
        if solver is None:
            solver = ADAM(**kwargs)

        if log_error:
            l2_err_log = [[np.zeros(n_iter) for j in range(d)]
                          for i in range(d)]
        else:
            l2_err_log = None
        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            kernel_2 = mhp_2._kernel_matrix[i][j]
            ker_param_2 = param_2[i][j]
            res_ij = kernel.get_l2_projection(kernel_2, ker_param_2,
                                              n_iter=n_iter,
                                              params_0=ker_0[i][j],
                                              try_sbf=try_sbf,
                                              solver=solver,
                                              log_error=log_error,
                                              rng=ker_0[i][j],
                                              verbose=verbose, **kwargs)
            param_1[i][j] = copy.deepcopy(res_ij['params'])
            if log_error:
                l2_err_log[i][j] = copy.deepcopy(res_ij['log'])
            error[i, j] = res_ij['error']
        res = {'params': param_1, 'error': error, 'log': l2_err_log}
        return res

    def get_l2_dist(self, mhp_2, ker_param_1=None, ker_param_2=None,
                    weights=None):
        if ker_param_1 is None:
            ker_param_1 = self.fitted_ker_param
        if ker_param_2 is None:
            ker_param_2 = mhp_2.fitted_ker_param
        adjacency_1 = self.make_adjacency_matrix(kernel_param=ker_param_1)
        adjacency_2 = mhp_2.make_adjacency_matrix(kernel_param=ker_param_2)
        d = self.d
        matrix_l2 = np.zeros((d, d))
        # Compute matrix of Wasserstein distances
        for i in range(1, d):
            for j in range(i):
                if adjacency_1[i][j] > 0. and adjacency_2[i][j] > 0.:
                    kernel_1 = self._kernel_matrix[i][j]
                    kernel_2 = mhp_2._kernel_matrix[i][j]
                    matrix_l2[i][j] = kernel_1.l2_dist(kernel_2,
                                                       ker_param_1[i][j],
                                                       ker_param_2[i][j])
        matrix_l2 = matrix_l2+matrix_l2.T
        # Linear weights
        if weights is None:
            weights = np.ones((d, d))
        Q = weights*matrix_l2
        return np.sum(Q)

    # Wasserstein
    def get_wass_dist(self, mhp_2, ker_param_1=None, ker_param_2=None,
                      weights=None, wass_grid=None):
        if ker_param_1 is None:
            ker_param_1 = self.fitted_ker_param
        if ker_param_2 is None:
            ker_param_2 = mhp_2.fitted_ker_param
        adjacency_1 = self.make_adjacency_matrix(kernel_param=ker_param_1)
        adjacency_2 = mhp_2.make_adjacency_matrix(kernel_param=ker_param_2)
        d = self.d
        matrix_wass = np.zeros((d, d))
        if wass_grid is None:
            wass_grid = np.linspace(0., 50, 10**4)
        # Compute matrix of Wasserstein distances
        for i in range(1, d):
            for j in range(i):
                if adjacency_1[i][j] > 0. and adjacency_2[i][j] > 0.:
                    def make_funcs():
                        def tilde_phi_1(t):
                            phi_vals = self.phi[i][j](t, ker_param_1[i][j])
                            return phi_vals/adjacency_1[i][j]

                        def tilde_phi_2(t):
                            phi_vals = mhp_2.phi[i][j](t, ker_param_1[i][j])
                            return phi_vals/adjacency_2[i][j]
                        return tilde_phi_1, tilde_phi_2
                    tilde_phi_1, tilde_phi_2 = make_funcs()
                    matrix_wass[i][j] = un.wass(tilde_phi_1, tilde_phi_2,
                                                wass_grid)
        matrix_wass = matrix_wass+matrix_wass.T
        # Linear weights
        if weights is None:
            weights = np.ones((d, d))
        Q = weights*matrix_wass
        return np.sum(Q)

# =============================================================================
# Graphic methods
# =============================================================================
    def plot_kernels(self, kernel_param=None, t_min=0., t_max=10.,
                     n_samples=1000, index_from_one=False, log_scale=False,
                     axs=None, save=False, filename='image.png',
                     show=False, **kwargs):
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("kernel_param must be specified.")
        return gt.plot_kernels(self.phi, kernel_param, t_min=t_min,
                               t_max=t_max, n_samples=n_samples,
                               index_from_one=index_from_one,
                               log_scale=log_scale, axs=axs,
                               save=save, filename=filename, show=show,
                               **kwargs)

    def plot_adjacency_matrix(self, adjacency=None, kernel_param=None,
                              event_names=None,
                              index_from_one=False, annotate=False,
                              cmap="Blues", save=False,
                              filename='image.png', show=True, **kwargs):
        if adjacency is None:
            if self.is_fitted:
                if self.fitted_adjacency is None:
                    if kernel_param is None:
                        kernel_param = self.fitted_ker_param
                    adjacency = self.make_adjacency_matrix()
                    self.fitted_adjacency = adjacency
                else:
                    adjacency = self.fitted_adjacency
            else:
                if kernel_param is not None:
                    adjacency = self.make_adjacency_matrix(kernel_param)
                else:
                    raise ValueError("adjacency must be specified.")

        return gt.plot_adjacency_matrix(adjacency, event_names=event_names,
                                        index_from_one=index_from_one,
                                        annotate=annotate, cmap=cmap,
                                        save=save, filename=filename,
                                        show=show, **kwargs)

    def plot_solver_path(self, true_mu=None, true_ker_param=None, min_mu=None,
                         min_ker_param=None, plot_derivatives=False,
                         derivatives_zero=False, axes=None, save=False,
                         filename='image.png', show=False, **kwargs):
        if not self.is_fitted:
            raise ValueError("MHP must be fitted before plotting solver path")
        fit_log = self.fit_log
        matrix_n_param = self.matrix_n_param
        mu_names = self.mu_names
        ker_param_names = self.ker_param_names

        return gt.plot_solver_path(fit_log, matrix_n_param, mu_names,
                                   ker_param_names, true_mu=true_mu,
                                   true_ker_param=true_ker_param,
                                   min_mu=min_mu, min_ker_param=min_ker_param,
                                   plot_derivatives=plot_derivatives,
                                   derivatives_zero=derivatives_zero,
                                   axes=axes, save=save, filename=filename,
                                   show=show, **kwargs)

    def plot_solver_path_seq(self, true_mu=None, true_ker_param=None,
                             min_mu=None, min_ker_param=None, axes=None,
                             save=False,
                             filename='image.png', show=False, **kwargs):
        d = self.d
        if not self.is_fitted:
            raise ValueError("MHP must be fitted before plotting solver path")
        fit_log = self.fit_log
        n_iter = fit_log.n_iter
        matrix_n_param = self.matrix_n_param
        mu_names = self.mu_names
        ker_param_names = self.ker_param_names

        # Mu
        for i in range(d):
            fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                                     sharey=False, **kwargs)
            # Parameter
            axes[0].plot(fit_log.mu[i], color=gt.standard_colors[0])
            if true_mu is not None:
                axes[0].hlines(true_mu[i], 0, n_iter[i]+1,
                               colors=gt.standard_colors[1],
                               linestyles='solid')
            if min_mu is not None:
                axes[0].hlines(min_mu[i], 0, n_iter[i]+1,
                               colors=gt.standard_colors[2],
                               linestyles='solid')
            axes[0].set(ylabel='Parameter')

            # Derivative
            axes[1].plot(fit_log.grad_mu[i], color=gt.standard_colors[0])
            axes[1].hlines(0., 0, n_iter[i], colors='grey',
                           linestyles='dashed')
            axes[1].set(ylabel='Derivative')

            # Legend
            axes[0].set(xlabel='Iteration')
            axes[1].set(xlabel='Iteration')
            fig.suptitle('Updates of '+mu_names[i]+' (baseline '+str(i)+')')
            fig.tight_layout()
            fig.show()

        # Kernel Parameters
        for i, j in itertools.product(range(d), range(d)):
            for ix_param in range(matrix_n_param[i][j]):
                fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True,
                                         sharey=False, **kwargs)
                # Parameter
                axes[0].plot([fit_log.ker[i][j][n][ix_param]
                              for n in range(n_iter[i]+1)],
                             color=gt.standard_colors[0])
                if true_ker_param is not None:
                    axes[0].hlines(true_ker_param[i][j][ix_param],
                                   0, n_iter[i]+1,
                                   colors=gt.standard_colors[1],
                                   linestyles='solid')
                if min_ker_param is not None:
                    axes[0].hlines(min_ker_param[i][j][ix_param], 0,
                                   n_iter[i]+1,
                                   colors=gt.standard_colors[2],
                                   linestyles='solid')
                axes[0].set(ylabel='Parameter')

                # Derivative
                axes[1].plot([fit_log.grad_ker[i][j][n][ix_param]
                              for n in range(n_iter[i])],
                             color=gt.standard_colors[0])
                axes[1].hlines(0., 0, n_iter[i], colors='grey',
                               linestyles='dashed')
                axes[1].set(ylabel='Derivative')
                # Legend
                axes[0].set(xlabel='Iteration')
                axes[1].set(xlabel='Iteration')
                fig.suptitle('Updates of '+ker_param_names[i][j][ix_param]
                             + ' (kernel '+str(i)+'←'+str(j)+')')
                fig.tight_layout()
                fig.show()

# =============================================================================
# Serialization
# =============================================================================
    def save(self, file, **kwargs):
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_out = open(file_mu, "wb", **kwargs)
        pickle.dump(self.fitted_mu, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        pickle_out = open(file_ker, "wb", **kwargs)
        pickle.dump(self.fitted_ker_param, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        pickle_out = open(file_residuals, "wb", **kwargs)
        pickle.dump(self.fit_residuals, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        pickle_out = open(file_adjacency, "wb", **kwargs)
        pickle.dump(self.fitted_adjacency, pickle_out)
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
        fitted_mu = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        file_ker = file+'_fitted_ker'
        pickle_in = open(file_ker, "rb")
        fitted_ker_param = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        pickle_in = open(file_residuals, "rb")
        fitted_residuals = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        pickle_in = open(file_adjacency, "rb")
        fitted_adjacency = pickle.load(pickle_in)

        self.clear_fit()

        self.is_fitted = True
        self.fitted_mu = fitted_mu
        self.fitted_ker_param = fitted_ker_param
        self.fit_residuals = fitted_residuals
        self.fitted_adjacency = fitted_adjacency
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
                if estimator_type == MHPStratEstim:
                    estimators[k] = MHPStratEstim()
                elif estimator_type == MHPExactEstim:
                    estimators[k] = MHPExactEstim()
                # Load value
                suffix = 'estimator_'+str(k)
                if file.endswith('.pickle'):
                    file_fit_estim_k = file[:-7]+'_'+suffix+'.pickle'
                else:
                    file_fit_estim_k = file+'_'+suffix
                estimators[k].load(file_fit_estim_k, self, process_path)
            self.fit_estim = estimators
