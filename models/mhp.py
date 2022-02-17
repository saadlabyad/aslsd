# License: BSD 3 clause

import bisect
import copy
import itertools

import numpy as np
from tqdm import tqdm

from aslsd.estimators.adaptive_stratified_estimator import AdaptiveStratified
from aslsd.estimators.estimator import Estimator
from aslsd.evaluation import goodness_of_fit as gof
from aslsd.events.process_path import ProcessPath
from aslsd.optim_logging.optim_logger import OptimLogger
from aslsd.plots import graphic_tools as gt
from aslsd.solvers.adam import ADAM
from aslsd.solvers.solver import Solver
from aslsd.utilities import useful_functions as uf


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
        DESCRIPTION. The default is None.

    diff_phi : `list` of `list` of `function`
        DESCRIPTION. The default is None.

    psi : `list` of `list` of `function`
        DESCRIPTION. The default is None.

    diff_psi : `list` of `list` of `function`
        DESCRIPTION. The default is None.

    upsilon : `list` of `list` of `list` of `function`
        DESCRIPTION. The default is None.

    diff_sim_upsilon : `list` of `list` of `function`
        DESCRIPTION. The default is None.

    diff_cross_upsilon : `list` of `list` of `list` of `function`
        DESCRIPTION. The default is None.

    is_fitted : `bool`
        DESCRIPTION. The default is False.

    fitted_mu : `numpy.ndarray`
        Fitted baseline.

    fitted_ker_param : `numpy.ndarray`
        Fitted kernel parameters.

    fit_residuals : `numpy.ndarray`
        Fit residuals.

    fitted_adjacency : `numpy.ndarray`
        Adjacency matrix for fitted kernel parameters.

    fit_log : TYPE, optional
        DESCRIPTION. The default is None.

    """

    def __init__(self, _kernel_matrix, d=None, matrix_n_param=None,
                 n_ker_param=None, ix_map=None, interval_map=None,
                 mu_names=None, ker_param_names=None,
                 param_bounds=None, phi=None, diff_phi=None, psi=None,
                 diff_psi=None, upsilon=None, diff_sim_upsilon=None,
                 diff_cross_upsilon=None, is_fitted=False,
                 fitted_mu=None, fitted_ker_param=None,
                 fit_residuals=None, fitted_adjacency=None, fit_log=None):
        """
        Constructor of objects of class MHP.

        Parameters
        ----------
        _kernel_matrix : `list` of `list` of `KernelModel`
            Matrix of kernel models.

        """
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
        self.param_bounds = self.get_param_bounds()
        self.make_kernel_functionals()

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
        Get lists storing the mapping for parameter indices.
        Denote by :math:`d` the dimensionality of the MHP model. For all
        :math:`k \\in [d]`, denote by :math:`n_k` the total number of
        parameters of all kernels :math:`(\\phi_{kj})_{j \\in [d]}`. We flatten
        The list
        `ix_map` is such that for all :math:`k \\in [d]`, and for all
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
            DESCRIPTION.

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

    # Omega
    def is_sbf(self):
        d = self.d
        for i, j in itertools.product(range(d), range(d)):
            if not self._kernel_matrix[i][j].is_sbf():
                return False
        return True

    # Bounds
    def get_param_bounds(self):
        d = self.d
        bnds = [[self._kernel_matrix[i][j].get_param_bounds()
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
    def make_kernel_functionals(self):
        d = self.d
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

    # Fit
    def clear_fit(self):
        self.is_fitted = False
        self.fitted_mu = None
        self.fitted_ker_param = None
        self.fit_residuals = None
        self.fitted_adjacency = None
        self.fit_log = None

    def fit(self, list_times, T_f, kappa=None, varpi=None, x_0=None,
            n_iter=1000, solvers=None, estimators=None, logger=None, seed=1234,
            verbose=False, clear=True, write=True, **kwargs):
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
        list_times : TYPE
            DESCRIPTION.
        T_f : TYPE
            DESCRIPTION.
        kappa : TYPE, optional
            DESCRIPTION. The default is None.
        varpi : TYPE, optional
            DESCRIPTION. The default is None.
        x_0 : TYPE, optional
            DESCRIPTION. The default is None.
        n_iter : TYPE, optional
            DESCRIPTION. The default is 1000.
        solvers : TYPE, optional
            DESCRIPTION. The default is None.
        estimators : TYPE, optional
            DESCRIPTION. The default is None.
        logger : TYPE, optional
            DESCRIPTION. The default is None.
        seed : TYPE, optional
            DESCRIPTION. The default is 1234.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.
        clear : TYPE, optional
            DESCRIPTION. The default is True.
        write : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fitted_mu : TYPE
            DESCRIPTION.
        fitted_ker_param : TYPE
            DESCRIPTION.

        """
        rng = np.random.default_rng(seed)

        # Clear saved data in case already fitted
        self.clear_fit()

        # Data
        d = self.d
        process_path = ProcessPath(list_times, T_f, kappa=kappa, varpi=varpi)

        # Model
        mu_bnds = [10**-10 for k in range(d)]
        bnds = self.matrix2tensor_params(mu_bnds, self.param_bounds)

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if x_0 is None:
            ref_mu = kwargs.get('ref_mu', None)
            ref_ker_param = kwargs.get('ref_ker_param', None)
            range_ref = kwargs.get('range_ref', 0.1)
            target_bratio = kwargs.get('target_bratio', 0.6)
            max_omega = kwargs.get('max_omega', 1.)
            true_omega = kwargs.get('true_omega', None)
            max_param = kwargs.get('max_param', 5.)
            min_mu = kwargs.get('min_mu', 0.)
            max_mu = kwargs.get('max_mu', None)
            mu_0, ker_0 = self.get_random_param(ref_mu=ref_mu,
                                                ref_ker_param=ref_ker_param,
                                                range_ref=range_ref,
                                                target_bratio=target_bratio,
                                                max_omega=max_omega,
                                                true_omega=true_omega,
                                                max_param=max_param,
                                                min_mu=min_mu, max_mu=max_mu,
                                                flatten=False, rng=rng)
            x_0 = self.matrix2tensor_params(mu_0, ker_0)
        else:
            mu_0, ker_0 = self.tensor2matrix_params(x_0)

        # Initialize Estimators
        if estimators is None:
            estimators = [AdaptiveStratified(**kwargs) for k in range(d)]
        else:
            if issubclass(type(estimators), Estimator):
                estimators = [copy.deepcopy(estimators) for k in range(d)]
        for k in range(d):
            estimators[k].k = k
            estimators[k].n_iter = n_iter[k]
            estimators[k].initialize(process_path, self)
            estimators[k].intialize_logs()
            estimators[k].set_stratification(**kwargs)

        # Initialize Solvers
        if solvers is None:
            solvers = ADAM(**kwargs)
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]

        # Initialize logger
        logger = OptimLogger(d, n_iter, **kwargs)

        # Scheme
        x = [None]*d
        for k in range(d):
            x_k = x_0[k]
            logger.log_param(k, 0, x_k)
            bounds_k = bnds[k]
            n_iter_k = n_iter[k]

            for t in tqdm(range(n_iter_k), disable=not verbose):
                # Compute LSE gradient estimate for parameters x_k
                g_t = estimators[k].lse_k_grad_estimate(x_k, rng)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration then project into space of parameters
                x_k = solvers.iterate(t, x_k, g_t)
                x_k = np.maximum(x_k, bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        fitted_mu, fitted_ker_param = self.tensor2matrix_params(x)
        self.is_fitted = True
        self.fitted_mu = fitted_mu
        self.fitted_ker_param = fitted_ker_param

        logger.process_logs(self)
        logger.mu_0 = mu_0
        logger.ker_0 = ker_0
        self.fit_log = logger
        return fitted_mu, fitted_ker_param

    def make_adjacency_matrix(self, kernel_param=None):
        log_fitted_adjacency = False
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
                log_fitted_adjacency = True
            else:
                raise ValueError("kernel_param must be specified.")
        d = self.d
        adjacency = [[self._kernel_matrix[i][j].l1_norm(kernel_param[i][j])
                      for j in range(d)] for i in range(d)]
        if log_fitted_adjacency:
            self.fitted_adjacency = adjacency
        return adjacency

    def get_branching_ratio(self, kernel_param):
        adjacency = self.make_adjacency_matrix(kernel_param)
        return np.max(np.absolute(np.linalg.eigvals(adjacency)))

    def get_random_param(self, ref_mu=None, ref_ker_param=None, range_ref=0.1,
                         target_bratio=0.6, max_omega=1., true_omega=None,
                         max_param=5.,
                         min_mu=0., max_mu=None, flatten=False, seed=1234,
                         rng=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        d = self.d

        # Mu
        if ref_mu is None:
            if not isinstance(min_mu, (list, np.ndarray)):
                min_mu = np.ones(d)*min_mu
            if max_mu is None:
                max_mu = max(max(min_mu), 1.)
            if not isinstance(max_mu, (list, np.ndarray)):
                max_mu = np.ones(d)*max_mu
            mu = np.zeros(d)
            for i in range(d):
                mu[i] = rng.uniform(low=min_mu[i], high=max_mu[i], size=1)[0]
        else:
            mu = np.zeros(d)
            for i in range(d):
                mu[i] = rng.uniform(low=max(0., (1.-range_ref)*ref_mu[i]),
                                    high=(1+range_ref)*ref_mu[i], size=1)[0]

        # Kernels
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if ref_ker_param is None:
            if not isinstance(max_param, (list, np.ndarray)):
                float_max = max_param
                max_param = [[[None for x
                               in range(self._kernel_matrix[i][j].n_param)]
                              for j in range(d)] for i in range(d)]
                for i, j in itertools.product(range(d), range(d)):
                    n_param = self._kernel_matrix[i][j].n_param
                    vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                    bnds = self.param_bounds[i][j]
                    for x in range(n_param):
                        if x in vec_ix_omega:
                            max_param[i][j][x] = max_omega
                        else:
                            max_param[i][j][x] = max(float_max, bnds[x])

            for i, j in itertools.product(range(d), range(d)):
                kernel_param[i][j] = []
                n_param = self._kernel_matrix[i][j].n_param
                vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                bnds = self.param_bounds[i][j]
                for x in range(n_param):
                    val = rng.uniform(low=bnds[x], high=max_param[i][j][x],
                                      size=1)[0]
                    kernel_param[i][j].append(val)
        else:
            for i, j in itertools.product(range(d), range(d)):
                kernel_param[i][j] = []
                n_param = self._kernel_matrix[i][j].n_param
                bnds = self.param_bounds[i][j]
                for x in range(n_param):
                    val = rng.uniform(low=max(bnds[x],
                                              (1.-range_ref)
                                              * ref_ker_param[i][j][x]),
                                      high=(1.+range_ref)
                                      * ref_ker_param[i][j][x],
                                      size=1)[0]
                    kernel_param[i][j].append(val)

        # Rescaling
        branching_ratio = self.get_branching_ratio(kernel_param)
        if branching_ratio > 0.:
            scaling = target_bratio/branching_ratio
        for i, j in itertools.product(range(d), range(d)):
            kernel_param[i][j] = np.array(kernel_param[i][j])
            if branching_ratio > 0.:
                vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                kernel_param[i][j][vec_ix_omega] = (scaling*kernel_param[i][j][vec_ix_omega])

        # Flatten
        if flatten:
            return self.matrix2tensor_params(mu, kernel_param)
        else:
            return mu, kernel_param

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
        process_path : TYPE
            DESCRIPTION.
        mu : TYPE, optional
            DESCRIPTION. The default is None.
        kernel_param : TYPE, optional
            DESCRIPTION. The default is None.
        sampling : TYPE, optional
            DESCRIPTION. The default is False.
        sample_size : TYPE, optional
            DESCRIPTION. The default is 10**3.
        seed : TYPE, optional
            DESCRIPTION. The default is 1234.
        write : TYPE, optional
            DESCRIPTION. The default is True.
        verbose : TYPE, optional
            DESCRIPTION. The default is False.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        residuals : TYPE
            DESCRIPTION.

        """
        
        if mu is None or kernel_param is None:
            if self.is_fitted:
                mu = self.fitted_mu
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("Both mu and kernel_param must be specified.")
        residuals = gof.get_residuals(process_path, self.psi, mu,
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

    # Simulation
    def simulate(self, T_f, mu=None, kernel_param=None, seed=1234,
                 verbose=False):
        if mu is None:
            mu = self.fitted_mu
            if mu is None:
                raise ValueError("Missing value for Mu")
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
            if kernel_param is None:
                raise ValueError("Missing value for Kernel parameters")
        mu = np.array(mu)
        d = self.d
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        adjacency = self.make_adjacency_matrix(kernel_param)
        rng = np.random.default_rng(seed)

        branching_ratio = self.get_branching_ratio(kernel_param)
        if branching_ratio >= 1:
            raise ValueError("Cannot simulate from unstable MHP: ",
                             "The branching ratio of this MHP is ",
                             branching_ratio, " > 1.")
        if verbose:
            print('Simulating events...')
        # Step 1. Generate immigrants
        # Number of immigrants
        Nim = rng.poisson(mu*T_f)

        # Location of immigrants
        generations = [[rng.uniform(low=0.0, high=T_f, size=Nim[i])]
                       for i in range(d)]

        def sum_generation(L, index):
            return sum([len(L[i][index]) for i in range(d)])

        ix_gen = 1
        #   Step 2. Fill via repeated generations
        while sum_generation(generations, ix_gen-1):
            for j in range(d):
                if len(generations[j][ix_gen-1]) > 0:
                    for i in range(d):
                        # Set number of offspring
                        Noff = rng.poisson(adjacency[i][j],
                                           size=len(generations[j][ix_gen-1]))
                        parenttimes = generations[j][ix_gen-1].repeat(Noff)
                        offsets = offset_gens[i][j](rng, N=Noff.sum())
                        offspringtime = parenttimes + offsets
                        generations[i] = generations[i]+[np.array([x for x in offspringtime if x < T_f])]

            ix_gen += 1
        list_times = [np.array(sorted([x for sublist in generations[i]
                                       for x in sublist])) for i in range(d)]
        if verbose:
            n_tot = sum([len(L) for L in list_times])
            print('Simulation Complete, ', n_tot, ' events simulated.')
        return list_times

    def simu_multipath(self, path_res, t_res, x_min, x_max, mu=None,
                       kernel_param=None, seed=1234, verbose=False,
                       disc_type='log', base_seed=1234):
        d = self.d
        rng = np.random.default_rng(base_seed)
        vec_seeds = rng.choice(10**5, size=path_res, replace=False)

        if disc_type == 'log':
            T_f = 10**x_max
        elif disc_type == 'linear':
            T_f = x_max
        list_Tf = uf.discretize_space(x_min, x_max, t_res, disc_type)
        list_paths = [[[] for j in range(path_res)] for i in range(t_res)]
        for j in range(path_res):
            seed = vec_seeds[j]
            list_times = self.simulate(T_f, mu=mu, kernel_param=kernel_param,
                                       seed=seed, verbose=verbose)
            for i in range(t_res):
                local_Tf = list_Tf[i]
                list_n_f = [bisect.bisect_left(list_times[index_dim],
                                               local_Tf)-1
                            for index_dim in range(d)]
                list_paths[i][j] = [list_times[0][:list_n_f[index_dim]+1]
                                    for index_dim in range(d)]
        return list_Tf, list_paths

    # Metrics
    # L2 projection
    def get_l2_projection(self, mhp_2, param_2, n_iter=1000,
                          solver=None, log_error=False, rng=None,
                          seed=1234,
                          verbose=False, **kwargs):
        d = self.d
        if rng is None:
            rng = np.random.default_rng(seed)
        # Model
        bnds = self.param_bounds

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
        mu_0, ker_0 = self.get_random_param(ref_mu=ref_mu,
                                            ref_ker_param=ref_ker_param,
                                            range_ref=range_ref,
                                            target_bratio=target_bratio,
                                            max_omega=max_omega,
                                            true_omega=true_omega,
                                            max_param=max_param,
                                            min_mu=min_mu, max_mu=max_mu,
                                            flatten=False, rng=rng)

        param_1 = [[None for j in range(d)] for i in range(d)]

        if solver is None:
            solver = ADAM(**kwargs)

        if log_error:
            l2_err_log = [[np.zeros(n_iter) for j in range(d)] for i in range(d)]
        else:
            l2_err_log = None
        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            kernel_2 = mhp_2._kernel_matrix[i][j]
            ker_param_2 = param_2[i][j]
            res_ij = kernel.get_l2_projection(kernel_2, ker_param_2, n_iter= n_iter, params_0=ker_0[i][j],
                                              solver=solver, log_error=log_error, rng=ker_0[i][j],
                                              verbose=verbose, **kwargs)
            param_1[i][j] = copy.deepcopy(res_ij['params'])
            if log_error:
                l2_err_log[i][j] = copy.deepcopy(res_ij['log'])
        res = {'params': param_1, 'log': l2_err_log}
        return res
    
    # Plots
    def plot_kernels(self, kernel_param=None, t_min=0., t_max=10.,
                     n_samples=1000, index_from_one=False, log_scale=False,
                     dpi=300, axs=None, save=False, filename='image.png',
                     show=False, **kwargs):
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("kernel_param must be specified.")
        return gt.plot_kernels(self.phi, kernel_param, t_min=t_min,
                               t_max=t_max, n_samples=n_samples,
                               index_from_one=index_from_one,
                               log_scale=log_scale, dpi=dpi, axs=axs,
                               save=save, filename=filename, show=show,
                               **kwargs)

    def plot_adjacency_matrix(self, adjacency=None, kernel_param=None,
                              event_names=None,
                              index_from_one=False, annotate=False,
                              cmap="Blues", dpi=300, save=False,
                              filename='image.png', show=True):
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
                                        annotate=annotate, cmap=cmap, dpi=dpi,
                                        save=save, filename=filename,
                                        show=show)

    def plot_solver_path(self, true_mu=None, true_ker_param=None, min_mu=None,
                         min_ker_param=None, plot_derivatives=False,
                         display_derivatives_zero=False, figsize=(10, 10),
                         dpi=300, pad=300, axs=None, save=False,
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
                                   display_derivatives_zero=display_derivatives_zero,
                                   figsize=figsize, dpi=dpi, pad=pad, axs=axs,
                                   save=save, filename=filename, show=show,
                                   **kwargs)
