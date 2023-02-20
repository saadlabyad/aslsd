# License: BSD 3 clause

import bisect
import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.marks.void_mark import VoidMark
from aslsd.baselines.baseline import BaselineModel
from aslsd.basis_baselines.basis_baseline_constant import ConstantBaseline
from aslsd.impact_functions.impact_function import ImpactFunction
from aslsd.basis_impacts.basis_impact_constant import ConstantImpact
from aslsd.estimators.mtlh_estimator import MtlhStratified
from aslsd.estimators.estimator import Estimator
from aslsd.evaluation import goodness_of_fit as gof
from aslsd.events.process_path import ProcessPath
from aslsd.optim_logging.optim_logger import OptimLogger
from aslsd.plots import graphic_tools as gt
from aslsd.solvers.adam import ADAM
from aslsd.solvers.solver import Solver
from aslsd.utilities import useful_functions as uf


class MTLH:
    """
    Class for Extended Linear Hawkes processes (ELHP) models.

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

    def __init__(self, _kernel_matrix, _baselines_vec=None,
                 _impact_matrix=None, vec_marks=None,
                 index_from_one=False, d=None,
                 matrix_n_param=None,
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
        self.d = len(_kernel_matrix)
        self.is_fitted = False
        self.index_from_one = index_from_one
        self.baselines_vec = _baselines_vec
        self.kernel_matrix = _kernel_matrix
        self.impact_matrix = _impact_matrix

        vec_marks_ = self.make_marks(vec_marks)
        self.vec_marks = vec_marks_

    # Marks
    def make_marks(self, vec_marks):
        d = self.d
        if vec_marks is None:
            vec_marks_ = [VoidMark() for i in range(d)]
        else:
            if uf.is_array(vec_marks):
                vec_marks_ = copy.deepcopy(vec_marks)
            else:
                vec_marks_ = [copy.deepcopy(vec_marks) for i in range(d)]
        return vec_marks_

    def make_expected_impact(self, impact_param):
        d = self.d
        if impact_param is None:
            impact_param = self.fitted_imp_param
            if impact_param is None:
                raise ValueError("Missing value for Impact parameters")
        expected_impact_matrix = np.zeros((d, d))
        for i, j in itertools.product(range(d), range(d)):
            impact = self._impact_matrix[i][j]
            expected_impact_matrix[i][j] = self.vec_marks[j].get_expected_impact(impact, impact_param[i][j])
        return expected_impact_matrix

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
        # Initialisation if L is None
        if L is None:
            L = [BaselineModel([ConstantBaseline()]) for i in range(self.d)]

        if hasattr(self, 'vector_n_param_mu'):
            old_vector_n_param_mu = copy.deepcopy(self.vector_n_param_mu)
        else:
            old_vector_n_param_mu = None
        # Baseline attributes only
        self._baselines_vec = copy.deepcopy(L)
        self.d = len(L)
        self.vector_n_param_mu = self.get_vector_n_param_mu()
        # Parameters names
        self.mu_param_names = self.get_mu_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.mu_param_bounds = self.get_mu_param_bounds()

        # General updates
        if hasattr(self, '_kernel_matrix'):
            if (old_vector_n_param_mu is not None) and (self.vector_n_param_mu != old_vector_n_param_mu):
                ix_map_ker, interval_map_ker = self.make_maps_ker()
                self.ix_map_ker = ix_map_ker
                self.interval_map_ker = interval_map_ker
            if hasattr(self, '_impact_matrix'):
                if (old_vector_n_param_mu is not None) and (self.vector_n_param_mu != old_vector_n_param_mu):
                    ix_map_imp, interval_map_imp = self.make_maps_imp()
                    self.ix_map_imp = ix_map_imp
                    self.interval_map_imp = interval_map_imp
                self.make_functionals()
                self.n_param_k = self.get_n_param_k()

    @baselines_vec.deleter
    def baselines_vec(self):
        del self._baselines_vec

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
        if hasattr(self, 'matrix_n_param_ker'):
            old_matrix_n_param_ker = copy.deepcopy(self.matrix_n_param_ker)
        else:
            old_matrix_n_param_ker = None
        # Kernel attributes only
        self._kernel_matrix = copy.deepcopy(M)
        self.d = len(M)
        self.matrix_n_param_ker = self.get_matrix_n_param_ker()
        # Parameters indexing
        ix_map_ker, interval_map_ker = self.make_maps_ker()
        self.ix_map_ker = ix_map_ker
        self.interval_map_ker = interval_map_ker
        # Parameters names
        self.ker_param_names = self.get_ker_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.ker_param_bounds = self.get_ker_param_bounds()

        # General updates
        if hasattr(self, '_impact_matrix'):
            if (old_matrix_n_param_ker is not None) and (self.matrix_n_param_ker != old_matrix_n_param_ker):
                ix_map_imp, interval_map_imp = self.make_maps_imp()
                self.ix_map_imp = ix_map_imp
                self.interval_map_imp = interval_map_imp
            if hasattr(self, '_baselines_vec'):
                self.make_functionals()
                self.n_param_k = self.get_n_param_k()

    @kernel_matrix.deleter
    def kernel_matrix(self):
        del self._kernel_matrix

    # Impact matrix
    @property
    def impact_matrix(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._impact_matrix

    @impact_matrix.setter
    def impact_matrix(self, M):
        # Initialisation if M is None:
        if M is None:
            M = [[ImpactFunction([ConstantImpact()]) for j in range(self.d)]
                 for i in range(self.d)]

        # Impact attributes only
        self._impact_matrix = copy.deepcopy(M)
        self.d = len(M)
        self.matrix_n_param_imp = self.get_matrix_n_param_imp()
        # Parameters indexing
        ix_map_imp, interval_map_imp = self.make_maps_imp()
        self.ix_map_imp = ix_map_imp
        self.interval_map_imp = interval_map_imp
        # Parameters names
        self.imp_param_names = self.get_imp_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.imp_param_bounds = self.get_imp_param_bounds()

        # General updates
        if hasattr(self, '_baselines_vec') and hasattr(self, '_impact_matrix'):
            self.make_functionals()
            self.n_param_k = self.get_n_param_k()

    @impact_matrix.deleter
    def impact_matrix(self):
        del self._impact_matrix

    # N params
    def get_vector_n_param_mu(self):
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
        vec_n_param_mu = [self._baselines_vec[i].n_param for i in range(d)]
        return vec_n_param_mu

    def get_matrix_n_param_ker(self):
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
        mat_n_param_ker = [[self._kernel_matrix[i][j].n_param for j in range(d)]
                           for i in range(d)]
        return mat_n_param_ker

    def get_matrix_n_param_imp(self):
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
        mat_n_param_imp = [[self._impact_matrix[i][j].n_param for j in range(d)]
                            for i in range(d)]
        return mat_n_param_imp

    def get_n_param_k(self):
        d = self.d
        n_param_k = [None]*d
        for k in range(d):
            n_param_k[k] = (self.vector_n_param_mu[k]
                            + sum(self.matrix_n_param_ker[k])
                            + sum(self.matrix_n_param_imp[k]))
        return n_param_k

    # Parameters map
    def make_maps_ker(self):
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
        ix_map_ker = [[None for i in range(self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k]))]
                      for k in range(d)]
        interval_map_ker = [[None for i in range(d)] for k in range(d)]
        for k in range(d):
            ix_map_ker[k][0] = -1
            ix_ker = 0
            ix_param = 0
            ix_left = self.vector_n_param_mu[k]
            x = self.vector_n_param_mu[k]
            n_param_k = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])
            while x < n_param_k:
                ix_map_ker[k][x] = {'ker': ix_ker, 'par': ix_param}
                if ix_param == self._kernel_matrix[k][ix_ker].n_param-1:
                    interval_map_ker[k][ix_ker] = [ix_left, x+1]
                    ix_ker += 1
                    ix_param = 0
                    ix_left = x+1
                    x += 1
                else:
                    ix_param += 1
                    x += 1
        return ix_map_ker, interval_map_ker

    def make_maps_imp(self):
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
        ix_map_imp = [[None for i in range(self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])+sum(self.matrix_n_param_imp[k]))]
                      for k in range(d)]
        interval_map_imp = [[None for i in range(d)] for k in range(d)]
        for k in range(d):
            ix_map_imp[k][0] = -1
            ix_imp = 0
            ix_param = 0
            ix_left = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])
            x = ix_left
            n_param_k = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])+sum(self.matrix_n_param_imp[k])
            while x < n_param_k:
                ix_map_imp[k][x] = {'imp': ix_imp, 'par': ix_param}
                if ix_param == self._impact_matrix[k][ix_imp].n_param-1:
                    interval_map_imp[k][ix_imp] = [ix_left, x+1]
                    ix_imp += 1
                    ix_param = 0
                    ix_left = x+1
                    x += 1
                else:
                    ix_param += 1
                    x += 1
        return ix_map_imp, interval_map_imp

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
        mu_param : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        d = self.d
        mu_param = np.array([tensor_param[i][:self.vector_n_param_mu[i]] for i in range(d)])
        kernel_param = np.array([[tensor_param[i][self.interval_map_ker[i][j][0]:
                                                  self.interval_map_ker[i][j][1]]
                                  for j in range(d)] for i in range(d)],
                                dtype=object)
        impact_param = np.array([[tensor_param[i][self.interval_map_imp[i][j][0]:
                                                  self.interval_map_imp[i][j][1]]
                                  for j in range(d)] for i in range(d)],
                                dtype=object)
        return mu_param, kernel_param, impact_param

    def xk2matrix_params(self, k, x_k):
        """
        Convert the list of flat vectors of parameters to a vector of
        background rates `mu` and a matrix of kernel parameters `kernel_param`.

        Parameters
        ----------
        tensor_param : TYPE
            DESCRIPTION.

        Returns
        -------
        mu_param : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        d = self.d
        x_mu_k = copy.deepcopy(x_k[:self.vector_n_param_mu[k]])
        x_ker_k = np.array([x_k[self.interval_map_ker[k][i][0]:
                                self.interval_map_ker[k][i][1]]
                            for i in range(d)], dtype=object)
        x_imp_k = np.array([x_k[self.interval_map_imp[k][i][0]:
                                self.interval_map_imp[k][i][1]]
                            for i in range(d)], dtype=object)
        return x_mu_k, x_ker_k, x_imp_k

    def matrix2tensor_params(self, mu_param, kernel_param, impact_param):
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
            # mu parameters
            x_k.extend(copy.deepcopy(mu_param[k]))
            # kernel parameters
            for i in range(d):
                x_k.extend(copy.deepcopy(kernel_param[k][i]))
            # Impact parameters
            for i in range(d):
                x_k.extend(copy.deepcopy(impact_param[k][i]))
            x[k] = np.array(copy.deepcopy(x_k), dtype=object)
        return np.array(x)

    def tensor2matrix_solverpaths(self, tensor_paths):
        #   Convert the list of parameters
        #   tensor_paths is a tensor of paths of solvers per parameters
        d = self.d
        list_n = [len(tensor_paths[k]) for k in range(d)]
        mu_param_paths = [[tensor_paths[k][n][:self.vector_n_param_mu[k]] for n in range(list_n[k])]
                          for k in range(d)]
        kernel_param_paths = [[[tensor_paths[i][n][self.interval_map_ker[i][j][0]:
                                                   self.interval_map_ker[i][j][1]]
                                for n in range(list_n[i])] for j in range(d)]
                              for i in range(d)]
        impact_param_paths = [[[tensor_paths[i][n][self.interval_map_imp[i][j][0]:
                                                   self.interval_map_imp[i][j][1]]
                                for n in range(list_n[i])] for j in range(d)]
                              for i in range(d)]
        return mu_param_paths, kernel_param_paths, impact_param_paths

    def make_xk(self, k, mu_param=None, kernel_param=None, impact_param=None):
        d = self.d

        if mu_param is None:
            mu_param = self.fitted_mu_param
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
        if impact_param is None:
            impact_param = self.fitted_imp_param

        x = self.matrix2tensor_params(mu_param, kernel_param, impact_param)
        x_k = x[k]
        return x_k

    # Omega
    def is_sbf(self):
        d = self.d
        for i, j in itertools.product(range(d), range(d)):
            if not self._kernel_matrix[i][j].is_sbf():
                return False
        return True

    # Bounds
    def get_mu_param_bounds(self):
        d = self.d
        mu_bnds = [self._baselines_vec[i].get_param_bounds() for i in range(d)]
        return mu_bnds

    def get_ker_param_bounds(self):
        d = self.d
        ker_bnds = [[self._kernel_matrix[i][j].get_param_bounds()
                     for j in range(d)] for i in range(d)]
        return ker_bnds

    def get_imp_param_bounds(self):
        d = self.d
        imp_bnds = [[self._impact_matrix[i][j].get_param_bounds()
                     for j in range(d)] for i in range(d)]
        return imp_bnds

    # Param names
    def get_mu_param_names(self, index_from_one=False):
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
                    mu_param_names[0][ix_param] = (vec_names[ix_basis][ix_2][:-1] + '_{'+str(ix_basis+int(index_from_one))+'}$')

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
                        mu_param_names[i][ix_param] = vec_names[ix_basis][ix_2][:-1] + '_{'+str(i+int(index_from_one))+'}$'
                else:
                    for ix_param in range(n_param):
                        ix_basis = mu.ix_map[ix_param]['mu']
                        ix_2 = mu.ix_map[ix_param]['par']
                        mu_param_names[i][ix_param] = vec_names[ix_basis][ix_2][:-1] + '_{'+str(i+int(index_from_one))+','+str(ix_basis+int(index_from_one))+'}$'
        return mu_param_names

    def get_ker_param_names(self, index_from_one=False):
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

        if d == 1:
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
        return ker_param_names

    def get_imp_param_names(self, index_from_one=False):
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

        if d == 1:
            impact = self._impact_matrix[0][0]
            vec_names = impact.get_vec_param_names()
            n_param = impact.n_param
            imp_param_names = [[None for j in range(d)] for i in range(d)]
            imp_param_names[0][0] = [None]*n_param
            for ix_param in range(n_param):
                ix_imp = impact.ix_map[ix_param]['imp']
                ix_2 = impact.ix_map[ix_param]['par']
                if impact.n_basis_imp == 1:
                    imp_param_names[0][0][ix_param] = vec_names[ix_imp][ix_2]
                else:
                    imp_param_names[0][0][ix_param] = (vec_names[ix_imp][ix_2][:-1] + '_{'+str(ix_imp+int(index_from_one))+'}$')
        else:
            imp_param_names = [[None for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                impact = self._impact_matrix[i][j]
                vec_names = impact.get_vec_param_names()
                n_param = impact.n_param
                imp_param_names[i][j] = [None]*n_param
                if impact.n_basis_imp == 1:
                    for ix_param in range(n_param):
                        ix_imp = impact.ix_map[ix_param]['imp']
                        ix_2 = impact.ix_map[ix_param]['par']
                        imp_param_names[i][j][ix_param] = vec_names[ix_imp][ix_2][:-1] + '_{'+str(i+int(index_from_one))+','+str(j+int(index_from_one))+'}$'
                else:
                    for ix_param in range(n_param):
                        ix_imp = impact.ix_map[ix_param]['imp']
                        ix_2 = impact.ix_map[ix_param]['par']
                        imp_param_names[i][j][ix_param] = vec_names[ix_imp][ix_2][:-1] + '_{'+str(i+int(index_from_one))+','+str(j+int(index_from_one))+','+str(ix_imp+int(index_from_one))+'}$'
        return imp_param_names

    # Kernel functionals
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
        self.K = [[None for j in range(d)] for i in range(d)]
        self.diff_K = [[None for j in range(d)] for i in range(d)]

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
                func_u = kernel_ki.make_upsilon()

                def upsilon(t, s, params_1, params_2):
                    return func_u(kernel_kj, t, s, params_1, params_2)
                self.upsilon[i][j][k] = upsilon
                diff_func_u = kernel_ki.make_diff_cross_upsilon()

                def diff_cross_upsilon(t, s, ix_func, ix_diff, params_1,
                                       params_2):
                    return diff_func_u(kernel_kj, t, s, ix_func, ix_diff,
                                       params_1, params_2)
                self.diff_cross_upsilon[i][j][k] = diff_cross_upsilon
            else:
                raise NotImplementedError("No available interaction"
                                          " between kernel", k, ",", i,
                                          " and kernel ", k, ",", j)

        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            baseline = self._baselines_vec[i]
            func_K = kernel.make_K()

            def K(t, s, params_ker, params_mu):
                return func_K(baseline, t, s, params_ker, params_mu)
            self.K[i][j] = K
            diff_func_K = kernel.make_diff_K()

            def diff_K(t, s, ix_func, ix_diff, params_ker, params_mu):
                return diff_func_K(baseline, t, s, ix_func, ix_diff, params_ker,
                                   params_mu)
            self.diff_K[i][j] = diff_K

        # Impact
        self.impact = [[None for j in range(d)] for i in range(d)]
        self.diff_impact = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            impact_function = self._impact_matrix[i][j]
            impact = impact_function.make_impact()
            self.impact[i][j] = impact
            diff_impact = impact_function.make_diff_impact()
            self.diff_impact[i][j] = diff_impact

    # Estimator functions
    def init_estimator(self, estimator, k):
        # Ixs book-keeping
        estimator.n_param_k = self.n_param_k[k]
        estimator.vector_n_param_mu = self.vector_n_param_mu
        estimator.matrix_n_param_ker = self.matrix_n_param_ker
        estimator.matrix_n_param_imp = self.matrix_n_param_imp
        estimator.ix_map_ker = self.ix_map_ker
        estimator.ix_map_imp = self.ix_map_imp
        estimator.interval_map_ker = self.interval_map_ker
        estimator.interval_map_imp = self.interval_map_imp
        estimator.xk2matrix_params = self.xk2matrix_params
        # Functionals
        estimator.phi = self.phi
        estimator.diff_phi = self.diff_phi
        estimator.upsilon = self.upsilon
        estimator.diff_sim_upsilon = self.diff_sim_upsilon
        estimator.diff_cross_upsilon = self.diff_cross_upsilon
        estimator.M = self.M
        estimator.diff_M = self.diff_M
        estimator.mu = self.mu
        estimator.diff_mu = self.diff_mu
        estimator.K = self.K
        estimator.diff_K = self.diff_K
        estimator.impact = self.impact
        estimator.diff_impact = self.diff_impact

    # Fit
    def init_logger(self, logger):
        d = self.d
        n_iter = logger.n_iter
        n_param_k = self.n_param_k
        if logger.is_log_param:
            logger.param_logs = [np.zeros((n_iter[k]+1, n_param_k[k]))
                                 for k in range(d)]
            logger.mu = [[None for x in range(n_iter[i]+1)] for i in range(d)]
            logger.ker = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                          for i in range(d)]
            logger.imp = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                          for i in range(d)]
        if logger.is_log_grad:
            logger.grad_logs = [np.zeros((n_iter[k], n_param_k[k]))
                                for k in range(d)]
            logger.grad_mu = [[None for x in range(n_iter[i]+1)] for i in range(d)]
            logger.grad_ker = [[[None for x in range(n_iter[i])]
                                for j in range(d)] for i in range(d)]
            logger.grad_imp = [[[None for x in range(n_iter[i])]
                                for j in range(d)] for i in range(d)]

        logger.mu_0 = None
        logger.ker_0 = None
        logger.imp_0 = None

    def process_logs(self, logger):
        d = self.d
        if logger.is_log_param:
            # Mu
            for i in range(d):
                for ix in range(logger.n_iter[i]+1):
                    logger.mu[i][ix] = logger.param_logs[i][ix][:self.vector_n_param_mu[i]]

            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]+1):
                    # Kernel
                    logger.ker[i][j][ix] = logger.param_logs[i][ix][self.interval_map_ker[i][j][0]:self.interval_map_ker[i][j][1]]
                    # Impact
                    logger.imp[i][j][ix] = logger.param_logs[i][ix][self.interval_map_imp[i][j][0]:self.interval_map_imp[i][j][1]]
        if logger.is_log_grad:
            # Mu
            for i in range(d):
                for ix in range(logger.n_iter[i]):
                    logger.grad_mu[i][ix] = logger.grad_logs[i][ix][:self.vector_n_param_mu[i]]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]):
                    # Kernel
                    logger.grad_ker[i][j][ix] = logger.grad_logs[i][ix][self.interval_map_ker[i][j][0]:self.interval_map_ker[i][j][1]]
                    # Impact
                    logger.grad_imp[i][j][ix] = logger.grad_logs[i][ix][self.interval_map_imp[i][j][0]:self.interval_map_imp[i][j][1]]
        if logger.is_log_lse:
            for k in range(d):
                self.lse[k] = self.estimator_logs[k]['lse']
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
                logger.allocs[k]['phi'] = logger.estimator_logs[k]['allocs']['phi']
                logger.allocs[k]['upsilon'] = logger.estimator_logs[k]['allocs']['upsilon']

    def clear_fit(self):
        """
        Delete all previously saved results and logs from the
        corresponding attributes of the MHP object.

        """
        self.is_fitted = False
        self.fitted_mu_param = None
        self.fitted_ker_param = None
        self.fitted_imp_param = None
        self.fit_residuals = None
        self.fitted_adjacency = None
        self.fit_log = None

    def fit(self, list_times, T_f, list_marks=None, kappa=None, varpi=None,
            x_0=None,
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
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        kappa : TYPE, optional
            DESCRIPTION. The default is None.
        varpi : TYPE, optional
            DESCRIPTION. The default is None.
        x_0 : `list` of `numpy.ndarray`, optional
            x_0[k] is the initial guess for parameters of problem k. The
            default is None.
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
        rng = np.random.default_rng(seed)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Data
        d = self.d
        process_path = ProcessPath(list_times, T_f, list_marks=list_marks,
                                   kappa=kappa, varpi=varpi)

        # Model
        bnds = self.matrix2tensor_params(self.mu_param_bounds,
                                         self.ker_param_bounds,
                                         self.imp_param_bounds)

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if x_0 is None:
            ref_mu_param = kwargs.get('ref_mu_param', None)
            ref_ker_param = kwargs.get('ref_ker_param', None)
            ref_imp_param = kwargs.get('ref_imp_param', None)
            range_ref_mu = kwargs.get('range_ref_mu', 0.1)
            range_ref_ker = kwargs.get('range_ref_ker', 0.1)
            range_ref_imp = kwargs.get('range_ref_imp', 0.1)
            min_mu_param = kwargs.get('min_mu_param', None)
            max_mu_param = kwargs.get('max_mu_param', None)
            target_bratio = kwargs.get('target_bratio', 0.6)
            max_omega = kwargs.get('max_omega', 1.)
            true_omega = kwargs.get('true_omega', None)
            max_ker_param = kwargs.get('max_ker_param', 5.)
            max_imp_param = kwargs.get('max_imp_param', 5.)

            mu_0, ker_0, imp_0 = self.get_random_param(ref_mu_param=ref_mu_param,
                                                       ref_ker_param=ref_ker_param,
                                                       ref_imp_param=ref_imp_param,
                                                       range_ref_mu=range_ref_mu,
                                                       range_ref_ker=range_ref_ker,
                                                       range_ref_imp=range_ref_imp,
                                                       min_mu_param=min_mu_param,
                                                       max_mu_param=max_mu_param,
                                                       target_bratio=target_bratio,
                                                       max_omega=max_omega,
                                                       true_omega=true_omega,
                                                       max_ker_param=max_ker_param,
                                                       max_imp_param=max_imp_param,
                                                       flatten=False, rng=rng)
            x_0 = self.matrix2tensor_params(mu_0, ker_0, imp_0)
        else:
            mu_0, ker_0, imp_0 = self.tensor2matrix_params(x_0)

        # Initialize Estimators
        if estimators is None:
            estimators = [MtlhStratified(**kwargs) for k in range(d)]
        else:
            if issubclass(type(estimators), Estimator):
                estimators = [copy.deepcopy(estimators) for k in range(d)]
        for k in range(d):
            estimators[k].k = k
            estimators[k].n_iter = n_iter[k]
            estimators[k].initialize(process_path, self)
            estimators[k].set_stratification(list_times, **kwargs)
            estimators[k].intialize_logs()

        # Initialize Solvers
        if solvers is None:
            solvers = [ADAM(**kwargs) for k in range(d)]
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]

        # Initialize logger
        logger = OptimLogger(d, n_iter, **kwargs)
        self.init_logger(logger)

        # Scheme
        x = [None]*d
        for k in range(d):
            x_k = x_0[k]
            logger.log_param(k, 0, x_k)
            bounds_k = bnds[k]
            n_iter_k = n_iter[k]

            for t in tqdm(range(n_iter_k), disable=not verbose):
                # Compute LSE gradient estimate for parameters x_k
                g_t = estimators[k].lse_k_grad_estimate(x_k, grad_alloc=True,
                                                        rng=rng)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration then project into space of parameters
                x_k = solvers[k].iterate(t, x_k, g_t)
                x_k = np.maximum(x_k, bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        fitted_mu, fitted_ker_param, fitted_imp_param = self.tensor2matrix_params(x)
        if write:
            self.is_fitted = True
            self.fitted_mu = fitted_mu
            self.fitted_ker_param = fitted_ker_param
            self.fitted_imp_param = fitted_imp_param
            self.process_logs(logger)
            logger.mu_0 = mu_0
            logger.ker_0 = ker_0
            logger.imp_0 = imp_0
            self.fit_log = logger
        return fitted_mu, fitted_ker_param, fitted_imp_param

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
        adjacency = [[self._kernel_matrix[i][j].l1_norm(kernel_param[i][j])
                      for j in range(d)] for i in range(d)]
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

    def get_random_param(self, ref_mu_param=None, ref_ker_param=None,
                         ref_imp_param=None, range_ref_mu=0.1,
                         range_ref_ker=0.1, range_ref_imp=0.1,
                         min_mu_param=None, max_mu_param=None,
                         target_bratio=0.6, max_omega=1., true_omega=None,
                         max_ker_param=5., max_imp_param=5.,
                         flatten=False,
                         seed=1234,
                         rng=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        d = self.d

        mu_bnds = self.get_mu_param_bounds()
        ker_bnds = self.get_ker_param_bounds()
        imp_bnds = self.get_imp_param_bounds()

        # Mu
        if ref_mu_param is None:
            if min_mu_param is None:
                min_mu_param = copy.deepcopy(mu_bnds)
            else:
                if not uf.is_array(min_mu_param):
                    min_mu_float = copy.deepcopy(min_mu_param)
                    min_mu_param = [min_mu_float*np.ones(self.vector_n_param_mu[i]) for i in range(d)]
            if max_mu_param is None:
                max_mu_param = [1.+min_mu_param[i] for i in range(d)]
            if not uf.is_array(max_mu_param):
                max_mu_float = copy.deepcopy(max_mu_param)
                max_mu_param = [max_mu_float*np.ones(self.vector_n_param_mu[i]) for i in range(d)]
            mu_param = np.array([np.zeros(self.vector_n_param_mu[i]) for i in range(d)], dtype=object)
            for i in range(d):
                n_param = self.vector_n_param_mu[i]
                for ix in range(n_param):
                    mu_param[i][ix] = rng.uniform(low=min_mu_param[i][ix],
                                                  high=max_mu_param[i][ix],
                                                  size=1)[0]
        else:
            dist_mu_bnds = [ref_mu_param[i]-mu_bnds[i] for i in range(d)]
            lower_mu_params = [ref_mu_param[i]-range_ref_mu*dist_mu_bnds[i] for i in range(d)]
            upper_mu_params = [ref_mu_param[i] for i in range(d)]

            mu_param = np.array([np.zeros(self.vector_n_param_mu[i]) for i in range(d)], dtype=object)
            for i in range(d):
                mu_param[i][ix] = rng.uniform(low=lower_mu_params[i][ix],
                                              high=upper_mu_params[i][ix],
                                              size=1)[0]

        # Kernels
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if ref_ker_param is None:
            if not isinstance(max_ker_param, (list, np.ndarray)):
                float_max = copy.deepcopy(max_ker_param)
                max_ker_param = [[[None for x
                                   in range(self.matrix_n_param_ker[i][j])]
                                  for j in range(d)] for i in range(d)]
                for i, j in itertools.product(range(d), range(d)):
                    n_param = self.matrix_n_param_ker[i][j]
                    vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                    for x in range(n_param):
                        if x in vec_ix_omega:
                            max_ker_param[i][j][x] = max_omega
                        else:
                            max_ker_param[i][j][x] = max(float_max, ker_bnds[i][j][x])

            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_ker[i][j]
                kernel_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=ker_bnds[i][j][ix],
                                      high=max_ker_param[i][j][ix],
                                      size=1)[0]
                    kernel_param[i][j][ix] = val
        else:
            dist_ker_bnds = [[ref_ker_param[i][j]-ker_bnds[i][j] for j in range(d)] for i in range(d)]
            lower_ker_params = [[ref_ker_param[i][j]-range_ref_ker*dist_ker_bnds[i][j] for j in range(d)] for i in range(d)]
            upper_ker_params = [[ref_ker_param[i][j]+range_ref_ker*dist_ker_bnds[i][j] for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_ker[i][j]
                kernel_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=lower_ker_params[i][j][ix],
                                      high=upper_ker_params[i][j][ix],
                                      size=1)[0]
                    kernel_param[i][j][ix] = val
        # Rescaling
        branching_ratio = self.get_branching_ratio(kernel_param=kernel_param)
        if branching_ratio > 0.:
            scaling = target_bratio/branching_ratio
        for i, j in itertools.product(range(d), range(d)):
            kernel_param[i][j] = np.array(kernel_param[i][j])
            if branching_ratio > 0.:
                vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                kernel_param[i][j][vec_ix_omega] = (scaling*kernel_param[i][j][vec_ix_omega])

        # Impacts
        impact_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if ref_imp_param is None:
            if not uf.is_array(max_imp_param):
                float_max = copy.deepcopy(max_imp_param)
                max_imp_param = [[[None for x
                                   in range(self.matrix_n_param_imp[i][j])]
                                  for j in range(d)] for i in range(d)]
                for i, j in itertools.product(range(d), range(d)):
                    n_param = self.matrix_n_param_imp[i][j]
                    for ix in range(n_param):
                        max_imp_param[i][j][ix] = max(float_max, imp_bnds[i][j][ix])

            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_imp[i][j]
                impact_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=imp_bnds[i][j][ix],
                                      high=max_ker_param[i][j][ix],
                                      size=1)[0]
                    impact_param[i][j][ix] = val
        else:
            dist_imp_bnds = [[ref_imp_param[i][j]-imp_bnds[i][j] for j in range(d)] for i in range(d)]
            lower_imp_params = [[ref_imp_param[i][j]-range_ref_imp*dist_imp_bnds[i][j] for j in range(d)] for i in range(d)]
            upper_imp_params = [[ref_imp_param[i][j]+range_ref_imp*dist_imp_bnds[i][j] for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_imp[i][j]
                impact_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=lower_imp_params[i][j][ix],
                                      high=upper_imp_params[i][j][ix],
                                      size=1)[0]
                    impact_param[i][j][ix] = val

        # Flatten
        if flatten:
            return self.matrix2tensor_params(mu_param, kernel_param,
                                             impact_param)
        else:
            return mu_param, kernel_param, impact_param

    # Residuals
    def get_residuals(self, process_path, mu_param=None, kernel_param=None,
                      impact_param=None, expected_impact_matrix=None,
                      sampling=False, sample_size=10**3,
                      seed=1234, write=True, verbose=False):
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
        if mu_param is None:
            if self.is_fitted:
                mu_param = self.fitted_mu_param
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("mu_param must be specified.")

        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("kernel_param must be specified.")

        if impact_param is None:
            if self.is_fitted:
                impact_param = self.fitted_imp_param
            else:
                raise ValueError("impact_param must be specified.")
        if expected_impact_matrix is None:
            expected_impact_matrix = self.make_expected_impact(impact_param)
        residuals = gof.get_residuals_mtlh(process_path, self.mu_compensator,
                                           self.psi, self.impact,
                                           expected_impact_matrix,
                                           mu_param, kernel_param, impact_param,
                                           sampling=sampling,
                                           sample_size=sample_size, seed=seed,
                                           verbose=verbose)
        if self.is_fitted and write:
            self.fitted_expected_impact_matrix = expected_impact_matrix
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
    def simulate(self, T_f, mu_param=None, kernel_param=None,
                 impact_param=None, rng=None, seed=1234,
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

        if mu_param is None:
            mu_param = self.fitted_mu_param
            if mu_param is None:
                raise ValueError("Missing value for Baseline parameters")
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
            if kernel_param is None:
                raise ValueError("Missing value for Kernel parameters")
        if impact_param is None:
            impact_param = self.fitted_imp_param
            if impact_param is None:
                raise ValueError("Missing value for Impact parameters")

        d = self.d
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        adjacency = self.make_adjacency_matrix(kernel_param)
        if rng is None:
            rng = np.random.default_rng(seed)

        branching_ratio = self.get_branching_ratio(adjacency=adjacency)
        if branching_ratio >= 1:
            raise ValueError("Cannot simulate from unstable MHP: ",
                             "The branching ratio of this MHP is ",
                             branching_ratio, " > 1.")
        if verbose:
            print('Simulating events...')
        # Step 1. Generate immigrants
        # Location of immigrants
        generations = [[self._baselines_vec[i].simulate(T_f, mu_param[i],
                                                        rng=rng)] for i in range(d)]
        raw_marks = [[self.vec_marks[i].simulate(size=len(generations[i][0]),
                                                 rng=rng)] for i in range(d)]
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen

        def sum_generation(L, index):
            return sum([len(L[i][index]) for i in range(d)])

        ix_gen = 1
        #   Step 2. Fill via repeated generations
        while sum_generation(generations, ix_gen-1):
            for k in range(d):
                generations[k].append(np.array([]))
                raw_marks[k].append(np.array([]))
            for j in range(d):
                # Simulate the offspring of the "ix_gen-1"th generation of
                # events of type j
                if len(generations[j][ix_gen-1]) > 0:
                    for i in range(d):
                        # Set number of offspring
                        parent_marks = raw_marks[j][ix_gen-1]
                        parent_impacts = self._impact_matrix[i][j].impact(parent_marks,
                                                                          impact_param[i][j])
                        Noff = rng.poisson(adjacency[i][j]*parent_impacts,
                                           size=len(generations[j][ix_gen-1]))
                        parenttimes = generations[j][ix_gen-1].repeat(Noff)
                        offsets = offset_gens[i][j](rng, N=Noff.sum())
                        offspringtime = parenttimes + offsets
                        generations[i][ix_gen] = np.append(generations[i][ix_gen], np.array([x for x in offspringtime if x < T_f]))
                        n_valid_kids = len(np.array([x for x in offspringtime if x < T_f]))
                        raw_marks[i][ix_gen] = np.append(raw_marks[i][ix_gen],
                                                         self.vec_marks[i].simulate(size=n_valid_kids,
                                                                                    rng=rng))
            ix_gen += 1

        if verbose:
            print('Sorting results ...')
        list_times_ = [[x for sublist in generations[i]
                        for x in sublist] for i in range(d)]
        list_marks_ = [[x for sublist in raw_marks[i]
                   for x in sublist] for i in range(d)]

        list_marks = [np.array([x for _, x in sorted(zip(list_times_[i], list_marks_[i]))]) for i in range(d)]
        list_times = [np.array(sorted(list_times_[i])) for i in range(d)]
        if verbose:
            n_tot = sum([len(L) for L in list_times])
            print('Simulation Complete, ', n_tot, ' events simulated.')
        return list_times, list_marks

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
                                              solver=solver,
                                              log_error=log_error,
                                              rng=ker_0[i][j],
                                              verbose=verbose, **kwargs)
            param_1[i][j] = copy.deepcopy(res_ij['params'])
            if log_error:
                l2_err_log[i][j] = copy.deepcopy(res_ij['log'])
        res = {'params': param_1, 'log': l2_err_log}
        return res

    # Plots
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
                         derivatives_zero=False, axs=None, save=False,
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
                                   axs=axs, save=save, filename=filename,
                                   show=show, **kwargs)

    # Serialization
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
        file_residuals = file+'_fitted_residuals'
        pickle_out = open(file_residuals, "wb", **kwargs)
        pickle.dump(self.fit_residuals, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        file_adjacency = file+'_fitted_adj'
        pickle_out = open(file_adjacency, "wb", **kwargs)
        pickle.dump(self.fitted_adjacency, pickle_out)
        pickle_out.close()

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
        fitted_ker = pickle.load(pickle_in)

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
        self.fitted_ker_param = fitted_ker
        self.fit_residuals = fitted_residuals
        self.fitted_adjacency = fitted_adjacency
