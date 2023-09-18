# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np

import aslsd.utilities.useful_functions as uf


class BasisImpact(ABC):
    """
    Abstract class for parametric families of basis kernels.

    | A basis kernel with parameters :math:`\\vartheta:=(\\vartheta_1, \\dots, \\vartheta_{n_{\\textrm{param}}})` is characterized by a basis function 
    
    .. math::
        f_{\\vartheta}:[0,+\\infty) \\to [0,+\\infty).
    
    We assume each parameter :math:`\\vartheta_i` leaves in a half-open interval :math:`[b_i, +\\infty)`.

    | In our code, we use sums of basis kernels :math:`\\Big(f^{(r)}_{\\vartheta^{(r)}}\\Big)_{r \\in [n_b]}` to define Kernels :math:`\\phi_{\\theta}` of an MHP, such that for all :math:`t \\geq 0`

    .. math::
        \\phi_{\\theta}(t) := \\sum_{r=1}^{n_{\\textrm{b}}}
        f^{(r)}_{\\vartheta^{(r)}}(t),

    where :math:`\\theta` is the vector of parameters of kernel :math:`\\phi_{\\theta}`,
    obtainend by concatenating the vectors of parameters :math:`\\vartheta^{(r)}`
    of each of the :math:`n_b` basis kernels.


    Attributes
    ----------
    n_fixed_vars : `int`
        DESCRIPTION. The default is 0.
    ix_map : TYPE, optional
        DESCRIPTION. The default is None.
    dict_interactions : TYPE, optional
        DESCRIPTION. The default is None.
    phi : `function`
        DESCRIPTION. The default is None.
    diff_phi : `function`
        DESCRIPTION. The default is None.

    """

    def __init__(self, fixed_indices=None, fixed_vars=None):
        """
        Initialize BasisKernel.

        Parameters
        ----------
        fixed_indices : `int` or `list` or `numpy.ndarray`, default=None
            Array of indices of parameters of the basis kernel that we want to fix
            to given values, that will be given by fixed_vars.
            If `int`then broadcasted to a unidimensional list.
    
        fixed_vars : `float` or `list` or `numpy.ndarray`, default=None
            Array of values of parameters of the basis kernel that we wish to fix.
            If `float`then broadcasted to a unidimensional list.
            Must be of the same length as fixed_indices; fixed_vars[p] is the value
            to which we want to impose for the parameter of index fixed_indices[p].

        """

        if fixed_indices is None:
            self.fixed_indices = []
            self.fixed_vars = []
            self.n_fixed_vars = 0
        elif isinstance(fixed_indices, (list, np.ndarray)):
            self.n_fixed_vars = len(fixed_vars)
            # Sort the list of fixed indices
            mixed_list = [[fixed_indices[i], fixed_vars[i]]
                          for i in range(self.n_fixed_vars)]
            mixed_list = sorted(mixed_list, key=lambda x: x[0])
            self.fixed_indices = [x[0] for x in mixed_list]
            self.fixed_vars = [x[1] for x in mixed_list]

        else:
            self.fixed_indices = [fixed_indices]
            self.fixed_vars = [fixed_vars]
            self.n_fixed_vars = 1

        self.ix_map = self.make_ix_map()

        self.make_impact_functionals()

    # Input dimension
    @abstractmethod
    def get_mark_dim(self):
        pass

    # Number of parameters
    @abstractmethod
    def get_n_vars(self):
        pass

    def get_n_param(self):
        """
        Get the number of non-fixed parameters of the basis kernel.

        Returns
        -------
        `int`
            Number of parameters.

        Notes
        ------
            Each parametric class of basis kernels is parametrized by a number
            of variables :math:`n_{var}`. Users can fix a number
            :math:`n_{fixed}` of these variables, by explicitly specifying
            the indices of . We refer to the non-fixed variables as parameters
            of the BasisKernel object. This function outputs the number of
            parameters, given by :math:`n_{param} = n_{var}-n_{fixed}`.


        """
        return self.get_n_vars()-self.n_fixed_vars

    # Bounds
    @abstractmethod
    def get_var_lower_bounds(self):
        pass

    @abstractmethod
    def get_var_upper_bounds(self):
        pass

    def get_param_lower_bounds(self):
        """
        Get the list of lower bounds of the domain of each parameter of the
        basis kernel.

        Let :math:`\\vartheta:=(\\vartheta_1, \\dots, \\vartheta_{n_{\\textrm{param}}})` denote the parameters of the basis kernel. 
        Each parameter :math:`\\vartheta_i` lives in a half-open interval :math:`[b_i, +\\infty)`.
        This method returns the vector :math:`(b_1, \\dots, b_{n_{\\textrm{param}}})`.

        Returns
        -------
        `list`
            List of lower bounds of the domain of each parameter of the
            basis kernel.

        Notes
        ------
            By parameters, we mean the non-fixed paramters.

        """
        var_bnds = self.get_var_lower_bounds()
        n_vars = len(var_bnds)
        param_bnds = np.array([var_bnds[ix] for ix in range(n_vars)
                               if ix not in self.fixed_indices])
        return param_bnds

    def get_param_upper_bounds(self):
        """
        Get the list of lower bounds of the domain of each parameter of the
        basis kernel.

        Let :math:`\\vartheta:=(\\vartheta_1, \\dots, \\vartheta_{n_{\\textrm{param}}})` denote the parameters of the basis kernel. 
        Each parameter :math:`\\vartheta_i` lives in a half-open interval :math:`[b_i, +\\infty)`.
        This method returns the vector :math:`(b_1, \\dots, b_{n_{\\textrm{param}}})`.

        Returns
        -------
        `list`
            List of lower bounds of the domain of each parameter of the
            basis kernel.

        Notes
        ------
            By parameters, we mean the non-fixed paramters.

        """
        var_bnds = self.get_var_upper_bounds()
        n_vars = len(var_bnds)
        param_bnds = np.array([var_bnds[ix] for ix in range(n_vars)
                               if ix not in self.fixed_indices])
        return param_bnds

    # Parameter names
    @abstractmethod
    def get_var_names(self):
        pass

    def get_param_names(self):
        """
        Get the list of names of the parameters of the basis kernel.

        Returns
        -------
        `list`
            List of names of the parameters of the basis kernel.

        Notes
        ------
            By parameters, we mean the non-fixed paramters.
            If the method get_var_names is not properly implemented in a child
            class, that is, if the method is set to return `None`, we generate
            parameter names as letters of the alphabet, except letters 'l' and
            'o' following pycodestyle E741.
            In the unlikely case where the basis kernel has more than 24
            parameters in the basis kernel, this will raise a ValueError.

        """
        var_names = self.get_var_names()
        if var_names is not None:
            n_vars = len(var_names)
            return [var_names[i] for i in range(n_vars)
                    if i not in self.fixed_indices]

        else:
            n_param = self.get_n_param()
            return uf.get_alphabet_range(n_param)

    # Impact functionals
    def make_vars(self, params):
        """
        Get the array of variables of the kernel.

        Parameters
        ----------
        params : `list` or `numpy.ndarray`
            Array of (non-fixed) parameters of the kernel.

        Returns
        -------
        variables : `numpy.ndarray`
            Array of variables (i.e. fixed and non-fixed parameters) of the
            basis kernel.

        """

        # A simple case: when no parameter has been fixed
        if self.n_fixed_vars == 0:
            return params
        else:
            n_vars = self.get_n_vars()
            variables = np.zeros(n_vars)
            ix_param = 0
            ix_fixed = 0
            for i in range(n_vars):
                if i in self.fixed_indices:
                    variables[i] = self.fixed_vars[ix_fixed]
                    ix_fixed += 1
                else:
                    variables[i] = params[ix_param]
                    ix_param += 1
            return variables

    def make_ix_map(self):
        n_vars = self.get_n_vars()
        ix_map = []
        for i in range(n_vars):
            if i not in self.fixed_indices:
                ix_map.append(i)
        return ix_map

    @abstractmethod
    def make_impact(self, xi, vars_):
        pass

    @abstractmethod
    def make_diff_impact(self, xi, ix_diff, vars_):
        pass

    def make_impact_functionals(self):
        def impact(xi, params):
            vars_ = self.make_vars(params)
            return self.make_impact(xi, vars_)
        self.impact = impact

        def diff_impact(xi, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_impact(xi, ix_diff_scaled, vars_)
        self.diff_impact = diff_impact
