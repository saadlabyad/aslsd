# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np

import aslsd.utilities.useful_functions as uf


class BasisBaseline(ABC):
    """
    Abstract class for parametric families of basis baselines.

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
    psi : `function`
        DESCRIPTION. The default is None.
    diff_psi : `function`
        DESCRIPTION. The default is None.
    upsilon : `function`
        DESCRIPTION. The default is None.
    diff_sim_upsilon : `function`
        DESCRIPTION. The default is None.
    diff_cross_upsilon : `function`
        DESCRIPTION. The default is None.

    Notes
    ------
        To avoid any confusion, we add that the notion of (sums of) basis
        kernels is different from the notion of (sums of) basis
        functions discussed in Cartea, Á., Cohen, S. N., and Labyad, S.,
        (November 22, 2021) 'Gradient-based estimation of linear Hawkes
        processes with general kernels'. `Available at SSRN. <https://ssrn.com/abstract=3969208>`_
        In particular, we do not assume that all basis kernels only have one
        parameter, and are proportional to a function with fixed parameter
        values. That is, we do not assume that all basis kernels :math:`f`
        satisfy a structure like

        .. math::
            f_{\\vartheta} = \\omega g,

        where :math:`g` is some function and :math:`\\vartheta=(\\omega)`.

        | Users can fix arbitrary values for some of the components of the vector of parameters :math:`\\vartheta` of the basis kernel. When fitting an MHP, fixed parameters will not be optimized. To avoid confusion, we refer to the vector of all parameters (wether fixed or not) as the variables of the basis kernel, and by abuse we refer to the vector of non-fixed parameters as parameters.

    """

    def __init__(self, fixed_indices=None, fixed_vars=None, n_fixed_vars=0,
                 ix_map=None, dict_interactions=None, mu=None, diff_mu=None,
                 M=None, diff_M=None, K=None, diff_K=None):
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

        self.dict_interactions = self.make_dict_interactions()

        self.make_baseline_functionals()
        self.make_simulation_functionals()

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
        param_bnds = np.array([var_bnds[i] for i in range(n_vars)
                               if i not in self.fixed_indices])
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
        param_bnds = np.array([var_bnds[i] for i in range(n_vars)
                               if i not in self.fixed_indices])
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

        """
        var_names = self.get_var_names()
        if var_names is not None:
            n_vars = len(var_names)
            return [var_names[i] for i in range(n_vars)
                    if i not in self.fixed_indices]
        # If the method get_var_names is not properly implemented in a child
        # class, that is, if the method is set to return `None`, we generate
        # parameter names as letters of the alphabet, except letters 'l' and
        # 'o' following pycodestyle E741.
        # In the unlikely case where the basis kernel has more than 24
        # parameters in the basis kernel, this will raise a ValueError.
        else:
            n_param = self.get_n_param()
            return uf.get_alphabet_range(n_param)

    # Availabe interactions
    @abstractmethod
    def get_interactions(self, is_reverse=False):
        pass

    def make_dict_interactions(self):
        """
        

        Returns
        -------
        dict_interactions : `dict`
            DESCRIPTION.

        """
        dict_interactions = {'dir': {}, 'rev': {}}
        list_inter_dir = self.get_interactions(is_reverse=False)
        for x in list_inter_dir:
            dict_interactions['dir'][str(type(x))] = True
        list_inter_rev = self.get_interactions(is_reverse=True)
        for x in list_inter_rev:
            dict_interactions['rev'][str(type(x))] = False
        return dict_interactions

    def is_compatible(self, basis_kern_2, is_reverse=False):
        """
        

        Parameters
        ----------
        basis_kern_2 : BasisKernel
            DESCRIPTION.
        is_reverse : `bool`, optional
            Check if . The default is False.

        Returns
        -------
        bool
            True if the basis kernel is compatible with `basis_kern_2`, False
            otherwise.

        """
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        if ker_type in self.get_interactions(is_reverse=is_reverse):
            return True
        return False

    # Kernel functionals
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
        """
        

        Returns
        -------
        ix_map : `list`
            DESCRIPTION.

        """
        n_vars = self.get_n_vars()
        ix_map = []
        for i in range(n_vars):
            if i not in self.fixed_indices:
                ix_map.append(i)
        return ix_map

    @abstractmethod
    def make_mu(self, t, vars_):
        pass

    @abstractmethod
    def make_diff_mu(self, t, ix_diff, vars_):
        pass

    @abstractmethod
    def make_M(self, t, vars_):
        pass

    @abstractmethod
    def make_diff_M(self, t, ix_diff, vars_):
        pass

    @abstractmethod
    def make_K(self, basis_kernel, t, s, vars_ker, vars_mu):
        pass

    @abstractmethod
    def make_diff_K(self, basis_kernel, t, s, ix_func, ix_diff, vars_ker,
                    vars_mu):
        pass

    def make_baseline_functionals(self):
        """
        Set the attributes corresponding to kernel functionals and their
        derivatives, as functions of (non-fixed) parameters.
        The functional :math:`\\psi` is defined for all :math:`t \\geq 0` by

        .. math::
            \\psi(t) := \\int_{[0,t]} f(u)du

        The functional :math:`\\Upsilon` is defined for all :math:`t,s \\geq 0`
        and for all functions :math:`\\tilde{f}` by

        .. math::
            \\Upsilon[f, \\tilde{f}](t,s) := \\int_{[0,t]} f(u)\\tilde{f}(u+s)du


        In our implementation we denote the functions:

        * :math:`f` by the attribute `phi`;
        * :math:`\\frac{\\partial f}{\\partial \\vartheta_p}` by the attribute `diff_phi`;
        * :math:`\\psi` by the attribute `psi`;
        * :math:`\\frac{\\partial \\psi}{\\partial \\vartheta_p}` by the attribute `diff_psi`;
        * :math:`\\Upsilon[\\cdot, f]` by the attribute `upsilon_rev`;
        * :math:`\\Upsilon[f, \\cdot]` by the attribute `upsilon`;
        * :math:`\\frac{\\partial \\Upsilon[f, f]}{\\partial \\vartheta_p}` by the attribute `diff_sim_upsilon`;
        * :math:`\\frac{\\partial \\Upsilon[\\cdot, f]}{\\partial \\vartheta_p}` by the attribute `diff_cross_upsilon_rev`;
        * :math:`\\frac{\\partial \\Upsilon[f, \\cdot]}{\\partial \\vartheta_p}` by the attribute `diff_cross_upsilon`;

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        Notes
        ------
            See Cartea, Á., Cohen, S. N., and Labyad, S., (November 22, 2021)
            'Gradient-based estimation of linear Hawkes processes with general
            kernels'.
            `Available at SSRN. <https://ssrn.com/abstract=3969208>`_

        """

        def mu(t, params):
            vars_ = self.make_vars(params)
            return self.make_mu(t, vars_)
        self.mu = mu

        def diff_mu(t, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_mu(t, ix_diff_scaled, vars_)
        self.diff_mu = diff_mu

        def M(t, params):
            vars_ = self.make_vars(params)
            return self.make_M(t, vars_)
        self.M = M

        def diff_M(t, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_M(t, ix_diff_scaled, vars_)
        self.diff_M = diff_M

        def K(kernel, t, s, params_ker, params_mu):
            vars_mu = self.make_vars(params_mu)
            res = 0.
            for ix_ker in range(kernel.n_basis_ker):
                basis_kernel = kernel._basis_kernels[ix_ker]
                start = kernel.interval_map[ix_ker][0]
                end = kernel.interval_map[ix_ker][1]
                params_basis_ker = params_ker[start:end]
                vars_basis_ker = basis_kernel.make_vars(params_basis_ker)
                res += self.make_K(basis_kernel, t, s, vars_basis_ker, vars_mu)
            return res
        self.K = K

        def diff_K(kernel, t, s, ix_func, ix_diff, params_ker,
                   params_mu):
            vars_mu = self.make_vars(params_mu)
            if ix_func == 1:
                # Derivative wrt kernel
                ix_ker = kernel.ix_map[ix_diff]['ker']
                ix_diff_scaled = kernel.ix_map[ix_diff]['par']
                basis_kernel = kernel._basis_kernels[ix_ker]
                start = kernel.interval_map[ix_ker][0]
                end = kernel.interval_map[ix_ker][1]
                params_basis_ker = params_ker[start:end]
                vars_basis_ker = basis_kernel.make_vars(params_basis_ker)
                return self.make_diff_K(basis_kernel, t, s, ix_func,
                                        ix_diff_scaled, vars_basis_ker,
                                        vars_mu)
            elif ix_func == 2:
                res = 0.
                for ix_ker in range(kernel.n_basis_ker):
                    basis_kernel = kernel._basis_kernels[ix_ker]
                    start = kernel.interval_map[ix_ker][0]
                    end = kernel.interval_map[ix_ker][1]
                    params_basis_ker = params_ker[start:end]
                    vars_basis_ker = basis_kernel.make_vars(params_basis_ker)
                    res += self.make_diff_K(basis_kernel, t, s, ix_func,
                                            ix_diff, vars_basis_ker, vars_mu)
                return res
        self.diff_K = diff_K

    # Simulatiom
    @abstractmethod
    def make_compensator(self, t, vars_):
        pass

    @abstractmethod
    def make_inv_compensator(self, y, vars_):
        pass

    @abstractmethod
    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass

    def make_simulation_functionals(self):
        def compensator(t, params):
            vars_ = self.make_vars(params)
            return self.make_compensator(t, vars_)
        self.compensator = compensator

        def inverse_compensator(y, params):
            vars_ = self.make_vars(params)
            return self.make_inv_compensator(y, vars_)
        self.inverse_compensator = inverse_compensator

        def intensity_bound(params):
            vars_ = self.make_vars(params)
            return self.make_intensity_bound(vars_)
        self.intensity_bound = intensity_bound
