# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np

import aslsd.utilities.useful_functions as uf


class BasisKernel(ABC):
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
    allow_simu : `bool`
        DESCRIPTION. The default is True.
    sim_func : `function`
        DESCRIPTION. The default is None.
    l1_norm : `function`
        DESCRIPTION. The default is None.
    diff_l1_norm : `function`
        DESCRIPTION. The default is None.
    l2_norm : `function`
        DESCRIPTION. The default is None.
    l2_dot : `function`
        DESCRIPTION. The default is None.
    l2_distance : `function`
        DESCRIPTION. The default is None.
    kl_divergence : `function`
        DESCRIPTION. The default is None.
    diff_kl_divergence : `function`
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
                 ix_map=None, dict_interactions=None, phi=None, diff_phi=None,
                 psi=None, diff_psi=None, upsilon=None, diff_sim_upsilon=None,
                 diff_cross_upsilon=None, allow_simu=True, sim_func=None,
                 l1_norm=None, diff_l1_norm=None, l2_norm=None, l2_dot=None,
                 l2_distance=None, kl_divergence=None,
                 diff_kl_divergence=None):
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
        # Fix variables
        if fixed_indices is None:
            self.fixed_indices = np.array([], dtype=int)
            self.fixed_vars = np.array([])
            self.n_fixed_vars = 0
        else:
            if not uf.is_array(fixed_indices):
                fixed_indices = np.array([fixed_indices], dtype=int)
                fixed_vars = np.array([fixed_vars])
            self.n_fixed_vars = len(fixed_vars)
            # Sort the list of fixed indices
            mixed_list = [[fixed_indices[i], fixed_vars[i]]
                          for i in range(self.n_fixed_vars)]
            mixed_list = sorted(mixed_list, key=lambda x: x[0])
            self.fixed_indices = np.array([x[0] for x in mixed_list],
                                          dtype=int)
            self.fixed_vars = np.array([x[1] for x in mixed_list])

        # Book-keeping for parameter indices
        self.list_is_active_var = self.make_list_is_active_var()
        self.ix_map = self.make_ix_map()

        # Interactions with other kernels
        self.dict_interactions = self.make_dict_interactions()

        # Kernel interactions
        self.make_kernel_functionals()

        # Simulation
        self.allow_simu = allow_simu
        if self.allow_simu:
            def simu_func(rng, params, size=1):
                vars_ = self.make_vars(params)
                return self.make_simu_func(rng, vars_, size=size)
            self.simu_func = simu_func

        # L_p metrics
        self.make_l1_metrics()
        self.make_l2_metrics()

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

    def make_list_is_active_var(self):
        n_vars = self.get_n_vars()
        list_is_active_var = np.ones(n_vars, dtype=bool)
        list_is_active_var[self.fixed_indices] = False
        return list_is_active_var

    # Omega
    def has_omega_param(self):
        """
        Check if this instance of basis kernel has an :math:`\\omega`
        parameter, that is, if the basis kernel function :math:`f`
        satisfies a structure like

        .. math::
            f_{\\vartheta} = \\omega g_{\\varphi}

        where :math:`g_{\\varphi}` is some function parametrized
        by :math:`\\varphi` and :math:`\\omega` is a positive parameter.
        In this case, the parameters of the basis kernel are

        .. math::
            \\vartheta^\\intercal=(\\omega, \\varphi^\\intercal).

        We allow for the vector of parameters :math:`\\varphi^\\intercal` to
        be empty in this definition.

        Returns
        -------
        `bool`
            True if the basis kernel has an omega type parameter, False
            otherwise.

        Notes
        ------
            By parameters, we mean the non-fixed paramters.

        """
        return (0 != self.fixed_indices[0])

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
            # A common case, SBF: all parameters but the first have been fixed
            if self.has_omega_param() and self.get_n_param() == 1:
                return np.concatenate(([params[0]], self.fixed_vars))
            # General case
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
    def make_phi(self, t, vars_):
        pass

    @abstractmethod
    def make_diff_phi(self, t, ix_diff, vars_):
        pass

    @abstractmethod
    def make_psi(self, t, vars_):
        pass

    @abstractmethod
    def make_diff_psi(self, t, ix_diff, vars_):
        pass

    @abstractmethod
    def make_upsilon(self, basis_kern_2, t, s, vars_1, vars_2):
        pass

    @abstractmethod
    def make_upsilon_rev(self, basis_kern_2, t, s, vars_2, vars_1):
        pass

    @abstractmethod
    def make_diff_sim_upsilon(self, t, s, ix_diff, vars_):
        pass

    @abstractmethod
    def make_diff_cross_upsilon(self, basis_kern_2, t, s, ix_func, ix_diff,
                                vars_1, vars_2):
        pass

    @abstractmethod
    def make_diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass

    @abstractmethod
    def make_K(self, baseline, t, s, vars_ker, params_mu):
        pass

    @abstractmethod
    def make_diff_K(self, baseline, t, s, ix_func, ix_diff, vars_ker, params_mu):
        pass

    def make_kernel_functionals(self):
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

        def phi(t, params):
            vars_ = self.make_vars(params)
            return self.make_phi(t, vars_)
        self.phi = phi

        def diff_phi(t, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_phi(t, ix_diff_scaled, vars_)
        self.diff_phi = diff_phi

        def psi(t, params):
            vars_ = self.make_vars(params)
            return self.make_psi(t, vars_)
        self.psi = psi

        def diff_psi(t, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_psi(t, ix_diff_scaled, vars_)
        self.diff_psi = diff_psi

        def upsilon_rev(basis_kern_2, t, s, params_2, params_1):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            return self.make_upsilon_rev(basis_kern_2, t, s, vars_2, vars_1)
        self.upsilon_rev = upsilon_rev

        def upsilon(basis_kern_2, t, s, params_1, params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if self.is_compatible(basis_kern_2, is_reverse=False):
                return self.make_upsilon(basis_kern_2, t, s, vars_1, vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                return basis_kern_2.make_upsilon_rev(self, t, s, vars_1,
                                                     vars_2)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.upsilon = upsilon

        def diff_sim_upsilon(t, s, ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_sim_upsilon(t, s, ix_diff_scaled, vars_)
        self.diff_sim_upsilon = diff_sim_upsilon

        def diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                   params_2, params_1):
            vars_2 = basis_kern_2.make_vars(params_2)
            vars_1 = self.make_vars(params_1)
            if ix_func == 1:
                ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
            elif ix_func == 2:
                ix_diff_scaled = self.ix_map[ix_diff]
            return self.diff_cross_upsilon_rev(basis_kern_2, t, s, ix_func,
                                               ix_diff_scaled, vars_2, vars_1)
        self.diff_cross_upsilon_rev = diff_cross_upsilon_rev

        def diff_cross_upsilon(basis_kern_2, t, s, ix_func, ix_diff,
                               params_1, params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if ix_func == 1:
                ix_diff_scaled = self.ix_map[ix_diff]
            elif ix_func == 2:
                ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
            if self.is_compatible(basis_kern_2, is_reverse=False):
                return self.make_diff_cross_upsilon(basis_kern_2, t, s,
                                                    ix_func, ix_diff_scaled,
                                                    vars_1, vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                if ix_func == 1:
                    ix_diff_scaled = self.ix_map[ix_diff]
                elif ix_func == 2:
                    ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
                return basis_kern_2.diff_cross_upsilon_rev(self, t, s, ix_func,
                                                           ix_diff_scaled,
                                                           vars_1, vars_2)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.diff_cross_upsilon = diff_cross_upsilon

        def K(baseline, t, s, params_ker, params_mu):
            vars_ker = self.make_vars(params_ker)
            return self.make_K(baseline, t, s, vars_ker, params_mu)
        self.K = K

        def diff_K(baseline, t, s, ix_func, ix_diff, params_ker, params_mu):
            vars_ker = self.make_vars(params_ker)
            if ix_func == 1:
                ix_diff_scaled = self.ix_map[ix_diff]
            elif ix_func == 2:
                ix_diff_scaled = ix_diff
            return self.make_diff_K(baseline, t, s, ix_func, ix_diff_scaled,
                                    vars_ker, params_mu)
        self.diff_K = diff_K

    # Simulatiom
    @abstractmethod
    def make_simu_func(self, rng, vars_, size=1):
        pass

    # Metrics
    @abstractmethod
    def make_l1_norm(self, vars_):
        pass

    @abstractmethod
    def make_diff_l1_norm(self, ix_diff, vars_):
        pass

    def make_l1_metrics(self):
        """
        Set the function attributes that comupte the :math:`L_{1}` norm of
        the basis kernel :math:`f` and its derivatives, as functions of
        (non-fixed) parameters.
        The :math:`L_{1}` norm of the basis kernel :math:`f` is defined by

        .. math::
            \\| f_{\\vartheta} \\|_1 := \\int_{[0,+\\infty]} |f_{\\vartheta}(u)|du.

        In our implementation we denote the functions:

        * :math:`\\| f_{\\vartheta} \\|_1` by the attribute `l1_norm`;
        * :math:`\\frac{\\partial `\\| f_{\\vartheta} \\|_1`}{\\partial \\vartheta_p}` by the attribute `diff_l1_norm`.

        """

        def l1_norm(params):
            vars_ = self.make_vars(params)
            return self.make_l1_norm(vars_)
        self.l1_norm = l1_norm

        def diff_l1_norm(ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_l1_norm(ix_diff_scaled, vars_)
        self.diff_l1_norm = diff_l1_norm

    @abstractmethod
    def make_l2_norm(self, vars_):
        pass

    @abstractmethod
    def make_diff_l2_norm(self, ix_diff, vars_):
        pass

    @abstractmethod
    def make_l2_dot(self, basis_kern_2, vars_1, vars_2):
        pass

    @abstractmethod
    def make_diff_l2_dot(self, basis_kern_2, ix_func, ix_diff, vars_1, vars_2):
        pass

    def make_l2_metrics(self):
        """
        Set the function attributes that compute :math:`L_{2}` norms, metrics
        inner-products, and their derivatives as functions of (non-fixed)
        parameters.
        Let :math:`f_{\\vartheta}` denote the basis kernel function with
        parameters :math:`\\vartheta`.
        The :math:`L_{2}` norm of the basis kernel function
        :math:`f_{\\vartheta}` is defined by

        .. math::
            \\| f_{\\vartheta} \\|_2^2 := \\int_{[0,+\\infty]} |f_{\\vartheta}(u)|^2du.

        The :math:`L_{2}` inner product between the basis kernel function
        :math:`f_{\\vartheta}` and some other basis kernel function
        :math:`g_{\\vartheta^\\prime}` is defined by

        .. math::
            \\langle f,g\\rangle_2 := \\int_{[0,+\\infty]} f_{\\vartheta}(u)g_{\\vartheta^\\prime}(u)du.

        The :math:`L_{2}` metric between :math:`f_{\\vartheta}` and
        :math:`g_{\\vartheta^\\prime}` is defined by

        .. math::
            d_2(f_{\\vartheta}, g_{\\vartheta^\\prime}) := \\| f_{\\vartheta}-g_{\\vartheta^\\prime} \\|_2.

        In our implementation we denote the functions:

        * :math:`\\| f_{\\vartheta} \\|_2` by the attribute `l2_norm`;
        * :math:`\\frac{\\partial `\\| f_{\\vartheta} \\|_2`}{\\partial \\vartheta_p}` by the attribute `diff_l2_norm`.

        """
        def l2_norm(params):
            vars_ = self.make_vars(params)
            return self.make_l2_norm(vars_)
        self.l2_norm = l2_norm

        def diff_l2_norm(ix_diff, params):
            vars_ = self.make_vars(params)
            ix_diff_scaled = self.ix_map[ix_diff]
            return self.make_diff_l2_norm(ix_diff_scaled, vars_)
        self.diff_l2_norm = diff_l2_norm

        def l2_dot(basis_kern_2, params_1, params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if self.is_compatible(basis_kern_2, is_reverse=False):
                return self.make_l2_dot(basis_kern_2, vars_1, vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                return basis_kern_2.make_l2_dot(self, vars_2, vars_1)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.l2_dot = l2_dot

        def diff_l2_dot(basis_kern_2, ix_func, ix_diff, params_1, params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if self.is_compatible(basis_kern_2, is_reverse=False):
                if ix_func == 1:
                    ix_diff_scaled = self.ix_map[ix_diff]
                elif ix_func == 2:
                    ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
                return self.make_diff_l2_dot(basis_kern_2, ix_func,
                                             ix_diff_scaled, vars_1, vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                if ix_func == 1:
                    ix_func_scaled = 2
                    ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
                elif ix_func == 2:
                    ix_func_scaled = 1
                    ix_diff_scaled = self.ix_map[ix_diff]
                return basis_kern_2.make_diff_l2_dot(self, ix_func_scaled,
                                                     ix_diff_scaled, vars_2,
                                                     vars_1)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.diff_l2_dot = diff_l2_dot

        def l2_distance(basis_kern_2, params_1, params_2):
            norm_kern_1 = self.l2_norm(params_1)
            norm_kern_2 = basis_kern_2.l2_norm(params_2)
            dot_prod = self.l2_dot(basis_kern_2, params_1, params_2)
            return norm_kern_1 + norm_kern_2 - 2*dot_prod
        self.l2_distance = l2_distance

    # KL divergence
    @abstractmethod
    def make_kl_divergence(self, basis_kern_2, vars_1, vars_2):
        pass

    @abstractmethod
    def make_kl_divergence_rev(self, basis_kern_2, vars_2, vars_1):
        pass

    @abstractmethod
    def make_diff_kl_divergence(self, basis_kern_2, ix_func, ix_diff, vars_1,
                                vars_2):
        pass

    @abstractmethod
    def make_diff_kl_divergence_rev(self, basis_kern_2, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass

    def make_kl_functionals(self):
        """
        Set the attributes that compute the Kullback–Leibler (KL) divergence
        and its derivatives as functions of (non-fixed)
        parameters.
        Let :math:`f_{\\vartheta}` denote the basis kernel function with
        parameters :math:`\\vartheta`. The KL divergence between the basis kernel function
        :math:`f_{\\vartheta}` and some other (nonzero) basis kernel function
        :math:`g_{\\vartheta^\\prime}` is defined by

        .. math::
            D_{\\mathrm{KL}}(f_{\\vartheta} \\| g_{\\vartheta^\\prime):=\\int_{[0,+\\infty]} f_{\\vartheta}(x) \\log \\left(\\frac{f_{\\vartheta}(x)}{g_{\\vartheta^\\prime(x)}\\right) d x

        In our implementation we denote the functions:

        * :math:`\\| f_{\\vartheta} \\|_2` by the attribute `kl_divergence`;
        * :math:`\\frac{\\partial `\\| f_{\\vartheta} \\|_2`}{\\partial \\vartheta_p}` by the attribute `diff_kl_divergence`.

        """

        def kl_divergence(basis_kern_2, params_1, params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if self.is_compatible(basis_kern_2, is_reverse=False):
                return self.make_kl_divergence(basis_kern_2, vars_1, vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                return basis_kern_2.make_kl_divergence_rev(self, vars_2,
                                                           vars_1)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.kl_divergence = kl_divergence

        def diff_kl_divergence(basis_kern_2, ix_func, ix_diff, params_1,
                               params_2):
            vars_1 = self.make_vars(params_1)
            vars_2 = basis_kern_2.make_vars(params_2)
            if ix_func == 1:
                ix_diff_scaled = self.ix_map[ix_diff]
            elif ix_func == 2:
                ix_diff_scaled = basis_kern_2.ix_map[ix_diff]
            if self.is_compatible(basis_kern_2, is_reverse=False):
                return self.make_diff_kl_divergence(basis_kern_2, ix_func,
                                                    ix_diff_scaled, vars_1,
                                                    vars_2)
            elif basis_kern_2.is_compatible(self, is_reverse=True):
                return basis_kern_2.make_diff_kl_divergence(self, ix_func,
                                                            ix_diff_scaled,
                                                            vars_2, vars_1)
            raise NotImplementedError("No available interaction between these",
                                      "basis kernels")
        self.diff_kl_divergence = diff_kl_divergence
