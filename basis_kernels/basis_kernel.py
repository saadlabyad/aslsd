# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np


class BasisKernel(ABC):
    """
    Abstract class for parametric families of basis kernels
    :math:`f_{\\vartheta}:[0,+\\infty) \to [0,+\\infty)`.
    Kernels :math:`\\phi` of an MHP will be defined as sums of basis kernels,
    .. math::
        \\phi_{\\theta}(t) := \\sum_{r=1}^{n_{\textrm{b}}}
        f^{(r)}_{\\vartheta_r}(t)

    where :math:`\\theta` is the vector of parameters of kernel :math:`\\phi`,
    obtainend by concatenating the vectors of parameters :math:`\\vartheta_r`
    of each of the :math:`n_b` basis kernels
    :math:`f^{(r)}_{\\vartheta}`, :math:`r \\in [n_b]`.


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

    Returns
    -------
    kappa : `list` of `list` of np.ndarray
        Description.

    varpi : `list` of `list` of np.ndarray
        Description.

    Notes
    ------
        To avoid any confusion, we add that the notion of (sums of) basis
        kernels is different from the notion of (sums of) basis
        functions discussed in Cartea, Á. and Cohen, S. N. and Labyad, S.,
        'Gradient-based estimation of linear Hawkes processes with general
        kernels'(November 22, 2021).
        Available at SSRN: https://ssrn.com/abstract=3969208
        In particular, we do not assume that all basis kernels only have one
        parameter, and are proportional to a function with fixed parameter
        values. That is, we do not assume that all basis kernels :math:`f`
        satisfy a structure like

        .. math::
            f_{\\vartheta} = \\omega g

        where :math:`g` is some function and :math:`\\vartheta=\\omega`.
    """

    def __init__(self, fixed_indices=None, fixed_vars=None, n_fixed_vars=0,
                 ix_map=None, dict_interactions=None, phi=None, diff_phi=None,
                 psi=None, diff_psi=None, upsilon=None, diff_sim_upsilon=None,
                 diff_cross_upsilon=None, allow_simu=True, sim_func=None,
                 l1_norm=None, diff_l1_norm=None, l2_norm=None, l2_dot=None,
                 l2_distance=None, kl_divergence=None,
                 diff_kl_divergence=None):
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

        self.make_kernel_functionals()

        self.allow_simu = allow_simu
        if self.allow_simu:
            def simu_func(rng, params, size=1):
                vars_ = self.make_vars(params)
                return self.make_simu_func(rng, vars_, size=size)
            self.simu_func = simu_func

        self.make_l1_metrics()
        self.make_l2_metrics()

    # Number of parameters
    @abstractmethod
    def get_n_vars(self):
        pass

    def get_n_param(self):
        return self.get_n_vars()-self.n_fixed_vars

    # Omega
    def has_omega_param(self):
        return (0 != self.fixed_indices[0])

    # Bounds
    @abstractmethod
    def get_var_bounds(self):
        pass

    def get_param_bounds(self):
        bnds = self.get_var_bounds()
        n_vars = len(bnds)
        return [bnds[i] for i in range(n_vars) if i not in self.fixed_indices]

    # Param names
    @abstractmethod
    def get_var_names(self):
        pass

    def get_param_names(self):
        """
        Get the list of names of the parameters of the basis kernel after
        fixing those we wanted to fix.

        Returns
        -------
        `list`

        """
        var_names = self.get_var_names()
        n_vars = len(var_names)
        return [var_names[i] for i in range(n_vars)
                if i not in self.fixed_indices]

    # Availabe interactions
    @abstractmethod
    def get_interactions(self, is_reverse=False):
        pass

    def make_dict_interactions(self):
        dict_interactions = {'dir': {}, 'rev': {}}
        list_inter_dir = self.get_interactions(is_reverse=False)
        for x in list_inter_dir:
            dict_interactions['dir'][str(type(x))] = True
        list_inter_rev = self.get_interactions(is_reverse=True)
        for x in list_inter_rev:
            dict_interactions['rev'][str(type(x))] = False
        return dict_interactions

    def is_compatible(self, basis_kern_2, is_reverse=False):
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
            Array of all parameters (i.e. fixed and non-fixed) of the kernel.

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

    def make_kernel_functionals(self):
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
