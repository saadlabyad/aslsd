# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel


class NullKernel(BasisKernel):
    """
    Class for null kernels. The associated basis function
    is defined for all :math:`t \\geq 0` by

    .. math::
        f_{\\vartheta}(t) := 0.

    Notes
    ------
        The closed form expressions for the derivatives of the functions
        :math:`\\Phi`, :math:`\\Psi`, and :math:`\\Upsilon` are available in
        Cartea, Á., Cohen, S. N., and Labyad, S., (November 22, 2021)
        'Gradient-based estimation of linear Hawkes processes with general
        kernels'.
        `Available at SSRN. <https://ssrn.com/abstract=3969208>`_

    """

    # Number of parameters
    def get_n_vars(self):
        return 0

    # Bounds
    def get_var_bounds(self):
        return np.array([])

    # Param names
    def get_var_names(self):
        return []

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return ['NullKernel', 'ExponentialKernel', 'GaussianKernel',
                    'TriangularKernel', 'RayleighKernel', 'BasisKernelMC']

    # Kernel functionals
    def make_phi(self, t, vars_):
        return 0.*t

    def make_diff_phi(self, t, ix_diff, vars_):
        return 0.*t

    def make_psi(self, t, vars_):
        return 0.*t

    def make_diff_psi(self, t, ix_diff, vars_):
        return 0.*t

    def make_upsilon(self, basis_kern_2, t, s, vars_1, vars_2):
        return 0.*t

    def make_upsilon_rev(self, basis_kern_2, t, s, vars_2, vars_1):
        return 0.*t

    def make_diff_sim_upsilon(self, t, s, ix_diff, vars_):
        return 0.*t

    def make_diff_cross_upsilon(self, basis_kern_2, t, s, ix_func, ix_diff,
                                vars_1, vars_2):
        return 0.*t

    def make_diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                    vars_2, vars_1):
        return 0.*t

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        return np.zeros(size)

    # Metrics
    def make_l1_norm(self, vars_):
        return 0.

    def make_diff_l1_norm(self, ix_diff, vars_):
        return 0.

    def make_l2_norm(self, vars_):
        return 0.

    def make_diff_l2_norm(self, ix_diff, vars_):
        return 0.

    def make_l2_dot(self, basis_kern_2, vars_1, vars_2):
        return 0.

    def make_diff_l2_dot(self, basis_kern_2, ix_func, ix_diff, vars_1, vars_2):
        return 0.

    # KL divergence
    def make_kl_divergence(self, basis_kern_2, vars_1, vars_2):
        pass

    def make_kl_divergence_rev(self, basis_kern_2, vars_2, vars_1):
        pass

    def make_diff_kl_divergence(self, basis_kern_2, ix_func, ix_diff, vars_1,
                                vars_2):
        pass

    def make_diff_kl_divergence_rev(self, basis_kern_2, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass
