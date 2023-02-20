# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


class TriangularKernel(BasisKernel):
    """
    Class for triangular basis kernels. The associated basis function
    is defined for all :math:`t \\geq 0` by

    .. math::
        f_{\\vartheta}(t) := \\omega \\left(\\frac{t-\\alpha}{\\beta} \\mathbb{1}_{\\left\\{0 \\leq t-\\alpha \\leq \\beta\\right\\}}-\\frac{t-\\alpha-\\beta-\\delta}{\\delta} \\mathbb{1}_{\\left\\{0 \\leq t-\\alpha-\\beta \\leq \\delta\\right\\}}\\right).

    The parameters of this basis kernel are given by

    .. math::
        \\vartheta := (\\omega, \\alpha, \\beta, \\delta).

    where

    * :math:`\\omega \\geq 0` controls the :math:`L_{1}` norm of :math:`f`;
    * :math:`\\alpha \\geq 0` is the left corner of the triangle;
    * :math:`\\beta \\geq 0` is the distance to the altitude foot;
    * :math:`\\delta \\geq 0` is the distance between the altitude foot and the right corner of the triangle.

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
        return 4

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-10 for i in range(4)])

    # Param names
    def get_var_names(self):
        omega = '$\u03C9$'
        alpha = '$\u03B1$'
        beta = '$\u03B2$'
        delta = '$\u03B4$'
        return [omega, alpha, beta, delta]

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return ['NullKernel', 'TriangularKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        omega = vars_[0]
        alpha = vars_[1]
        beta = vars_[2]
        delta = vars_[3]
        return omega*uf.tri(t, alpha, beta, delta)

    def make_diff_phi(self, t, ix_diff, vars_):
        pass

    def make_psi(self, t, vars_):
        pass

    def make_diff_psi(self, t, ix_diff, vars_):
        pass

    def make_upsilon(self, basis_kern_2, t, s, vars_1, vars_2):
        pass

    def make_upsilon_rev(self, basis_kern_2, t, s, vars_2, vars_1):
        pass

    def make_diff_sim_upsilon(self, t, s, ix_diff, vars_):
        pass

    def make_diff_cross_upsilon(self, basis_kern_2, t, s, ix_func, ix_diff,
                                vars_1, vars_2):
        pass

    def make_diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass

    def make_K(self, baseline, t, s, vars_ker, params_mu):
        pass

    def make_diff_K(self, baseline, t, s, ix_func, ix_diff, vars_ker, params_mu):
        pass

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        alpha = vars_[1]
        beta = vars_[2]
        delta = vars_[3]
        left = alpha
        mode = alpha+beta
        right = mode+delta
        return rng.triangular(left, mode, right, size=size)

    # Metrics
    def make_l1_norm(self, vars_):
        omega = vars_[0]
        beta = vars_[2]
        delta = vars_[3]
        return 0.5*omega*(beta+delta)

    def make_diff_l1_norm(self, ix_diff, vars_):
        pass

    def make_l2_norm(self, vars_):
        pass

    def make_diff_l2_norm(self, ix_diff, vars_):
        pass

    def make_l2_dot(self, basis_kern_2, vars_1, vars_2):
        pass

    def make_diff_l2_dot(self, basis_kern_2, ix_func, ix_diff, vars_1, vars_2):
        pass

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
