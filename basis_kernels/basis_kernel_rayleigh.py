# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


class RayleighKernel(BasisKernel):

    # Number of parameters
    def get_n_vars(self):
        return 2

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-10, 10**-10])

    # Param names
    def get_var_names(self):
        omega = '$\u03C9$'
        beta = '$\u03B2$'
        return [omega, beta]

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return ['NullKernel', 'RayleighKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        omega = vars_[0]
        beta = vars_[1]
        return omega*(t/beta**2)*np.exp(-0.5*(t/beta)**2)

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

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        return rng.rayleigh(scale=vars_[1], size=size)

    # Metrics
    def make_l1_norm(self, vars_):
        return vars_[0]

    def make_diff_l1_norm(self, ix_diff, vars_):
        # Differential wrt Omega
        if ix_diff == 0:
            return 1.
        # Differential wrt Beta
        elif ix_diff == 1:
            return 0.

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
