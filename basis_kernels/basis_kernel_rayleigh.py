# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


class NullKernel(BasisKernel):

    # Number of parameters
    def get_n_vars(self):
        pass

    # Bounds
    def get_var_bounds(self):
        pass

    # Param names
    def get_var_names(self):
        pass

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        pass

    # Kernel functionals
    def make_phi(self, t, vars_):
        pass

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
        pass

    # Metrics
    def make_l1_norm(self, vars_):
        pass

    def make_diff_l1_norm(self, ix_diff, vars_):
        pass

    def make_l2_norm(self, vars_):
        pass

    def make_diff_l2_norm(self, ix_diff, vars_):
        pass

    def make_l2_dot(self, basis_kern_2, vars_1, vars_2):
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
