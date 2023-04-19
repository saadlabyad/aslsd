# License: BSD 3 clause

import numpy as np

from aslsd.functionals.baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf


class ConstantBaseline(BasisBaseline):
    # Number of parameters
    def get_n_vars(self):
        return 1

    def get_var_bounds(self):
        return [10**(-10)]

    def get_var_names(self):
        return ['$\u03BC$']

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return []

    def make_mu(self, t, vars_):
        if uf.is_array(t):
            return vars_[0]*np.ones(len(t))
        else:
            return vars_[0]

    def make_diff_mu(self, t, ix_diff, vars_):
        if uf.is_array(t):
            return np.ones(len(t))
        else:
            return 1.

    def make_M(self, t, vars_):
        if uf.is_array(t):
            return (vars_[0]**2)*np.ones(len(t))
        else:
            return vars_[0]**2

    def make_diff_M(self, t, ix_diff, vars_):
        if uf.is_array(t):
            return 2.*vars_[0]*np.ones(len(t))
        else:
            return 2.*vars_[0]

    # Simulatiom
    def make_compensator(self, t, vars_):
        return vars_[0]*t

    def make_inv_compensator(self, y, vars_):
        return y/vars_[0]

    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass