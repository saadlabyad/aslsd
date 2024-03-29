# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class ConstantImpact(BasisImpact):
    def __init__(self, mark_dim=1, fixed_indices=None, fixed_vars=None):
        self.mark_dim = mark_dim
        BasisImpact.__init__(self, fixed_indices=fixed_indices,
                             fixed_vars=fixed_vars)

    # Input dimension
    def get_mark_dim(self):
        return self.mark_dim

    # Number of variables
    def get_n_vars(self):
        return 0

    # Bounds
    def get_var_lower_bounds(self):
        return np.arrray([])

    def get_var_upper_bounds(self):
        return np.arrray([])

    # Parameter names
    def get_var_names(self):
        pass

    # Impact functionals
    def make_impact(self, xi, vars_):
        if uf.is_array(xi):
            return np.ones(len(xi))
        else:
            return 1.

    def make_diff_impact(self, xi, ix_diff, vars_):
        pass
