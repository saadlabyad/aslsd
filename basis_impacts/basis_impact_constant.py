# License: BSD 3 clause

import numpy as np

from aslsd.basis_impacts.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class ConstantImpact(BasisImpact):

    # Number of parameters
    def get_n_vars(self):
        return 0

    # Bounds
    def get_var_bounds(self):
        return []

    # Parameter names
    def get_var_names(self):
        pass

    # Impact functionals
    def make_impact(self, t, vars_):
        if uf.is_array(t):
            return np.ones(len(t))
        else:
            return 1.

    def make_diff_impact(self, t, ix_diff, vars_):
        pass
