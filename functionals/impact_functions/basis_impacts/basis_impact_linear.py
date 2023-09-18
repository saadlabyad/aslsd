# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class LinearImpact(BasisImpact):
    def __init__(self, mark_dim=1, fixed_indices=None, fixed_vars=None):
        self.mark_dim = mark_dim
        BasisImpact.__init__(self, fixed_indices=fixed_indices,
                             fixed_vars=fixed_vars)

    # Input dimension
    def get_mark_dim(self):
        return self.mark_dim

    # Number of variables
    def get_n_vars(self):
        return self.mark_dim

    # Bounds
    def get_var_lower_bounds(self):
        return (10**-10)*np.ones(self.mark_dim)

    def get_var_upper_bounds(self):
        return np.inf*np.ones(self.mark_dim)

    # Parameter names
    def get_var_names(self):
        var_names = [r'$a_{'+str(ix)+'}$' for ix in range(self.mark_dim)]
        return var_names

    # Impact functionals
    def make_impact(self, xi, vars_):
        if uf.is_array(xi):
            res = 1.+np.sum(xi*vars_, axis=xi.ndim-1)
            return res
        else:
            return 1.+xi*vars_[0]

    def make_diff_impact(self, xi, ix_diff, vars_):
        if uf.is_array(xi):
            if xi.ndim == 1:
                return xi[ix_diff]
            elif xi.ndim == 2:
                return xi[:, ix_diff]
            else:
                raise ValueError('xi does not have the right dimension.')
        else:
            return xi
