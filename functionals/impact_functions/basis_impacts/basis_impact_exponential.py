# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class ExponentialImpact(BasisImpact):

    # Input dimension
    def get_mark_dim(self):
        return 1

    # Number of Variables
    def get_n_vars(self):
        return 1

    # Bounds
    def get_var_lower_bounds(self):
        return np.array([10**-10])

    def get_var_upper_bounds(self):
        return np.array([np.inf])

    # Parameter names
    def get_var_names(self):
        beta = '$\u03B2$'
        return [beta]

    # Impact functionals
    def make_impact(self, xi, vars_):
        if uf.is_array(xi):
            res = np.exp(-vars_[0]*xi)
            if xi.ndim == 1:
                return res[0]
            elif xi.ndim == 2:
                return res
        else:
            self.make_impact(np.array([xi]), vars_)

    def make_diff_impact(self, xi, ix_diff, vars_):
        if uf.is_array(xi):
            # Derivative wrt \beta
            if ix_diff == 0:
                res = -xi*np.exp(-vars_[0]*xi)
                if xi.ndim == 1:
                    return res[0]
                elif xi.ndim == 2:
                    return res
            else:
                raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                                 "valid differentiation index for basis impacts",
                                 "of type ", str(self.__class__))
        else:
            self.make_diff_impact(np.array([xi]), ix_diff, vars_)
