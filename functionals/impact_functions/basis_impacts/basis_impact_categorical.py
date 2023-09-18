# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class CategoricalImpact(BasisImpact):
    def __init__(self, J, fixed_indices=None, fixed_vars=None):
        self.J = J
        BasisImpact.__init__(self, fixed_indices=fixed_indices,
                             fixed_vars=fixed_vars)

    # Input dimension
    def get_mark_dim(self):
        return 1

    # Number of Variables
    def get_n_vars(self):
        return self.J-1

    # Bounds
    def get_var_lower_bounds(self):
        return (10**-10)*np.ones(self.J-1)

    def get_var_upper_bounds(self):
        return np.inf*np.ones(self.J-1)

    # Parameter names
    def get_var_names(self):
        return ['$\u03B2_{'+str(k+1)+'}$' for k in range(self.J-1)]

    # Impact functionals
    def make_impact(self, xi, vars_):
        if uf.is_array(xi):
            res = np.ones(len(xi))
            sub_ixs = np.where((xi > 0))[0]
            sub_vals = xi[sub_ixs]-1
            sub_vals = sub_vals.astype(int)
            sub_vals = sub_vals.flatten()
            res[sub_ixs] = vars_[sub_vals]
            if xi.ndim == 1:
                return res[0]
            elif xi.ndim == 2:
                return res
        else:
            self.make_impact(np.array([xi]), vars_)

    def make_diff_impact(self, xi, ix_diff, vars_):
        if uf.is_array(xi):
            # Derivative wrt \beta
            if ix_diff < self.J-1:
                res = np.zeros(len(xi))
                ixs_p = np.where((xi == ix_diff+1))[0]
                res[ixs_p] = 1.
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
