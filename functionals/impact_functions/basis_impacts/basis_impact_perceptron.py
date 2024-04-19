# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf
from aslsd.utilities.useful_numerics import logistic_function


def diff_logsitic_func(x=None, f=None):
    if f is None:
        f = logistic_function(x)
    res = f*(1.-f)
    return res


class PerceptronImpact(BasisImpact):
    def __init__(self, mark_dim=1, fixed_indices=None, fixed_vars=None):
        self.mark_dim = mark_dim
        BasisImpact.__init__(self, fixed_indices=fixed_indices,
                             fixed_vars=fixed_vars)

    # Input dimension
    def get_mark_dim(self):
        return self.mark_dim

    # Number of variables
    def get_n_vars(self):
        return 1+self.mark_dim

    # Bounds
    def get_var_lower_bounds(self):
        n_vars = self.get_n_vars()
        res = np.ones(n_vars)
        res[0] = 10**-10
        res[1:] = -np.inf
        return res

    def get_var_upper_bounds(self):
        n_vars = self.get_n_vars()
        return np.inf*np.ones(n_vars)

    # Parameter names
    def get_var_names(self):
        n_vars = self.get_n_vars()
        var_names = [r'$a_{'+str(ix)+'}$' for ix in range(n_vars)]
        return var_names

    # Impact functionals
    def make_impact(self, xi, vars_):
        # Vectorization
        return_float = False
        if not uf.is_array(xi):
            return_float = True
            xi = np.array([xi])
        # Computations
        f_bias = logistic_function(vars_[0])
        a = np.array(vars_[1:])
        a = a.reshape((1, len(vars_)))
        a_xi = vars_[0]+np.sum(xi*a, axis=xi.ndim-1)
        f_xi = logistic_function(a_xi)
        res = f_xi/f_bias
        if return_float:
            res = res[0]
        return res

    def make_diff_impact(self, xi, ix_diff, vars_):
        # Vectorization
        return_float = False
        if not uf.is_array(xi):
            return_float = True
            xi = np.array([xi])
        # Computations
        f_bias = logistic_function(vars_[0])
        impact_xi = self.make_impact(xi, vars_)
        if ix_diff == 0:
            res = f_bias*impact_xi*(1.-impact_xi)
        else:
            xi_p = xi[ix_diff-1]
            res = xi_p*impact_xi*(1.-f_bias*impact_xi)
        if return_float:
            res = res[0]
        return res
