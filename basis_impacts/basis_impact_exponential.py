# License: BSD 3 clause

import numpy as np

from aslsd.basis_impacts.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class ExponentialImpact(BasisImpact):

    # Number of parameters
    def get_n_vars(self):
        return 1

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-10])

    # Parameter names
    def get_var_names(self):
        beta = '$\u03B2$'
        return [beta]

    # Impact functionals
    def make_impact(self, zeta, vars_):
        return vars_[0]*np.exp(-vars_[0]*zeta)

    def make_diff_impact(self, zeta, ix_diff, vars_):
        # Derivative wrt \beta
        if ix_diff == 0:
            return np.exp(-vars_[0]*zeta)*(1.-vars_[0]*zeta)
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis impacts",
                             "of type ", str(self.__class__))
