# License: BSD 3 clause

import numpy as np

from aslsd.functionals.baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf


class LinearBaseline(BasisBaseline):
    # Number of parameters
    def get_n_vars(self):
        return 2

    def get_var_lower_bounds(self):
        return np.array([10**-10, 10**-10])

    def get_var_upper_bounds(self):
        return np.array([np.inf, np.inf])

    def get_var_names(self):
        return ['$a$', '$b$']

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return []

    def make_mu(self, t, vars_):
        return vars_[0]*t+vars_[1]

    def make_diff_mu(self, t, ix_diff, vars_):
        if ix_diff == 0:
            return t
        elif ix_diff == 1:
            if uf.is_array(t):
                return np.ones(len(t))
            else:
                return 1.

    def make_M(self, t, vars_):
        a, b = vars_
        return ((a**2)/3.)*t**2+a*b*t+b**2

    def make_diff_M(self, t, ix_diff, vars_):
        a, b = vars_
        if ix_diff == 0:
            return (2./3.)*a*t**2+b*t
        elif ix_diff == 1:
            return a*t+2.*b

    # Simulatiom
    def make_compensator(self, t, vars_):
        return 0.5*vars_[0]*t**2+vars_[1]*t

    def make_inv_compensator(self, y, vars_):
        return (-vars_[1]+np.sqrt(vars_[1]**2+2.*vars_[0]*y))/(vars_[0])

    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass
