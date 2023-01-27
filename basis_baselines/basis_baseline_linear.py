# License: BSD 3 clause

import numpy as np

from aslsd.basis_baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf


class LinearBaseline(BasisBaseline):
    # Number of parameters
    def get_n_vars(self):
        return 2

    def get_var_bounds(self):
        return [10**-10, 10**-10]

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
        return ((vars_[0]**2)/3.)*t**2+vars_[0]*vars_[1]*t+vars_[1]**2

    def make_diff_M(self, t, ix_diff, vars_):
        if ix_diff == 0:
            return (2.*vars_[0]/3.)*t**2+vars_[1]*t
        elif ix_diff == 1:
            return vars_[0]*t+2.*vars_[1]

    def make_K(self, basis_ker, t, s, vars_ker, vars_mu):
        pass

    def make_diff_K(self, basis_ker, t, s, ix_func, ix_diff, vars_ker,
                    vars_mu):
        pass

    # Simulatiom
    def make_compensator(self, t, vars_):
        return 0.5*vars_[0]*t**2+vars_[1]*t

    def make_inv_compensator(self, y, vars_):
        return (-vars_[1]+np.sqrt(vars_[1]**2+2.*vars_[0]*y))/(vars_[0])

    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass
