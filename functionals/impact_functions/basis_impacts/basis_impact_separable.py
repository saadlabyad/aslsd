# License: BSD 3 clause

import itertools

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact
from aslsd.utilities import useful_functions as uf


class SeparableImpact(BasisImpact):
    def __init__(self, basis_impacts):
        self.basis_impacts = basis_impacts
        self.n_basis_imp = len(basis_impacts)
        # Mark Dimension
        self.mark_dim = 0
        self.vec_mark_dims = np.zeros(self.n_basis_imp, dtype=int)
        self.dim_bounds = np.zeros((self.n_basis_imp, 2), dtype=int)
        for ix in range(self.n_basis_imp):
            basis_mark_dim = self.basis_impacts[ix].get_mark_dim()
            self.dim_bounds[ix][0] = self.mark_dim
            self.dim_bounds[ix][1] = self.mark_dim+basis_mark_dim
            self.vec_mark_dims[ix] = basis_mark_dim
            self.mark_dim += basis_mark_dim
        # Number of variables
        self.n_vars = 0
        self.var_ixs_bounds = np.zeros((self.n_basis_imp, 2), dtype=int)
        for ix in range(self.n_basis_imp):
            basis_n_param = self.basis_impacts[ix].get_n_param()
            self.var_ixs_bounds[ix, 0] = self.n_vars
            self.var_ixs_bounds[ix, 1] = (self.var_ixs_bounds[ix, 0]
                                          + basis_n_param)
            self.n_vars += basis_n_param
        # Variables bounds and variables names
        self.var_lower_bounds = np.zeros(self.n_vars)
        self.var_upper_bounds = np.zeros(self.n_vars)
        self.var_names = [None]*self.n_vars
        for ix in range(self.n_basis_imp):
            start = self.var_ixs_bounds[ix][0]
            end = self.var_ixs_bounds[ix][1]
            self.var_lower_bounds[start:end] = self.basis_impacts[ix].get_param_lower_bounds()
            self.var_upper_bounds[start:end] = self.basis_impacts[ix].get_param_upper_bounds()
            self.var_names[start:end] = self.basis_impacts[ix].get_param_names()
        # Map between ixs
        prod_map = self.make_basis_selection()
        self.prod_map = prod_map
        # Constructor
        BasisImpact.__init__(self)

    # Input dimension
    def get_mark_dim(self):
        return self.mark_dim

    # Number of variables
    def get_n_vars(self):
        return self.n_vars

    # Bounds
    def get_var_lower_bounds(self):
        return self.var_lower_bounds

    def get_var_upper_bounds(self):
        return self.var_upper_bounds

    # Parameter names
    def get_var_names(self):
        return self.var_names

    # Interfacing
    def select_xi(self, xi, ix_func):
        start = self.dim_bounds[ix_func][0]
        end = self.dim_bounds[ix_func][1]
        if xi.ndim == 1:
            return xi[start:end]
        elif xi.ndim == 2:
            return xi[:, start:end]

    def select_var(self, vars_, ix_func):
        start = self.var_ixs_bounds[ix_func][0]
        end = self.var_ixs_bounds[ix_func][1]
        return vars_[start:end]

    def make_basis_selection(self):
        prod_map = [{'ix_func': None, 'ix_param': None}
                    for ix in range(self.n_vars)]
        ix_counter = 0
        for ix_func in range(self.n_basis_imp):
            basis_imp = self.basis_impacts[ix_func]
            for ix_param in range(basis_imp.get_n_param()):
                prod_map[ix_counter]['ix_func'] = ix_func
                prod_map[ix_counter]['ix_param'] = ix_param
                ix_counter += 1
        return prod_map

    # Impact functionals
    def make_impact(self, xi, vars_):
        if xi.ndim == 1:
            res = 1.
        elif xi.ndim == 2:
            res = np.ones(len(xi))
        for ix_func in range(self.n_basis_imp):
            basis_imp = self.basis_impacts[ix_func]
            xi_f = self.select_xi(xi, ix_func)
            vars_f = self.select_var(vars_, ix_func)
            res *= basis_imp.impact(xi_f, vars_f)
        return res

    def make_diff_impact(self, xi, ix_diff, vars_):
        ix_func_diff = self.prod_map[ix_diff]['ix_func']
        ix_param = self.prod_map[ix_diff]['ix_param']

        # Derivative
        basis_imp = self.basis_impacts[ix_func_diff]
        vars_f = self.select_var(vars_, ix_func_diff)
        xi_f = self.select_xi(xi, ix_func_diff)
        res = basis_imp.diff_impact(xi_f, ix_param, vars_f)

        for ix_func in itertools.chain(range(ix_func_diff),
                                       range(ix_func_diff+1, self.n_basis_imp)):
            basis_imp = self.basis_impacts[ix_func]
            xi_f = self.select_xi(xi, ix_func)
            vars_f = self.select_var(vars_, ix_func)
            res *= basis_imp.impact(xi_f, vars_f)
        return res
