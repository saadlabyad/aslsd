# License: BSD 3 clause

import numpy as np
import pandas as pd

from aslsd.stats.marks.mark import Mark
from aslsd.utilities import useful_statistics as us


class IndProdMark(Mark):
    def __init__(self, list_marks):
        self.list_marks = list_marks
        self.n_mark_types = len(list_marks)
        # mark dim
        self.mark_dim = 0
        self.vec_mark_dims = np.zeros(self.n_mark_types, dtype=int)
        self.dim_bounds = np.zeros((self.n_mark_types, 2), dtype=int)
        for ix in range(self.n_mark_types):
            basis_mark_dim = self.list_marks[ix].get_mark_dim()
            self.dim_bounds[ix][0] = self.mark_dim
            self.dim_bounds[ix][1] = self.mark_dim+basis_mark_dim
            self.vec_mark_dims[ix] = basis_mark_dim
            self.mark_dim += basis_mark_dim
        # Initialise Abstract class
        Mark.__init__(self, default_exp_imp='exact')

    def get_mark_dim(self):
        return self.mark_dim

    def simulate(self, size=1, rng=None, seed=1234):
        if size == 0:
            return np.zeros(0)
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Generation
        marks = np.zeros((size, self.get_mark_dim()))
        for ix in range(self.n_mark_types):
            mark = self.list_marks[ix]
            start = self.dim_bounds[ix][0]
            end = self.dim_bounds[ix][1]
            marks[:, start:end] = mark.simulate(size=size, rng=rng, seed=1234)
        return marks

    def exact_expected_basis_impact(self, basis_impact, imp_params):
        pass

    def estimate_mark_probas(self, data):
        pass
