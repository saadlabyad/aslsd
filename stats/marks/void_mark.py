# License: BSD 3 clause

import numpy as np

from aslsd.stats.marks.mark import Mark


class VoidMark(Mark):

    def get_mark_dim(self):
        return 0

    def simulate(self, size=1, rng=None, seed=1234):
        if type(size) != tuple:
            xi = np.zeros((size, self.get_mark_dim()))
        else:
            xi = np.zeros(size)
        return xi

    def exact_expected_basis_impact(self, basis_impact, basis_imp_params):
        pass
