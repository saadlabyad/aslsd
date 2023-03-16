# License: BSD 3 clause

import numpy as np

from aslsd.stats.marks.mark import Mark


class VoidMark(Mark):

    def simulate(self, size=1, rng=None, seed=1234):
        return np.zeros(size)

    def exact_expected_basis_impact(self, basis_impact, mark_params,
                                    imp_params):
        pass
