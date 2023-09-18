# License: BSD 3 clause

import numpy as np
import pandas as pd

from aslsd.stats.marks.mark import Mark
from aslsd.utilities import useful_statistics as us


class CategoricalMark(Mark):
    def __init__(self, J, probas=None, data=None):
        self.J = J
        self.marks_set = np.arange(J)
        # Probability vector
        if probas is None:
            if data is None:
                probas = (1./J)*np.zeros(J)
            else:
                probas = self.estimate_mark_probas(data)
        probas[-1] = 1.-np.sum(probas[:-1])
        self.probas = np.array(probas)
        # Initialise Abstract class
        Mark.__init__(self, default_exp_imp='exact')

    def get_mark_dim(self):
        return 1

    def simulate(self, size=1, rng=None, seed=1234):
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Generation
        xi = rng.choice(self.marks_set, size=size, replace=True, p=self.probas)
        if type(size) != tuple:
            xi = xi.reshape((size, self.get_mark_dim()))
        return xi

    def exact_expected_basis_impact(self, basis_impact, imp_params):
        # Compute beta
        vec_beta = np.ones(self.J)
        vec_beta[1:] = 0.+imp_params
        # Compute expectation
        mean = np.sum(self.probas*vec_beta)
        return mean

    def estimate_mark_probas(self, data):
        count_dict = dict(pd.Series(data).value_counts())
        empirical_counts = np.array([count_dict[k] for k in range(self.J)])
        empirical_probas = empirical_counts/np.sum(empirical_counts)
        return empirical_probas
