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
                probas[-1] = 1.-np.sum(probas[:-1])
            else:
                probas = self.estimate_mark_probas(data)
        else:
            probas[-1] = 1.-np.sum(probas[:-1])

        self.probas = np.array(probas)
        # Initialise Abstract class
        Mark.__init__(self, default_exp_imp='exact')

    def simulate(self, size=1, rng=None, seed=1234):
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)
        # Generation
        marks = rng.choice(self.marks_set, size=size, replace=True,
                           p=self.probas)
        return marks

    def exact_expected_basis_impact(self, basis_impact, imp_params):
        # Compute beta
        vec_beta = np.zeros(self.J)
        vec_beta[:-1] = 0.+imp_params
        vec_beta[-1] = 1.-np.sum(imp_params)
        # Compute expectation
        mean = np.sum(self.probas*vec_beta)
        return mean

    def estimate_mark_probas(self, data):
        count_dict = dict(pd.Series(data).value_counts())
        empirical_counts = np.array([count_dict[k] for k in range(self.J)])
        empirical_probas = empirical_counts/np.sum(empirical_counts)
        return empirical_probas
