# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np


class Mark(ABC):
    def __init__(self, mark_params=None, default_exp_imp='estimate'):
        self.mark_params = mark_params
        self.default_exp_imp = default_exp_imp

    # Expected Impact
    def exact_expected_impact(self, impact_function, imp_params):
        mark_params = self.mark_params
        res = 0.
        for ix_imp in range(impact_function.n_basis_imp):
            basis_imp = impact_function._basis_impacts[ix_imp]
            res += self.exact_expected_basis_impact(basis_imp, mark_params,
                                                    imp_params[basis_imp.interval_map[ix_imp][0]:basis_imp.interval_map[ix_imp][1]])
        return res

    def estimate_expected_impact(self, impact_function, imp_params, n_mc=10**5,
                                 rng=None, seed=1234):
        if rng is None:
            rng = np.random.default_rng(seed)
        marks = self.simulate(size=n_mc, rng=rng)
        return np.mean(impact_function.impact(marks, imp_params))

    def get_expected_impact(self, impact_function, imp_params, n_mc=10**5,
                            rng=None, seed=1234):
        if self.default_exp_imp == 'estimate':
            return self.estimate_expected_impact(impact_function, imp_params,
                                                 n_mc=n_mc, rng=rng, seed=seed)
        elif self.default_exp_imp == 'exact':
            return self.exact_expected_impact(impact_function, imp_params)

    @abstractmethod
    def simulate(self, size=1, rng=None, seed=1234):
        pass

    @abstractmethod
    def exact_expected_basis_impact(self, basis_impact, mark_params,
                                    imp_params):
        pass
