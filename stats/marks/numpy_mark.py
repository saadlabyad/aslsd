# License: BSD 3 clause

import numpy as np

from aslsd.stats.marks.mark import Mark


class NumpyMark(Mark):
    def __init__(self, rv_name='uniform', param_names=None, mark_params=None,
                 default_exp_imp='estimate'):
        self.rv_name = rv_name
        if param_names is None:
            dict_params = {}
        else:
            dict_params = {}
            assert len(param_names) == len(mark_params), "param_names and mark_params should be of the same size"
            for ix in range(len(param_names)):
                dict_params[param_names[ix]] = mark_params[ix]
        self.dict_params = dict_params
        Mark.__init__(self, mark_params=mark_params,
                      default_exp_imp=default_exp_imp)

    def simulate(self, size=1, rng=None, seed=1234):
        if rng is None:
            rng = np.random.default_rng(seed)
        generator = getattr(rng, self.rv_name)
        return generator(size=size, **self.dict_params)

    def exact_expected_basis_impact(self, basis_impact, mark_params,
                                    imp_params):
        pass
