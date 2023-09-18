# License: BSD 3 clause

import numpy as np

from aslsd.stats.marks.mark import Mark


class NumpyMark(Mark):
    def __init__(self, rv_name='uniform', mark_dim=1, param_names=None,
                 mark_params=None,
                 default_exp_imp='estimate'):
        self.rv_name = rv_name
        self.mark_dim = mark_dim
        if param_names is None:
            dict_params = {}
        else:
            dict_params = {}
            assert len(param_names) == len(mark_params), "param_names and mark_params should be of the same size"
            for ix in range(len(param_names)):
                dict_params[param_names[ix]] = mark_params[ix]
        self.dict_params = dict_params
        Mark.__init__(self, default_exp_imp=default_exp_imp)

    def get_mark_dim(self):
        return self.mark_dim

    def simulate(self, size=1, rng=None, seed=1234):
        if rng is None:
            rng = np.random.default_rng(seed)
        generator = getattr(rng, self.rv_name)
        xi = generator(size=size, **self.dict_params)
        if type(size) != tuple:
            xi = xi.reshape((size, self.get_mark_dim()))
        return xi

    def exact_expected_basis_impact(self, basis_impact, basis_imp_params):
        pass
