# License: BSD 3 clause

import numpy as np

from aslsd.events import time_ordering


class ProcessPath():
    def __init__(self, list_times, T_f, d=None, n_events=None, eta=None,
                 list_times2end=None, kappa=None, varpi=None, lag_sizes=None):

        self.list_times = list_times
        self.T_f = T_f
        self.d = len(self.list_times)
        self.n_events = np.array([len(L) for L in self.list_times])
        self.eta = self.n_events/T_f

        self.list_times2end = [T_f - L for L in list_times]

        if kappa is None or varpi is None:
            kappa, varpi = time_ordering.get_kappa_varpi(list_times, T_f)
        self.kappa = kappa
        self.varpi = varpi

        if lag_sizes is None:
            lag_sizes = time_ordering.get_lag_sizes(list_times, self.kappa,
                                                    self.varpi)
        self.lag_sizes = lag_sizes
