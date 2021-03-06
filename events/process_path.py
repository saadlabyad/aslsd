# License: BSD 3 clause

import numpy as np

from aslsd.events import time_ordering


class ProcessPath():
    """
    Class for paths of point processes.

    A d-dimensional orderly point process is a random sequence of times

    .. math::
        \\mathcal{T}=\\left\\{t_{m}^{i}: m \\in \\mathbb{N}^{*}, i \\in[d], t_{m}^{i}<t_{m+1}^{i}\\right\\}

    The associated counting process :math:`\\boldsymbol{N}` is defined for
    :math:`t \\geq 0` by

    .. math::
        \\boldsymbol{N}_{t}:=\\left(N_{t}^{i}\\right)_{i \\in[d]}, \\quad \\text { where } \\quad N_{t}^{i}:=\\sum_{i=1}^{+\\infty} \\mathbb{1}_{\\left\\{t \\geq t_{i}^{j}\\right\\}} .

    This class models a path of a point process :math:`\\mathcal{T}` up to
    a terminal time :math:`T`.

    Attributes
    ----------
    list_times : `list` of `numpy.ndarray`
        list_times[i] is the sorted array of events of type :math:`i`.
    T_f : `float`
        Terminal time.
    d : `int`
        Dimension of the process.
    n_events : `list` of `int`
        n_events[i] is the number of events of type :math:`i`.
    eta : `numpy.ndarray`
        eta[i] is the global event rate :math:`\\eta_{T}^{i}:=\\frac{N_{T}^{i}}{T}`.
    list_times2end : `list` of `numpy.ndarray`
        list_times2end[i] is the array of times remaining times :math:`T-t^i_m`.
    kappa : `list` of `list` of `numpy.ndarray`
        kappa[j][i][m] is the index of the latest event of type :math:`j`
        preceding event time :math:`t_m^i`.
    varpi : `list` of `list` of `numpy.ndarray`
        varpi[i][j][h] is the smallest index :math:`m` such that there
        exists at least h events of type :math:`j` preceding event time
        :math:`t_m^i`.
    lag_sizes : `list` of `list` of `numpy.ndarray`
        lag_sizes[i][j][h] is the number of events of type :math:`i` with at
        least :math:`h` predecessors of type :math:`j`.
        By convention, matrix_lag_sizes[i][j][0]=0.

    """

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
