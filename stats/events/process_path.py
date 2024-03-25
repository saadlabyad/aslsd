# License: BSD 3 clause

import bisect
import copy
import pickle

import numpy as np
from scipy.interpolate import interp1d

from aslsd.stats.events import time_ordering


class ProcessPath():
    """
    Class for a finite path of a point processes.

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

    def __init__(self, list_times=None, T_f=1., list_marks=None,
                 n_events=None, eta=None,
                 list_times2end=None, kappa=None, varpi=None, lag_sizes=None,
                 book_keeping=True):
        if list_times is None:
            list_times = []
        self.T_f = T_f
        self.d = len(list_times)
        # Event data
        self.list_times = [0.+np.array(list_times[i]) for i in range(self.d)]
        if list_marks is None:
            self.list_marks = None
        else:
            self.list_marks = [0+np.array(list_marks[i])
                               for i in range(self.d)]
        # Basic statistics
        self.n_events = np.array([len(L) for L in self.list_times])
        self.eta = self.n_events/T_f
        # Book-keeping
        self.book_keeping = book_keeping
        if book_keeping:
            if list_times2end is None:
                self.list_times2end = [T_f - L for L in list_times]
            else:
                self.list_times2end = list_times2end

            if kappa is None or varpi is None:
                kappa, varpi = time_ordering.get_kappa_varpi(list_times, T_f)
            self.kappa = kappa
            self.varpi = varpi

            if lag_sizes is None:
                lag_sizes = time_ordering.get_lag_sizes(list_times, self.kappa,
                                                        self.varpi)
            self.lag_sizes = lag_sizes
        else:
            self.list_times2end = list_times2end
            self.kappa = kappa
            self.varpi = varpi
            self.lag_sizes = lag_sizes

# =============================================================================
# List operations
# =============================================================================
    def get_book_keeping(self):
        self.list_times2end = [self.T_f - L for L in self.list_times]

        kappa, varpi = time_ordering.get_kappa_varpi(self.list_times, self.T_f)
        self.kappa = kappa
        self.varpi = varpi

        lag_sizes = time_ordering.get_lag_sizes(self.list_times, self.kappa,
                                                self.varpi)
        self.lag_sizes = lag_sizes

    def clone(self):
        list_times = self.list_times
        process_path = ProcessPath(self.list_times, self.T_f, d=self.d,
                                   list_marks=self.list_marks,
                                   n_events=self.n_events,
                                   eta=self.eta,
                                   list_times2end=self.list_times2end,
                                   kappa=self.kappa, varpi=self.varpi,
                                   lag_sizes=self.lag_sizes,
                                   book_keeping=self.book_keeping)
        return process_path

    def truncate(self, T_trunc):
        d = self.d
        if T_trunc >= self.T_f:
            self.clone()
        list_n_f = [bisect.bisect_left(self.list_times[i], T_trunc)-1
                    for i in range(self.d)]

        trunc_times = [self.list_times[i][:list_n_f[i]+1] for i in range(d)]
        if self.list_marks is not None:
            trunc_marks = [self.list_marks[i][:list_n_f[i]+1]
                           for i in range(d)]
        else:
            trunc_marks = None
        # Book-keeping
        if self.kappa is None or self.varpi is None:
            trunc_kappa = None
            trunc_varpi = None
        else:
            # Kappa
            trunc_kappa = [[self.kappa[j][i][:list_n_f[i]+1]
                            for i in range(d)] for j in range(d)]
            # Varpi
            trunc_varpi = [[self.varpi[i][j][:list_n_f[j]+1]
                            for j in range(d)] for i in range(d)]

        # Return
        trunc_path = ProcessPath(trunc_times, T_trunc, d=self.d,
                                 list_marks=trunc_marks,
                                 n_events=None,
                                 eta=None,
                                 list_times2end=None,
                                 kappa=trunc_kappa, varpi=trunc_varpi,
                                 lag_sizes=None,
                                 book_keeping=self.book_keeping)
        return trunc_path

    def get_counting_process(self):
        d = self.d
        counting_process = [None]*d
        for i in range(d):
            # Times
            times = np.zeros(len(self.list_times[i])+2)
            times[1:-1] = self.list_times[i]
            times[-1] = self.T_f
            # Counts
            vals = np.arange(len(self.list_times[i])+2)
            vals[-1] = len(self.list_times[i])
            time_func = interp1d(times, vals, kind='previous')
            # Counting process
            counting_process[i] = time_func
        self.counting_process = counting_process
        return counting_process

    def censor(self, del_ixs, book_keeping=False):
        d = self.d
        # Times
        censored_times = copy.deepcopy(self.list_times)
        censored_marks = copy.deepcopy(self.list_marks)
        for i in range(d):
            if (del_ixs[i] is not None) and (len(del_ixs[i]) > 0):
                censored_times[i] = np.delete(censored_times[i], del_ixs[i])
                if censored_marks is not None:
                    censored_marks[i] = np.delete(censored_marks[i],
                                                  del_ixs[i])
        return ProcessPath(censored_times, self.T_f, d=d,
                           list_marks=censored_marks,
                           book_keeping=book_keeping)

    def get_inter_arrival_times(self):
        list_times = self.list_times
        ia_times = [None]*self.d
        for i in range(self.d):
            ia_times[i] = list_times[i][1:]-list_times[i][:-1]
        return ia_times

# =============================================================================
# Statistics
# =============================================================================
    def get_events_rate(self, interval_bounds):
        rate = [None]*self.d
        for i in range(self.d):
            interval_sizes = interval_bounds[1:]-interval_bounds[:-1]
            rate[i] = np.histogram(self.times[i],
                                   interval_bounds[i])[0]/interval_sizes
        return rate

# =============================================================================
# Serialization
# =============================================================================
    def save(self, file, **kwargs):
        dict_attr = {'list_times': self.list_times,
                     'list_marks': self.list_marks, 'T_f': self.T_f,
                     'd': self.d, 'n_events': self.n_events,
                     'eta': self.eta, 'book_keeping': self.book_keeping}
        if self.book_keeping:
            dict_attr['list_times2end'] = self.list_times2end
            dict_attr['kappa'] = self.kappa
            dict_attr['varpi'] = self.varpi
            dict_attr['lag_sizes'] = self.lag_sizes

        pickle_out = open(file, "wb", **kwargs)
        pickle.dump(dict_attr, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        saving_file = open(file, "rb")
        dict_attr = pickle.load(saving_file)
        saving_file.close()
        self.list_times = dict_attr['list_times']
        self.list_marks = dict_attr['list_marks']
        self.T_f = dict_attr['T_f']
        self.d = dict_attr['d']
        self.n_events = dict_attr['n_events']
        self.eta = dict_attr['eta']
        self.book_keeping = dict_attr['book_keeping']
        if self.book_keeping:
            self.list_times2end = dict_attr['list_times2end']
            self.kappa = dict_attr['kappa']
            self.varpi = dict_attr['varpi']
            self.lag_sizes = dict_attr['lag_sizes']
