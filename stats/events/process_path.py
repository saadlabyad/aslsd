# License: BSD 3 clause

import bisect
import copy
import pickle
import itertools

import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from aslsd.stats.events import time_ordering
from aslsd.stats.events.path_event import PathEvent
import aslsd.utilities.useful_functions as uf


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

    def __init__(self, list_times=None, T_f=1., T_i=0., list_marks=None,
                 n_events=None, eta=None,
                 list_times2end=None, kappa=None, varpi=None, lag_sizes=None,
                 book_keeping=True):
        if list_times is None:
            list_times = []
        self.T_f = T_f
        self.T_i = T_i
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
        if T_f == 0.:
            self.eta = 0.
        else:
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
        # Trivial cases
        if T_trunc == 0:
            list_times = [np.array([]) for i in range(d)]
            return ProcessPath(list_times, T_trunc)
        if T_trunc >= self.T_f:
            return self.clone()
        # General cases
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
        trunc_path = ProcessPath(trunc_times, T_trunc,
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

    def get_next_event(self, t_ref):
        if uf.is_array(t_ref):
            times = np.concatenate(self.list_times)
            event_types = np.concatenate([i*np.ones(len(self.list_times[i]))
                                          for i in range(self.d)])
            ixs_sort = np.argsort(times)+0
            times = times[ixs_sort]
            event_types = event_types[ixs_sort]
            if self.list_marks is not None:
                marks = np.concatenate(self.list_marks)
                marks = marks[ixs_sort]
            ixs_next = np.searchsorted(times, t_ref, side='right')
            res = [None]*len(ixs_next)
            for ix in range(len(ixs_next)):
                t_next = times[ixs_next[ix]]
                dim_next = event_types[ixs_next[ix]]
                mark_next = None
                if self.list_marks is not None:
                    mark_next = marks[ixs_next[ix]]
                next_event = PathEvent(time=t_next, dim=dim_next,
                                       mark=mark_next)
                res[ix] = next_event
            return res
        else:
            res = self.get_next_event(np.array([t_ref]))
            return res[0]

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

    def get_sample_grid(self, sample_period):
        n_bins = int(self.T_f/sample_period)
        grid = sample_period*np.arange(n_bins+1)
        return grid

    def get_sample_hist(self, sample_period, grid=None):
        d = self.d
        if grid is None:
            grid = self. get_sample_grid(sample_period)
        # Compute
        sample_hist = np.array([np.histogram(self.list_times[i],
                                             bins=grid)[0]
                                for i in range(d)])
        return sample_hist

    def get_empirical_intensity(self, sample_period, grid=None,
                                sample_hist=None):
        d = self.d
        if sample_hist is None:
            # Compute
            sample_hist = self.get_sample_hist(sample_period, grid=grid)
        emp_int = [None]*d
        for i in range(d):
            emp_int[i] = sample_hist[i]/sample_period
        return emp_int

    def get_empirical_covariance(self, sample_period, grid=None, max_tau=None,
                                 sample_hist=None,
                                 verbose=False):
        d = self.d
        n_bins = int(self.T_f/sample_period)
        if sample_hist is None:
            # Compute
            sample_hist = self.get_sample_hist(sample_period, grid=grid)
        if max_tau is None:
            max_lag = n_bins-1
        else:
            max_lag = int(max_tau/sample_period)
        max_time = (max_lag+1)*sample_period

        # Compute cov
        sample_cov = np.zeros((d, d, max_lag+1))
        for lag in tqdm(range(max_lag+1), disable=not verbose):
            for i, j in itertools.product(range(d), range(d)):
                n_j = sample_hist[j, :n_bins-lag]
                n_i = sample_hist[i, lag:]
                sample_cov[i, j, lag] = (np.mean(n_j*n_i)
                                         - np.mean(n_j)*np.mean(n_i))
        sample_cov /= sample_period
        return sample_cov

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
