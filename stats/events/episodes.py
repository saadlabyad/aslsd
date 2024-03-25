# License: BSD 3 clause

import bisect
import copy
import pickle

import numpy as np
from scipy.interpolate import interp1d

from aslsd.stats.events.process_path import ProcessPath


class Episodes():
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

    def __init__(self, list_paths=None, T_f=1.):
        if list_paths is None:
            list_paths = []
        self.list_paths = list_paths
        self.n_episodes = len(self.list_paths)
        self.T_f = T_f
        # Dimension
        if self.n_episodes == 0:
            self.d = 0
        else:
            self.d = self.list_paths[0].d
        # Basic statistics
        self.list_n_events = np.zeros((self.d, self.n_episodes))
        self.list_eta = np.zeros((self.d, self.n_episodes))
        for ix in range(self.n_episodes):
            self.list_n_events[:, ix] = self.list_paths[ix].n_events
            self.list_eta[:, ix] = self.list_paths[ix].eta
        if self.n_episodes > 0:
            self.average_n_events = np.mean(self.list_n_events, axis=1)
            self.total_n_events = np.sum(self.list_n_events, axis=1, dtype=int)
            self.average_eta = np.mean(self.list_eta, axis=1)

# =============================================================================
# List operations
# =============================================================================
    def get_book_keeping(self):
        pass

    def clone(self):
        pass

    def truncate(self, T_trunc):
        list_trunc_paths = [None]*self.n_episodes
        for ix in range(self.n_episodes):
            list_trunc_paths[ix] = self.list_paths.truncate(T_trunc)
        trunc_episodes = Episodes(list_paths=list_trunc_paths, T_f=T_trunc)
        return trunc_episodes

    def get_counting_process(self):
        pass

    def censor(self, del_ixs, book_keeping=False):
        pass

    def get_inter_arrival_times(self):
        pass

# =============================================================================
# Statistics
# =============================================================================
    def get_events_rate(self, interval_bounds):
        rate = [np.zeros(len(interval_bounds[i])-1) for i in range(self.d)]
        for ix in range(self.n_episodes):
            path_rate = self.list_paths.get_events_rate(interval_bounds)
            for i in range(self.d):
                rate[i] += path_rate[i]
        return rate

# =============================================================================
# Serialization
# =============================================================================
    def save(self, file, **kwargs):
        for ix in range(self.n_episodes):
            self.list_paths[ix].save(file+'_path_'+str(ix))
        dict_attr = {'n_episodes': self.n_episodes, 'T_f': self.T_f,
                     'd': self.d, 'list_n_events': self.list_n_events,
                     'list_eta': self.list_eta,
                     'average_n_events': self.average_n_events,
                     'total_n_events': self.total_n_events,
                     'average_eta': self.average_eta}
        pickle_out = open(file+'_stats_attr', "wb", **kwargs)
        pickle.dump(dict_attr, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        saving_file = open(file+'_stats_attr', "rb")
        dict_attr = pickle.load(saving_file)
        saving_file.close()
        self.n_episodes = dict_attr['n_episodes']
        self.T_f = dict_attr['T_f']
        self.d = dict_attr['d']
        self.list_n_events = dict_attr['list_n_events']
        self.list_eta = dict_attr['list_eta']
        self.average_n_events = dict_attr['average_n_events']
        self.total_n_events = dict_attr['total_n_events']
        self.average_eta = dict_attr['average_eta']
        list_paths = [None]*self.n_episodes
        for ix in range(self.n_episodes):
            process_path = ProcessPath()
            process_path.load(file+'_path_'+str(ix))
            list_paths[ix] = process_path
        self.list_paths = list_paths
