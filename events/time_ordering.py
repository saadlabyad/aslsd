# License: BSD 3 clause

import bisect
import itertools

import numpy as np


def get_kappa_varpi(list_times, T_f):
    """
    Compute the functions :math:`\\kappa` and :math:`\\varpi`, needed for
    chronological ordering of event times across dimensions, and return them as
    arrays.

    * :math:`\\kappa(j,i,m)` is the index of the latest event of type :math:`j`
    preceding event time :math:`t_m^i`.

    * :math:`\\varpi(i,j,h)` is the smallest index :math:`m` such that there
    exists at least h events of type :math:`j` preceding event time
    :math:`t_m^i`.

    Parameters
    ----------
    list_times : `list` of np.ndarray
        list_times[i] is the sorted array of events of type :math:`i`.

    T_f : `float`
        Terminal time.

    Returns
    -------
    kappa : `list` of `list` of `numpy.ndarray`
        Description.

    varpi : `list` of `list` of `numpy.ndarray`
        Description.

    Raises
    ------
    IndexError
        If `T_f` is smaller than `list_times[i][-1]` for some `i`.

    """

    d = len(list_times)
    sizes = [len(list_times[i]) for i in range(d)]

    kappa = [[[] for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        if i == j:
            kappa[i][j] = np.arange(sizes[i]+1, dtype=np.int64)-1
        else:
            kappa[i][j] = [-1 for n in range(sizes[j]+1)]
            old_index = 0
            new_index = 0
            for n in range(sizes[j]):
                old_index = max(new_index, 0)
                new_index = bisect.bisect_left(list_times[i], list_times[j][n],
                                               lo=old_index, hi=sizes[i])-1
                kappa[i][j][n] = new_index
            kappa[i][j][-1] = bisect.bisect_left(list_times[i], T_f, lo=0,
                                                 hi=sizes[i]-1)
            kappa[i][j] = np.array(kappa[i][j], dtype=np.int64)

    varpi = [[[] for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        if i == j:
            varpi[i][j] = np.arange(sizes[i], dtype=np.int64)
        if i != j:
            varpi[i][j] = [0 for h in range(kappa[j][i][-2]+2)]
            old_index = 0
            new_index = -1
            lo_val = 0
            for h in range(1, kappa[j][i][-2]+2):
                lo_val = max(0, new_index)
                new_index = bisect.bisect_left(kappa[j][i], h-1, lo=lo_val,
                                               hi=len(kappa[j][i]))
                varpi[i][j][h] = new_index
            varpi[i][j] = np.array(varpi[i][j], dtype=np.int64)

    return kappa, varpi


def get_lag_sizes(list_times, kappa, varpi):
    """
    Compute the number of event of type :math:`i` with at least :math:`h`
    predecessors of type :math:`j`, for all :math:`i,j \\in [d]`,
    :math:`h \\in [\\kappa(j,i,N^i_T)]`.

    Parameters
    ----------
    list_times : `list` of np.ndarray
        list_times[i] is the sorted array of events of type :math:`i`.

    kappa : `list` of `list` of `numpy.ndarray`
        DESCRIPTION.

    varpi : `list` of `list` of `numpy.ndarray`
        DESCRIPTION.

    Returns
    -------
    lag_sizes : `list` of `list` of np.ndarray
        lag_sizes[i][j][h] is the number of events of type :math:`i` with at
        least :math:`h` predecessors of type :math:`j`.
        By convention, matrix_lag_sizes[i][j][0]=0.

    """

    d = len(list_times)
    lag_sizes = [[[None]*(kappa[j][i][len(list_times[i])-1]+2)
                  for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        lag_sizes[i][j][0] = 0
        lag_sizes[i][j][1: kappa[j][i][len(list_times[i])-1]+2] = len(list_times[i])-varpi[i][j][1: kappa[j][i][len(list_times[i])-1]+2]
    return lag_sizes
