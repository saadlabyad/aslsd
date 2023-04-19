# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:08:07 2021

@author: saadl
"""
import inspect
import itertools
import os
import sys
import unittest

import numpy as np
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import time_ordering


def test_kappa(list_times, kappa, m_max=None, verbose=False):
    d = len(list_times)

    for i, j in itertools.product(range(d), range(d)):
        if (m_max is not None) and (m_max < len(list_times[i])):
            stop_index = m_max
        else:
            stop_index = len(list_times[i])

        for m in tqdm(range(stop_index), disable=not verbose):
            kappa_m = kappa[j][i][m]
            if kappa_m > -1 and kappa_m <= len(list_times[j])-2:
                t_im = list_times[i][m]
                if not ((list_times[j][kappa_m] < t_im)
                        & (list_times[j][kappa_m+1] >= t_im)):
                    return False
    return True


def test_varpi(list_times, varpi, h_max=None, verbose=False):
    d = len(list_times)

    for i, j in itertools.product(range(d), range(d)):
        if ((h_max is not None) and (h_max >= 1)
                and (h_max < len(varpi[i][j]))):
            stop_index = h_max
        else:
            stop_index = len(varpi[i][j])

        for h in tqdm(range(1, stop_index), disable=not verbose):
            varpi_h = varpi[i][j][h]
            t_jn = list_times[j][h-1]
            if varpi_h >= 1:
                if not ((list_times[i][varpi_h] > t_jn)
                        & (list_times[i][varpi_h-1] <= t_jn)):
                    return False
            else:
                print('i, j: ', i, j)
                print('h', h)
                print('list_times[i][0]', list_times[i][0])
                print('t_jn', t_jn)
                if list_times[i][0] <= t_jn:
                    return False
    return True


# list_times = [[1., 2., 4. ,5], [0., 0.5, 2.5]]
list_times = [None]*2
list_times[0] = np.array([1.75869445, 1.80061029, 1.82564278, 1.91613725,
                          1.92903655, 1.9636614, 2.00255562, 2.0473686,
                          2.13208584, 2.15969973])
list_times[1] = np.array([1.72669995, 1.75597784, 1.7805469, 1.78599852,
                          1.79899688, 1.79990949, 1.8040088, 2.02449236,
                          2.04648297, 2.18191544])
T_f = 6.
kappa, varpi = time_ordering.get_kappa_varpi(list_times, T_f)

test_varpi(list_times, varpi)
# # matrix_kappa[j][i][10]=1 #    Introduce an error for sanity check
# test_kappa(matrix_kappa,list_times,i,j,m_max=None)


# # matrix_varpi[i][j][10]=1 #    Introduce an error for sanity check
# test_varkappa(matrix_varkappa,matrix_kappa,list_times,i,j,m_max=None)

if __name__ == '__main__':
    unittest.main()
