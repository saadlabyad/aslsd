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


def test_kappa(list_times, matrix_kappa, m_max=None):
    d = len(list_times)

    for i, j in itertools.product(range(d), range(d)):
        if (m_max is not None) and (m_max < len(list_times[i])):
            stop_index = m_max
        else:
            stop_index = len(list_times[i])

        for m in tqdm(range(stop_index)):
            kappa_m = matrix_kappa[j][i][m]
            if kappa_m > -1 and kappa_m <= len(list_times[j])-2:
                t_im = list_times[i][m]
                if not ((list_times[j][kappa_m] < t_im)
                        & (list_times[j][kappa_m+1] >= t_im)):
                    return False
    return True


def test_varkappa(list_times, matrix_varpi, h_max=None):
    d = len(list_times)

    for i, j in itertools.product(range(d), range(d)):
        if ((h_max is not None) and (h_max >= 1)
                and (h_max < len(matrix_varpi[i][j]))):
            stop_index = h_max
        else:
            stop_index = len(matrix_varpi[i][j])

        for h in tqdm(range(1, stop_index)):
            varpi_h = matrix_varpi[i][j][h]
            t_jn = list_times[j][h-1]
            if varpi_h >= 1:
                if not ((list_times[i][varpi_h] > t_jn)
                        & (list_times[i][varpi_h-1] <= t_jn)):
                    return False
    return True

# # matrix_kappa[j][i][10]=1 #    Introduce an error for sanity check
# test_kappa(matrix_kappa,list_times,i,j,m_max=None)


# # matrix_varkappa[i][j][10]=1 #    Introduce an error for sanity check
# test_varkappa(matrix_varkappa,matrix_kappa,list_times,i,j,m_max=None)

if __name__ == '__main__':
    unittest.main()
