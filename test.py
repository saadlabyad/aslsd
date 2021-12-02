# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:08:07 2021

@author: saadl
"""
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aslsd.basis_kernels.basis_kernel_gaussian import GaussianKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP


g = GaussianKernel()
kernel = KernelModel(g)
mhp = MHP([[kernel]])

# Define true parameter values
mu = np.array([1.5])
omega = 0.5
beta = 0.5
delta = 3.
kernel_param = [[np.array([omega, beta, delta])]]


mhp.plot_kernels(mu=mu, kernel_param=kernel_param)


def get_times(n_execs, func_basis, func_ker, func_mhp, *args):
    exec_times = {}
    exec_times['basis'] = np.zeros(n_execs)
    exec_times['ker'] = np.zeros(n_execs)
    exec_times['mhp'] = np.zeros(n_execs)
    for x in range(n_execs):
        start = timer()
        z = func_basis(*args)
        end = timer()
        exec_times['basis'][x] = end-start

        start = timer()
        z = func_ker(*args)
        end = timer()
        exec_times['ker'][x] = end-start

        start = timer()
        z = func_mhp(*args)
        end = timer()
        exec_times['mhp'][x] = end-start
    for key in exec_times.keys():
        print(np.mean(exec_times[key])*10**6, u"\u00B1", np.std(exec_times[key])*10**6, 'microseconds for ', str(key))


func_basis = g.diff_psi
func_ker = kernel.diff_psi
func_mhp = mhp.diff_psi[0][0]
args = [0., 1, kernel_param[0][0]]
get_times(10**6, func_basis, func_ker, func_mhp, *args)
# fig = plt.figure(dpi=300)
# t = np.linspace(0, 10, 1000)
# true_y = true_mhp.phi[0][0](t, true_kernel_param[0][0])
# plt.plot(t, true_y, color='darkorange', label='Ground truth')
# y = mhp.phi[0][0](t, mhp.fitted_ker_param[0][0])
# plt.plot(t, y, color='steelblue', label='ASLSD')
# plt.legend()
# fig.show()

# # Time it

