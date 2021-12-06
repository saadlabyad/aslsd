"""
=========================================
Fit Exp2D
=========================================
In this example, we simulate a groudn truth bidimensional MHP with exponential
kernels.
We estimate the parameters of this MHP using ASLSD, plot the path of the
solver, the fitted kernels and the adjacency matrix of the fit.
"""

import itertools
import os
import sys
# add the path of packages to system path
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import numpy as np

from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP

# Define a ground truth MHP
d = 2
true_kernel_matrix = [[KernelModel([ExponentialKernel()]) for j in range(d)]
                      for i in range(d)]
true_mhp = MHP(true_kernel_matrix)

# Define true parameter values
true_mu = np.array([1.5, 1.])

true_omega = np.ones((2, 2))
true_omega[0][0] = .2
true_omega[0][1] = .6
true_omega[1][0] = .7
true_omega[1][1] = .1

true_beta = np.ones((2, 2))
true_beta[0][0] = 1.
true_beta[0][1] = 1.5
true_beta[1][0] = 2.
true_beta[1][1] = 1.3

true_ker_param = np.ones((d, d, 2))
for i, j in itertools.product(range(d), range(d)):
    true_ker_param[i][j][0] = true_omega[i][j]
    true_ker_param[i][j][1] = true_beta[i][j]

# Simulate a path of the ground truth
T_f = 10**5
list_times = true_mhp.simulate(T_f, mu=true_mu, kernel_param=true_ker_param,
                               seed=1234, verbose=True)

# Define and fit a model
kernel_matrix = [[KernelModel([ExponentialKernel()]) for j in range(d)]
                 for i in range(d)]
mhp = MHP(kernel_matrix)
kwargs = {'is_log_param': True, 'is_log_grad': True,
          'learning_rate_type': 'exponential'}
mhp.fit(list_times, T_f, n_iter=1000, seed=1234, verbose=True, **kwargs)
fit_log = mhp.fit_log

# Visualize results
mhp.plot_solver_path(true_mu=true_mu, true_ker_param=true_ker_param)
mhp.plot_kernels()
mhp.plot_adjacency_matrix()
