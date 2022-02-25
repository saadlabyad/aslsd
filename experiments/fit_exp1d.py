# License: BSD 3 clause

"""
Fit Exp1D

In this example, we simulate a unidimensional (ground truth) MHP with
exponential kernel.
We estimate the parameters of this MHP using ASLSD.
"""
import os
import sys
# add the path of packages to system path
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import matplotlib.pyplot as plt
import numpy as np

from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP

# Define a ground truth MHP
true_kernel = KernelModel([ExponentialKernel()])
true_mhp = MHP([[true_kernel]])

# Define true parameter values
true_mu = np.array([1.5])
true_omega = 0.2
true_beta = 1.
true_ker_param = [[np.array([true_omega, true_beta])]]

# Simulate a path of the ground truth
T_f = 10**6
list_times = true_mhp.simulate(T_f, mu=true_mu, kernel_param=true_ker_param,
                               seed=1234, verbose=True)

# Define and fit a model
kernel = KernelModel(ExponentialKernel())
mhp = MHP([[kernel]])
kwargs = {'is_log_param': True, 'is_log_grad': True}
mhp.fit(list_times, T_f, n_iter=1000, seed=12, verbose=True, **kwargs)
fit_log = mhp.fit_log

# Visualize results
fig = plt.figure(dpi=300)
t = np.linspace(0, 10, 1000)
true_y = true_mhp.phi[0][0](t, true_ker_param[0][0])
plt.plot(t, true_y, color='darkorange', label='Ground truth')
y = mhp.phi[0][0](t, mhp.fitted_ker_param[0][0])
plt.plot(t, y, color='steelblue', label='ASLSD')
plt.legend()
fig.show()

mhp.plot_solver_path(true_mu=true_mu, true_ker_param=true_ker_param,
                     plot_derivatives=True, derivatives_zero=True)
