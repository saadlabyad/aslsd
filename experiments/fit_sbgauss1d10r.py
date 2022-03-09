# License: BSD 3 clause

"""
In this example, we simulate a unidimensional (ground truth) MHP with a
multimodal Gaussian kernel with three modes.

We estimate the parameters of this MHP using ASLSD, with a SBF Gaussian model
with ten modes.
"""

import os
import sys
# add the path of packages to system path
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aslsd.basis_kernels.basis_kernel_gaussian import GaussianKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP

# Define a ground truth MHP
true_kernel = KernelModel([GaussianKernel(), GaussianKernel(),
                           GaussianKernel()])
true_mhp = MHP([[true_kernel]])

# Define true parameter values
true_mu = np.array([0.01])
true_omega = np.array([0.2, 0.3, 0.4])
true_beta = np.array([0.4, 0.6, 0.8])
true_delta = np.array([1., 3., 8.])

true_ker_param = [[np.zeros(9)]]
for ix in range(3):
    true_ker_param[0][0][3*ix] = true_omega[ix]
    true_ker_param[0][0][3*ix+1] = true_beta[ix]
    true_ker_param[0][0][3*ix+2] = true_delta[ix]


# Simulate a path of the ground truth
T_f = 10**7
list_times = true_mhp.simulate(T_f, mu=true_mu, kernel_param=true_ker_param,
                               seed=1234, verbose=True)

# Visualize simulated data
inter_arrivals = pd.Series(list_times[0][1:]-list_times[0][:-1])
fig = plt.figure(dpi=300)
inter_arrivals[inter_arrivals < 10].hist(bins=200)
for z in true_delta:
    plt.axvline(x=z, color='firebrick')
plt.title('Histogram of inter-arrival times')
fig.show()

# Define a model
basis_kernels = [GaussianKernel(fixed_indices=[1, 2],
                                fixed_vars=[0.5, float(ix_bk)])
                 for ix_bk in range(10)]
kernel = KernelModel(basis_kernels)
mhp = MHP([[kernel]])

# Fit the model
kwargs = {'is_log_param': True, 'is_log_grad': True}
mhp.fit(list_times, T_f, n_iter=1000, seed=12, verbose=True, **kwargs)
fit_log = mhp.fit_log

# L2 projection of the model
res = mhp.get_l2_projection(true_mhp, true_ker_param, n_iter=1000,
                            log_error=True, verbose=True)
l2_projection_params = res['params']


# Visualize results
fig = plt.figure(dpi=300)
t = np.linspace(0, 10, 1000)
true_y = true_kernel.phi(t, true_ker_param[0][0])
plt.plot(t, true_y, color='darkorange', label='Ground truth')
y = mhp.phi[0][0](t, mhp.fitted_ker_param[0][0])
plt.plot(t, y, color='steelblue', label='ASLSD')
l2_proj_y = mhp.phi[0][0](t, l2_projection_params[0][0])
plt.plot(t, l2_proj_y, color='firebrick', linestyle='dashed',
         label=r'$L_2$ projection')
plt.legend()
fig.show()

mhp.plot_solver_path(true_mu=true_mu, true_ker_param=true_ker_param,
                     plot_derivatives=True, derivatives_zero=True)
