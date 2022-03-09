# License: BSD 3 clause

"""
Reproduce the results from the epidemic propagation application in the
"Applications" section of our paper.

In this application, we model the propagation of malaria in China.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aslsd.basis_kernels.basis_kernel_gaussian import GaussianKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP
from aslsd.models.hom_poisson import HomPoisson
from aslsd.events.process_path import ProcessPath

data_filepath = "data/malaria_china_data.csv"


# Data preprocessing
df = pd.read_csv(data_filepath)
times = df[df['time_jitter'] > 0.]['time_jitter'].values
T_f = times[-1]+(times[-1]-times[-2])
list_times = [times]
process_path = ProcessPath(list_times, T_f)
d = len(list_times)


# Model definition
model_dict = {}
colors_dict = {}
# Poisson
model_dict['Poisson'] = HomPoisson(1)
colors_dict['Poisson'] = 'steelblue'
# Gaussian 1R
kernel_g1d1r = KernelModel(GaussianKernel())
mhp_g1d1r = MHP([[kernel_g1d1r]])
model_dict['Gauss1D1R'] = mhp_g1d1r
colors_dict['Gauss1D1R'] = 'darkorange'
# Gaussian 10R
fixed_betas_g1d10r = np.ones(10)*1.9
fixed_deltas_g1d10r = np.linspace(0., 20., 10)

basis_kernels_g1d10r = [GaussianKernel(fixed_indices=[1, 2],
                                       fixed_vars=[fixed_betas_g1d10r[ix_bk],
                                                   fixed_deltas_g1d10r[ix_bk]])
                        for ix_bk in range(10)]
kernel_g1d10r = KernelModel(basis_kernels_g1d10r)
mhp_g1d10r = MHP([[kernel_g1d10r]])
model_dict['SbfGauss1D10R'] = mhp_g1d10r
colors_dict['SbfGauss1D10R'] = 'darkgreen'


# Fitting models
for key in model_dict.keys():
    if key != 'Poisson':
        kwargs = {'is_log_param': True, 'is_log_grad': True}
        model_dict[key].fit(list_times, T_f, n_iter=1000, seed=12,
                            verbose=True, **kwargs)
    else:
        model_dict[key].fit(list_times, T_f)


# Visualize results
t_min = 0.
t_max = 10.
n_samples = 10**3
fig = plt.figure(dpi=300)
x_phi = np.linspace(t_min, t_max, n_samples)
for key in model_dict.keys():
    if key != 'Poisson':
        ker_param = model_dict[key].fitted_ker_param
        y_phi = model_dict[key].phi[0][0](x_phi, ker_param[0][0])
        plt.plot(x_phi, y_phi, label=key, color=colors_dict[key])
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\phi(t)$')
plt.title('Kernel plot')
fig.show()


# Goodness-of-fit
for key in model_dict.keys():
    if key != 'Poisson':
        model_dict[key].get_residuals(process_path)
    else:
        model_dict[key].get_residuals(list_times)

fig, axs = plt.subplots(1, 2, sharex=False, sharey=False, dpi=300)
for key in model_dict.keys():
    kwargs = {'label': key, 'color': colors_dict[key]}
    axs[0] = model_dict[key].qq_plot(0, ax=axs[0], **kwargs)
    kwargs = {'color': colors_dict[key]}
    axs[1] = model_dict[key].qq_plot(0, ax=axs[1], style='uniform',
                                     substract_yx=True, **kwargs)
axs[0].set_title('QQ-plot')
axs[1].set_title('Transformed probability plot')
fig.legend(loc='center left')
fig.tight_layout()
fig.suptitle('Model residuals', y=1.1)
fig.show()
