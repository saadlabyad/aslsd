# License: BSD 3 clause

"""
Reproduce the results from the news propagation application in the
"Applications" section of our paper.

In this application, we are interested in the diffusion of information across
different media platforms.

In this first example, we aggregate all publication timestamps into a path of
a one-dimensional point process.
"""

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from aslsd.applications.memetracker.preprocessing import memetracker_preprocessing as mtp
from aslsd.applications.memetracker.preprocessing import memetracker_webdicts as mweb
from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.events.process_path import ProcessPath
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP
from aslsd.models.hom_poisson import HomPoisson

# Data preprocessing
memes_filepath = "data/memetracker_royalwedding_data.txt"
break_ind = 10**5
web_dict, cascades_list = mtp.read_cascade(memes_filepath, break_ind)

countries_filepath = "data/memetracker_webdict.csv"
final_webdict = mweb.load_countrywise_webdict(countries_filepath)
# mcp.webdict2csv(web_dict, keyword, title=None)
times, web_ids = mtp.flatten_cascades(cascades_list, mean=0.0, std=0.,
                                      base_seed=1234, discard_collisions=True)
web_countries = [final_webdict[site]['country'] for site in web_ids]

min_date = datetime.utcfromtimestamp(times[0]*3600).strftime('%Y-%m-%d %H:%M:%S')
max_date = datetime.utcfromtimestamp(times[-1]*3600).strftime('%Y-%m-%d %H:%M:%S')
print('Min date in dataset', min_date)
print('Max date in dataset', max_date)

rescale_factor = 1.
date_rec_min = datetime(2011, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
t_rec_min = date_rec_min.timestamp()/3600.
date_rec_max = datetime(2012, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
t_rec_max = date_rec_max.timestamp()/3600.
times = mtp.rescale_times(times, rescale_factor, t_rec_min, t_rec_max)

list_times = [times]
print('Number of events after filtering: ', len(times))
T_f = t_rec_max-t_rec_min
d = len(list_times)
process_path = ProcessPath(list_times, T_f)


# Model definition
model_dict = {}
colors_dict = {}
# Poisson
model_dict['Poisson'] = HomPoisson(1)
colors_dict['Poisson'] = 'steelblue'
# Exponential 1D1R
kernel_exp1d1r = KernelModel(ExponentialKernel())
mhp_exp1d1r = MHP([[kernel_exp1d1r]])
model_dict['Exp1D1R'] = mhp_exp1d1r
colors_dict['Exp1D1R'] = 'darkorange'
# Exponential 1D6R
basis_kernels_exp1d6r = [ExponentialKernel() for ix_bk in range(6)]
kernel_exp1d6r = KernelModel(basis_kernels_exp1d6r)
mhp_exp1d6r = MHP([[kernel_exp1d6r]])
model_dict['Exp1D6R'] = mhp_exp1d6r
colors_dict['Exp1D6R'] = 'darkgreen'


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
        true_ker_param = model_dict[key].fitted_ker_param
        y_phi = model_dict[key].phi[0][0](x_phi, true_ker_param[0][0])
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
