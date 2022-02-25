# License: BSD 3 clause

from datetime import datetime, timezone
import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aslsd.applications.memetracker.preprocessing import memetracker_preprocessing as mtp
from aslsd.applications.memetracker.preprocessing import memetracker_webdicts as mweb
from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.events.process_path import ProcessPath
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP

# Data preprocessing
memes_filepath = "data/memetracker_royalwedding_data.txt"
break_ind = 10**6
web_dict, cascades_list = mtp.read_cascade(memes_filepath, break_ind)
countries_filepath = "data/memetracker_webdict.csv"
final_webdict = mweb.load_countrywise_webdict(countries_filepath)
times, web_ids = mtp.flatten_cascades(cascades_list, mean=0.0, std=0.,
                                      base_seed=1234, discard_collisions=True)
web_countries = [final_webdict[site]['country'] for site in web_ids]
for i in range(len(web_countries)):
    if web_countries[i] == ' ':
        web_countries[i] = 'Unknown'
df = pd.DataFrame.from_dict({'time': times, 'id': web_ids,
                             'country': web_countries})
df['country'].value_counts()
df['country'].value_counts().plot(kind='bar')
times_us = np.array(df.loc[df['country'] == 'United States of America']['time'])
times_uk = np.array(df.loc[df['country'] == 'United Kingdom']['time'])
list_times = [times_us, times_uk]
d = len(list_times)
min_data_date=datetime.utcfromtimestamp(times[0]*3600).strftime('%Y-%m-%d %H:%M:%S')
max_data_date=datetime.utcfromtimestamp(times[-1]*3600).strftime('%Y-%m-%d %H:%M:%S')
print('Min data date', min_data_date)
print('Max data date', max_data_date)
rescale_factor = 1.
date_rec_min = datetime(2011, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
t_rec_min = date_rec_min.timestamp()/3600.
date_rec_max = datetime(2012, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
t_rec_max = date_rec_max.timestamp()/3600.
for i in range(d):
    list_times[i] = mtp.rescale_times(list_times[i], rescale_factor, t_rec_min,
                                      t_rec_max)
T_f = t_rec_max-t_rec_min
process_path = ProcessPath(list_times, T_f)


# Model definition
model_dict = {}
colors_dict = {}
# Exponential
kernel_matrix_exp2d1r = [[KernelModel([ExponentialKernel()]) for j in range(d)]
                         for i in range(d)]
mhp_exp2d1r = MHP(kernel_matrix_exp2d1r)
model_dict['Exp2D1R'] = mhp_exp2d1r
colors_dict['Exp2D1R'] = 'steelblue'
# Exponential 2D3R
kernel_matrix_exp2d3r = [[KernelModel([ExponentialKernel()
                                       for ix in range(3)]) for j in range(d)]
                         for i in range(d)]
mhp_exp2d3r = MHP(kernel_matrix_exp2d3r)
model_dict['Exp2D3R'] = mhp_exp2d3r
colors_dict['Exp2D3R'] = 'darkorange'


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
fig, axs = plt.subplots(d, d, sharex=True, sharey=False, dpi=300)
x_phi = np.linspace(t_min, t_max, n_samples)
for key in model_dict.keys():
    for i, j in itertools.product(range(d), range(d)):
        ker_param = model_dict[key].fitted_ker_param
        y_phi = model_dict[key].phi[i][j](x_phi, ker_param[0][0])
        axs[i, j].plot(x_phi, y_phi, label=key, color=colors_dict[key])
plt.legend()
for i in range(d):
    for j in range(d):
        axs[i, j].set_ylabel(r'$\phi_{'+str(i)+','+str(j)+'}(t)$')
        axs[d-1, j].set_xlabel(r'$t$')
fig.suptitle('Kernel plot')
fig.tight_layout()
fig.show()


# Goodness-of-fit
for key in model_dict.keys():
    if key != 'Poisson':
        model_dict[key].get_residuals(process_path, verbose=True)
    else:
        model_dict[key].get_residuals(list_times)

fig, axs = plt.subplots(d, 2, sharex=False, sharey=False, dpi=300)
for i in range(d):
    for key in model_dict.keys():
        if i == 0:
            kwargs = {'label': key, 'color': colors_dict[key]}
        else:
            kwargs = {'color': colors_dict[key]}
        axs[i, 0] = model_dict[key].qq_plot(i, ax=axs[i, 0], **kwargs)
        kwargs = {'color': colors_dict[key]}
        axs[i, 1] = model_dict[key].qq_plot(i, ax=axs[i, 1], style='uniform',
                                            substract_yx=True, **kwargs)
axs[0, 0].set_title('QQ-plot')
axs[0, 1].set_title('Transformed probability plot')
fig.legend(loc='center right')
fig.tight_layout()
fig.suptitle('Model residuals', y=1.1)
