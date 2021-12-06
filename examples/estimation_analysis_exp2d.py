"""
=========================================
Gradient estimation analysis
=========================================
In this example, we simulate a bidimensional MHP with exponential kernels.
We compare the distribution of LSE gradient estimations based on ASLSD
compared with the exact values of the gradient.
We compute the exact gradient in this case using the recurrence formulae
for exponential MHP.
"""

import itertools
import os
import sys
# add the path of packages to system path
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP
from aslsd.estimators.adaptive_stratified_estimator import AdaptiveStratified
from aslsd.events.process_path import ProcessPath
from aslsd.exponential_mhp.exp_hawkes import RecurrentExponential

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
process_path = ProcessPath(list_times, T_f)

# Define a model
kernel_matrix = [[KernelModel([ExponentialKernel()]) for j in range(d)]
                 for i in range(d)]
mhp = MHP(kernel_matrix)

# Define estimators
estimators = [AdaptiveStratified() for k in range(d)]
for k in range(d):
    estimators[k].k = k
    estimators[k].n_iter = 1000
    estimators[k].initialize(process_path, mhp)
    estimators[k].intialize_logs()
    estimators[k].set_stratification()

# Define parameter values x for which we want to analyse the gradient
x = [np.array([true_mu[k], true_omega[k][0], true_beta[k][0], true_omega[k][1],
               true_beta[k][1]]) for k in range(d)]

# Compute the exact gradient in x
recurrent_exp = RecurrentExponential(mhp)
exact_grad = [recurrent_exp.lse_k_grad(k, x[k], T_f, list_times,
                                       process_path.list_times2end,
                                       process_path.n_events,
                                       process_path.varpi,
                                       process_path.kappa,
                                       process_path.eta) for k in range(d)]
# Compute gradient estimates in x
N_samples = 500
seed = 1234
rng = np.random.default_rng(seed)
list_grads = [[None for x in range(N_samples)] for k in range(d)]
for k in range(d):
    for n in tqdm(range(N_samples)):
        grad_n = estimators[k].lse_k_grad_estimate(x[k], rng=rng)
        list_grads[k][n] = grad_n

# Visualisations
list_grads_mu = [pd.Series([y[0] for y in list_grads[k]]) for k in range(d)]
exact_grad_mu = [exact_grad[k] for k in range(d)]
for k in range(d):
    fig = plt.figure(dpi=300)
    list_grads_mu[k].hist(bins=100)
    plt.axvline(exact_grad_mu[k], color='red', linestyle='dashed')
    plt.title(r'$\mu_'+str(k)+'$ Derivative')
    fig.show()

list_grads_omega = [[pd.Series([x[1+2*j] for x in list_grads[i]])
                     for j in range(d)] for i in range(d)]
exact_grad_omega = [[exact_grad[i][1+2*j] for j in range(d)] for i in range(d)]
for i, j in itertools.product(range(d), range(d)):
    fig = plt.figure(dpi=300)
    list_grads_omega[i][j].hist(bins=100)
    plt.axvline(list_grads_omega[i][j].mean(), color='grey',
                linestyle='dashed')
    plt.axvline(exact_grad_omega[i][j], color='red', linestyle='dashed')
    plt.title(r'$\omega_{'+str(i)+','+str(j)+'}$ derivative')
    fig.show()

list_grads_beta = [[pd.Series([x[1+2*j+1] for x in list_grads[i]])
                    for j in range(d)] for i in range(d)]
exact_beta_der = [[exact_grad[i][1+2*j+1] for j in range(d)] for i in range(d)]
for i, j in itertools.product(range(d), range(d)):
    fig = plt.figure(dpi=300)
    list_grads_beta[i][j].hist(bins=100)
    plt.axvline(list_grads_beta[i][j].mean(), color='grey', linestyle='dashed')
    plt.axvline(exact_beta_der[i][j], color='red', linestyle='dashed')
    plt.title(r'$\beta_{'+str(i)+','+str(j)+'}$ derivative')
    fig.show()
