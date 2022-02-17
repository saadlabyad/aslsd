# License: BSD 3 clause

import os,sys,inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import tikzplotlib

import memetracker_cascades_preprocessing as mcp

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from time_structures import get_kappa_varkappa
import gof_residuals
import l1_norm
import func_compiler
import func_maker
from hawkes_sgd import fit_momentum_stratified,fit_momentum_adastrat,fit_adam_stratified, fit_adam_adastrat
import mhp_simulation
from model_specification import make_matrix_param, get_param_directory, get_mhp_param_names,make_bounds,make_index_converter, list2matrix_parameters, list2matrix_solverpaths

from func_maker import make_model
import exp_hawkes
import graphic_tools as gt
from postprocessing_estimation import fit_multipath, get_error_multipath_param
from tick_models import fit_tick_sumexp, fit_tick_wh
import error_analysis
from func_compiler import compile_phi
import model_generation
import useful_functions as uf
from l2_projection_sbf_kernel import get_l2_projection_sbf_kernel

# =============================================================================
# Data preprocessing
# =============================================================================
# keyword='prince william-william-kate middleton-kate-middleton-westminster-watch-marriage-queen-king-elizabeth-charles'
keyword='royal-wedding'
break_ind=10**5
web_dict, cascades_list=mcp.read_cascade(keyword,break_ind)
final_webdict=mcp.load_countrywise_webdict(keyword) 
# mcp.webdict2csv(web_dict,keyword,title=None)

# fig=plt.figure(dpi=300)
# plt.plot([c['cascade size'] for c in cascades_list],color='steelblue')
# plt.axhline(y=np.mean(np.array([c['cascade size'] for c in cascades_list])), color='firebrick', linestyle='-')
# plt.xlabel('Cascade Index')
# plt.ylabel('# Events in the cascade')
# plt.title(keyword+' keyword')
# fig.show()

# fig=plt.figure(dpi=300)
# plt.plot([c['cascade times'][0] for c in cascades_list],color='steelblue')
# # plt.axhline(y=np.mean(np.array([c['cascade size'] for c in cascades_list])), color='firebrick', linestyle='-')
# plt.xlabel('Cascade Index')
# plt.ylabel('Time of first event in the cascade')
# plt.title(keyword+' keyword')
# fig.show()

# =============================================================================
# Aggregate event cascades
# =============================================================================
# times=mcp.flatten_cascades(cascades_list,mean=0.0, std=10**-5,base_seed=1234)
times,web_ids=mcp.flatten_cascades(cascades_list,mean=0.0, std=0.,base_seed=1234,discard_collisions=True)
web_countries=[final_webdict[site]['country'] for site in web_ids]
# =============================================================================
# Windsorize
# =============================================================================
min_data_date=datetime.utcfromtimestamp(times[0]*3600).strftime('%Y-%m-%d %H:%M:%S')
max_data_date=datetime.utcfromtimestamp(times[-1]*3600).strftime('%Y-%m-%d %H:%M:%S')
print('Min data date',min_data_date)
print('Max data date',max_data_date)

rescale_factor=1.

# date_rec_min=datetime(2011,7,1,0,0,0, tzinfo=timezone.utc)
date_rec_min=datetime(2011,11,1,0,0,0, tzinfo=timezone.utc)
t_rec_min=date_rec_min.timestamp()/3600.

date_rec_max=datetime(2012,3,1,0,0,0, tzinfo=timezone.utc)
t_rec_max=date_rec_max.timestamp()/3600.

times=mcp.rescale_times(times,rescale_factor,t_rec_min,t_rec_max)

# =============================================================================
# Visualization
# =============================================================================
# fig=plt.figure(dpi=300)

# plt.plot(times,[i for i in range(len(times))],color='steelblue')
# plt.plot([0,times[-1]],[0,times[-1]],color='red')
# # t_special_event_1=datetime(2011,7,23,13,0,0, tzinfo=timezone.utc).timestamp()-t_rec_min
# # plt.axvline(x=t_special_event_1, color='firebrick', linestyle='-')

# plt.xlabel('Publication time')
# plt.ylabel('Cumulative # of pubications')
# plt.title(keyword+' keyword')
# fig.show()

# print('# of events before Spec. Ev. :',len(times[(times<=t_special_event_1)]))
# print('# of events after Spec. Ev. :',len(times[(times>=t_special_event_1)]))

# =============================================================================
# Final data
# =============================================================================
tensor_event_times=[times]
print('Number of events after filtering: ',len(times))

T_f=t_rec_max-t_rec_min
d=len(tensor_event_times)
list_event_counts=[len(tensor_event_times[i]) for i in range(d)]
matrix_kappa,matrix_varkappa =get_kappa_varkappa(tensor_event_times,T_f)
eta=[len(L)/T_f for L in tensor_event_times]

# =============================================================================
# Poisson
# =============================================================================
print('Poisson')
list_r_hompoisson=[1]
fixed_param_hompoisson=np.array([[1.]],dtype=object)
list_ker_types_hompoisson=['SB Exponential']

matrix_model_hompoisson=[[{'Type':'SB Exponential','r':1,'Fixed parameters':fixed_param_hompoisson} for j in range(d)] for i in range(d)]

mu_hompoisson=np.array([len(tensor_event_times[i])/T_f for i in range(d)])

kernel_param_hompoisson=np.zeros( (d,d,1) )

phi_hompoisson,diff_phi_hompoisson,psi_hompoisson,diff_psi_hompoisson,upsilon_hompoisson,diff_sim_upsilon_hompoisson,diff_cross_upsilon_hompoisson=make_model(matrix_model_hompoisson)
matrix_phi_hompoisson=func_compiler.compile_phi(phi_hompoisson,kernel_param_hompoisson,d)

#   Resiudal analysis
#       Compute residuals
psi_hompoisson=func_maker.make_psi(matrix_model_hompoisson)
matrix_psi_hompoisson=func_compiler.compile_psi(psi_hompoisson,kernel_param_hompoisson,d)

residuals_hompoisson=gof_residuals.get_residuals_homogeneouspoisson(d,mu_hompoisson,tensor_event_times)
# #       KS test
# gof_residuals.ks_test_residuals_k(residuals_hompoisson[0])
# #       QQ plot
# gof_residuals.qq_plot_residuals_k(residuals_hompoisson[0],proba=False,log_scale=False,reference='uniform')
# #       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_hompoisson[0],color='steelblue')
# fig.show()


# =============================================================================
# Fit WH
# =============================================================================
print('WH')
max_support=10
mu_wh, matrix_phi_wh=fit_tick_wh(tensor_event_times,claw_method="lin", delta_lag=1., min_lag=5e-4, max_lag=100, quad_method="gauss", n_quad=10**3,min_support=1e-4, max_support=max_support, n_threads=4)

fig=plt.figure(dpi=300)
x_phi_wh=np.linspace(0,max_support,10**4)
y_phi_wh=matrix_phi_wh[0][0](x_phi_wh)
plt.plot(x_phi_wh,y_phi_wh)
fig.show()

# print('L1 norm',l1_norm.get_l1_norm_nonparam(matrix_phi_wh[0][0]))

#   Residual analysis
#       Compute residuals
matrix_psi_wh=func_compiler.compile_psi_numeric(matrix_phi_wh,d,x_max=50.,int_res=10**4)
residuals_wh=gof_residuals.get_residuals(d,mu_wh,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_wh,sampling=False,sample_size=10**3,seed=1234)
#       KS test
gof_residuals.ks_test_residuals_k(residuals_wh[0])

#       QQ plot
# gof_residuals.qq_plot_residuals_k(residuals_wh[0])

#       Plot residuals
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_wh[0],color='steelblue')
# # plt.xscale('log')
# plt.yscale('log')
# fig.show()

# =============================================================================
# Fit SumExp
# =============================================================================
print('SumExp')
beta_sumexp=np.array([0.5,1.,1.5,2.,5.,10.])

matrix_model_sumexp=model_generation.instantiate_matrix_model(d,kernel_type='SB Exponential')
model_generation.clone_fixed_param(matrix_model_sumexp,beta_sumexp,len(beta_sumexp),d)

mu_sumexp, kernel_param_sumexp=fit_tick_sumexp(tensor_event_times,beta_sumexp)
phi_sumexp=func_maker.make_phi(matrix_model_sumexp)

matrix_phi_sumexp=func_compiler.compile_phi(phi_sumexp,kernel_param_sumexp,d)

adj_matrix_sumexp=l1_norm.make_adjacency_matrix(matrix_model_sumexp,kernel_param_sumexp,d)
radius_sumexp=l1_norm.get_spectral_radius(adj_matrix_sumexp)


# fig=plt.figure(dpi=300)
# x=np.linspace(0,10**1,10**4)
# y=matrix_phi_sumexp[0][0](x)
# plt.plot(x,y)
# fig.show()


#   Resiudal test
#       Compute residuals
psi_sumexp=func_maker.make_psi(matrix_model_sumexp)
matrix_psi_sumexp=func_compiler.compile_psi(psi_sumexp,kernel_param_sumexp,d)

# fig=plt.figure(dpi=300)
# x=np.linspace(0,10**1,10**4)
# y=matrix_psi_sumexp[0][0](x)
# plt.plot(x,y)
# fig.show()

residuals_sumexp=gof_residuals.get_residuals(d,mu_sumexp,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_sumexp,sampling=False,sample_size=10**3,seed=1234)
#       KS test
# gof_residuals.ks_test_residuals_k(residuals_sumexp[0])

#       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_sumexp[0])

#       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_sumexp[0],color='steelblue')
# fig.show()
# =============================================================================
# aslsd - Exponential r=1
# =============================================================================
print('aslsd - Exponential r=1')
# Model Definition
matrix_model_aslsd_exp1=[[{'Type':'Exponential','r':1,'Fixed parameters':[]} for j in range(d)] for i in range(d)]
mu_names_aslsd_exp1,ker_names_aslsd_exp1=get_mhp_param_names(matrix_model_aslsd_exp1,d)

# Learning parameters
base_seed=1245
vec_learning_rate=[10**-1]
vec_epsilon=[10**-8]
vec_momentum_1=[0.9*10**0]
vec_momentum_2=[0.999*10**0]
vec_n_iter=[2000]

# Stratification config
upsilon_config=[]
phi_config=[]
psi_config=[]
upsilonzero_config=[]

# Single step adastrat
for k in range(d):
    # k_upsilon_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
    upsilon_config.append({'EMA weight':[[0.6]],'n_K':[[1]],'list_Delta_Q':[[ [3*10**4] ]],'bucket_size':[[1]],'h_min':[[1]],'h_max':[[30]], 'B_p_total':[[ 10**4]],'n_clusters': [[10]]})
    # k_phi_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
    phi_config.append({ 'EMA weight':[0.6],'n_K':[1],'list_Delta_Q':[ [3*10**4] ],'bucket_size':[1],'h_min':[1],'h_max':[30],  'B_p_total': [10**4],'n_clusters': [10]})
    # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
    psi_config.append({'B_spec':[10**4],'B_p_total':[1*10**4],'n_buckets':[10]})
    # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
    upsilonzero_config.append({'B_spec':[10**4],'B_p_total':[1*10**4],'n_buckets':[10]})
penalty_type=None
penalty_kwargs=None

# fitted_aslsd_exp1=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_exp1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
fitted_aslsd_exp1=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_exp1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=penalty_type,penalty_kwargs=penalty_kwargs,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=True)

mu_aslsd_exp1=fitted_aslsd_exp1['Mu']
kernel_param_aslsd_exp1=fitted_aslsd_exp1['Kernel']

# Make model for additionnal checks
matrix_param_aslsd_exp1=make_matrix_param(matrix_model_aslsd_exp1)
param_directory_aslsd_exp1=get_param_directory(d,matrix_param_aslsd_exp1)
phi_aslsd_exp1,diff_phi_aslsd_exp1,psi_aslsd_exp1,diff_psi_aslsd_exp1,upsilon_aslsd_exp1,diff_sim_upsilon_aslsd_exp1,diff_cross_upsilon_aslsd_exp1=make_model(matrix_model_aslsd_exp1)

matrix_phi_aslsd_exp1=func_compiler.compile_phi(phi_aslsd_exp1,kernel_param_aslsd_exp1,d)

adj_matrix_aslsd_exp1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_exp1,kernel_param_aslsd_exp1,d)
radius_aslsd_exp1=l1_norm.get_spectral_radius(adj_matrix_aslsd_exp1)

#   Resiudal analysis
#       Compute residuals
psi_aslsd_exp1=func_maker.make_psi(matrix_model_aslsd_exp1)
matrix_psi_aslsd_exp1=func_compiler.compile_psi(psi_aslsd_exp1,kernel_param_aslsd_exp1,d)
residuals_aslsd_exp1=gof_residuals.get_residuals(d,mu_aslsd_exp1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_exp1,sampling=False,sample_size=10**3,seed=1234)

#       KS test
gof_residuals.ks_test_residuals_k(residuals_aslsd_exp1[0])

#       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp1[0])

#       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_exp1[0],color='steelblue')
# plt.yscale('log')
# fig.show()


#   Fit log check
fit_log_aslsd_exp1=fitted_aslsd_exp1['Fit log']
fig, axs = plt.subplots(2, 2, sharex=False, sharey=False,dpi=300)
cumul_fig_count=0
#   Mu
axs[0, 0].plot(fit_log_aslsd_exp1['Updates']['Parameters']['Mu'][0], color=gt.standard_colors[0])
axs[0, 0].set(ylabel=r'$\mu$ updates')
axs[0, 0].xaxis.set_visible(False)
    
#   Omega
axs[0, 1].plot([x[0] for x in fit_log_aslsd_exp1['Updates']['Parameters']['Kernel'][0][0]], color=gt.standard_colors[0])
axs[0, 1].set(ylabel=r'$\omega$ updates')

#   Beta
axs[1, 0].plot([x[1] for x in fit_log_aslsd_exp1['Updates']['Parameters']['Kernel'][0][0]], color=gt.standard_colors[0])
axs[1, 0].set(ylabel=r'$\beta$ updates')

#   Delta
axs[1, 1].set_axis_off()

axs[1, 0].set(xlabel='Iteration')
axs[0, 1].set(xlabel='Iteration')
#   Try
fig.tight_layout()
# =============================================================================
# aslsd - Exponential r=3
# =============================================================================
# Model Definition
# matrix_model_aslsd_exp3=[[{'Type':'Exponential','r':3,'Fixed parameters':[]} for j in range(d)] for i in range(d)]
# mu_names_aslsd_exp3,ker_names_aslsd_exp3=get_mhp_param_names(matrix_model_aslsd_exp3,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[100 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[100] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[100],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[100],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_exp3=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_exp3,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_exp3=fitted_aslsd_exp3['Mu']
# kernel_param_aslsd_exp3=fitted_aslsd_exp3['Kernel']

# # Make model for additionnal checks
# matrix_param_aslsd_exp3=make_matrix_param(matrix_model_aslsd_exp3)
# param_directory_aslsd_exp3=get_param_directory(d,matrix_param_aslsd_exp3)
# phi_aslsd_exp3,diff_phi_aslsd_exp3,psi_aslsd_exp3,diff_psi_aslsd_exp3,upsilon_aslsd_exp3,diff_sim_upsilon_aslsd_exp3,diff_cross_upsilon_aslsd_exp3=make_model(matrix_model_aslsd_exp3)

# matrix_phi_aslsd_exp3=func_compiler.compile_phi(phi_aslsd_exp3,kernel_param_aslsd_exp3,d)

# adj_matrix_aslsd_exp3=l1_norm.make_adjacency_matrix(matrix_model_aslsd_exp3,kernel_param_aslsd_exp3,d)
# radius_aslsd_exp3=l1_norm.get_spectral_radius(adj_matrix_aslsd_exp3)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# # y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# # plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# # fig.show()


# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_exp3=func_maker.make_psi(matrix_model_aslsd_exp3)
# matrix_psi_aslsd_exp3=func_compiler.compile_psi(psi_aslsd_exp3,kernel_param_aslsd_exp3,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_exp3=gof_residuals.get_residuals(d,mu_aslsd_exp3,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_exp3)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_exp3[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp3[0])

# #       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_exp3[0],color='steelblue')
# fig.show()

# =============================================================================
# aslsd - Exponential r=6
# =============================================================================
print('aslsd - Exponential r=6')
# Model Definition
matrix_model_aslsd_exp6=[[{'Type':'Exponential','r':6,'Fixed parameters':[]} for j in range(d)] for i in range(d)]
mu_names_aslsd_exp6,ker_names_aslsd_exp6=get_mhp_param_names(matrix_model_aslsd_exp6,d)

# Learning parameters
base_seed=1245
vec_learning_rate=[10**-1]
vec_epsilon=[10**-8]
vec_momentum_1=[0.9*10**0]
vec_momentum_2=[0.999*10**0]
vec_n_iter=[2000]

# Stratification config
upsilon_config=[]
phi_config=[]
psi_config=[]
upsilonzero_config=[]

# Single step adastrat
for k in range(d):
    # k_upsilon_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
    upsilon_config.append({'EMA weight':[[0.6]],'n_K':[[1]],'list_Delta_Q':[[ [3*10**4] ]],'bucket_size':[[1]],'h_min':[[1]],'h_max':[[30]], 'B_p_total':[[ 10**4]],'n_clusters': [[10]]})
    # k_phi_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
    phi_config.append({ 'EMA weight':[0.6],'n_K':[1],'list_Delta_Q':[ [3*10**4] ],'bucket_size':[1],'h_min':[1],'h_max':[30],  'B_p_total': [10**4],'n_clusters': [10]})
    # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
    psi_config.append({'B_spec':[10**4],'B_p_total':[1*10**4],'n_buckets':[10]})
    # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
    upsilonzero_config.append({'B_spec':[10**4],'B_p_total':[1*10**4],'n_buckets':[10]})
penalty_type=None
penalty_kwargs=None

fitted_aslsd_exp6=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_exp6,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=penalty_type,penalty_kwargs=penalty_kwargs,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=True)
mu_aslsd_exp6=fitted_aslsd_exp6['Mu']
kernel_param_aslsd_exp6=fitted_aslsd_exp6['Kernel']

# # Make model for additionnal checks
matrix_param_aslsd_exp6=make_matrix_param(matrix_model_aslsd_exp6)
param_directory_aslsd_exp6=get_param_directory(d,matrix_param_aslsd_exp6)
phi_aslsd_exp6,diff_phi_aslsd_exp6,psi_aslsd_exp6,diff_psi_aslsd_exp6,upsilon_aslsd_exp6,diff_sim_upsilon_aslsd_exp6,diff_cross_upsilon_aslsd_exp6=make_model(matrix_model_aslsd_exp6)

matrix_phi_aslsd_exp6=func_compiler.compile_phi(phi_aslsd_exp6,kernel_param_aslsd_exp6,d)

adj_matrix_aslsd_exp6=l1_norm.make_adjacency_matrix(matrix_model_aslsd_exp6,kernel_param_aslsd_exp6,d)
radius_aslsd_exp6=l1_norm.get_spectral_radius(adj_matrix_aslsd_exp6)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# # y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# # plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# # fig.show()


# #   Resiudal analysis
# #       Compute residuals
psi_aslsd_exp6=func_maker.make_psi(matrix_model_aslsd_exp6)
matrix_psi_aslsd_exp6=func_compiler.compile_psi(psi_aslsd_exp6,kernel_param_aslsd_exp6,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

residuals_aslsd_exp6=gof_residuals.get_residuals(d,mu_aslsd_exp6,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_exp6,sampling=False,sample_size=10**3,seed=1234)
# #       KS test
gof_residuals.ks_test_residuals_k(residuals_aslsd_exp6[0])

# #       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp6[0])

# #       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_exp6[0],color='steelblue')
# fig.show()

# =============================================================================
# aslsd - SB Gaussian r=1
# =============================================================================
# r=1
# fixed_param_aslsd_sbgauss1=np.array([0.2,0.3])

# # Model Definition
# matrix_model_aslsd_sbgauss1=[[{'Type':'SB Gaussian','r':r,'Fixed parameters':fixed_param_aslsd_sbgauss1} for j in range(d)] for i in range(d)]
# mu_names_aslsd_sbgauss1,ker_names_aslsd_sbgauss1=get_mhp_param_names(matrix_model_aslsd_sbgauss1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_sbgauss1=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_sbgauss1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_sbgauss1=fitted_aslsd_sbgauss1['Mu']
# kernel_param_aslsd_sbgauss1=fitted_aslsd_sbgauss1['Kernel']


# # Make model for additionnal checks
# matrix_param_aslsd_sbgauss1=make_matrix_param(matrix_model_aslsd_sbgauss1)
# param_directory_aslsd_sbgauss1=get_param_directory(d,matrix_param_aslsd_sbgauss1)
# phi_aslsd_sbgauss1,diff_phi_aslsd_sbgauss1,psi_aslsd_sbgauss1,diff_psi_aslsd_sbgauss1,upsilon_aslsd_sbgauss1,diff_sim_upsilon_aslsd_sbgauss1,diff_cross_upsilon_aslsd_sbgauss1=make_model(matrix_model_aslsd_sbgauss1)

# matrix_phi_aslsd_sbgauss1=func_compiler.compile_phi(phi_aslsd_sbgauss1,kernel_param_aslsd_sbgauss1,d)

# adj_matrix_aslsd_sbgauss1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_sbgauss1,kernel_param_aslsd_sbgauss1,d)
# radius_aslsd_sbgauss1=l1_norm.get_spectral_radius(adj_matrix_aslsd_sbgauss1)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_sbgauss1=np.linspace(0,50,10**4)
# # y_phi_aslsd_sbgauss1=matrix_phi_aslsd_sbgauss1[0][0](x_phi_aslsd_sbgauss1)
# # plt.plot(x_phi_aslsd_sbgauss1,y_phi_aslsd_sbgauss1)
# # fig.show()


# #   Resiudal analysis

# #       Compute residuals
# psi_aslsd_sbgauss1=func_maker.make_psi(matrix_model_aslsd_sbgauss1)
# matrix_psi_aslsd_sbgauss1=func_compiler.compile_psi(psi_aslsd_sbgauss1,kernel_param_aslsd_sbgauss1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_sbgauss1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_sbgauss1=gof_residuals.get_residuals(d,mu_aslsd_sbgauss1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_sbgauss1)

# #       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_sbgauss1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss1[0])

# =============================================================================
# aslsd - SB Gaussian r=20
# =============================================================================
# print('aslsd - SB Gaussian r=20')
# r=20
# fixed_param_aslsd_sbgauss20=np.zeros(2*r)

# #   SBF config 1
# # array_delta_aslsd_sbgauss20=np.linspace(0.,10.,20)
# # for l in range(r):
# #     #   Beta
# #     fixed_param_aslsd_sbgauss20[2*l]=0.25
# #     #   Delta
# #     fixed_param_aslsd_sbgauss20[2*l+1]=array_delta_aslsd_sbgauss20[l]

# #   SBF config 2
# n_deltas_01=15
# n_deltas_14=5
# array_delta_aslsd_sbgauss20=np.concatenate((np.linspace(0.,1.,n_deltas_01, endpoint=False),np.linspace(1.,4.,n_deltas_14)))
# for l in range(r):
#     #   Beta
#     if l<r-1:
#         fixed_param_aslsd_sbgauss20[2*l]=0.5*(array_delta_aslsd_sbgauss20[l+1]-array_delta_aslsd_sbgauss20[l])
#     else:
#         fixed_param_aslsd_sbgauss20[2*l]=0.5*(array_delta_aslsd_sbgauss20[l]-array_delta_aslsd_sbgauss20[l-1])
#     #   Delta
#     fixed_param_aslsd_sbgauss20[2*l+1]=array_delta_aslsd_sbgauss20[l]

# # Model Definition
# matrix_model_aslsd_sbgauss20=[[{'Type':'SB Gaussian','r':r,'Fixed parameters':fixed_param_aslsd_sbgauss20} for j in range(d)] for i in range(d)]
# mu_names_aslsd_sbgauss20,ker_names_aslsd_sbgauss20=get_mhp_param_names(matrix_model_aslsd_sbgauss20,d)


# projection_exp1_sbgauss20=get_l2_projection_sbf_kernel(matrix_model_aslsd_exp1[0][0],kernel_param_aslsd_exp1[0][0],matrix_model_aslsd_sbgauss20[0][0],n_iter=10**4,base_seed=1245,omega_0='random',target_radius=0.7,learning_rate=10**-2,momentum_1=0.9,momentum_2=0.999,epsilon=10**-8,record_updates=True)
# omega_ref=projection_exp1_sbgauss20['omega']    
# ref_param_sbgauss20={}
# ref_param_sbgauss20['Mu']=mu_aslsd_exp1
# ref_param_sbgauss20['Kernel']=[[omega_ref]]

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-1]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[1000]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
#     upsilon_config.append({'EMA weight':[[0.6]],'n_K':[[1]],'list_Delta_Q':[[ [3*10**4] ]],'bucket_size':[[1]],'h_min':[[1]],'h_max':[[30]], 'B_p_total':[[ 10**4]],'n_clusters': [[10]]})
#     # k_phi_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
#     phi_config.append({ 'EMA weight':[0.6],'n_K':[1],'list_Delta_Q':[ [3*10**4] ],'bucket_size':[1],'h_min':[1],'h_max':[30],  'B_p_total': [10**4],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[10**4],'B_p_total':[2*10**4],'n_buckets':[20]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[10**4],'B_p_total':[2*10**4],'n_buckets':[20]})
# penalty_type=None
# penalty_kwargs=None

# fitted_aslsd_sbgauss20=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_sbgauss20,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=penalty_type,penalty_kwargs=penalty_kwargs,base_seed=base_seed,x_0='random',ref_param=ref_param_sbgauss20,range_ref=0.,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)

# mu_aslsd_sbgauss20=fitted_aslsd_sbgauss20['Mu']
# kernel_param_aslsd_sbgauss20=fitted_aslsd_sbgauss20['Kernel']

# # Make model for additionnal checks
# matrix_param_aslsd_sbgauss20=make_matrix_param(matrix_model_aslsd_sbgauss20)
# param_directory_aslsd_sbgauss20=get_param_directory(d,matrix_param_aslsd_sbgauss20)
# phi_aslsd_sbgauss20,diff_phi_aslsd_sbgauss20,psi_aslsd_sbgauss20,diff_psi_aslsd_sbgauss20,upsilon_aslsd_sbgauss20,diff_sim_upsilon_aslsd_sbgauss20,diff_cross_upsilon_aslsd_sbgauss20=make_model(matrix_model_aslsd_sbgauss20)

# matrix_phi_aslsd_sbgauss20=func_compiler.compile_phi(phi_aslsd_sbgauss20,kernel_param_aslsd_sbgauss20,d)

# adj_matrix_aslsd_sbgauss20=l1_norm.make_adjacency_matrix(matrix_model_aslsd_sbgauss20,kernel_param_aslsd_sbgauss20,d)
# radius_aslsd_sbgauss20=l1_norm.get_spectral_radius(adj_matrix_aslsd_sbgauss20)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_sbgauss20=np.linspace(0,10,10**4)
# # y_phi_aslsd_sbgauss20=matrix_phi_aslsd_sbgauss20[0][0](x_phi_aslsd_sbgauss20)
# # plt.plot(x_phi_aslsd_sbgauss20,y_phi_aslsd_sbgauss20)
# # fig.show()

# #   Iniital 
# matrix_phi_aslsd_sbgauss20_initial=func_compiler.compile_phi(phi_aslsd_sbgauss20,ref_param_sbgauss20['Kernel'],d)

# fig=plt.figure(dpi=300)
# x_phi_aslsd_sbgauss20=np.linspace(0,10,10**4)
# y_phi_aslsd_sbgauss20_initial=matrix_phi_aslsd_sbgauss20_initial[0][0](x_phi_aslsd_sbgauss20)
# plt.plot(x_phi_aslsd_sbgauss20,y_phi_aslsd_sbgauss20_initial,color='blue',label='L2 proj Exp1 VS SbGauss20')
# y_phi_aslsd_exp1=matrix_phi_aslsd_exp1[0][0](x_phi_aslsd_sbgauss20)
# plt.plot(x_phi_aslsd_sbgauss20,y_phi_aslsd_exp1,'--',color='orange',label='Exp1')
# y_phi_aslsd_sbgauss20=matrix_phi_aslsd_sbgauss20[0][0](x_phi_aslsd_sbgauss20)
# plt.plot(x_phi_aslsd_sbgauss20,y_phi_aslsd_sbgauss20,color='red',label='Fitted SbGauss20')
# plt.legend()
# fig.show()

# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_sbgauss20=func_maker.make_psi(matrix_model_aslsd_sbgauss20)
# matrix_psi_aslsd_sbgauss20=func_compiler.compile_psi(psi_aslsd_sbgauss20,kernel_param_aslsd_sbgauss20,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_sbgauss20[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_sbgauss20=gof_residuals.get_residuals(d,mu_aslsd_sbgauss20,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_sbgauss20,sampling=False,sample_size=10**3,seed=1234)

# #       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_sbgauss20[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss20[0])


# =============================================================================
# aslsd - Gaussian r=1
# =============================================================================
# print('aslsd - Gaussian r=1')
# r=1
# fixed_param_aslsd_gauss1=np.array([])

# # Model Definition
# matrix_model_aslsd_gauss1=[[{'Type':'Gaussian','r':1,'Fixed parameters':fixed_param_aslsd_gauss1} for j in range(d)] for i in range(d)]
# mu_names_aslsd_gauss1,ker_names_aslsd_gauss1=get_mhp_param_names(matrix_model_aslsd_gauss1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[2*10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
#     upsilon_config.append({'EMA weight':[[0.6]],'n_K':[[1]],'list_Delta_Q':[[ [3*10**4] ]],'bucket_size':[[1]],'h_min':[[1]],'h_max':[[20]], 'B_p_total':[[ 10**4]],'n_clusters': [[100]]})
#     # k_phi_config={'EMA weight':,'n_K':,'list_Delta_Q':,'bucket_size':,'h_min':,'h_max': , 'B_p_total':,'n_clusters': }
#     phi_config.append({ 'EMA weight':[0.6],'n_K':[1],'list_Delta_Q':[ [3*10**4] ],'bucket_size':[1],'h_min':[1],'h_max':[20],  'B_p_total': [10**4],'n_clusters': [100]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[10**4],'B_p_total':[2*10**4],'n_buckets':[20]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[10**4],'B_p_total':[2*10**4],'n_buckets':[20]})
# penalty_type=None
# penalty_kwargs=None

# fitted_aslsd_gauss1=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_gauss1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=penalty_type,penalty_kwargs=penalty_kwargs,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)


# mu_aslsd_gauss1=fitted_aslsd_gauss1['Mu']
# kernel_param_aslsd_gauss1=fitted_aslsd_gauss1['Kernel']


# # Make model for additionnal checks
# matrix_param_aslsd_gauss1=make_matrix_param(matrix_model_aslsd_gauss1)
# param_directory_aslsd_gauss1=get_param_directory(d,matrix_param_aslsd_gauss1)
# phi_aslsd_gauss1,diff_phi_aslsd_gauss1,psi_aslsd_gauss1,diff_psi_aslsd_gauss1,upsilon_aslsd_gauss1,diff_sim_upsilon_aslsd_gauss1,diff_cross_upsilon_aslsd_gauss1=make_model(matrix_model_aslsd_gauss1)

# matrix_phi_aslsd_gauss1=func_compiler.compile_phi(phi_aslsd_gauss1,kernel_param_aslsd_gauss1,d)

# adj_matrix_aslsd_gauss1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_gauss1,kernel_param_aslsd_gauss1,d)
# radius_aslsd_gauss1=l1_norm.get_spectral_radius(adj_matrix_aslsd_gauss1)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_gauss1=np.linspace(0,50,10**4)
# # y_phi_aslsd_gauss1=matrix_phi_aslsd_gauss1[0][0](x_phi_aslsd_gauss1)
# # plt.plot(x_phi_aslsd_gauss1,y_phi_aslsd_gauss1)
# # fig.show()


# #   Resiudal analysis

# #       Compute residuals
# psi_aslsd_gauss1=func_maker.make_psi(matrix_model_aslsd_gauss1)
# matrix_psi_aslsd_gauss1=func_compiler.compile_psi(psi_aslsd_gauss1,kernel_param_aslsd_gauss1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_sbgauss1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_gauss1=gof_residuals.get_residuals(d,mu_aslsd_gauss1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_gauss1,sampling=True,sample_size=10**3,seed=1234)

# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_gauss1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss1[0])

# # =============================================================================
# # aslsd - Gaussian r=6
# # =============================================================================
# r=6
# fixed_param_aslsd_gauss6=np.array([])

# # Model Definition
# matrix_model_aslsd_gauss6=[[{'Type':'Gaussian','r':r,'Fixed parameters':fixed_param_aslsd_gauss6} for j in range(d)] for i in range(d)]
# mu_names_aslsd_gauss6,ker_names_aslsd_gauss6=get_mhp_param_names(matrix_model_aslsd_gauss6,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_gauss6=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_gauss6,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_gauss6=fitted_aslsd_gauss6['Mu']
# kernel_param_aslsd_gauss6=fitted_aslsd_gauss6['Kernel']


# # Make model for additionnal checks
# matrix_param_aslsd_gauss6=make_matrix_param(matrix_model_aslsd_gauss6)
# param_directory_aslsd_gauss6=get_param_directory(d,matrix_param_aslsd_gauss6)
# phi_aslsd_gauss6,diff_phi_aslsd_gauss6,psi_aslsd_gauss6,diff_psi_aslsd_gauss6,upsilon_aslsd_gauss6,diff_sim_upsilon_aslsd_gauss6,diff_cross_upsilon_aslsd_gauss6=make_model(matrix_model_aslsd_gauss6)

# matrix_phi_aslsd_gauss6=func_compiler.compile_phi(phi_aslsd_gauss6,kernel_param_aslsd_gauss6,d)

# adj_matrix_aslsd_gauss6=l1_norm.make_adjacency_matrix(matrix_model_aslsd_gauss6,kernel_param_aslsd_gauss6,d)
# radius_aslsd_gauss6=l1_norm.get_spectral_radius(adj_matrix_aslsd_gauss6)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_gauss6=np.linspace(0,50,10**4)
# # y_phi_aslsd_gauss6=matrix_phi_aslsd_gauss6[0][0](x_phi_aslsd_gauss6)
# # plt.plot(x_phi_aslsd_gauss6,y_phi_aslsd_gauss6)
# # fig.show()


# #   Resiudal analysis

# #       Compute residuals
# psi_aslsd_gauss6=func_maker.make_psi(matrix_model_aslsd_gauss6)
# matrix_psi_aslsd_gauss6=func_compiler.compile_psi(psi_aslsd_gauss6,kernel_param_aslsd_gauss6,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_sbgauss1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_gauss6=gof_residuals.get_residuals(d,mu_aslsd_gauss6,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_gauss6,sampling=True,sample_size=10**3,seed=1234)

# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_gauss6[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss1[0])

# =============================================================================
# aslsd - Gaussian r=3
# =============================================================================
# r=3

# # fixed_param_aslsd=np.array([0.5,0.5])
# fixed_param_aslsd_sbgauss3=np.array([0.2,0.3,1.,11.,1.,24.])

# # Model Definition
# matrix_model_aslsd_sbgauss3=[[{'Type':'SB Gaussian','r':r,'Fixed parameters':fixed_param_aslsd_sbgauss3} for j in range(d)] for i in range(d)]
# mu_names_aslsd_sbgauss3,ker_names_aslsd_sbgauss3=get_mhp_param_names(matrix_model_aslsd_sbgauss3,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[5*10**2]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_sbgauss3=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_sbgauss3,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_sbgauss3=fitted_aslsd_sbgauss3['Mu']
# kernel_param_aslsd_sbgauss3=fitted_aslsd_sbgauss3['Kernel']


# # Make model for additionnal checks
# matrix_param_aslsd_sbgauss3=make_matrix_param(matrix_model_aslsd_sbgauss3)
# param_directory_aslsd_sbgauss3=get_param_directory(d,matrix_param_aslsd_sbgauss3)
# phi_aslsd_sbgauss3,diff_phi_aslsd_sbgauss3,psi_aslsd_sbgauss3,diff_psi_aslsd_sbgauss3,upsilon_aslsd_sbgauss3,diff_sim_upsilon_aslsd_sbgauss3,diff_cross_upsilon_aslsd_sbgauss3=make_model(matrix_model_aslsd_sbgauss3)

# matrix_phi_aslsd_sbgauss3=func_compiler.compile_phi(phi_aslsd_sbgauss3,kernel_param_aslsd_sbgauss3,d)

# adj_matrix_aslsd_sbgauss3=l1_norm.make_adjacency_matrix(matrix_model_aslsd_sbgauss3,kernel_param_aslsd_sbgauss3,d)
# radius_aslsd_sbgauss3=l1_norm.get_spectral_radius(adj_matrix_aslsd_sbgauss3)

# fig=plt.figure(dpi=300)
# x_phi_aslsd_sbgauss3=np.linspace(0,50,10**4)
# y_phi_aslsd_sbgauss3=matrix_phi_aslsd_sbgauss3[0][0](x_phi_aslsd_sbgauss3)
# plt.plot(x_phi_aslsd_sbgauss3,y_phi_aslsd_sbgauss3)
# fig.show()


# #   Resiudal test
# psi_aslsd_sbgauss3=func_maker.make_psi(matrix_model_aslsd_sbgauss3)
# matrix_psi_aslsd_sbgauss3=func_compiler.compile_psi(psi_aslsd_sbgauss3,kernel_param_aslsd_sbgauss3,d)

# fig=plt.figure(dpi=300)
# x=np.linspace(0,10**1,10**4)
# y=matrix_psi_aslsd_sbgauss3[0][0](x)
# plt.plot(x,y)
# fig.show()




# # residuals_aslsd_sbgauss3=gof_residuals.get_residuals(d,mu_aslsd_sbgauss3,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_sbgauss3)
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_sbgauss3[0])

# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss3[0])

# =============================================================================
# aslsd - Mixed: SB Gaussian r=1 + SB Exponential r=1
# =============================================================================
# #   {'List r','List fixed parameters','List kernel types','Parameter index bounds','Index converter'}

# # fixed_param_aslsd=np.array([0.5,0.5])
# list_r_aslsd_mixed1=[1,1]
# list_fixed_param_aslsd_mixed1=np.array([[0.2,0.15],[.5]],dtype=object)
# list_ker_types_aslsd_mixed1=['SB Gaussian','SB Exponential']
# # Model Definition
# matrix_model_aslsd_mixed1=[[{'Type':'Mixed','List r':list_r_aslsd_mixed1,'List fixed parameters':list_fixed_param_aslsd_mixed1,'List kernel types':list_ker_types_aslsd_mixed1} for j in range(d)] for i in range(d)]
# # mu_names_aslsd_mixed1,ker_names_aslsd_mixed1=get_mhp_param_names(matrix_model_aslsd_mixed1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_mixed1=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_mixed1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_mixed1=fitted_aslsd_mixed1['Mu']
# kernel_param_aslsd_mixed1=fitted_aslsd_mixed1['Kernel']




# #   Make model for additionnal checks
# matrix_param_aslsd_mixed1=make_matrix_param(matrix_model_aslsd_mixed1)
# param_directory_aslsd_mixed1=get_param_directory(d,matrix_param_aslsd_mixed1)
# phi_aslsd_mixed1,diff_phi_aslsd_mixed1,psi_aslsd_mixed1,diff_psi_aslsd_mixed1,upsilon_aslsd_mixed1,diff_sim_upsilon_aslsd_mixed1,diff_cross_upsilon_aslsd_mixed1=make_model(matrix_model_aslsd_mixed1)
# matrix_phi_aslsd_mixed1=func_compiler.compile_phi(phi_aslsd_mixed1,kernel_param_aslsd_mixed1,d)
# #adj_matrix_aslsd_mixed1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_mixed1,kernel_param_aslsd_mixed1,d)
# #radius_aslsd_mixed1=l1_norm.get_spectral_radius(adj_matrix_aslsd_mixed1)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# # y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# # plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# # fig.show()


# #############################################################################
# # #   Tests
# # t_test=np.linspace(0,10**3,10**4)
# # s_test=np.linspace(0,10**1,10**4)
# # list_param=np.array([1.,1.])

# # # val_test=psi_aslsd_mixed1(0,0,list_param,t_test)
# # # diff_index=0
# # # val_test=diff_psi_aslsd_mixed1(0,0,list_param,t_test,diff_index)

# # # val_test=upsilon_aslsd_mixed1(0,0,0,list_param,list_param,t_test,1000.)

# # diff_index=0
# # # print(matrix_model_aslsd_mixed1[0][0]['Index converter'](diff_index) )
# # val_test=diff_sim_upsilon_aslsd_mixed1(0,0,list_param,t_test,10**5,diff_index)

# # fig=plt.figure(dpi=300)

# # plt.plot(t_test,val_test)
# # fig.show()
# #############################################################################

# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_mixed1=func_maker.make_psi(matrix_model_aslsd_mixed1)
# matrix_psi_aslsd_mixed1=func_compiler.compile_psi(psi_aslsd_mixed1,kernel_param_aslsd_mixed1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_mixed1=gof_residuals.get_residuals(d,mu_aslsd_mixed1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_mixed1)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_mixed1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_mixed1[0])

# #       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_mixed1[0],color='steelblue')
# fig.show()

# =============================================================================
# aslsd - Mixed: Gaussian r=1 + Exponential r=1
# =============================================================================
# #   {'List r','List fixed parameters','List kernel types','Parameter index bounds','Index converter'}

# # fixed_param_aslsd=np.array([0.5,0.5])
# list_r_aslsd_mixed1=[1,1]
# list_fixed_param_aslsd_mixed1=np.array([[],[]],dtype=object)
# list_ker_types_aslsd_mixed1=['Gaussian','Exponential']
# # Model Definition
# matrix_model_aslsd_mixed1=[[{'Type':'Mixed','List r':list_r_aslsd_mixed1,'List fixed parameters':list_fixed_param_aslsd_mixed1,'List kernel types':list_ker_types_aslsd_mixed1} for j in range(d)] for i in range(d)]
# # mu_names_aslsd_mixed1,ker_names_aslsd_mixed1=get_mhp_param_names(matrix_model_aslsd_mixed1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_mixed1=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_mixed1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_mixed1=fitted_aslsd_mixed1['Mu']
# kernel_param_aslsd_mixed1=fitted_aslsd_mixed1['Kernel']




# #   Make model for additionnal checks
# matrix_param_aslsd_mixed1=make_matrix_param(matrix_model_aslsd_mixed1)
# param_directory_aslsd_mixed1=get_param_directory(d,matrix_param_aslsd_mixed1)
# phi_aslsd_mixed1,diff_phi_aslsd_mixed1,psi_aslsd_mixed1,diff_psi_aslsd_mixed1,upsilon_aslsd_mixed1,diff_sim_upsilon_aslsd_mixed1,diff_cross_upsilon_aslsd_mixed1=make_model(matrix_model_aslsd_mixed1)
# matrix_phi_aslsd_mixed1=func_compiler.compile_phi(phi_aslsd_mixed1,kernel_param_aslsd_mixed1,d)
# #adj_matrix_aslsd_mixed1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_mixed1,kernel_param_aslsd_mixed1,d)
# #radius_aslsd_mixed1=l1_norm.get_spectral_radius(adj_matrix_aslsd_mixed1)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# # y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# # plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# # fig.show()

# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_mixed1=func_maker.make_psi(matrix_model_aslsd_mixed1)
# matrix_psi_aslsd_mixed1=func_compiler.compile_psi(psi_aslsd_mixed1,kernel_param_aslsd_mixed1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_mixed1=gof_residuals.get_residuals(d,mu_aslsd_mixed1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_mixed1,sampling=True,sample_size=10**3,seed=1234)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_mixed1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_mixed1[0])

# #       Resdiual plot
# # fig=plt.figure(dpi=300)
# # plt.plot(times[:-1],residuals_aslsd_mixed1[0],color='steelblue')
# # fig.show()

# =============================================================================
# aslsd - Mixed: Gaussian r=6 + Exponential r=1
# =============================================================================
# #   {'List r','List fixed parameters','List kernel types','Parameter index bounds','Index converter'}

# # fixed_param_aslsd=np.array([0.5,0.5])
# list_r_aslsd_mixed_gauss6_exp1=[6,1]
# list_fixed_param_aslsd_mixed_gauss6_exp1=np.array([[],[]],dtype=object)
# list_ker_types_aslsd_mixed_gauss6_exp1=['Gaussian','Exponential']
# # Model Definition
# matrix_model_aslsd_mixed_gauss6_exp1=[[{'Type':'Mixed','List r':list_r_aslsd_mixed_gauss6_exp1,'List fixed parameters':list_fixed_param_aslsd_mixed_gauss6_exp1,'List kernel types':list_ker_types_aslsd_mixed_gauss6_exp1} for j in range(d)] for i in range(d)]
# # mu_names_aslsd_mixed_gauss6_exp1,ker_names_aslsd_mixed_gauss6_exp1=get_mhp_param_names(matrix_model_aslsd_mixed_gauss6_exp1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]
# vec_momentum_2=[0.999*10**0]
# vec_n_iter=[10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30]], 'max_B_spec_h':[[1000 ]], 'B_p_total':[[ 10**3]],'n_clusters': [[10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30], 'max_B_spec_h':[1000] , 'B_p_total': [10**3],'n_clusters': [10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[1000],'B_p_total':[1000],'n_buckets':[10]})

# fitted_aslsd_mixed_gauss6_exp1=fit_adam_stratified(tensor_event_times,T_f,matrix_model_aslsd_mixed_gauss6_exp1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,base_seed=base_seed,vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)
# mu_aslsd_mixed_gauss6_exp1=fitted_aslsd_mixed_gauss6_exp1['Mu']
# kernel_param_aslsd_mixed_gauss6_exp1=fitted_aslsd_mixed_gauss6_exp1['Kernel']




# #   Make model for additionnal checks
# matrix_param_aslsd_mixed_gauss6_exp1=make_matrix_param(matrix_model_aslsd_mixed_gauss6_exp1)
# param_directory_aslsd_mixed_gauss6_exp1=get_param_directory(d,matrix_param_aslsd_mixed_gauss6_exp1)
# phi_aslsd_mixed_gauss6_exp1,diff_phi_aslsd_mixed_gauss6_exp1,psi_aslsd_mixed_gauss6_exp1,diff_psi_aslsd_mixed_gauss6_exp1,upsilon_aslsd_mixed_gauss6_exp1,diff_sim_upsilon_aslsd_mixed_gauss6_exp1,diff_cross_upsilon_aslsd_mixed_gauss6_exp1=make_model(matrix_model_aslsd_mixed_gauss6_exp1)
# matrix_phi_aslsd_mixed_gauss6_exp1=func_compiler.compile_phi(phi_aslsd_mixed_gauss6_exp1,kernel_param_aslsd_mixed_gauss6_exp1,d)
# #adj_matrix_aslsd_mixed_gauss6_exp1=l1_norm.make_adjacency_matrix(matrix_model_aslsd_mixed_gauss6_exp1,kernel_param_aslsd_mixed_gauss6_exp1,d)
# #radius_aslsd_mixed_gauss6_exp1=l1_norm.get_spectral_radius(adj_matrix_aslsd_mixed_gauss6_exp1)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_mixed_gauss6_exp1=np.linspace(0,50,10**4)
# # y_phi_aslsd_mixed_gauss6_exp1=matrix_phi_aslsd_mixed_gauss6_exp1[0][0](x_phi_aslsd_mixed_gauss6_exp1)
# # plt.plot(x_phi_aslsd_mixed_gauss6_exp1,y_phi_aslsd_mixed_gauss6_exp1)
# # fig.show()

# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_mixed_gauss6_exp1=func_maker.make_psi(matrix_model_aslsd_mixed_gauss6_exp1)
# matrix_psi_aslsd_mixed_gauss6_exp1=func_compiler.compile_psi(psi_aslsd_mixed_gauss6_exp1,kernel_param_aslsd_mixed_gauss6_exp1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed_gauss6_exp1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_mixed_gauss6_exp1=gof_residuals.get_residuals(d,mu_aslsd_mixed_gauss6_exp1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_mixed_gauss6_exp1,sampling=True,sample_size=10**3,seed=1234)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_mixed_gauss6_exp1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_mixed_gauss6_exp1[0])

# #       Resdiual plot
# # fig=plt.figure(dpi=300)
# # plt.plot(times[:-1],residuals_aslsd_mixed_gauss6_exp1[0],color='steelblue')
# # fig.show()
# =============================================================================
# Custom model 1
# =============================================================================
# val_0=3.
# beta_exp=6.
# omega_exp=val_0/beta_exp
# r_gauss=5
# r_exp=1
# list_r_custom1=[r_gauss,r_exp]
# list_fixed_param_custom1=np.array([[0.05,0.48,0.05,1.,0.05,2.,0.05,3.,0.05,4.],[beta_exp]],dtype=object)
# list_ker_types_custom1=['SB Gaussian','SB Exponential']

# matrix_model_custom1=[[{'Type':'Mixed','List r':list_r_custom1,'List fixed parameters':list_fixed_param_custom1,'List kernel types':list_ker_types_custom1} for j in range(d)] for i in range(d)]
# matrix_model_custom1[0][0]['List fixed parameters']=list_fixed_param_custom1
# matrix_param_custom1=make_matrix_param(matrix_model_custom1)
# matrix_model_custom1=make_index_converter(matrix_model_custom1,matrix_param_custom1)


# mu_custom1=np.array([0.3])

# kernel_param_custom1=np.ones( (1,1,r_gauss+r_exp) )
# kernel_param_custom1[0][0][:r_gauss]=.06
# kernel_param_custom1[0][0][-1]=omega_exp

# phi_custom1,diff_phi_custom1,psi_custom1,diff_psi_custom1,upsilon_custom1,diff_sim_upsilon_custom1,diff_cross_upsilon_custom1=make_model(matrix_model_custom1)
# matrix_phi_custom1=func_compiler.compile_phi(phi_custom1,kernel_param_custom1,d)

# fig=plt.figure(dpi=1200)
# x_phi=np.linspace(0,5,10**4)
# y_phi_wh=matrix_phi_wh[0][0](x_phi)
# y_phi_custom1=matrix_phi_custom1[0][0](x_phi)
# plt.plot(x_phi,y_phi_wh,color='firebrick',label='WH')
# plt.plot(x_phi,y_phi_custom1,color='blue',label='Custom')
# plt.legend()
# plt.xlabel(r'$t$')
# plt.ylabel(r'$\phi_{0,0}(t)$')
# # plt.yscale('log')
# fig.show()


# # #   Resiudal analysis
# # #       Compute residuals
# psi_custom1=func_maker.make_psi(matrix_model_custom1)
# matrix_psi_custom1=func_compiler.compile_psi(psi_custom1,kernel_param_custom1,d)

# residuals_custom1=gof_residuals.get_residuals(d,mu_custom1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_custom1,sampling=True,sample_size=10**3,seed=1234)
# # #       KS test
# # # gof_residuals.ks_test_residuals_k(residuals_custom1[0])
# # #       QQ plot
# # # gof_residuals.qq_plot_residuals_k(residuals_custom1[0])
# # #       Resdiual plot
# # # fig=plt.figure(dpi=300)
# # # plt.plot(times[:-1],residuals_custom1[0],color='steelblue')
# # # fig.show()

# =============================================================================
# Comparison
# =============================================================================
model_dict={}

#   Benchmarks
model_dict['WH']={'phi':matrix_phi_wh,'residuals':residuals_wh[0],'color':'firebrick'}
model_dict['SumExp']={'phi':matrix_phi_sumexp,'residuals':residuals_sumexp[0],'color':'darkorange'}
# model_dict['Poisson']={'phi':matrix_phi_hompoisson,'residuals':residuals_hompoisson[0],'color':'gold'}

#   Personal

#       Mixed
# model_dict['ASLSD mixed1']={'phi':matrix_phi_aslsd_mixed1,'residuals':residuals_aslsd_mixed1[0],'color':'steelblue'}
# model_dict['ASLSD mixed gauss6 exp1']={'phi':matrix_phi_aslsd_mixed_gauss6_exp1,'residuals':residuals_aslsd_mixed_gauss6_exp1[0],'color':'steelblue'}
# model_dict['Custom1']={'phi':matrix_phi_custom1,'residuals':residuals_custom1[0],'color':'cyan'}

#       Exponential
model_dict['ASLSD Exp1']={'phi':matrix_phi_aslsd_exp1,'residuals':residuals_aslsd_exp1[0],'color':'springgreen'}
# model_dict['ASLSD exp3']={'phi':matrix_phi_aslsd_exp3,'residuals':residuals_aslsd_exp3[0],'color':'lime'}
model_dict['ASLSD Exp6']={'phi':matrix_phi_aslsd_exp6,'residuals':residuals_aslsd_exp6[0],'color':'steelblue'}

#       Gaussian
# model_dict['ASLSD sbgauss1']={'phi':matrix_phi_aslsd_sbgauss1,'residuals':residuals_aslsd_sbgauss1[0],'color':'darkorange'}
# model_dict['ASLSD gauss1']={'phi':matrix_phi_aslsd_gauss1,'residuals':residuals_aslsd_gauss1[0],'color':'black'}
# model_dict['ASLSD gauss6']={'phi':matrix_phi_aslsd_gauss6,'residuals':residuals_aslsd_gauss6[0],'color':'green'}
# model_dict['ASLSD sbgauss20']={'phi':matrix_phi_aslsd_sbgauss20,'residuals':residuals_aslsd_sbgauss20[0],'color':'black'}

#   Select models to display
def select_model_feature(model_dict,model_names,feature_key):
    return [model_dict[model_name][feature_key] for model_name in model_names]

active_model_names=['WH','SumExp','ASLSD Exp1','ASLSD Exp6']
# active_model_names=['WH','SumExp','Poisson','ASLSD sbgauss20']
# active_model_names=['SumExp','ASLSD exp6']
n_active_models=len(active_model_names)
active_model_phis=select_model_feature(model_dict,active_model_names,'phi')
active_model_residuals=select_model_feature(model_dict,active_model_names,'residuals')
active_model_colors=select_model_feature(model_dict,active_model_names,'color')

# =============================================================================
# Save Data
# =============================================================================
# file = open('memetracker_royal_1d_modeldict.obj', 'w') 
# pickle.dump(model_dict, file)


# =============================================================================
# Make plots
# =============================================================================
#   Phi plots
i_phi=0
j_phi=0
fig=plt.figure(dpi=1200)
x_phi=np.linspace(0,10,10**4)
y_phi_models=[None]*n_active_models
for m in range(n_active_models):
    if active_model_names[m]!='Poisson':
        y_phi_models[m]=active_model_phis[m][i_phi][j_phi](x_phi)
        plt.plot(x_phi,y_phi_models[m],color=active_model_colors[m],label=active_model_names[m])
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\phi_{'+str(i_phi)+str(j_phi)+'}(t)$')
# plt.yscale('log')
fig.show()

#   Residual analysis
#       Q-Q Plots
gof_residuals.qq_plot_residuals_vect(active_model_residuals,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=True,markersize=5,dpi=1200)

#       Probability Plots
gof_residuals.qq_plot_residuals_vect(active_model_residuals,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=5,display_line45=False,dpi=1200)

#       Probability Plots minus y=x
gof_residuals.qq_plot_residuals_vect(active_model_residuals,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,dpi=1200)


#       KS test
ks_test_level=0.05
for m in range(n_active_models):
    pvalue=gof_residuals.ks_test_residuals_k(active_model_residuals[m]).pvalue
    if pvalue>ks_test_level:
        test_outcome='Success'
    else:
        test_outcome='Failure'
    print('Model '+active_model_names[m]+' p-value: ',pvalue,', KS test ',test_outcome)


# =============================================================================
# Summary
# =============================================================================
# Part I: Q-Q Plot and rescaled probability plot
ks_critical=1.628

fig, axs = plt.subplots(1, 2, sharex=False, sharey=False,figsize=(8, 3),dpi=300)

axs[0]=gof_residuals.qq_plot_residuals_vect(active_model_residuals,active_model_colors,proba=False,n_ref_data=10**7,label=False,list_labels=active_model_names,log_scale=True,markersize=2,max_points=1000,dpi=300,ax=axs[0])
axs[0].set_title('Q-Q plot')

axs[1]=gof_residuals.qq_plot_residuals_vect(active_model_residuals,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,normalize=True,max_points=1000,dpi=300,ax=axs[1])
xmin,xmax = axs[1].get_xlim()
axs[1].hlines(y=ks_critical,xmin=xmin,xmax=xmax,linestyle='dashed', linewidth=1, color='black')
axs[1].hlines(y=-ks_critical,xmin=xmin,xmax=xmax,linestyle='dashed', linewidth=1, color='black')
axs[1].set_title('Rescaled probability plot')
fig.show()

tikzplotlib.save("tex/memetracker1d_residual_plots.tex")

# Part II: Fitted kernels
fig=plt.figure(dpi=300)
x_phi=np.linspace(0,4,10**3)
y_phi_models=[None]*n_active_models
for m in range(n_active_models):
    if active_model_names[m]!='Poisson':
        y_phi_models[m]=active_model_phis[m][0][0](x_phi)
        plt.plot(x_phi,y_phi_models[m],color=active_model_colors[m],label=active_model_names[m])
plt.legend()
plt.xlabel(r'$t$')
plt.ylabel(r'$\phi(t)$')
# plt.yscale('log')
fig.show()

tikzplotlib.save("tex/memetracker1d_fitted_kernels.tex")