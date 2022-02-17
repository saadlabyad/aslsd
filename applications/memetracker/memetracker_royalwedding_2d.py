# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:47:03 2021

@author: saadl
"""
import os,sys,inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
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
break_ind=10**6
web_dict, cascades_list=mcp.read_cascade(keyword,break_ind)
final_webdict=mcp.load_countrywise_webdict(keyword) 
  
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
for i in range(len(web_countries)):
    if web_countries[i]==' ':
        web_countries[i]='Unknown'


df=pd.DataFrame.from_dict({'time':times,'id':web_ids,'country':web_countries})
df['country'].value_counts()
df['country'].value_counts().plot(kind='bar')

times_us=np.array(df.loc[df['country']=='United States of America']['time'] )
times_uk=np.array(df.loc[df['country']=='United Kingdom']['time'] )

# fig=plt.figure(dpi=300)
# plt.plot(times_us,[i for i in range(len(times_us))],color='steelblue')
# plt.plot(times_uk,[i for i in range(len(times_uk))],color='green')
# plt.xlabel('Publication time')
# plt.ylabel('Cumulative # of publications')
# plt.title(keyword+' keyword, pre-filter')
# fig.show()


tensor_event_times=[times_us,times_uk]
d=len(tensor_event_times)

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

for i in range(d):
    tensor_event_times[i]=mcp.rescale_times(tensor_event_times[i],rescale_factor,t_rec_min,t_rec_max)

# =============================================================================
# Visualization
# =============================================================================
# fig=plt.figure(dpi=1200)
# plt.plot(tensor_event_times[0],[i for i in range(len(tensor_event_times[0]))],color='steelblue',label='US')
# plt.plot(tensor_event_times[1],[i for i in range(len(tensor_event_times[1]))],color='green',label='UK')
# # plt.plot([0,times[-1]],[0,times[-1]],color='red')
# # t_special_event_1=datetime(2011,7,23,13,0,0, tzinfo=timezone.utc).timestamp()-t_rec_min
# # plt.axvline(x=t_special_event_1, color='firebrick', linestyle='-')
# plt.xlabel('Publication time')
# plt.ylabel('Cumulative # of publications')
# plt.title(keyword+' keyword')
# plt.legend()
# fig.show()

# print('# of events before Spec. Ev. :',len(times[(times<=t_special_event_1)]))
# print('# of events after Spec. Ev. :',len(times[(times>=t_special_event_1)]))

# =============================================================================
# Final data
# =============================================================================
total_times=copy.deepcopy(list(tensor_event_times[0]))
total_times.extend(list(tensor_event_times[1]))
total_times.sort()
total_times=np.array(total_times)

print('Number of events after filtering: ',len(total_times))

T_f=t_rec_max-t_rec_min

list_event_counts=[len(tensor_event_times[i]) for i in range(d)]
matrix_kappa,matrix_varkappa =get_kappa_varkappa(tensor_event_times,T_f)
eta=[len(tensor_event_times[i])/T_f for i in range(d)]

# =============================================================================
# Poisson
# =============================================================================
print('Fit Poisson')
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
residuals_hompoisson=gof_residuals.get_residuals_homogeneouspoisson(d,mu_hompoisson,tensor_event_times)

# #       KS test
# gof_residuals.ks_test_residuals_k(residuals_hompoisson[0])
# gof_residuals.ks_test_residuals_k(residuals_hompoisson[1])

# #       QQ plot
# gof_residuals.qq_plot_residuals_k(residuals_hompoisson[0],proba=False,log_scale=False,reference='uniform')
# gof_residuals.qq_plot_residuals_k(residuals_hompoisson[1],proba=False,log_scale=False,reference='uniform')

#       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(tensor_event_times[0][:-1],residuals_hompoisson[0],color='steelblue')
# plt.plot(tensor_event_times[1][:-1],residuals_hompoisson[1],color='steelblue')
# fig.show()

# =============================================================================
# Fit WH
# =============================================================================
print('Fit WH')
max_support=10
mu_wh, matrix_phi_wh=fit_tick_wh(tensor_event_times,claw_method="lin", delta_lag=1., min_lag=5e-4, max_lag=1000, quad_method="gauss", n_quad=10**3,min_support=1e-4, max_support=max_support, n_threads=4)
for i in range(d):
    for j in range(d):
        fig=plt.figure(dpi=300)
        x_phi_wh=np.linspace(0,max_support,10**4)
        y_phi_wh=matrix_phi_wh[i][j](x_phi_wh)
        plt.plot(x_phi_wh,y_phi_wh)
        plt.title(r'$\phi_{'+str(i)+str(j)+'}$')
        fig.show()

# print('L1 norm',l1_norm.get_l1_norm_nonparam(matrix_phi_wh[0][0]))

#   Residual analysis
#       Compute residuals
matrix_psi_wh=func_compiler.compile_psi_numeric(matrix_phi_wh,d,x_max=50.,int_res=10**4)
residuals_wh=gof_residuals.get_residuals(d,mu_wh,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_wh,sampling=False,sample_size=10**3,seed=1234)
#       KS test
gof_residuals.ks_test_residuals_k(residuals_wh[1])

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
vec_learning_rate=[10**-1,10**-1]
vec_epsilon=[10**-8,10**-8]
vec_momentum_1=[0.9*10**0,0.9*10**0]
vec_momentum_2=[0.999*10**0,0.999*10**0]
vec_n_iter=[2000,2000]

# Stratification config
upsilon_config=[]
phi_config=[]
psi_config=[]
upsilonzero_config=[]

for k in range(d):
    # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
    upsilon_config.append({'EMA weight': [[0.6,0.1],[0.6,0.6]],'n_K': [[1,1],[1,1]],'list_Delta_Q': [[[3*10**4],[3*10**4]],[[3*10**4],[3*10**4]]],'bucket_size': [[1,1],[1,1]], 'h_min':[[1,1 ],[1,1 ]], 'h_max':[[30,30 ],[30,30 ]], 'B_p_total':[[ 10**4,10**4],[ 10**4,10**4]],'n_clusters': [[10,10],[10,10]]})
    # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
    phi_config.append({'EMA weight': [0.6,0.6],'n_K': [1,1],'list_Delta_Q':[ [3*10**4],[3*10**4] ],'bucket_size':[1,1],'h_min':[1,1],'h_max':[30,30], 'B_p_total': [10**4,10**4],'n_clusters': [10,10]})
    # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
    psi_config.append({'B_spec':[10**4,10**4],'B_p_total':[10**4,10**4],'n_buckets':[10,10]})
    # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
    upsilonzero_config.append({'B_spec':[10**4,10**4],'B_p_total':[10**4,10**4],'n_buckets':[10,10]})

#   Penalty Configuration
#       Type 1: None
penalty_type=None
penalty_kwargs=None

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
gt.plot_adj_matrix(adj_matrix_aslsd_exp1,['US','UK'],cmap="Blues",dpi=300,show=True,annotate=False)
# fig=plt.figure(dpi=300)
# x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# fig.show()


#   Resiudal analysis
#       Compute residuals
psi_aslsd_exp1=func_maker.make_psi(matrix_model_aslsd_exp1)
matrix_psi_aslsd_exp1=func_compiler.compile_psi(psi_aslsd_exp1,kernel_param_aslsd_exp1,d)

# fig=plt.figure(dpi=300)
# x=np.linspace(0,10**1,10**4)
# y=matrix_psi_aslsd_mixed1[0][0](x)
# plt.plot(x,y)
# fig.show()

residuals_aslsd_exp1=gof_residuals.get_residuals(d,mu_aslsd_exp1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_exp1,sampling=False,sample_size=10**3,seed=1234)
#       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_exp1[0])
# gof_residuals.ks_test_residuals_k(residuals_aslsd_exp1[1])

# #       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp1[0],reference='uniform')
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp1[1],reference='uniform')

#       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_exp1[0],color='steelblue')
# plt.yscale('log')
# fig.show()

# =============================================================================
# aslsd - Exponential r=3
# =============================================================================
print('aslsd - Exponential r=3')
# Model Definition
matrix_model_aslsd_exp3=[[{'Type':'Exponential','r':3,'Fixed parameters':[]} for j in range(d)] for i in range(d)]
mu_names_aslsd_exp3,ker_names_aslsd_exp3=get_mhp_param_names(matrix_model_aslsd_exp3,d)

# Learning parameters
base_seed=1245
vec_learning_rate=[10**-2,10**-2]
vec_epsilon=[10**-8,10**-8]
vec_momentum_1=[0.9*10**0,0.9*10**0]
vec_momentum_2=[0.999*10**0,0.999*10**0]
vec_n_iter=[2*10**3,2*10**3]

# Stratification config
upsilon_config=[]
phi_config=[]
psi_config=[]
upsilonzero_config=[]

for k in range(d):
    # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
    upsilon_config.append({'EMA weight': [[0.6,0.1],[0.6,0.6]],'n_K': [[1,1],[1,1]],'list_Delta_Q': [[[3*10**4],[3*10**4]],[[3*10**4],[3*10**4]]],'bucket_size': [[1,1],[1,1]], 'h_min':[[1,1 ],[1,1 ]], 'h_max':[[30,30 ],[30,30 ]], 'B_p_total':[[ 10**4,10**4],[ 10**4,10**4]],'n_clusters': [[10,10],[10,10]]})
    # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
    phi_config.append({'EMA weight': [0.6,0.6],'n_K': [1,1],'list_Delta_Q':[ [3*10**4],[3*10**4] ],'bucket_size':[1,1],'h_min':[1,1],'h_max':[30,30], 'B_p_total': [10**4,10**4],'n_clusters': [10,10]})
    # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
    psi_config.append({'B_spec':[10**4,10**4],'B_p_total':[10**4,10**4],'n_buckets':[10,10]})
    # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
    upsilonzero_config.append({'B_spec':[10**4,10**4],'B_p_total':[10**4,10**4],'n_buckets':[10,10]})

#   Penalty Configuration
#       Type 1: None
penalty_type=None
penalty_kwargs=None

fitted_aslsd_exp3=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_exp3,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=penalty_type,penalty_kwargs=penalty_kwargs,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=True)

mu_aslsd_exp3=fitted_aslsd_exp3['Mu']
kernel_param_aslsd_exp3=fitted_aslsd_exp3['Kernel']

# Make model for additionnal checks
matrix_param_aslsd_exp3=make_matrix_param(matrix_model_aslsd_exp3)
param_directory_aslsd_exp3=get_param_directory(d,matrix_param_aslsd_exp3)
phi_aslsd_exp3,diff_phi_aslsd_exp3,psi_aslsd_exp3,diff_psi_aslsd_exp3,upsilon_aslsd_exp3,diff_sim_upsilon_aslsd_exp3,diff_cross_upsilon_aslsd_exp3=make_model(matrix_model_aslsd_exp3)

matrix_phi_aslsd_exp3=func_compiler.compile_phi(phi_aslsd_exp3,kernel_param_aslsd_exp3,d)

adj_matrix_aslsd_exp3=l1_norm.make_adjacency_matrix(matrix_model_aslsd_exp3,kernel_param_aslsd_exp3,d)
radius_aslsd_exp3=l1_norm.get_spectral_radius(adj_matrix_aslsd_exp3)

# fig=plt.figure(dpi=300)
# x_phi_aslsd_mixed1=np.linspace(0,50,10**4)
# y_phi_aslsd_mixed1=matrix_phi_aslsd_mixed1[0][0](x_phi_aslsd_mixed1)
# plt.plot(x_phi_aslsd_mixed1,y_phi_aslsd_mixed1)
# fig.show()


#   Resiudal analysis
#       Compute residuals
psi_aslsd_exp3=func_maker.make_psi(matrix_model_aslsd_exp3)
matrix_psi_aslsd_exp3=func_compiler.compile_psi(psi_aslsd_exp3,kernel_param_aslsd_exp3,d)

# fig=plt.figure(dpi=300)
# x=np.linspace(0,10**1,10**4)
# y=matrix_psi_aslsd_mixed1[0][0](x)
# plt.plot(x,y)
# fig.show()

residuals_aslsd_exp3=gof_residuals.get_residuals(d,mu_aslsd_exp3,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_exp3,sampling=False,sample_size=10**3,seed=1234)
#       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_exp3[0])

#       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_exp3[0])

#       Resdiual plot
# fig=plt.figure(dpi=300)
# plt.plot(times[:-1],residuals_aslsd_exp3[0],color='steelblue')
# fig.show()

# # =============================================================================
# # aslsd - SB Gaussian r=1
# # =============================================================================
# r=1
# fixed_param_aslsd_sbgauss1=np.array([0.2,0.3])

# # Model Definition
# matrix_model_aslsd_sbgauss1=[[{'Type':'SB Gaussian','r':r,'Fixed parameters':fixed_param_aslsd_sbgauss1} for j in range(d)] for i in range(d)]
# mu_names_aslsd_sbgauss1,ker_names_aslsd_sbgauss1=get_mhp_param_names(matrix_model_aslsd_sbgauss1,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2]
# vec_epsilon=[10**-8]
# vec_momentum_1=[0.9*10**0]# Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2,10**-2]
# vec_epsilon=[10**-8,10**-8]
# vec_momentum_1=[0.9*10**0,0.9*10**0]
# vec_momentum_2=[0.999*10**0,0.999*10**0]
# vec_n_iter=[2*10**3,2*10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30,30],[30,30]], 'max_B_spec_h':[[100,100 ],[100,100 ]], 'B_p_total':[[ 10**3,10**3],[ 10**3,10**3]],'n_clusters': [[10,10],[10,10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30,30], 'max_B_spec_h':[100,100] , 'B_p_total': [10**3,10**3],'n_clusters': [10,10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})

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

# residuals_aslsd_sbgauss1=gof_residuals.get_residuals(d,mu_aslsd_sbgauss1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_sbgauss1,sampling=False,sample_size=10**3,seed=1234)

# #       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_sbgauss1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_sbgauss1[0])

# =============================================================================
# aslsd - Gaussian r=1
# =============================================================================
# print('aslsd - Gaussian r=1')
# r=1

# # Model Definition
# matrix_model_aslsd_gauss1=[[{'Type':'Gaussian','r':r,'Fixed parameters':[]} for j in range(d)] for i in range(d)]
# mu_names_aslsd_gauss1,ker_names_aslsd_gauss1=get_mhp_param_names(matrix_model_aslsd_gauss1,d)


# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2,10**-2]
# vec_epsilon=[10**-8,10**-8]
# vec_momentum_1=[0.9*10**0,0.9*10**0]
# vec_momentum_2=[0.999*10**0,0.999*10**0]
# vec_n_iter=[2*10**3,2*10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'EMA weight': [[0.6,0.6],[0.6,0.6]],'n_K': [[1,1],[1,1]],'list_Delta_Q': [[[3*10**4],[3*10**4]],[[3*10**4],[3*10**4]]],'bucket_size': [[1,1],[1,1]], 'h_min':[[1,1 ],[1,1 ]], 'h_max':[[20,20 ],[20,20 ]], 'B_p_total':[[ 10**3,10**3],[ 10**3,10**3]],'n_clusters': [[10,10],[10,10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'EMA weight': [0.6,0.6],'n_K': [1,1],'list_Delta_Q':[ [3*10**4],[3*10**4] ],'bucket_size':[1,1],'h_min':[1,1],'h_max':[20,20], 'B_p_total': [10**3,10**3],'n_clusters': [10,10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})

# fitted_aslsd_gauss1=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_gauss1,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=None,penalty_kwargs=None,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)


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
# # y=matrix_psi_aslsd_gauss1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_gauss1=gof_residuals.get_residuals(d,mu_aslsd_gauss1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_gauss1,sampling=False,sample_size=10**3,seed=1234)

# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_gauss1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_gauss1[0])

# =============================================================================
# aslsd - SB Gaussian r=20
# =============================================================================
# print('aslsd - Gaussian r=20')
# r=20

# matrix_fixed_param_aslsd_sbgauss20=[[np.zeros(2*r) for j in range(d)] for i in range(d)]

# for i in range(d):
#     for j in range(d):

#         array_delta_aslsd_sbgauss20=np.linspace(0.,10.,20)
#         for l in range(r):
#             #   Beta
#             matrix_fixed_param_aslsd_sbgauss20[i][j][2*l]=0.25
        
#             #   Delta
#             matrix_fixed_param_aslsd_sbgauss20[i][j][2*l+1]=array_delta_aslsd_sbgauss20[l]
    
# # Model Definition
# matrix_model_aslsd_gauss20=[[{'Type':'SB Gaussian','r':r,'Fixed parameters':matrix_fixed_param_aslsd_sbgauss20[i][j]} for j in range(d)] for i in range(d)]
# mu_names_aslsd_gauss20,ker_names_aslsd_gauss20=get_mhp_param_names(matrix_model_aslsd_gauss20,d)

# # Learning parameters
# base_seed=1245
# vec_learning_rate=[10**-2,10**-2]
# vec_epsilon=[10**-8,10**-8]
# vec_momentum_1=[0.9*10**0,0.9*10**0]
# vec_momentum_2=[0.999*10**0,0.999*10**0]
# vec_n_iter=[2*10**3,2*10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'EMA weight': [[0.6,0.6],[0.6,0.6]],'n_K': [[1,1],[1,1]],'list_Delta_Q': [[[3*10**4],[3*10**4]],[[3*10**4],[3*10**4]]],'bucket_size': [[1,1],[1,1]], 'h_min':[[1,1 ],[1,1 ]], 'h_max':[[20,20 ],[20,20 ]], 'B_p_total':[[ 10**3,10**3],[ 10**3,10**3]],'n_clusters': [[10,10],[10,10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'EMA weight': [0.6,0.6],'n_K': [1,1],'list_Delta_Q':[ [3*10**4],[3*10**4] ],'bucket_size':[1,1],'h_min':[1,1],'h_max':[20,20], 'B_p_total': [10**3,10**3],'n_clusters': [10,10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})

# fitted_aslsd_gauss20=fit_adam_adastrat(tensor_event_times,T_f,matrix_model_aslsd_gauss20,vec_learning_rate,vec_epsilon,vec_n_iter,upsilon_config,phi_config,psi_config,upsilonzero_config,penalty_type=None,penalty_kwargs=None,base_seed=base_seed,x_0='random',vec_momentum_1=vec_momentum_1,vec_momentum_2=vec_momentum_2,record_updates=False)


# mu_aslsd_gauss20=fitted_aslsd_gauss20['Mu']
# kernel_param_aslsd_gauss20=fitted_aslsd_gauss20['Kernel']


# # Make model for additionnal checks
# matrix_param_aslsd_gauss20=make_matrix_param(matrix_model_aslsd_gauss20)
# param_directory_aslsd_gauss20=get_param_directory(d,matrix_param_aslsd_gauss20)
# phi_aslsd_gauss20,diff_phi_aslsd_gauss20,psi_aslsd_gauss20,diff_psi_aslsd_gauss20,upsilon_aslsd_gauss20,diff_sim_upsilon_aslsd_gauss20,diff_cross_upsilon_aslsd_gauss20=make_model(matrix_model_aslsd_gauss20)

# matrix_phi_aslsd_gauss20=func_compiler.compile_phi(phi_aslsd_gauss20,kernel_param_aslsd_gauss20,d)

# adj_matrix_aslsd_gauss20=l1_norm.make_adjacency_matrix(matrix_model_aslsd_gauss20,kernel_param_aslsd_gauss20,d)
# radius_aslsd_gauss20=l1_norm.get_spectral_radius(adj_matrix_aslsd_gauss20)

# # fig=plt.figure(dpi=300)
# # x_phi_aslsd_gauss20=np.linspace(0,50,10**4)
# # y_phi_aslsd_gauss20=matrix_phi_aslsd_gauss20[0][0](x_phi_aslsd_gauss20)
# # plt.plot(x_phi_aslsd_gauss20,y_phi_aslsd_gauss20)
# # fig.show()


# #   Resiudal analysis

# #       Compute residuals
# psi_aslsd_gauss20=func_maker.make_psi(matrix_model_aslsd_gauss20)
# matrix_psi_aslsd_gauss20=func_compiler.compile_psi(psi_aslsd_gauss20,kernel_param_aslsd_gauss20,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_gauss20[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_gauss20=gof_residuals.get_residuals(d,mu_aslsd_gauss20,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_gauss20,sampling=False,sample_size=10**3,seed=1234)

#       KS test
# gof_residuals.ks_test_residuals_k(residuals_aslsd_gauss20[0])

#       Q-Q plot
# gof_residuals.qq_plot_residuals_k(residuals_aslsd_gauss20[0])

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




# # residuals_aslsd_sbgauss3=gof_residuals.get_residuals(d,mu_aslsd_sbgauss3,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_sbgauss3,sampling=False,sample_size=10**3,seed=1234)
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
# vec_learning_rate=[10**-2,10**-2]
# vec_epsilon=[10**-8,10**-8]
# vec_momentum_1=[0.9*10**0,0.9*10**0]
# vec_momentum_2=[0.999*10**0,0.999*10**0]
# vec_n_iter=[2*10**3,2*10**3]

# # Stratification config
# upsilon_config=[]
# phi_config=[]
# psi_config=[]
# upsilonzero_config=[]

# for k in range(d):
#     # k_upsilon_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     upsilon_config.append({'h_spec': [[30,30],[30,30]], 'max_B_spec_h':[[100,100 ],[100,100 ]], 'B_p_total':[[ 10**3,10**3],[ 10**3,10**3]],'n_clusters': [[10,10],[10,10]]})
#     # k_phi_config={'h_spec': , 'max_B_spec_h': , 'B_p_total':,'n_clusters':  }
#     phi_config.append({'h_spec': [30,30], 'max_B_spec_h':[100,100] , 'B_p_total': [10**3,10**3],'n_clusters': [10,10]})
#     # k_psi_config={'B_spec':,'B_p_total':,'n_buckets':}
#     psi_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})
#     # k_upsilonzero_config={'B_spec':,'B_p_total':,'n_buckets':}
#     upsilonzero_config.append({'B_spec':[100,100],'B_p_total':[1000,1000],'n_buckets':[10,10]})

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


#############################################################################
# #   Tests
# t_test=np.linspace(0,10**3,10**4)
# s_test=np.linspace(0,10**1,10**4)
# list_param=np.array([1.,1.])

# # val_test=psi_aslsd_mixed1(0,0,list_param,t_test)
# # diff_index=0
# # val_test=diff_psi_aslsd_mixed1(0,0,list_param,t_test,diff_index)

# # val_test=upsilon_aslsd_mixed1(0,0,0,list_param,list_param,t_test,1000.)

# diff_index=0
# # print(matrix_model_aslsd_mixed1[0][0]['Index converter'](diff_index) )
# val_test=diff_sim_upsilon_aslsd_mixed1(0,0,list_param,t_test,10**5,diff_index)

# fig=plt.figure(dpi=300)

# plt.plot(t_test,val_test)
# fig.show()
#############################################################################

# #   Resiudal analysis
# #       Compute residuals
# psi_aslsd_mixed1=func_maker.make_psi(matrix_model_aslsd_mixed1)
# matrix_psi_aslsd_mixed1=func_compiler.compile_psi(psi_aslsd_mixed1,kernel_param_aslsd_mixed1,d)

# # fig=plt.figure(dpi=300)
# # x=np.linspace(0,10**1,10**4)
# # y=matrix_psi_aslsd_mixed1[0][0](x)
# # plt.plot(x,y)
# # fig.show()

# residuals_aslsd_mixed1=gof_residuals.get_residuals(d,mu_aslsd_mixed1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_aslsd_mixed1,sampling=False,sample_size=10**3,seed=1234)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_aslsd_mixed1[0])

# #       Q-Q plot
# # gof_residuals.qq_plot_residuals_k(residuals_aslsd_mixed1[0])

# #       Resdiual plot
# # fig=plt.figure(dpi=300)
# # plt.plot(times[:-1],residuals_aslsd_mixed1[0],color='steelblue')
# # fig.show()

# =============================================================================
# Custom model 1
# =============================================================================
# list_r_custom1=[1,1]
# list_fixed_param_custom1=np.array([[0.2,0.15],[.5]],dtype=object)
# list_ker_types_custom1=['SB Gaussian','SB Exponential']

# matrix_model_custom1=[[{'Type':'Mixed','List r':list_r_custom1,'List fixed parameters':list_fixed_param_custom1,'List kernel types':list_ker_types_custom1} for j in range(d)] for i in range(d)]
# matrix_model_custom1[0][0]['List fixed parameters']=np.array([[0.2,0.15],[.5]],dtype=object)
# matrix_param_custom1=make_matrix_param(matrix_model_custom1)
# matrix_model_custom1=make_index_converter(matrix_model_custom1,matrix_param_custom1)


# mu_custom1=np.array([0.3])

# kernel_param_custom1=np.ones( (1,1,2) )
# kernel_param_custom1[0][0][0]=.49
# kernel_param_custom1[0][0][1]=.2

# phi_custom1,diff_phi_custom1,psi_custom1,diff_psi_custom1,upsilon_custom1,diff_sim_upsilon_custom1,diff_cross_upsilon_custom1=make_model(matrix_model_custom1)
# matrix_phi_custom1=func_compiler.compile_phi(phi_custom1,kernel_param_custom1,d)

# #   Resiudal analysis
# #       Compute residuals
# psi_custom1=func_maker.make_psi(matrix_model_custom1)
# matrix_psi_custom1=func_compiler.compile_psi(psi_custom1,kernel_param_custom1,d)

# residuals_custom1=gof_residuals.get_residuals(d,mu_custom1,tensor_event_times,matrix_kappa,matrix_varkappa,list_event_counts,matrix_psi_custom1,sampling=False,sample_size=10**3,seed=1234)
# #       KS test
# # gof_residuals.ks_test_residuals_k(residuals_custom1[0])
# #       QQ plot
# # gof_residuals.qq_plot_residuals_k(residuals_custom1[0])
# #       Resdiual plot
# # fig=plt.figure(dpi=300)
# # plt.plot(times[:-1],residuals_custom1[0],color='steelblue')
# # fig.show()

# =============================================================================
# Comparison
# =============================================================================
model_dict={}
#   Benchmarks
model_dict['WH']={'phi':matrix_phi_wh,'residuals 0':residuals_wh[0],'residuals 1':residuals_wh[1],'color':'firebrick'}
model_dict['SumExp']={'phi':matrix_phi_sumexp,'residuals 0':residuals_sumexp[0],'residuals 1':residuals_sumexp[1],'color':'darkorange'}
# model_dict['Poisson']={'phi':matrix_phi_hompoisson,'residuals 0':residuals_hompoisson[0],'residuals 1':residuals_hompoisson[1],'color':'gold'}

#   Personal
# model_dict['ASLSD Mixed1']={'phi':matrix_phi_aslsd_mixed1,'residuals 0':residuals_aslsd_mixed1[0],'residuals 1':residuals_aslsd_mixed1[1],'color':'steelblue'}
# model_dict['custom1']={'phi':matrix_phi_custom1,'residuals 0':residuals_custom1[0],'color':'cyan'}

#       Exponential
model_dict['ASLSD Exp1']={'phi':matrix_phi_aslsd_exp1,'residuals 0':residuals_aslsd_exp1[0],'residuals 1':residuals_aslsd_exp1[1],'color':'springgreen'}
model_dict['ASLSD Exp3']={'phi':matrix_phi_aslsd_exp3,'residuals 0':residuals_aslsd_exp3[0],'residuals 1':residuals_aslsd_exp3[1],'color':'steelblue'}

#       Gaussian
# model_dict['ASLSD SbGauss1']={'phi':matrix_phi_aslsd_sbgauss1,'residuals 0':residuals_aslsd_sbgauss1[0],'residuals 1':residuals_aslsd_sbgauss1[1],'color':'darkorange'}
# model_dict['ASLSD Gauss1']={'phi':matrix_phi_aslsd_gauss1,'residuals 0':residuals_aslsd_gauss1[0],'residuals 1':residuals_aslsd_gauss1[1],'color':'darkorange'}




# =============================================================================
# Make plots
# =============================================================================

#   Select models to display
def select_model_feature(model_dict,model_names,feature_key):
    return [model_dict[model_name][feature_key] for model_name in model_names]

active_model_names=['WH','SumExp','ASLSD Exp1','ASLSD Exp3']
n_active_models=len(active_model_names)
active_model_phis=select_model_feature(model_dict,active_model_names,'phi')
active_model_residuals0=select_model_feature(model_dict,active_model_names,'residuals 0')
active_model_residuals1=select_model_feature(model_dict,active_model_names,'residuals 1')
active_model_colors=select_model_feature(model_dict,active_model_names,'color')


# #   Phi plots
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=False,dpi=300)
# for i_phi in range(d):
#     for j_phi in range(d):
        

#         x_phi=np.linspace(0,10,10**4)
#         y_phi_models=[None]*n_active_models
#         for m in range(n_active_models):
#             if active_model_names[m]!='Poisson':
#                 y_phi_models[m]=active_model_phis[m][i_phi][j_phi](x_phi)
#                 axs[i_phi,j_phi].plot(x_phi,y_phi_models[m],color=active_model_colors[m],label=active_model_names[m])
#         plt.xlabel(r'$t$')
#         axs[i_phi,j_phi].set(ylabel=r'$\phi_{'+str(i_phi+1)+str(j_phi+1)+'}(t)$')
#         # fig.show()
# axs[1, 0].set(xlabel=r'$t$')
# axs[1, 1].set(xlabel=r'$t$')
# fig.tight_layout()
# #   Legend option 1
# axs[1, 1].legend()
# #   Legend option 2
# # handles, labels = axs[1, 1].get_legend_handles_labels()
# # fig.legend(handles, labels, loc='lower center')



# #   Residual analysis
# #       Q-Q Plots
# gof_residuals.qq_plot_residuals_vect(active_model_residuals0,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=True,markersize=5,dpi=1200)

# #       Probability Plots
# gof_residuals.qq_plot_residuals_vect(active_model_residuals0,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=5,display_line45=False,dpi=1200)

# #       Probability Plots minus y=x
# gof_residuals.qq_plot_residuals_vect(active_model_residuals0,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,dpi=1200)
# gof_residuals.qq_plot_residuals_vect(active_model_residuals1,active_model_colors,proba=False,n_ref_data=10**7,label=True,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,dpi=1200)

# #   Concatenate residual plots
# fig, axs = plt.subplots(2, 2, sharex=False, sharey=False,dpi=300)
# for i_phi in range(d):
#     for j_phi in range(d):
#         put_label=False
#         if i_phi==0 and j_phi==0:
#             put_label=True
#         # label=( (i_phi,j_phi) ==(d-1,d-1))
#         # print('label',label)
#         if i_phi==0:
#             resdiual_vect=active_model_residuals0
#         elif i_phi==1:
#             resdiual_vect=active_model_residuals1
        
#         if j_phi==0:
#             gof_residuals.qq_plot_residuals_vect(resdiual_vect,active_model_colors,proba=False,n_ref_data=10**7,label=put_label,list_labels=active_model_names,log_scale=True,markersize=5,dpi=1200,ax=axs[i_phi,j_phi])
#         elif j_phi==1:
#             gof_residuals.qq_plot_residuals_vect(resdiual_vect,active_model_colors,proba=False,n_ref_data=10**7,label=put_label,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,dpi=1200,ax=axs[i_phi,j_phi])
# fig.tight_layout()
# #   Legend option 1
# # axs[1, 1].legend()
# #   Legend option 2
# # handles, labels = axs[1, 1].get_legend_handles_labels()
# # fig.legend(handles, labels, loc='lower center')


# #       KS test
ks_test_level=0.05
for m in range(n_active_models):
    pvalue=gof_residuals.ks_test_residuals_k(active_model_residuals1[m]).pvalue
    if pvalue>ks_test_level:
        test_outcome='Success'
    else:
        test_outcome='Failure'
    print('Model '+active_model_names[m]+' p-value: ',pvalue,', KS test ',test_outcome)
# =============================================================================
# Save data
# =============================================================================
# Residuals
df_res_0 = pd.DataFrame.from_dict({'SumExp': model_dict['SumExp']['residuals 0'],
                                   'ASLSD Exp1': model_dict['ASLSD Exp1']['residuals 0'],
                                   'ASLSD Exp3': model_dict['ASLSD Exp3']['residuals 0']})
df_res_0.to_csv("saved_data/memetracker2d/residuals_0.csv")

df_res_1 = pd.DataFrame.from_dict({'SumExp': model_dict['SumExp']['residuals 1'],
                                   'ASLSD Exp1': model_dict['ASLSD Exp1']['residuals 1'],
                                   'ASLSD Exp3': model_dict['ASLSD Exp3']['residuals 1']})
df_res_1.to_csv("saved_data/memetracker2d/residuals_1.csv")

# Part II: Fitted kernels
active_model_names=['WH','SumExp','ASLSD Exp1','ASLSD Exp3']
n_active_models=len(active_model_names)
active_model_phis=select_model_feature(model_dict,active_model_names,'phi')
active_model_residuals0=select_model_feature(model_dict,active_model_names,'residuals 0')
active_model_residuals1=select_model_feature(model_dict,active_model_names,'residuals 1')
active_model_colors=select_model_feature(model_dict,active_model_names,'color')
dict_phi = [[{} for j in range(d)] for i in range(d)]
pd_phi = [[None for j in range(d)] for i in range(d)]
for i_phi in range(d):
    for j_phi in range(d):
        x_phi=np.linspace(0,10,10**3)
        y_phi_models=[None]*n_active_models
        for m in range(n_active_models):
            if active_model_names[m]!='Poisson':
                y_phi_models[m]=active_model_phis[m][i_phi][j_phi](x_phi)
                dict_phi[i_phi][j_phi][active_model_names[m]] = y_phi_models[m]
        pd_phi[i_phi][j_phi] = pd.DataFrame.from_dict(dict_phi[i_phi][j_phi] )
        pd_phi[i_phi][j_phi].to_csv("saved_data/memetracker2d/phi_"+str(i_phi)+str(j_phi)+".csv")
                
    
# =============================================================================
# Summary
# =============================================================================
# Part I: Q-Q Plot and rescaled probability plot
active_model_names=['SumExp','ASLSD Exp1','ASLSD Exp3']
n_active_models=len(active_model_names)
active_model_phis=select_model_feature(model_dict,active_model_names,'phi')
active_model_residuals0=select_model_feature(model_dict,active_model_names,'residuals 0')
active_model_residuals1=select_model_feature(model_dict,active_model_names,'residuals 1')
active_model_colors=select_model_feature(model_dict,active_model_names,'color')

ks_critical=1.628

fig, axs = plt.subplots(2, 2, sharex=False, sharey=False,dpi=300)
for i_phi in range(d):
    for j_phi in range(d):
        put_label=False
        if i_phi==0 and j_phi==0:
            put_label=True
        # label=( (i_phi,j_phi) ==(d-1,d-1))
        # print('label',label)
        if i_phi==0:
            resdiual_vect=active_model_residuals0
        elif i_phi==1:
            resdiual_vect=active_model_residuals1
        
        if j_phi==0:
            gof_residuals.qq_plot_residuals_vect(resdiual_vect,active_model_colors,proba=False,n_ref_data=10**7,label=put_label,list_labels=active_model_names,log_scale=True,markersize=2,max_points=500,dpi=300,ax=axs[i_phi,j_phi])
        elif j_phi==1:
            gof_residuals.qq_plot_residuals_vect(resdiual_vect,active_model_colors,proba=False,n_ref_data=10**7,label=put_label,list_labels=active_model_names,log_scale=False,reference='uniform',markersize=1,display_line45=False,substract_yx=True,normalize=True,max_points=500,dpi=300,ax=axs[i_phi,j_phi])
            xmin,xmax = axs[i_phi,j_phi].get_xlim()
            #   KS test level at 0.99
            axs[i_phi,j_phi].hlines(y=ks_critical,xmin=xmin,xmax=xmax,linestyle='dashed', linewidth=1, color='black')
            axs[i_phi,j_phi].hlines(y=-ks_critical,xmin=xmin,xmax=xmax,linestyle='dashed', linewidth=1, color='black')

# fig.tight_layout()
#   Legend option 1
# axs[1, 1].legend()
#   Legend option 2
# handles, labels = axs[1, 1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center')
tikzplotlib.save("tex/memetracker2d_residual_plots.tex")

# Part II: Fitted kernels
active_model_names=['WH','SumExp','ASLSD Exp1','ASLSD Exp3']
n_active_models=len(active_model_names)
active_model_phis=select_model_feature(model_dict,active_model_names,'phi')
active_model_residuals0=select_model_feature(model_dict,active_model_names,'residuals 0')
active_model_residuals1=select_model_feature(model_dict,active_model_names,'residuals 1')
active_model_colors=select_model_feature(model_dict,active_model_names,'color')

fig, axs = plt.subplots(2, 2, sharex=True, sharey=False,dpi=300)
for i_phi in range(d):
    for j_phi in range(d):
        x_phi=np.linspace(0,10,10**3)
        y_phi_models=[None]*n_active_models
        for m in range(n_active_models):
            if active_model_names[m]!='Poisson':
                y_phi_models[m]=active_model_phis[m][i_phi][j_phi](x_phi)
                axs[i_phi,j_phi].plot(x_phi,y_phi_models[m],color=active_model_colors[m],label=active_model_names[m])

        axs[i_phi,j_phi].plot(x_phi,np.zeros(len(x_phi)),linestyle='dashed', linewidth=1, color='grey',alpha=1.)

        plt.xlabel(r'$t$')
        axs[i_phi,j_phi].set(ylabel=r'$\phi_{'+str(i_phi+1)+str(j_phi+1)+'}(t)$')
        # fig.show()
axs[1, 0].set(xlabel=r'$t$')
axs[1, 1].set(xlabel=r'$t$')
#   Legend option 1
axs[1, 1].legend()
#   Legend option 2
# handles, labels = axs[1, 1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='lower center')
tikzplotlib.save("tex/memetracker2d_fitted_kernels.tex")

# Visualize results
t_min = 0.
t_max = 10.
n_samples = 10**3
fig = plt.figure(dpi=300)
x_phi = np.linspace(t_min, t_max, n_samples)
for i, j in itertools.product(range(d), range(d)):
    y_phi = true_mhp.phi[i][j](x_phi, true_ker_param[i][j])
    axs[i, j].plot(x_phi, y_phi, color='darkorange')
axs = mhp.plot_kernels(t_min=0., t_max=10., n_samples=10**3, axs=axs)

# Goodness-of-fit
n_samples = 10**3
fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, dpi=300)
for key in model_dict.keys():
    axs[0] = model_dict[key].qq_plot(0, ax=axs[0])
    axs[1] = model_dict[key].qq_plot(0, ax=axs[1], style='uniform',
                                     substract_yx=True)
