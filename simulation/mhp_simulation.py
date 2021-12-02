# License: BSD 3 clause

import numpy as np
import random
from numpy import log,exp
import multiprocessing as mp
from scipy.stats import norm,poisson,uniform
import bisect

import useful_functions as uf
import l1_norm


# =============================================================================
# Generate multi-path
# =============================================================================
def simu_multipath_1d(sim_func,path_res,t_res,x_min,x_max,disc_type='log',base_seed=1234):
    np.seed=base_seed
    seed=base_seed
    
    if disc_type=='log':
        T_f = 10**x_max
    elif disc_type=='linear':
        T_f=x_max
    
    list_Tf=uf.discretize_space(x_min,x_max,t_res,disc_type)
    
    list_paths=[[[] for j in range(path_res)] for i in range(t_res)]

    for j in range(path_res):
        seed+=np.random.randint(10**4)
        tensor_event_times=sim_func(T_f,seed)
        for i in range(t_res):
            local_Tf=list_Tf[i]
            n_f=bisect.bisect_left(tensor_event_times[0],local_Tf)-1
            #Check
    #        if n_f<len(tensor_event_times[0])-1:
    #            assert (tensor_event_times[0][n_f]<=local_Tf) and (tensor_event_times[0][n_f+1]>local_Tf)            
            list_paths[i][j]=[tensor_event_times[0][:n_f+1]]
    return list_Tf,list_paths

def simu_multipath(d,sim_func,path_res,t_res,x_min,x_max,disc_type='log',base_seed=1234):
    np.seed=base_seed
    seed=base_seed
    
    if disc_type=='log':
        T_f = 10**x_max
    elif disc_type=='linear':
        T_f=x_max
    
    list_Tf=uf.discretize_space(x_min,x_max,t_res,disc_type)
    
    list_paths=[[[] for j in range(path_res)] for i in range(t_res)]

    for j in range(path_res):
        seed+=np.random.randint(10**4)
        tensor_event_times=sim_func(T_f,seed)
        for i in range(t_res):
            local_Tf=list_Tf[i]
            list_n_f=[ bisect.bisect_left(tensor_event_times[index_dim],local_Tf)-1 for index_dim in range(d)]

            #Check
    #        if n_f<len(tensor_event_times[0])-1:
    #            assert (tensor_event_times[0][n_f]<=local_Tf) and (tensor_event_times[0][n_f+1]>local_Tf)            
            list_paths[i][j]=[tensor_event_times[0][:list_n_f[index_dim]+1] for index_dim in range(d)]
    return list_Tf,list_paths
