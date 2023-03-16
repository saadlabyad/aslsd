# License: BSD 3 clause

import copy
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tqdm import tqdm

from aslsd.utilities import graphic_tools as gt
from aslsd.utilities import useful_functions as uf


def get_descendants_times(ancestors, mhp, T_f, mu=None,
                          kernel_param=None, rng=None, seed=1234):
    if mu is None:
        mu = mhp.fitted_mu
        if mu is None:
            raise ValueError("Missing value for Mu")
    if kernel_param is None:
        kernel_param = mhp.fitted_ker_param
        if kernel_param is None:
            raise ValueError("Missing value for Kernel parameters")
    mu = np.array(mu)
    d = mhp.d
    offset_gens = [[None for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        offset_gens[i][j] = mhp._kernel_matrix[i][j].make_offset_gen(
            kernel_param[i][j])

    adjacency = mhp.make_adjacency_matrix(kernel_param)
    if rng is None:
        rng = np.random.default_rng(seed)

    branching_ratio = mhp.get_branching_ratio(adjacency=adjacency)
    if branching_ratio >= 1:
        raise ValueError("Cannot simulate from unstable MHP: ",
                         "The branching ratio of this MHP is ",
                         branching_ratio, " > 1.")

    descendants = [[copy.deepcopy(ancestors[i])] for i in range(d)]
    # Step 1. Generate immigrants
    # Number of immigrants
    Nim = rng.poisson(mu*T_f)

    # Location of immigrants
    generations = [[rng.uniform(low=0.0, high=T_f, size=Nim[i])]
                   for i in range(d)]
    # generations is a list such that generations[i][ix_gen] contains
    # the times of events of type i of generation ix_gen

    def sum_generation(L, index):
        return sum([len(L[i][index]) for i in range(d)])

    ix_gen = 1
    #   Step 2. Fill via repeated generations
    while sum_generation(descendants, ix_gen-1):
        for k in range(d):
            generations[k].append(np.array([]))
            descendants[k].append(np.array([]))
        for j in range(d):
            # Simulate the offspring of the "ix_gen-1"th generation of
            checks_j = 1.*np.in1d(generations[j][ix_gen-1],
                                  descendants[j][ix_gen-1])

            # events of type j
            if len(generations[j][ix_gen-1]) > 0:
                for i in range(d):
                    # Set number of offspring
                    Noff = rng.poisson(adjacency[i][j],
                                       size=len(generations[j][ix_gen-1]))
                    # if (i == j) and (Noff.sum() > 0):
                    #     print('Error: i=', i, ' Noff.sum()=', Noff.sum())
                    checks_i_j = checks_j.repeat(Noff)
                    parenttimes = generations[j][ix_gen-1].repeat(Noff)
                    filtered_parenttimes = parenttimes*checks_i_j
                    offsets = offset_gens[i][j](rng, N=Noff.sum())
                    filtered_offsets = offsets*checks_i_j
                    offspringtime = parenttimes + offsets
                    filtered_offspringtime = filtered_parenttimes+filtered_offsets
                    filtered_offspringtime = filtered_offspringtime[filtered_offspringtime != 0.]
                    descendants[i][ix_gen] = np.append(descendants[i][ix_gen], np.array([x for x in filtered_offspringtime if x < T_f]))
                    # print('descendants[i] for i=',i, ': ', descendants[i])
                    generations[i][ix_gen] = np.append(generations[i][ix_gen],
                                               np.array([x for x in offspringtime if x < T_f]))
                    # if (i == j):
                    #     print('New, i=', i, ' j=', j)
                    #     print('checks_i_j', type(checks_i_j))
                    #     print('parenttimes', parenttimes)
                    #     print('filtered_parenttimes', filtered_parenttimes)
                    #     print('offsets', offsets)
                    #     print('filtered_offsets', filtered_offsets)
                    #     print('offspringtime', offspringtime)
                    #     print('filtered_offspringtime', filtered_offspringtime)
                    #     print('descendants', descendants)
            
        ix_gen += 1
        # print([len(descendants[i][ix_gen-1]) for i in range(d)])
        # print('generation', ix_gen)
    list_descendants = [np.array(sorted([x for sublist in descendants[i]
                                         for x in sublist])) for i in range(d)]
    return list_descendants


def get_baseline_events(mhp, T_f, mu=None, kernel_param=None,
                        rng=None, seed=1234):
    if mu is None:
        mu = mhp.fitted_mu
        if mu is None:
            raise ValueError("Missing value for Mu")
    if kernel_param is None:
        kernel_param = mhp.fitted_ker_param
        if kernel_param is None:
            raise ValueError("Missing value for Kernel parameters")
    mu = np.array(mu)
    d = mhp.d
    offset_gens = [[None for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        offset_gens[i][j] = mhp._kernel_matrix[i][j].make_offset_gen(
            kernel_param[i][j])

    adjacency = mhp.make_adjacency_matrix(kernel_param)
    if rng is None:
        rng = np.random.default_rng(seed)

    branching_ratio = mhp.get_branching_ratio(adjacency=adjacency)
    if branching_ratio >= 1:
        raise ValueError("Cannot simulate from unstable MHP: ",
                         "The branching ratio of this MHP is ",
                         branching_ratio, " > 1.")

    # Step 1. Generate immigrants
    # Number of immigrants
    Nim = rng.poisson(mu*T_f)

    # Immigrants
    immigrants = [sorted(list(rng.uniform(low=0.0, high=T_f, size=Nim[i]))) for i in range(d)]
    immigrants = [np.array(immigrants[i]) for i in range(d)]
    return immigrants


def get_relaxtion_time(d, t_ev, descendants):
    lifespan = [None]*d
    for j in range(d):
        L = descendants[j]
        if len(L) == 0:
            lifespan[j] = 0.
        else:
            lifespan[j] = np.max(L)
    relaxation_time = max(lifespan)-t_ev
    return relaxation_time
