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
from aslsd.utilities import useful_statistics as us


# Simulation
def get_descendants_times(mhp, ref_times, T_f, T_i=0., history=None, mu=None,
                          kernel_param=None, rng=None, seed=1234,
                          verbose=False, return_sim=False):
    """
    Simulate a path of the MHP.

    Parameters
    ----------
    T_f : `float`
        Terminal time.
    mu : `numpy.ndarray`, optional
        Vector of baseline parameters. The default is None, in that case
        fitted baseline parameters will be used if they are stored in the
        corresponding attribute of the MHP object.
    kernel_param : `numpy.ndarray`, optional
        Matrix of kernel parameters. The default is None, in that case
        fitted kernel parameters will be used if they are stored in the
        corresponding attribute of the MHP object.
    seed : `int`, optional
        Seed for the random number generator. The default is 1234.
    verbose : `bool`, optional
        If True, print progression information. The default is False.

    Raises
    ------
    ValueError
        Raise an error if the baseline or the kernel parameters are not
        specified and there is no fitted baseline or kernel parameters
        saved as an atrribute.

    Returns
    -------
    list_times : `list` of `numpy.ndarray`
        List of simulated jump times for each dimension.

    """
    d = mhp.d
    # RNG
    rng = us.make_rng(rng=rng, seed=seed)
    # Prepare parameters
    mu, kernel_param = mhp.load_param(mu=mu, kernel_param=kernel_param)
    # Adjacency matrix
    adjacency = mhp.make_adjacency_matrix(kernel_param)
    branching_ratio = mhp.get_branching_ratio(adjacency=adjacency)
    if branching_ratio >= 1:
        raise ValueError("Cannot simulate from unstable MHP: ",
                         "The branching ratio of this MHP is ",
                         branching_ratio, " > 1.")
    # Offset generators
    offset_gens = [[None for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        offset_gens[i][j] = mhp._kernel_matrix[i][j].make_offset_gen(
            kernel_param[i][j])

    # Adjust history
    if history is None:
        history = [[] for i in range(d)]

    # Adjust Reference times
    n_ref_times = [None]*d
    for i in range(d):
        if ref_times[i] is None:
            ref_times[i] = []
        n_ref_times[i] = len(ref_times[i])

    # Start simulation
    if verbose:
        print('Simulating events...')

    # Step 0.1. Intialise Generations
    # generations is a list such that generations[i][ix_gen] contains
    # the times of events of type i of generation ix_gen
    generations = [None]*d
    for i in range(d):
        if len(history[i]) > 0:
            generations[i] = [copy.deepcopy(np.array(history[i]))]
        else:
            generations[i] = [[]]

    # Step 0.2. Intialise ref_generations
    # ref_generations is a list such that generations[j][m][i][ix_gen] contains
    # the times of events of type i of generation ix_gen of descendants of
    # ref_times[i][m]
    ref_generations = [None]*d
    for j in range(d):
        if n_ref_times[j] > 0:
            ref_generations[j] = [None]*n_ref_times[j]
            for m in range(n_ref_times[j]):
                ref_generations[j][m] = [None]*d
                for i in range(d):
                    ref_generations[j][m][i] = [[]]
                ref_generations[j][m][j][0].append(ref_times[j][m])
        else:
            ref_generations[j] = [[]]

    # Step 1. Generate immigrants
    # Number of immigrants
    Nim = rng.poisson(mu*(T_f-T_i))
    # Location of immigrants
    immigrants = [None]*d
    for i in range(d):
        immigrants[i] = rng.uniform(low=T_i, high=T_f, size=Nim[i])
        if len(generations[i][0]) == 0:
            generations[i][0] = copy.deepcopy(immigrants[i])
        else:
            generations[i][0] = np.concatenate((generations[i][0],
                                                immigrants[i]))

    #   Step 2. Fill via repeated generations
    def sum_generation(L, index):
        return sum([len(L[i][index]) for i in range(d)])

    ix_gen = 1
    ix_gen_ref = [[1 for m in range(n_ref_times[k])] for k in range(d)]
    while sum_generation(generations, ix_gen-1):
        for k in range(d):
            generations[k].append(np.array([]))
        update_ref = True
        for j in range(d):
            # Simulate the offspring of the "ix_gen-1"th generation of
            # events of type j
            if len(generations[j][ix_gen-1]) > 0:
                checks_j = [[1.*np.in1d(generations[j][ix_gen-1],
                                        ref_generations[k][m][j][ix_gen_ref[k][m]-1])
                             for m in range(n_ref_times[k])] for k in range(d)]
                for i in range(d):
                    
                    # Set number of offspring
                    Noff = rng.poisson(adjacency[i][j],
                                       size=len(generations[j][ix_gen-1]))
                    parenttimes = generations[j][ix_gen-1].repeat(Noff)
                    offsets = offset_gens[i][j](rng, N=Noff.sum())
                    offspringtime = parenttimes + offsets
                    generations[i][ix_gen] = np.append(
                        generations[i][ix_gen],
                        np.array([x for x in offspringtime
                                  if (x < T_f) and (x > T_i)]))
                    # Offspring of reference times
                    for k in range(d):
                        for m in range(n_ref_times[k]):
                            checks_i_j = checks_j[k][m].repeat(Noff)
                            # filtered_parenttimes = parenttimes*checks_i_j
                            # filtered_offsets = offsets*checks_i_j
                            # filtered_offspringtime = filtered_parenttimes+filtered_offsets
                            # filtered_offspringtime = filtered_offspringtime[filtered_offspringtime != 0.]
                            ref_ixs = np.where((checks_i_j == 1.))[0]
                            # print(Noff)
                            # Increment ix_gen_ref[k][m] only once, the first
                            # time it gets hit i.e. 
                            if (len(ref_ixs) > 0) and update_ref:
                                for p in range(d):
                                    ref_generations[k][m][p].append(np.array([]))
                                ix_gen_ref[k][m] += 1
                                update_ref = False
                            if len(ref_ixs) > 0:
                                filtered_offspringtime = offspringtime[ref_ixs]
                                ref_generations[k][m][i][ix_gen_ref[k][m]-1] = np.append(ref_generations[k][m][i][ix_gen_ref[k][m]-1],
                                    np.array([x for x in filtered_offspringtime
                                              if (x < T_f) and (x > T_i)]))

        ix_gen += 1
    list_times = [np.array(sorted([x for sublist in generations[i]
                                   for x in sublist])) for i in range(d)]
    # Filtering
    for i in range(d):
        list_times[i] = list_times[i][list_times[i] > T_i]
    # Descendants
    list_descendants = [None]*d
    for k in range(d):
        list_descendants[k] = [None]*n_ref_times[k]
        for m in range(n_ref_times[k]):
            list_descendants[k][m] = [np.array(sorted(
                [x for sublist in ref_generations[k][m][i]
                 for x in sublist])) for i in range(d)]
            for i in range(d):
                if len(list_descendants[k][m][i]) > 0:
                    list_descendants[k][m][i] = list_descendants[k][m][i][
                        list_descendants[k][m][i] > T_i]
            if len(list_descendants[k][m][k]) > 0:
                list_descendants[k][m][k] = list_descendants[k][m][k][
                    list_descendants[k][m][k] > ref_times[k][m]]
    # Simulation complete
    if verbose:
        n_tot = sum([len(L) for L in list_times])
        print('Simulation Complete, ', n_tot, ' events simulated.')
    # Return
    if return_sim:
        return list_times, list_descendants
    else:
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
    lifespan = np.zeros(d)
    for j in range(d):
        L = descendants[j]
        if len(L) == 0:
            lifespan[j] = 0.
        else:
            lifespan[j] = np.max(L)
    relaxation_time = np.max(lifespan)-t_ev
    return relaxation_time
