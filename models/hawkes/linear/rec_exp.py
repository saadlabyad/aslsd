# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.stats.marks.void_mark import VoidMark
from aslsd.functionals.baselines.baseline import BaselineModel
from aslsd.functionals.baselines.basis_baselines.basis_baseline_constant import ConstantBaseline
from aslsd.functionals.impact_functions.impact_function import ImpactFunction
from aslsd.functionals.impact_functions.basis_impacts.basis_impact_constant import ConstantImpact
from aslsd.functionals.kernels.basis_kernels.\
    basis_kernel_exponential import ExponentialKernel
from aslsd.stats.residual_analysis import goodness_of_fit as gof
from aslsd.stats.events.process_path import ProcessPath
from aslsd.optimize.optim_logging.optim_logger import OptimLogger
from aslsd.optimize.solvers.adam import ADAM
from aslsd.optimize.solvers.solver import Solver
from aslsd.utilities import useful_functions as uf
from aslsd.utilities import useful_statistics as us
from aslsd.utilities import graphic_tools as gt


# =============================================================================
# Exponential Markov tricks
# =============================================================================
def update_exp_sum(S_m, m, i, j, beta, kappa, list_times, U):
    # S^{(m)}_{ijk,q} := \sum_{n=1}^{\kappa(j,i,m)} \exp(-beta_{kjq} (t^i_m-t^j_n)) U_n
    # return S^{(m+1)}_{ijk,q} given S^{(m)}_{ijk,q}
    t_m1 = list_times[i][m+1]
    t_m = list_times[i][m]
    exp_term = np.exp(-beta*(t_m1-t_m))*S_m
    if kappa[j][i][m] == kappa[j][i][m+1]:
        return exp_term
    else:
        ix_start = kappa[j][i][m]+1
        ix_end = kappa[j][i][m+1]
        U_n = U[ix_start:ix_end+1]
        t_diff = t_m1-list_times[j][ix_start:ix_end+1]
        return exp_term+np.sum(np.exp(-beta*t_diff)*U_n)


def get_exp_sum(i, j, beta, varpi, kappa, list_times, U):
    res = np.zeros(len(list_times[i]))
    # Initialise
    t_m1 = list_times[i][varpi[i][j][1]]
    t_diff = t_m1-list_times[j][:kappa[j][i][varpi[i][j][1]]+1]
    U_n = U[:kappa[j][i][varpi[i][j][1]]+1]
    S_m = np.sum(np.exp(-beta*t_diff)*U_n)
    res[varpi[i][j][1]] = S_m
    # recurrence
    for m in range(varpi[i][j][1]+1, len(list_times[i])):
        S_m = update_exp_sum(S_m, m-1, i, j, beta, kappa, list_times, U)
        res[m] = S_m
    return res


def get_exp_sum_reccurence_lag_h(beta, kappa, varpi, list_times, i, j, h):
    #   Compute \sum_{n=varpi[i][j][h]}^{N^i_T} \exp(-beta (t_m-t_{\kappa(j,i,m)-h+1}))
    indices_m = [m for m in range(varpi[i][j][h], len(list_times[i]))]
    indices_n = [kappa[j][i][m]-h+1 for m in indices_m]
    return np.sum(np.exp(-beta*(list_times[i][indices_m]-list_times[j][indices_n])))


# =============================================================================
# Exponential variables
# =============================================================================
# b_ijk_pq
def get_b_ijk_pq(beta_kip, beta_kjq):
    b_ijk_pq = (beta_kip*beta_kjq)/(beta_kip+beta_kjq)
    return b_ijk_pq


def get_b_k_tensor(k, kernel_matrix, ker_param_k):
    d = len(kernel_matrix)
    b_k = [[[[0.
              for q in range(kernel_matrix[k][j].n_basis_ker)]
             for p in range(kernel_matrix[k][i].n_basis_ker)]
            for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        kernel_ki = kernel_matrix[k][i]
        ker_param_ki = ker_param_k[i]
        ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
        r_ki = kernel_ki.n_basis_ker
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for p, q in itertools.product(range(r_ki), range(r_kj)):
            beta_kip = ker_vars_ki[2*p+1]
            beta_kjq = ker_vars_kj[2*q+1]
            b_ijk_pq = get_b_ijk_pq(beta_kip, beta_kjq)
            b_k[i][j][p][q] = b_ijk_pq
    return b_k


# b_1_ijk_pq
def get_b_1_ijk_pq(beta_kip, beta_kjq):
    b_1_ijk_pq = beta_kip/(beta_kip+beta_kjq)
    return b_1_ijk_pq


def get_b_1_k_tensor(k, kernel_matrix, ker_param_k):
    d = len(kernel_matrix)
    b_1_k = [[[[0.
              for q in range(kernel_matrix[k][j].n_basis_ker)]
             for p in range(kernel_matrix[k][i].n_basis_ker)]
              for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        kernel_ki = kernel_matrix[k][i]
        ker_param_ki = ker_param_k[i]
        ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
        r_ki = kernel_ki.n_basis_ker
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for p, q in itertools.product(range(r_ki), range(r_kj)):
            beta_kip = ker_vars_ki[2*p+1]
            beta_kjq = ker_vars_kj[2*q+1]
            b_1_ijk_pq = get_b_1_ijk_pq(beta_kip, beta_kjq)
            b_1_k[i][j][p][q] = b_1_ijk_pq
    return b_1_k


# b_1_ijk_pq
def get_b_2_ijk_pq(beta_kip, beta_kjq):
    b_2_ijk_pq = beta_kjq/(beta_kip+beta_kjq)
    return b_2_ijk_pq


def get_b_2_k_tensor(k, kernel_matrix, ker_param_k):
    d = len(kernel_matrix)
    b_2_k = [[[[0.
              for q in range(kernel_matrix[k][j].n_basis_ker)]
             for p in range(kernel_matrix[k][i].n_basis_ker)]
              for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        kernel_ki = kernel_matrix[k][i]
        ker_param_ki = ker_param_k[i]
        ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
        r_ki = kernel_ki.n_basis_ker
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for p, q in itertools.product(range(r_ki), range(r_kj)):
            beta_kip = ker_vars_ki[2*p+1]
            beta_kjq = ker_vars_kj[2*q+1]
            b_2_ijk_pq = get_b_2_ijk_pq(beta_kip, beta_kjq)
            b_2_k[i][j][p][q] = b_2_ijk_pq
    return b_2_k


# epsilon_ijk_pq
def get_epsilon_ijk_pq(i, beta_kip, beta_kjq, list_times2end):
    epsilon_ijk_pq = np.exp(-(beta_kip+beta_kjq)*list_times2end[i])
    return epsilon_ijk_pq


def get_epsilon_k_tensor(k, kernel_matrix, ker_param_k, list_times2end):
    d = len(kernel_matrix)
    epsilon_k = [[[[0.
                    for q in range(kernel_matrix[k][j].n_basis_ker)]
                   for p in range(kernel_matrix[k][i].n_basis_ker)]
                  for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        kernel_ki = kernel_matrix[k][i]
        ker_param_ki = ker_param_k[i]
        ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
        r_ki = kernel_ki.n_basis_ker
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for p, q in itertools.product(range(r_ki), range(r_kj)):
            beta_kip = ker_vars_ki[2*p+1]
            beta_kjq = ker_vars_kj[2*q+1]
            epsilon_ijk_pq = get_epsilon_ijk_pq(i, beta_kip, beta_kjq,
                                                list_times2end)
            epsilon_k[i][j][p][q] = epsilon_ijk_pq
    return epsilon_k


# =============================================================================
# Impact quantities
# =============================================================================
# impact_ki
def get_impact_ki(k, i, impact, imp_param_k, list_marks):
    impact_ki = impact[k][i](list_marks[i], imp_param_k[i])
    return impact_ki


def get_impact_k_tensor(k, impact, imp_param_k, list_marks):
    d = len(impact)
    impact_k = [get_impact_ki(k, i, impact, imp_param_k, list_marks)
                for i in range(d)]
    return impact_k


# twimpact_ki
def get_twimpact_ki(i, impact_ki, list_times):
    twimpact_ki = impact_ki*list_times[i]
    return twimpact_ki


def get_twimpact_k_tensor(k, impact_k, list_times):
    d = len(list_times)
    twimpact_k = [get_twimpact_ki(i, impact_k[i], list_times)
                  for i in range(d)]
    return twimpact_k


# S_imp_ijk_q
def get_S_imp_ijk_q(i, j, beta_kjq, varpi, kappa, list_times, impact_kj):
    S_imp_ijk_q = get_exp_sum(i, j, beta_kjq, varpi, kappa, list_times,
                              impact_kj)
    return S_imp_ijk_q


def get_S_imp_k_tensor(k, kernel_matrix, ker_param_k,
                       varpi, kappa, list_times, impact_k):
    d = len(kernel_matrix)
    S_imp_k = [[[0. for q in range(kernel_matrix[k][j].n_basis_ker)]
                for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        impact_kj = impact_k[j]
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for q in range(r_kj):
            beta_kjq = ker_vars_kj[2*q+1]
            S_imp_ijk_q = get_S_imp_ijk_q(i, j, beta_kjq, varpi, kappa,
                                          list_times, impact_kj)
            S_imp_k[i][j][q] = S_imp_ijk_q
    return S_imp_k


# S_twimp_ijk_q
def get_S_twimp_ijk_q(i, j, beta_kjq, varpi, kappa, list_times, twimpact_kj):
    S_twimp_ijk_q = get_exp_sum(i, j, beta_kjq, varpi, kappa, list_times,
                                twimpact_kj)
    return S_twimp_ijk_q


def get_S_twimp_k_tensor(k, kernel_matrix, ker_param_k,
                         varpi, kappa, list_times, twimpact_k):
    d = len(kernel_matrix)
    S_twimp_k = [[[0. for q in range(kernel_matrix[k][j].n_basis_ker)]
                  for j in range(d)] for i in range(d)]
    for i, j in itertools.product(range(d), range(d)):
        twimpact_kj = twimpact_k[j]
        kernel_kj = kernel_matrix[k][j]
        ker_param_kj = ker_param_k[j]
        ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
        r_kj = kernel_kj.n_basis_ker
        for q in range(r_kj):
            beta_kjq = ker_vars_kj[2*q+1]
            S_twimp_ijk_q = get_S_twimp_ijk_q(i, j, beta_kjq, varpi, kappa,
                                              list_times, twimpact_kj)
            S_twimp_k[i][j][q] = S_twimp_ijk_q
    return S_twimp_k


# =============================================================================
# Psi
# =============================================================================
def get_sum_psi_ki_p(i, omega_kip, beta_kip, list_times2end):
    n_i = len(list_times2end[i])
    exp_term = np.exp(-beta_kip*list_times2end[i])
    res = omega_kip*(n_i-np.sum(exp_term))
    return res


def get_diff_omega_sum_psi_ki_p(i, beta_kip, list_times2end):
    n_i = len(list_times2end[i])
    exp_term = np.exp(-beta_kip*list_times2end[i])
    res = n_i-np.sum(exp_term)
    return res


def get_diff_beta_sum_psi_ki_p(i, omega_kip, beta_kip, list_times2end):
    exp_term = np.exp(-beta_kip*list_times2end[i])
    prod_term = list_times2end[i]*exp_term
    res = omega_kip*np.sum(prod_term)
    return res


# =============================================================================
# K - Constant Baseline case
# =============================================================================
def get_sum_K_ki_p_constbase(i, omega_kip, beta_kip, mu_k, impact_ki,
                             list_times2end):
    prod = (1.-np.exp(-beta_kip*list_times2end[i]))*impact_ki
    res = mu_k*omega_kip*np.sum(prod)
    return res


# Derivatives wrt Kernel
def get_diff_omega_sum_K_ki_p_constbase(i, beta_kip, mu_k, impact_ki,
                                        list_times2end):
    prod = (1.-np.exp(-beta_kip*list_times2end[i]))*impact_ki
    res = mu_k*np.sum(prod)
    return res


def get_diff_beta_sum_K_ki_p_constbase(i, omega_kip, beta_kip, mu_k, impact_ki,
                                       list_times2end):
    prod = list_times2end[i]*np.exp(-beta_kip*list_times2end[i])*impact_ki
    res = mu_k*omega_kip*np.sum(prod)
    return res


# Derivative wrt baseline
def get_diff_mu_sum_K_ki_p_constbase(i, omega_kip, beta_kip, impact_ki,
                                     list_times2end):
    prod = (1.-np.exp(-beta_kip*list_times2end[i]))*impact_ki
    res = omega_kip*np.sum(prod)
    return res

# =============================================================================
# K - Piecewise Constant Baseline case
# =============================================================================

# =============================================================================
# Upsilon zero
# =============================================================================
def get_sum_upsilonzero_ki_pq(omega_kip, omega_kiq, b_iik_pq, impact_ki,
                              epsilon_iik_pq):
    prod = (impact_ki**2)*(1.-epsilon_iik_pq)
    res = omega_kip*omega_kiq*b_iik_pq*np.sum(prod)
    return res


# Diff upsilonzero cross
def get_diff_omegakip_sum_upsilonzero_ki_pq(omega_kiq, b_iik_pq, impact_ki,
                                            epsilon_iik_pq):
    prod = (impact_ki**2)*(1.-epsilon_iik_pq)
    res = omega_kiq*b_iik_pq*np.sum(prod)
    return res


def get_diff_omegakiq_sum_upsilonzero_ki_pq(omega_kip, b_iik_pq,
                                            impact_ki, epsilon_iik_pq):
    prod = (impact_ki**2)*(1.-epsilon_iik_pq)
    res = omega_kip*b_iik_pq*np.sum(prod)
    return res


def get_diff_betakip_sum_upsilonzero_ki_pq(i, omega_kip, beta_kip, omega_kiq,
                                           b_2_iik_pq, impact_ki,
                                           epsilon_iik_pq,
                                           list_times2end):
    times = b_2_iik_pq+(list_times2end[i]*beta_kip-b_2_iik_pq)*epsilon_iik_pq
    prod = (impact_ki**2)*times
    res = omega_kip*omega_kiq*b_2_iik_pq*np.sum(prod)
    return res


def get_diff_betakiq_sum_upsilonzero_ki_pq(i, omega_kip, omega_kiq, beta_kiq,
                                           b_1_iik_pq, impact_ki, epsilon_iik_pq,
                                           list_times2end):
    times = b_1_iik_pq+(list_times2end[i]*beta_kiq-b_1_iik_pq)*epsilon_iik_pq
    prod = (impact_ki**2)*times
    res = omega_kip*omega_kiq*b_1_iik_pq*np.sum(prod)
    return res


# Diff upsilonzero self
def get_diff_omega_sum_upsilonzero_ki_pp(omega_kip, beta_kip, impact_ki,
                                         epsilon_iik_pp):
    prod = (impact_ki**2)*(1.-epsilon_iik_pp)
    res = omega_kip*beta_kip*np.sum(prod)
    return res


def get_diff_beta_sum_upsilonzero_ki_pp(i, omega_kip, beta_kip, impact_ki,
                                        epsilon_iik_pp, list_times2end):
    times = 1.+(2.*beta_kip*list_times2end[i]-1.)*epsilon_iik_pp
    prod = (impact_ki**2)*times
    res = 0.5*omega_kip**2*np.sum(prod)
    return res


# =============================================================================
# Phi terms
# =============================================================================
def get_sum_phi_kj_q(k, j, omega_kjq, beta_kjq, S_imp_kjk_q, varpi):
    min_m = varpi[k][j][1]
    res = omega_kjq*beta_kjq*np.sum(S_imp_kjk_q[min_m:])
    return res


def get_diff_omega_sum_phi_kj_q(k, j, beta_kjq, S_imp_kjk_q, varpi):
    min_m = varpi[k][j][1]
    res = beta_kjq*np.sum(S_imp_kjk_q[min_m:])
    return res


def get_diff_beta_sum_phi_kj_q(k, j, omega_kjq, beta_kjq, S_imp_kjk_q,
                               S_twimp_kjk_q, varpi, list_times):
    min_m = varpi[k][j][1]
    # Term 1
    times = list_times[k][min_m:]
    prod_1 = (1.-beta_kjq*times)*S_imp_kjk_q[min_m:]
    res_1 = omega_kjq*np.sum(prod_1)
    # Term 2
    res_2 = omega_kjq*beta_kjq*np.sum(S_twimp_kjk_q[min_m:])
    # Add
    res = res_1+res_2
    return res


# =============================================================================
# Upsilon terms
# =============================================================================
def get_sum_upsilon_ijk_pq(i, j, omega_kip, omega_kjq, b_ijk_pq,
                           impact_ki, S_imp_ijk_q, epsilon_ijk_pq, varpi):
    min_m = varpi[i][j][1]
    prod = impact_ki[min_m:]*(1.-epsilon_ijk_pq[min_m:])*S_imp_ijk_q[min_m:]
    res = omega_kip*omega_kjq*b_ijk_pq*np.sum(prod)
    return res


# Diff Upsilon Cross
def get_diff_omega_i_sum_upsilon_ijk_pq(i, j, omega_kjq, b_ijk_pq,
                                        impact_ki, S_imp_ijk_q, epsilon_ijk_pq,
                                        varpi):
    min_m = varpi[i][j][1]
    prod = impact_ki[min_m:]*(1.-epsilon_ijk_pq[min_m:])*S_imp_ijk_q[min_m:]
    res = omega_kjq*b_ijk_pq*np.sum(prod)
    return res


def get_diff_omega_j_sum_upsilon_ijk_pq(i, j, omega_kip, b_ijk_pq,
                                        impact_ki, S_imp_ijk_q, epsilon_ijk_pq,
                                        varpi):
    min_m = varpi[i][j][1]
    prod = impact_ki[min_m:]*(1.-epsilon_ijk_pq[min_m:])*S_imp_ijk_q[min_m:]
    res = omega_kip*b_ijk_pq*np.sum(prod)
    return res


def get_diff_beta_i_sum_upsilon_ijk_pq(i, j, omega_kip, beta_kip, omega_kjq,
                                       b_2_ijk_pq, impact_ki, S_imp_ijk_q,
                                       epsilon_ijk_pq, varpi, list_times, T_f):
    min_m = varpi[i][j][1]
    t_term = b_2_ijk_pq+(beta_kip*(T_f-list_times[i][min_m:])-b_2_ijk_pq)*epsilon_ijk_pq[min_m:]
    prod = impact_ki[min_m:]*t_term*S_imp_ijk_q[min_m:]
    res = omega_kip*omega_kjq*b_2_ijk_pq*np.sum(prod)
    return res


def get_diff_beta_j_sum_upsilon_ijk_pq(i, j, omega_kip, beta_kip, omega_kjq,
                                       beta_kjq, b_ijk_pq, b_1_ijk_pq,
                                       impact_ki, S_imp_ijk_q, S_twimp_ijk_q,
                                       epsilon_ijk_pq, varpi, list_times, T_f):
    min_m = varpi[i][j][1]
    # Term 1
    times = list_times[i][min_m:]
    t_term = b_1_ijk_pq-beta_kjq*times+(T_f*beta_kjq-b_1_ijk_pq)*epsilon_ijk_pq[min_m:]
    prod_1 = impact_ki[min_m:]*t_term*S_imp_ijk_q[min_m:]*S_imp_ijk_q[min_m:]
    res_1 = omega_kip*omega_kjq*b_1_ijk_pq*np.sum(prod_1)
    # Term 2
    prod_2 = impact_ki[min_m:]*(1.-epsilon_ijk_pq[min_m:])*S_twimp_ijk_q[min_m:]
    res_2 = omega_kip*omega_kjq*beta_kjq*b_ijk_pq*np.sum(prod_2)
    # Add
    res = res_1+res_2
    return res


# Diff Upsilon self
def get_diff_omega_sum_upsilon_iik_pp(i, omega_kip, beta_kip,
                                      impact_ki, S_imp_iik_p, epsilon_iik_pp,
                                      varpi):
    min_m = varpi[i][i][1]
    prod = impact_ki[min_m:]*(1.-epsilon_iik_pp[min_m:])*S_imp_iik_p[min_m:]
    res = omega_kip*beta_kip*np.sum(prod)
    return res


def get_diff_beta_sum_upsilon_iik_pp(i, omega_kip, beta_kip,
                                     impact_ki, S_imp_iik_p, S_twimp_iik_p,
                                     epsilon_iik_pp,
                                     varpi, list_times, T_f):
    min_m = varpi[i][i][1]
    times = list_times[i][min_m:]
    # Normal term
    term_1 = 1.-beta_kip*times+(beta_kip*(2.*T_f-times)-1.)*epsilon_iik_pp[min_m:]
    prod_1 = impact_ki[min_m:]*term_1*S_imp_iik_p[min_m:]
    res_1 = 0.5*(omega_kip**2)*np.sum(prod_1)
    # Time weighted term
    prod_2 = impact_ki[min_m:]*(1.-epsilon_iik_pp[min_m:])*S_twimp_iik_p[min_m:]
    res_2 = 0.5*(omega_kip**2)*beta_kip*np.sum(prod_2)
    # Add them
    res = res_1+res_2
    return res


# =============================================================================
# Class Definition: Recurrent Exponential
# =============================================================================
class RecurrentExponential:
    def __init__(self, _kernel_matrix, _baselines_vec=None,
                 _impact_matrix=None, vec_marks=None,
                 index_from_one=False, K_from_baseline=True,
                 precomp_imp=None, precomp_S=None, precomp_eps=None,
                 max_array_size=10**10):
        """
        Constructor of objects of class MHP.

        Parameters
        ----------
        _kernel_matrix : `list` of `list` of `KernelModel`
            Matrix of kernel models.

        """
        self.clear_fit()
        self.d = len(_kernel_matrix)
        self.is_fitted = False
        self.index_from_one = index_from_one
        self.K_from_baseline = K_from_baseline
        self.baselines_vec = _baselines_vec
        self.kernel_matrix = _kernel_matrix
        self.impact_matrix = _impact_matrix

        vec_marks_ = self.make_marks(vec_marks)
        self.vec_marks = vec_marks_

        # Precomp
        self.max_array_size = max_array_size
        self.precomp_imp = precomp_imp
        self.precomp_S = precomp_S
        self.precomp_eps = precomp_eps

    # Marks
    def make_marks(self, vec_marks):
        d = self.d
        if vec_marks is None:
            vec_marks_ = [VoidMark() for i in range(d)]
        else:
            if uf.is_array(vec_marks):
                vec_marks_ = copy.deepcopy(vec_marks)
            else:
                vec_marks_ = [copy.deepcopy(vec_marks) for i in range(d)]
        return vec_marks_

    def make_expected_impact(self, impact_param=None):
        d = self.d
        if impact_param is None:
            impact_param = self.fitted_imp_param
            if impact_param is None:
                raise ValueError("Missing value for Impact parameters")
        expected_impact_matrix = np.zeros((d, d))
        for i, j in itertools.product(range(d), range(d)):
            impact = self._impact_matrix[i][j]
            expected_impact_matrix[i][j] = self.vec_marks[j].get_expected_impact(impact, impact_param[i][j])
        return expected_impact_matrix

    # Baselines vector
    @property
    def baselines_vec(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._baselines_vec

    @baselines_vec.setter
    def baselines_vec(self, L):
        # Initialisation if L is None
        if L is None:
            L = [BaselineModel([ConstantBaseline()]) for i in range(self.d)]

        if hasattr(self, 'vector_n_param_mu'):
            old_vector_n_param_mu = copy.deepcopy(self.vector_n_param_mu)
        else:
            old_vector_n_param_mu = None
        # Baseline attributes only
        self._baselines_vec = copy.deepcopy(L)
        self.d = len(L)
        self.vector_n_param_mu = self.get_vector_n_param_mu()
        # Parameters names
        self.mu_param_names = self.get_mu_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.mu_param_lower_bounds = self.get_mu_param_lower_bounds()
        self.mu_param_upper_bounds = self.get_mu_param_upper_bounds()

        # General updates
        if hasattr(self, '_kernel_matrix'):
            if (old_vector_n_param_mu is not None) and (self.vector_n_param_mu != old_vector_n_param_mu):
                ix_map_ker, interval_map_ker = self.make_maps_ker()
                self.ix_map_ker = ix_map_ker
                self.interval_map_ker = interval_map_ker
            if hasattr(self, '_impact_matrix'):
                if (old_vector_n_param_mu is not None) and (self.vector_n_param_mu != old_vector_n_param_mu):
                    ix_map_imp, interval_map_imp = self.make_maps_imp()
                    self.ix_map_imp = ix_map_imp
                    self.interval_map_imp = interval_map_imp
                self.make_functionals()
                self.n_param_k = self.get_n_param_k()
                self.n_param = sum(self.n_param_k)

    @baselines_vec.deleter
    def baselines_vec(self):
        del self._baselines_vec

    # Kernel matrix
    @property
    def kernel_matrix(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._kernel_matrix

    @kernel_matrix.setter
    def kernel_matrix(self, M):
        if hasattr(self, 'matrix_n_param_ker'):
            old_matrix_n_param_ker = copy.deepcopy(self.matrix_n_param_ker)
        else:
            old_matrix_n_param_ker = None
        # Kernel attributes only
        self._kernel_matrix = copy.deepcopy(M)
        self.d = len(M)
        self.matrix_n_param_ker = self.get_matrix_n_param_ker()
        # Parameters indexing
        ix_map_ker, interval_map_ker = self.make_maps_ker()
        self.ix_map_ker = ix_map_ker
        self.interval_map_ker = interval_map_ker
        # Parameters names
        self.ker_param_names = self.get_ker_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.ker_param_lower_bounds = self.get_ker_param_lower_bounds()
        self.ker_param_upper_bounds = self.get_ker_param_upper_bounds()
        # Fixed betas
        self.is_fixed_betas = True
        for i, j in itertools.product(range(self.d), range(self.d)):
            for r in range(M[i][j].n_basis_ker):
                if 1 not in M[i][j].basis_kernels[r].fixed_indices:
                    self.is_fixed_betas = False
                    break

        # General updates
        if hasattr(self, '_impact_matrix'):
            if (old_matrix_n_param_ker is not None) and (self.matrix_n_param_ker != old_matrix_n_param_ker):
                ix_map_imp, interval_map_imp = self.make_maps_imp()
                self.ix_map_imp = ix_map_imp
                self.interval_map_imp = interval_map_imp
            if hasattr(self, '_baselines_vec'):
                self.make_functionals()
                self.n_param_k = self.get_n_param_k()
                self.n_param = sum(self.n_param_k)

    @kernel_matrix.deleter
    def kernel_matrix(self):
        del self._kernel_matrix

    # Impact matrix
    @property
    def impact_matrix(self):
        """
        Kernel matrix of the MHP.
        Setting the kernel matrix to a new value will automatically modify the
        values of the other attributes that depend on it.

        """
        return self._impact_matrix

    @impact_matrix.setter
    def impact_matrix(self, M):
        # Initialisation if M is None:
        if M is None:
            M = [[ImpactFunction([ConstantImpact()]) for j in range(self.d)]
                 for i in range(self.d)]

        # Impact attributes only
        self._impact_matrix = copy.deepcopy(M)
        self.d = len(M)
        self.matrix_n_param_imp = self.get_matrix_n_param_imp()
        # Parameters indexing
        ix_map_imp, interval_map_imp = self.make_maps_imp()
        self.ix_map_imp = ix_map_imp
        self.interval_map_imp = interval_map_imp
        # Parameters names
        self.imp_param_names = self.get_imp_param_names(index_from_one=self.index_from_one)
        # Parameters bounds
        self.imp_param_lower_bounds = self.get_imp_param_lower_bounds()
        self.imp_param_upper_bounds = self.get_imp_param_upper_bounds()
        # self.is_fixed_impacts
        self.is_fixed_impacts = True
        for i, j in itertools.product(range(self.d), range(self.d)):
            if M[i][j].n_param > 0:
                self.is_fixed_impacts = False
                break

        # General updates
        if hasattr(self, '_baselines_vec') and hasattr(self, '_impact_matrix'):
            self.make_functionals()
            self.n_param_k = self.get_n_param_k()
            self.n_param = sum(self.n_param_k)

    @impact_matrix.deleter
    def impact_matrix(self):
        del self._impact_matrix

    # N params
    def get_vector_n_param_mu(self):
        """
        Get the matrix of number of parameters per kernel model.

        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list` of `list` of `int`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        vec_n_param_mu = [self._baselines_vec[i].n_param for i in range(d)]
        return vec_n_param_mu

    def get_matrix_n_param_ker(self):
        """
        Get the matrix of number of parameters per kernel model.

        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list` of `list` of `int`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        mat_n_param_ker = [[self._kernel_matrix[i][j].n_param for j in range(d)]
                           for i in range(d)]
        return mat_n_param_ker

    def get_matrix_n_param_imp(self):
        """
        Get the matrix of number of parameters per kernel model.

        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list` of `list` of `int`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        mat_n_param_imp = [[self._impact_matrix[i][j].n_param for j in range(d)]
                            for i in range(d)]
        return mat_n_param_imp

    def get_n_param_k(self):
        d = self.d
        n_param_k = [None]*d
        for k in range(d):
            n_param_k[k] = (self.vector_n_param_mu[k]
                            + sum(self.matrix_n_param_ker[k])
                            + sum(self.matrix_n_param_imp[k]))
        return n_param_k

    # Parameters map
    def make_maps_ker(self):
        """
        Compute the mapping between the MHP parameter indices and the flattened
        vectors of parameters.

        Denote by :math:`d` the dimensionality of the MHP model. For all
        :math:`i, j \\in [d]`, denote the vector of parameters of the kernel
        :math:`\\phi_{i j}` by :math:`\\vartheta_{i j}`, the dimension of
        :math:`\\vartheta_{i j}` by :math:`\\rho_{i j}`, and the total number
        of event type :math:`k` parameters of the model by

        .. math::
            n^{(k)}_{\\text {param }}:=1+\\sum_{i=1}^{d} \\rho_{k i},
        where the additive term :math:`1` in the right hand side correspond to
        the baseline parameter.

        For all :math:`k \\in[d]`, concatenate the vectors of event type
        :math:`k` kernel parameters as

        .. math::
            \\vartheta_{k}^{\\top}=\\left(\\vartheta_{k 1}^{\\top}, \\ldots, \\vartheta_{k d}^{\\top}\\right),

        and define the vectors of event type :math:`k` MHP parameters
        :math:`\\theta_{k}^{\\top}=\\left(\\mu_{k}, \\vartheta_{k}^{\\top}\\right)`.
        Finally, define the vector of parameters :math:`\\theta` of the MHP by

        .. math::
            \\theta^{\\top}=\\left(\\theta_{1}^{\\top}, \\ldots, \\theta_{d}^{\\top}\\right) .

        The vector of event type :math:`k` MHP parameters, :math:`\\theta_{k}`,
        is the vector of dimension :math:`n^{(k)}_{\\text {param }}` containing
        the baseline parameter :math:`\\mu_{k}` and the parameters of all
        kernels :math:`(\\phi_{kj})_{j \\in [d]}`.

        The list `ix_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`i \\in n_k`, `ix_map[k][i]` is a dictionary with keys `ker` and
        `par`. The value `ix_map[k][i]['ker']` is an integer representing the
        index `j` such that the `i`-th parameter of the flat vector of
        parameters is a parameter of the kernel :math:`\\phi_{kj}`. By default,
        since the `0`-th parameter of the flat vector corresponds to the
        background rate :math:`\\mu_k`, we set `ix_map[k][0]['ker']` to `-1`.
        Now let `j=ix_map[k][i]['ker']`. The value `ix_map[k][i]['par']` is an
        integer representing the
        index `p` such that the `i`-th parameter of the flat vector of
        parameters is the `p`-th parameter of the kernel :math:`\\phi_{kj}`.
        The list `interval_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`j \\in [d]`, `interval_map[k][j]` is a list of two integers
        `[p,q]` such that `p` (resp. `q-1`) is the index of the first (resp.
        last) parameter of kernel :math:`\\phi_{kj}` in the flat vector of
        parameters.

        Returns
        -------
        ix_map : `list` of `list` of `dict`
            DESCRIPTION.
        interval_map : `list` of `list` of `int`
            Matrix of indices of first and last parameters of kernels in the
            flat vector of parameters.

        """
        d = self.d
        ix_map_ker = [[None for i in range(self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k]))]
                      for k in range(d)]
        interval_map_ker = [[None for i in range(d)] for k in range(d)]
        for k in range(d):
            ix_map_ker[k][0] = -1
            ix_ker = 0
            ix_param = 0
            ix_left = self.vector_n_param_mu[k]
            x = self.vector_n_param_mu[k]
            n_param_k = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])
            while x < n_param_k:
                ix_map_ker[k][x] = {'ker': ix_ker, 'par': ix_param}
                if ix_param == self._kernel_matrix[k][ix_ker].n_param-1:
                    interval_map_ker[k][ix_ker] = [ix_left, x+1]
                    ix_ker += 1
                    ix_param = 0
                    ix_left = x+1
                    x += 1
                else:
                    ix_param += 1
                    x += 1
        return ix_map_ker, interval_map_ker

    def make_maps_imp(self):
        """
        Compute the mapping between the MHP parameter indices and the flattened
        vectors of parameters.

        Denote by :math:`d` the dimensionality of the MHP model. For all
        :math:`i, j \\in [d]`, denote the vector of parameters of the kernel
        :math:`\\phi_{i j}` by :math:`\\vartheta_{i j}`, the dimension of
        :math:`\\vartheta_{i j}` by :math:`\\rho_{i j}`, and the total number
        of event type :math:`k` parameters of the model by

        .. math::
            n^{(k)}_{\\text {param }}:=1+\\sum_{i=1}^{d} \\rho_{k i},
        where the additive term :math:`1` in the right hand side correspond to
        the baseline parameter.

        For all :math:`k \\in[d]`, concatenate the vectors of event type
        :math:`k` kernel parameters as

        .. math::
            \\vartheta_{k}^{\\top}=\\left(\\vartheta_{k 1}^{\\top}, \\ldots, \\vartheta_{k d}^{\\top}\\right),

        and define the vectors of event type :math:`k` MHP parameters
        :math:`\\theta_{k}^{\\top}=\\left(\\mu_{k}, \\vartheta_{k}^{\\top}\\right)`.
        Finally, define the vector of parameters :math:`\\theta` of the MHP by

        .. math::
            \\theta^{\\top}=\\left(\\theta_{1}^{\\top}, \\ldots, \\theta_{d}^{\\top}\\right) .

        The vector of event type :math:`k` MHP parameters, :math:`\\theta_{k}`,
        is the vector of dimension :math:`n^{(k)}_{\\text {param }}` containing
        the baseline parameter :math:`\\mu_{k}` and the parameters of all
        kernels :math:`(\\phi_{kj})_{j \\in [d]}`.

        The list `ix_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`i \\in n_k`, `ix_map[k][i]` is a dictionary with keys `ker` and
        `par`. The value `ix_map[k][i]['ker']` is an integer representing the
        index `j` such that the `i`-th parameter of the flat vector of
        parameters is a parameter of the kernel :math:`\\phi_{kj}`. By default,
        since the `0`-th parameter of the flat vector corresponds to the
        background rate :math:`\\mu_k`, we set `ix_map[k][0]['ker']` to `-1`.
        Now let `j=ix_map[k][i]['ker']`. The value `ix_map[k][i]['par']` is an
        integer representing the
        index `p` such that the `i`-th parameter of the flat vector of
        parameters is the `p`-th parameter of the kernel :math:`\\phi_{kj}`.
        The list `interval_map` is such that for all :math:`k \\in [d]`, and for all
        :math:`j \\in [d]`, `interval_map[k][j]` is a list of two integers
        `[p,q]` such that `p` (resp. `q-1`) is the index of the first (resp.
        last) parameter of kernel :math:`\\phi_{kj}` in the flat vector of
        parameters.

        Returns
        -------
        ix_map : `list` of `list` of `dict`
            DESCRIPTION.
        interval_map : `list` of `list` of `int`
            Matrix of indices of first and last parameters of kernels in the
            flat vector of parameters.

        """
        d = self.d
        ix_map_imp = [[None for i in range(self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])+sum(self.matrix_n_param_imp[k]))]
                      for k in range(d)]
        interval_map_imp = [[None for i in range(d)] for k in range(d)]
        for k in range(d):
            ix_map_imp[k][0] = -1
            ix_imp = 0
            ix_param = 0
            ix_left = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])
            x = ix_left
            n_param_k = self.vector_n_param_mu[k]+sum(self.matrix_n_param_ker[k])+sum(self.matrix_n_param_imp[k])
            while x < n_param_k:
                ix_map_imp[k][x] = {'imp': ix_imp, 'par': ix_param}
                if ix_param == self._impact_matrix[k][ix_imp].n_param-1:
                    interval_map_imp[k][ix_imp] = [ix_left, x+1]
                    ix_imp += 1
                    ix_param = 0
                    ix_left = x+1
                    x += 1
                else:
                    ix_param += 1
                    x += 1
        return ix_map_imp, interval_map_imp

    def xk2matrix_params(self, k, x_k):
        """
        Convert the list of flat vectors of parameters to a vector of
        background rates `mu` and a matrix of kernel parameters `kernel_param`.

        Parameters
        ----------
        tensor_param : TYPE
            DESCRIPTION.

        Returns
        -------
        mu_param : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        d = self.d
        # Mu terms
        x_mu_k = np.zeros(self.vector_n_param_mu[k])
        for ix in range(self.vector_n_param_mu[k]):
            x_mu_k[ix] = x_k[ix]

        # Kernel terms
        x_ker_k = np.array([np.zeros(self.matrix_n_param_ker[k][i])
                            for i in range(d)], dtype=object)
        for i in range(d):
            for ix in range(self.matrix_n_param_ker[k][i]):
                ix_mod = self.interval_map_ker[k][i][0]+ix
                x_ker_k[i][ix] = x_k[ix_mod]

        # Impact terms
        x_imp_k = np.array([np.zeros(self.matrix_n_param_imp[k][i])
                            for i in range(d)], dtype=object)
        for i in range(d):
            for ix in range(self.matrix_n_param_imp[k][i]):
                ix_mod = self.interval_map_imp[k][i][0]+ix
                x_imp_k[i][ix] = x_k[ix_mod]
        # Result
        return x_mu_k, x_ker_k, x_imp_k

    def tensor2matrix_params(self, tensor_param):
        """
        Convert the list of flat vectors of parameters to a vector of
        background rates `mu` and a matrix of kernel parameters `kernel_param`.

        Parameters
        ----------
        tensor_param : TYPE
            DESCRIPTION.

        Returns
        -------
        mu_param : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        d = self.d
        mu_param = [None for i in range(d)]
        kernel_param = [None for i in range(d)]
        impact_param = [None for i in range(d)]

        for k in range(d):
            x_mu_k, x_ker_k, x_imp_k = self.xk2matrix_params(k,
                                                             tensor_param[k])
            mu_param[k] = 0+x_mu_k
            kernel_param[k] = 0+x_ker_k
            impact_param[k] = 0+x_imp_k

        mu_param = np.array(mu_param, dtype=object)
        kernel_param = np.array(kernel_param, dtype=object)
        impact_param = np.array(impact_param, dtype=object)
        return mu_param, kernel_param, impact_param

    def matrix2tensor_params(self, mu_param, kernel_param, impact_param):
        """
        Convert the vector of background rates `mu` and the matrix of kernel
        parameters `kernel_param` to the list of flat vectors of parameters.

        Parameters
        ----------
        mu : TYPE
            Vector of baselines.
        kernel_param : TYPE
            Matrix of kernel parameters.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        d = self.d
        x = [np.zeros(self.n_param_k[k]) for k in range(d)]
        for k in range(d):
            # mu parameters
            x[k][:self.vector_n_param_mu[k]] = 0+mu_param[k]
            # kernel parameters
            start = self.vector_n_param_mu[k]
            for i in range(d):
                end = start+len(kernel_param[k][i])-1
                x[k][start:end+1] = 0+kernel_param[k][i]
                start = end+1
            # Impact parameters
            for i in range(d):
                end = start+len(impact_param[k][i])-1
                x[k][start:end+1] = 0+impact_param[k][i]
                start = end+1
        return np.array(x, dtype=object)

    def tensor2matrix_solverpaths(self, x_paths):
        #   Convert the list of parameters
        #   x_paths is a tensor of paths of solvers per parameters
        d = self.d
        list_n = [len(x_paths[k]) for k in range(d)]
        # Mu
        mu_param_paths = [np.zeros((list_n[k],
                                    self.vector_n_param_mu[k]))
                          for k in range(d)]
        for k in range(d):
            mu_param_paths[k] = np.array([x_paths[k][n][:self.vector_n_param_mu[k]]
                                          for n in range(list_n[k])])
        # Kernels
        kernel_param_paths = [[np.zeros((list_n[i],
                                         self.matrix_n_param_ker[i][j]))
                               for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            for ix in range(self.matrix_n_param_ker[i][j]):
                ix_param = self.interval_map_ker[i][j][0]+ix
                kernel_param_paths[i][j][:, ix_param] = np.array([x_paths[i][n][ix_param]
                                                                  for n in range(list_n[i])])
        # Impacts
        impact_param_paths = [[np.zeros((list_n[i],
                                         self.matrix_n_param_imp[i][j]))
                               for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            for ix in range(self.matrix_n_param_imp[i][j]):
                ix_param = self.interval_map_imp[i][j][0]+ix
                impact_param_paths[i][j][:, ix_param] = np.array([x_paths[i][n][ix_param]
                                                                  for n in range(list_n[i])])
        return mu_param_paths, kernel_param_paths, impact_param_paths

    def make_xk(self, k, mu_param=None, kernel_param=None, impact_param=None):
        d = self.d

        if mu_param is None:
            mu_param = self.fitted_mu_param
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
        if impact_param is None:
            impact_param = self.fitted_imp_param

        x = self.matrix2tensor_params(mu_param, kernel_param, impact_param)
        x_k = x[k]
        return x_k

    # Omega
    def is_sbf(self):
        d = self.d
        for i, j in itertools.product(range(d), range(d)):
            if not self._kernel_matrix[i][j].is_sbf():
                return False
        return True

    # Bounds
    def get_mu_param_lower_bounds(self):
        d = self.d
        mu_bnds = [self._baselines_vec[i].get_param_lower_bounds()
                   for i in range(d)]
        return mu_bnds

    def get_mu_param_upper_bounds(self):
        d = self.d
        mu_bnds = [self._baselines_vec[i].get_param_upper_bounds()
                   for i in range(d)]
        return mu_bnds

    def get_ker_param_lower_bounds(self):
        d = self.d
        ker_bnds = [[self._kernel_matrix[i][j].get_param_lower_bounds()
                     for j in range(d)] for i in range(d)]
        return ker_bnds

    def get_ker_param_upper_bounds(self):
        d = self.d
        ker_bnds = [[self._kernel_matrix[i][j].get_param_upper_bounds()
                     for j in range(d)] for i in range(d)]
        return ker_bnds

    def get_imp_param_lower_bounds(self):
        d = self.d
        imp_bnds = [[self._impact_matrix[i][j].get_param_lower_bounds()
                     for j in range(d)] for i in range(d)]
        return imp_bnds

    def get_imp_param_upper_bounds(self):
        d = self.d
        imp_bnds = [[self._impact_matrix[i][j].get_param_upper_bounds()
                     for j in range(d)] for i in range(d)]
        return imp_bnds

    # Param names
    def get_mu_param_names(self, index_from_one=False):
        """
        Get the matrix of parameter names per kernel model.
        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list`
            Matrix of number of parameters per kernel model.

        """
        d = self.d
        mu_param_names = [None]*d
        if d == 1:
            mu = self._baselines_vec[0]
            vec_names = mu.get_vec_param_names()
            n_param = mu.n_param
            mu_param_names[0] = [None]*n_param
            for ix_param in range(n_param):
                ix_basis = mu.ix_map[ix_param]['mu']
                ix_2 = mu.ix_map[ix_param]['par']
                if mu.n_basis_mus == 1:
                    mu_param_names[0][ix_param] = vec_names[ix_basis][ix_2]
                else:
                    sub = str(ix_basis+int(index_from_one))
                    s = vec_names[ix_basis][ix_2]
                    mu_param_names[0][ix_param] = uf.add_subscript(s, sub)

        else:
            for i in range(d):
                mu = self._baselines_vec[i]
                vec_names = mu.get_vec_param_names()
                n_param = mu.n_param
                mu_param_names[i] = [None]*n_param
                if mu.n_basis_mus == 1:
                    for ix_param in range(n_param):
                        ix_basis = mu.ix_map[ix_param]['mu']
                        ix_2 = mu.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))
                        s = vec_names[ix_basis][ix_2]
                        mu_param_names[i][ix_param] = uf.add_subscript(s, sub)
                else:
                    for ix_param in range(n_param):
                        ix_basis = mu.ix_map[ix_param]['mu']
                        ix_2 = mu.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(ix_basis+int(index_from_one))+','
                        s = vec_names[ix_basis][ix_2]
                        mu_param_names[i][ix_param] = uf.add_subscript(s, sub)
        return mu_param_names

    def get_ker_param_names(self, index_from_one=False):
        """
        Get the matrix of parameter names per kernel model.
        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list`
            Matrix of number of parameters per kernel model.

        """
        d = self.d

        if d == 1:
            kernel = self._kernel_matrix[0][0]
            vec_names = kernel.get_vec_param_names()
            n_param = kernel.n_param
            ker_param_names = [[None for j in range(d)] for i in range(d)]
            ker_param_names[0][0] = [None]*n_param
            for ix_param in range(n_param):
                ix_ker = kernel.ix_map[ix_param]['ker']
                ix_2 = kernel.ix_map[ix_param]['par']
                if kernel.n_basis_ker == 1:
                    ker_param_names[0][0][ix_param] = vec_names[ix_ker][ix_2]
                else:
                    sub = str(ix_ker+int(index_from_one))
                    s = vec_names[ix_ker][ix_2]
                    ker_param_names[0][0][ix_param] = uf.add_subscript(s, sub)
        else:
            ker_param_names = [[None for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                kernel = self._kernel_matrix[i][j]
                vec_names = kernel.get_vec_param_names()
                n_param = kernel.n_param
                ker_param_names[i][j] = [None]*n_param
                if kernel.n_basis_ker == 1:
                    for ix_param in range(n_param):
                        ix_ker = kernel.ix_map[ix_param]['ker']
                        ix_2 = kernel.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(j+int(index_from_one))
                        s = vec_names[ix_ker][ix_2]
                        ker_param_names[i][j][ix_param] = uf.add_subscript(s, sub)
                else:
                    for ix_param in range(n_param):
                        ix_ker = kernel.ix_map[ix_param]['ker']
                        ix_2 = kernel.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(j+int(index_from_one))+','+str(ix_ker+int(index_from_one))
                        s = vec_names[ix_ker][ix_2]
                        ker_param_names[i][j][ix_param] = uf.add_subscript(s, sub)
        return ker_param_names

    def get_imp_param_names(self, index_from_one=False):
        """
        Get the matrix of parameter names per kernel model.
        If we denote by :math:`M` this matrix and by :math:`d` the
        dimensionality of the MHP model, then :math:`M` is a :math:`d\\times d`
        matrix which entry :math:`M_{ij}` is the number of parameters of kernel
        :math:`\\phi_{ij}`.

        Returns
        -------
        mat_n_param : `list`
            Matrix of number of parameters per kernel model.

        """
        d = self.d

        if d == 1:
            impact = self._impact_matrix[0][0]
            vec_names = impact.get_vec_param_names()
            n_param = impact.n_param
            imp_param_names = [[None for j in range(d)] for i in range(d)]
            imp_param_names[0][0] = [None]*n_param
            for ix_param in range(n_param):
                ix_imp = impact.ix_map[ix_param]['imp']
                ix_2 = impact.ix_map[ix_param]['par']
                if impact.n_basis_imp == 1:
                    imp_param_names[0][0][ix_param] = vec_names[ix_imp][ix_2]
                else:
                    sub = str(ix_imp+int(index_from_one))
                    s = vec_names[ix_imp][ix_2]
                    imp_param_names[0][0][ix_param] = uf.add_subscript(s, sub)
        else:
            imp_param_names = [[None for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                impact = self._impact_matrix[i][j]
                vec_names = impact.get_vec_param_names()
                n_param = impact.n_param
                imp_param_names[i][j] = [None]*n_param
                if impact.n_basis_imp == 1:
                    for ix_param in range(n_param):
                        ix_imp = impact.ix_map[ix_param]['imp']
                        ix_2 = impact.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(j+int(index_from_one))
                        s = vec_names[ix_imp][ix_2]
                        imp_param_names[i][j][ix_param] = uf.add_subscript(s, sub)
                else:
                    for ix_param in range(n_param):
                        ix_imp = impact.ix_map[ix_param]['imp']
                        ix_2 = impact.ix_map[ix_param]['par']
                        sub = str(i+int(index_from_one))+','+str(j+int(index_from_one))+','+str(ix_imp+int(index_from_one))
                        s = vec_names[ix_imp][ix_2]
                        imp_param_names[i][j][ix_param] = uf.add_subscript(s, sub)
        return imp_param_names

    # Kernel functionals
    def make_functionals(self):
        d = self.d

        # Mu
        self.mu_compensator = [self._baselines_vec[i].compensator for i in range(d)]
        self.mu = [None for i in range(d)]
        self.diff_mu = [None for i in range(d)]
        self.M = [None for i in range(d)]
        self.diff_M = [None for i in range(d)]
        for i in range(d):
            baseline = self._baselines_vec[i]
            mu = baseline.make_mu()
            self.mu[i] = mu
            diff_mu = baseline.make_diff_mu()
            self.diff_mu[i] = diff_mu
            M = baseline.make_M()
            self.M[i] = M
            diff_M = baseline.make_diff_M()
            self.diff_M[i] = diff_M

        # Kernels
        self.phi = [[None for j in range(d)] for i in range(d)]
        self.diff_phi = [[None for j in range(d)] for i in range(d)]
        self.psi = [[None for j in range(d)] for i in range(d)]
        self.diff_psi = [[None for j in range(d)] for i in range(d)]
        self.upsilon = [[[None for k in range(d)] for j in range(d)]
                        for i in range(d)]
        self.diff_sim_upsilon = [[None for j in range(d)]
                                 for i in range(d)]
        self.diff_cross_upsilon = [[[None for k in range(d)]
                                    for j in range(d)] for i in range(d)]
        self.K = [[None for j in range(d)] for i in range(d)]
        self.diff_K = [[None for j in range(d)] for i in range(d)]

        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            phi = kernel.make_phi()
            self.phi[i][j] = phi
            diff_phi = kernel.make_diff_phi()
            self.diff_phi[i][j] = diff_phi
            psi = kernel.make_psi()
            self.psi[i][j] = psi
            diff_psi = kernel.make_diff_psi()
            self.diff_psi[i][j] = diff_psi
            diff_sim_upsilon = kernel.make_diff_sim_upsilon()
            self.diff_sim_upsilon[i][j] = diff_sim_upsilon

        for i, j, k in itertools.product(range(d), range(d), range(d)):
            kernel_ki = self._kernel_matrix[k][i]
            kernel_kj = self._kernel_matrix[k][j]
            if kernel_ki.is_compatible(kernel_kj):
                func_u = kernel_ki.make_upsilon()

                def upsilon(t, s, params_1, params_2):
                    return func_u(kernel_kj, t, s, params_1, params_2)
                self.upsilon[i][j][k] = upsilon
                diff_func_u = kernel_ki.make_diff_cross_upsilon()

                def diff_cross_upsilon(t, s, ix_func, ix_diff, params_1,
                                       params_2):
                    return diff_func_u(kernel_kj, t, s, ix_func, ix_diff,
                                       params_1, params_2)
                self.diff_cross_upsilon[i][j][k] = diff_cross_upsilon
            else:
                raise NotImplementedError("No available interaction"
                                          " between kernel", k, ",", i,
                                          " and kernel ", k, ",", j)

        for i, j in itertools.product(range(d), range(d)):
            kernel = self._kernel_matrix[i][j]
            baseline = self._baselines_vec[i]
            if self.K_from_baseline:
                func_K = baseline.make_K()

                def K(t, s, params_ker, params_mu):
                    return func_K(kernel, t, s, params_ker, params_mu)
                diff_func_K = baseline.make_diff_K()

                def diff_K(t, s, ix_func, ix_diff, params_ker, params_mu):
                    return diff_func_K(kernel, t, s, ix_func, ix_diff,
                                       params_ker, params_mu)
            else:
                func_K = kernel.make_K()

                def K(t, s, params_ker, params_mu):
                    return func_K(baseline, t, s, params_ker, params_mu)
                diff_func_K = kernel.make_diff_K()

                def diff_K(t, s, ix_func, ix_diff, params_ker, params_mu):
                    return diff_func_K(baseline, t, s, ix_func, ix_diff,
                                       params_ker, params_mu)

            self.K[i][j] = K
            self.diff_K[i][j] = diff_K

        # Impact
        self.impact = [[None for j in range(d)] for i in range(d)]
        self.diff_impact = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            impact_function = self._impact_matrix[i][j]
            impact = impact_function.make_impact()
            self.impact[i][j] = impact
            diff_impact = impact_function.make_diff_impact()
            self.diff_impact[i][j] = diff_impact

# =============================================================================
# pre-computations
# =============================================================================
    def is_computable_imp(self, process_path):
        d = process_path.d
        sizes = [len(process_path.list_marks[i]) for i in range(d)]
        total_size = sum(sizes)
        return (total_size <= self.max_array_size)

    def is_computable_S(self, process_path):
        d = process_path.d
        res = np.zeros(d, dtype=bool)
        for k in range(d):
            s = 0
            for i, j in itertools.product(range(d), range(d)):
                for q in range(self.kernel_matrix[k][j].n_basis_ker):
                    s += len(process_path.list_times[i])
            res[k] = (s <= self.max_array_size)
        return res

    def is_computable_eps(self, process_path):
        d = process_path.d
        res = np.zeros(d, dtype=bool)
        for k in range(d):
            s = 0
            for i, j in itertools.product(range(d), range(d)):
                for p in range(self.kernel_matrix[k][i].n_basis_ker):
                    for q in range(self.kernel_matrix[k][j].n_basis_ker):
                        s += len(process_path.list_times[i])
            res[k] = (s <= self.max_array_size)
        return res

# =============================================================================
# Assign qts
# =============================================================================
    def assign_impact_ki(self, k, i, imp_param_k, list_marks):
        if self.impact_k is None:
            impact_ki = get_impact_ki(k, i, self.impact, imp_param_k,
                                      list_marks)
        else:
            impact_ki = self.impact_k[i]
        return impact_ki

    def assign_twimpact_ki(self, i, list_times, impact_ki=None, k=None,
                           imp_param_k=None, list_marks=None):
        if self.twimpact_k is None:
            if impact_ki is None:
                impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                                  list_marks)
            twimpact_ki = get_twimpact_ki(i, impact_ki, list_times)
        else:
            twimpact_ki = self.twimpact_k[i]
        return twimpact_ki

    def assign_epsilon_ijk_pq(self, i, j, p, q, beta_kip, beta_kjq,
                              list_times2end):
        if self.epsilon_k is None:
            epsilon_ijk_pq = get_epsilon_ijk_pq(i, beta_kip, beta_kjq,
                                                list_times2end)
        else:
            epsilon_ijk_pq = self.epsilon_k[i][j][p][q]
        return epsilon_ijk_pq

    def assign_S_imp_ijk_q(self, i, j, q, beta_kjq, varpi, kappa, list_times,
                           impact_kj=None, k=None, imp_param_k=None,
                           list_marks=None):
        if self.S_imp_k is None:
            if impact_kj is None:
                impact_kj = self.assign_impact_ki(k, j, imp_param_k,
                                                  list_marks)
            S_imp_ijk_q = get_S_imp_ijk_q(i, j, beta_kjq, varpi, kappa,
                                          list_times, impact_kj)
        else:
            S_imp_ijk_q = self.S_imp_k[i][j][q]
        return S_imp_ijk_q

    def assign_S_twimp_ijk_q(self, i, j, q, beta_kjq, varpi, kappa, list_times,
                             twimpact_kj=None, impact_kj=None, k=None,
                             imp_param_k=None, list_marks=None):
        if self.S_twimp_k is None:
            if twimpact_kj is None:
                twimpact_kj = self.assign_twimpact_ki(j, list_times,
                                                      impact_ki=impact_kj, k=k,
                                                      imp_param_k=imp_param_k,
                                                      list_marks=list_marks)
            S_twimp_ijk_q = get_S_twimp_ijk_q(i, j, beta_kjq, varpi, kappa,
                                              list_times,
                                              twimpact_kj)
        else:
            S_twimp_ijk_q = self.S_twimp_k[i][j][q]
        return S_twimp_ijk_q

# =============================================================================
# LSE
# =============================================================================
    def get_lse_k(self, k, process_path, x_k=None):
        # Exact LSE_k of an Exponential MHP
        d = self.d
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        T_f = process_path.T_f
        kappa = process_path.kappa
        varpi = process_path.varpi

        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)

        res = 0.

        # M term
        M_term = self.M[k](T_f, mu_param_k)
        res += M_term

        # Mu term
        mu_term = np.sum(self.mu[k](list_times[k], mu_param_k))
        res -= 2.*(mu_term/T_f)

        # Single Sums
        # Sum K
        for i in range(d):
            ker_param_ki = ker_param_k[i]
            K_vals = self.K[k][i](list_times2end[i], list_times[i],
                                  ker_param_ki, mu_param_k)
            impact_ki = self.assign_impact_ki(k, i, imp_param_k, list_marks)
            K_term = np.sum(K_vals*impact_ki)
            res += 2.*(K_term/T_f)

        # Sum Upsilonzero
        for i in range(d):
            # Upsilon_iik(T-t^i_m,0)
            kernel_ki = self.kernel_matrix[k][i]
            r_ki = kernel_ki.n_basis_ker
            ker_param_ki = ker_param_k[i]
            ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
            for p, q in itertools.product(range(r_ki), range(r_ki)):
                omega_kip = ker_vars_ki[2*p]
                beta_kip = ker_vars_ki[2*p+1]
                omega_kiq = ker_vars_ki[2*q]
                beta_kiq = ker_vars_ki[2*q+1]
                # Precomp
                b_iik_pq = self.b_k[i][i][p][q]
                impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                                  list_marks)
                epsilon_iik_pq = self.assign_epsilon_ijk_pq(i, i, p, q,
                                                            beta_kip, beta_kiq,
                                                            list_times2end)
                # Comp
                upsilonzero_term = get_sum_upsilonzero_ki_pq(omega_kip,
                                                             omega_kiq,
                                                             b_iik_pq,
                                                             impact_ki,
                                                             epsilon_iik_pq)
                res += upsilonzero_term/T_f

        # Double sums
        # Upsilon
        for i, j in itertools.product(range(d), range(d)):
            # Get variables of kernel ki
            kernel_ki = self.kernel_matrix[k][i]
            r_ki = kernel_ki.n_basis_ker
            ker_param_ki = ker_param_k[i]
            ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
            # Get variables of kernel kj
            kernel_kj = self.kernel_matrix[k][j]
            r_kj = kernel_kj.n_basis_ker
            ker_param_kj = ker_param_k[j]
            ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
            # Computations
            for p, q in itertools.product(range(r_ki), range(r_kj)):
                omega_kip = ker_vars_ki[2*p]
                beta_kip = ker_vars_ki[2*p+1]
                omega_kjq = ker_vars_kj[2*q]
                beta_kjq = ker_vars_kj[2*q+1]
                # Precomp
                b_ijk_pq = self.b_k[i][j][p][q]
                impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                                  list_marks)
                S_imp_ijk_q = self.assign_S_imp_ijk_q(i, j, q, beta_kjq, varpi,
                                                      kappa, list_times)
                epsilon_ijk_pq = self.assign_epsilon_ijk_pq(i, j, p, q,
                                                            beta_kip, beta_kjq,
                                                            list_times2end)
                # Comp
                upsilon_term = get_sum_upsilon_ijk_pq(i, j, omega_kip,
                                                      omega_kjq,
                                                      b_ijk_pq,  impact_ki,
                                                      S_imp_ijk_q,
                                                      epsilon_ijk_pq,
                                                      varpi)
                res += 2.*(upsilon_term/T_f)
        # Phi
        for j in range(d):
            # Get variables of kernel kj
            kernel_kj = self.kernel_matrix[k][j]
            r_kj = kernel_kj.n_basis_ker
            ker_param_kj = ker_param_k[j]
            ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
            # Computations
            phi_term_j = 0.
            for q in range(r_kj):
                omega_kjq = ker_vars_kj[2*q]
                beta_kjq = ker_vars_kj[2*q+1]
                # precomp
                S_imp_kjk_q = self.assign_S_imp_ijk_q(k, j, q, beta_kjq, varpi,
                                                      kappa, list_times)
                # Comp
                phi_term_j += get_sum_phi_kj_q(k, j, omega_kjq, beta_kjq,
                                            S_imp_kjk_q, varpi)
            res -= 2.*(phi_term_j/T_f)

        return res

    # Derivatives of LSE functions
    def diff_lse_mu_kr(self, k, r, process_path, x_k=None):
        # Exact LSE_k of an Exponential MHP
        d = self.d
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        T_f = process_path.T_f

        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)

        res = 0.

        # M term
        diff_M_term = self.diff_M[k](T_f, r, mu_param_k)
        res += diff_M_term

        # Mu term
        diff_mu_term = np.sum(self.diff_mu[k](list_times[k], r, mu_param_k))
        res -= 2.*(diff_mu_term/T_f)

        # K term
        ix_func = 2
        for i in range(d):
            ker_param_ki = ker_param_k[i]
            diff_K_vals = self.diff_K[k][i](list_times2end[i], list_times[i],
                                            ix_func, r,
                                            ker_param_ki, mu_param_k)
            impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                              process_path.list_marks)
            diff_K_term = np.sum(diff_K_vals*impact_ki)
            res += 2.*(diff_K_term/T_f)
        return res

    def diff_lse_omega_kfr(self, k, f, r, process_path, ix_ker_param=None,
                           x_k=None):
        # Exact Diff LSE_k of an Exponential MHP
        d = self.d
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        T_f = process_path.T_f
        kappa = process_path.kappa
        varpi = process_path.varpi

        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)

        kernel_kf = self.kernel_matrix[k][f]        
        r_kf = kernel_kf.n_basis_ker
        ker_param_kf = ker_param_k[f]
        ker_vars_kf = kernel_kf.make_vars(ker_param_kf)
        omega_kfr = ker_vars_kf[2*r]
        beta_kfr = ker_vars_kf[2*r+1]
        if ix_ker_param is None:
            ix_ker_param = 2*r

        # Precomputations
        impact_kf = self.assign_impact_ki(k, f, imp_param_k, list_marks)

        res = 0.
        # Single Sums
        # Sum K
        ix_func = 1
        diff_K_vals = self.diff_K[k][f](list_times2end[f], list_times[f],
                                        ix_func, ix_ker_param, ker_param_kf,
                                        mu_param_k)
        diff_K_term = np.sum(diff_K_vals*impact_kf)
        res += 2.*(diff_K_term/T_f)

        # Sum Upsilonzero
        # Self
        epsilon_ffk_rr = self.assign_epsilon_ijk_pq(f, f, r, r, beta_kfr,
                                                    beta_kfr,
                                                    list_times2end)
        uzero_self = get_diff_omega_sum_upsilonzero_ki_pp(omega_kfr, beta_kfr,
                                                          impact_kf,
                                                          epsilon_ffk_rr)
        # Cross
        uzero_cross_1 = 0.
        for q in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfq = ker_vars_kf[2*q]
            beta_kfq = ker_vars_kf[2*q+1]
            # Precom
            b_ffk_rq = self.b_k[f][f][r][q]
            epsilon_ffk_rq = self.assign_epsilon_ijk_pq(f, f, r, q, beta_kfr,
                                                        beta_kfq,
                                                        list_times2end)
            # comp
            uzero_cross_1 += get_diff_omegakip_sum_upsilonzero_ki_pq(omega_kfq,
                                                                     b_ffk_rq,
                                                                     impact_kf,
                                                                     epsilon_ffk_rq)
        uzero_cross_2 = 0.
        for p in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfp = ker_vars_kf[2*p]
            beta_kfp = ker_vars_kf[2*p+1]
            # Precomp
            b_ffk_pr = self.b_k[f][f][p][r]
            epsilon_ffk_pr = self.assign_epsilon_ijk_pq(f, f, p, r, beta_kfp,
                                                        beta_kfr,
                                                        list_times2end)
            # Comp
            uzero_cross_2 += get_diff_omegakiq_sum_upsilonzero_ki_pq(omega_kfp,
                                                                     b_ffk_pr,
                                                                     impact_kf,
                                                                     epsilon_ffk_pr)
        uzero_cross = uzero_cross_1+uzero_cross_2
        uzero = uzero_self+uzero_cross
        res += uzero/T_f

        # Double sums
        # Upsilon
        # upsilon self
        # Precomp
        S_imp_ffk_r = self.assign_S_imp_ijk_q(f, f, r, beta_kfr, varpi, kappa,
                                              list_times, impact_kj=impact_kf,
                                              k=k, imp_param_k=imp_param_k,
                                              list_marks=list_marks)
        epsilon_ffk_rr = self.assign_epsilon_ijk_pq(f, f, r, r, beta_kfr,
                                                    beta_kfr, list_times2end)
        # Comp
        ups_self = get_diff_omega_sum_upsilon_iik_pp(f, omega_kfr, beta_kfr,
                                                     impact_kf, S_imp_ffk_r,
                                                     epsilon_ffk_rr,
                                                     varpi)
        for q in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfq = ker_vars_kf[2*q]
            # Precomp
            b_ffk_rq = self.b_k[f][f][r][q]
            S_imp_ffk_q = self.assign_S_imp_ijk_q(f, f, q, beta_kfq, varpi,
                                                  kappa, list_times,
                                                  impact_kj=impact_kf, k=k, 
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
            epsilon_ffk_rq = self.assign_epsilon_ijk_pq(f, f, r, q, beta_kfr,
                                                        beta_kfq,
                                                        list_times2end)
            # Comp
            ups_self += get_diff_omega_i_sum_upsilon_ijk_pq(f, f, omega_kfq,
                                                            b_ffk_rq,
                                                            impact_kf,
                                                            S_imp_ffk_q,
                                                            epsilon_ffk_rq,
                                                            varpi)
        for p in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfp = ker_vars_kf[2*p]
            beta_kfp = ker_vars_kf[2*p+1]
            # Precomp
            b_ffk_pr = self.b_k[f][f][p][r]
            S_imp_ffk_r = self.assign_S_imp_ijk_q(f, f, r, beta_kfr, varpi,
                                                  kappa, list_times,
                                                  impact_kj=impact_kf, k=k,
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
            epsilon_ffk_pr = self.assign_epsilon_ijk_pq(f, f, p, r, beta_kfp,
                                                        beta_kfr,
                                                        list_times2end)
            # Comp
            ups_self += get_diff_omega_j_sum_upsilon_ijk_pq(f, f, omega_kfp,
                                                            b_ffk_pr,
                                                            impact_kf,
                                                            S_imp_ffk_r,
                                                            epsilon_ffk_pr,
                                                            varpi)
        # Upsilon cross 1
        ups_cross_1 = 0.
        for j in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel kj
            kernel_kj = self.kernel_matrix[k][j]
            r_kj = kernel_kj.n_basis_ker
            ker_param_kj = ker_param_k[j]
            ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
            for q in range(r_kj):
                omega_kjq = ker_vars_kj[2*q]
                beta_kjq = ker_vars_kj[2*q+1]
                # Precomp
                b_fjk_rq = self.b_k[f][j][r][q]
                impact_kj = self.assign_impact_ki(k, j, imp_param_k, list_marks)
                S_imp_fjk_q = self.assign_S_imp_ijk_q(f, j, q, beta_kjq, varpi,
                                                      kappa, list_times,
                                                      impact_kj=impact_kj, k=k,
                                                      imp_param_k=imp_param_k,
                                                      list_marks=list_marks)
                epsilon_fjk_rq = self.assign_epsilon_ijk_pq(f, j, r, q,
                                                            beta_kfr, beta_kjq,
                                                            list_times2end)
                # Comp
                ups_cross_1 += get_diff_omega_i_sum_upsilon_ijk_pq(f, j,
                                                                   omega_kjq,
                                                                   b_fjk_rq,
                                                                   impact_kf,
                                                                   S_imp_fjk_q,
                                                                   epsilon_fjk_rq,
                                                                   varpi)
        # Upsilon cross 2
        ups_cross_2 = 0.
        for i in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel kj
            kernel_ki = self.kernel_matrix[k][i]
            r_ki = kernel_ki.n_basis_ker
            ker_param_ki = ker_param_k[i]
            ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
            for p in range(r_ki):
                omega_kip = ker_vars_ki[2*p]
                beta_kip = ker_vars_ki[2*p+1]
                # Precomp
                b_ifk_pr = self.b_k[i][f][p][r]
                impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                                  list_marks)
                S_imp_ifk_r = self.assign_S_imp_ijk_q(i, f, r, beta_kfr, varpi,
                                                      kappa, list_times,
                                                      impact_kj=impact_kf, k=k,
                                                      imp_param_k=imp_param_k,
                                                      list_marks=list_marks)

                epsilon_ifk_pr = self.assign_epsilon_ijk_pq(i, f, p, r,
                                                            beta_kip, beta_kfr,
                                                            list_times2end)
                # Comp
                ups_cross_2 += get_diff_omega_j_sum_upsilon_ijk_pq(i, f,
                                                                   omega_kip,
                                                                   b_ifk_pr,
                                                                   impact_ki,
                                                                   S_imp_ifk_r,
                                                                   epsilon_ifk_pr,
                                                                   varpi)
        ups_cross = ups_cross_1+ups_cross_2
        ups_term = ups_self+ups_cross
        res += 2.*(ups_term/T_f)
        # Phi
        S_imp_kfk_r = self.assign_S_imp_ijk_q(k, f, r, beta_kfr, varpi, kappa,
                                              list_times, impact_kj=impact_kf,
                                              k=k,
                                              imp_param_k=imp_param_k,
                                              list_marks=list_marks)
        phi_term = get_diff_omega_sum_phi_kj_q(k, f, beta_kfr, S_imp_kfk_r,
                                               varpi)
        res -= 2.*(phi_term/T_f)

        return res

    def diff_lse_beta_kfr(self, k, f, r, process_path, ix_ker_param=None,
                          x_k=None):
        # Exact Diff LSE_k of an Exponential MHP
        d = self.d
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        T_f = process_path.T_f
        varpi = process_path.varpi
        kappa = process_path.kappa

        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)

        kernel_kf = self.kernel_matrix[k][f]
        r_kf = kernel_kf.n_basis_ker
        ker_param_kf = ker_param_k[f]
        ker_vars_kf = kernel_kf.make_vars(ker_param_kf)
        omega_kfr = ker_vars_kf[2*r]
        beta_kfr = ker_vars_kf[2*r+1]
        if ix_ker_param is None:
            ix_ker_param = 2*r+1

        # Precomputations
        impact_kf = self.assign_impact_ki(k, f, imp_param_k, list_marks)
        epsilon_ffk_rr = self.assign_epsilon_ijk_pq(f, f, r, r, beta_kfr,
                                                    beta_kfr, list_times2end)

        # Compute result
        res = 0.

        # Single Sums
        # Sum K
        ix_func = 1
        diff_K_vals = self.diff_K[k][f](list_times2end[f], list_times[f],
                                        ix_func, ix_ker_param, ker_param_kf,
                                        mu_param_k)
        diff_K_term = np.sum(diff_K_vals*impact_kf)
        res += 2.*(diff_K_term/T_f)

        # Sum Upsilonzero
        # Self
        uzero_self = get_diff_beta_sum_upsilonzero_ki_pp(f, omega_kfr,
                                                         beta_kfr,
                                                         impact_kf,
                                                         epsilon_ffk_rr,
                                                         list_times2end)
        # Cross
        uzero_cross_1 = 0.
        for q in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfq = ker_vars_kf[2*q]
            beta_kfq = ker_vars_kf[2*q+1]
            # Precomp
            b_2_ffk_pq = self. b_2_k[f][f][r][q]
            epsilon_ffk_rq = self.assign_epsilon_ijk_pq(f, f, r, q, beta_kfr,
                                                        beta_kfq,
                                                        list_times2end)
            # Comp
            uzero_cross_1 += get_diff_betakip_sum_upsilonzero_ki_pq(f,
                                                                    omega_kfr,
                                                                    beta_kfr,
                                                                    omega_kfq,
                                                                    b_2_ffk_pq,
                                                                    impact_kf,
                                                                    epsilon_ffk_rq,
                                                                    list_times2end)
        uzero_cross_2 = 0.
        for p in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfp = ker_vars_kf[2*p]
            beta_kfp = ker_vars_kf[2*p+1]
            # Precomp
            b_1_ffk_pr = self. b_1_k[f][f][p][r]
            epsilon_ffk_pr = self.assign_epsilon_ijk_pq(f, f, p, r, beta_kfp,
                                                        beta_kfr,
                                                        list_times2end)
            # Comp
            uzero_cross_2 += get_diff_betakiq_sum_upsilonzero_ki_pq(f,
                                                                    omega_kfp,
                                                                    omega_kfr,
                                                                    beta_kfr,
                                                                    b_1_ffk_pr,
                                                                    impact_kf,
                                                                    epsilon_ffk_pr,
                                                                    list_times2end)
        uzero_cross = uzero_cross_1+uzero_cross_2
        uzero = uzero_self+uzero_cross
        res += uzero/T_f

        # Double sums
        # Upsilon
        # upsilon self
        # Precomp
        S_imp_ffk_r = self.assign_S_imp_ijk_q(f, f, r, beta_kfr, varpi, kappa,
                                              list_times, impact_kj=impact_kf,
                                              k=k, imp_param_k=imp_param_k,
                                              list_marks=list_marks)
        twimpact_kf = self.assign_twimpact_ki(f, list_times,
                                              impact_ki=impact_kf,
                                              k=k, imp_param_k=imp_param_k,
                                              list_marks=list_marks)
        S_twimp_ffk_r = self.assign_S_twimp_ijk_q(f, f, r, beta_kfr, varpi,
                                                  kappa, list_times,
                                                  twimpact_kj=twimpact_kf,
                                                  impact_kj=impact_kf, k=k,
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
        # Comp
        ups_self = get_diff_beta_sum_upsilon_iik_pp(f, omega_kfr, beta_kfr,
                                                    impact_kf, S_imp_ffk_r,
                                                    S_twimp_ffk_r,
                                                    epsilon_ffk_rr,
                                                    varpi, list_times, T_f)
        for q in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfq = ker_vars_kf[2*q]
            beta_kfq = ker_vars_kf[2*q+1]
            # Precomp
            b_2_ffk_rq = self.b_2_k[f][f][r][q]
            S_imp_ffk_q = self.assign_S_imp_ijk_q(f, f, q, beta_kfq, varpi,
                                                  kappa, list_times,
                                                  impact_kj=impact_kf, k=k,
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
            epsilon_ffk_rq = self.assign_epsilon_ijk_pq(f, f, r, q, beta_kfr,
                                                        beta_kfq,
                                                        list_times2end)
            # Comp
            ups_self += get_diff_beta_i_sum_upsilon_ijk_pq(f, f, omega_kfr,
                                                           beta_kfr, omega_kfq,
                                                           b_2_ffk_rq,
                                                           impact_kf,
                                                           S_imp_ffk_q,
                                                           epsilon_ffk_rq,
                                                           varpi, list_times,
                                                           T_f)
        for p in itertools.chain(range(r), range(r+1, r_kf)):
            omega_kfp = ker_vars_kf[2*p]
            beta_kfp = ker_vars_kf[2*p+1]
            # Precomp
            b_ffk_pr = self.b_k[f][f][p][r]
            b_1_ffk_pr = self.b_1_k[f][f][p][r]
            S_imp_ffk_q = self.assign_S_imp_ijk_q(f, f, q, beta_kfq, varpi,
                                                  kappa, list_times,
                                                  impact_kj=impact_kf, k=k,
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
            epsilon_ffk_pr = self.assign_epsilon_ijk_pq(f, f, p, r, beta_kfp,
                                                        beta_kfr,
                                                        list_times2end)
            # Comp
            ups_self += get_diff_beta_j_sum_upsilon_ijk_pq(f, f, omega_kfp,
                                                           beta_kfp, omega_kfr,
                                                           beta_kfr, b_ffk_pr,
                                                           b_1_ffk_pr,
                                                           impact_kf,
                                                           S_imp_ffk_q,
                                                           S_twimp_ffk_r,
                                                           epsilon_ffk_pr,
                                                           varpi, list_times,
                                                           T_f)
        # Upsilon cross 1
        ups_cross_1 = 0.
        for j in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel kj
            kernel_kj = self.kernel_matrix[k][j]
            r_kj = kernel_kj.n_basis_ker
            ker_param_kj = ker_param_k[j]
            ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
            for q in range(r_kj):
                omega_kjq = ker_vars_kj[2*q]
                beta_kjq = ker_vars_kj[2*q+1]
                # Precomp
                b_2_fjk_rq = self.b_2_k[f][j][r][q]
                impact_kj = self.assign_impact_ki(k, j, imp_param_k,
                                                  list_marks)
                S_imp_fjk_q = self.assign_S_imp_ijk_q(f, j, q, beta_kjq, varpi,
                                                      kappa, list_times,
                                                      impact_kj=impact_kj, k=k,
                                                      imp_param_k=imp_param_k,
                                                      list_marks=list_marks)
                epsilon_fjk_rq = self.assign_epsilon_ijk_pq(f, j, r, q,
                                                            beta_kfr, beta_kjq,
                                                            list_times2end)
                # Comp
                ups_cross_1 += get_diff_beta_i_sum_upsilon_ijk_pq(f, j,
                                                                  omega_kfr,
                                                                  beta_kfr,
                                                                  omega_kjq,
                                                                  b_2_fjk_rq,
                                                                  impact_kf,
                                                                  S_imp_fjk_q,
                                                                  epsilon_fjk_rq,
                                                                  varpi,
                                                                  list_times,
                                                                  T_f)
        # Upsilon cross 2
        ups_cross_2 = 0.
        for i in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel kj
            kernel_ki = self.kernel_matrix[k][i]
            r_ki = kernel_ki.n_basis_ker
            ker_param_ki = ker_param_k[i]
            ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
            # precomputations
            impact_ki = self.assign_impact_ki(k, i, imp_param_k, list_marks)
            for p in range(r_ki):
                omega_kip = ker_vars_ki[2*p]
                beta_kip = ker_vars_ki[2*p+1]
                # Precomp
                b_ifk_pr = self.b_k[i][f][p][r]
                b_1_ifk_pr = self.b_1_k[i][f][p][r]
                S_imp_ifk_r = self.assign_S_imp_ijk_q(i, f, r, beta_kfr, varpi,
                                                      kappa, list_times,
                                                      impact_kj=impact_kf, k=k,
                                                      imp_param_k=imp_param_k,
                                                      list_marks=list_marks)
                S_twimp_ifk_r = self.assign_S_twimp_ijk_q(i, f, r, beta_kfr,
                                                          varpi, kappa,
                                                          list_times,
                                                          twimpact_kj=twimpact_kf,
                                                          impact_kj=impact_kf, k=k,
                                                          imp_param_k=imp_param_k,
                                                          list_marks=list_marks)
                epsilon_ifk_pr = self.assign_epsilon_ijk_pq(i, f, p, r,
                                                            beta_kip, beta_kfr,
                                                            list_times2end)
                # Result
                ups_cross_2 += get_diff_beta_j_sum_upsilon_ijk_pq(i, f,
                                                                  omega_kip,
                                                                  beta_kip,
                                                                  omega_kfr,
                                                                  beta_kfr,
                                                                  b_ifk_pr,
                                                                  b_1_ifk_pr,
                                                                  impact_ki,
                                                                  S_imp_ifk_r,
                                                                  S_twimp_ifk_r,
                                                                  epsilon_ifk_pr,
                                                                  varpi,
                                                                  list_times,
                                                                  T_f)
        ups_cross = ups_cross_1+ups_cross_2
        ups_term = ups_self+ups_cross
        res += 2.*(ups_term/T_f)
        # Phi
        S_imp_kfk_r = self.assign_S_imp_ijk_q(k, f, r, beta_kfr, varpi, kappa,
                                              list_times,
                                              impact_kj=impact_kf, k=k,
                                              imp_param_k=imp_param_k,
                                              list_marks=list_marks)
        S_twimp_kfk_r = self.assign_S_twimp_ijk_q(k, f, r, beta_kfr, varpi,
                                                  kappa, list_times,
                                                  twimpact_kj=twimpact_kf,
                                                  impact_kj=impact_kf, k=k,
                                                  imp_param_k=imp_param_k,
                                                  list_marks=list_marks)
        phi_term = get_diff_beta_sum_phi_kj_q(k, f, omega_kfr, beta_kfr,
                                              S_imp_kfk_r, S_twimp_kfk_r,
                                              varpi, list_times)
        res -= 2.*(phi_term/T_f)

        return res

    def diff_lse_theta_kfr_impact(self, k, f, r, process_path, x_k=None):
        # Exact LSE_k of an Exponential MHP
        d = self.d
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        T_f = process_path.T_f
        varpi = process_path.varpi
        kappa = process_path.kappa

        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)

        kernel_kf = self.kernel_matrix[k][f]
        r_kf = kernel_kf.n_basis_ker
        ker_param_kf = ker_param_k[f]
        ker_vars_kf = kernel_kf.make_vars(ker_param_kf)

        # Pre-computations
        diff_impact_kf_r = self.diff_impact[k][f](list_marks[f], r,
                                                  imp_param_k[f])
        impact_kf = self.assign_impact_ki(k, f, imp_param_k, list_marks)
        diff_sq_impact_kf_r = 2.*impact_kf*diff_impact_kf_r
        # S_diff_r_imp_kfk_q
        S_diff_r_imp_kfk = [get_S_imp_ijk_q(k, f, ker_vars_kf[2*q+1], varpi,
                                            kappa, list_times, impact_kf)
                            for q in range(r_kf)]
        if f == k:
            S_diff_r_imp_ffk = S_diff_r_imp_kfk
        else:
            S_diff_r_imp_ffk = [get_S_imp_ijk_q(f, f, ker_vars_kf[2*q+1],
                                                varpi, kappa, list_times,
                                                impact_kf)
                                for q in range(r_kf)]
        # Compute result
        res = 0.
        # Single Sums
        # Sum K
        K_vals = self.K[k][f](list_times2end[f], list_times[f],
                              ker_param_kf, mu_param_k)
        K_term = np.sum(K_vals*diff_impact_kf_r)
        res += 2.*(K_term/T_f)

        # Sum Upsilonzero
        # Upsilon_ffk(T-t^i_m,0)
        upsilonzero_term = 0.
        for p, q in itertools.product(range(r_kf), range(r_kf)):
            # Pre-computations
            omega_kfp = ker_vars_kf[2*p]
            beta_kfp = ker_vars_kf[2*p+1]
            omega_kfq = ker_vars_kf[2*q]
            beta_kfq = ker_vars_kf[2*q+1]
            b_ffk_pq = self.b_k[f][f][p][q]
            # epsilon_ffk_pq
            epsilon_ffk_pq, = self.assign_epsilon_ijk_pq(f, f, p, q, beta_kfp,
                                                         beta_kfq,
                                                         list_times2end)
            # Compute term
            upsilonzero_term += get_sum_upsilonzero_ki_pq(omega_kfp,
                                                          omega_kfq,
                                                          b_ffk_pq,
                                                          diff_sq_impact_kf_r,
                                                          epsilon_ffk_pq)
        res += upsilonzero_term/T_f

        # Double sums
        # Upsilon self
        upsilon_self = 0.
        # Computations
        for p, q in itertools.product(range(r_kf), range(r_kf)):
            # Pre-computations
            omega_kfp = ker_vars_kf[2*p]
            omega_kfq = ker_vars_kf[2*q]
            b_ffk_pq = self.b_k[f][f][p][q]
            S_imp_ffk_q = self.S_imp_k[f][f][q]
            S_diff_r_imp_ffk_q = S_diff_r_imp_ffk[q]
            # epsilon_ffk_pq
            # Compute term
            uself_term_1 = get_sum_upsilon_ijk_pq(f, f, omega_kfp,
                                                  omega_kfq,
                                                  b_ffk_pq,  diff_impact_kf_r,
                                                  S_imp_ffk_q, epsilon_ffk_pq,
                                                  varpi)
            uself_term_2 = get_sum_upsilon_ijk_pq(f, f, omega_kfp,
                                                  omega_kfq,
                                                  b_ffk_pq,  impact_kf,
                                                  S_diff_r_imp_ffk_q,
                                                  epsilon_ffk_pq,
                                                  varpi)
            upsilon_self += uself_term_1+uself_term_2

        # Upsilon cross
        upsilon_cross = 0.
        for i in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel ki
            kernel_ki = self.kernel_matrix[k][i]
            r_ki = kernel_ki.n_basis_ker
            ker_param_ki = ker_param_k[i]
            ker_vars_ki = kernel_ki.make_vars(ker_param_ki)
            impact_ki = self.assign_impact_ki(k, i, imp_param_k,
                                              list_marks)
            for p, q in itertools.product(range(r_ki), range(r_kf)):
                omega_kip = ker_vars_ki[2*p]
                omega_kfq = ker_vars_kf[2*q]
                b_ifk_pq = self.b_k[i][f][p][q]
                # S_diff_r_imp_ifk
                if i == k:
                    S_diff_r_imp_ifk_q = S_diff_r_imp_kfk[q]
                elif (k != f) and (i == f):
                    S_diff_r_imp_ifk_q = S_diff_r_imp_ffk[q]
                else:
                    S_diff_r_imp_ifk_q = get_S_imp_ijk_q(i, f, beta_kfq, varpi,
                                                         kappa, list_times,
                                                         impact_kf)
                # epsilon
                if self.epsilon_k is None:
                    epsilon_ifk_pq = get_epsilon_ijk_pq(i, beta_kfp, beta_kfq,
                                                        list_times2end)
                else:
                    epsilon_ifk_pq = self.epsilon_k[i][f][p][q]
                # Computations
                upsilon_cross += get_sum_upsilon_ijk_pq(i, f, omega_kip,
                                                        omega_kfq,
                                                        b_ifk_pq,  impact_ki,
                                                        S_diff_r_imp_ifk_q,
                                                        epsilon_ifk_pq,
                                                        varpi)

        for j in itertools.chain(range(f), range(f+1, d)):
            # Get variables of kernel kj
            kernel_kj = self.kernel_matrix[k][j]
            r_kj = kernel_kj.n_basis_ker
            ker_param_kj = ker_param_k[j]
            ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
            for p, q in itertools.product(range(r_kf), range(r_kj)):
                omega_kfp = ker_vars_kf[2*p]
                omega_kjq = ker_vars_kj[2*q]
                beta_kjq = ker_vars_kj[2*q+1]
                b_fjk_pq = self.b_k[f][k][p][q]
                S_imp_fjk_q = self.S_imp_k[f][j][q]
                epsilon_fjk_pq = self.assign_epsilon_ijk_pq(f, j, p, q,
                                                            beta_kfp,
                                                            beta_kjq,
                                                            list_times2end)

                # Compute
                upsilon_cross += get_sum_upsilon_ijk_pq(f, j, omega_kfp,
                                                        omega_kjq,
                                                        b_fjk_pq,
                                                        diff_impact_kf_r,
                                                        S_imp_fjk_q,
                                                        epsilon_fjk_pq,
                                                        varpi)
        upsilon_term = upsilon_self+upsilon_cross
        res += 2.*(upsilon_term/T_f)

        # Phi
        # Get variables of kernel kj
        # Computations
        phi_term = 0.
        for q in range(r_kf):
            omega_kfq = ker_vars_kf[2*q]
            beta_kfq = ker_vars_kf[2*q+1]
            S_diff_r_imp_kfk_q = S_diff_r_imp_kfk[q]
            phi_term += get_sum_phi_kj_q(k, f, omega_kfq, beta_kfq,
                                         S_diff_r_imp_kfk_q, varpi)
        res -= 2.*(phi_term/T_f)

        return res

    # Derivatives of LSE functions
    def compute_grad_lse_k(self, k, process_path, x_k=None):
        d = process_path.d
        # Parameters
        if x_k is None:
            x_k = self.make_xk(k)
        mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k, x_k)
        n_param_k = len(x_k)
        grad = np.zeros(n_param_k)

        ix_param = 0
        # Baseline parameters
        for ix_mu_param in range(len(mu_param_k)):
            grad[ix_param] = self.diff_lse_mu_kr(k, ix_mu_param, process_path,
                                                 x_k=x_k)
            ix_param += 1

        # Derivative with respect to kernel parameters
        for f in range(d):
            kernel_kf = self.kernel_matrix[k][f]
            r_kf = kernel_kf.n_basis_ker
            ix_ker_param = 0
            for r in range(r_kf):
                # Omega
                if kernel_kf.list_is_active_var[2*r]:
                    grad[ix_param] = self.diff_lse_omega_kfr(k, f, r,
                                                             process_path,                                                             
                                                             x_k=x_k)

                    ix_ker_param += 1
                    ix_param += 1
                # Beta
                if kernel_kf.list_is_active_var[2*r+1]:
                    grad[ix_param] = self.diff_lse_beta_kfr(k, f, r,
                                                            process_path,
                                                            ix_ker_param=ix_ker_param,
                                                            x_k=x_k)
                    ix_ker_param += 1
                    ix_param += 1

        # Derivatives wrt impact parameters
        for f in range(d):
            impact_func_kf = self.impact_matrix[k][f]
            for r in range(impact_func_kf.n_param):
                grad[ix_param] = self.diff_lse_theta_kfr_impact(k, f, r,
                                                                process_path,
                                                                x_k=x_k)
                ix_param += 1
        return grad

    # Logging
    def init_logger(self, logger):
        d = self.d
        n_iter = logger.n_iter
        n_param_k = self.n_param_k
        if logger.is_log_param:
            logger.param_logs = [np.zeros((n_iter[k]+1, n_param_k[k]))
                                 for k in range(d)]
            logger.mu = [[None for x in range(n_iter[i]+1)] for i in range(d)]
            logger.ker = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                          for i in range(d)]
            logger.imp = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                          for i in range(d)]
        if logger.is_log_grad:
            logger.grad_logs = [np.zeros((n_iter[k], n_param_k[k]))
                                for k in range(d)]
            logger.grad_mu = [[None for x in range(n_iter[i]+1)] for i in range(d)]
            logger.grad_ker = [[[None for x in range(n_iter[i])]
                                for j in range(d)] for i in range(d)]
            logger.grad_imp = [[[None for x in range(n_iter[i])]
                                for j in range(d)] for i in range(d)]
        if logger.is_log_lse:
            logger.lse = np.array([np.zeros(n_iter[k]) for k in range(d)])

        logger.mu_0 = None
        logger.ker_0 = None
        logger.imp_0 = None

    def process_logs(self, logger):
        d = self.d
        if logger.is_log_param:
            # Mu
            for i in range(d):
                for ix in range(logger.n_iter[i]+1):
                    logger.mu[i][ix] = logger.param_logs[i][ix][:self.vector_n_param_mu[i]]

            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]+1):
                    # Kernel
                    if self.matrix_n_param_ker[i][j] > 0:
                        logger.ker[i][j][ix] = logger.param_logs[i][ix][self.interval_map_ker[i][j][0]:self.interval_map_ker[i][j][1]]
                    # Impact
                    if self.matrix_n_param_imp[i][j] > 0:
                        logger.imp[i][j][ix] = logger.param_logs[i][ix][self.interval_map_imp[i][j][0]:self.interval_map_imp[i][j][1]]
        if logger.is_log_grad:
            # Mu
            for i in range(d):
                for ix in range(logger.n_iter[i]):
                    logger.grad_mu[i][ix] = logger.grad_logs[i][ix][:self.vector_n_param_mu[i]]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(logger.n_iter[i]):
                    # Kernel
                    if self.matrix_n_param_ker[i][j] > 0:
                        logger.grad_ker[i][j][ix] = logger.grad_logs[i][ix][self.interval_map_ker[i][j][0]:self.interval_map_ker[i][j][1]]
                    # Impact
                    if self.matrix_n_param_imp[i][j] > 0:
                        logger.grad_imp[i][j][ix] = logger.grad_logs[i][ix][self.interval_map_imp[i][j][0]:self.interval_map_imp[i][j][1]]

    # Fit
    def clear_fit(self):
        """
        Delete all previously saved results and fit logs.

        """
        self.is_fitted = False
        self.fitted_mu_param = None
        self.fitted_ker_param = None
        self.fitted_imp_param = None
        self.fit_residuals = None
        self.fitted_adjacency = None
        self.fit_log = None

    def init_precomp(self, k,  ker_param_k, imp_param_k, process_path,
                     is_comp_imp, is_comp_S, is_comp_eps):
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        varpi = process_path.varpi
        kappa = process_path.kappa

        # Betas only
        self.b_k = get_b_k_tensor(k, self.kernel_matrix, ker_param_k)
        self.b_1_k = get_b_1_k_tensor(k, self.kernel_matrix, ker_param_k)
        self.b_2_k = get_b_2_k_tensor(k, self.kernel_matrix, ker_param_k)

        # Impact only: self.precomp_imp
        if (self.precomp_imp is None and is_comp_imp) or self.precomp_imp:
            self.impact_k = get_impact_k_tensor(k, self.impact,
                                                imp_param_k,
                                                list_marks)
            self.twimpact_k = get_twimpact_k_tensor(k, self.impact_k,
                                                    list_times)
        else:
            self.impact_k = None
            self.twimpact_k = None

        # self.precomp_S
        if (self.precomp_S is None and is_comp_S[k]) or self.precomp_S:
            self.S_imp_k = get_S_imp_k_tensor(k, self.kernel_matrix,
                                              ker_param_k, varpi, kappa,
                                              list_times, self.impact_k)
            if not self.is_fixed_betas:
                self.S_twimp_k = get_S_twimp_k_tensor(k, self.kernel_matrix,
                                                      ker_param_k,
                                                      varpi, kappa, list_times,
                                                      self.twimpact_k)
            else:
                self.S_twimp_k = None
        else:
            self.S_imp_k = None
            self.S_twimp_k = None

        # self.precomp_eps
        if (self.precomp_eps is None and is_comp_eps[k]) or self.precomp_eps:
            self.epsilon_k = get_epsilon_k_tensor(k, self.kernel_matrix,
                                                  ker_param_k,
                                                  list_times2end)
        else:
            self.epsilon_k = None

    def update_precomp(self, k,  ker_param_k, imp_param_k, process_path,
                       is_comp_imp, is_comp_S, is_comp_eps):
        # Time quantities
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        list_marks = process_path.list_marks
        varpi = process_path.varpi
        kappa = process_path.kappa

        # Betas only
        if not self.is_fixed_betas:
            self.b_k = get_b_k_tensor(k, self.kernel_matrix, ker_param_k)
            self.b_1_k = get_b_1_k_tensor(k, self.kernel_matrix, ker_param_k)
            self.b_2_k = get_b_2_k_tensor(k, self.kernel_matrix, ker_param_k)

        # Impact only: self.precomp_imp
        if (self.precomp_imp is None and is_comp_imp) or self.precomp_imp:
            self.impact_k = get_impact_k_tensor(k, self.impact,
                                                imp_param_k,
                                                list_marks)
            if not self.is_fixed_betas:
                self.twimpact_k = get_twimpact_k_tensor(k, self.impact_k,
                                                        list_times)

        # self.precomp_S
        if (self.precomp_S is None and is_comp_S[k]) or self.precomp_S:
            if (not self.is_fixed_betas) or (not self.is_fixed_impacts):
                self.S_imp_k = get_S_imp_k_tensor(k, self.kernel_matrix,
                                                  ker_param_k, varpi, kappa,
                                                  list_times, self.impact_k)
            if not self.is_fixed_betas:
                self.S_twimp_k = get_S_twimp_k_tensor(k, self.kernel_matrix,
                                                      ker_param_k,
                                                      varpi, kappa, list_times,
                                                      self.twimpact_k)

        # self.precomp_eps
        if (self.precomp_eps is None and is_comp_eps[k]) or self.precomp_eps:
            if not self.is_fixed_betas:
                self.epsilon_k = get_epsilon_k_tensor(k, self.kernel_matrix,
                                                      ker_param_k,
                                                      list_times2end)

    def fit(self, process_path, x_0=None,
            n_iter=1000, solvers=None, estimators=None, logger=None,
            rng=None, seed=1234,
            verbose=False, clear=True, write=True, **kwargs):
        """
        Fit the RecExp model to one data path.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The least squares error (LSE) of this model for these observations is
        defined as

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\frac{1}{T} \\sum_{k=1}^{d} \\int_{0}^{T} \\lambda_{k}(t)^{2} \\mathrm{~d} t-\\frac{2}{T} \\sum_{k=1}^{d} \\sum_{m=1}^{N_{T}^{k}} \\lambda_{k}\\left(t_{m}^{k}\\right).

        For a homogeneous Poisson model, this simplifies to

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\sum_{k=1}^{d} \\bigg( \\mu_{k}^{2} -2 \\frac{N_{T}^{k}}{T} \\bigg).

        Parameters
        ----------
        process_path : `aslsd.ProcessPath`
            Data path to fit model to.
        x_0 : `list` of `numpy.ndarray`, optional
            x_0[k] is the initial guess for parameters of problem k. The
            default is None.
        n_iter : `list` or `int`, optional
            n_iter[k] is the number of iterations of the the optimisation
            algorithm for problem k. If  n_iter is of type `int`, it will be
            converted to a d-dimensional array where each entry is equal to
            that integer. The default is 1000.
        solvers : `list` of `aslsd.Solver`, optional
            solvers[k] is the optimization solver for problem k. The default
            is None.
        estimators : `list` of `aslsd.Esimtator`, optional
            estimators[k] is the gradient estimator for problem k. The default
            is None.
        logger : `aslsd.OptimLogger`, optional
            DESCRIPTION. The default is None.
        seed : `int`, optional
            Seed for the random number generator. The default is 1234.
        verbose : `bool`, optional
            If True, print progression information. The default is False.
        clear : `bool`, optional
            If true, delete all previously saved results and logs from the
            corresponding attributes of the MHP object. The default is True.
        write : `bool`, optional
            If true, save the estimation results and logs in the corresponding
            attributes of the MHP object. The default is True.
        **kwargs : `dict`
            Additional keyword arguments.

        Returns
        -------
        fitted_mu : `numpy.ndarray`
            Fitted baselines.
        fitted_ker_param : `numpy.ndarray`
            Fitted kernel parameters.

        """
        rng = us.make_rng(rng=rng, seed=seed)
        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Data
        d = self.d

        # Model bounds
        lower_bnds = self.matrix2tensor_params(self.mu_param_lower_bounds,
                                               self.ker_param_lower_bounds,
                                               self.imp_param_lower_bounds)
        upper_bnds = self.matrix2tensor_params(self.mu_param_upper_bounds,
                                               self.ker_param_upper_bounds,
                                               self.imp_param_upper_bounds)

        # Solver
        if not uf.is_array(n_iter):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if x_0 is None:
            ref_mu_param = kwargs.get('ref_mu_param', None)
            ref_ker_param = kwargs.get('ref_ker_param', None)
            ref_imp_param = kwargs.get('ref_imp_param', None)
            range_ref_mu = kwargs.get('range_ref_mu', 0.1)
            range_ref_ker = kwargs.get('range_ref_ker', 0.1)
            range_ref_imp = kwargs.get('range_ref_imp', 0.1)
            min_mu_param = kwargs.get('min_mu_param', None)
            max_mu_param = kwargs.get('max_mu_param', None)
            target_bratio = kwargs.get('target_bratio', 0.6)
            max_omega = kwargs.get('max_omega', 1.)
            true_omega = kwargs.get('true_omega', None)
            max_ker_param = kwargs.get('max_ker_param', 5.)
            max_imp_param = kwargs.get('max_imp_param', 5.)

            mu_0, ker_0, imp_0 = self.get_random_param(ref_mu_param=ref_mu_param,
                                                       ref_ker_param=ref_ker_param,
                                                       ref_imp_param=ref_imp_param,
                                                       range_ref_mu=range_ref_mu,
                                                       range_ref_ker=range_ref_ker,
                                                       range_ref_imp=range_ref_imp,
                                                       min_mu_param=min_mu_param,
                                                       max_mu_param=max_mu_param,
                                                       target_bratio=target_bratio,
                                                       max_omega=max_omega,
                                                       true_omega=true_omega,
                                                       max_ker_param=max_ker_param,
                                                       max_imp_param=max_imp_param,
                                                       flatten=False, rng=rng)
            x_0 = self.matrix2tensor_params(mu_0, ker_0, imp_0)
        else:
            mu_0, ker_0, imp_0 = self.tensor2matrix_params(x_0)

        # Initialize Solvers
        if solvers is None:
            solvers = [ADAM(**kwargs) for k in range(d)]
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]

        # Initialize logger
        logger = OptimLogger(d, n_iter, **kwargs)
        self.init_logger(logger)

        # Precomp
        is_comp_imp = self.is_computable_imp(process_path)
        is_comp_S = self.is_computable_S(process_path)
        is_comp_eps = self.is_computable_eps(process_path)       

        # Scheme
        x = [None]*d
        for k in range(d):
            x_k = x_0[k]
            logger.log_param(k, 0, x_k)
            lower_bounds_k = lower_bnds[k]
            upper_bounds_k = upper_bnds[k]
            n_iter_k = n_iter[k]
            # Precomp
            mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k,
                                                                         x_k)
            self.init_precomp(k,  ker_param_k, imp_param_k, process_path,
                              is_comp_imp, is_comp_S, is_comp_eps)
            for t in tqdm(range(n_iter_k), disable=not verbose):
                # Compute LSE gradient estimate for parameters x_k
                g_t = self.compute_grad_lse_k(k, process_path, x_k=x_k)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration
                x_k = solvers[k].iterate(t, x_k, g_t)
                # Project into space of parameters
                x_k = np.clip(x_k, lower_bounds_k, upper_bounds_k)
                logger.log_param(k, t+1, x_k)
                # Update precomps
                mu_param_k, ker_param_k, imp_param_k = self.xk2matrix_params(k,
                                                                             x_k)
                self.update_precomp(k, ker_param_k, imp_param_k,
                                    process_path, is_comp_imp, is_comp_S,
                                    is_comp_eps)
            x[k] = x_k
        fitted_mu_param, fitted_ker_param, fitted_imp_param = self.tensor2matrix_params(x)
        if write:
            self.is_fitted = True
            self.fitted_mu_param = fitted_mu_param
            self.fitted_ker_param = fitted_ker_param
            self.fitted_imp_param = fitted_imp_param
            self.process_logs(logger)
            logger.mu_0 = mu_0
            logger.ker_0 = ker_0
            logger.imp_0 = imp_0
            self.fit_log = logger
        return fitted_mu_param, fitted_ker_param, fitted_imp_param

    def make_adjacency_matrix(self, kernel_param=None):
        """
        Compute the adjacency matrix of the MHP.

        The adjacency matrix :math:`A` of an MHP is the :math:`d\\times d`
        matrix of :math:`L_{1}` norms of kernels; that is, for all
        :math:`i,j \\in [d]` the corresponding entry of this matrix is given by

        .. math::
            A_{ij} := \\int_{[0,+\\infty]} |\\phi_{ij}(u)|du..


        Parameters
        ----------
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters at which to evaluate the adjacency
            matrix. The default is None, in that case fitted kernel
            parameters will be used if they are stored in the corresponding
            attribute of the MHP object.

        Raises
        ------
        ValueError
            Raise an error if the kernel parameters not specified and there
            are no kernel parameters saved as an atrribute.

        Returns
        -------
        adjacency : `numpy.ndarray`
            Adjacency matrix of the MHP.

        """
        log_fitted_adjacency = False
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
                log_fitted_adjacency = True
            else:
                raise ValueError("kernel_param must be specified.")
        d = self.d
        adjacency = [[self._kernel_matrix[i][j].l1_norm(kernel_param[i][j])
                      for j in range(d)] for i in range(d)]
        if log_fitted_adjacency:
            self.fitted_adjacency = adjacency
        return adjacency

    def get_branching_ratio(self, adjacency=None, kernel_param=None):
        """
        Compute the branching ratio of the MHP.

        The branching ratio of an MHP is equal to the spectral radius of
        its adjacency matrix; that is, the maximum of the absolute values of
        the eigenvalues of the adjacency matrix.

        Parameters
        ----------
        adjacency : `numpy.ndarray`, optional
            Adjacency matrix of the MHP. The default is None, in that case it
            will be computed.

        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters at which to evaluate the adjacency
            matrix. The default is None, in that case fitted kernel
            parameters will be used if they are stored in the corresponding
            attribute of the MHP object.

        Raises
        ------
        ValueError
            Raise an error if the adjacency matrix is not specified, the kernel
            parameters not specified and there are no kernel parameters saved
            as an atrribute.

        Returns
        -------
         adjacency : `float`
            Branching ratio of the MHP.

        """
        if adjacency is None:
            if kernel_param is None:
                if self.is_fitted:
                    kernel_param = self.fitted_ker_param
                else:
                    raise ValueError("kernel_param must be specified.")
            adjacency = self.make_adjacency_matrix(kernel_param)
        bratio = np.max(np.absolute(np.linalg.eigvals(adjacency)))
        return bratio

    def get_random_param(self, ref_mu_param=None, ref_ker_param=None,
                         ref_imp_param=None, range_ref_mu=0.1,
                         range_ref_ker=0.1, range_ref_imp=0.1,
                         min_mu_param=None, max_mu_param=None,
                         target_bratio=0.6, max_omega=1., true_omega=None,
                         max_ker_param=5., max_imp_param=5.,
                         flatten=False,
                         seed=1234,
                         rng=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        d = self.d

        mu_lower_bnds = self.get_mu_param_lower_bounds()
        mu_upper_bnds = self.get_mu_param_upper_bounds()
        ker_lower_bnds = self.get_ker_param_lower_bounds()
        ker_upper_bnds = self.get_ker_param_upper_bounds()
        imp_lower_bnds = self.get_imp_param_lower_bounds()
        imp_upper_bnds = self.get_imp_param_upper_bounds()

        # Mu
        if ref_mu_param is None:
            if min_mu_param is None:
                min_mu_param = copy.deepcopy(mu_lower_bnds)
            else:
                if not uf.is_array(min_mu_param):
                    min_mu_float = copy.deepcopy(min_mu_param)
                    min_mu_param = [min_mu_float*np.ones(self.vector_n_param_mu[i]) for i in range(d)]
            if max_mu_param is None:
                max_mu_param = [1.+min_mu_param[i] for i in range(d)]
            if not uf.is_array(max_mu_param):
                max_mu_float = copy.deepcopy(max_mu_param)
                max_mu_param = [max_mu_float*np.ones(self.vector_n_param_mu[i]) for i in range(d)]
            mu_param = np.array([np.zeros(self.vector_n_param_mu[i]) for i in range(d)], dtype=object)
            for i in range(d):
                n_param = self.vector_n_param_mu[i]
                for ix in range(n_param):
                    mu_param[i][ix] = rng.uniform(low=min_mu_param[i][ix],
                                                  high=max_mu_param[i][ix],
                                                  size=1)[0]
        else:
            dist_mu_bnds = [ref_mu_param[i]-mu_lower_bnds[i] for i in range(d)]
            lower_mu_params = [ref_mu_param[i]-range_ref_mu*dist_mu_bnds[i] for i in range(d)]
            upper_mu_params = [ref_mu_param[i] for i in range(d)]

            mu_param = np.array([np.zeros(self.vector_n_param_mu[i]) for i in range(d)], dtype=object)
            for i in range(d):
                mu_param[i][ix] = rng.uniform(low=lower_mu_params[i][ix],
                                              high=upper_mu_params[i][ix],
                                              size=1)[0]

        # Kernels
        kernel_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if ref_ker_param is None:
            if not isinstance(max_ker_param, (list, np.ndarray)):
                float_max = copy.deepcopy(max_ker_param)
                max_ker_param = [[[None for x
                                   in range(self.matrix_n_param_ker[i][j])]
                                  for j in range(d)] for i in range(d)]
                for i, j in itertools.product(range(d), range(d)):
                    n_param = self.matrix_n_param_ker[i][j]
                    vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                    for x in range(n_param):
                        if x in vec_ix_omega:
                            max_ker_param[i][j][x] = max_omega
                        else:
                            max_ker_param[i][j][x] = max(float_max,
                                                         ker_lower_bnds[i][j][x])

            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_ker[i][j]
                kernel_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=ker_lower_bnds[i][j][ix],
                                      high=max_ker_param[i][j][ix],
                                      size=1)[0]
                    kernel_param[i][j][ix] = val
        else:
            dist_ker_bnds = [[ref_ker_param[i][j]-ker_lower_bnds[i][j] for j in range(d)] for i in range(d)]
            lower_ker_params = [[ref_ker_param[i][j]-range_ref_ker*dist_ker_bnds[i][j] for j in range(d)] for i in range(d)]
            upper_ker_params = [[ref_ker_param[i][j]+range_ref_ker*dist_ker_bnds[i][j] for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_ker[i][j]
                kernel_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    val = rng.uniform(low=lower_ker_params[i][j][ix],
                                      high=upper_ker_params[i][j][ix],
                                      size=1)[0]
                    kernel_param[i][j][ix] = val
        # Rescaling
        branching_ratio = self.get_branching_ratio(kernel_param=kernel_param)
        if branching_ratio > 0.:
            scaling = target_bratio/branching_ratio
        for i, j in itertools.product(range(d), range(d)):
            kernel_param[i][j] = np.array(kernel_param[i][j])
            if branching_ratio > 0.:
                vec_ix_omega = self._kernel_matrix[i][j].ix_omegas()
                if len(vec_ix_omega) > 0:
                    kernel_param[i][j][vec_ix_omega] = (scaling
                                                        * kernel_param[i][j][vec_ix_omega])

        # Impacts
        impact_param = np.array([[None for j in range(d)]
                                 for i in range(d)], dtype=object)
        if ref_imp_param is None:
            if not uf.is_array(max_imp_param):
                float_max = copy.deepcopy(max_imp_param)
                max_imp_param = [[[None for x
                                   in range(self.matrix_n_param_imp[i][j])]
                                  for j in range(d)] for i in range(d)]
                for i, j in itertools.product(range(d), range(d)):
                    n_param = self.matrix_n_param_imp[i][j]
                    for ix in range(n_param):
                        max_imp_param[i][j][ix] = max(float_max,
                                                      imp_lower_bnds[i][j][ix])

            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_imp[i][j]
                impact_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    lo_lim = imp_lower_bnds[i][j][ix]
                    hi_lim = min(max_imp_param[i][j][ix],
                                 imp_upper_bnds[i][j][ix])
                    hi_lim = max(hi_lim, lo_lim)
                    val = rng.uniform(low=lo_lim, high=hi_lim, size=1)[0]
                    impact_param[i][j][ix] = val
        else:
            dist_imp_bnds = [[ref_imp_param[i][j]-imp_lower_bnds[i][j] for j in range(d)] for i in range(d)]
            lower_imp_params = [[ref_imp_param[i][j]-range_ref_imp*dist_imp_bnds[i][j] for j in range(d)] for i in range(d)]
            upper_imp_params = [[ref_imp_param[i][j]+range_ref_imp*dist_imp_bnds[i][j] for j in range(d)] for i in range(d)]
            for i, j in itertools.product(range(d), range(d)):
                n_param = self.matrix_n_param_imp[i][j]
                impact_param[i][j] = np.zeros(n_param)
                for ix in range(n_param):
                    lo_lim = lower_imp_params[i][j][ix]
                    hi_lim = min(upper_imp_params[i][j][ix],
                                 imp_upper_bnds[i][j][ix])
                    hi_lim = max(hi_lim, lo_lim)
                    val = rng.uniform(low=lo_lim, high=hi_lim, size=1)[0]
                    impact_param[i][j][ix] = val

        # Flatten
        if flatten:
            return self.matrix2tensor_params(mu_param, kernel_param,
                                             impact_param)
        else:
            return mu_param, kernel_param, impact_param

    # Residuals
    def get_residuals(self, process_path, mu_param=None, kernel_param=None,
                      impact_param=None, expected_impact_matrix=None,
                      verbose=False, write=True):
        """
        Compute the residuals of the model.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        Let :math:`k \\in [d]`, define the compensator of :math:`N^k`
        for all :math:`t \\geq 0` by

        .. math::
            \\Lambda_{k}(t):=\\int_{[0,t]}\\lambda_k(t)\\mathrm{~d} t.

        For all :math:`m \\in \\mathbb{N}^{*}, k \\in [d]`, let

        .. math::
            s_{m}^{k}=\\Lambda_{k}\\left(t_{m}^{k}\\right).

        For each :math:`k \\in[d]`, define the point process

        .. math::
            \\mathcal{S}^{k}:=\\left\\{s_{m}^{k}: m \\in \\mathbb{N}^{*}\\right\\};

        then :math:`\\left(\mathcal{S}^{k}\\right)_{k \\in[d]}` are independent
        standard Poisson processes. The inter-arrival times of
        :math:`\\mathcal{S}^{k}` ('residuals'), for a model that fits the data
        well must therefore be close to a standard exponential distribution.        

        Parameters
        ----------
        process_path : `aslsd.ProcessPath`
            DESCRIPTION.
        mu : `numpy.ndarray`, optional
            Vector of baseline parameters. The default is None, in that case
            fitted baseline parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        kernel_param : `numpy.ndarray`, optional
            Matrix of kernel parameters. The default is None, in that case
            fitted kernel parameters will be used if they are stored in the
            corresponding attribute of the MHP object.
        sampling : `bool`, optional
            If True, subsample the residuals. The default is False.
        sample_size : `int`, optional
            Size of the subsample of residuals. The default is 10**3. Only
            used if sampling is True.
        seed : `int`, optional
            Seed of the random number generator. The default is 1234. Only
            used if sampling is true.
        write : `bool`, optional
            If true, save computed residuals in the corresponding
            attributes of the MHP object. The default is True.
        verbose : `bool`, optional
            If True, print progress bar. The default is False.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        residuals : `list` of `numpy.ndarray`
            Residuals of the fitted model.

        """
        if mu_param is None:
            if self.is_fitted:
                mu_param = self.fitted_mu_param
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("mu_param must be specified.")

        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("kernel_param must be specified.")

        if impact_param is None:
            if self.is_fitted:
                impact_param = self.fitted_imp_param
            else:
                raise ValueError("impact_param must be specified.")
        if expected_impact_matrix is None:
            expected_impact_matrix = self.make_expected_impact(impact_param)
        # Path
        d = process_path.d
        list_times = process_path.list_times
        varpi = process_path.varpi
        kappa = process_path.kappa
        # Compute residuals
        residuals = [None]*d
        for k in range(d):
            if verbose:
                print('Computing residuals, dimension k=', str(k), ' ...')
            N_k = process_path.n_events[k]
            varpi_kj1 = np.array([process_path.varpi[k][j][1]
                                  for j in range(d)], dtype=int)
            kernel_part = np.zeros((d, N_k))
            U_ones = np.ones(N_k)
            # I. Computations for Poisson part
            times_comp_poisson = self.mu_compensator[k](list_times[k],
                                                        mu_param[k])
            poisson_residuals = times_comp_poisson[1:]-times_comp_poisson[:-1]
            # II. Kernels
            for j in range(d):
                kernel_kj = self.kernel_matrix[k][j]
                r_kj = kernel_kj.n_basis_ker
                ker_param_kj = kernel_param[k][j]
                ker_vars_kj = kernel_kj.make_vars(ker_param_kj)
                # Kappa vector
                kappa_vec = np.zeros(N_k)
                for m in range(varpi_kj1, N_k):
                    kappa_vec[m] = float(kappa[j][k][m])
                for q in range(r_kj):
                    omega_kjq = ker_vars_kj[2*q]
                    beta_kjq = ker_vars_kj[2*q+1]
                    S_one_kjk_q = get_exp_sum(k, j, beta_kjq, varpi, kappa,
                                              list_times, U_ones)
                    kernel_part[varpi_kj1:] = omega_kjq*(kappa_vec[varpi_kj1:]
                                                         - S_one_kjk_q[varpi_kj1:])
            # Group results
            kernel_residuals = (kernel_part[:, 1:]
                                - kernel_part[:, :-1]).sum(axis=0)
            residuals[k] = poisson_residuals+kernel_residuals

        # Save
        if self.is_fitted and write:
            self.fitted_expected_impact_matrix = expected_impact_matrix
            self.fit_residuals = residuals
        return residuals

    def ks_test_residuals(self, residuals=None):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.ks_test_residuals(residuals)

    def qq_plot(self, i, residuals=None, labels=None, style='exponential',
                substract_yx=False, normalize=False, max_points=None,
                display_line45=True, log_scale=False, ax=None, save=False,
                filename='image.png', show=False, **kwargs):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.qq_plot(residuals[i], n_models=1, labels=labels,
                           style=style, substract_yx=substract_yx,
                           normalize=normalize, max_points=max_points,
                           display_line45=display_line45, log_scale=log_scale,
                           ax=ax, save=save, filename=filename, show=show,
                           **kwargs)

    # Simulation
    def simulate(self, T_f, mu_param=None, kernel_param=None,
                 impact_param=None, rng=None, seed=1234,
                 verbose=False):
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

        if mu_param is None:
            mu_param = self.fitted_mu_param
            if mu_param is None:
                raise ValueError("Missing value for Baseline parameters")
        if kernel_param is None:
            kernel_param = self.fitted_ker_param
            if kernel_param is None:
                raise ValueError("Missing value for Kernel parameters")
        if impact_param is None:
            impact_param = self.fitted_imp_param
            if impact_param is None:
                raise ValueError("Missing value for Impact parameters")

        d = self.d
        offset_gens = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            offset_gens[i][j] = self._kernel_matrix[i][j].make_offset_gen(
                kernel_param[i][j])

        adjacency = self.make_adjacency_matrix(kernel_param)
        # RNG
        rng = us.make_rng(rng=rng, seed=seed)

        branching_ratio = self.get_branching_ratio(adjacency=adjacency)
        if branching_ratio >= 1:
            raise ValueError("Cannot simulate from unstable MHP: ",
                             "The branching ratio of this MHP is ",
                             branching_ratio, " > 1.")
        if verbose:
            print('Simulating events...')
        # Step 1. Generate immigrants
        # Location of immigrants
        generations = [[self._baselines_vec[i].simulate(T_f, mu_param[i],
                                                        rng=rng)] for i in range(d)]
        raw_marks = [[self.vec_marks[i].simulate(size=len(generations[i][0]),
                                                 rng=rng)] for i in range(d)]
        # generations is a list such that generations[i][ix_gen] contains
        # the times of events of type i of generation ix_gen

        def sum_generation(L, index):
            return sum([len(L[i][index]) for i in range(d)])

        ix_gen = 1
        #   Step 2. Fill via repeated generations
        while sum_generation(generations, ix_gen-1):
            for k in range(d):
                generations[k].append(np.array([]))
                raw_marks[k].append(np.array([]))
            for j in range(d):
                # Simulate the offspring of the "ix_gen-1"th generation of
                # events of type j
                if len(generations[j][ix_gen-1]) > 0:
                    for i in range(d):
                        # Set number of offspring
                        parent_marks = raw_marks[j][ix_gen-1]
                        parent_impacts = self._impact_matrix[i][j].impact(parent_marks,
                                                                          impact_param[i][j])
                        Noff = rng.poisson(adjacency[i][j]*parent_impacts,
                                           size=len(generations[j][ix_gen-1]))
                        parenttimes = generations[j][ix_gen-1].repeat(Noff)
                        offsets = offset_gens[i][j](rng, N=Noff.sum())
                        offspringtime = parenttimes + offsets
                        generations[i][ix_gen] = np.append(generations[i][ix_gen], np.array([x for x in offspringtime if x < T_f]))
                        n_valid_kids = len(np.array([x for x in offspringtime if x < T_f]))
                        if n_valid_kids > 0:
                            if len(raw_marks[i][ix_gen]) > 0:
                                raw_marks[i][ix_gen] = np.append(raw_marks[i][ix_gen],
                                                                 self.vec_marks[i].simulate(size=n_valid_kids,
                                                                                            rng=rng),
                                                                 axis=0)
                            else:
                                raw_marks[i][ix_gen] = self.vec_marks[i].simulate(size=n_valid_kids,
                                                                                            rng=rng)

            ix_gen += 1

        if verbose:
            print('Sorting results ...')
        list_times_ = [[x for sublist in generations[i]
                        for x in sublist] for i in range(d)]
        list_marks_ = [[x for sublist in raw_marks[i]
                   for x in sublist] for i in range(d)]

        list_marks = [np.array([x for _, x in sorted(zip(list_times_[i], list_marks_[i]))]) for i in range(d)]
        list_times = [np.array(sorted(list_times_[i])) for i in range(d)]
        for i in range(d):
            list_marks[i] = list_marks[i].reshape((len(list_times[i]),
                                                   self.vec_marks[i].get_mark_dim()))
        if verbose:
            n_tot = sum([len(L) for L in list_times])
            print('Simulation Complete, ', n_tot, ' events simulated.')
        return list_times, list_marks

    def simu_multipath(self, path_res, t_res, x_min, x_max, mu=None,
                       kernel_param=None, seed=1234, verbose=False,
                       disc_type='log', base_seed=1234):
        d = self.d
        rng = np.random.default_rng(base_seed)
        vec_seeds = rng.choice(10**5, size=path_res, replace=False)

        if disc_type == 'log':
            T_f = 10**x_max
        elif disc_type == 'linear':
            T_f = x_max
        list_Tf = uf.discretize_space(x_min, x_max, t_res, disc_type)
        list_paths = [[[] for j in range(path_res)] for i in range(t_res)]
        for j in range(path_res):
            seed = vec_seeds[j]
            list_times = self.simulate(T_f, mu=mu, kernel_param=kernel_param,
                                       seed=seed, verbose=verbose)
            for i in range(t_res):
                local_Tf = list_Tf[i]
                list_n_f = [bisect.bisect_left(list_times[index_dim],
                                               local_Tf)-1
                            for index_dim in range(d)]
                list_paths[i][j] = [list_times[0][:list_n_f[index_dim]+1]
                                    for index_dim in range(d)]
        return list_Tf, list_paths

    # Metrics
    # L2 projection
    def get_l2_projection(self, mhp_2, param_2, n_iter=1000,
                          solver=None, log_error=False, rng=None,
                          seed=1234,
                          verbose=False, **kwargs):
        return self._mhp.get_l2_projection(mhp_2, param_2, n_iter=1000,
                                           solver=solver, log_error=log_error,
                                           rng=rng, seed=1234, verbose=verbose,
                                           **kwargs)

    # Plots
    def plot_kernels(self, kernel_param=None, t_min=0., t_max=10.,
                     n_samples=1000, index_from_one=False, log_scale=False,
                     axs=None, save=False, filename='image.png',
                     show=False, **kwargs):
        if kernel_param is None:
            if self.is_fitted:
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("kernel_param must be specified.")
        return gt.plot_kernels(self.phi, kernel_param, t_min=t_min,
                               t_max=t_max, n_samples=n_samples,
                               index_from_one=index_from_one,
                               log_scale=log_scale, axs=axs,
                               save=save, filename=filename, show=show,
                               **kwargs)

    def plot_adjacency_matrix(self, adjacency=None, kernel_param=None,
                              event_names=None,
                              index_from_one=False, annotate=False,
                              cmap="Blues", save=False,
                              filename='image.png', show=True, **kwargs):
        if adjacency is None:
            if self.is_fitted:
                if self.fitted_adjacency is None:
                    if kernel_param is None:
                        kernel_param = self.fitted_ker_param
                    adjacency = self.make_adjacency_matrix()
                    self.fitted_adjacency = adjacency
                else:
                    adjacency = self.fitted_adjacency
            else:
                if kernel_param is not None:
                    adjacency = self.make_adjacency_matrix(kernel_param)
                else:
                    raise ValueError("adjacency must be specified.")

        return gt.plot_adjacency_matrix(adjacency, event_names=event_names,
                                        index_from_one=index_from_one,
                                        annotate=annotate, cmap=cmap,
                                        save=save, filename=filename,
                                        show=show, **kwargs)

    def plot_solver_path(self, true_mu=None, true_ker_param=None, min_mu=None,
                         min_ker_param=None, plot_derivatives=False,
                         derivatives_zero=False, axs=None, save=False,
                         filename='image.png', show=False, **kwargs):
        if not self.is_fitted:
            raise ValueError("MHP must be fitted before plotting solver path")
        fit_log = self.fit_log
        matrix_n_param = self.matrix_n_param
        mu_names = self.mu_names
        ker_param_names = self.ker_param_names

        return gt.plot_solver_path(fit_log, matrix_n_param, mu_names,
                                   ker_param_names, true_mu=true_mu,
                                   true_ker_param=true_ker_param,
                                   min_mu=min_mu, min_ker_param=min_ker_param,
                                   plot_derivatives=plot_derivatives,
                                   derivatives_zero=derivatives_zero,
                                   axs=axs, save=save, filename=filename,
                                   show=show, **kwargs)

    # Serialization
    def save(self, file, **kwargs):
        # Parameters
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_out = open(file_mu, "wb", **kwargs)
        pickle.dump(self.fitted_mu_param, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        pickle_out = open(file_ker, "wb", **kwargs)
        pickle.dump(self.fitted_ker_param, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_imp = file+'_fitted_imp.pickle'
        else:
            file_imp = file+'_fitted_imp'
        pickle_out = open(file_imp, "wb", **kwargs)
        pickle.dump(self.fitted_imp_param, pickle_out)
        pickle_out.close()

        # Residuals
        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        pickle_out = open(file_residuals, "wb", **kwargs)
        pickle.dump(self.fit_residuals, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        pickle_out = open(file_adjacency, "wb", **kwargs)
        pickle.dump(self.fitted_adjacency, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_fit_log = file+'_fit_log.pickle'
        else:
            file_fit_log = file+'_fit_log'
        pickle_out = open(file_fit_log, "wb", **kwargs)
        pickle.dump(self.fit_log, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        # Parameters
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_in = open(file_mu, "rb")
        fitted_mu_param = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        file_ker = file+'_fitted_ker'
        pickle_in = open(file_ker, "rb")
        fitted_ker_param = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_imp = file+'_fitted_imp.pickle'
        else:
            file_imp = file+'_fitted_imp'
        file_imp = file+'_fitted_imp'
        pickle_in = open(file_imp, "rb")
        fitted_imp_param = pickle.load(pickle_in)

        # Residuals
        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        pickle_in = open(file_residuals, "rb")
        fitted_residuals = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        pickle_in = open(file_adjacency, "rb")
        fitted_adjacency = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_fit_log = file+'_fit_log.pickle'
        else:
            file_fit_log = file+'_fit_log'
        pickle_in = open(file_fit_log, "rb")
        fit_log = pickle.load(pickle_in)

        self.clear_fit()

        self.is_fitted = True
        self.fitted_mu_param = fitted_mu_param
        self.fitted_ker_param = fitted_ker_param
        self.fitted_imp_param = fitted_imp_param
        self.fit_residuals = fitted_residuals
        self.fitted_adjacency = fitted_adjacency
        self.fit_log = fit_log
