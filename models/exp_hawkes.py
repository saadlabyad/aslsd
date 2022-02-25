# License: BSD 3 clause

import copy
import itertools

import numpy as np

from aslsd.models.mhp import MHP
from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.utilities import useful_functions as uf


def update_exp_sum_recurrence(E_m, m, beta, kappa, list_times, i, j):
    # E_m:=\sum_{n=1}^{\kappa(j,i,m)} \exp(-beta (t_m-t_n))
    # return E_{m+1} given E_{m}
    t_m1 = list_times[i][m+1]
    t_m = list_times[i][m]
    exp_term = np.exp(-beta*(t_m1-t_m))*E_m
    if kappa[j][i][m] == kappa[j][i][m+1]:
        return exp_term
    else:
        return exp_term+np.sum(np.exp(-beta*(t_m1-list_times[j][kappa[j][i][m]+1:kappa[j][i][m+1]+1])))


def get_exp_sum_reccurence_lag_h(beta, kappa, varpi, list_times, i, j, h):
    #   Compute \sum_{n=varpi[i][j][h]}^{N^i_T} \exp(-beta (t_m-t_{\kappa(j,i,m)-h+1}))
    indices_m = [m for m in range(varpi[i][j][h], len(list_times[i]))]
    indices_n = [kappa[j][i][m]-h+1 for m in indices_m]
    return np.sum(np.exp(-beta*(list_times[i][indices_m]-list_times[j][indices_n])))


def get_exp_lse_k(d, k, p, r, T_f, list_times, list_times2end, n_events, varpi,
                  kappa, eta, mhp, x_k):
    # Exact LSE of an Exponential MHP
    mu_k = x_k[0]
    eta_k = eta[k]

    res = mu_k**2-2*eta_k*mu_k

    for i in range(d):
        x_ki = x_k[mhp.interval_map[k][i][0]:mhp.interval_map[k][i][1]]
        #   Psi_ki(T-t^i_m)
        res += 2*mu_k*np.sum(mhp.psi[k][i](list_times2end[i], x_ki))/T_f
        #   Upsilon_iik(T-t^i_m,0)
        res += np.sum(mhp.upsilon[i][i][k](list_times2end[i], 0., x_ki, x_ki))/T_f

    # Upsilon
    for i, j in itertools.product(range(d), range(d)):
        x_ki = x_k[mhp.interval_map[k][i][0]:mhp.interval_map[k][i][1]]
        r_ki = int(len(x_ki)/2)
        x_kj = x_k[mhp.interval_map[k][j][0]:mhp.interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        for ix_ker1, ix_ker2 in itertools.product(range(r_ki), range(r_kj)):
            upsilon_term = 0.
            omega_ki = x_ki[2*ix_ker1]
            beta_ki = x_ki[2*ix_ker1+1]
            omega_kj = x_kj[2*ix_ker2]
            beta_kj = x_kj[2*ix_ker2+1]
            E_m = np.sum(np.exp(-beta_kj*(list_times[i][varpi[i][j][1]]-list_times[j][:kappa[j][i][varpi[i][j][1]]+1])))
            upsilon_term += E_m*(1.-np.exp(-(beta_ki+beta_kj)*(T_f-list_times[i][varpi[i][j][1]])))
            for m in range(varpi[i][j][1]+1, n_events[i]):
                t_m = list_times[i][m]
                E_m = update_exp_sum_recurrence(E_m, m-1, beta_kj, kappa, list_times, i, j)
                upsilon_term += E_m*(1.-np.exp(-(beta_ki+beta_kj)*(T_f-t_m)))
            upsilon_term = omega_ki*omega_kj*beta_ki*beta_kj*(1./(beta_ki+beta_kj))*upsilon_term
            res += 2.*(upsilon_term/T_f)

    # Phi
    for j in range(d):
        x_kj = x_k[mhp.interval_map[k][j][0]:mhp.interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        for ix_ker2 in range(r_kj):
            omega_kj = x_kj[2*ix_ker2]
            beta_kj = x_kj[2*ix_ker2+1]

            term_phi = 0.
            E_m = np.sum(np.exp(-beta_kj*(list_times[k][varpi[k][j][1]]-list_times[j][:kappa[j][k][varpi[k][j][1]]+1])))
            term_phi = E_m
            for m in range(varpi[k][j][1]+1, n_events[k]):
                t_m = list_times[k][m]
                E_m = update_exp_sum_recurrence(E_m, m-1, beta_kj, kappa, list_times, k, j)
                term_phi += E_m
            term_phi = omega_kj*beta_kj*term_phi

            res -= (2./T_f)*term_phi

    return res


def diff_exp_lse_omega_kpr(d, k, p, r, T_f, list_times, list_times2end,
                           n_events, varpi, kappa, mhp, x_k):
    x_kp = x_k[mhp.interval_map[k][p][0]:mhp.interval_map[k][p][1]]
    r_kp = int(len(x_kp)/2)
    ix_ker = r//2
    omega_kpr = x_kp[2*ix_ker]
    beta_kpr = x_kp[2*ix_ker+1]
    # Exact derivative of the LSE of an Exponential MHP with respect to the
    # rth parameter of phi_kp
    res = 0.
    # Cross Upsilon
    # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
    for i in itertools.chain(range(p), range(p+1, d)):
        x_ki = x_k[mhp.interval_map[k][i][0]:mhp.interval_map[k][i][1]]
        r_ki = int(len(x_ki)/2)
        for ix_ker1 in range(r_ki):
            diff_term = 0.
            omega_kil = x_ki[2*ix_ker1]
            beta_kil = x_ki[2*ix_ker1+1]
            E_m = np.sum(np.exp(-beta_kpr*(list_times[i][varpi[i][p][1]]-list_times[p][:kappa[p][i][varpi[i][p][1]]+1])))
            diff_term += E_m*(1.-np.exp(-(beta_kil+beta_kpr)*(T_f-list_times[i][varpi[i][p][1]])))
            for m in range(varpi[i][p][1]+1, n_events[i]):
                t_m = list_times[i][m]
                E_m = update_exp_sum_recurrence(E_m, m-1, beta_kpr, kappa, list_times, i, p)
                diff_term += E_m*(1.-np.exp(-(beta_kil+beta_kpr)*(T_f-t_m)))
            diff_term = omega_kil*beta_kil*beta_kpr*(1./(beta_kil+beta_kpr))*diff_term
            res += (2./T_f)*diff_term

    # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
    for j in itertools.chain(range(p), range(p+1, d)):
        x_kj = x_k[mhp.interval_map[k][j][0]:mhp.interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        for ix_ker2 in range(r_kj):
            diff_term = 0.
            omega_kjl = x_kj[2*ix_ker2]
            beta_kjl = x_kj[2*ix_ker2+1]
            E_m = np.sum(np.exp(-beta_kjl*(list_times[p][varpi[p][j][1]]-list_times[j][:kappa[j][p][varpi[p][j][1]]+1])))
            diff_term += E_m*(1.-np.exp(-(beta_kpr+beta_kjl)*(T_f-list_times[p][varpi[p][j][1]])))
            for m in range(varpi[p][j][1]+1, n_events[p]):
                t_m = list_times[p][m]
                E_m = update_exp_sum_recurrence(E_m, m-1, beta_kjl, kappa, list_times, p, j)
                diff_term += E_m*(1.-np.exp(-(beta_kpr+beta_kjl)*(T_f-t_m)))
            diff_term = omega_kjl*beta_kpr*beta_kjl*(1./(beta_kpr+beta_kjl))*diff_term
            res += (2./T_f)*diff_term

    # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
    diff_term_sim_upsilon = 0.
    E_m = np.exp(-beta_kpr*(list_times[p][1]-list_times[p][0]))
    diff_term_sim_upsilon += E_m*(1.-np.exp(-2*beta_kpr*(T_f-list_times[p][1])))
    for m in range(2, n_events[p]):
        t_m = list_times[p][m]
        E_m = update_exp_sum_recurrence(E_m, m-1, beta_kpr, kappa, list_times,
                                        p, p)
        diff_term_sim_upsilon += E_m*(1.-np.exp(-2*beta_kpr*(T_f-t_m)))
    diff_term_sim_upsilon *= omega_kpr*beta_kpr
    res += (2./T_f)*diff_term_sim_upsilon

    for ix_ker1 in itertools.chain(range(ix_ker), range(ix_ker+1, r_kp)):
        diff_term_sim_upsilon = 0.
        omega_kpl = x_kp[2*ix_ker1]
        beta_kpl = x_kp[2*ix_ker1+1]
        E_m = np.sum(np.exp(-beta_kpl*(list_times[p][varpi[p][p][1]]-list_times[p][:kappa[p][p][varpi[p][p][1]]+1])))
        diff_term_sim_upsilon += E_m*(1.-np.exp(-(beta_kpr+beta_kpl)*(T_f-list_times[p][varpi[p][p][1]])))
        for m in range(varpi[p][p][1]+1, n_events[p]):
            t_m = list_times[p][m]
            E_m = update_exp_sum_recurrence(E_m, m-1, beta_kpl, kappa,
                                            list_times, p, p)
            diff_term_sim_upsilon += E_m*(1.-np.exp(-(beta_kpr+beta_kpl)*(T_f-t_m)))
        diff_term_sim_upsilon = omega_kpl*beta_kpr*beta_kpl*(1./(beta_kpr+beta_kpl))*diff_term
        res += (2./T_f)*diff_term

    for ix_ker2 in itertools.chain(range(ix_ker), range(ix_ker+1, r_kp)):
        diff_term_sim_upsilon = 0.
        omega_kpl = x_kp[2*ix_ker2]
        beta_kpl = x_kp[2*ix_ker2+1]
        E_m = np.sum(np.exp(-beta_kpr*(list_times[p][varpi[p][p][1]]-list_times[p][:kappa[p][p][varpi[p][p][1]]+1])))
        diff_term_sim_upsilon += E_m*(1.-np.exp(-(beta_kpr+beta_kpl)*(T_f-list_times[p][varpi[p][p][1]])))
        for m in range(varpi[p][p][1]+1, n_events[p]):
            t_m = list_times[p][m]
            E_m = update_exp_sum_recurrence(E_m, m-1, beta_kpr, kappa, list_times, p, p)
            diff_term_sim_upsilon += E_m*(1.-np.exp(-(beta_kpr+beta_kpl)*(T_f-t_m)))
        diff_term_sim_upsilon = omega_kpl*beta_kpr*beta_kpl*(1./(beta_kpr+beta_kpl))*diff_term
        res += 2.*(diff_term/T_f)

    # Term 4 : Phi_{kp}
    diff_term_phi = 0.
    E_m = np.sum(np.exp(-beta_kpr*(list_times[k][varpi[k][p][1]]-list_times[p][:kappa[p][k][varpi[k][p][1]]+1])))
    diff_term_phi = E_m
    for m in range(varpi[k][p][1]+1, n_events[k]):
        t_m = list_times[k][m]
        E_m = update_exp_sum_recurrence(E_m, m-1, beta_kpr, kappa, list_times, k, p)
        diff_term_phi += E_m
    diff_term_phi = beta_kpr*diff_term_phi
    res -= (2./T_f)*diff_term_phi

    # Term 5 : Psi_{kp}
    mu_k = x_k[0]
    diff_term_psi = np.sum(mhp.diff_psi[k][p](list_times2end[p], r, x_kp))
    res += (2./T_f)*mu_k*diff_term_psi

    # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
    diff_term_sim_upsilonzero = np.sum(mhp.diff_sim_upsilon[k][p](list_times2end[p], 0., r, x_kp))
    res += diff_term_sim_upsilonzero/T_f
    return res


def diff_exp_lse_theta_kpr(d, k, p, r, T_f, list_times, list_times2end,
                           n_events, varpi, kappa, eta, mhp, x_k):
    if (r % 2) == 0:
        return diff_exp_lse_omega_kpr(d, k, p, r, T_f, list_times,
                                      list_times2end, n_events, varpi, kappa,
                                      mhp, x_k)
    else:
        def func(x):
            y_k = copy.deepcopy(x_k)
            y_k[mhp.interval_map[k][p][0]+r] = x
            return get_exp_lse_k(d, k, p, r, T_f, list_times, list_times2end,
                                 n_events, varpi, kappa, eta, mhp, y_k)
        z = x_k[mhp.interval_map[k][p][0]+r]
        return uf.finite_diff(func, z, epsilon=10**-3,
                              diff_type='central difference', diff_index=0)


class RecurrentExponential:
    def __init__(self, mhp):

        if type(mhp) != MHP:
            raise ValueError("The constructor of RecurrentExponential ",
                             "objects takes MHP objects in argument")
        d = mhp.d
        for i, j in itertools.product(range(d), range(d)):
            for basis_kernel in mhp.kernel_matrix[i][j]._basis_kernels:
                if type(basis_kernel) != ExponentialKernel:
                    raise ValueError("All basis kernels must be exponential")
        self.mhp = mhp

    def diff_lse_mu_k(self, k, x_k, list_times2end, T_f, eta):
        d = self.mhp.d
        mu_k = x_k[0]
        eta_k = eta[k]
        res = 2*(mu_k-eta_k)
        for i in range(d):
            x_ki = x_k[self.mhp.interval_map[k][i][0]:self.mhp.interval_map[k][i][1]]
            res += 2.*(np.sum(self.mhp.psi[k][i](list_times2end[i], x_ki))
                       / T_f)
        return res

    def lse_k_grad(self, k, x_k, T_f, list_times, list_times2end, n_events,
                   varpi, kappa, eta):
        d = self.mhp.d
        n_param_k = len(x_k)
        grad = np.zeros(n_param_k)

        # Derivative with respect to \mu_k
        grad[0] = self.diff_lse_mu_k(k, x_k, list_times2end, T_f, eta)

        # Derivative with respect to kernel parameters
        for ix_param in range(1, n_param_k):
            p = self.mhp.ix_map[k][ix_param]['ker']
            r = self.mhp.ix_map[k][ix_param]['par']
            grad[ix_param] = diff_exp_lse_theta_kpr(d, k, p, r, T_f,
                                                    list_times, list_times2end,
                                                    n_events, varpi, kappa,
                                                    eta, self.mhp, x_k)

        return grad
