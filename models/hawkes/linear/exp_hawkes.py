# License: BSD 3 clause

import copy
import itertools
import pickle

import numpy as np
from tqdm import tqdm

from aslsd.models.mhp import MHP
from aslsd.basis_kernels.basis_kernel_exponential import ExponentialKernel
from aslsd.evaluation import goodness_of_fit as gof
from aslsd.events.process_path import ProcessPath
from aslsd.optim_logging.optim_logger import OptimLogger
from aslsd.plots import graphic_tools as gt
from aslsd.solvers.adam import ADAM
from aslsd.solvers.solver import Solver
from aslsd.utilities import useful_functions as uf

# Exponential Markov tricks
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


# LSE functions
def get_sum_psi_ki(d, k, i, psi, x_ki, process_path=None, T_f=None,
                   list_times2end=None, normalize=True):
    if process_path is not None:
        T_f = process_path.T_f
        list_times2end = process_path.list_times2end

    norm = T_f**-float(normalize)
    return np.sum(psi[k][i](list_times2end[i], x_ki))*norm


def get_sum_upsilonzero_ki(d, k, i, upsilon, x_ki, process_path=None, T_f=None,
                           list_times2end=None, normalize=True):
    if process_path is not None:
        T_f = process_path.T_f
        list_times2end = process_path.list_times2end

    norm = T_f**-float(normalize)
    #   Upsilon_iik(T-t^i_m,0)
    return np.sum(upsilon[i][i][k](list_times2end[i], 0., x_ki, x_ki))*norm


def get_sum_phi_kjr(k, j, omega_kj, beta_kj, process_path=None,
                    list_times=None,
                    T_f=None, n_events=None, kappa=None, varpi=None,
                    normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi

    E_m = np.sum(np.exp(-beta_kj*(list_times[k][varpi[k][j][1]]-list_times[j][:kappa[j][k][varpi[k][j][1]]+1])))
    res = E_m
    for m in range(varpi[k][j][1]+1, n_events[k]):
        E_m = update_exp_sum_recurrence(E_m, m-1, beta_kj, kappa, list_times,
                                        k, j)
        res += E_m
    res = omega_kj*beta_kj*res
    norm = T_f**-float(normalize)
    res *= norm
    return res


def get_sum_phi_kj(k, j, r_kj, x_kj, process_path=None,
                   list_times=None,
                   T_f=None, n_events=None, kappa=None, varpi=None,
                   normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi

    res = 0.
    for ix_ker in range(r_kj):
        omega_kj = x_kj[2*ix_ker]
        beta_kj = x_kj[2*ix_ker+1]
        res += get_sum_phi_kjr(k, j, omega_kj, beta_kj, list_times=list_times,
                               T_f=T_f, n_events=n_events, kappa=kappa,
                               varpi=varpi, normalize=normalize)
    return res


def get_sum_phi_k(d, k, x_k, interval_map, process_path=None, list_times=None,
                  T_f=None, n_events=None, kappa=None, varpi=None,
                  normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi

    res = 0.
    for j in range(d):
        x_kj = x_k[interval_map[k][j][0]:interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        res += get_sum_phi_kj(k, j, r_kj, x_kj, list_times=list_times,
                              T_f=T_f, n_events=n_events, kappa=kappa,
                              varpi=varpi, normalize=normalize)
    return res


def get_sum_upsilon_ijkr(i, j, omega_ki, beta_ki, omega_kj, beta_kj,
                         process_path=None,
                         list_times=None,
                         T_f=None, n_events=None, kappa=None, varpi=None,
                         normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi

    upsilon_term = 0.
    E_m = np.sum(np.exp(-beta_kj*(list_times[i][varpi[i][j][1]]-list_times[j][:kappa[j][i][varpi[i][j][1]]+1])))
    upsilon_term += E_m*(1.-np.exp(-(beta_ki+beta_kj)*(T_f-list_times[i][varpi[i][j][1]])))
    for m in range(varpi[i][j][1]+1, n_events[i]):
        t_m = list_times[i][m]
        E_m = update_exp_sum_recurrence(E_m, m-1, beta_kj, kappa, list_times,
                                        i, j)
        upsilon_term += E_m*(1.-np.exp(-(beta_ki+beta_kj)*(T_f-t_m)))
    upsilon_term = omega_ki*omega_kj*beta_ki*beta_kj*(1./(beta_ki+beta_kj))*upsilon_term
    norm = T_f**-float(normalize)
    upsilon_term *= norm
    return upsilon_term


def get_sum_upsilon_ijk(i, j, k, x_ki, x_kj, r_ki, r_kj,
                        process_path=None,
                        list_times=None,
                        T_f=None, n_events=None, kappa=None, varpi=None,
                        normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi
    res = 0.
    for ix_ker1, ix_ker2 in itertools.product(range(r_ki), range(r_kj)):
        omega_ki, beta_ki = x_ki[2*ix_ker1:2*ix_ker1+2]
        omega_kj, beta_kj = x_kj[2*ix_ker2:2*ix_ker2+2]
        res += get_sum_upsilon_ijkr(i, j, omega_ki, beta_ki, omega_kj, beta_kj,
                                    list_times=list_times, T_f=T_f,
                                    n_events=n_events, kappa=kappa,
                                    varpi=varpi, normalize=normalize)
    return res


def get_sum_upsilon_k(d, k, x_k, interval_map, process_path=None,
                      list_times=None,
                      T_f=None, n_events=None, kappa=None, varpi=None,
                      normalize=True):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi
    res = 0.
    for i, j in itertools.product(range(d), range(d)):
        x_ki = x_k[interval_map[k][i][0]:interval_map[k][i][1]]
        r_ki = int(len(x_ki)/2)
        x_kj = x_k[interval_map[k][j][0]:interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        res += get_sum_upsilon_ijk(i, j, k, x_ki, x_kj, r_ki, r_kj,
                                   list_times=list_times,
                                   T_f=T_f, n_events=n_events, kappa=kappa,
                                   varpi=varpi, normalize=normalize)
    return res


def get_exp_lse_k(d, k, x_k, interval_map, psi, upsilon,
                  process_path=None, T_f=None, list_times=None,
                  list_times2end=None, n_events=None, varpi=None, kappa=None,
                  eta_k=None, eta=None):
    # Exact LSE_k of an Exponential MHP
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi
        eta = process_path.eta

    if eta_k is None:
        eta_k = eta[k]

    mu_k = x_k[0]
    res = mu_k**2-2*eta_k*mu_k
    for i in range(d):
        x_ki = x_k[interval_map[k][i][0]:interval_map[k][i][1]]
        #   Psi_ki(T-t^i_m)
        res += mu_k*2*get_sum_psi_ki(d, k, i, psi, x_ki, T_f=T_f,
                                     list_times2end=list_times2end,
                                     normalize=True)
        #   Upsilon_iik(T-t^i_m,0)
        res += get_sum_upsilonzero_ki(d, k, i, upsilon, x_ki, T_f=T_f,
                                      list_times2end=list_times2end,
                                      normalize=True)

    # Upsilon
    res += 2.*get_sum_upsilon_k(d, k, x_k, interval_map, list_times=list_times,
                                T_f=T_f, n_events=n_events, kappa=kappa,
                                varpi=varpi, normalize=True)

    # Phi
    res -= 2.*get_sum_phi_k(d, k, x_k, interval_map, list_times=list_times,
                            T_f=T_f, n_events=n_events, kappa=kappa,
                            varpi=varpi, normalize=True)

    return res


# Derivatives of LSE functions
def diff_exp_lse_mu_k(d, k, x_k, interval_map, psi,
                      process_path=None, list_times2end=None, T_f=None,
                      eta_k=None, eta=None):

    # Time quantities
    if process_path is None:
        if eta_k is None:
            if eta is None:
                eta_k = len(list_times2end[k])/T_f
            else:
                eta_k = eta[k]
    else:
        list_times2end = process_path.list_times2end
        T_f = process_path.T_f
        eta_k = process_path.eta[k]

    mu_k = x_k[0]
    res = 2*(mu_k-eta_k)
    for i in range(d):
        x_ki = x_k[interval_map[k][i][0]:interval_map[k][i][1]]
        res += 2.*(np.sum(psi[k][i](list_times2end[i], x_ki))
                   / T_f)
    return res


def diff_exp_lse_omega_kpr(d, k, p, ix_ker, x_k, interval_map,
                           process_path=None, T_f=None,
                           list_times=None, list_times2end=None,
                           n_events=None, varpi=None, kappa=None):
    # Time quantities
    if process_path is not None:
        list_times = process_path.list_times
        list_times2end = process_path.list_times2end
        T_f = process_path.T_f
        n_events = process_path.n_events
        kappa = process_path.kappa
        varpi = process_path.varpi

    x_kp = x_k[interval_map[k][p][0]:interval_map[k][p][1]]
    mu_k = x_k[0]
    r_kp = int(len(x_kp)/2)
    omega_kpr = x_kp[2*ix_ker]
    beta_kpr = x_kp[2*ix_ker+1]
    # Exact derivative of the LSE of an Exponential MHP with respect to the
    # rth parameter of phi_kp
    res = 0.
    # Cross Upsilon
    # Loop 1: Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
    for i in itertools.chain(range(p), range(p+1, d)):
        x_ki = x_k[interval_map[k][i][0]:interval_map[k][i][1]]
        r_ki = int(len(x_ki)/2)
        x_kj = [1., beta_kpr]
        r_kj = 1
        res += 2.*get_sum_upsilon_ijk(i, p, k, x_ki, x_kj, r_ki, r_kj,
                                      list_times=list_times,
                                      T_f=T_f, n_events=n_events,
                                      kappa=kappa,
                                      varpi=varpi, normalize=True)

    # Loop 2: Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
    for j in itertools.chain(range(p), range(p+1, d)):
        x_ki = [1., beta_kpr]
        r_ki = 1
        x_kj = x_k[interval_map[k][j][0]:interval_map[k][j][1]]
        r_kj = int(len(x_kj)/2)
        res += 2.*get_sum_upsilon_ijk(p, j, k, x_ki, x_kj, r_ki, r_kj,
                                      list_times=list_times,
                                      T_f=T_f, n_events=n_events,
                                      kappa=kappa,
                                      varpi=varpi, normalize=True)

    # Term 3 : Upsilon_{ppk}=phi_{kp}phi_{kp}
    x_ki = [1., beta_kpr]
    r_ki = 1
    x_kj = [1., beta_kpr]
    r_kj = 1
    res += 4.*omega_kpr*get_sum_upsilon_ijk(p, p, k, x_ki, x_kj, r_ki, r_kj,
                                            list_times=list_times,
                                            T_f=T_f, n_events=n_events,
                                            kappa=kappa,
                                            varpi=varpi, normalize=True)

    for ix_ker1 in itertools.chain(range(ix_ker), range(ix_ker+1, r_kp)):
        omega_kpl = x_kp[2*ix_ker1]
        beta_kpl = x_kp[2*ix_ker1+1]
        x_ki = [omega_kpl, beta_kpl]
        r_ki = 1
        x_kj = [1., beta_kpr]
        r_kj = 1
        res += 2.*get_sum_upsilon_ijk(p, p, k, x_ki, x_kj, r_ki, r_kj,
                                      list_times=list_times,
                                      T_f=T_f, n_events=n_events,
                                      kappa=kappa,
                                      varpi=varpi, normalize=True)

    for ix_ker2 in itertools.chain(range(ix_ker), range(ix_ker+1, r_kp)):
        omega_kpl = x_kp[2*ix_ker2]
        beta_kpl = x_kp[2*ix_ker2+1]
        x_ki = [1., beta_kpr]
        r_ki = 1
        x_kj = [omega_kpl, beta_kpl]
        r_kj = 1
        res += 2.*get_sum_upsilon_ijk(p, p, k, x_ki, x_kj, r_ki, r_kj,
                                      list_times=list_times,
                                      T_f=T_f, n_events=n_events,
                                      kappa=kappa,
                                      varpi=varpi, normalize=True)

    # Term 4 : Phi_{kp}
    res -= 2.*get_sum_phi_kj(k, p, 1, [1., beta_kpr],
                             list_times=list_times,
                             T_f=T_f, n_events=n_events, kappa=kappa,
                             varpi=varpi,
                             normalize=True)

    # Term 5 : Psi_{kp}
    res += mu_k*2*np.sum(1.-np.exp(-beta_kpr*list_times2end[p]))/T_f

    # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
    res += omega_kpr*beta_kpr*np.sum(1.-np.exp(-2.*beta_kpr*list_times2end[p]))/T_f

    return res


def diff_exp_lse_theta_kpr(d, k, p, r, x_k, interval_map, psi=None,
                           upsilon=None, process_path=None,
                           T_f=None, list_times=None, list_times2end=None,
                           n_events=None, varpi=None, kappa=None, eta_k=None,
                           eta=None, epsilon=10**-3,
                           diff_type='central difference'):
    if (r % 2) == 0:
        ix_ker = r//2
        return diff_exp_lse_omega_kpr(d, k, p, ix_ker, x_k, interval_map,
                                      process_path=process_path, T_f=T_f,
                                      list_times=list_times,
                                      list_times2end=list_times2end,
                                      n_events=n_events, varpi=varpi,
                                      kappa=kappa)
    else:
        def func(x):
            y_k = copy.deepcopy(x_k)
            y_k[interval_map[k][p][0]+r] = x
            return get_exp_lse_k(d, k, y_k, interval_map, psi, upsilon,
                                 process_path=process_path, T_f=T_f,
                                 list_times=list_times,
                                 list_times2end=list_times2end,
                                 n_events=n_events, varpi=varpi, kappa=kappa,
                                 eta_k=eta_k, eta=eta)
        z = x_k[interval_map[k][p][0]+r]
        return uf.finite_diff(func, z, epsilon=10**-3,
                              diff_type='central difference', diff_index=0)


class RecurrentExponential:
    def __init__(self, _kernel_matrix, d=None, matrix_n_param=None,
                 n_ker_param=None, ix_map=None, interval_map=None,
                 mu_names=None, ker_param_names=None,
                 param_bounds=None, phi=None, diff_phi=None, psi=None,
                 diff_psi=None, upsilon=None, diff_sim_upsilon=None,
                 diff_cross_upsilon=None, is_fitted=False,
                 fitted_mu=None, fitted_ker_param=None,
                 fit_residuals=None, fitted_adjacency=None, fit_log=None):
        """
        Constructor of objects of class MHP.

        Parameters
        ----------
        _kernel_matrix : `list` of `list` of `KernelModel`
            Matrix of kernel models.

        """
        self.kernel_matrix = _kernel_matrix

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
        self._kernel_matrix = M
        d = len(M)
        self.d = d
        for i, j in itertools.product(range(d), range(d)):
            for basis_kernel in M[i][j]._basis_kernels:
                if type(basis_kernel) != ExponentialKernel:
                    raise ValueError("All basis kernels must be exponential")
        self._mhp = MHP(M)
        self.matrix_n_param = self.get_n_param()
        self.n_ker_param = sum([sum(self.matrix_n_param[i])
                                for i in range(self.d)])

        ix_map, interval_map = self.make_maps()
        self.ix_map = ix_map
        self.interval_map = interval_map

        param_names = self.get_param_names()
        self.mu_names = param_names['mu']
        self.ker_param_names = param_names['kernels']
        self.param_bounds = self.get_param_bounds()
        self.make_kernel_functionals()
        self.clear_fit()

    @kernel_matrix.deleter
    def kernel_matrix(self):
        del self._kernel_matrix

    # N params
    def get_n_param(self):
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
        return self._mhp.get_n_param()

    # Parameters map
    def make_maps(self):
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
        return self._mhp.make_maps()

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
        mu : TYPE
            Vector of background rates.
        kernel_param : TYPE
            Matrix of kernel parameters.

        """
        return self._mhp.tensor2matrix_params(tensor_param)

    def matrix2tensor_params(self, mu, kernel_param):
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
        return self._mhp.matrix2tensor_params(mu, kernel_param)

    def tensor2matrix_solverpaths(self, tensor_paths):
        #   Convert the list of parameters
        #   tensor_paths is a tensor of paths of solvers per parameters
        return self._mhp.tensor2matrix_solverpaths(tensor_paths)

    # Omega
    def is_sbf(self):
        return self._mhp.is_sbf()

    # Bounds
    def get_param_bounds(self):
        return self._mhp.get_param_bounds()

    # Param names
    def get_param_names(self, index_from_one=False):
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
        return self._mhp.get_param_names(index_from_one=index_from_one)

    # Kernel functionals
    def make_kernel_functionals(self):
        d = self.d
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
                func = kernel_ki.make_upsilon()

                def upsilon(t, s, params_1, params_2):
                    return func(kernel_kj, t, s, params_1, params_2)
                self.upsilon[i][j][k] = upsilon
                diff_func = kernel_ki.make_diff_cross_upsilon()

                def diff_cross_upsilon(t, s, ix_func, ix_diff, params_1,
                                       params_2):
                    return diff_func(kernel_kj, t, s, ix_func, ix_diff,
                                     params_1, params_2)
                self.diff_cross_upsilon[i][j][k] = diff_cross_upsilon
            else:
                raise NotImplementedError("No available interaction"
                                          " between kernel", k, ",", i,
                                          " and kernel ", k, ",", j)

    # Evaluation of Reccurent Exponential functionals
    def make_xk(self, k, mu_k=None, mu=None, ker_param=None):
        d = self.d
        if ker_param is None:
            ker_param = self.fitted_ker_param
        if mu_k is None:
            if mu is None:
                if self.fitted_mu is None:
                    mu = np.zeros(d)
                else:
                    mu = self.fitted_mu
        else:
            mu = np.zeros(d)
            mu[k] = mu_k
        x = self.matrix2tensor_params(mu, ker_param)
        x_k = x[k]
        return x_k

    def get_sum_psi_ki(self, k, i, x_ki=None, process_path=None, T_f=None,
                       list_times2end=None, normalize=True):
        psi = self.psi
        d = self.d
        # x_ki
        if x_ki is None:
            x_ki = self.fitted_ker_param[k][i]
        return get_sum_psi_ki(d, k, i, psi, x_ki,
                              process_path=process_path, T_f=T_f,
                              list_times2end=list_times2end,
                              normalize=normalize)

    def get_sum_upsilonzero_ki(self, k, i, x_ki=None, process_path=None,
                               T_f=None, list_times2end=None, normalize=True):
        upsilon = self.upsilon
        d = self.d
        # x_ki
        if x_ki is None:
            x_ki = self.fitted_ker_param[k][i]
        return get_sum_upsilonzero_ki(d, k, i, upsilon, x_ki,
                                      process_path=process_path, T_f=T_f,
                                      list_times2end=list_times2end,
                                      normalize=normalize)

    def get_sum_phi_kjr(self, k, j, ix_ker=None, omega_kj=None, beta_kj=None,
                        process_path=None, list_times=None,
                        T_f=None, n_events=None, kappa=None, varpi=None,
                        normalize=True):
        if (omega_kj is None) or (beta_kj is None):
            x_kj = self.fitted_ker_param[k][j]
            if omega_kj is None:
                omega_kj = x_kj[2*ix_ker]
            if beta_kj is None:
                beta_kj = x_kj[2*ix_ker+1]
        return get_sum_phi_kjr(k, j, omega_kj, beta_kj,
                               process_path=process_path,
                               list_times=list_times, T_f=T_f,
                               n_events=n_events, kappa=kappa, varpi=varpi,
                               normalize=normalize)

    def get_sum_phi_kj(self, k, j, x_kj=None, process_path=None,
                       list_times=None, T_f=None, n_events=None, kappa=None,
                       varpi=None, normalize=True):
        if x_kj is None:
            x_kj = self.fitted_ker_param[k][j]
        r_kj = len(x_kj)//2
        return get_sum_phi_kj(k, j, r_kj, x_kj, process_path=process_path,
                              list_times=list_times, T_f=T_f,
                              n_events=n_events, kappa=kappa, varpi=varpi,
                              normalize=normalize)

    def get_sum_phi_k(self, k, x_k=None, ker_param=None, process_path=None,
                      list_times=None, T_f=None, n_events=None, kappa=None,
                      varpi=None, normalize=True):
        d = self.d
        if x_k is None:
            x_k = self.make_xk(k, ker_param=ker_param)
        return get_sum_phi_k(d, k, x_k, self.interval_map,
                             process_path=process_path, list_times=list_times,
                             T_f=T_f, n_events=n_events, kappa=kappa,
                             varpi=varpi, normalize=normalize)

    def get_sum_upsilon_ijkr(self, i, j, k, ix_ker1=None, ix_ker2=None,
                             omega_ki=None,
                             beta_ki=None, omega_kj=None, beta_kj=None,
                             process_path=None, list_times=None, T_f=None,
                             n_events=None, kappa=None, varpi=None,
                             normalize=True):
        if (omega_ki is None) or (beta_ki is None):
            x_ki = self.fitted_ker_param[k][i]
            if omega_ki is None:
                omega_ki = x_ki[2*ix_ker1]
            if beta_ki is None:
                beta_ki = x_ki[2*ix_ker1+1]
        if (omega_kj is None) or (beta_kj is None):
            x_kj = self.fitted_ker_param[k][j]
            if omega_kj is None:
                omega_kj = x_kj[2*ix_ker2]
            if beta_kj is None:
                beta_kj = x_kj[2*ix_ker2+1]
        return get_sum_upsilon_ijkr(i, j, omega_ki, beta_ki, omega_kj, beta_kj,
                                    process_path=process_path,
                                    list_times=list_times, T_f=T_f,
                                    n_events=n_events, kappa=kappa,
                                    varpi=varpi, normalize=normalize)

    def get_sum_upsilon_ijk(self, i, j, k, x_ki=None, x_kj=None,
                            process_path=None, list_times=None, T_f=None,
                            n_events=None, kappa=None, varpi=None,
                            normalize=True):
        if x_ki is None:
            x_ki = self.fitted_ker_param[k][i]
        r_ki = len(x_ki)//2
        if x_kj is None:
            x_kj = self.fitted_ker_param[k][j]
        r_kj = len(x_kj)//2
        return get_sum_upsilon_ijk(i, j, k, x_ki, x_kj, r_ki, r_kj,
                                   process_path=process_path,
                                   list_times=list_times, T_f=T_f,
                                   n_events=n_events, kappa=kappa, varpi=varpi,
                                   normalize=normalize)

    def get_sum_upsilon_k(self, k, x_k=None, ker_param=None, process_path=None,
                          list_times=None,
                          T_f=None, n_events=None, kappa=None, varpi=None,
                          normalize=True):
        d = self.d
        if x_k is None:
            x_k = self.make_xk(k, ker_param=ker_param)
        return get_sum_upsilon_k(d, k, x_k, self.interval_map,
                                 process_path=process_path,
                                 list_times=list_times, T_f=T_f,
                                 n_events=n_events, kappa=kappa, varpi=varpi,
                                 normalize=normalize)

    def get_exp_lse_k(self, k, x_k=None, ker_param=None, process_path=None,
                      T_f=None, list_times=None, list_times2end=None,
                      n_events=None, varpi=None, kappa=None, eta_k=None,
                      eta=None):
        # Exact LSE_k of an Exponential MHP
        d = self.d
        psi = self.psi
        upsilon = self.upsilon
        if x_k is None:
            x_k = self.make_xk(k, ker_param=ker_param)
        return get_exp_lse_k(d, k, x_k, self.interval_map, psi, upsilon,
                             process_path=process_path, T_f=T_f,
                             list_times=list_times,
                             list_times2end=list_times2end,
                             n_events=n_events, varpi=varpi, kappa=kappa,
                             eta_k=eta_k, eta=eta)

    # Derivatives of LSE functions
    def diff_exp_lse_mu_k(self, k, x_k=None, mu_k=None, mu=None,
                          ker_param=None,
                          process_path=None, list_times2end=None, T_f=None,
                          eta_k=None, eta=None):
        d = self.d
        psi = self.psi
        if x_k is None:
            x_k = self.make_xk(k, mu_k=None, mu=None, ker_param=ker_param)
        return diff_exp_lse_mu_k(d, k, x_k, self.interval_map, psi,
                                 process_path=process_path,
                                 list_times2end=list_times2end, T_f=T_f,
                                 eta_k=eta_k, eta=eta)

    def diff_exp_lse_omega_kpr(self, d, k, p, ix_ker, x_k=None, mu_k=None,
                               mu=None, ker_param=None,
                               process_path=None, T_f=None,
                               list_times=None, list_times2end=None,
                               n_events=None, varpi=None, kappa=None):
        d = self.d
        if x_k is None:
            x_k = self.make_xk(k, mu_k=mu_k, mu=mu, ker_param=ker_param)
        return diff_exp_lse_omega_kpr(d, k, p, ix_ker, x_k, self.interval_map,
                                      process_path=process_path, T_f=T_f,
                                      list_times=list_times,
                                      list_times2end=list_times2end,
                                      n_events=n_events, varpi=varpi,
                                      kappa=kappa)

    def diff_exp_lse_theta_kpr(self, k, p, r, x_k=None, mu_k=None, mu=None,
                               ker_param=None, process_path=None,
                               T_f=None, list_times=None, list_times2end=None,
                               n_events=None, varpi=None, kappa=None,
                               eta_k=None, eta=None, epsilon=10**-3,
                               diff_type='central difference'):
        d = self.d
        interval_map = self.interval_map
        psi = self.psi
        upsilon = self.upsilon
        if x_k is None:
            x_k = self.make_xk(k, mu_k=mu_k, mu=mu, ker_param=ker_param)
        return diff_exp_lse_theta_kpr(d, k, p, r, x_k, interval_map, psi=psi,
                                      upsilon=upsilon,
                                      process_path=process_path,
                                      T_f=T_f, list_times=list_times,
                                      list_times2end=list_times2end,
                                      n_events=n_events, varpi=varpi,
                                      kappa=kappa, eta_k=eta_k, eta=eta,
                                      epsilon=epsilon, diff_type=diff_type)

    def get_grad_lse_k(self, k, x_k=None, mu_k=None, mu=None, ker_param=None,
                       process_path=None, T_f=None, list_times=None,
                       list_times2end=None, n_events=None, varpi=None,
                       kappa=None, eta_k=None, eta=None):
        if x_k is None:
            x_k = self.make_xk(k, mu_k=mu_k, mu=mu, ker_param=ker_param)
        n_param_k = len(x_k)
        grad = np.zeros(n_param_k)

        # Derivative with respect to \mu_k
        grad[0] = self.diff_exp_lse_mu_k(k, x_k=x_k, process_path=process_path,
                                         list_times2end=list_times2end,
                                         T_f=T_f, eta_k=eta_k, eta=eta)

        # Derivative with respect to kernel parameters
        for ix_param in range(1, n_param_k):
            p = self._mhp.ix_map[k][ix_param]['ker']
            r = self._mhp.ix_map[k][ix_param]['par']
            res = self.diff_exp_lse_theta_kpr(k, p, r, x_k=x_k,
                                              process_path=process_path,
                                              T_f=T_f,
                                              list_times=list_times,
                                              list_times2end=list_times2end,
                                              n_events=n_events, varpi=varpi,
                                              kappa=kappa, eta_k=eta_k,
                                              eta=eta, epsilon=10**-3,
                                              diff_type='central difference')
            grad[ix_param] = res
        return grad

    # Fit
    def clear_fit(self):
        """
        Delete all previously saved results and logs from the
        corresponding attributes of the MHP object.

        """
        self.is_fitted = False
        self._mhp.is_fitted = False
        self.fitted_mu = None
        self._mhp.fitted_mu = None
        self.fitted_ker_param = None
        self._mhp.fitted_ker_param = None
        self.fit_residuals = None
        self._mhp.fit_residuals = None
        self.fitted_adjacency = None
        self._mhp.fitted_adjacency = None
        self.fit_log = None
        self._mhp.fit_log = None

    def fit(self, list_times, T_f, kappa=None, varpi=None, x_0=None,
            n_iter=1000, solvers=None, estimators=None, logger=None, seed=1234,
            verbose=False, clear=True, write=True, **kwargs):
        """
        Fit the MHP model to some observations.

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
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        kappa : TYPE, optional
            DESCRIPTION. The default is None.
        varpi : TYPE, optional
            DESCRIPTION. The default is None.
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
        rng = np.random.default_rng(seed)

        # Clear saved data in case already fitted
        if clear:
            self.clear_fit()

        # Data
        d = self.d
        process_path = ProcessPath(list_times, T_f, kappa=kappa, varpi=varpi)

        # Model
        mu_bnds = [10**-10 for k in range(d)]
        bnds = self.matrix2tensor_params(mu_bnds, self.param_bounds)

        # Solver
        if not isinstance(n_iter, (list, np.ndarray)):
            n_iter = [n_iter for k in range(d)]

        # Initialisation
        if x_0 is None:
            ref_mu = kwargs.get('ref_mu', None)
            ref_ker_param = kwargs.get('ref_ker_param', None)
            range_ref = kwargs.get('range_ref', 0.1)
            target_bratio = kwargs.get('target_bratio', 0.6)
            max_omega = kwargs.get('max_omega', 1.)
            true_omega = kwargs.get('true_omega', None)
            max_param = kwargs.get('max_param', 5.)
            min_mu = kwargs.get('min_mu', 0.)
            max_mu = kwargs.get('max_mu', None)
            mu_0, ker_0 = self.get_random_param(ref_mu=ref_mu,
                                                ref_ker_param=ref_ker_param,
                                                range_ref=range_ref,
                                                target_bratio=target_bratio,
                                                max_omega=max_omega,
                                                true_omega=true_omega,
                                                max_param=max_param,
                                                min_mu=min_mu, max_mu=max_mu,
                                                flatten=False, rng=rng)
            x_0 = self.matrix2tensor_params(mu_0, ker_0)
        else:
            mu_0, ker_0 = self.tensor2matrix_params(x_0)

        # Initialize Solvers
        if solvers is None:
            solvers = ADAM(**kwargs)
        else:
            if issubclass(type(solvers), Solver):
                solvers = [copy.deepcopy(solvers) for k in range(d)]

        # Initialize logger
        logger = OptimLogger(d, n_iter, **kwargs)

        # Scheme
        x = [None]*d
        for k in range(d):
            x_k = x_0[k]
            logger.log_param(k, 0, x_k)
            bounds_k = bnds[k]
            n_iter_k = n_iter[k]

            for t in tqdm(range(n_iter_k), disable=not verbose):
                # Compute LSE gradient estimate for parameters x_k
                g_t = self.get_grad_lse_k(k, x_k=x_k,
                                          process_path=process_path)
                logger.log_grad(k, t, g_t)
                # Apply solver iteration then project into space of parameters
                x_k = solvers.iterate(t, x_k, g_t)
                x_k = np.maximum(x_k, bounds_k)
                logger.log_param(k, t+1, x_k)
            esimator_k_log = estimators[k].get_log()
            logger.estimator_logs[k] = esimator_k_log
            x[k] = x_k
        fitted_mu, fitted_ker_param = self.tensor2matrix_params(x)
        if write:
            self.is_fitted = True
            self._mhp.is_fitted = True
            self.fitted_mu = fitted_mu
            self._mhp.fitted_mu = fitted_mu
            self.fitted_ker_param = fitted_ker_param
            self._mhp.fitted_ker_param = fitted_ker_param
            logger.process_logs(self)
            logger.mu_0 = mu_0
            logger.ker_0 = ker_0
            self.fit_log = logger
            self._mhp.fit_log = logger
        return fitted_mu, fitted_ker_param

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

    def get_random_param(self, ref_mu=None, ref_ker_param=None, range_ref=0.1,
                         target_bratio=0.6, max_omega=1., true_omega=None,
                         max_param=5.,
                         min_mu=0., max_mu=None, flatten=False, seed=1234,
                         rng=None):
        return self._mhp.get_random_param(ref_mu=ref_mu,
                                          ref_ker_param=ref_ker_param,
                                          range_ref=range_ref,
                                          target_bratio=target_bratio,
                                          max_omega=max_omega,
                                          true_omega=true_omega,
                                          max_param=max_param, min_mu=min_mu,
                                          max_mu=max_mu, flatten=flatten,
                                          seed=seed, rng=rng)

    # Residuals
    def get_residuals(self, process_path, mu=None, kernel_param=None,
                      sampling=False, sample_size=10**3, seed=1234, write=True,
                      verbose=False):
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
        if mu is None or kernel_param is None:
            if self.is_fitted:
                mu = self.fitted_mu
                kernel_param = self.fitted_ker_param
            else:
                raise ValueError("Both mu and kernel_param must be specified.")
        residuals = gof.get_residuals(process_path, self.psi, mu,
                                      kernel_param, sampling=sampling,
                                      sample_size=sample_size, seed=seed,
                                      verbose=verbose)
        if self.is_fitted and write:
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
    def simulate(self, T_f, mu=None, kernel_param=None, seed=1234,
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
        return self._mhp.simulate(T_f, mu=mu, kernel_param=kernel_param,
                                  seed=1234, verbose=verbose)

    def simu_multipath(self, path_res, t_res, x_min, x_max, mu=None,
                       kernel_param=None, seed=1234, verbose=False,
                       disc_type='log', base_seed=1234):
        return self._mhp.simu_multipath(path_res, t_res, x_min, x_max, mu=mu,
                                        kernel_param=kernel_param, seed=seed,
                                        verbose=verbose, disc_type=disc_type,
                                        base_seed=base_seed)

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
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_out = open(file_mu, "wb", **kwargs)
        pickle.dump(self.fitted_mu, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        pickle_out = open(file_ker, "wb", **kwargs)
        pickle.dump(self.fitted_ker_param, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_residuals = file+'_fitted_residuals.pickle'
        else:
            file_residuals = file+'_fitted_residuals'
        file_residuals = file+'_fitted_residuals'
        pickle_out = open(file_residuals, "wb", **kwargs)
        pickle.dump(self.fit_residuals, pickle_out)
        pickle_out.close()

        if file.endswith('.pickle'):
            file_adjacency = file+'_fitted_adj.pickle'
        else:
            file_adjacency = file+'_fitted_adj'
        file_adjacency = file+'_fitted_adj'
        pickle_out = open(file_adjacency, "wb", **kwargs)
        pickle.dump(self.fitted_adjacency, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        if file.endswith('.pickle'):
            file_mu = file+'_fitted_mu.pickle'
        else:
            file_mu = file+'_fitted_mu'
        pickle_in = open(file_mu, "rb")
        fitted_mu = pickle.load(pickle_in)

        if file.endswith('.pickle'):
            file_ker = file+'_fitted_ker.pickle'
        else:
            file_ker = file+'_fitted_ker'
        file_ker = file+'_fitted_ker'
        pickle_in = open(file_ker, "rb")
        fitted_ker = pickle.load(pickle_in)

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

        self.clear_fit()

        self.is_fitted = True
        self.fitted_mu = fitted_mu
        self.fitted_ker_param = fitted_ker
        self.fit_residuals = fitted_residuals
        self.fitted_adjacency = fitted_adjacency
