# License: BSD 3 clause

import numpy as np

from aslsd.functionals.kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


# =============================================================================
# Upsilon
# =============================================================================
dict_upsilon = {}


def upsilon_sqpow_null(t, s, vars_1, vars_2):
    return 0.*t


dict_upsilon['NullKernel'] = upsilon_sqpow_null


def upsilon_sqpow_sqpow_szero(t, vars_1, vars_2):
    # s=0
    omega, beta = vars_1
    omega_2, beta_2 = vars_2
    # Computations
    v_t = 1.+beta*t
    # Inverses
    cubic_inv_t = v_t**-3
    # result
    res = omega**2*(beta/3.)*(1.-cubic_inv_t)
    return res


def upsilon_sqpow_sqpow_spos(t, s, vars_1, vars_2):
    omega, beta = vars_1
    omega_2, beta_2 = vars_2
    # Computations
    v_t = 1.+beta*t
    v_s = 1.+beta*s
    v_ts = 1.+beta*(t+s)
    # Inverses
    inv_t = 1./v_t
    inv_s = 1./v_s
    inv_ts = 1./v_ts
    # Logs
    log_t = np.log(v_t)
    log_s = np.log(v_s)
    log_ts = np.log(v_ts)
    # result
    inv_term = (1.-inv_t+inv_s-inv_ts)/(beta*s**2)
    log_term = 2.*(log_ts-log_t-log_s)/((beta**2)*s**3)
    res = omega**2*(inv_term+log_term)
    return res


def upsilon_sqpow_sqpow(t, s, vars_1, vars_2):
    omega, beta = vars_1
    omega_2, beta_2 = vars_2
    if not uf.is_array(s) and s == 0.:
        res = upsilon_sqpow_sqpow_szero(t, vars_1, vars_2)
    else:
        res = upsilon_sqpow_sqpow_spos(t, s, vars_1, vars_2)
    return res


dict_upsilon['SquaredPowerLawKernel'] = upsilon_sqpow_sqpow


dict_diff_cross_upsilon = {}


def diff_cross_upsilon_sqpow_null(t, s, ix_func, ix_diff, vars_1, vars_2):
    return 0.*t


dict_diff_cross_upsilon['NullKernel'] = diff_cross_upsilon_sqpow_null


def diff_cross_upsilon_sqpow_sqpow(t, s, ix_func, ix_diff, vars_1, vars_2):
    omega_1, beta_1 = vars_1
    omega_2, beta_2 = vars_2
    if ix_func == 1:
        # Derivative with respect to omega_1
        if ix_diff == 0:
            pass
        # Differential wrt beta_1
        elif ix_diff == 1:
            pass
    if ix_func == 2:
        # Derivative with respect to \omega_2
        if ix_diff == 0:
            pass
        # Derivative with respect to \beta_2
        elif ix_diff == 1:
            pass


dict_diff_cross_upsilon['SquaredPowerLawKernel'] = diff_cross_upsilon_sqpow_sqpow

# =============================================================================
# K
# =============================================================================
dict_K = {}
dict_diff_K = {}


# Constant
def K_sqpow_const(t, s, vars_ker, vars_basis_mu):
    b = vars_basis_mu[0]
    omega, beta = vars_ker
    pass


dict_K['ConstantBaseline'] = K_sqpow_const


def diff_K_sqpow_const(t, s, ix_func, ix_diff, vars_ker, vars_basis_mu):
    b = vars_basis_mu[0]
    omega, beta = vars_ker
    exp_t = np.exp(-beta*t)
    # Derivatives wrt Kernel parameters
    if ix_func == 1:
        if ix_diff == 0:
            # Derivative wrt Omega
            pass
        elif ix_diff == 1:
            # Derivative wrt Beta
            pass
    # Derivatives wrt Mu parameters
    elif ix_func == 2:
        if ix_diff == 0:
            # b
            pass


dict_diff_K['ConstantBaseline'] = diff_K_sqpow_const

# =============================================================================
# L2 norm
# =============================================================================
dict_l2_dot = {}


def l2_dot_sqpow_null(vars_1, vars_2):
    return 0.


dict_l2_dot['NullKernel'] = l2_dot_sqpow_null


def l2_dot_sqpow_sqpow(vars_1, vars_2):
    omega_1, beta_1 = vars_1
    omega_2, beta_2 = vars_2
    pass


dict_l2_dot['SquaredPowerLawKernel'] = l2_dot_sqpow_sqpow


dict_diff_l2_dot = {}


def diff_l2_dot_sqpow_null(ix_func, ix_diff, vars_1, vars_2):
    return 0.


dict_diff_l2_dot['NullKernel'] = diff_l2_dot_sqpow_null


def diff_l2_dot_sqpow_sqpow(ix_func, ix_diff, vars_1, vars_2):
    omega_1, beta_1 = vars_1
    omega_2, beta_2 = vars_2
    if ix_func == 1:
        if ix_diff == 0:
            return omega_2*((beta_1*beta_2)/(beta_1+beta_2))
        if ix_diff == 1:
            pass
    elif ix_func == 2:
        if ix_diff == 0:
            return omega_1*((beta_1*beta_2)/(beta_1+beta_2))
        if ix_diff == 1:
            pass


dict_diff_l2_dot['SquaredPowerLawKernel'] = diff_l2_dot_sqpow_sqpow


# =============================================================================
# Basis Kernel Class
# =============================================================================
class SquaredPowerLawKernel(BasisKernel):
    """
    Class for squared power law basis kernels. The associated basis function
    is defined for all :math:`t \\geq 0` by

    .. math::
        f_{\\vartheta}(t) := \\omega\\alpha\\beta\\frac{1}{(1+\\beta t)^{1+\\alpha}},

    where :math:`\\alpha=2`. For more general power kernels (with varying exponent
    rate :math:`\\alpha`), see the BasisKernelMC class with PowerLaw.
    The parameters of this basis kernel are given by

    .. math::
        \\vartheta := (\\omega,\\beta).

    where

    * :math:`\\omega \\geq 0` controls the :math:`L_{1}` norm of :math:`f`;
    * :math:`\\beta > 0` is the exponent of :math:`f`.

    Notes
    ------
        The closed form expressions for the derivatives of the functions
        :math:`\\Phi`, :math:`\\Psi`, and :math:`\\Upsilon` are available in
        Cartea, Á., Cohen, S. N., and Labyad, S., (November 22, 2021)
        'Gradient-based estimation of linear Hawkes processes with general
        kernels'.
        `Available at SSRN. <https://ssrn.com/abstract=3969208>`_

    """

    def __init__(self, fixed_indices=None, fixed_vars=None,
                 ix_map=None, allow_simu=True, sim_func=None):
        self.alpha = 1.
        BasisKernel.__init__(self, fixed_indices=fixed_indices,
                             fixed_vars=fixed_vars, allow_simu=allow_simu,
                             sim_func=sim_func)

    # Number of parameters
    def get_n_vars(self):
        return 2

    # Bounds
    def get_var_lower_bounds(self):
        return np.array([10**-10, 10**-10])

    def get_var_upper_bounds(self):
        return np.array([np.inf, np.inf])

    # Param names
    def get_var_names(self):
        omega = '$\u03C9$'
        beta = '$\u03B2$'
        return [omega, beta]

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return ['NullKernel', 'SquaredPowerLawKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        alpha = self.alpha
        omega, beta = vars_
        v = 1+beta*t
        res = omega*alpha*beta*v**(-1.-alpha)
        return res

    def make_diff_phi(self, t, ix_diff, vars_):
        alpha = self.alpha
        omega, beta = vars_
        v = 1+beta*t
        # Derivative wrt \omega
        if ix_diff == 0:
            return alpha*beta*v**(-1.-alpha)
        # Derivative wrt \beta
        elif ix_diff == 1:
            return omega*alpha*(1.-(alpha*beta)*t)*v**(-2.-alpha)
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_psi(self, t, vars_):
        alpha = self.alpha
        omega, beta = vars_
        v = 1+beta*t
        res = omega*(1.-v**-alpha)
        return res

    def make_diff_psi(self, t, ix_diff, vars_):
        alpha = self.alpha
        omega, beta = vars_
        v = 1.+beta*t
        if ix_diff == 0:
            # Derivative wrt Omega
            return 1.-v**-alpha
        elif ix_diff == 1:
            # Derivative wrt Beta
            return omega*alpha*t*v**(-1.-alpha)
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_upsilon(self, basis_kern_2, t, s, vars_1, vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_upsilon[ker_type](t, s, vars_1, vars_2)

    def make_upsilon_rev(self, basis_kern_2, t, s, vars_2, vars_1):
        pass

    def diff_sim_upsilon_szero(self, t, ix_diff, vars_):
        alpha = self.alpha
        omega, beta = vars_
        v_t = 1.+beta*t
        cubic_inv_t = v_t**-3.
        # Derivative wrt \omega
        if ix_diff == 0:
            return 2.*omega*(beta/3.)*(1.-cubic_inv_t)
        # Derivative wrt \beta
        elif ix_diff == 1:
            res = (omega**2)*(((1.-cubic_inv_t)/3.)+beta*t*v_t**-4)
            return res
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_diff_sim_upsilon_spos(self, t, s, ix_diff, vars_):
        alpha = self.alpha
        omega, beta = vars_
        # Computations
        v_t = 1.+beta*t
        v_s = 1.+beta*s
        v_ts = 1.+beta*(t+s)
        # Inverses
        inv_t = 1./v_t
        inv_s = 1./v_s
        inv_ts = 1./v_ts
        # Logs
        log_t = np.log(v_t)
        log_s = np.log(v_s)
        log_ts = np.log(v_ts)
        # result
        log_term = log_ts-log_t-log_s
        # Derivative wrt \omega
        if ix_diff == 0:
            inv_term = 1.-inv_t+inv_s-inv_ts
            res = 2.*omega*(inv_term/(beta*s**2)+2.*log_term/((beta**2)*s**3))
            return res
        # Derivative wrt \beta
        elif ix_diff == 1:
            beta_s_2 = (beta*s)**-2
            beta_s_3 = (beta*s)**-3
            sum_beta_s = beta_s_2+2.*beta_s_3
            diff_beta_s = beta_s_2-2.*beta_s_3
            inv_term = (-sum_beta_s+sum_beta_s*inv_t-diff_beta_s*inv_s
                        + diff_beta_s*inv_ts)
            sq_inv_term = t*inv_t**2-s*inv_s**2+(t+s)*inv_ts**2
            res = (omega**2)*(inv_term+sq_inv_term/(beta*s**2)
                              - 4.*log_term*beta_s_3)
            return res
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_diff_sim_upsilon(self, t, s, ix_diff, vars_):
        if not uf.is_array(s) and s == 0.:
            res = self.diff_sim_upsilon_szero(t, ix_diff, vars_)
        else:
            res = self.make_diff_sim_upsilon_spos(t, s, ix_diff, vars_)
        return res

    def make_diff_cross_upsilon(self, basis_kern_2, t, s, ix_func, ix_diff,
                                vars_1, vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_diff_cross_upsilon[ker_type](t, s, ix_func, ix_diff,
                                                 vars_1, vars_2)

    def make_diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass

    def make_K(self, baseline, t, s, vars_ker, params_mu):
        res = 0.
        for ix_mu in range(baseline.n_basis_mus):
            basis_mu = baseline._basis_mus[ix_mu]
            mu_type = str(type(basis_mu))
            mu_type = mu_type.split('.')[-1][:-2]
            params_basis_mu = params_mu[baseline.interval_map[ix_mu][0]:baseline.interval_map[ix_mu][1]]
            vars_basis_mu = basis_mu.make_vars(params_basis_mu)
            res += dict_K[mu_type](t, s, vars_ker, vars_basis_mu)
        return res

    def make_diff_K(self, baseline, t, s, ix_func, ix_diff, vars_ker, params_mu):
        if ix_func == 1:
            res = 0.
            for ix_mu in range(baseline.n_basis_mus):
                basis_mu = baseline._basis_mus[ix_mu]
                mu_type = str(type(basis_mu))
                mu_type = mu_type.split('.')[-1][:-2]
                params_basis_mu = params_mu[baseline.interval_map[ix_mu][0]:baseline.interval_map[ix_mu][1]]
                vars_basis_mu = basis_mu.make_vars(params_basis_mu)
                res += dict_diff_K[mu_type](t, s, ix_func, ix_diff, vars_ker,
                                            vars_basis_mu)
            return res
        elif ix_func == 2:
            ix_mu = baseline.ix_map[ix_diff]['mu']
            ix_diff_scaled = baseline.ix_map[ix_diff]['par']
            basis_mu = baseline._basis_mus[ix_mu]
            mu_type = str(type(basis_mu))
            mu_type = mu_type.split('.')[-1][:-2]
            params_basis_mu = params_mu[baseline.interval_map[ix_mu][0]:baseline.interval_map[ix_mu][1]]
            vars_basis_mu = basis_mu.make_vars(params_basis_mu)
            return dict_diff_K[mu_type](t, s, ix_func, ix_diff_scaled,
                                        vars_ker, vars_basis_mu)

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        alpha = self.alpha
        omega, beta = vars_
        return rng.exponential(scale=1./beta, size=size)

    # Metrics
    def make_l1_norm(self, vars_):
        alpha = self.alpha
        omega, beta = vars_
        return omega

    def make_diff_l1_norm(self, ix_diff, vars_):
        # Differential wrt Omega
        if ix_diff == 0:
            return 1.
        # Differential wrt Beta
        elif ix_diff == 1:
            return 0.

    def make_l2_norm(self, vars_):
        alpha = self.alpha
        omega, beta = vars_
        res = omega**2*beta/3.

    def make_diff_l2_norm(self, ix_diff, vars_):
        alpha = self.alpha
        omega, beta = vars_
        # Differential wrt Omega
        if ix_diff == 0:
            return 2*omega*beta/3
        # Differential wrt Beta
        elif ix_diff == 1:
            return (omega**2)/3.

    def make_l2_dot(self, basis_kern_2, vars_1, vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_l2_dot[ker_type](vars_1, vars_2)

    def make_diff_l2_dot(self, basis_kern_2, ix_func, ix_diff, vars_1, vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_diff_l2_dot[ker_type](ix_func, ix_diff, vars_1, vars_2)

    # KL divergence
    def make_kl_divergence(self, basis_kern_2, vars_1, vars_2):
        pass

    def make_kl_divergence_rev(self, basis_kern_2, vars_2, vars_1):
        pass

    def make_diff_kl_divergence(self, basis_kern_2, ix_func, ix_diff, vars_1,
                                vars_2):
        pass

    def make_diff_kl_divergence_rev(self, basis_kern_2, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass
