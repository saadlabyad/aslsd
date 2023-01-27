# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


dict_upsilon = {}


def upsilon_exponential_null(t, s, vars_1, vars_2):
    return 0.*t


dict_upsilon['NullKernel'] = upsilon_exponential_null


def upsilon_exponential_exponential(t, s, vars_1, vars_2):
    return (vars_1[0]*vars_2[0]*((vars_1[1]*vars_2[1])/(vars_1[1]+vars_2[1]))
            * np.exp(-vars_2[1]*s)*(1-np.exp(-(vars_1[1]+vars_2[1])*t)))


dict_upsilon['ExponentialKernel'] = upsilon_exponential_exponential


def upsilon_exponential_gaussian(t, s, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    d_s = delta_2-s-beta_1*beta_2**2
    return (omega_1*omega_2*beta_1*np.exp(-beta_1
                                          * (delta_2-s-0.5*beta_1*beta_2**2))
            * uf.gaussian_int(-d_s/beta_2, (t-d_s)/beta_2))


dict_upsilon['GaussianKernel'] = upsilon_exponential_gaussian


dict_diff_cross_upsilon = {}


def diff_cross_upsilon_exp_null(t, s, ix_func, ix_diff, vars_1, vars_2):
    return 0.*t


dict_diff_cross_upsilon['NullKernel'] = diff_cross_upsilon_exp_null


def diff_cross_upsilon_exp_exp(t, s, ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    if ix_func == 1:
        # Derivative with respect to omega_1
        if ix_diff == 0:
            return (omega_2*((beta_1*beta_2)/(beta_1+beta_2))
                    * np.exp(-beta_2*s)*(1.-np.exp(-(beta_1+beta_2)*t)))
        # Differential wrt beta_1
        elif ix_diff == 1:
            return (omega_1*omega_2*np.exp(-beta_2*s)
                    * ((uf.share_ratio(beta_2, beta_1)**2)
                       * (1.-np.exp(-(beta_2+beta_1)*t))
                       + uf.prod_ratio(beta_2, beta_1)*t*np.exp(
                           -(beta_2+beta_1)*t)))
    if ix_func == 2:
        # Derivative with respect to \omega_2
        if ix_diff == 0:
            return (omega_1*((beta_1*beta_2)/(beta_1+beta_2))*np.exp(-beta_2*s)
                    * (1.-np.exp(-(beta_1+beta_2)*t)))
        # Derivative with respect to \beta_2
        elif ix_diff == 1:
            return (omega_1*omega_2*np.exp(-beta_2*s)
                    * ((uf.share_ratio(beta_1, beta_2)**2)
                       * (1.-np.exp(-(beta_1+beta_2)*t))
                       + uf.prod_ratio(beta_2, beta_1)
                       * ((t+s)*np.exp(-(beta_1+beta_2)*t)-s)))


dict_diff_cross_upsilon['ExponentialKernel'] = diff_cross_upsilon_exp_exp


def diff_cross_upsilon_exp_gauss(t, s, ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    d_s = delta_2-s-beta_1*beta_2**2
    if ix_func == 1:
        # Differential wrt \omega_1
        if ix_diff == 0:
            # Overflow problem of the exonential for this upsilon ...
            return np.where(-beta_1*(delta_2-s-0.5*beta_1*beta_2**2) > 10**2,
                            0., (omega_2*beta_1
                                 * np.exp(-beta_1*(delta_2
                                                   - s-0.5*beta_1*beta_2**2))
                                 * uf.gaussian_int(-d_s/beta_2,
                                                   (t-d_s)/beta_2)))

        elif ix_diff == 1:
            # Differential wrt \beta_1
            cdf_term = ((1./beta_1-delta_2+s+0.5*beta_1*beta_2**2)
                        * uf.gaussian_int(-d_s/beta_2, (t-d_s)/beta_2))
            pdf_term = beta_2*(uf.gaussian_pdf((t-d_s)/beta_2)
                               - uf.gaussian_pdf(-d_s/beta_2))
            # Overflow problem of the exonential for this upsilon ...
            return np.where(-beta_1*(delta_2-s-0.5*beta_1*beta_2**2) > 10**2,
                            0., (omega_1*omega_2*beta_1
                                 * np.exp(-beta_1*(delta_2-s
                                                   - 0.5*beta_1*beta_2**2))
                                 * (cdf_term+pdf_term)))

    elif ix_func == 2:
        # Differential wrt \omega_2
        if ix_diff == 0:
            # Overflow problem of the exonential for this upsilon ...
            return np.where(-beta_1*(delta_2-s-0.5*beta_1*beta_2**2) > 10**2,
                            0., (omega_1*beta_1
                                 * np.exp(-beta_1*(delta_2-s
                                                   - 0.5*beta_1*beta_2**2))
                                 * uf.gaussian_int(-d_s/beta_2,
                                                   (t-d_s)/beta_2)))

        # Differential wrt \beta_2
        elif ix_diff == 1:
            cdf_term = (beta_2*beta_1**2)*uf.gaussian_int(-d_s/beta_2,
                                                          (t-d_s)/beta_2)
            pdf_term = ((beta_1+(delta_2-s-t)/beta_2**2)
                        * uf.gaussian_pdf((t-d_s)/beta_2)
                        - ((beta_1+(delta_2-s)/beta_2**2)
                           * uf.gaussian_pdf(-d_s/beta_2)))
            # Overflow problem of the exonential for this upsilon ...
            return np.where(-beta_1*(delta_2-s-0.5*beta_1*beta_2**2) > 10**2,
                            0., (omega_2*omega_1*beta_1
                                 * np.exp(-beta_1*(delta_2-s
                                                   - 0.5*beta_1*beta_2**2))
                                 * (cdf_term+pdf_term)))

        # Differential wrt \delta_2
        elif ix_diff == 2:
            cdf_term = beta_1*uf.gaussian_int(-d_s/beta_2, (t-d_s)/beta_2)
            pdf_term = (1./beta_2)*(uf.gaussian_pdf((t-d_s)/beta_2)
                                    - uf.gaussian_pdf(-d_s/beta_2))
            # Overflow problem of the exonential for this upsilon ...
            return -np.where(-beta_1*(delta_2-s-0.5*beta_1*beta_2**2) > 10**2,
                             0., (omega_2*omega_1*beta_1
                                  * np.exp(-beta_1*(delta_2-s
                                                    - 0.5*beta_1*beta_2**2))
                                  * (cdf_term+pdf_term)))


dict_diff_cross_upsilon['GaussianKernel'] = diff_cross_upsilon_exp_gauss


dict_l2_dot = {}


def l2_dot_exp_null(vars_1, vars_2):
    return 0.


dict_l2_dot['NullKernel'] = l2_dot_exp_null


def l2_dot_exp_exp(vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    return omega_1*omega_2*((beta_1*beta_2)/(beta_1+beta_2))


dict_l2_dot['ExponentialKernel'] = l2_dot_exp_exp


def l2_dot_exp_gauss(vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    return (omega_1*omega_2*beta_1
            * np.exp(-beta_1*(delta_2-0.5*beta_1*beta_2**2))
            * uf.normal_cdf((delta_2-beta_1*beta_2**2)/beta_2))


dict_l2_dot['GaussianKernel'] = l2_dot_exp_gauss

dict_diff_l2_dot = {}


def diff_l2_dot_exp_null(ix_func, ix_diff, vars_1, vars_2):
    return 0.


dict_diff_l2_dot['NullKernel'] = diff_l2_dot_exp_null


def diff_l2_dot_exp_exp(ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
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


dict_diff_l2_dot['ExponentialKernel'] = diff_l2_dot_exp_exp


def diff_l2_dot_exp_gauss(ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    if ix_func == 1:
        if ix_diff == 0:
            return (omega_2*beta_1
                    * np.exp(-beta_1*(delta_2-0.5*beta_1*beta_2**2))
                    * uf.normal_cdf((delta_2-beta_1*beta_2**2)/beta_2))
        if ix_diff == 1:
            pass
    elif ix_func == 2:
        if ix_diff == 0:
            return (omega_1*beta_1
                    * np.exp(-beta_1*(delta_2-0.5*beta_1*beta_2**2))
                    * uf.normal_cdf((delta_2-beta_1*beta_2**2)/beta_2))
        if ix_diff == 1:
            pass


dict_diff_l2_dot['GaussianKernel'] = diff_l2_dot_exp_gauss


class ExponentialKernel(BasisKernel):
    """
    Class for exponential basis kernels. The associated basis function
    is defined for all :math:`t \\geq 0` by

    .. math::
        f_{\\vartheta}(t) := \\omega\\beta\\exp(-\\beta t).

    The parameters of this basis kernel are given by

    .. math::
        \\vartheta := (\\omega,\\beta).

    where

    * :math:`\\omega \\geq 0` controls the :math:`L_{1}` norm of :math:`f`;
    * :math:`\\beta > 0` is the decay rate of :math:`f`.

    Notes
    ------
        The closed form expressions for the derivatives of the functions
        :math:`\\Phi`, :math:`\\Psi`, and :math:`\\Upsilon` are available in
        Cartea, Á., Cohen, S. N., and Labyad, S., (November 22, 2021)
        'Gradient-based estimation of linear Hawkes processes with general
        kernels'.
        `Available at SSRN. <https://ssrn.com/abstract=3969208>`_

    """

    # Number of parameters
    def get_n_vars(self):
        return 2

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-10, 10**-10])

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
            return ['NullKernel', 'ExponentialKernel', 'GaussianKernel',
                    'TriangularKernel', 'RayleighKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        return vars_[0]*vars_[1]*np.exp(-vars_[1]*t)

    def make_diff_phi(self, t, ix_diff, vars_):
        # Derivative wrt \omega
        if ix_diff == 0:
            return vars_[1]*np.exp(-vars_[1]*t)
        # Derivative wrt \beta
        elif ix_diff == 1:
            return vars_[0]*np.exp(-vars_[1]*t)*(1.-vars_[1]*t)
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_psi(self, t, vars_):
        return vars_[0]*(1-np.exp(-vars_[1]*t))

    def make_diff_psi(self, t, ix_diff, vars_):
        if ix_diff == 0:
            # Derivative wrt Omega
            return 1-np.exp(-vars_[1]*t)
        elif ix_diff == 1:
            # Derivative wrt Beta
            return vars_[0]*t*np.exp(-vars_[1]*t)
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

    def make_diff_sim_upsilon(self, t, s, ix_diff, vars_):

        # Derivative wrt \omega
        if ix_diff == 0:
            return (vars_[0]*vars_[1]*np.exp(-vars_[1]*s)
                    * (1-np.exp(-2*vars_[1]*t)))
        # Derivative wrt \beta
        elif ix_diff == 1:
            return (0.5*(vars_[0]**2)*np.exp(-vars_[1]*s)
                    * (1.-vars_[1]*s+np.exp(-2*vars_[1]*t)
                       * (vars_[1]*(2*t+s)-1)))
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_diff_cross_upsilon(self, basis_kern_2, t, s, ix_func, ix_diff,
                                vars_1, vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_diff_cross_upsilon[ker_type](t, s, ix_func, ix_diff,
                                                 vars_1, vars_2)

    def make_diff_cross_upsilon_rev(self, basis_kern_2, t, s, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        beta = vars_[1]
        return rng.exponential(scale=1./beta, size=size)

    # Metrics
    def make_l1_norm(self, vars_):
        return vars_[0]

    def make_diff_l1_norm(self, ix_diff, vars_):
        # Differential wrt Omega
        if ix_diff == 0:
            return 1.
        # Differential wrt Beta
        elif ix_diff == 1:
            return 0.

    def make_l2_norm(self, vars_):
        omega = vars_[0]
        beta = vars_[1]
        return 0.5*beta*omega**2

    def make_diff_l2_norm(self, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        # Differential wrt Omega
        if ix_diff == 0:
            return omega*beta
        # Differential wrt Beta
        elif ix_diff == 1:
            return 0.5*(omega**2)

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
