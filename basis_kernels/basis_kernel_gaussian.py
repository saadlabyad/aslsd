# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf

dict_upsilon = {}


def upsilon_gauss_null(t, s, vars_1, vars_2):
    return 0.*t


dict_upsilon['NullKernel'] = upsilon_gauss_null


def upsilon_gauss_gauss(t, s, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    sum_beta = np.sqrt(beta_1**2+beta_2**2)
    b_s = (beta_1*beta_2)/sum_beta
    d_s = (uf.share_ratio(beta_2**2, beta_1**2)*delta_1
           + uf.share_ratio(beta_1**2, beta_2**2)*(delta_2-s))
    return (omega_1*omega_2*uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
            * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))


dict_upsilon['GaussianKernel'] = upsilon_gauss_gauss


def upsilon_gauss_exp(t, s, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]

    d_s = delta_1-beta_2*beta_1**2
    return (omega_1*omega_2*beta_2
            * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
            * uf.gaussian_int(-d_s/beta_1, (t-d_s)/beta_1))


dict_upsilon['ExponentialKernel'] = upsilon_gauss_exp

dict_diff_cross_upsilon = {}


def diff_cross_upsilon_gauss_null(t, s, ix_func, ix_diff, vars_1, vars_2):
    return 0.*t


dict_diff_cross_upsilon['NullKernel'] = diff_cross_upsilon_gauss_null


def diff_cross_upsilon_gauss_gauss(t, s, ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    sum_beta = np.sqrt(beta_1**2+beta_2**2)
    b_s = (beta_1*beta_2)/sum_beta
    d_s = (uf.share_ratio(beta_2**2, beta_1**2)*delta_1
           + uf.share_ratio(beta_1**2, beta_2**2)*(delta_2-s))
    if ix_func == 1:
        # Derivative with respect to \omega_1
        if ix_diff == 0:
            return (omega_2*uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))

        # Derivative with respect to \beta_1
        elif ix_diff == 1:
            cdf_term = ((beta_1/sum_beta**2)
                        * (((s-delta_2+delta_1)**2/sum_beta**2)-1)
                        * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))
            offset_coeff = 2*beta_1*(delta_2-delta_1-s)/sum_beta**2
            pdf_term = -((beta_2/(beta_1*sum_beta**2))
                         * ((offset_coeff+((t-d_s)/beta_1))
                            * uf.gaussian_pdf((t-d_s)/b_s)
                            - (offset_coeff-d_s/beta_1)
                            * uf.gaussian_pdf(-d_s/b_s)))
            return (omega_1*omega_2
                    * uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * (cdf_term+pdf_term))

        # Derivative with respect to \delta_1
        elif ix_diff == 2:
            cdf_term = (((delta_2-s-delta_1)/sum_beta**2)
                        * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))
            pdf_term = -((beta_2/(beta_1*sum_beta**2))
                         * (uf.gaussian_pdf((t-d_s)/b_s)
                            - uf.gaussian_pdf(-d_s/b_s)))
            return (omega_1*omega_2
                    * uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * (cdf_term+pdf_term))
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index")

    elif ix_func == 2:
        # Derivative with respect to \omega_2
        if ix_diff == 0:
            return (omega_1*uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))

        # Derivative with respect to \beta_2
        elif ix_diff == 1:
            cdf_term = ((beta_2/sum_beta**2)
                        * (((s-delta_2+delta_1)**2/sum_beta**2)-1)
                        * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))
            offset_coeff = 2*beta_2*(delta_1-delta_2+s)/sum_beta**2
            pdf_term = -((beta_1/(beta_2*sum_beta**2))
                         * ((offset_coeff-((t-d_s)/beta_2))
                            * uf.gaussian_pdf((t-d_s)/b_s)
                            - (offset_coeff+d_s/beta_2)
                            * uf.gaussian_pdf(-d_s/b_s)))
            return (omega_2*omega_1
                    * uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * (cdf_term+pdf_term))

        # Derivative with respect to \delta_2
        elif ix_diff == 2:
            cdf_term = (((delta_1+s-delta_2)/sum_beta**2)
                        * uf.gaussian_int(-d_s/b_s, (t-d_s)/b_s))
            pdf_term = -((beta_1/(beta_2*sum_beta**2))
                         * (uf.gaussian_pdf((t-d_s)/b_s)
                            - uf.gaussian_pdf(-d_s/b_s)))
            return (omega_2*omega_1
                    * uf.gaussian_pdf(s, delta_2-delta_1, sum_beta)
                    * (cdf_term+pdf_term))
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index")


dict_diff_cross_upsilon['GaussianKernel'] = diff_cross_upsilon_gauss_gauss


def diff_cross_upsilon_gauss_exp(t, s, ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]

    if ix_func == 1:
        # Derivative with respect to \omega_1
        if ix_diff == 0:
            d_s = delta_1-beta_2*beta_1**2
            # Overflow problem of the exponential for this upsilon ...
            return np.where(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2) > 10**2,
                            0., omega_2*beta_2
                            * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
                            * uf.gaussian_int(-d_s/beta_1, (t-d_s)/beta_1))

        # Derivative with respect to \beta_1
        elif ix_diff == 1:
            d_s = delta_1+s-beta_2*beta_1**2
            cdf_term = (beta_1*(beta_2**2)
                        * uf.gaussian_int(-d_s/beta_1, (t-d_s)/beta_1))
            pdf_term = ((beta_2-((t-delta_1)/beta_1**2))
                        * uf.gaussian_pdf((t-d_s)/beta_1)
                        - (beta_2+(delta_1/beta_1**2))
                        * uf.gaussian_pdf(-d_s/beta_1))
            return (omega_1*omega_2*beta_2
                    * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
                    * (cdf_term+pdf_term))

        # Derivative with respect to \delta_1
        elif ix_diff == 2:
            d_s = (delta_1+s-beta_2*beta_1**2)/beta_1
            cdf_term = -beta_2*uf.gaussian_int(-d_s, t/beta_1-d_s)
            pdf_term = (-1./beta_1)*(uf.gaussian_pdf(t/beta_1-d_s)
                                     - uf.gaussian_pdf(-d_s))
            return (omega_1*omega_2*beta_2
                    * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
                    * (cdf_term+pdf_term))
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index")
    elif ix_func == 2:
        # Derivative with respect to \omega_2
        if ix_diff == 0:
            d_s = delta_1-beta_2*beta_1**2
            return (omega_1*beta_2
                    * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
                    * uf.gaussian_int(-d_s/beta_1, (t-d_s)/beta_1))

        # Derivative with respect to \beta_2
        elif ix_diff == 1:
            d_s = (delta_1-beta_2*beta_1**2)/beta_1
            cdf_term = (((1./beta_2)-delta_1-s+0.5*beta_2*beta_1**2)
                        * uf.gaussian_int(-d_s, (t/beta_1)-d_s))
            pdf_term = beta_1*(uf.gaussian_pdf((t/beta_1)-d_s)
                               - uf.gaussian_pdf(-d_s))
            return (omega_2*omega_1*beta_2
                    * np.exp(-beta_2*(delta_1+s-0.5*beta_2*beta_1**2))
                    * (cdf_term+pdf_term))
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index")


dict_diff_cross_upsilon['ExponentialKernel'] = diff_cross_upsilon_gauss_exp


dict_l2_dot = {}


def l2_dot_gauss_null(vars_1, vars_2):
    return 0.


dict_l2_dot['NullKernel'] = l2_dot_gauss_null


def l2_dot_gauss_gauss(vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    sum_beta = np.sqrt(beta_1**2+beta_2**2)
    b_s = (beta_1*beta_2)/sum_beta
    d_s = (uf.share_ratio(beta_2**2, beta_1**2)*delta_1
           + uf.share_ratio(beta_1**2, beta_2**2)*(delta_2))
    return (omega_1*omega_2*uf.gaussian_pdf(0., delta_2-delta_1, sum_beta)
            * uf.normal_cdf(d_s/b_s))


dict_l2_dot['GaussianKernel'] = l2_dot_gauss_gauss


def l2_dot_gauss_exp(vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]

    return (omega_1*omega_2*beta_2
            * np.exp(-beta_2*(delta_1-0.5*beta_2*beta_1**2))
            * uf.normal_cdf((delta_1-beta_2*beta_1**2)/beta_1))


dict_l2_dot['ExponentialKernel'] = l2_dot_gauss_exp


dict_diff_l2_dot = {}


def diff_l2_dot_gauss_null(ix_func, ix_diff, vars_1, vars_2):
    return 0.


dict_diff_l2_dot['NullKernel'] = l2_dot_gauss_null


def diff_l2_dot_gauss_gauss(ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    if ix_func == 1:
        if ix_diff == 0:
            sum_beta = np.sqrt(beta_1**2+beta_2**2)
            b_s = (beta_1*beta_2)/sum_beta
            d_s = (uf.share_ratio(beta_2**2, beta_1**2)*delta_1
                   + uf.share_ratio(beta_1**2, beta_2**2)*(delta_2))
            return (omega_2*uf.gaussian_pdf(0., delta_2-delta_1, sum_beta)
                    * uf.normal_cdf(d_s/b_s))
        elif ix_diff == 1:
            return
        elif ix_diff == 2:
            return
    elif ix_func == 2:
        if ix_diff == 0:
            sum_beta = np.sqrt(beta_1**2+beta_2**2)
            b_s = (beta_1*beta_2)/sum_beta
            d_s = (uf.share_ratio(beta_2**2, beta_1**2)*delta_1
                   + uf.share_ratio(beta_1**2, beta_2**2)*(delta_2))
            return (omega_1*uf.gaussian_pdf(0., delta_2-delta_1, sum_beta)
                    * uf.normal_cdf(d_s/b_s))
        elif ix_diff == 1:
            return
        elif ix_diff == 2:
            return


dict_diff_l2_dot['GaussianKernel'] = diff_l2_dot_gauss_gauss


def diff_l2_dot_gauss_exp(ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]

    if ix_func == 1:
        if ix_diff == 0:
            return (omega_2*beta_2
                    * np.exp(-beta_2*(delta_1-0.5*beta_2*beta_1**2))
                    * uf.normal_cdf((delta_1-beta_2*beta_1**2)/beta_1))
        elif ix_diff == 1:
            return
        elif ix_diff == 2:
            return
    elif ix_func == 2:
        if ix_diff == 0:
            return (omega_1*beta_2
                    * np.exp(-beta_2*(delta_1-0.5*beta_2*beta_1**2))
                    * uf.normal_cdf((delta_1-beta_2*beta_1**2)/beta_1))
        elif ix_diff == 1:
            return


dict_diff_l2_dot['ExponentialKernel'] = diff_l2_dot_gauss_exp

dict_kl_divergence = {}


def kl_divergence_gauss_gauss(basis_kern_2, vars_1, vars_2):
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    return (np.log(beta_2/beta_1)-0.5
            + ((beta_1**2+(delta_1-delta_2)**2)/(2*beta_2**2)))


dict_kl_divergence['GaussianKernel'] = kl_divergence_gauss_gauss

dict_diff_kl_divergence = {}


def diff_kl_divergence_gauss_gauss(basis_kern_2, ix_func, ix_diff, vars_1,
                                   vars_2):
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]

    if ix_func == 1:
        # Derivative with respect to \beta_1
        if ix_diff == 1:
            return -(1./beta_1)+beta_1/beta_2**2
        # Derivative with respect to \delta_1
        elif ix_diff == 2:
            return (delta_1-delta_2)/beta_2**2
    elif ix_func == 2:
        # Derivative with respect to \beta_2
        if ix_diff == 1:
            # Diff beta_2
            return 1./beta_2-(beta_1**2+(delta_1-delta_2)**2)/beta_2**3
        # Derivative with respect to \delta_2
        elif ix_diff == 2:
            return -(delta_1-delta_2)/beta_2**2
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index")


dict_diff_kl_divergence['GaussianKernel'] = diff_kl_divergence_gauss_gauss


class GaussianKernel(BasisKernel):
    """
    Class for Gaussian basis kernels. The associated basis function
    is defined for all :math:`t \\geq 0` by

    .. math::
        f_{\\vartheta}(t) := \\frac{\\omega}{\\beta\\sqrt{2\\pi}}\\exp\\bigg(-\\frac{(t-\\delta)^2}{2\\beta^2}\\bigg).

    The parameters of this basis kernel are given by

    .. math::
        \\vartheta := (\\omega,\\beta,\\delta).

    where

    * :math:`\\omega \\geq 0` controls the :math:`L_{1}` norm of :math:`f`;
    * :math:`\\beta > 0` is the standard deviation of the Gaussian;
    * :math:`\\delta > 0` is the mean of the Gaussian.

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
        return 3

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-8, 10**-8, 0.])

    # Param names
    def get_var_names(self):
        omega = '$\u03C9$'
        beta = '$\u03B2$'
        delta = '$\u03B4$'
        return [omega, beta, delta]

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return ['NullKernel', 'ExponentialKernel', 'GaussianKernel',
                    'TriangularKernel', 'RayleighKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        return omega*uf.gaussian_pdf(t, delta, beta)

    def make_diff_phi(self, t, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        # Derivative with respect to \omega
        if ix_diff == 0:
            return uf.gaussian_pdf(t, delta, beta)
        # Derivative with respect to \beta
        elif ix_diff == 1:
            return ((omega/beta)*uf.gaussian_pdf(t, delta, beta)
                    * (((t-delta)/beta)**2-1.))
        # Derivative with respect to \delta
        elif ix_diff == 2:
            return omega*((t-delta)/beta**2)*uf.gaussian_pdf(t, delta, beta)

    def make_psi(self, t, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        return omega*uf.gaussian_int(-delta/beta, (t-delta)/beta)

    def make_diff_psi(self, t, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        # Derivative with respect to \omega
        if ix_diff == 0:
            return uf.gaussian_int(-delta/beta, (t-delta)/beta)
        # Derivative with respect to \beta
        elif ix_diff == 1:
            return (omega/beta**2)*((delta-t)*uf.gaussian_pdf((t-delta)/beta)-delta*uf.gaussian_pdf(-delta/beta))
        # Derivative with respect to \delta
        elif ix_diff == 2:
            return -(omega/beta)*(uf.gaussian_pdf((t-delta)/beta)
                                  - uf.gaussian_pdf(-delta/beta))
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
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        # Derivative with respect to \omega
        if ix_diff == 0:
            return (2.*omega*uf.gaussian_pdf(s, 0., np.sqrt(2.)*beta)*uf.gaussian_int(-np.sqrt(2.)*delta/beta, np.sqrt(2.)*(t-delta)/beta))
        # Derivative with respect to \beta
        elif ix_diff == 1:
            cdf_term = (((s**2-2*beta**2)/(2*beta**3))
                        * uf.gaussian_int(-np.sqrt(2)*(delta-0.5*s)/beta,
                                          np.sqrt(2)*(t-delta+0.5*s)/beta))
            pdf_term = -((np.sqrt(2)*(t-delta+0.5*s)/beta**2)
                         * uf.gaussian_pdf((np.sqrt(2)*(t-delta+0.5*s)/beta))
                         - (np.sqrt(2)*(-delta+0.5*s)/beta**2)
                         * uf.gaussian_pdf(-(np.sqrt(2)*(delta-0.5*s)/beta)))
            return (omega**2*uf.gaussian_pdf(s, 0., np.sqrt(2.)*beta)
                    * (cdf_term+pdf_term))
        # Derivative with respect to \delta
        elif ix_diff == 2:
            return -((omega**2*np.sqrt(2)/beta)
                     * uf.gaussian_pdf(s, 0., np.sqrt(2.)*beta)
                     * (uf.gaussian_pdf(np.sqrt(2)*(t-delta+s/2.)/beta)
                        - uf.gaussian_pdf(-np.sqrt(2)*(delta-s/2.)/beta)))
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

    def make_K(self, baseline, t, s, vars_ker, params_mu):
        pass

    def make_diff_K(self, baseline, t, s, ix_func, ix_diff, vars_ker, params_mu):
        pass

    # Simulatiom
    def make_simu_func(self, rng, vars_, size=1):
        beta = vars_[1]
        delta = vars_[2]
        return rng.normal(loc=delta, scale=beta, size=size)

    # Metrics
    def make_l1_norm(self, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        return omega*uf.normal_cdf(delta/beta)

    def make_diff_l1_norm(self, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        # Derivative with respect to \omega
        if ix_diff == 0:
            return uf.normal_cdf(delta/beta)
        # Derivative with respect to \belta
        elif ix_diff == 1:
            return
        # Derivative with respect to \delta
        elif ix_diff == 2:
            return (omega/beta)*uf.gaussian_pdf(-delta/beta, 0., 0.)
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_l2_norm(self, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]

        return (omega**2*uf.gaussian_pdf(0., 0., np.sqrt(2)*beta)
                * uf.normal_cdf(np.sqrt(2)*delta/beta))

    def make_diff_l2_norm(self, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]

        # Derivative with respect to \omega
        if ix_diff == 0:
            return (2.*omega*uf.gaussian_pdf(0., 0., np.sqrt(2.)*beta)
                    * uf.normal_cdf(np.sqrt(2.)*delta/beta))
        # Derivative with respect to \beta
        elif ix_diff == 1:
            return
        # Derivative with respect to \delta
        elif ix_diff == 2:
            return
        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

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
        return dict_diff_kl_divergence[str(type(basis_kern_2))](vars_1, vars_2)

    def make_kl_divergence_rev(self, basis_kern_2, vars_2, vars_1):
        pass

    def make_diff_kl_divergence(self, basis_kern_2, ix_func, ix_diff, vars_1,
                                vars_2):
        ker_type = str(type(basis_kern_2))
        ker_type = ker_type.split('.')[-1][:-2]
        return dict_diff_kl_divergence[ker_type](ix_func, ix_diff, vars_1,
                                                 vars_2)

    def make_diff_kl_divergence_rev(self, basis_kern_2, ix_func, ix_diff,
                                    vars_2, vars_1):
        pass
