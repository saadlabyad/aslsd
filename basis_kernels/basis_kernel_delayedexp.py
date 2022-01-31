# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel import BasisKernel
from aslsd.utilities import useful_functions as uf


dict_upsilon = {}


def upsilon_delayedexp_null(t, s, vars_1, vars_2):
    return 0.*t


dict_upsilon['NullKernel'] = upsilon_delayedexp_null


def upsilon_delayedexp_delayedexp(t, s, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    d_s = delta_2-delta_1
    b_s = np.piecewise(t, [d_s < s, d_s == s, d_s > s], [beta_2, 0., beta_1])
    res = (omega_1*omega_2*uf.prod_ratio(beta_1, beta_2)
           * (np.exp(-b_s*np.abs(s-delta_2+delta_1))
              - np.exp(-beta_1*(t-delta_1)-beta_2*(t+s-delta_2))))
    return res


dict_upsilon['DelayedExponentialKernel'] = upsilon_delayedexp_delayedexp

dict_diff_cross_upsilon = {}


def diff_cross_upsilon_delayedexp_null(t, s, ix_func, ix_diff, vars_1, vars_2):
    return 0.*t


dict_diff_cross_upsilon['NullKernel'] = diff_cross_upsilon_delayedexp_null


def diff_cross_upsilon_delayedexp_delayedexp(t, s, ix_func, ix_diff, vars_1,
                                             vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    delta_1 = vars_1[2]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    delta_2 = vars_2[2]
    d_s = delta_2-delta_1
    b_s = np.piecewise(t, [d_s < s, d_s == s, d_s > s], [beta_2, 0., beta_1])
    if ix_func == 1:
        # Derivative with respect to omega_1
        if ix_diff == 0:
            res = (omega_2*uf.prod_ratio(beta_1, beta_2)
                   * (np.exp(-b_s*np.abs(s-delta_2+delta_1))
                      - np.exp(-beta_1*(t-delta_1)-beta_2*(t+s-delta_2))))
            return res
        # Differential wrt beta_1
        elif ix_diff == 1:
            pass
        # Differential wrt delta_1
        elif ix_diff == 2:
            pass
    if ix_func == 2:
        # Derivative with respect to \omega_2
        if ix_diff == 0:
            res = (omega_1*uf.prod_ratio(beta_1, beta_2)
                   * (np.exp(-b_s*np.abs(s-delta_2+delta_1))
                      - np.exp(-beta_1*(t-delta_1)-beta_2*(t+s-delta_2))))
            return res
        # Derivative with respect to \beta_2
        elif ix_diff == 1:
            pass
        # Derivative with respect to \delta_2
        elif ix_diff == 2:
            pass


dict_diff_cross_upsilon['DelayedExponentialKernel'] = diff_cross_upsilon_delayedexp_delayedexp

dict_l2_dot = {}


def l2_dot_delayedexp_null(vars_1, vars_2):
    return 0.


dict_l2_dot['NullKernel'] = l2_dot_delayedexp_null


def l2_dot_delayedexp_delayedexp(vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    pass


dict_l2_dot['DelayedExponentialKernel'] = l2_dot_delayedexp_delayedexp

dict_diff_l2_dot = {}


def diff_l2_dot_delayedexp_null(ix_func, ix_diff, vars_1, vars_2):
    return 0.


dict_diff_l2_dot['NullKernel'] = diff_l2_dot_delayedexp_null


def diff_l2_dot_delayedexp_delayedexp(ix_func, ix_diff, vars_1, vars_2):
    omega_1 = vars_1[0]
    beta_1 = vars_1[1]
    omega_2 = vars_2[0]
    beta_2 = vars_2[1]
    if ix_func == 1:
        if ix_diff == 0:
            pass
        if ix_diff == 1:
            pass
    elif ix_func == 2:
        if ix_diff == 0:
            pass
        if ix_diff == 1:
            pass


dict_diff_l2_dot['DelayedExponentialKernel'] = diff_l2_dot_delayedexp_delayedexp


class DelayedExponentialKernel(BasisKernel):

    # Number of parameters
    def get_n_vars(self):
        return 3

    # Bounds
    def get_var_bounds(self):
        return np.array([10**-10, 10**-10, 10**-10])

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
            return ['NullKernel', 'DelayedExponentialKernel']

    # Kernel functionals
    def make_phi(self, t, vars_):
        delta = vars_[2]
        return np.piecewise(t, [t < delta, t >= delta],
                            [0., lambda t: vars_[0]*vars_[1]*np.exp(-vars_[1]*(t-delta))])

    def make_diff_phi(self, t, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        # Derivative wrt \omega
        if ix_diff == 0:
            return np.piecewise(t, [t < delta, t >= delta],
                                [0., lambda t: beta*np.exp(-beta*(t-delta))])
        # Derivative wrt \beta
        elif ix_diff == 1:
            return np.piecewise(t, [t < delta, t >= delta],
                                [0., lambda t: omega*np.exp(-beta*(t-delta))*(1.-beta*(t-delta))])

        else:
            raise ValueError("The argument ix_diff = ", ix_diff, "is not a ",
                             "valid differentiation index for basis kernels",
                             "of type ", str(self.__class__))

    def make_psi(self, t, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        return np.piecewise(t, [t < delta, t >= delta],
                            [0., lambda t: omega*(1-np.exp(-beta*(t-delta)))])

    def make_diff_psi(self, t, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        delta = vars_[2]
        if ix_diff == 0:
            # Derivative wrt Omega
            return np.piecewise(t, [t < delta, t >= delta],
                                [0., lambda t: 1-np.exp(-beta*(t-delta))])
        elif ix_diff == 1:
            # Derivative wrt Beta
            return np.piecewise(t, [t < delta, t >= delta],
                                [0., lambda t: omega*(t-delta)*np.exp(-beta*(t-delta)) ])
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
        # Derivative wrt \omega
        if ix_diff == 0:
            return (omega*beta*np.exp(-beta*s)
                    * (1-np.exp(-2*beta*t)))
            return np.piecewise(t, [t < delta, t >= delta],
                                [0., lambda t: 1-np.exp(-beta*(t-delta))])
        # Derivative wrt \beta
        elif ix_diff == 1:
            pass
        # Derivative wrt \delta
        elif ix_diff == 2:
            pass
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
        delta = vars_[2]
        return delta+rng.exponential(scale=1./beta, size=size)

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
        pass

    def make_diff_l2_norm(self, ix_diff, vars_):
        omega = vars_[0]
        beta = vars_[1]
        # Differential wrt Omega
        if ix_diff == 0:
            pass
        # Differential wrt Beta
        elif ix_diff == 1:
            pass

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
