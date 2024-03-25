# License: BSD 3 clause

import numpy as np

from aslsd.functionals.baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf
import aslsd.functionals.baselines.periodic_baselines.kepler_equation as ke
import aslsd.functionals.baselines.periodic_baselines.\
    periodic_mu_inversion as pmi


def K_exp(basis_kernel, t, s, vars_ker, vars_mu):
    omega, beta = vars_ker
    alpha, a, b, delta = vars_mu
    psi_term = (alpha+delta)*omega*(1.-np.exp(-beta*t))
    coeff = (omega*beta)/(beta**2+a**2)
    integral = coeff*(
        beta*np.cos(a*s+b)-a*np.sin(a*s+b)
        - np.exp(-beta*t)*(beta*np.cos(a*(t+s)+b)-a*np.sin(a*(t+s)+b)))
    res = psi_term + alpha*integral
    return res


def diff_K_exp(basis_kernel, t, s, ix_func, ix_diff, vars_ker,
               vars_mu):
    omega, beta = vars_ker
    alpha, a, b, delta = vars_mu
    if ix_func == 1:
        # Derivative wrt kernel
        if ix_diff == 0:
            # Derivative wrt omega
            psi_term = (alpha+delta)*(1.-np.exp(-beta*t))
            coeff = beta/(beta**2+a**2)
            integral = coeff*(
                beta*np.cos(a*s+b)-a*np.sin(a*s+b)
                - np.exp(-beta*t)*(beta*np.cos(a*(t+s)+b)-a*np.sin(a*(t+s)+b)))
            res = psi_term + alpha*integral
            return res
        elif ix_diff == 1:
            # Derivative wrt beta
            pass
    elif ix_func == 2:
        # Derivative wrt baseline
        if ix_diff == 0:
            # Derivative wrt alpha
            pass
        elif ix_diff == 1:
            # Derivative wrt a
            pass
        elif ix_diff == 2:
            # Derivative wrt b
            pass
        elif ix_diff == 3:
            # Derivative wrt delta
            psi_term = 1.-np.exp(-beta*t)
            return psi_term


class CosineBaseline(BasisBaseline):
    def __init__(self, inverse_kepler=None, inv_kepler_type='Newton',
                 inv_kepler_args=None,
                 fixed_indices=None, fixed_vars=None, n_fixed_vars=0,
                 ix_map=None, dict_interactions=None, mu=None, diff_mu=None,
                 M=None, diff_M=None):
        self.pos_min_f = 1.
        self.T_p = 2.*np.pi
        self.intercept = True
        self.f = np.cos

        def diff_f(t):
            return -np.sin(t)
        self.diff_f = diff_f
        self.F = np.sin

        def F_q(t):
            return 0.5*(t+np.sin(t)*np.cos(t))
        self.F_q = F_q

        # Simulation
        if inv_kepler_args is None:
            inv_kepler_args = {}
        if inverse_kepler is None:
            if inv_kepler_type == 'Kapetyn':
                inv_kepler_func = ke.get_inv_kepler_kapetyn
            elif inv_kepler_type == 'Newton':
                inv_kepler_func = ke.get_inv_kepler_newton
            inverse_kepler = ke.extend_inv_kepler_func(inv_kepler_func,
                                                       **inv_kepler_args)
        self.inverse_kepler = inverse_kepler

        def inverse_h(y, epsilon=.5):
            t = self.inverse_kepler(np.pi+y, epsilon=epsilon)-np.pi
            return t
        self.inverse_h = inverse_h
        BasisBaseline.__init__(self, fixed_indices=fixed_indices,
                               fixed_vars=fixed_vars,
                               n_fixed_vars=n_fixed_vars, ix_map=ix_map,
                               dict_interactions=dict_interactions, mu=mu,
                               diff_mu=diff_mu, M=M, diff_M=diff_M)

    # Number of parameters
    def get_n_vars(self):
        return 4

    def get_var_lower_bounds(self):
        return np.array([10**-5, 10**(-10), 0., 10**(-10)])

    def get_var_upper_bounds(self):
        return np.array([np.inf, np.inf, self.T_p, np.inf])

    def get_var_names(self):
        return ['$\u03B1$', 'a', 'b', '$\u03B4$']

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return []

    def make_mu(self, t, vars_):
        return pmi.get_mu(t, self.f, vars_=vars_, pos_min_f=self.pos_min_f,
                          intercept=self.intercept)

    def make_diff_mu(self, t, ix_diff, vars_):
        alpha = vars_[0]
        a = vars_[1]
        b = vars_[2]

        if ix_diff == 0:
            # Derivative wrt \alpha
            return self.pos_min_f+self.f(a*t+b)
        elif ix_diff == 1:
            # Derivative wrt a
            return alpha*t*self.diff_f(a*t+b)
        elif ix_diff == 2:
            # Derivative wrt b
            return alpha*self.diff_f(a*t+b)
        elif ix_diff == 3:
            # Derivative wrt \delta
            if uf.is_array(t):
                return np.ones(len(t))
            else:
                return 1.

    def make_M(self, t, vars_):
        return pmi.get_M(t, self.T_p, self.F_q, F=self.F, vars_=vars_,
                         pos_min_f=self.pos_min_f, intercept=self.intercept)

    def make_diff_M(self, t, ix_diff, vars_):
        return pmi.get_diff_M(t, ix_diff, self.T_p, self.f, self.F_q, F=self.F,
                              vars_=vars_, pos_min_f=self.pos_min_f,
                              intercept=self.intercept)

    # Interactions with kernels
    def make_K(self, basis_kernel, t, s, vars_ker, vars_mu):
        psi = basis_kernel.make_psi
        if uf.is_array(t) or uf.is_array(s):
            pass
        else:
            res = self.make_K(basis_kernel, np.array([t]), np.array([s]),
                              vars_ker, vars_mu)
            return res[0]

    def make_diff_K(self, basis_kernel, t, s, ix_func, ix_diff, vars_ker,
                    vars_mu):
        if uf.is_array(t) or uf.is_array(s):
            if ix_func == 1:
                pass
            elif ix_func == 2:
                pass
        else:
            res = self.make_diff_K(basis_kernel, np.array([t]), np.array([s]),
                                   ix_func, ix_diff, vars_ker, vars_mu)
            return res[0]

    # Simulatiom
    def make_compensator(self, t, vars_):
        alpha, a, b, c = pmi.make_variables(vars_, pos_min_f=self.pos_min_f,
                                            intercept=self.intercept)
        T_p = self.T_p
        k, theta = pmi.t2ktheta(t, T_p, a, b)
        res = (alpha/a)*(self.F(a*t+b)-self.F(b))
        if c > 0.:
            res += c*t
        return res

    def make_inv_compensator(self, y, vars_):
        alpha, a, b, c = pmi.make_variables(vars_, pos_min_f=self.pos_min_f,
                                            intercept=self.intercept)
        T_p = self.T_p
        t = pmi.get_t(y, T_p, self.F, alpha, a, b, c=c, inverse_F=None,
                      inverse_h=self.inverse_h)
        return t

    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass
