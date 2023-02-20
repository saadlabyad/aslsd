# License: BSD 3 clause

import numpy as np

from aslsd.basis_baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf
import aslsd.periodic_baselines.kepler_equation as ke
import aslsd.periodic_baselines.periodic_mu_inversion as pmi


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

    def get_var_bounds(self):
        return np.array([10**-5, 10**(-10), 0., 10**(-10)])

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
