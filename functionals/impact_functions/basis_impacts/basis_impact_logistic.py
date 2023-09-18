# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact


def get_impact_exp(beta, delta):
    res = 4.*(1.-(2.*delta-1.)**2)*beta
    return res


def logistic_func(x, omega, beta, delta):
    impact_ratio = (x-delta)/(1.-(2.*x-1.)**2)
    impact_exp = get_impact_exp(beta, delta)
    return omega/(1.+np.exp(-impact_exp*impact_ratio))


def diff_beta_logistic_func(x, omega, beta, delta):
    impact_ratio = (x-delta)/(1.-(2.*x-1.)**2)
    delta_term = 4.*(1.-(2.*delta-1.)**2)
    A = np.exp(-delta_term*beta*impact_ratio)
    diff_A = -delta_term*impact_ratio*A
    res = -omega*diff_A/(1.+A)**2
    return res


def diff_delta_logistic_func(x, omega, beta, delta):
    x_sq_term = 1./(1.-(2.*x-1.)**2)
    impact_ratio = (x-delta)*x_sq_term
    delta_term = 4.*(1.-(2.*delta-1.)**2)
    A = np.exp(-delta_term*beta*impact_ratio)
    offset = 1-1.5*delta
    x_lin_term = (2.*delta-1)*(x+offset)+1.
    diff_A = (16.*beta*x_sq_term)*x_lin_term*A
    res = -omega*diff_A/(1.+A)**2
    return res


class LogisticImpact(BasisImpact):

    # Input dimension
    def get_mark_dim(self):
        return 1

    # Number of Variables
    def get_n_vars(self):
        return 3

    # Bounds
    def get_var_lower_bounds(self):
        return np.array([0., 10**-10, 10**-10])

    def get_var_upper_bounds(self):
        return np.array([np.inf, np.inf, 1.])

    # Parameter names
    def get_var_names(self):
        omega = '$\u03C9$'
        beta = '$\u03B2$'
        delta = '$\u03B4$'
        return [omega, beta, delta]

    # Impact functionals
    def make_impact(self, xi, vars_):
        omega, beta, delta = vars_
        if xi.ndim == 1:
            if xi[0] == 0.:
                return 1.
            elif xi[0] == 1.:
                return 1.+omega
            else:
                return 1.+logistic_func(xi[0], omega, beta, delta)
        elif xi.ndim == 2:
            res = np.ones(len(xi))
            ixs_1 = np.where(xi == 1)[0]
            res[ixs_1] = 1.+omega
            ixs_else = np.where((xi > 0.) & (xi < 1.))
            if len(ixs_else) > 0:
                res[ixs_else[0]] = 1.+logistic_func(xi[ixs_else], omega, beta,
                                                    delta)
        else:
            raise ValueError('xi does not have the right dimension.')
        return res

    def make_diff_impact(self, xi, ix_diff, vars_):
        omega, beta, delta = vars_
        if xi.ndim == 1:
            if ix_diff == 0:
                # Derivative wrt omega
                if xi[0] == 0.:
                    return 0.
                elif xi[0] == 1.:
                    return 1.
                else:
                    return logistic_func(xi[0], 1., beta, delta)
            elif ix_diff == 1:
                # Derivative wrt beta
                if (xi[0] == 0) or (xi[0] == 1):
                    return 0.
                else:
                    return diff_beta_logistic_func(xi[0], omega, beta, delta)
            elif ix_diff == 2:
                # Derivative wrt delta
                if (xi[0] == 0) or (xi[0] == 1):
                    return 0.
                else:
                    return diff_delta_logistic_func(xi[0], omega, beta, delta)
        elif xi.ndim == 2:
            res = np.zeros(len(xi))
            ixs_1 = np.where(xi == 1)[0]
            ixs_else = np.where((xi > 0.) & (xi < 1.))
            if ix_diff == 0:
                # Derivative wrt omega
                res[ixs_1] = 1.
                if len(ixs_else) > 0:
                    res[ixs_else[0]] = logistic_func(xi[ixs_else], 1., beta,
                                                     delta)
                    return res
            elif ix_diff == 1:
                # Derivative wrt beta
                res[ixs_else[0]] = diff_beta_logistic_func(xi[ixs_else], omega,
                                                           beta, delta)
                return res
            elif ix_diff == 2:
                # Derivative wrt delta
                res[ixs_else[0]] = diff_delta_logistic_func(xi[ixs_else],
                                                            omega, beta, delta)
                return res
        else:
            raise ValueError('xi does not have the right dimension.')
        return res
