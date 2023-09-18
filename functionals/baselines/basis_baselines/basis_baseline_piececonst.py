# License: BSD 3 clause
import copy

import numpy as np

from aslsd.functionals.baselines.basis_baseline import BasisBaseline
import aslsd.utilities.useful_functions as uf


class PieceConstBaseline(BasisBaseline):
    def __init__(self, beta=None, b=None,
                 fixed_indices=None, fixed_vars=None, n_fixed_vars=0,
                 ix_map=None, dict_interactions=None, mu=None, diff_mu=None,
                 M=None, diff_M=None):
        # Specify beta
        if beta is None:
            self.beta = np.array([0., np.inf])
            self.n_int = 1
        else:
            self.beta = self.process_beta(beta)
            self.n_int = len(self.beta)-1
        # List of interval sizes
        self.interval_sizes = self.beta[1:]-self.beta[:-1]
        # Cumulative weighted interval sizes
        self.fix_b(b)

        # Book keeping functions
        def g(t):
            return np.digitize(t, self.beta)-1
        self.g = g

        def inv_G(y, b=None, cum_weighted_intervals=None):
            if cum_weighted_intervals is None:
                if self.cum_weighted_intervals is not None:
                    return np.digitize(y, self.cum_weighted_intervals)
                else:
                    cum_weighted_intervals = self.get_cum_weighted_intervals(
                        b, self.interval_sizes)
                    return np.digitize(y, cum_weighted_intervals)
            else:
                cum_weighted_intervals = cum_weighted_intervals.astype('float')
                return np.digitize(y, cum_weighted_intervals)
        self.inv_G = inv_G
        BasisBaseline.__init__(self, fixed_indices=fixed_indices,
                               fixed_vars=fixed_vars,
                               n_fixed_vars=n_fixed_vars, ix_map=ix_map,
                               dict_interactions=dict_interactions, mu=mu,
                               diff_mu=diff_mu, M=M, diff_M=diff_M)

    # For init
    def process_beta(self, L):
        Q = np.array(list(set(L)))
        Q = np.sort(Q)
        Q = Q[Q >= 0.]
        if Q[0] != 0.:
            Q = np.insert(Q, 0, 0.)
        if Q[-1] != np.inf:
            Q = np.append(Q, np.inf)
        return Q

    def get_cum_weighted_intervals(self, b, interval_sizes):
        b = b.astype('float')
        weighted_sizes = b*interval_sizes
        # cum_weighted_intervals = np.zeros(len(weighted_sizes)+1)
        # cum_weighted_intervals[1:] = np.cumsum(weighted_sizes)
        cum_weighted_intervals = np.cumsum(weighted_sizes)
        return cum_weighted_intervals

    def get_squared_cum_weighted_intervals(self, b, interval_sizes):
        squared_weighted_sizes = b**2*interval_sizes
        # squared_cum_weighted_intervals = np.zeros(len(squared_weighted_sizes)+1)
        # squared_cum_weighted_intervals[1:] = np.cumsum(squared_weighted_sizes)
        squared_cum_weighted_intervals = np.cumsum(squared_weighted_sizes)
        return squared_cum_weighted_intervals

    def fix_b(self, b):
        self.b = b
        if b is None:
            self.cum_weighted_intervals = None
            self.squared_cum_weighted_intervals = None
        else:
            cum_weighted_intervals = self.get_cum_weighted_intervals(
                b, self.interval_sizes)
            self.cum_weighted_intervals = cum_weighted_intervals
            squared_cum_weighted_intervals = self.get_squared_cum_weighted_intervals(b, self.interval_sizes)
            self.squared_cum_weighted_intervals = squared_cum_weighted_intervals

    # Number of parameters
    def get_n_vars(self):
        return self.n_int

    def get_var_lower_bounds(self):
        return 10**-10*np.ones(self.n_int)

    def get_var_upper_bounds(self):
        return np.inf*np.ones(self.n_int)

    def get_var_names(self):
        return ['$b_{'+str(ix)+'}$' for ix in range(self.n_int)]

    # Availabe interactions
    def get_interactions(self, is_reverse=False):
        if is_reverse:
            return []
        else:
            return []

    def make_mu(self, t, vars_):
        return vars_[self.g(t)]

    def make_diff_mu(self, t, ix_diff, vars_):
        if not uf.is_array(t):
            return float((t >= self.beta[ix_diff])
                         and (t <= self.beta[ix_diff+1]))
        # Vectorized
        res = np.zeros(len(t))
        res[(t >= self.beta[ix_diff]) & (t <= self.beta[ix_diff+1])] = 1.
        return res

    def make_M(self, t, vars_):
        # Cum weighted
        if self.squared_cum_weighted_intervals is None:
            squared_cum_weighted_intervals = self.get_squared_cum_weighted_intervals(vars_, self.interval_sizes)
        else:
            squared_cum_weighted_intervals = self.squared_cum_weighted_intervals
        # Scalar version
        if not uf.is_array(t):
            if t == 0.:
                return 0.
            else:
                g_t = self.g(t)
                res = vars_[g_t]**2*(t-self.beta[g_t])
                if g_t > 0:
                    res += squared_cum_weighted_intervals[g_t-1]
                res /= t
                return res
        # Vector version
        else:
            res = np.zeros(len(t))
            nonzeros_ixs = np.where((t > 0.))[0]
            g_t = self.g(t)
            res[nonzeros_ixs] = vars_[g_t]**2*(t-self.beta[g_t])

            filtered_ixs = np.where((g_t > 0))[0]
            res[filtered_ixs] += squared_cum_weighted_intervals[g_t[filtered_ixs]-1]
            res[nonzeros_ixs] /= t[nonzeros_ixs]
            return res

    def make_diff_M(self, t, ix_diff, vars_):
        g_t = self.g(t)
        # Scalar version
        if not uf.is_array(t):
            if t == 0.:
                return 0.
            else:
                if g_t > ix_diff:
                    return 2.*vars_[ix_diff]*(self.beta[ix_diff+1]-self.beta[ix_diff])/t
                elif g_t == ix_diff:
                    return 2.*vars_[ix_diff]*(t-self.beta[ix_diff])/t
                else:
                    return 0.
        # Vector version
        else:
            res = np.zeros(len(t))
            ixs_greater = np.where((g_t > ix_diff) & (t > 0.))
            ixs_equal = np.where((g_t == ix_diff) & (t > 0.))
            res[ixs_greater] = 2.*vars_[ix_diff]*(self.beta[ix_diff+1]-self.beta[ix_diff])/t[ixs_greater]
            res[ixs_equal] = 2.*vars_[ix_diff]*(t[ixs_equal]-self.beta[ix_diff])/t[ixs_equal]
            return res

    # Simulatiom
    def make_compensator(self, t, vars_):
        g_t = self.g(t)
        res = vars_[g_t]*(t-self.beta[g_t])
        # Cum weighted
        if self.cum_weighted_intervals is None:
            cum_weighted_intervals = self.get_cum_weighted_intervals(vars_, self.interval_sizes)
        else:
            cum_weighted_intervals = self.cum_weighted_intervals
        # Scalar version
        if not uf.is_array(t):
            if g_t > 0:
                res += cum_weighted_intervals[g_t-1]
            return res
        # Vector version
        filtered_ixs = np.where((g_t > 0))[0]
        res[filtered_ixs] += cum_weighted_intervals[g_t[filtered_ixs]-1]
        return res

    def make_inv_compensator(self, y, vars_):
        # Cum weighted
        if self.cum_weighted_intervals is None:
            cum_weighted_intervals = self.get_cum_weighted_intervals(vars_, self.interval_sizes)
        else:
            cum_weighted_intervals = self.cum_weighted_intervals
        # G(y)
        G_y = self.inv_G(y, b=vars_,
                         cum_weighted_intervals=cum_weighted_intervals)
        res = copy.deepcopy(y)
        # Scalar version
        if not uf.is_array(y):
            if G_y > 0:
                res -= cum_weighted_intervals[G_y-1]
        # Vector version
        else:
            filtered_ixs = np.where((G_y > 0))[0]
            res[filtered_ixs] -= cum_weighted_intervals[G_y[filtered_ixs]-1]
        res = self.beta[G_y]+(res/vars_[G_y])
        return res

    def make_intensity_bound(self, vars_):
        # Return basis baseline object which intensity upper bounds that of
        # the basis baseline object.
        pass

    # Data Processing
    def get_events_rate(self, times):
        return np.histogram(times, self.beta)[0]/self.interval_sizes
