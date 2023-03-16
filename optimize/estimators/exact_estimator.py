# License: BSD 3 clause

import itertools

import numpy as np

from aslsd.optimize.estimators.estimator import Estimator


class ExactEstimator(Estimator):
    def __init__(self, **kwargs):
        pass

    def get_log(self):
        pass

    def diff_lse_mu_k(self, x_k, rng=None, seed=None):
        d = self.d
        k = self.k
        T_f = self.T_f
        eta_k = self.eta[k]
        mu_k = x_k[0]
        res = 2*(mu_k-eta_k)
        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            res += 2.*(np.sum(self.psi[k][i](self.list_times2end[i], x_ki))
                       / T_f)
        return res

    def diff_lse_theta_kpr(self, p, r, x_k, rng=None, seed=None):
        # Derivative of Z with respect to the rth parameter of phi_kp
        # Exact computation
        d = self.d
        k = self.k
        T_f = self.T_f
        mu_k = x_k[0]
        x_kp = x_k[self.interval_map[k][p][0]:self.interval_map[k][p][1]]
        ix_diff = r

        res = 0.

        # Cross Upsilon
        # Loop 1: \Upsilon_{ipk}=phi_{ki}phi_{kp} , i!=p
        for i in itertools.chain(range(p), range(p+1, d)):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            ix_func = 2
            diff_term = 0.
            for m in range(self.varpi[i][p][1], self.n_events[i]):
                t_m = self.list_times[i][m]
                timediff = t_m-self.list_times[p][:self.kappa[p][i][m]+1]
                diff_term += np.sum(self.diff_cross_upsilon[i][p][k](T_f-t_m,
                                                                     timediff,
                                                                     ix_func,
                                                                     ix_diff,
                                                                     x_ki,
                                                                     x_kp))
            res += (2./T_f)*diff_term

        # Loop 2: \Upsilon_{pjk}=phi_{kp}phi_{kj} , j!=p
        for j in itertools.chain(range(p), range(p+1, d)):
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            ix_func = 1
            diff_term = 0.
            for m in range(self.varpi[p][j][1], self.n_events[p]):
                t_m = self.list_times[p][m]
                timediff = t_m-self.list_times[j][:self.kappa[j][p][m]+1]
                diff_term += np.sum(self.diff_cross_upsilon[p][j][k](T_f-t_m,
                                                                     timediff,
                                                                     ix_func,
                                                                     ix_diff,
                                                                     x_kp,
                                                                     x_kj))
            res += (2./T_f)*diff_term

        # Term 3 : \Upsilon_{ppk}=phi_{kp}phi_{kp}
        diff_term = 0.
        for m in range(self.varpi[p][p][1], self.n_events[p]):
            t_m = self.list_times[p][m]
            timediff = t_m-self.list_times[p][:self.kappa[p][p][m]+1]
            diff_term += np.sum(self.diff_sim_upsilon[k][p](T_f-t_m, timediff,
                                                            ix_diff, x_kp))
        res += (2.*diff_term/T_f)

        # Term 4 : \phi_{kp}
        diff_term = 0.
        for m in range(self.varpi[k][p][1], self.n_events[k]):
            t_m = self.list_times[k][m]
            timediff = t_m-self.list_times[p][:self.kappa[p][k][m]+1]
            diff_term += np.sum(self.diff_phi[k][p](timediff, ix_diff, x_kp))
        res -= (2./T_f)*diff_term

        # Term 5 : \psi_{kp}
        res += 2.*mu_k*(np.sum(self.diff_psi[k][p](self.list_times2end[p],
                                                   ix_diff, x_kp))/T_f)

        # Term 6 : Self Upsilon at zero : Upsilon_{ppk}=phi_{kp}phi_{kp}
        res += np.sum(self.diff_sim_upsilon[k][p](self.list_times2end[p], 0.,
                                                  ix_diff, x_kp))/T_f
        return res

    def lse_k_grad_estimate(self, x_k, rng=None, seed=None):
        k = self.k
        grad = np.zeros(self.n_param_k)

        # Derivative with respect to \mu_k
        grad[0] = self.diff_lse_mu_k(x_k)

        # Derivative with respect to kernel parameters
        for ix_param in range(1, self.n_param_k):
            p = self.ix_map[k][ix_param]['ker']
            r = self.ix_map[k][ix_param]['par']
            grad[ix_param] = self.diff_lse_theta_kpr(p, r, x_k)

        return grad

    def lse_k_estimate(self, x_k, rng=None, seed=None):
        d = self.d
        k = self.k
        T_f = self.T_f
        eta_k = self.eta[k]
        mu_k = x_k[0]

        res = 0.
        res += mu_k**2-2*eta_k*mu_k

        for i in range(d):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            # Psi_ki(T-t^i_m)
            res += (2*mu_k
                    * np.sum(self.psi[k][i](x_ki, self.list_times2end[i]))/T_f)
            # Upsilon_iik(T-t^i_m,0)
            res += np.sum(self.upsilon[i][i][k](x_ki, x_ki,
                                                self.list_times2end[i],
                                                0.))/T_f

        for i, j in itertools.product(range(d), range(d)):
            x_ki = x_k[self.interval_map[k][i][0]:self.interval_map[k][i][1]]
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            for m in range(self.varpi[i][j][1], self.n_events[i]):
                t_m = self.list_times[i][m]
                array_timediff = t_m-self.list_times[j][:self.kappa[j][i][m]+1]
                #   Upsilon_ijk (T-t^i_m,t^i_m-t^j_n)
                res += (2./T_f)*np.sum(self.upsilon[i][j][k](x_ki, x_kj,
                                                             T_f-t_m,
                                                             array_timediff))

        for j in range(d):
            x_kj = x_k[self.interval_map[k][j][0]:self.interval_map[k][j][1]]
            for m in range(self.varpi[k][j][1], self.n_events[k]):
                t_m = self.list_times[k][m]
                array_timediff = t_m-self.list_times[j][:self.kappa[j][k][m]+1]
                #   Phi_kj (t^k_m-t^j_n)
                res -= (2./T_f)*np.sum(self.phi[k][j](x_kj, array_timediff))

        return res
