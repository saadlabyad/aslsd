# License: BSD 3 clause

import itertools

import numpy as np


class HomPoisson:
    def __init__(self, d, param_names=None,
                 param_bounds=None, phi=None, diff_phi=None, psi=None,
                 diff_psi=None, upsilon=None, diff_sim_upsilon=None,
                 diff_cross_upsilon=None,
                 fitted_mu=None, fitted_ker_param=None):
        self.d = d
        self.param_names = self.get_param_names()
        self.para_bounds = self.get_param_bounds()
        self.make_kernel_functionals()

    # Parameters map
    def make_ix_map(self):
        pass

    # Omega
    def ix_omegas(self):
        # Outputs the list of indices of the first parameter of each basis
        # kernel.
        # In case of an MHP where each basis kernel has an omega type
        # parameter, this is useful to control the L1 normal of each kernel,
        # and therefore the spectral radius of the adjacency matrix of the
        # MHP. In the context of random initialization of an MHP, this allows
        # to avoid parameter values that make the MHP unstable.
        return [x[0] for x in self.interval_map]

    # Bounds
    def get_param_bounds(self):
        d = self.d
        bnds = [[self.kernel_matrix[i][j].get_param_bounds()
                 for j in range(d)] for i in range(d)]
        return bnds

    # Param names
    def get_param_names(self):
        d = self.d
        param_names = [[None for j in range(d)] for i in range(d)]
        for i, j in itertools.product(range(d), range(d)):
            kernel = self.kernel_matrix[i][j]
            vec_param_names = kernel.get_vec_param_names(self)
            n_param = kernel.n_basis_ker
            param_names[i][j] = [None]*n_param
            for ix_param_scaled in range(n_param):
                ix_ker = kernel.ix_map[ix_param_scaled]['ker']
                ix_param = kernel.ix_map[ix_param_scaled]['par']
                param_names[i][j][ix_param_scaled] = vec_param_names[ix_ker]
                + '_{'+str(i)+','+str(j)+','+str(ix_param)+'}'
        return param_names

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
            kernel = self.kernel_matrix[i][j]
            self.phi[i][j] = kernel.make_phi()
            self.diff_phi[i][j] = kernel.make_diff_phi()
            self.psi[i][j] = kernel.make_psi()
            self.diff_psi[i][j] = kernel.make_diff_psi()
            self.diff_sim_upsilon[i][j] = kernel.make_diff_sim_upsilon()

        for i, j, k in itertools.product(range(d), range(d), range(d)):
            kernel_ki = self.kernel_matrix[k][i]
            kernel_kj = self.kernel_matrix[k][j]
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

    def fit(self, list_times, T_f):
        fitted_mu = np.array([len(L) for L in list_times])/T_f
        self.fitted_mu = fitted_mu

    # Simulation
    def simulate(self, T_f, mu=None, seed=1234):
        if mu is None:
            mu = self.fitted_mu
            if mu is None:
                raise ValueError("Missing value for Mu")
        d = self.d
        list_times = [None]*d
        rng = np.random.default_rng(seed)
        for i in range(d):
            # Number of immigrants
            Nim = rng.poisson(mu[i]*T_f)
            generations = rng.uniform(low=0., high=T_f, size=Nim)
            np.sort(generations)
            list_times[i] = generations
        return list_times

    # Evaluation
    def get_residuals(self,  list_times, T_f):
        pass
