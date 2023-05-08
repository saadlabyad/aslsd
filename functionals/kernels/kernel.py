# License: BSD 3 clause

import copy
import itertools

import numpy as np
from tqdm import tqdm

from aslsd.functionals.kernels.basis_kernel import BasisKernel
from aslsd.optimize.solvers.adam import ADAM


class KernelModel():
    """
    Class for parametric kernel models of MHP.

    | Let :math:`\\theta` denote the vector of parameters of the kernel. A kernel :math:`\\phi_{\\theta}` is a sums of basis kernels :math:`\\Big(f^{(r)}_{\\vartheta^{(r)}}\\Big)_{r \\in [n_b]}`, :math:`n_b \\geq 1`.
    For all :math:`t \\geq 0`, define

    .. math::
        \\phi_{\\theta}(t) := \\sum_{r=1}^{n_{\\textrm{b}}}
        f^{(r)}_{\\vartheta^{(r)}}(t).

    The vector of parameters of kernel :math:`\\phi_{\\theta}` is obtainend by
    concatenating the vectors of parameters :math:`\\vartheta^{(r)}`
    of each of the :math:`n_b` basis kernels.

    Parameters
    ----------
    _basis_kernels : `BasisKernel` or `list` or `numpy.ndarray`
        Array of events of basis kernels.
        If `BasisKernel`then broadcasted to a unidimensional list.

    """

    def __init__(self, _basis_kernels, n_basis_ker=0, is_null=None,
                 vec_n_param=None,
                 n_param=0, ix_map=None, interval_map=None, phi=None,
                 diff_phi=None, psi=None, diff_psi=None,  upsilon=None,
                 diff_sim_upsilon=None, diff_cross_upsilon=None,
                 allow_simu=True, sim_func=None, l1_norm=None,
                 diff_l1_norm=None, l2_norm=None, l2_dot=None,
                 l2_distance=None, kl_divergence=None,
                 diff_kl_divergence=None):

        self.basis_kernels = _basis_kernels

    # List of basis kernels
    @property
    def basis_kernels(self):
        return self._basis_kernels

    @basis_kernels.setter
    def basis_kernels(self, L):
        if isinstance(L, (list, np.ndarray)):
            self.n_basis_ker = len(L)
        else:
            self.n_basis_ker = 1
            L = [L]

        self._basis_kernels = L

        self.is_null = self.test_nullity()

        self.vec_n_param = [self._basis_kernels[i].get_n_param()
                            for i in range(self.n_basis_ker)]

        self.n_param = sum(self.vec_n_param)

        ix_map, interval_map = self.make_maps()
        self.ix_map = ix_map
        self.interval_map = interval_map

        self.make_kernel_functionals()

        self.make_l1_metrics()

        self.make_l2_metrics()

    @basis_kernels.deleter
    def basis_kernels(self):
        del self._basis_kernels

    # Parameters map
    def make_maps(self):
        ix_map = [None]*self.n_param
        ix_ker = 0
        ix_param = 0
        i = 0
        while i < self.n_param:
            ix_map[i] = {'ker': ix_ker, 'par': ix_param}
            if ix_param == self.vec_n_param[ix_ker]-1:
                ix_ker += 1
                ix_param = 0
                i += 1
            else:
                ix_param += 1
                i += 1
        interval_map = [None]*self.n_basis_ker
        ix_left = 0
        ix_right = self.vec_n_param[0]
        for ix_ker in range(self.n_basis_ker):
            interval_map[ix_ker] = [ix_left, ix_right]
            ix_left = ix_right
            ix_right += self.vec_n_param[ix_ker]
        return ix_map, interval_map

    # Null
    def test_nullity(self):
        for ix_ker in range(self.n_basis_ker):
            if str(type(self._basis_kernels[ix_ker])) != "NullKernel":
                return False
        return True

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

    def is_sbf(self):
        for ix_ker in range(self.n_basis_ker):
            if self._basis_kernels[ix_ker].get_n_param() > 1:
                return False
            if not self._basis_kernels[ix_ker].has_omega_param():
                return False
        return True

    # Bounds
    def get_param_bounds(self):
        bnds = [None]*self.n_param
        for i in range(self.n_param):
            ix_ker = self.ix_map[i]['ker']
            ix_param = self.ix_map[i]['par']
            bnds[i] = self._basis_kernels[ix_ker].get_param_bounds()[ix_param]
        return bnds

    # Param names
    def get_vec_param_names(self):
        vec_param_names = [self._basis_kernels[i].get_param_names()
                           for i in range(self.n_basis_ker)]
        return vec_param_names

    # Availabe interactions
    def is_compatible(self, kernel_2):
        basis_kers_2 = kernel_2._basis_kernels
        for basis_ker_1, basis_ker_2 in itertools.product(self._basis_kernels,
                                                          basis_kers_2):
            # Upsilon 1,2
            if ((not basis_ker_1.is_compatible(basis_ker_2, is_reverse=False))
                and (not basis_ker_2.is_compatible(basis_ker_1,
                                                   is_reverse=True))):
                return False
            # Upsilon 2,1
            if ((not basis_ker_2.is_compatible(basis_ker_1, is_reverse=False))
                and (not basis_ker_1.is_compatible(basis_ker_2,
                                                   is_reverse=True))):
                return False
        return True

    # Kernel functionals
    def make_phi(self):
        def phi(t, params):
            res = 0.
            for ix_ker in range(self.n_basis_ker):
                res += self._basis_kernels[ix_ker].phi(t, params[self.interval_map[ix_ker][0]:self.interval_map[ix_ker][1]])
            return res
        return phi

    def make_diff_phi(self):
        def diff_phi(t, ix_diff, params):
            ix_ker = self.ix_map[ix_diff]['ker']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = self._basis_kernels[ix_ker].diff_phi(t, ix_diff_scaled,
                                                       params[self.interval_map[ix_ker][0]:self.interval_map[ix_ker][1]])
            return res
        return diff_phi

    def make_psi(self):
        def psi(t, params):
            res = 0.
            for ix_ker in range(self.n_basis_ker):
                res += self._basis_kernels[ix_ker].psi(t, params[self.interval_map[ix_ker][0]:self.interval_map[ix_ker][1]])
            return res
        return psi

    def make_diff_psi(self):
        def diff_psi(t, ix_diff, params):
            ix_ker = self.ix_map[ix_diff]['ker']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = self._basis_kernels[ix_ker].diff_psi(t, ix_diff_scaled, params[self.interval_map[ix_ker][0]:self.interval_map[ix_ker][1]])
            return res
        return diff_psi

    def make_upsilon(self):
        def upsilon(kernel_2, t, s, params_1, params_2):
            res = 0.
            n_b_ker_1 = self.n_basis_ker
            n_b_ker_2 = kernel_2.n_basis_ker
            for ix_ker_1, ix_ker_2 in itertools.product(range(n_b_ker_1),
                                                        range(n_b_ker_2)):
                ker1_left = self.interval_map[ix_ker_1][0]
                ker1_right = self.interval_map[ix_ker_1][1]
                ker2_left = kernel_2.interval_map[ix_ker_2][0]
                ker2_right = kernel_2.interval_map[ix_ker_2][1]

                res += self._basis_kernels[ix_ker_1].upsilon(kernel_2._basis_kernels[ix_ker_2], t, s, params_1[ker1_left:ker1_right], params_2[ker2_left:ker2_right])
            return res
        return upsilon

    def make_diff_sim_upsilon(self):
        def diff_sim_upsilon(t, s, ix_diff, params):
            res = 0.
            ix_ker = self.ix_map[ix_diff]['ker']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            ker_lft = self.interval_map[ix_ker][0]
            ker_rgt = self.interval_map[ix_ker][1]
            res += self._basis_kernels[ix_ker].diff_sim_upsilon(t, s, ix_diff_scaled, params[ker_lft:ker_rgt])
            # Upsilon ix_ker, i
            for ix_ker_2 in itertools.chain(range(ix_ker),
                                            range(ix_ker+1, self.n_basis_ker)):
                ker2_lft = self.interval_map[ix_ker_2][0]
                ker2_rgt = self.interval_map[ix_ker_2][1]

                ix_func = 1
                res += self._basis_kernels[ix_ker].diff_cross_upsilon(self._basis_kernels[ix_ker_2], t, s, ix_func, ix_diff_scaled, params[ker_lft:ker_rgt], params[ker2_lft:ker2_rgt])

            # Upsilon i, ix_ker
            for ix_ker_1 in itertools.chain(range(ix_ker),
                                            range(ix_ker+1, self.n_basis_ker)):
                ker1_lft = self.interval_map[ix_ker_1][0]
                ker1_rgt = self.interval_map[ix_ker_1][1]

                ix_func = 2
                res += self._basis_kernels[ix_ker_1].diff_cross_upsilon(self._basis_kernels[ix_ker], t, s, ix_func, ix_diff_scaled, params[ker1_lft:ker1_rgt], params[ker_lft:ker_rgt])
            return res
        return diff_sim_upsilon

    def make_diff_cross_upsilon(self):
        def diff_cross_upsilon(kernel_2, t, s, ix_func, ix_diff, params_1,
                               params_2):
            if ix_func == 1:
                res = 0.
                ix_ker = self.ix_map[ix_diff]['ker']
                ix_diff_scaled = self.ix_map[ix_diff]['par']
                ker_lft = self.interval_map[ix_ker][0]
                ker_rgt = self.interval_map[ix_ker][1]

                # Upsilon ix_ker, i
                for ix_ker_2 in range(kernel_2.n_basis_ker):
                    ker2_lft = kernel_2.interval_map[ix_ker_2][0]
                    ker2_rgt = kernel_2.interval_map[ix_ker_2][1]

                    res += self._basis_kernels[ix_ker].diff_cross_upsilon(kernel_2._basis_kernels[ix_ker_2], t, s, 1, ix_diff_scaled, params_1[ker_lft:ker_rgt], params_2[ker2_lft:ker2_rgt])
                return res
            elif ix_func == 2:
                res = 0.
                ix_ker = kernel_2.ix_map[ix_diff]['ker']
                ix_diff_scaled = kernel_2.ix_map[ix_diff]['par']
                ker_lft = kernel_2.interval_map[ix_ker][0]
                ker_rgt = kernel_2.interval_map[ix_ker][1]
                # Upsilon i, ix_ker
                for ix_ker_1 in range(self.n_basis_ker):
                    ker1_lft = self.interval_map[ix_ker_1][0]
                    ker1_rgt = self.interval_map[ix_ker_1][1]
    
                    res += self._basis_kernels[ix_ker_1].diff_cross_upsilon(kernel_2._basis_kernels[ix_ker], t, s, 2, ix_diff_scaled, params_1[ker1_lft:ker1_rgt], params_2[ker_lft:ker_rgt])
                return res
        return diff_cross_upsilon

    def make_K(self):
        def K(baseline, t, s, params_ker, params_mu):
            res = 0.
            n_b_ker = self.n_basis_ker
            for ix_ker in range(n_b_ker):
                ker_left = self.interval_map[ix_ker][0]
                ker_right = self.interval_map[ix_ker][1]
                res += self._basis_kernels[ix_ker].K(baseline, t, s,
                                                     params_ker[ker_left:ker_right],
                                                     params_mu)
            return res
        return K

    def make_diff_K(self):
        def diff_K(baseline, t, s, ix_func, ix_diff, params_ker, params_mu):
            if ix_func == 1:
                res = 0.
                ix_ker = self.ix_map[ix_diff]['ker']
                ix_diff_scaled = self.ix_map[ix_diff]['par']
                ker_lft = self.interval_map[ix_ker][0]
                ker_rgt = self.interval_map[ix_ker][1]
                # Upsilon i, ix_ker
                res += self._basis_kernels[ix_ker].diff_K(baseline, t, s, 1,
                                                          ix_diff_scaled,
                                                          params_ker[ker_lft:ker_rgt],
                                                          params_mu)
                return res
            elif ix_func == 2:
                res = 0.
                # Upsilon ix_ker, i
                for ix_ker in range(self.n_basis_ker):
                    ker_lft = self.interval_map[ix_ker][0]
                    ker_rgt = self.interval_map[ix_ker][1]
                    res += self._basis_kernels[ix_ker].diff_K(baseline, t, s, 2,
                                                              ix_diff,
                                                              params_ker[ker_lft:ker_rgt],
                                                              params_mu)
                return res
        return diff_K

    def make_kernel_functionals(self):
        phi = self.make_phi()
        self.phi = phi

        diff_phi = self.make_diff_phi()
        self.diff_phi = diff_phi

        psi = self.make_psi()
        self.psi = psi

        diff_psi = self.make_diff_psi()
        self.diff_psi = diff_psi

        upsilon = self.make_upsilon()
        self.upsilon = upsilon

        diff_sim_upsilon = self.make_diff_sim_upsilon()
        self.diff_sim_upsilon = diff_sim_upsilon

        diff_cross_upsilon = self.make_diff_cross_upsilon()
        self.diff_cross_upsilon = diff_cross_upsilon

        K = self.make_K()
        self.K = K

        diff_K = self.make_diff_K()
        self.diff_K = diff_K

    # Operators overload
    def append(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_kernels = list(self._basis_kernels)+list(obj_2._basis_kernels)
            self.basis_kernels = basis_kernels
        if issubclass(type(obj_2), BasisKernel):
            L = self._basis_kernels
            L.append(obj_2)
            self.basis_kernels = L

    def __add__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_kernels = list(self._basis_kernels)+list(obj_2._basis_kernels)
            return KernelModel(basis_kernels)
        elif issubclass(type(obj_2), BasisKernel):
            basis_kernels = self._basis_kernels
            basis_kernels.append(obj_2)
            return KernelModel(basis_kernels)
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))

    def __iadd__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_kernels = self._basis_kernels
            self.basis_kernels = basis_kernels+list(obj_2._basis_kernels)
            return self
        elif issubclass(type(obj_2), BasisKernel):
            L = self._basis_kernels
            L.append(obj_2)
            self.basis_kernels = L
            return self
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))

    # Simfunc
    def make_offset_gen(self, params):
        basis_kernels = self._basis_kernels
        n_basis_ker = self.n_basis_ker
        interval_map = self.interval_map
        vec_branching = np.array(
            [basis_kernels[ix_ker].l1_norm(params[interval_map[ix_ker][0]:
                                                  interval_map[ix_ker][1]])
             for ix_ker in range(n_basis_ker)])
        branching_ratio = np.sum(vec_branching)

        def offset_gen(rng, N=1):
            if branching_ratio == 0:
                return []
            else:
                times = []
                if n_basis_ker == 1:
                    while (len(times) <= N):
                        # Generate additional random times, corresponds to
                        # the expected number that will be needed+100.
                        N_target = int((N+1-len(times))/(1-branching_ratio)+100)
                        simtimes = basis_kernels[0].simu_func(rng, params,
                                                              size=N_target)
                        times += [x for x in simtimes if x > 0]
                else:
                    vec_probas = vec_branching/branching_ratio
                    while (len(times) <= N):
                        N_target = int((N+1-len(times))/(1-branching_ratio)+100)
                        N_basisfunc_list = rng.binomial(n=N_target, p=vec_probas)
                        simtimes_basisfunc = [None]*n_basis_ker
                        for ix_ker in range(n_basis_ker):
                            # Generate additional random times, corresponds to
                            # the expected number that will be needed+100.
                            simtimes_basisfunc[ix_ker] = basis_kernels[ix_ker].simu_func(rng, params[interval_map[ix_ker][0]: interval_map[ix_ker][1]], size=N_basisfunc_list[ix_ker])
                            times += [x for x in simtimes_basisfunc[ix_ker]
                                      if x > 0]
                        rng.shuffle(times)
                return times[:N]
        return offset_gen

    # L1 metrics
    def make_l1_norm(self):
        interval_map = copy.deepcopy(self.interval_map)
        basis_kernels = copy.deepcopy(self._basis_kernels)

        def l1_norm(params):
            res = 0.
            for ix_ker in range(self.n_basis_ker):
                res += basis_kernels[ix_ker].l1_norm(params[interval_map[ix_ker][0]:interval_map[ix_ker][1]])
            return res
        return l1_norm

    def make_diff_l1_norm(self):
        interval_map = copy.deepcopy(self.interval_map)
        basis_kernels = copy.deepcopy(self._basis_kernels)

        def diff_l1_norm(ix_diff, params):
            ix_ker = self.ix_map[ix_diff]['ker']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            return basis_kernels[ix_ker].diff_l1_norm(ix_diff_scaled, params[interval_map[ix_ker][0]:interval_map[ix_ker][1]])

        return diff_l1_norm

    def make_l1_metrics(self):
        l1_norm = self.make_l1_norm()
        self.l1_norm = l1_norm

        diff_l1_norm = self.make_diff_l1_norm()
        self.diff_l1_norm = diff_l1_norm

    def get_l1_perbasis(self, params):
        interval_map = copy.deepcopy(self.interval_map)
        basis_kernels = copy.deepcopy(self._basis_kernels)
        l1_per_basis = np.zeros(self.n_basis_ker)
        for ix_ker in range(self.n_basis_ker):
            l1_per_basis[ix_ker] = basis_kernels[ix_ker].l1_norm(params[interval_map[ix_ker][0]:interval_map[ix_ker][1]])
        return l1_per_basis

    # L2 metrics
    def make_l2_norm(self):
        interval_map = copy.deepcopy(self.interval_map)
        basis_kernels = copy.deepcopy(self._basis_kernels)
        n_basis_ker = self.n_basis_ker
        def l2_norm(params):
            res = 0.
            for ix_ker_1, ix_ker_2 in itertools.product(range(n_basis_ker), range(n_basis_ker)):
                res += basis_kernels[ix_ker_1].l2_dot(basis_kernels[ix_ker_2], params[interval_map[ix_ker_1][0]:interval_map[ix_ker_1][1]], params[interval_map[ix_ker_2][0]:interval_map[ix_ker_2][1]])
            return res
        return l2_norm

    def make_diff_l2_norm(self):
        interval_map = copy.deepcopy(self.interval_map)
        basis_kernels = copy.deepcopy(self._basis_kernels)
        n_basis_ker = self.n_basis_ker
        def diff_l2_norm(ix_diff, params):
            ix_ker = self.ix_map[ix_diff]['ker']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = basis_kernels[ix_ker].diff_l2_norm(ix_diff_scaled, params[interval_map[ix_ker][0]:interval_map[ix_ker][1]])
            
            for ix_ker_2 in itertools.chain(range(ix_ker), range(ix_ker+1, self.n_basis_ker)):
                ix_func = 1
                res += 2*basis_kernels[ix_ker].diff_l2_dot(basis_kernels[ix_ker_2], ix_func, ix_diff_scaled, params[interval_map[ix_ker][0]:interval_map[ix_ker][1]], params[interval_map[ix_ker_2][0]:interval_map[ix_ker_2][1]])
            return res
        return diff_l2_norm

    def make_l2_dot(self):
        interval_map_1 = copy.deepcopy(self.interval_map)
        basis_kernels_1 = copy.deepcopy(self._basis_kernels)
        n_basis_ker_1 = self.n_basis_ker

        def l2_dot(kernel_2, params_1, params_2):
            interval_map_2 = copy.deepcopy(kernel_2.interval_map)
            basis_kernels_2 = copy.deepcopy(kernel_2._basis_kernels)
            n_basis_ker_2 = kernel_2.n_basis_ker
            res = 0.
            for ix_ker_1, ix_ker_2 in itertools.product(range(n_basis_ker_1), range(n_basis_ker_2)):
                res += basis_kernels_1[ix_ker_1].l2_dot(basis_kernels_2[ix_ker_2], params_1[interval_map_1[ix_ker_1][0]:interval_map_1[ix_ker_1][1]], params_2[interval_map_2[ix_ker_2][0]:interval_map_2[ix_ker_2][1]])
            return res
        return l2_dot

    def make_diff_l2_dot(self):
        # Asssuming that the kernels are different !
        # If you wish to compute for the case where kernels are the same,
        # please use diff_l2_norm
        interval_map_1 = copy.deepcopy(self.interval_map)
        basis_kernels_1 = copy.deepcopy(self._basis_kernels)
        n_basis_ker_1 = self.n_basis_ker

        def diff_l2_dot(kernel_2, ix_func, ix_diff, params_1, params_2):
            interval_map_2 = copy.deepcopy(kernel_2.interval_map)
            basis_kernels_2 = copy.deepcopy(kernel_2._basis_kernels)
            n_basis_ker_2 = kernel_2.n_basis_ker
            if ix_func == 1:
                ix_ker_1 = self.ix_map[ix_diff]['ker']
                ix_diff_scaled = self.ix_map[ix_diff]['par']
                res = 0.
                for ix_ker_2 in range(n_basis_ker_2):
                    res += basis_kernels_1[ix_ker_1].diff_l2_dot(basis_kernels_2[ix_ker_2], 1, ix_diff_scaled, params_1[interval_map_1[ix_ker_1][0]:interval_map_1[ix_ker_1][1]], params_2[interval_map_2[ix_ker_2][0]:interval_map_2[ix_ker_2][1]])
                return res
            if ix_func == 2:
                ix_ker_2 = kernel_2.ix_map[ix_diff]['ker']
                ix_diff_scaled_2 = kernel_2.ix_map[ix_diff]['par']
                res = 0.
                for ix_ker_1 in range(n_basis_ker_1):
                    res += basis_kernels_2[ix_ker_2].diff_l2_dot(basis_kernels_1[ix_ker_1], 1, ix_diff_scaled, params_2[interval_map_2[ix_ker_2][0]:interval_map_2[ix_ker_2][1]], params_1[interval_map_1[ix_ker_1][0]:interval_map_1[ix_ker_1][1]])
                return res
        return diff_l2_dot

    def make_l2_metrics(self):
        l2_norm = self.make_l2_norm()
        self.l2_norm = l2_norm

        diff_l2_norm = self.make_diff_l2_norm()
        self.diff_l2_norm = diff_l2_norm

        l2_dot = self.make_l2_dot()
        self.l2_dot = l2_dot

        diff_l2_dot = self.make_diff_l2_dot()
        self.diff_l2_dot = diff_l2_dot

    # L2 projection
    def get_l2_projection_loss(self, params, kernel_2, ker_param_2,
                               sbf=False, Q=None, c=None):
        norm_ker_2 = kernel_2.l2_norm(ker_param_2)
        res = norm_ker_2
        if sbf:
            # Quadratic term
            res += params.dot(Q.dot(params))
            # Linear term
            res += c.dot(params)
            # Constant term
            res += norm_ker_2
            return 0.5*res
        else:
            return 0.5*(self.l2_norm(params)
                        - 2.*self.l2_dot(kernel_2, params, ker_param_2)
                        + norm_ker_2)

    def diff_l2_projection_loss(self, params, kernel_2, ker_param_2,
                                sbf=False, Q=None, c=None):
        n_param = len(params)
        grad = np.zeros(n_param)
        if sbf:
            return Q.dot(params)+c
        else:
            for ix_param in range(n_param):
                grad[ix_param] = (self.diff_l2_norm(ix_param, params)
                                  - 2.*self.diff_l2_dot(kernel_2, 1, ix_param,
                                                        params, ker_param_2))
        return grad

    def make_l2_proj_Q(self):
        n_param = self.n_param
        Q = np.zeros((n_param, n_param))
        for m in range(n_param):
            for n in range(m+1):
                # Compute Q[m][n] for m <= n
                Q[m][n] = self._basis_kernels[m].l2_dot(self._basis_kernels[n],
                                                        np.ones(1),
                                                        np.ones(1))
                # If m!=n, Q[n][m] = Q[m][n]
                if m != n:
                    Q[n][m] = Q[m][n]
        return Q

    def make_l2_proj_c(self, kernel_2, ker_param_2):
        n_param = self.n_param
        c = np.zeros(n_param)
        for m in range(n_param):
            n_basis_ker_2 = len(kernel_2._basis_kernels)
            interval_map_2 = copy.deepcopy(kernel_2.interval_map)
            for ix_ker_2 in range(n_basis_ker_2):
                basis_ker_2 = kernel_2._basis_kernels[ix_ker_2]
                basis_ker_param_2 = ker_param_2[interval_map_2[ix_ker_2][0]:interval_map_2[ix_ker_2][1]]
                c[m] -= self._basis_kernels[m].l2_dot(basis_ker_2,
                                                      np.ones(1),
                                                      basis_ker_param_2)
        return c

    def get_l2_projection(self, kernel_2, ker_param_2, n_iter=1000,
                          params_0=None, try_sbf=True, Q=None, c=None,
                          solver=None, log_error=False, rng=None,
                          seed=1234,
                          verbose=False, **kwargs):
        # Random number generator
        if rng is None:
            rng = np.random.default_rng(seed)
        # Bounds and number of parameters
        bnds = self.get_param_bounds()
        n_param = self.n_param
        # Initialise parameters
        if params_0 is None:
            params_0 = bnds+rng.uniform(low=0.0, high=1., size=n_param)
        params = copy.deepcopy(params_0)
        # Initialise solver and logger
        if solver is None:
            solver = ADAM()
        if log_error:
            l2_err_log = np.zeros(n_iter)
        else:
            l2_err_log = None
        # Initialise quadratic program if sbf
        if self.is_sbf() and try_sbf:
            sbf = True
            # Quadratic term coefficients
            if Q is None:
                Q = self.make_l2_proj_Q()
            # Linear term coefficients
            if c is None:
                c = self.make_l2_proj_c(kernel_2, ker_param_2)
        else:
            sbf = False
            Q = None
            c = None

        # Numerical scheme
        for t in tqdm(range(n_iter), disable=not verbose):
            # Compute L2 projection loss gradient estimate for parameters x_k
            g_t = self.diff_l2_projection_loss(params, kernel_2, ker_param_2,
                                               sbf=sbf, Q=Q, c=c)
            # Apply solver iteration then project into space of parameters
            params = solver.iterate(t, params, g_t)
            params = np.maximum(params, bnds)

            if log_error:
                l2_err_log[t] = self.get_l2_projection_loss(params, kernel_2,
                                                            ker_param_2,
                                                            sbf=sbf, Q=Q, c=c)
        res = {'params': params, 'log': l2_err_log}
        return res

    # KL divergence
