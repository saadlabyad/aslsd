# License: BSD 3 clause

import copy
import itertools

import numpy as np
from tqdm import tqdm

from aslsd.basis_baselines.basis_baseline import BasisBaseline
from aslsd.evaluation import goodness_of_fit as gof


class BaselineModel():
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

    def __init__(self, _basis_mus, default_simu_method='inverse_cdf',
                 n_basis_mus=0,
                 vec_n_param=None,
                 n_param=0, ix_map=None, interval_map=None):

        self.basis_mus = _basis_mus
        self.default_simu_method = default_simu_method

    # List of basis baselines
    @property
    def basis_mus(self):
        return self._basis_mus

    @basis_mus.setter
    def basis_mus(self, L):
        if isinstance(L, (list, np.ndarray)):
            self.n_basis_mus = len(L)
        else:
            self.n_basis_mus = 1
            L = [L]

        self._basis_mus = L

        self.vec_n_param = [self._basis_mus[i].get_n_param()
                            for i in range(self.n_basis_mus)]

        self.n_param = sum(self.vec_n_param)

        ix_map, interval_map = self.make_maps()
        self.ix_map = ix_map
        self.interval_map = interval_map

        self.make_baseline_functionals()
        self.make_simulation_functionals()

    @basis_mus.deleter
    def basis_mus(self):
        del self._basis_mus

    # Parameters map
    def make_maps(self):
        ix_map = [None]*self.n_param
        ix_mu = 0
        ix_param = 0
        i = 0
        while i < self.n_param:
            ix_map[i] = {'mu': ix_mu, 'par': ix_param}
            if ix_param == self.vec_n_param[ix_mu]-1:
                ix_mu += 1
                ix_param = 0
                i += 1
            else:
                ix_param += 1
                i += 1
        interval_map = [None]*self.n_basis_mus
        ix_left = 0
        ix_right = self.vec_n_param[0]
        for ix_mu in range(self.n_basis_mus):
            interval_map[ix_mu] = [ix_left, ix_right]
            ix_left = ix_right
            ix_right += self.vec_n_param[ix_mu]
        return ix_map, interval_map

    # Bounds
    def get_param_bounds(self):
        bnds = np.zeros(self.n_param)
        for i in range(self.n_param):
            ix_mu = self.ix_map[i]['mu']
            ix_param = self.ix_map[i]['par']
            bnds[i] = self._basis_mus[ix_mu].get_param_bounds()[ix_param]
        return bnds

    # Param names
    def get_vec_param_names(self):
        vec_param_names = [self._basis_mus[i].get_param_names()
                           for i in range(self.n_basis_mus)]
        return vec_param_names

    # Availabe interactions
    def is_compatible(self, kernel):
        basis_kers = kernel._basis_kernels
        for basis_mu, basis_ker in itertools.product(self._basis_mus,
                                                     basis_kers):
            # K 1,2
            if ((not basis_mu.is_compatible(basis_ker,
                                            is_reverse=False))
                and (not basis_ker.is_compatible(basis_mu,
                                                 is_reverse=True))):
                return False
            # Upsilon 2,1
            if ((not basis_ker.is_compatible(basis_mu, is_reverse=False))
                and (not basis_mu.is_compatible(basis_ker,
                                                is_reverse=True))):
                return False
        return True

    # Baseline functionals
    def make_mu(self):
        def mu(t, params):
            res = 0.
            for ix_mu in range(self.n_basis_mus):
                res += self._basis_mus[ix_mu].mu(t, params[self.interval_map[ix_mu][0]:self.interval_map[ix_mu][1]])
            return res
        return mu

    def make_diff_mu(self):
        def diff_mu(t, ix_diff, params):
            ix_mu = self.ix_map[ix_diff]['mu']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = self._basis_mus[ix_mu].diff_mu(t, ix_diff_scaled,
                                                  params[self.interval_map[ix_mu][0]:self.interval_map[ix_mu][1]])
            return res
        return diff_mu

    def make_M(self):
        def M(t, params):
            res = 0.
            for ix_mu in range(self.n_basis_mus):
                res += self._basis_mus[ix_mu].M(t, params[self.interval_map[ix_mu][0]:self.interval_map[ix_mu][1]])
            return res
        return M

    def make_diff_M(self):
        def diff_M(t, ix_diff, params):
            ix_mu = self.ix_map[ix_diff]['mu']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = self._basis_mus[ix_mu].diff_M(t, ix_diff_scaled,
                                                params[self.interval_map[ix_mu][0]:self.interval_map[ix_mu][1]])
            return res
        return diff_M

    def make_baseline_functionals(self):
        mu = self.make_mu()
        self.mu = mu

        diff_mu = self.make_diff_mu()
        self.diff_mu = diff_mu

        M = self.make_M()
        self.M = M

        diff_M = self.make_diff_M()
        self.diff_M = diff_M

    # Simulation functionals
    def make_compensator(self):
        if len(self.basis_mus) == 1:
            compensator = self.basis_mus[0].compensator
            return compensator
        elif len(self.basis_mus) >= 2:
            pass

    def make_inv_compensator(self):
        if len(self.basis_mus) == 1:
            inverse_compensator = self.basis_mus[0].inverse_compensator
            return inverse_compensator
        elif len(self.basis_mus) >= 2:
            pass

    def make_intensity_bound(self):
        # Return baseline object which intensity upper bounds that of
        # the baseline object.
        pass

    def make_simulation_functionals(self):
        self.compensator = self.make_compensator()
        self.inverse_compensator = self.make_inv_compensator()
        self.intensity_bound = self.make_intensity_bound()

    # Simulation
    def simulate_inverse_cdf(self, T_f, params, rng=None, seed=1234):
        if rng is None:
            rng = np.random.default_rng(seed)
        # Get the compensator transformed terminal time
        T_hom = self.compensator(T_f, params)
        Nim = rng.poisson(T_hom)
        times_hom = rng.uniform(low=0.0, high=T_hom, size=Nim)
        times_hom.sort()
        times = self.inverse_compensator(times_hom, params)
        return times

    def simulate_thinning(self, T_f, params, rng=None, seed=1234):
        if rng is None:
            rng = np.random.default_rng(seed)
        upper_base = self.intensity_bound(params)
        times_upper = upper_base.simulate_inverse_cdf(T_f, [], rng=rng)
        N_upper = len(times_upper)
        if N_upper == 0:
            return np.array([], dtype=float)
        rejection_probas = rng.uniform(low=0., high=1., size=N_upper)
        ratios = self.mu(times_upper, params)/upper_base.mu(times_upper, [])
        ixs_accepted = np.where(rejection_probas <= ratios)[0]
        return times_upper[ixs_accepted]

    def simulate(self, T_f, params, rng=None, seed=1234):
        if self.default_simu_method == 'inverse_cdf':
            times = self.simulate_inverse_cdf(T_f, params, rng=rng, seed=seed)
        elif self.default_simu_method == 'thinning':
            times = self.simulate_thinning(T_f, params, rng=rng, seed=seed)
        return times

    # Residuals
    def get_residuals(self, times, params):
        times_comp = self.compensator(times, params)
        residuals = times_comp[1:]-times_comp[:-1]
        return residuals

    # Operators overload
    def append(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_mus = list(self._basis_mus)+list(obj_2._basis_mus)
            self.basis_mus = basis_mus
        if issubclass(type(obj_2), BasisBaseline):
            L = self._basis_mus
            L.append(obj_2)
            self.basis_mus = L

    def __add__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_mus = list(self._basis_mus)+list(obj_2._basis_mus)
            return BaselineModel(basis_mus)
        elif issubclass(type(obj_2), BasisBaseline):
            basis_mus = self._basis_mus
            basis_mus.append(obj_2)
            return BaselineModel(basis_mus)
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))

    def __iadd__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_mus = self._basis_mus
            self.basis_mus = basis_mus+list(obj_2._basis_mus)
            return self
        elif issubclass(type(obj_2), BasisBaseline):
            L = self._basis_mus
            L.append(obj_2)
            self.basis_mus = L
            return self
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))

