# License: BSD 3 clause

import numpy as np

from aslsd.functionals.impact_functions.basis_impact import BasisImpact


class ImpactFunction():
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
    _basis_impacts : `BasisKernel` or `list` or `numpy.ndarray`
        Array of events of basis kernels.
        If `BasisKernel`then broadcasted to a unidimensional list.

    """

    def __init__(self, _basis_impacts, n_basis_imp=0, is_null=None,
                 vec_n_param=None,
                 n_param=0, ix_map=None, interval_map=None, impact=None,
                 diff_impact=None):

        self.basis_impacts = _basis_impacts

    # List of basis kernels
    @property
    def basis_impacts(self):
        return self._basis_impacts

    @basis_impacts.setter
    def basis_impacts(self, L):
        if isinstance(L, (list, np.ndarray)):
            self.n_basis_imp = len(L)
        else:
            self.n_basis_imp = 1
            L = [L]

        self._basis_impacts = L

        self.vec_n_param = [self._basis_impacts[i].get_n_param()
                            for i in range(self.n_basis_imp)]

        self.n_param = sum(self.vec_n_param)

        ix_map, interval_map = self.make_maps()
        self.ix_map = ix_map
        self.interval_map = interval_map

        self.make_impact_functionals()

    @basis_impacts.deleter
    def basis_impacts(self):
        del self._basis_impacts

    # Parameters map
    def make_maps(self):
        ix_map = [None]*self.n_param
        ix_imp = 0
        ix_param = 0
        i = 0
        while i < self.n_param:
            ix_map[i] = {'imp': ix_imp, 'par': ix_param}
            if ix_param == self.vec_n_param[ix_imp]-1:
                ix_imp += 1
                ix_param = 0
                i += 1
            else:
                ix_param += 1
                i += 1
        interval_map = [None]*self.n_basis_imp
        ix_left = 0
        ix_right = self.vec_n_param[0]
        for ix_imp in range(self.n_basis_imp):
            interval_map[ix_imp] = [ix_left, ix_right]
            ix_left = ix_right
            ix_right += self.vec_n_param[ix_imp]
        return ix_map, interval_map

    # Bounds
    def get_param_bounds(self):
        bnds = [None]*self.n_param
        for i in range(self.n_param):
            ix_imp = self.ix_map[i]['imp']
            ix_param = self.ix_map[i]['par']
            bnds[i] = self._basis_impacts[ix_imp].get_param_bounds()[ix_param]
        return bnds

    # Param names
    def get_vec_param_names(self):
        vec_param_names = [self._basis_impacts[i].get_param_names()
                           for i in range(self.n_basis_imp)]
        return vec_param_names

    # Impact functionals
    def make_impact(self):
        def impact(xi, params):
            res = 0.
            for ix_imp in range(self.n_basis_imp):
                res += self._basis_impacts[ix_imp].impact(xi, params[self.interval_map[ix_imp][0]:self.interval_map[ix_imp][1]])
            return res
        return impact

    def make_diff_impact(self):
        def diff_impact(xi, ix_diff, params):
            ix_imp = self.ix_map[ix_diff]['imp']
            ix_diff_scaled = self.ix_map[ix_diff]['par']
            res = self._basis_impacts[ix_imp].diff_impact(xi, ix_diff_scaled,
                                                          params[self.interval_map[ix_imp][0]:self.interval_map[ix_imp][1]])
            return res
        return diff_impact

    def make_impact_functionals(self):
        impact = self.make_impact()
        self.impact = impact

        diff_impact = self.make_diff_impact()
        self.diff_impact = diff_impact

    # Operators overload
    def append(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_impacts = list(self._basis_impacts)+list(obj_2._basis_impacts)
            self.basis_impacts = basis_impacts
        if issubclass(type(obj_2), BasisImpact):
            L = self._basis_impacts
            L.append(obj_2)
            self.basis_impacts = L

    def __add__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_impacts = list(self._basis_impacts)+list(obj_2._basis_impacts)
            return ImpactFunction(basis_impacts)
        elif issubclass(type(obj_2), BasisImpact):
            basis_impacts = self._basis_impacts
            basis_impacts.append(obj_2)
            return ImpactFunction(basis_impacts)
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))

    def __iadd__(self, obj_2):
        if type(obj_2) == self.__class__:
            basis_impacts = self._basis_impacts
            self.basis_impacts = basis_impacts+list(obj_2._basis_impacts)
            return self
        elif issubclass(type(obj_2), BasisImpact):
            L = self._basis_impacts
            L.append(obj_2)
            self.basis_impacts = L
            return self
        else:
            raise TypeError("Summation is not possible for objects of type ",
                            type(obj_2))
