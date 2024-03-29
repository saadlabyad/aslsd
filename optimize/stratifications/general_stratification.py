# License: BSD 3 clause
import copy
import pickle

import numpy as np

from aslsd.utilities import useful_functions as uf


class GeneralStratification:
    def __init__(self, n=0, **kwargs):
        """
        General Monte-Carlo stratification for a set of n elements.

        Parameters
        ----------
        n : `int`, optional
            umber of elements in the estimated set. The default is 0.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.n = n
        if self.n > 0:
            self.initialize_strata(**kwargs)

    def initialize_strata(self, **kwargs):
        # Exact
        # m_spec is the index of the term starting from which we evaluate
        # functions at all indices
        self.start_exact_reverse = kwargs.get('start_exact_reverse', True)

        # Exact domain
        self.exact_domain = kwargs.get('exact_domain', None)
        if self.exact_domain is None:
            self.n_exact = kwargs.get('n_exact', 1000)
            self.n_exact = min(self.n, self.n_exact)  # Sanity check
            if self.n_exact > 0:
                ixs_exact = np.arange(self.n_exact)
                if self.start_exact_reverse:
                    self.exact_domain = self.n-self.n_exact+ixs_exact
                else:
                    self.exact_domain = ixs_exact
        else:
            self.n_exact = len(self.exact_domain)

        # Adaptive Strata
        self.n_nonadaptive_strata = kwargs.get('n_nonadaptive_strata', None)
        self.adaptive_strata = kwargs.get('adaptive_strata', None)
        if self.n == self.n_exact:
            self.n_adaptive_strata = 0
        else:
            if self.adaptive_strata is None:
                self.n_adaptive_strata = kwargs.get('n_adaptive_strata', 10)
                self.n_adaptive_strata = min(self.n_adaptive_strata,
                                             self.n-self.n_exact)
            else:
                self.n_adaptive_strata = len(self.adaptive_strata)
        if self.n_adaptive_strata > 0:
            n_samples_adaptive = kwargs.get('n_samples_adaptive', 10**3)
            if uf.is_array(n_samples_adaptive):
                self.n_adaptive_iters = len(self.n_samples_adaptive)
            else:
                self.n_adaptive_iters = kwargs.get('n_adaptive_iters', 1)
                if n_samples_adaptive is not None:
                    n_samples_adaptive = n_samples_adaptive*np.ones(self.n_adaptive_iters)

            if self.adaptive_strata is None:
                if self.start_exact_reverse:
                    r_bound_adaptive = self.n-self.n_exact-1
                    if self.n_nonadaptive_strata == 0:
                        l_bound_adaptive = 0
                    else:
                        l_bound_adaptive = max(0, r_bound_adaptive-10**3*self.n_adaptive_strata)
                else:
                    l_bound_adaptive = self.n_exact
                    if self.n_nonadaptive_strata == 0:
                        r_bound_adaptive = self.n-1
                    else:
                        r_bound_adaptive = min(self.n-1,
                                               l_bound_adaptive+10**3*self.n_adaptive_strata)
                # Now make adaptive strata bounds between l_bound and r_bound
                self.adaptive_strata_type = kwargs.get('adaptive_strata_type', 'linear')
                adaptive_strata = uf.partition_space(l_bound_adaptive,
                                                     r_bound_adaptive,
                                                     self.n_adaptive_strata,
                                                     self.adaptive_strata_type)
                self.adaptive_strata = copy.deepcopy(adaptive_strata)

            self.adaptive_strata_sizes = self.get_stratification_size(self.adaptive_strata)
            self.total_size_adaptive = np.sum(self.adaptive_strata_sizes)
            self.n_samples_adaptive = np.minimum(n_samples_adaptive,
                                                 self.total_size_adaptive)
            self.n_samples_adaptive = self.n_samples_adaptive.astype(int)
            self.adaptive_sample_size = np.sum(self.n_samples_adaptive)
            self.adaptive_strata_fpop = np.array([np.sqrt(size/float(size-1.)) for size in self.adaptive_strata_sizes])
            self.ema_weight = kwargs.get('ema_weight', .6)
            
            default_ada_rel_alloc_type = kwargs.get('default_ada_rel_alloc_type', 'uniform')
            default_ada_rel_alloc = self.get_default_rel_alloc(self.n_adaptive_strata, alloc_type=default_ada_rel_alloc_type, strata_sizes=self.adaptive_strata_sizes)
            self.adaptive_rel_alloc = copy.deepcopy(default_ada_rel_alloc)
            self.guess_rel_alloc = copy.deepcopy(default_ada_rel_alloc)
        else:
            self.total_size_adaptive = 0
            self.adaptive_sample_size = 0

        # Non-adaptive Strata
        self.nonadaptive_strata = kwargs.get('nonadaptive_strata', None)
        nonadaptive_sample_size = kwargs.get('nonadaptive_sample_size',
                                             10**4)
        self.total_size_nonadaptive = (self.n-self.n_exact
                                       - self.total_size_adaptive)
        if (self.total_size_nonadaptive == 0) or (nonadaptive_sample_size == 0):
            self.n_nonadaptive_strata = 0
        else:
            if self.nonadaptive_strata is None:
                self.n_nonadaptive_strata = min(10,
                                                self.total_size_nonadaptive)
            else:
                self.n_nonadaptive_strata = len(self.nonadaptive_strata)
        if self.n_nonadaptive_strata > 0:
            if self.nonadaptive_strata is None:
                if self.start_exact_reverse:
                    if self.n_adaptive_strata > 0:
                        r_bound_nonadaptive = self.adaptive_strata[0][0]-1
                    else:
                        r_bound_nonadaptive = self.n-self.n_exact-1
                    l_bound_nonadaptive = 0
                else:
                    if self.n_adaptive_strata > 0:
                        l_bound_nonadaptive = self.adaptive_strata[-1][1]+1
                    else:
                        l_bound_nonadaptive = self.n_exact
                    r_bound_nonadaptive = self.n-1
            # Now make non adaptive strata bounds between l_bound and r_bound
            self.nonadaptive_strata_type = kwargs.get('nonadaptive_strata_type', 'linear')
            nonadaptive_strata = uf.partition_space(l_bound_nonadaptive,
                                                    r_bound_nonadaptive,
                                                    self.n_nonadaptive_strata,
                                                    self.nonadaptive_strata_type)
            self.nonadaptive_strata = copy.deepcopy(nonadaptive_strata)
            self.nonadaptive_strata_sizes = self.get_stratification_size(self.nonadaptive_strata)
            nonadaptive_abs_alloc = kwargs.get('nonadaptive_abs_alloc', None)
            if nonadaptive_abs_alloc is None:
                nonadaptive_sample_size_perstratum = int(nonadaptive_sample_size/self.n_nonadaptive_strata)
                nonadaptive_sample_size_perstratum = max(1, nonadaptive_sample_size_perstratum)
                nonadaptive_abs_alloc = nonadaptive_sample_size_perstratum*np.ones(self.n_nonadaptive_strata, dtype=int)
            self.nonadaptive_abs_alloc = np.minimum(nonadaptive_abs_alloc, self.nonadaptive_strata_sizes)
            self.nonadaptive_sample_size = np.sum(self.nonadaptive_abs_alloc)
        else:
            self.total_size_nonadaptive = 0
            self.nonadaptive_sample_size = 0
        self.total_eval_size = self.n_exact+self.adaptive_sample_size+self.nonadaptive_sample_size

    def get_default_abs_alloc(self):
        return np.array([min(10**3, self.strata_sizes[x])
                         for x in range(self.n_strata)])

    def get_stratification_size(self, strata):
        strata_sizes = np.array([L[1]-L[0]+1 for L in strata], dtype=int)
        return strata_sizes

    def get_default_rel_alloc(self, n_strata, alloc_type='uniform',
                              strata_sizes=None, **kwargs):
        if alloc_type == 'uniform':
            rate = 1./n_strata
            rel_alloc = np.zeros(n_strata)
            rel_alloc[:-1] = 1./n_strata
            rel_alloc[-1] = 1.-np.sum(rel_alloc)
        elif alloc_type == 'proportional':
            rel_alloc = np.zeros(n_strata)
            strata_sizes_ = np.array(copy.deepcopy(strata_sizes))
            strata_sizes_ = strata_sizes_.astype(float)
            total_size = np.sum(strata_sizes_)
            rel_alloc = strata_sizes_/total_size
            rel_alloc[-1] = 1.-np.sum(rel_alloc[:-1])
        return rel_alloc

# =============================================================================
# Serialization
# =============================================================================
    def save(self, file, **kwargs):
        pickle_out = open(file, "wb", **kwargs)
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def load(self, file, **kwargs):
        pickle_in = open(file, "rb")
        strf = pickle.load(pickle_in)
        strf_attr = vars(strf)
        for attr_name in strf_attr.keys():
            setattr(self, attr_name, strf_attr[attr_name])
