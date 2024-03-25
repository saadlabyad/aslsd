# License: BSD 3 clause

import pickle

import numpy as np


class DoubleIxStratification:
    def __init__(self, h_max=0, **kwargs):
        self.h_max = h_max
        self.n_exact = 0
        if self.h_max > 0:
            self.initialize_strata(**kwargs)

    def initialize_strata(self, **kwargs):        
        self.n_adaptive_iters = kwargs.get('n_adaptive_iters', 1)
        self.n_samples_adaptive = kwargs.get('n_samples_adaptive', 10**4)
        
        self.adaptive_rel_alloc_0 = kwargs.get('adaptive_rel_alloc_0', None)
        self.adaptive_rel_alloc = kwargs.get('adaptive_rel_alloc', None)
        self.ema_weight = kwargs.get('ema_weight', 0.6)

        # Adaptive domain
        if isinstance(self.n_samples_adaptive, (list, np.ndarray)):
            self.n_adaptive_iters = len(self.n_samples_adaptive)
        else:
            self.n_samples_adaptive = np.array([self.n_samples_adaptive])
            self.n_adaptive_iters = 1
        
        adaptive_strata = kwargs.get('adaptive_strata', None)
        if adaptive_strata is None:
            max_lag_adaptive = kwargs.get('max_lag_adaptive', 30)
            adaptive_strata = self.get_default_adaptive_strata(max_lag_adaptive)
        self.adaptive_strata = adaptive_strata

        self.n_adaptive_strata = len(self.adaptive_strata)

        if self.adaptive_rel_alloc_0 is None:
            adaptive_rel_alloc_0 = self.get_default_adaptive_rel_alloc_0()
            self.adaptive_rel_alloc_0 = adaptive_rel_alloc_0
        else:
            if len(self.adaptive_rel_alloc_0) != self.n_adaptive_strata:
                raise ValueError("Incompatible lengths between adaptive ",
                                 "stratification and absolute allocation")
        self.guess_rel_alloc = 0+self.adaptive_rel_alloc_0
        self.adaptive_rel_alloc = 0+self.adaptive_rel_alloc_0

        # Non-adaptive domain
        # Strata def
        self.nonadaptive_strata = kwargs.get('nonadaptive_strata', None)
        if self.nonadaptive_strata is None:
            self.nonadaptive_strata = self.get_default_nonadaptive_strata()
        self.n_nonadaptive_strata = len(self.nonadaptive_strata)
        # Allocation
        self.nonadaptive_abs_alloc = kwargs.get('nonadaptive_abs_alloc', None)
        self.nonadaptive_sample_size = kwargs.get('nonadaptive_sample_size',
                                                  10**4)
        if self.nonadaptive_abs_alloc is not None:
            if len(self.nonadaptive_abs_alloc) != self.n_nonadaptive_strata:
                raise ValueError("Incompatible lengths between non adaptive ",
                                 "stratification and absolute allocation")

    def get_lag_strata_sizes(self, strata, lag_sizes, i, j):
        n_strata = len(strata)
        strata_sizes = [None]*n_strata
        for ix_strata in range(n_strata):
            lag_min = strata[ix_strata][0]
            lag_max = strata[ix_strata][1]
            strata_sizes[ix_strata] = np.sum(lag_sizes[i][j][lag_min:lag_max+1])
        return strata_sizes

    def get_lag_strata_finitepop(self):
        return [np.sqrt(size/float(size-1.))
                for size in self.adaptive_strata_sizes]

    def get_stratification_size(self, lag_sizes, i, j):
        adaptive_strata_sizes = self.get_lag_strata_sizes(self.adaptive_strata,
                                                          lag_sizes, i, j)
        self.adaptive_strata_sizes = adaptive_strata_sizes
        self.adaptive_strata_fpop = self.get_lag_strata_finitepop()

        for ix in range(self.n_adaptive_iters):
            self.n_samples_adaptive[ix] = min(self.n_samples_adaptive[ix], sum(self.adaptive_strata_sizes))
        nonadaptive_strata_sizes = self.get_lag_strata_sizes(self.nonadaptive_strata, lag_sizes, i, j)
        self.nonadaptive_strata_sizes = nonadaptive_strata_sizes
        if self.nonadaptive_abs_alloc is None:
            nonadaptive_sample_size_perstratum = int(self.nonadaptive_sample_size/self.n_nonadaptive_strata)
            nonadaptive_sample_size_perstratum = max(1, nonadaptive_sample_size_perstratum)
            nonadaptive_abs_alloc = nonadaptive_sample_size_perstratum*np.ones(self.n_nonadaptive_strata, dtype=int)
        self.nonadaptive_abs_alloc = np.minimum(nonadaptive_abs_alloc, self.nonadaptive_strata_sizes)
        self.nonadaptive_sample_size = np.sum(self.nonadaptive_abs_alloc)

    def get_default_adaptive_strata(self, max_lag_adaptive):
        adpative_h_max = min(max_lag_adaptive, self.h_max)
        return np.array([[h, h] for h in range(1, adpative_h_max)])

    def get_default_nonadaptive_strata(self):
        nonadaptive_h_min = self.adaptive_strata[-1][1]+1
        L = np.logspace(np.log10(nonadaptive_h_min), np.log10(self.h_max+1),
                        num=10)
        L[0] = nonadaptive_h_min
        L[-1] = self.h_max+1
        L = list(L)
        L = [int(x) for x in L]
        return [[L[i], L[i+1]-1] for i in range(len(L)-1)]

    def get_default_adaptive_rel_alloc_0(self):
        x = 1./self.n_adaptive_strata
        alloc = x*np.ones(self.n_adaptive_strata)
        return alloc

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
