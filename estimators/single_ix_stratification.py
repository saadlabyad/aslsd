# License: BSD 3 clause

import numpy as np


class SingleIxStratification:
    def __init__(self, n_events_i, **kwargs):
        self.n_events_i = n_events_i
        self.n_exact = kwargs.get('n_exact', 10**4)
        self.strata = kwargs.get('strata', None)
        self.n_strata = kwargs.get('n_strata', None)
        self.abs_alloc = kwargs.get('abs_alloc', None)

        self.n_exact = min(self.n_exact, self.n_events_i-1)

        # m_spec is the index of the term starting from which we evalue
        # functions at all indices
        self.m_spec = self.n_events_i-self.n_exact

        if self.strata is None:
            self.strata = self.get_default_strata()
        self.n_strata = len(self.strata)

        strata_sizes = self.get_stratification_size()
        self.strata_sizes = strata_sizes

        if self.abs_alloc is None:
            abs_alloc = self.get_default_abs_alloc()
            self.abs_alloc = abs_alloc
        else:
            if len(self.abs_alloc) == self.n_strata:
                self.abs_alloc = abs_alloc
            else:
                raise ValueError("Incompatible lengths between ",
                                 "stratification and absolute allocation")

    def get_default_strata(self):
        if self.m_spec <= 100:
            return [[0, self.m_spec-1]]
        else:
            L = np.logspace(0, np.log10(self.m_spec), num=10)
            # Correct for roundoff errors
            L[-1] = self.m_spec
            L = list(L)
            L = [int(x) for x in L]
            return [[L[i], L[i+1]-1] for i in range(len(L)-1)]

    def get_default_abs_alloc(self):
        return np.array([min(10**3, self.strata_sizes[x])
                         for x in range(self.n_strata)])

    def get_stratification_size(self):
        strata_sizes = np.array([L[1]-L[0]+1 for L in self.strata])
        return strata_sizes
