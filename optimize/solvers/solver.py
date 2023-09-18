# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np


class Solver(ABC):
    """
    Abstract class for optimization solvers.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        # Track Iteration
        self.t = 0
        # Learning rate
        self.learning_rate = self.make_learning_rate(**kwargs)
        # Clipping
        self.clipping = kwargs.get('clipping', False)
        self.clipping_window = kwargs.get('clipping_window', 10)
        self.clipping_tol = kwargs.get('clipping_tol', 1.1)
        self.clipping_offset = kwargs.get('clipping_offset', 1.)

    @abstractmethod
    def iterate(self, t, x, grad):
        pass

# =============================================================================
# Learning Rate
# =============================================================================
    def make_learning_rate(self, **kwargs):
        rate_type = kwargs.get('learning_rate_type', 'constant')
        rate_0 = kwargs.get('inital_learning_rate', 10**-2)

        # Constant rate
        if rate_type == 'constant':
            def func(t):
                return rate_0

        # Exponential rate
        elif rate_type == 'exponential':
            rate_0 = kwargs.get('inital_learning_rate', 10**-1)
            divider = kwargs.get('learning_rate_divider', 2)
            period = kwargs.get('learning_rate_period', 200)

            def func(t):
                q = t//period
                return rate_0/float(divider**q)

        # Linear rate
        elif rate_type == 'linear':
            rate_0 = kwargs.get('inital_learning_rate', 1.)
            offset = kwargs.get('learning_rate_offset', 1.)
            slope = kwargs.get('learning_rate_slope', .1)

            def func(t):
                return rate_0/(offset+slope*t)

        # Custom rate
        elif rate_type == 'custom':
            func = kwargs.get('learning_rate_custom', None)
            if func is None:
                raise ValueError("If 'rate_type' is 'custom', a rate",
                                 "function must be defined in",
                                 "'learning_rate_custom'.")
        return func

# =============================================================================
# CLipping
# =============================================================================
    def clip(self, grad):
        if not self.clipping:
            return grad
        if self.t == 0:
            self.list_clip_y = np.zeros((len(grad), self.clipping_window))
        if self.t < self.clipping_window:
            g_t = 0.+grad
            self.list_clip_y[:, self.t] = np.abs(g_t)
        else:
            # Clip gradient g_t
            y_max = np.max(self.list_clip_y, axis=1)
            y_ref = self.clipping_offset+self.clipping_tol*y_max
            g_t = np.clip(grad, -y_ref, y_ref)
            # Update reference
            self.list_clip_y[:, :-1] = self.list_clip_y[:, 1:]
            self.list_clip_y[:, -1] = np.abs(g_t)
        # Return result
        return g_t
