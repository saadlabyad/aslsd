# License: BSD 3 clause

from abc import ABC, abstractmethod


class Solver(ABC):
    """
    Abstract class for optimization solvers.

    Parameters
    ----------

    """

    def __init__(self, **kwargs):
        pass

    def make_learning_rate(self, **kwargs):
        rate_type = kwargs.get('learning_rate_type', 'constant')
        rate_0 = kwargs.get('inital_learning_rate', 10**-2)
        if rate_type == 'constant':
            def func(t):
                return rate_0
        elif rate_type == 'exponential':
            rate_0 = kwargs.get('inital_learning_rate', 10**-1)
            divider = kwargs.get('learning_rate_divider', 2)
            period = kwargs.get('learning_rate_period', 200)

            def func(t):
                q = t//period
                return rate_0/float(divider**q)
        elif rate_type == 'linear':
            divider = kwargs.get('learning_rate_divider', 1.)

            def func(t):
                return rate_0/float(divider+t)
        elif rate_type == 'custom':
            func = kwargs.get('learning_rate_custom', None)
            if func is None:
                raise ValueError("If 'rate_type' is 'custom', a rate",
                                 "function must be defined in",
                                 "'learning_rate_custom'.")
        return func

    # Number of parameters
    @abstractmethod
    def iterate(self, t, x, grad):
        pass
