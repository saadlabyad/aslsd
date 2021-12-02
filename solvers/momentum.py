# License: BSD 3 clause

from aslsd.solvers.solver import Solver


class Momentum(Solver):
    """
    Class for a momentum SGD optimizer.

    Parameters
    ----------

    Notes
    ------
        Qian, Ning. "On the momentum term in gradient descent learning
        algorithms." Neural networks 12.1 (1999): 145-151.
    """

    def __init__(self, **kwargs):
        self.momentum = kwargs.get('momentum', default=0.9)
        self.momentum_term = 0.

    def iterate(self, t, x, grad):
        delta_vt = self.momentum_term
        rate = self.learning_rate(t)
        delta_vt = -rate*grad+self.momentum*delta_vt
        self.momentum_term = delta_vt
        return x + delta_vt
