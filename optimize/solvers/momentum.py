# License: BSD 3 clause

from aslsd.optimize.solvers.solver import Solver


class Momentum(Solver):
    """
    Class for a momentum SGD optimizer.

    This solver performs parameter updates of the form

    .. math::
        \\theta_{k}^{(t+1)}=\\mathrm{proj}_{\\Theta_{k}}\\left(\\theta_{k}^{(t)}+\\Delta \\theta_{k}^{(t+1)}\\right),

    with

    .. math::
        \\Delta \\theta_{k}^{(t)} =-a_{\\mathrm{rate}}(t) \\cdot \\mathcal{G}_{\\boldsymbol{T}}^{(\\boldsymbol{k})}\\left(\\theta_{k}^{(t)}\\right)+a_{\\mathrm{M}} \\cdot \\Delta \\theta_{k}^{(t-1)} .

    The operator :math:`\\mathrm{proj}_{\\Theta_{k}}` denotes the projection
    operator on the set of parameters :math:`\\Theta_{k}`. The hyper-parameters
    of this solver are

    * :math:`a_{\\mathrm{M}}` is the gradient momentum term;
    * :math:`a_{\\mathrm{rate}}` is the time-dependent learning rate.

    Parameters
    ----------

    Notes
    ------
        Qian, Ning. 'On the momentum term in gradient descent learning
        algorithms'. Neural networks 12.1 (1999): 145-151.
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
