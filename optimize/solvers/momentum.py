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
        self.momentum_1 = kwargs.get('momentum_1', 0.9)
        self.momentum_term_1 = 0.
        Solver.__init__(self, **kwargs)

    def iterate(self, t, x, grad):
        g_t = self.clip(grad)
        delta_vt = self.momentum_term_1
        rate = self.learning_rate(t)
        delta_vt = -rate*g_t+self.momentum_1*delta_vt
        self.momentum_term_1 = delta_vt
        self.t += 1
        return x + delta_vt
