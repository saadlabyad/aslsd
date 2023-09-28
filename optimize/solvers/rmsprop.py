# License: BSD 3 clause

import numpy as np

from aslsd.optimize.solvers.solver import Solver


class RMSprop(Solver):
    """
    Class for ADAM optimizer.

    This solver performs parameter updates of the form

    .. math::
        \\theta_{k}^{(t+1)}=\\mathrm{proj}_{\\Theta_{k}}\\left(\\theta_{k}^{(t)}+\\Delta \\theta_{k}^{(t+1)}\\right),

    with

    .. math::
        g_{1}^{(t)} & =\\frac{a_{\\mathrm{M} 1} \\cdot g_{1}^{(t-1)}+\\left(1-a_{\\mathrm{M} 1}\\right) \\cdot \\mathcal{G}_{\\boldsymbol{T}}^{(\\boldsymbol{k})}\\left(\\theta_{k}^{(t)}\\right)}{1-a_{\\mathrm{M} 1}^{t}}; \\\\
        g_{2}^{(t)} & =\\frac{a_{\\mathrm{M} 2} \\cdot g_{2}^{(t-1)}+\\left(1-a_{\\mathrm{M} 2}\\right) \\cdot \\mathcal{G}_{\\boldsymbol{T}}^{(\\boldsymbol{k})} \\odot \\mathcal{G}_{\\boldsymbol{T}}^{(\\boldsymbol{k})}\\left(\\theta_{k}^{(t)}\\right)}{1-a_{\\mathrm{M} 2}^{t}}; \\\\
        \\Delta \\theta_{k}^{(t)} & =-a_{\\mathrm{rate}}(t) \\cdot \\frac{g_{1}^{(t)}}{\\sqrt{g_{2}^{(t)}}+a_{\\mathrm{E}}}.

    The operator :math:`\\mathrm{proj}_{\\Theta_{k}}` denotes the projection
    operator on the set of parameters :math:`\\Theta_{k}`. The hyper-parameters
    of this solver are

    * :math:`a_{\\mathrm{M} 1}` is the gradient momentum term;
    * :math:`a_{\\mathrm{M} 2}` is the squared gradient momentum term;
    * :math:`a_{\\mathrm{E}}` is a very small hyper-parameter to avoid division by zero;
    * :math:`a_{\\mathrm{rate}}` is the time-dependent learning rate.

    Parameters
    ----------

    Notes
    ------
        Diederik P Kingma and Jimmy Ba. 'Adam: A method for stochastic
        optimization'. In International Conference on Learning Representations,
        2015.
    """

    def __init__(self, **kwargs):
        self.momentum_2 = kwargs.get('momentum_2', 0.999)
        self.momentum_term_2 = 0.
        self.epsilon = kwargs.get('epsilon', 10**-8)
        Solver.__init__(self, **kwargs)

    def iterate(self, t, x, grad):
        g_t = self.clip(grad)

        v_t = self.momentum_term_2
        v_t_new = self.momentum_2*v_t+(1.-self.momentum_2)*g_t**2
        self.momentum_term_2 = v_t_new

        rate = self.learning_rate(t)

        self.t += 1

        return x-rate*g_t/(np.sqrt(v_t_new)+self.epsilon)
