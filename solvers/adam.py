# License: BSD 3 clause

import numpy as np

from aslsd.solvers.solver import Solver


class ADAM(Solver):
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
        self.learning_rate = self.make_learning_rate(**kwargs)
        self.momentum_1 = kwargs.get('momentum_1', 0.9)
        self.momentum_term_1 = 0.
        self.momentum_2 = kwargs.get('momentum_2', 0.999)
        self.momentum_term_2 = 0.
        self.epsilon = kwargs.get('epsilon', 10**-8)

    def iterate(self, t, x, grad):
        m_t = self.momentum_term_1
        m_t_new = self.momentum_1*m_t+(1.-self.momentum_1)*grad
        self.momentum_term_1 = m_t_new

        v_t = self.momentum_term_2
        v_t_new = self.momentum_2*v_t+(1.-self.momentum_2)*np.power(grad, 2)
        self.momentum_term_2 = v_t_new

        m_hat_t = m_t_new/(1.-np.power(self.momentum_1, t+1))
        v_hat_t = v_t_new/(1.-np.power(self.momentum_2, t+1))

        rate = self.learning_rate(t)

        return x-rate*m_hat_t/(np.sqrt(v_hat_t)+self.epsilon)
