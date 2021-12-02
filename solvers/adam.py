# License: BSD 3 clause

import numpy as np

from aslsd.solvers.solver import Solver


class ADAM(Solver):
    """
    Class for ADAM optimizer.

    Parameters
    ----------

    Notes
    ------
        Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic
        optimization. In International Conference on Learning Representations,
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
