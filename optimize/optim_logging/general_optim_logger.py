# License: BSD 3 clause

import itertools

import numpy as np


class GeneralOptimLogger:
    """
    Class for optimization logger.

    Attributes
    ----------
    d : `int`
        Dimension of the MHP.
    n_iter : `list` of `int`
        DESCRIPTION.

    is_log_param : `bool`
        DESCRIPTION.
    is_log_grad : `bool`
        DESCRIPTION.
    param_logs : `numpy.ndarray`
        DESCRIPTION.
    mu : `numpy.ndarray`
        DESCRIPTION.
    ker : `numpy.ndarray`
        DESCRIPTION.
    grad_logs : `numpy.ndarray`
        DESCRIPTION.
    grad_mu : `numpy.ndarray`
        DESCRIPTION.
    grad_ker : `numpy.ndarray`
        DESCRIPTION.
    estimator_logs : TYPE
        DESCRIPTION.
    mu_0 : `numpy.ndarray`
        DESCRIPTION.
    ker_0 : `numpy.ndarray`
        DESCRIPTION.

    **kwargs : TYPE
        DESCRIPTION.


    """

    def __init__(self, d, n_iter, **kwargs):
        self.d = d
        self.n_iter = n_iter
        self.is_log_param = kwargs.get('is_log_param', False)
        self.is_log_grad = kwargs.get('is_log_grad', False)
        self.is_log_loss = kwargs.get('is_log_loss', False)
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
        self.is_log_allocs = kwargs.get('is_log_allocs', False)
        if self.is_log_param:
            self.param_logs = [[None for x in range(n_iter[k]+1)]
                               for k in range(d)]
        if self.is_log_grad:
            self.grad_logs = [[None for x in range(n_iter[k])]
                              for k in range(d)]
        self.estimator_logs = [None]*d
        self.x_0 = None
        if self.is_log_loss:
            self.loss = [None for ix in range(d)]
        if self.is_log_ixs:
            self.samples = [None for ix in range(d)]
        if self.is_log_allocs:
            self.allocs = [None for ix in range(d)]

    def log_param(self, k, t, x_k):
        if self.is_log_param:
            self.param_logs[k][t] = x_k

    def log_grad(self, k, t, g_t):
        if self.is_log_grad:
            self.grad_logs[k][t] = g_t

    def process_logs(self, mhp):
        d = self.d
        for k in range(d):
            if self.is_log_grad:
                self.grad_logs[k] = np.array(self.grad_logs[k])
            if self.is_log_param:
                self.param_logs[k] = np.array(self.param_logs[k])
        # if self.is_log_loss:
        #     for k in range(d):
        #         self.loss[k] = self.estimator_logs[k]['loss']
        # if self.is_log_ixs:
        #     for k in range(d):
        #         self.samples[k] = {}
        #         self.samples[k]['psi'] = self.estimator_logs[k]['samples']['psi']
        #         self.samples[k]['upsilonzero'] = self.estimator_logs[k]['samples']['upsilonzero']
        #         self.samples[k]['phi'] = self.estimator_logs[k]['samples']['phi']
        #         self.samples[k]['upsilon'] = self.estimator_logs[k]['samples']['upsilon']
        # if self.is_log_allocs:
        #     for k in range(d):
        #         self.allocs[k] = {}
        #         self.allocs[k]['phi'] = self.estimator_logs[k]['allocs']['phi']
        #         self.allocs[k]['upsilon'] = self.estimator_logs[k]['allocs']['upsilon']
