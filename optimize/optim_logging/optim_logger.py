# License: BSD 3 clause

import itertools

import numpy as np


class OptimLogger:
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

    **kwargs : `dict`
        Reference to dictionary of keyword arguments.


    """

    def __init__(self, d, n_iter, **kwargs):
        self.d = d
        self.n_iter = n_iter
        self.is_log_param = kwargs.get('is_log_param', False)
        self.is_log_grad = kwargs.get('is_log_grad', False)
        self.is_log_lse = kwargs.get('is_log_lse', False)
        self.is_log_ixs = kwargs.get('is_log_ixs', False)
        self.is_log_allocs = kwargs.get('is_log_allocs', False)
        self.estimator_logs = [None]*d
        if self.is_log_lse:
            self.lse = [None for ix in range(d)]
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

    def log_lse(self, k, t, lse_t):
        if self.is_log_lse:
            self.lse_k[k][t] = lse_t
