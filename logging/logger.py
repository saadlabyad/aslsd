# License: BSD 3 clause

import itertools

import numpy as np


class Logger:
    def __init__(self, d, n_iter, **kwargs):
        self.d = d
        self.n_iter = n_iter
        self.is_log_param = kwargs.get('is_log_param', False)
        self.is_log_grad = kwargs.get('is_log_grad', False)
        if self.is_log_param:
            self.param_logs = [[None for x in range(n_iter[k]+1)]
                               for k in range(d)]
            self.mu = [np.zeros(n_iter[k]+1) for k in range(d)]
            self.ker = [[[None for x in range(n_iter[i]+1)] for j in range(d)]
                        for i in range(d)]
        if self.is_log_grad:
            self.grad_logs = [[None for x in range(n_iter[k])]
                              for k in range(d)]
            self.grad_mu = [[None for x in range(n_iter[k])]
                            for k in range(d)]
            self.grad_ker = [[[None for x in range(n_iter[i])]
                              for j in range(d)] for i in range(d)]
        self.estimator_logs = [None]*d
        self.mu_0 = None
        self.ker_0 = None

    def log_param(self, k, t, x_k):
        if self.is_log_param:
            self.param_logs[k][t] = x_k

    def log_grad(self, k, t, g_t):
        if self.is_log_grad:
            self.grad_logs[k][t] = g_t

    def process_logs(self, mhp):
        d = self.d
        if self.is_log_param:
            for i in range(d):
                for ix in range(self.n_iter[i]+1):
                    self.mu[i][ix] = self.param_logs[i][ix][0]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(self.n_iter[i]+1):
                    self.ker[i][j][ix] = self.param_logs[i][ix][mhp.interval_map[i][j][0]:mhp.interval_map[i][j][1]]
        if self.is_log_grad:
            for i in range(d):
                for ix in range(self.n_iter[i]):
                    self.grad_mu[i][ix] = self.grad_logs[i][ix][0]
            for i, j in itertools.product(range(d), range(d)):
                for ix in range(self.n_iter[i]):
                    self.grad_ker[i][j][ix] = self.grad_logs[i][ix][mhp.interval_map[i][j][0]:mhp.interval_map[i][j][1]]
