# License: BSD 3 clause

from abc import ABC, abstractmethod

import numpy as np

import aslsd.utilities.useful_functions as uf


class PointModel(ABC):
    def __init__(self, is_fitted=False):
        self.is_fitted = False

    # Parameters operations
    @abstractmethod
    def xk2matrix_params(self, k, x_k):
        pass

    @abstractmethod
    def matrix2tensor_params(self, *args):
        pass

    @abstractmethod
    def tensor2matrix_solverpaths(self, tensor_paths):
        pass

    @abstractmethod
    def make_xk(self, k, **kwargs):
        pass

    @abstractmethod
    def load_param(self, **kwargs):
        pass

    # Estimator functions
    @abstractmethod
    def init_estimator(self, estimator, k):
        pass

    # Logging
    @abstractmethod
    def init_logger(self, logger):
        pass

    @abstractmethod
    def process_logs(self, logger):
        pass

    # Fit
    @abstractmethod
    def get_random_param(self, flatten=False, rng=None, seed=1234, **kwargs):
        pass

    @abstractmethod
    def clear_fit(self):
        pass

    @abstractmethod
    def fit(self, process_path=None, list_process_paths=None, x_0=None,
            n_iter=1000, tol=10**-6, rng=None, seed=None, estimators=None,
            solvers=None,
            logger=None,
            verbose=False, clear=True, write=True, **kwargs):
        pass

    # Residuals
    def get_residuals(self, process_path=None, sampling=False, sample_size=10**3,
                      seed=1234, write=True, verbose=False, **kwargs):
        pass

    def ks_test_residuals(self, residuals=None):
        pass

    def qq_plot(self, i, residuals=None, labels=None, style='exponential',
                substract_yx=False, normalize=False, max_points=None,
                display_line45=True, log_scale=False, ax=None, save=False,
                filename='image.png', show=False, **kwargs):
        pass

    # Simulation
    def simulate(self, T_f, mu_param=None, kernel_param=None,
                 impact_param=None, rng=None, seed=1234,
                 verbose=False):
        pass

    def simu_multipath(self, path_res, t_res, x_min, x_max, mu=None,
                       kernel_param=None, seed=1234, verbose=False,
                       disc_type='log', base_seed=1234):
        pass

    # Serialization
    def save(self, file, **kwargs):
        pass

    def load(self, file, **kwargs):
        pass
