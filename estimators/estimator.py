# License: BSD 3 clause

from abc import ABC, abstractmethod


class Estimator(ABC):
    def __init__(self, **kwargs):
        self.k = None
        self.n_iter = None
        self.list_times = None
        self.T_f = None
        self.d = None,
        self.n_events = None
        self.eta = None
        self.list_times2end = None
        self.kappa = None,
        self.varpi = None
        self.lag_sizes = None,
        self.n_param_k = None
        self.ix_map = None
        self.interval_map = None,
        self.phi = None
        self.diff_phi = None
        self.psi = None
        self.diff_psi = None,
        self.upsilon = None
        self.diff_sim_upsilon = None
        self.diff_cross_upsilon = None

    def initialize(self, process_path, hawkes_model):
        # Data
        self.list_times = process_path.list_times
        self.list_marks = process_path.list_marks
        self.T_f = process_path.T_f
        self.d = process_path.d
        self.n_events = process_path.n_events
        self.eta = process_path.eta
        self.list_times2end = process_path.list_times2end
        self.kappa = process_path.kappa
        self.varpi = process_path.varpi
        self.lag_sizes = process_path.lag_sizes

        # Model
        k = self.k
        hawkes_model.init_estimator(self, k)

    @abstractmethod
    def get_log(self):
        pass

    @abstractmethod
    def lse_k_grad_estimate(self, x_k, rng=None, seed=None):
        pass

    @abstractmethod
    def lse_k_estimate(self, x_k, rng=None, seed=None):
        pass
