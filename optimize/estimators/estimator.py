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

    def initialize_model_data(self, model, data):
        # Data
        self.list_times = data.list_times
        self.list_marks = data.list_marks
        self.T_f = data.T_f
        self.d = data.d
        self.n_events = data.n_events
        self.eta = data.eta
        self.list_times2end = data.list_times2end
        self.kappa = data.kappa
        self.varpi = data.varpi
        self.lag_sizes = data.lag_sizes

        # Model
        k = self.k
        model.init_estimator(self, k)

    @abstractmethod
    def get_log(self):
        pass

    @abstractmethod
    def lse_k_grad_estimate(self, x_k, rng=None, seed=None):
        pass

    @abstractmethod
    def lse_k_estimate(self, x_k, rng=None, seed=None):
        pass
