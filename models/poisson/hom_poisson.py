# License: BSD 3 clause

import itertools

import numpy as np

from aslsd.stats.residual_analysis import goodness_of_fit as gof


class HomPoisson:
    """
    Class for homogeneous Poisson models of point processes.

    Let :math:`\\mathbf{N}` be a d-dimensional counting process with
    conditional intensity :math:`\\boldsymbol{\\lambda}`.
    We say that :math:`\\mathbf{N}` is a homogeneous Poisson process if there
    exists positive constants :math:`(\\mu_{i})_{i \\in[d]}` (that we call
    baselines), such that for all :math:`i \\in[d]` and for all
    :math:`t \\geq 0`, we have

    .. math::
        \\lambda_{i}(t):=\\mu_{i}.


    Attributes
    ----------
    d : `int`
        Dimension of the homogeneous Poisson process.
    index_from_one : `bool`, optional
        Start the indexing of baselines from 1 instead of 0. The default is False.
    mu_names : `list` of `str`, optional
        List of names of baseline parameters. The default is None.
    is_fitted : `bool`, optional
        True if this model has been fitted. The default is False.
    fitted_mu : `numpy.ndarray`, optional
        Array of fitted baselines. The default is None.
    fit_residuals : `list` of `numpy.ndarray`, optional
        List of fit residuals. The default is None.

    """

    def __init__(self, d, index_from_one=False, mu_names=None,
                 is_fitted=False, fitted_mu=None, fit_residuals=None):
        self.d = d
        if mu_names is None:
            self.mu_names = self.get_param_names(index_from_one=index_from_one)

    # Param names
    def get_param_names(self, index_from_one=False):
        """
        Get the standard names of baseline variables.

        By default, we denote the baselines by :math:`(\\mu_{i})_{i \\in[d]}`.

        Parameters
        ----------
        index_from_one : `bool`, optional
            Start the indexing of baselines from 1 instead of 0. The default is False.

        Returns
        -------
        mu_names : `list` of `str`
            List of names of baseline parameters.

        """
        d = self.d
        mu_names = ['$\u03BC_{'+str(i+int(index_from_one))+'}$'
                    for i in range(d)]
        return mu_names

    def clear_fit(self):
        self.is_fitted = False
        self.fitted_mu = None
        self.fitted_ker_param = None
        self.fit_residuals = None
        self.fitted_adjacency = None
        self.fit_log = None

    def fit(self, list_times, T_f, clear=True, write=True):
        """
        Fit the Homoegeneous Poisson model to some observations.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The least squares error (LSE) of this model for these observations is
        defined as

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\frac{1}{T} \\sum_{k=1}^{d} \\int_{0}^{T} \\lambda_{k}(t)^{2} \\mathrm{~d} t-\\frac{2}{T} \\sum_{k=1}^{d} \\sum_{m=1}^{N_{T}^{k}} \\lambda_{k}\\left(t_{m}^{k}\\right).

        For a homogeneous Poisson model, this simplifies to

        .. math::
            \\mathcal{R}_{T}(\\boldsymbol{\\mu}):=\\sum_{k=1}^{d} \\bigg( \\mu_{k}^{2} -2 \\frac{N_{T}^{k}}{T} \\bigg).

        This expression is minimized if for all :math:`k \\in [d]` we have

        .. math::
            \\mu_k^{*}:=\\frac{N_{T}^{k}}{T}.

        The optimal baseline vector :math:`\\boldsymbol{\\mu^{*}}` defined
        above is also the maximizer of the loglikelihood of these observations.

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times for each dimension.
        T_f : `float`
            Terminal time.
        clear : `bool`, optional
            If true, delete all previously saved results and logs from the
            corresponding attributes of the MHP object. The default is True.
        write : `bool`, optional
            If true, save the estimation results and logs in the corresponding
            attributes of the MHP object. The default is True.

        Returns
        -------
        fitted_mu : `numpy.ndarray`
            Fitted baselines.

        """
        if clear:
            self.clear_fit()
        fitted_mu = np.array([len(L) for L in list_times])/T_f
        if write:
            self.is_fitted = True
            self.fitted_mu = fitted_mu
        return fitted_mu

    # Simulation
    def simulate(self, T_f, mu=None, seed=1234):
        """
        Simulate a path of the homogeneous Poisson model.

        Parameters
        ----------
        T_f : `float`
            Terminal time.
        mu : `numpy.ndarray`, optional
            Vector of baselines. If None, use the fitted baselines. The default
            is None.
        seed : `int`, optional
            Seed of the random generator. The default is 1234.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        list_times : `list` of `numpy.ndarray`
            List of jump times per dimension.

        """
        if mu is None:
            mu = self.fitted_mu
            if mu is None:
                raise ValueError("Missing value for Mu")
        d = self.d
        list_times = [None]*d
        rng = np.random.default_rng(seed)
        for i in range(d):
            # Number of immigrants
            Nim = rng.poisson(mu[i]*T_f)
            generations = rng.uniform(low=0., high=T_f, size=Nim)
            generations.sort()
            list_times[i] = generations
        return list_times

    # Evaluation
    def get_residuals(self,  list_times, mu=None, write=True):
        """
        Compute the residuals of the model.

        We suppose that we observe a path of a d-dimensional counting process
        :math:`\\mathbf{N}` started at time :math:`0` up to some terminal time
        :math:`T`.

        The residuals of a homogeneous Poisson model :math:`\\boldsymbol{\\mu}`
        are defined by

        .. math::
            \\left\\{\\mu_k(t^k_{m+1}-t^k_{m}): k \\in [d], m \\in [N^k_T-1] \\right\\}.

        Parameters
        ----------
        list_times : `list` of `numpy.ndarray`
            List of jump times per dimension.
        mu : `numpy.ndarray`, optional
            Vector of baselines. If None, use the fitted baselines. The default
            is None.
        write : `bool`, optional
            Save the results. The default is True.

        Raises
        ------
        ValueError
            Raise an error if the baseline is not specified and there is no
            fitted baseline saved as an atrribute.

        Returns
        -------
        residuals : `list` of `numpy.ndarray`
            List of residuals per dimension.

        """
        if mu is None:
            if self.is_fitted:
                mu = self.fitted_mu
            else:
                raise ValueError("Mu must be specified.")

        d = self.d
        ia_times = [list_times[i][1:]-list_times[i][:-1] for i in range(d)]
        residuals = [mu[i]*ia_times[i] for i in range(d)]
        if self.is_fitted and write:
            self.fit_residuals = residuals
        return residuals

    def qq_plot(self, i, residuals=None, labels=None, style='exponential',
                substract_yx=False, normalize=False, max_points=None,
                display_line45=True, log_scale=False,
                ax=None, save=False, filename='image.png', show=False,
                **kwargs):
        if residuals is None:
            if self.fit_residuals is not None:
                residuals = self.fit_residuals
            else:
                raise ValueError("residuals must be specified.")
        return gof.qq_plot(residuals[i], n_models=1, labels=labels,
                           style=style, substract_yx=substract_yx,
                           normalize=normalize, max_points=max_points,
                           display_line45=display_line45, log_scale=log_scale,
                           ax=ax, save=save,
                           filename=filename, show=show, **kwargs)
