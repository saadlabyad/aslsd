aslsd
#####

Parametric estimation of multivariate Hawkes processes with general kernels.

This code is under active development, and is part of our paper `Gradient-based estimation of linear Hawkes processes with general kernels. <https://arxiv.org/abs/2111.10637>`_

Description
-----------

Multivariate Hawkes processes (MHP) are a fundamental class of point processes
with self-excitation. When estimating parameters for these processes, a
difficulty is that the two main error functionals, the log-likelihood and the least
squares error (LSE), as well as the evaluation of their gradients, have a quadratic
complexity in the number of observed events. In practice, this prohibits the use
of exact gradient-based algorithms for parameter estimation in many settings.
Furthermore, MHP models are not designed for non-stationary training data,
and they cannot incorporate event information besides their timestamps: we introduce
the marked time-dependent linear Hawkes (MTLH) model to overcome
these limitations. 

We construct an adaptive stratified sampling estimator of the
gradient of the LSE of Hawkes models. This results in the ASLSD algorithm, a
fast parametric estimation method for MHP and MTLH with general kernels,
applicable to large datasets, which compares favourably with existing methods.
We evaluate our algorithm on synthetic and real-world data.

This code implements

* the ASLSD algorithm for the estimation of MHP and MTLH models;
* exact cluster based simulation of MHP and MTLH models;
* residual analysis and other evaluation metrics for MHP and MTLH models.

Dependencies
------------

This code requires Python 3.7 or newer, as well as:

* `Numpy 1.19 <https://numpy.org/install/>`_
* `Scipy 1.6 <https://scipy.org/install/>`_
* `Pandas 1.2 <https://pandas.pydata.org/docs/getting_started/install.html>`_
* `tqdm 4.59 <https://github.com/tqdm/tqdm#installation>`_

Citation
------------

This code is part of our paper `Gradient-based estimation of linear Hawkes processes with general kernels. <https://arxiv.org/abs/2111.10637>`_

If you use this code as part of a scientific publication, please acknowledge our paper. ::

   @article{cartea2021gradient,
     title={Gradient-based estimation of linear Hawkes processes with general kernels},
     author={Cartea, {\'A}lvaro and Cohen, Samuel N and Labyad, Saad},
     journal={arXiv preprint arXiv:2111.10637},
     year={2021}
   }
