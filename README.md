# aslsd
Parametric estimation of multivariate Hawkes processes with general kernels.

This code is under active development, and is part of our paper [Gradient-based estimation of linear Hawkes processes with general kernels.](https://arxiv.org/abs/2111.10637)

## Description
Linear multivariate Hawkes processes (MHP) are a fundamental class of point processes with self-excitation. When estimating parameters for these processes, a difficulty is that the two main error functionals, the log-likelihood and the least squares error (LSE), as well as the evaluation of their
gradients, have a quadratic complexity in the number of observed events. In practice, this prohibits
the use of exact gradient-based algorithms for parameter estimation. 

We propose a stochastic optimization algorithm for parametric MHP estimation that does not directly evaluate the conditional intensity of the MHP; we
call this the ASLSD algorithm (Adaptively Stratified Least Squares Descent). This results in a fast parametric estimation
method for MHP with general kernels, applicable to large datasets, which compares favourably with
existing methods.

This code implements
* the ASLSD algorithm for the estimation of MHP;
* exact cluster based simulation of MHP;
* goodness-of-fit tests.

## Dependencies
This code requires Python 3.7 or newer, as well as:
* [Numpy 1.19](https://numpy.org/install/)
* [Scipy 1.6](https://scipy.org/install/)
* [Pandas 1.2](https://pandas.pydata.org/docs/getting_started/install.html)
* [tqdm 4.59](https://github.com/tqdm/tqdm#installation)

## Citation
This code is part of our paper

[Gradient-based estimation of linear Hawkes processes with general kernels.](https://arxiv.org/abs/2111.10637)

If you use this code as part of a scientific publication, please acknowledge our paper. You can use the bibtex entry below.
```
@article{cartea2021gradient,
  title={Gradient-based estimation of linear Hawkes processes with general kernels},
  author={Cartea, {\'A}lvaro and Cohen, Samuel N and Labyad, Saad},
  journal={arXiv preprint arXiv:2111.10637},
  year={2021}
}
```
