# Royal wedding
Reproduce the results from the news propagation application in the "Applications" section of our paper, [Gradient-based estimation of linear Hawkes processes with general kernels](https://arxiv.org/abs/2111.10637).
In this application, we are interested in the diffusion of information across different media platforms.

## Data
We use the data of  Gomez Rodriguez, Leskovec, and Sch¨olkopf (in their paper [Structure and Dynamics of Information Pathways in Online Media](https://dl.acm.org/doi/10.1145/2433396.2433402)). The authors compiled news articles from several websites that
mention a selection of keywords into the MemeTracker dataset. There exist different versions of the
MemeTracker dataset, with different data and different structures. We are interested in the one proposed
by Gomez Rodriguez et al. in the Stanford Network Analysis Project. We do not host this data in the `aslsd` Github repository for license purposes. This data is publicly available to download using [this link](http://snap.stanford.edu/infopath/data.html).
The user should start by downloading the file
```
memes-w5-all-2011-03-2012-02-n5000-call-nc10-cl2-cascades-keywords.tgz
```


Once the user has downloaded this file on their machine, they need to need to extract the following file from this archive:

```
memes-w5-all-2011-03-2012-02-n5000-call-nc10-cl2-keywords-in-body-prince william-william-kate middleton-kate-middleton-westminster-watch-marriage-queen-king-elizabeth-charles-cascades.txt
```

The user can then run 
```
memetracker_royalwedding_1d.py
```
or
```
memetracker_royalwedding_2d.py
```
which is located in this folder of the `aslsd` package, by specifying the variable `data_filepath` (this variable is a string representing the path to the downloaded data).


## Univariate models
Unwin et al. use a slightly modified univariate linear Hawkes model with a delayed Rayleigh kernel to study the transmission in this context. To fit their
model, given the typically small number of observations in the applications they consider, they compute exactly the log-likelihood of their observations and input it to a standard optimization
solver.

In our paper, we fit two models to this data using `aslsd`: 
* `SbfGauss1D10R` is an SBF Gaussian model with ten Gaussians
(with uniformly spaced means in [0, 20] and standard deviations equal to 1.9),
* `Gauss1D1R` is a non-SBF Gaussian model. 

In addition to this, we consider two benchmarks: 
* `Poisson` is a naive homogeneous Poisson model;
* `SumExp` is an SBF exponential MHP model, fitted using the algorithm of Martin Bompaire, Emmanuel Bacry, and Stéphane Gaïffas, in [Dual optimization for convex constrained objectives without the gradient-Lipschitz assumption](https://arxiv.org/abs/1807.03545#:~:text=Dual%20optimization%20for%20convex%20constrained%20objectives%20without%20the%20gradient%2DLipschitz%20assumption,-Martin%20Bompaire%2C%20Emmanuel&text=The%20minimization%20of%20convex%20objectives,finite%20sums%20of%20convex%20functions.). 

For `SumExp`, we use the implementation of this algorithm in the Python library `tick`, which can be installed [here](https://github.com/X-DataInitiative/tick).


## Bidimensional models