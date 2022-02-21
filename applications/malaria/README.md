# Malaria
Reproduce the results from the epidemic propagation application in the "Applications" section of our paper, [Gradient-based estimation of linear Hawkes processes with general kernels](https://arxiv.org/abs/2111.10637).

## Description
In this application, we model the propagation of malaria in China.  

## Data
We use the data of Unwin, Routledge, Flaxman, Rizoiu, Lai, Cohen, Weiss, Mishra, and Bhatt (In their paper [Using Hawkes processes to model imported and local malaria cases in near-elimination settings](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008830)) for the propagation of malaria in the Yunan province between 1 January 2011
and 24 September 2013. We do not host this data in the aslsd Github repository for license purposes. This data is publicly available to download from the Harvard Dataverse, in the publication: [Replication Data for: Using Hawkes Processes to model imported and local malaria cases in near-elimination settings](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YPRLIL).
The only file we use is
```
china_malaria.RData
```

which can be downloaded [using this link](https://dataverse.harvard.edu/api/access/datafile/4443458).
Once the user has downloaded this file on their machine, they need to convert it into a `csv` file. This can be done for example by using R, or the Python package `pyreadr` available [here](https://pypi.org/project/pyreadr/).
The user can then run 
```
malaria.py
```
which is located in this folder of the `aslsd` package, by specifying the variable `data_filepath`, which is the path to the downloaded data in `csv` format.


## Models
Unwin et al. use a slightly modified univariate linear Hawkes model with a delayed Rayleigh kernel to study the transmission in this context. To fit their
model, given the typically small number of observations in the applications they consider, they compute exactly the log-likelihood of their observations and input it to a standard optimization
solver.

In our paper, we fit two models using `aslsd`: 
* `SbfGauss1D10R` is an SBF Gaussian model with ten Gaussians
(with uniformly spaced means in [0, 20] and standard deviations equal to 1.9),
* `Gauss1D1R` is a non-SBF Gaussian model. 

In addition to this, we consider two benchmarks: 
* `Poisson` is a naive homogeneous Poisson model;
* `SumExp` is an SBF exponential MHP model, fitted using the algorithm of Martin Bompaire, Emmanuel Bacry, and Stéphane Gaïffas, in [Dual optimization for convex constrained objectives without the gradient-Lipschitz assumption](https://arxiv.org/abs/1807.03545#:~:text=Dual%20optimization%20for%20convex%20constrained%20objectives%20without%20the%20gradient%2DLipschitz%20assumption,-Martin%20Bompaire%2C%20Emmanuel&text=The%20minimization%20of%20convex%20objectives,finite%20sums%20of%20convex%20functions.). 

For `SumExp`, we use the implementation of this algorithm in the Python library `tick`, which can be installed [here](https://github.com/X-DataInitiative/tick).