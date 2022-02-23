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


Once the user has downloaded this archive on their machine, they need to extract the following file:

```
memes-w5-all-2011-03-2012-02-n5000-call-nc10-cl2-keywords-in-body-prince william-william-kate middleton-kate-middleton-westminster-watch-marriage-queen-king-elizabeth-charles-cascades.txt
```

We model mentions of the keyword related to the British Royal family, a few months after the wedding
of Prince William and Catherine Middleton on 29 April 2011. We limit ourselves to data between 1
November 2011 at midnight UTC and 1 March 2012 at midnight UTC. The data is timestamped in Unix
time in hours with a second resolution. When two or more posts have the same timestamp, we only keep
the event that appears first in the dataset.

To reproduce our results, the user can run 
```
memetracker_royalwedding_1d.py
```
or
```
memetracker_royalwedding_2d.py
```
which is located in this folder of the `aslsd` package, by specifying the variable `data_filepath` (this variable is a string representing the path to the downloaded data).


## Univariate models
First, we aggregate all publication timestamps into a path of a one-dimensional point process.

In our paper, we fit two models to this data using `aslsd`: 
* `Exp1D1R` is a non-SBF exponential model;
* `Exp1D6R` is a non-SBF exponential model with 6 exponentials. 

In addition to this, we consider two benchmarks: 
* `SumExp` is an SBF exponential MHP model, fitted using the algorithm of Martin Bompaire, Emmanuel Bacry, and Stéphane Gaïffas, in [Dual optimization for convex constrained objectives without the gradient-Lipschitz assumption](https://arxiv.org/abs/1807.03545#:~:text=Dual%20optimization%20for%20convex%20constrained%20objectives%20without%20the%20gradient%2DLipschitz%20assumption,-Martin%20Bompaire%2C%20Emmanuel&text=The%20minimization%20of%20convex%20objectives,finite%20sums%20of%20convex%20functions.). 
* `WH` is  non-parametric estimation method which solves a Wiener–Hopf system derived from the autocovariance of the MHP, fitted using the algorithm of  Emmanuel Bacry and Jean-Fran¸cois Muzy, in [Dual optimization for convex constrained objectives without the gradient-Lipschitz assumption](https://ieeexplore.ieee.org/document/7416001?arnumber=7416001). 

For `SumExp` and `WH`, we use the implementations of these algorithms in the Python library `tick`, which can be installed [here](https://github.com/X-DataInitiative/tick).


## Bidimensional models
Diffusion dynamics of news related to the British Royal family
might significantly differ between British news outlets, North American and Australian media, and those
from other nationalities. In the 5, 000 news websites that appear in the MemeTracker dataset, it is
not sufficient to use the top-level domain of the website to deduce its country. We manually verify the nationality of the media sources; the list of media nationalities is available as a csv file in the Applications
folder of our repository. We model publication times of US and UK articles related to this keyword as
a bi-dimensional MHP (dimension i = 1 corresponds to US articles and i = 2 to UK articles).

In our paper, we fit three models to this data using `aslsd`: 
* `Exp2D1R` is a non-SBF exponential model;
* `Exp2D3R` is a non-SBF exponential model with 3 exponentials;
* `Gauss2D1R` is a non-SBF Gaussian model. 

In addition to this, we consider the two usual benchmarks, `SumExp` and `WH`.
