# Malaria
Reproduce the results from malaria application in the "Applications" section of our paper, [Gradient-based estimation of linear Hawkes processes with general kernels](https://arxiv.org/abs/2111.10637).

## Description
In this application, we model the propagation of Malaria in China.  

## Data
We use the data of Unwin, Routledge, Flaxman, Rizoiu, Lai, Cohen, Weiss, Mishra, and Bhatt for the propagation of malaria in the Yunan province between 1 January 2011
and 24 September 2013. We do not host this data in the aslsd Github repository for license purposes. This data is publicly available to download from the Harvard Dataverse, in the publication: [Replication Data for: Using Hawkes Processes to model imported and local malaria cases in near-elimination settings](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YPRLIL).
The only file we use is
```
china_malaria.RData
```

which can be downloaded [using this link](https://dataverse.harvard.edu/api/access/datafile/4443458).
Once the user has downloaded this file on their machine, they need to run 
```
malaria.py
```

by specifying the variable rdata_filepath, which is the path to 

```
china_malaria.RData
```


## Models
In their paper [Using Hawkes processes to model imported and local malaria cases in near-elimination settings](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008830),
Unwin et al. use a slightly modified univariate linear Hawkes model with a delayed Rayleigh kernel to study the transmission in this context. To fit their
model, given the typically small number of observations in the applications they consider, they compute exactly the log-likelihood of their observations and input it to a standard optimization
solver.

In our paper, we consider.
The benchmarks used are WH and SumExp.