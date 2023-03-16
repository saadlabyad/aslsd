# License: BSD 3 clause

from aslsd.functionals.kernels.basis_kernels.\
    preloaded_mc import (mc_exponential,
                         mc_gamma,
                         mc_gaussian,
                         mc_power_law,
                         mc_delayed_power_law)

dict_preloaded = {}
dict_preloaded['Exponential'] = mc_exponential.dict_ker
dict_preloaded['Gamma'] = mc_gamma.dict_ker
dict_preloaded['Gaussian'] = mc_gaussian.dict_ker
dict_preloaded['PowerLaw'] = mc_power_law.dict_ker
dict_preloaded['DelayedPowerLaw'] = mc_delayed_power_law.dict_ker
