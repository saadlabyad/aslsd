# License: BSD 3 clause

from aslsd.functionals.kernels.basis_kernels.\
    preloaded_mc import (mc_exponential, mc_power_law, mc_delayed_power_law,
                         mc_uniform, mc_triangular, mc_gaussian, mc_gamma,
                         mc_rayleigh)

dict_preloaded = {}
# Markovian
dict_preloaded['Exponential'] = mc_exponential.dict_ker
# Heavy-tailed
dict_preloaded['PowerLaw'] = mc_power_law.dict_ker
dict_preloaded['DelayedPowerLaw'] = mc_delayed_power_law.dict_ker
# Dense
dict_preloaded['Uniform'] = mc_uniform.dict_ker
dict_preloaded['Triangular'] = mc_triangular.dict_ker
dict_preloaded['Gaussian'] = mc_gaussian.dict_ker
dict_preloaded['Gamma'] = mc_gamma.dict_ker
# Other non-monotonically decaying
dict_preloaded['Rayleigh'] = mc_rayleigh.dict_ker
