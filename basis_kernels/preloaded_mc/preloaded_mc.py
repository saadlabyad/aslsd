# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.preloaded_mc import mc_exponential
from aslsd.basis_kernels.preloaded_mc import mc_gamma
from aslsd.basis_kernels.preloaded_mc import mc_power_law

dict_preloaded = {}
dict_preloaded['Exponential'] = mc_exponential.dict_ker
dict_preloaded['Gamma'] = mc_gamma.dict_ker
dict_preloaded['PowerLaw'] = mc_power_law.dict_ker
