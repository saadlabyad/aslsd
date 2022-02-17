# License: BSD 3 clause

import numpy as np

from aslsd.basis_kernels.basis_kernel_gaussian import GaussianKernel
from aslsd.homogeneous_poisson.hom_poisson import HomPoisson
from aslsd.kernels.kernel import KernelModel
from aslsd.models.mhp import MHP

model_dict = {}

# Poisson
model_dict['Poisson'] = HomPoisson(1)

# Gaussian 1R
kernel_g1d1r = KernelModel(GaussianKernel())
mhp_g1d1r = MHP([[kernel_g1d1r]])
model_dict['Gauss1D1R'] = mhp_g1d1r

# Gaussian 10R
basis_kernels_g1d10r = [GaussianKernel(fixed_indices=[1, 2],
                                       fixed_vars=[0.5, float(ix_bk)])
                        for ix_bk in range(10)]
kernel_g1d10r = KernelModel(basis_kernels_g1d10r)
mhp_g1d10r = MHP([[kernel_g1d10r]])
model_dict['Gauss1D10R'] = mhp_g1d10r
