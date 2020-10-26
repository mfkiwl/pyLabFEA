import pytest
import os
import pylabfea as FE
import numpy as np
print('pyLabFEA version',FE.__version__)


mat = FE.Material()               # define  material
mat.elasticity(E=200.e3, nu=0.3)
mat.plasticity(sy=150., hill=[0.7,1.,1.4], khard=0.)   # define material with ideal isotropic plasticity

sig = np.zeros(6)
eps = np.zeros(6)
epl = np.zeros(6)
depl = np.zeros(6)
deps = np.zeros(6)
deps[0] = 0.0003
deps[1] = -0.3*deps[0]
deps[2] = -0.3*deps[0]
CV = np.array([[mat.C11, mat.C12, mat.C12, 0.,  0.,  0.], \
               [mat.C12, mat.C11, mat.C12, 0.,  0.,  0.], \
               [mat.C12, mat.C12, mat.C11, 0.,  0.,  0.], \
               [0.,  0.,  0.,  mat.C44, 0.,  0.], \
               [0.,  0.,  0.,  0.,  mat.C44, 0.], \
               [0.,  0.,  0.,  0.,  0.,  mat.C44]])

for kinc in range(5):
    fyld, sig, depl, gr_stiff = mat.response(sig, epl, deps, CV)
    eps += deps
    epl += depl
    print('\n>>> Output:', kinc, deps, depl)
    print('sig: ',np.around(sig, decimals=5), FE.Stress(sig).sJ2())
    print('eps: ',np.around(eps, decimals=6), FE.Strain(eps).eeq())
    print('epl: ',np.around(epl, decimals=6), FE.Strain(epl).eeq())
    hh = sig -  CV@(eps-epl)
    if FE.seq_J2(hh) > 1.:
        print('TEST failed', hh, CV@(eps-epl), eps-epl, mat.calc_yf(sig))

