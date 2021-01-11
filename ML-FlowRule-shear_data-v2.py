#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train ML flow rules to data sets for random and Goss textures

Created on Tue Jan  5 16:07:44 2021

@author: Alexander Hartmaier
"""

import pylabfea as FE
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import pickle
print('pyLabFEA version',FE.__version__)

def unit_stress(nd=90, nsh=3):
    '''Construct unit stresses that define the loading cases for RVE calculations 
    in the full stress space. Normal stresses are constructed from polar angle in 
    deviatoric principle stress plane, assuming an equiv. stress of 1 and a 
    hydrostatic stress of 0.
    Shear stress components are constructed via a unit quaternion that defines the 
    rotation from the principal stress space into the full stress space.
    Stress space is assumed to be given in a reference frame that corresponds to 
    material frame and applied load frame.
    
    Parameters
    ----------
    nd : int
        number of polar angles for construction of unit stresses (optional, default: 8)
    nsh : int
        number of variations in each shear stress component per polar angle
    
    Returns
    -------
    sig : (N,6) array 
        unit stresses in Voigt notation (s11, s22, s33, s23, s13, s12)
    N   : int
        number of generated unit stresses (=nd*nsh**3)
    '''
    #define ranges for polar angle, and quaternion vector for shear stresses
    # 8 stresses in deviatoric principle stress space are constructed
    # 3^3=27 shear stresses are constructed
    i0 = 12
    N = nd*nsh + i0
    print('Generating %i unit stresses, with %i polar angles and %i variations of shear components.' % (N, nd, nsh))
    theta = np.linspace(0,np.pi,nd, endpoint=False) # polar angle exclude endpoint due to peridicity
    sig = np.zeros((N,6))
    sig[0,:] = np.array([1., 0., 0., 0., 0.,0.])  # uniaxial load cases
    sig[1,:] = np.array([0., 1., 0., 0., 0.,0.])
    sig[2,:] = np.array([0., 0., 1., 0., 0.,0.])
    sig[3,:] = np.array([1., 1., 0., 0., 0.,0.]) # bi-axial load cases
    sig[4,:] = np.array([0., 1., 1., 0., 0.,0.])
    sig[5,:] = np.array([1., 0., 1., 0., 0.,0.])
    sig[6,:] = np.array([1., -1., 0., 0., 0.,0.]) # pure shear load cases
    sig[7,:] = np.array([0., 1., -1., 0., 0.,0.])
    sig[8,:] = np.array([-1., 0., 1., 0., 0.,0.])
    sig[9,:] = np.array([0., 0., 0., 1., 0.,0.]) # simple shear load cases
    sig[10,:] = np.array([0., 0., 0., 0., 1.,0.])
    sig[11,:] = np.array([0., 0., 0., 0., 0.,1.])
    seq = FE.seq_J2(sig[0:i0,:])
    sig[0:i0,:] /= seq[:,None]
    for i, pa in enumerate(theta):
        for j in range(nsh):
            ind = i0 + nsh*i + j
            sig[ind,0:3] = FE.sp_cart([1., theta[i], 0.])  # create deviatoric principal stress
            sig[ind,3+j] = (-1.)**i * (i%5)/2.
            seq = FE.seq_J2(sig[ind,:])
            if seq<0.01:
                sig[ind,3+j] *= -1.
                seq = FE.seq_J2(sig[ind,:])
            sig[ind,:] /= seq # normalize stresses
    return sig, N

def find_yloc(x, sig, mat):
    '''Function to expand unit stresses by factor and calculate yield function;
    used by search algorithm to find zeros of yield function.

    Parameters
    ----------
    x : (N,)-array
        Multiplyer for stress
    sig : (N,6) array
        unit stress
    
    Returns
    -------
    f : 1d-array
        Yield function evaluated at sig=x.sp
    '''
    
    N = len(sig)
    f = np.zeros(N)
    for i in range(N):
        f[i] = mat.calc_seq(sig[i,:]*x[i]) - mat.sy
    return f

#define Barlat material for Goss texture (RVE data, combined data set)
bp = [0.81766901, -0.36431565, 0.31238124, 0.84321164, -0.01812166, 0.8320893, 0.35952332,
      0.08127502, 1.29314957, 1.0956107, 0.90916744, 0.27655112, 1.090482, 1.18282173,
      -0.01897814, 0.90539357, 1.88256105, 0.0127306 ]
mat_bG = FE.Material(name='Goss-combined-Yld2004-18p')
mat_bG.elasticity(E=151220., nu=0.3)
mat_bG.plasticity(sy=46.76, barlat=bp[0:18], barlat_exp=8)

#create set of unit stresses and assess yield stresses
sunit, N = unit_stress(nd=60,nsh=3)
x1 = fsolve(find_yloc, np.ones(N)*mat_bG.sy, args=(sunit,mat_bG), xtol=1.e-5)
sig = sunit * x1[:,None]

#define material as basis for ML flow rule
data_GS = FE.Data(sig, None, name="Goss-Barlat", sdim=6, epl_crit=0.002, mirror=True)
mat_mlGb = FE.Material(name='ML-Goss-Barlat')       # define material 
#mat_mlGb.elasticity(E=mat_bG.E, nu=mat_bG.nu)  
#mat_mlGb.plasticity(sy=mat_bG.sy)
mat_mlGb.from_data(data_GS.mat_param)          # define microstructural parameters for material
sc = FE.s_cyl(mat_mlGb.msparam[0]['sig_yld'][0])
mat_mlGb.polar_plot_yl(data=sc, dname='full stress', arrow=True)
print(mat_bG.E, mat_mlGb.E)
print(mat_bG.nu, mat_mlGb.nu)
print(mat_bG.sy, mat_mlGb.sy)

#train SVC with combined data from random texture
mat_mlGb.train_SVC(C=10, gamma=3.)
mat_mlGb.export_MLparam()

# stress strain curves
mat_mlGb.calc_properties(verb=False, eps=0.0013, sigeps=True)
mat_mlGb.plot_stress_strain()

# plot yield locus with stress states
s=80
ax = mat_mlGb.plot_yield_locus(xstart=-1.8, xend=1.8, ref_mat=mat_bG, Nmesh=200)
stx = mat_mlGb.sigeps['stx']['sig'][:,0:2]/mat_mlGb.sy
sty = mat_mlGb.sigeps['sty']['sig'][:,0:2]/mat_mlGb.sy
et2 = mat_mlGb.sigeps['et2']['sig'][:,0:2]/mat_mlGb.sy
ect = mat_mlGb.sigeps['ect']['sig'][:,0:2]/mat_mlGb.sy
ax.scatter(stx[1:,0],stx[1:,1],s=s, c='#f0ff00', edgecolors='k')
ax.scatter(sty[1:,0],sty[1:,1],s=s, c='#f0ff00', edgecolors='k')
ax.scatter(et2[1:,0],et2[1:,1],s=s, c='#f0ff00', edgecolors='k')
ax.scatter(ect[1:,0],ect[1:,1],s=s, c='#f0ff00', edgecolors='k')
plt.show()

