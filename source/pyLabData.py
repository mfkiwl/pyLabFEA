#Module Data
'''Introduces a class for data resulting from virtual or physical mechanical tests
for use in pyLabFEA package.

uses NumPy, SciPy, MatPlotLib

Version: 2.0 (2020-03-31)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, March 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''
import pyLabFEM as FE
from pyLabMaterial import Material
import numpy as np
import json
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys

'================='
'define data class'
'================='
class Data(object):
    '''Define class for handling data from virtual mechanical tests in micromechanical 
    simulations and data from physical mechanical tests on materials with various 
    microstructures
    
    Parameters
    ----------
    msl  : list
        List with names of JOSN files with metadata for all microstructures
    path_data : str
        Trunc of pathname for data files
    path_json : str
        Trunc of pathname for JSON metadata files (Optional, default: path_data)
    name : str
        Name of Dataset (optional, default: 'Dataset')
    nth  : int
        Read only every nth lines of input file (optional, default: 1)
    epl_crit : float
        Critical plastic strain at which yield strength is defined (optional, default: 2.e-3)
    d_ep     : float
        Range around critical value of plastic strain in which flow stresses are
        evaluated (optional, default: 5.e-4)
        
    Attributes
    ----------
    msl   : list
    Nset  : int
    name  : str
    pd    : str
    pj    : str
    epc   : float
    dep   : float
    set   : list
    
    
    '''
    def __init__(self, msl, path_data, path_json=None, name='Dataset', nth=1, 
                            epl_crit=2.e-3, d_ep=5.e-4):
        self.msl  = msl
        self.Nset = len(msl)
        self.name = name
        self.pd = path_data
        if path_json is not None:
            self.pj = path_json
        else:
            self.pj = self.pd
        self.nth = nth
        self.epc = epl_crit
        self.dep = d_ep
        self.set = []
        for name in msl:
            self.set.append(self.Set(self, name))

        
    class Set(object):
        '''Define class for handling of a dataset for one individual material 
    
        Parameters
        ----------
        db   : object of type ``Data``
            Parent database from which properties are inherited
        name : str
            Name of JSON file with metadata for microstructure to be stored in this dataset
        
        Attributes
        ----------
        db : object of class ``Data``
        name : str
        N  : int
        eps, epl, eel, sig, ang : (N,) array
        Ndat : int
        sfc, sfpr, peeq : (Ndat,) array
        f_yld : (Ndat,) array
        C11, C12 : float
        E, nu : float
    
        '''
        def __init__(self, db, name):
            self.name = name
            self.db = db
            self.E = None
            with open(db.pj+name+'.json') as f:
                self.meta = json.load(f)

            'import data from CSV files of micromechanical simulations'
            sep   = self.meta['CSV_separator']
            file  = db.pd + self.meta['CSV_data']
            names = self.meta['CSV_format'].split(sep)
            AllData = pd.read_csv(file,header=1,names=names,sep=sep)
            sig_names = ['S11','S22','S33']
            epl_names = ['Ep11','Ep22','Ep33']
            eps_names = ['E11','E22','E33']
            eel_names = ['Ee11','Ee22','Ee33']
            ubc_names = ['StrainX', 'StrainY', 'StrainZ']
            self.eps = AllData[eps_names].to_numpy()[::db.nth]
            self.epl = AllData[epl_names].to_numpy()[::db.nth]
            self.eel = AllData[eel_names].to_numpy()[::db.nth]
            self.sig = AllData[sig_names].to_numpy()[::db.nth]
            self.ubc = AllData[ubc_names].to_numpy()[::db.nth]
            self.N = np.shape(self.sig)[0]
            print(self.N,' data points imported into database ',db.name,' from ',name,)
            
            'calculate eqiv. stresses and strains and theta values'
            sc_full   = FE.s_cyl(self.sig)   # transform stresses into cylindrical coordinates
            peeq_full = FE.eps_eq(self.epl)  # calculate equiv. plastic strain from data
            
            'Consistency checks'
            if np.amax(np.abs(sc_full[:,2])) > 1.:
                print('*** Warning: Large hydrostatic stresses: minimum p=%5.2f MPa, maximum p=%5.2f MPa' 
                      %(np.amin(sc_full[:,2]),np.amax(sc_full[:,2])))
            hh = self.eel - (self.eps-self.epl)
            if np.amax(FE.eps_eq(hh)) > 1.e-8:
                print('*** WARNING: Inconsistency in eps_el!')
            '''hh = self.ang - sc_full[:,1]
            if np.amax(np.abs(hh)) > 1.e-6:
                print('*** WARNING: Inconsistency in theta!')
                print(self.ang[0:self.N:2500], sc_full[0:self.N:2500,1], hh[0:self.N:2500])'''

            'filter load cases'
            self.load_case = []   # list of lists with data set indices belonging to one load case
            hh = []
            uvec = self.ubc[0,:]
            for i in range(self.N):
                if np.linalg.norm(self.ubc[i,:]-uvec) < 1.e-6:
                    hh.append(i)
                else:
                    self.load_case.append(hh)
                    hh = []
                    uvec = self.ubc[i,:]
                    
            '''
            'import Fourier coefficients'
            fc_name = db.pd + self.meta['Fourier_coeff']
            f = open(fc_name, 'r')
            for x in f:
                print(x[0], x)
            '''
            
            'select data points close to yield point => raw data'
            ind  = np.nonzero(np.logical_and(peeq_full>db.epc-db.dep, peeq_full<db.epc+db.dep))[0]
            scyl_raw = sc_full[ind]
            peeq_raw = peeq_full[ind]
            self.Ndat = len(ind)
            self.sy = np.average(scyl_raw[:,0])  # calculate average yield point
            print('Yield strength: %5.2f MPa, estimated from %i data sets' %(self.sy,self.Ndat))
            
            'mirror stress data in theta space'
            sc2 = np.zeros((self.Ndat,2))
            sc2[:,0] = scyl_raw[:,0]
            sc2[:,1] = scyl_raw[:,1]-np.pi
            
            'calculate associated load vector for boundary conditions'
            hs1 = self.sig[ind,:]   # mirrored stresses
            hs2 = FE.sp_cart(sc2)   # original stresses
            sign = np.sign(hs1*hs2) # filter stress components where sign has changed by mirroring
            ubc2 = self.ubc[ind]*sign    # change sign accordingly in BC vector

            'add mirrored data to flow stresses and augment plastic strain arrays accordingly'
            self.sfc_  = np.append(scyl_raw[:,0:2], sc2, axis=0) 
            self.peeq_ = np.append(peeq_raw, peeq_raw, axis=0)
            self.ubc_  = np.append(self.ubc[ind], ubc2, axis=0)
            self.sig_  = FE.sp_cart(self.sfc_) # transform back into 3D principle stress space
            self.Ndat *= 2
            
            'calculate yield function of mirrored raw data'
            self.f_yld_ = np.sign(self.peeq_ - db.epc)
            self.i_el_  = np.nonzero(self.f_yld_<0.)[0]
            self.i_pl_  = np.nonzero(self.f_yld_>=0.)[0]
            
            'filter load cases of mirrored raw data'
            self.lc_ = []   # list of lists with data set indices belonging to one load case
            hh = []
            uvec = self.ubc_[0,:]
            for i in range(self.Ndat):
                if np.linalg.norm(self.ubc_[i,:]-uvec) < 1.e-6:
                    hh.append(i)
                else:
                    self.lc_.append(hh)
                    hh = []
                    uvec = self.ubc_[i,:]
            print('Load cases: ',len(self.lc_))

            'calculate yield stress for each load case'
            hs = []
            ht = []
            self.Nlc = len(self.lc_)
            for i in range(self.Nlc):
                ind = self.lc_[i]
                iel = np.nonzero(self.f_yld_[ind]<0.)[0]  # find elastic data sets in load case
                ipl = np.nonzero(self.f_yld_[ind]>=0.)[0] # find plastic data sets in load case
                ds = self.sfc_[ind[ipl[0]],0] - self.sfc_[ind[iel[-1]],0]   # difference in first plastic sfc and last elastic sfc
                de = self.peeq_[ind[ipl[0]]]  - self.peeq_[ind[iel[-1]]]  # difference in first plastic peeq and last elastic peeq
                'interpolated value of crit. equiv. stress at yield onset'
                hs.append(self.sfc_[ind[iel[-1]],0] + (db.epc-self.peeq_[ind[iel[-1]]])*ds/de)   
                ht.append(self.sfc_[ind[iel[-1]],1]) # polar angle of load case
            self.syc = np.zeros((self.Nlc,3))  # critical cylindrical stress tensor at yield onset
            self.syc[:,0] = np.array(hs)
            self.syc[:,1] = np.array(ht)
            
        def prop_elastic(self):
            '''calculate estimates for elastic properties E, nu associated to data set
            Note: should be calculated from micromechnical data
            '''
            'select data points with eqiv. stress in range [0.1,0.4]sy'
            seq = FE.seq_J2(self.sig)
            ind = np.nonzero(np.logical_and(seq>0.1*self.sy, seq<0.4*self.sy))[0]
            seq = FE.seq_J2(self.sig[ind])
            eeq = FE.eps_eq(self.eel[ind])
            self.E = np.average(seq/eeq)
            
            print('Number', len(ind))
            ssig = 1.e-2
            seps = 1.e4
            a = seps*self.eel[ind,:]
            b = ssig*self.sig[ind,:]
            x = np.linalg.lstsq(a, b, rcond=None)
            c = x[0]*seps/ssig
            self.C11 = c[0,0]
            self.C12 = c[1,0]
            self.C13 = c[2,0]
            self.C21 = c[0,1]
            self.C22 = c[1,1]
            self.C23 = c[2,1]
            self.C31 = c[0,2]
            self.C32 = c[1,2]
            self.C33 = c[2,2]
            print('Residuals: ',x[1])
            #self.nu = self.C12/(self.C11 + self.C12) # estimate for isotropic materials
            #self.E = self.C12*(1.+self.nu)*(1.-2.*self.nu)/self.nu
            self.nu = 0.3
            print('***Microstructure ',self.name,' in database ',self.db.name)
            print('Estimated elasic constants: E=%5.2f GPa, nu=%4.2f' % (self.E/1000, self.nu))
            print('Estimated elasic tensor C_ij (GPa): \n%8.2f, %8.2f, %8.2f' 
                   % (self.C11/1000, self.C12/1000, self.C13/1000))
            print('%8.2f, %8.2f, %8.2f' 
                   % (self.C21/1000, self.C22/1000, self.C23/1000))
            print('%8.2f, %8.2f, %8.2f' 
                   % (self.C31/1000, self.C32/1000, self.C33/1000))
                   
        def augment_data(self, graph=True):
            '''Raw data is distributed over entire deviatoric plane to create a suited
            data set for training of SVC, graphical output of raw data together with 
            augmented data can be created.
            '''
            if self.E is None:
                self.prop_elastic()
                
            'define material'
            self.mat = Material(name='ML-'+self.name)
            self.mat.elasticity(E=self.E, nu=self.nu)
            self.mat.plasticity(sy=self.sy)
            self.mat.microstructure(texture=self.text_param)

            'augment raw data and create result vector (yield function)'
            self.sc_train, self.yf_train = self.mat.create_sig_data(syc=self.syc, Nseq=25, extend=False)
            self.sc_test, self.yf_test   = self.mat.create_sig_data(syc=self.syc[0:self.Nlc:10,:])

            if graph:
                print('Plot raw data and training data extended over deviatoric plane for data set ',self.name)
                fig, ax  = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
                Ncol = int(len(self.sc_train)/self.Nlc)
                ax.scatter(self.sc_train[:,1], self.sc_train[:,0], s=20, c=self.yf_train, cmap=plt.cm.Paired)
                #ax.scatter(self.sfc_[:,1], self.sfc_[:,0], s=20, c=self.f_yld_, cmap=plt.cm.Paired)
                ax.plot(self.syc[:,1], self.syc[:,0], '-m')
                ax.set_title('SVC yield function for data set '+self.name)
                ax.set_xlabel(r'$\theta$ (rad)', fontsize=20)
                ax.set_ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=20)
                ax.tick_params(axis="x", labelsize=16)
                ax.tick_params(axis="y", labelsize=16)
                plt.show()

            
        def plot_seq(self):
            plt.plot(self.peeq[self.i_el]*100,self.sfc[self.i_el,0],'.b')
            plt.plot(self.peeq[self.i_pl]*100,self.sfc[self.i_pl,0],'.r')
            plt.title('equiv. plastic strain vs. equiv. stress (mirrored data)')
            plt.xlabel(r'$\epsilon_{eq}^{pl}$ (%)')
            plt.ylabel(r'$\sigma_{eq}^\mathrm{J2}$ (MPa)')
            plt.show()
            
        def plot_theta(self):
            plt.plot(self.sfc[self.i_el,1],self.sfc[self.i_el,0],'.b')
            plt.plot(self.sfc[self.i_pl,1],self.sfc[self.i_pl,0],'.r')
            #plt.plot(t_dat[self.i_el],self.sfc[self.i_el,0],'.b')
            #plt.plot(t_dat[self.i_pl],self.sfc[self.i_pl,0],'.r')
            plt.title('polar angle vs. equiv. stress')
            plt.xlabel(r'$\theta$ (rad)')
            plt.ylabel(r'$\sigma_{eq}^\mathrm{J2}$ (MPa)')
            plt.show()
        
        def plot_flow_stress(self):
            plt.polar(self.sfc[self.i_pl,1],self.sfc[self.i_pl,0]/self.sy,'.r')
            plt.polar(self.sfc[self.i_el,1],self.sfc[self.i_el,0]/self.sy,'.b')
            #plt.polar(np.linspace(0.,2*np.pi,36), np.ones(36), '-k', linewidth=2)
            plt.title('polar angle vs. J2 equiv. stress')
            plt.legend(['data above yield crit.','data below yield crit.'], loc=(1.,0.8))
            plt.show()
            
    def train_SVC(self):
            '''Train SVC for all yield functions of the microstructures provided in the list ``Data.msl``. In First step, the 
            training data for each set is generated by creating stresses on the deviatoric plane and calculating their catgegorial
            yield function ("-1": elastic, "+1": plastic). Furthermore, axes in different dimensions for microstructural
            features are introduced that describe the relation between the different sets.
            '''
            
            train_sc, test_sc = mat_ml.setup_yf_SVM(xt, yt, cyl=True, C=10, gamma=4., fs=0.25, plot=True)   

            print('\n-------------------------\n')
            print('SVM classification fitted')
            print('-------------------------\n')
            print(mat_ml.svm_yf)
            print("Training data points:", self.Ndat," raw data points, ",len(xt),"augmented data points (polar angles)")
            print("Training set score: {} %".format(train_sc))
        