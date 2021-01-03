#Module pylabfea.data
'''Module pylabfea.data introduces the class ``Data`` for handling of data resulting 
from virtual or physical mechanical tests in the pyLabFEA package. This class provides the 
methods necessary for analyzing data. During this processing, all information is gathered 
that is required to define a material, i.e., all parameters for elasticity, plasticity, 
and microstructures are provided from the data. Materials are defined in 
module pylabfea.material based on the analyzed data of this module.

uses NumPy, SciPy, MatPlotLib, Pandas

Version: 3.2 (2020-10-26)
Author: Alexander Hartmaier, ICAMS/Ruhr-University Bochum, April 2020
Email: alexander.hartmaier@rub.de
distributed under GNU General Public License (GPLv3)'''

from pylabfea.basic import *
from pylabfea.model import Model
from pylabfea.material import Material
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from copy import deepcopy

sig_names = ['S11','S22','S33','S23','S13','S12']
theta_name = ['theta']
epl_names = ['Ep11','Ep22','Ep33','Ep23','Ep13','Ep12']
eps_names = ['E11','E22','E33','E23','E13','E12']
eel_names = ['Ee11','Ee22','Ee33','Ee23','Ee13','Ee12']
ubc_names = ['StrainX', 'StrainY', 'StrainZ', 'StrainYZ','StrainXZ','StrainXY']
sbc_names = ['StressX','StressY','StressZ','StressYZ','StressXZ','StressXY']
            
#=================
#define data class
#=================
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
        Trunc of pathname for JSON metadata files (optional, default: path_data)
    name : str
        Name of Dataset (optional, default: 'Dataset')
    sdim : int
        Dimensionality of stresses; if sdim=3 only principal stresses are considered (optional, default: 3)
    mirror : Boolean
        Indicate if stresses shall be doubled by periodic completion of cylindrical stresses
        on the deviatoric plane of the principal stress space (optional, default: False)
    nth  : int
        Read only every nth lines of input file (optional, default: 1)
    epl_crit : float
        Critical plastic strain at which yield strength is defined (optional, default: 2.e-3)
    d_ep     : float
        Range around critical value of plastic strain in which flow stresses are
        evaluated (optional, default: 5.e-4)
    npe : int
        Number of equiv. plastic strains used for work hardening
        
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
    sy_av : float
    E_av  : float
    nu_av : float
    flow_stress : cyl. stress
    mat_param : dictionary
        Contains available data for microstructural parameters ("texture", "work_hard", "flow_stress") to be transfered to material.microstructure
    '''
    def __init__(self, msl, path_data, path_json=None, name='Dataset', nth=1, sdim=3, mirror=False,
                            epl_crit=2.e-3, d_ep=5.e-4, epl_max=0.03, npe=1, plot=False):
        self.msl  = msl
        self.Nset = len(msl)
        self.name = name
        self.pd = path_data
        self.sdim = sdim
        self.mirror = mirror
        if path_json is not None:
            self.pj = path_json
        else:
            self.pj = self.pd
        self.nth = nth
        self.epc = epl_crit
        self.dep = d_ep
        self.set = []
        self.flow_stress = None
        self.predef = False
        #import and pre-process data of all microstructure
        for name in msl:
            self.set.append(self.Set(self, name, plot=plot))
        #calculate average properties over all microstructures
        prop = np.zeros(3)
        micr = []
        name = []
        sfl  = []
        peeq_min = 10.  # minimum final equiv. plastic strain reached in sets
        Nlc_min = 1000  # minimum number of load cases reached in sets
        ttyp = self.set[-1].texture_type
        for dset in self.set:
            prop += np.array([dset.sy, dset.E, dset.nu])
            micr.append(dset.texture_param)
            name.append(dset.texture_name)
            if self.flow_stress:
                peeq_min = np.minimum(peeq_min, dset.peeq_full[-10])
                Nlc_min = np.minimum(Nlc_min, 2*len(dset.load_case))
            else:
                peeq_min = self.epc
                Nlc_min = np.minimum(Nlc_min, dset.Nlc)
            if dset.texture_type!=ttyp and dset.texture_type!='Random':
                warnings.warn('Warning: Different texture types mixed in data set '+str(name))
        prop /= self.Nset  # average properties over all microstructures
        peeq_min = np.minimum(peeq_min, epl_max)
        self.sy_av = prop[0] # information needed when material.plasticity is initiated 
        self.E_av  = prop[1] # information needed when material.elasticity is initiated 
        self.nu_av = prop[2]
        print('\n###   Data set "%s"  ###' % (self.name))
        print('Type of microstructure: ', ttyp)
        print('Imported %i data sets for textures, with %i hardening stages and %i load cases each.' %(self.Nset, npe, Nlc_min))
        print('Averaged properties : E_av=%5.2f GPa, nu_av=%4.2f, sy_av=%4.2f MPa' % (self.E_av/1000, self.nu_av, self.sy_av))
        self.mat_param = {      # this information of the data will be copied into the material upon definition of its microstructure
            'ms_type'     : ttyp,      # unimodal texture type
            'Npl'         : npe,       # number of PEEQ values 
            'Nlc'         : Nlc_min,   # number of load cases covered in data (minimum of load cases reached over all sets)
            'Ntext'       : self.Nset, # number of microstructures covered by sets
            'texture'     : np.array(micr),  # texture parameter: mixture parameter ms_type wrt random
            'tx_name'     : name,      # list of names of different textures
            'peeq_max'    : peeq_min,  # maximum PEEQ covered in data (must be minimum of value reached over all sets)
            'epc'         : epl_crit,  # critical PEEQ for with yield stress is defined 
            'work_hard'   : np.linspace(epl_crit,peeq_min,npe) # values of PEEQ for which flow stresses are available
        }
        print(self.mat_param)
        #for each load case in each set: interpolate flow stress to fixed PEEQ grid
        Nlc = self.mat_param['Nlc']
        sf = np.zeros((self.Nset,npe,Nlc,3))  
        sflow_av = np.zeros((self.Nset,npe))
        iset = 0
        for dset in self.set:   # loop over data sets
            ipeeq=0
            if not self.flow_stress:
                #data for stress tensor at yield onset provided in data
                sf[iset,ipeeq,:,:] = dset.syc
                sflow_av[iset,ipeeq] = np.average(dset.syc[:,0])    # average seq over polar angles
            else:
                N0 = int(Nlc/2)
                N1 = len(dset.load_case)
                x0 = np.linspace(0,1,N0)
                x1 = np.linspace(0,1,N1)
                y1 = np.linspace(0,N1-1,N1)
                ilc = np.interp(x0,x1,y1).astype(int)
            
                for peeq in self.mat_param['work_hard']:  # loop over PEEQS for which flow stresses are required
                    hs = []
                    ht = []
                    for i in range(N0):  # loop over load cases for each level of PEEQ
                        #this should be generalized for the case that Nlc varies strongly over the sets
                        ind = dset.load_case[ilc[i]]  # indices of data points belonging to this load case
                        #select data points above and below peeq
                        iel = np.nonzero(dset.peeq_full[ind]<peeq)[0]  # find elastic data sets in load case
                        ipl = np.nonzero(dset.peeq_full[ind]>=peeq)[0] # find plastic data sets in load case
                        eel = dset.peeq_full[ind[iel[-1]]]  # last data point < peeq
                        epl = dset.peeq_full[ind[ipl[0]]]   # first data point > peeq
                        s0  = dset.sc_full[ind[iel[-1]],0]  # largest flow stress below peeq
                        s1  = dset.sc_full[ind[ipl[0]],0]   # smallest flow stress above peeq
                        ds = s1 - s0     # difference in seq b/w data points around peeq
                        de = epl  - eel  # difference in peeq for closest data points
                        sint = s0 + (peeq-eel)*ds/de # linearly interpolated equiv. flow stress at peeq
                        theta = dset.sc_full[ind[iel[-1]],1] # polar angle (not interpolated)
                        hs.append(sint)
                        hs.append(sint)  # append twice to consider tension compression symmetry in material
                        ht.append(theta) # append original polar angles
                        theta -= np.pi   # mirror polar angle
                        if theta<-np.pi:
                            theta += 2*np.pi
                        ht.append(theta) # append mirrored polar angles
                    ind = np.argsort(ht) # sort w.r.t. polar angle
                    sf[iset,ipeeq,:,0] = np.array(hs)[ind]   # first component: seq
                    sf[iset,ipeeq,:,1] = np.array(ht)[ind]   # second component: polar angle
                    sflow_av[iset,ipeeq] = np.average(hs)    # average seq over polar angles
                    ipeeq += 1
            iset += 1
        self.mat_param['flow_stress'] = sf       # cylindrical flow stresses at PEEQs in 'work_hard' for each texture
        self.mat_param['flow_seq_av'] = sflow_av # averaged equiv. flow stresses at PEEQs in 'work_hard' for each texture
 
    class Set(object):
        '''Define class for handling of a dataset for one individual material 
    
        Parameters
        ----------
        db   : object of type ``Data``
            Parent database from which properties are inherited
        name : str
            Name of JSON file with metadata for microstructure to be stored in this dataset
        plot : Boolean
            Graphical output of data in each set (optional, default: False)
        
        Attributes
        ----------
        db : object of class ``Data``
        name : str
        N  : int
            Number of imported data points
        Ndat : int
            Number of raw data points (filtered data lying around yield point mirrored wrt polar angle)
        Nlc  : int
            Number of load cases in raw data
        E, nu : float
            Elastic parameters obtained from data
        sy    : float
            Yield strength obtained from data
        texture_param : float
            Microstructure parameter for texture
        eps, epl, eel, sig, ubc, sc_full : (N,) array
        sfc_   : (Ndat,3) array
            Filtered cyl. stress tensor around yield point
        peeq_  : (Ndat,) array
            Filtered equiv. plastic strain around yield point
        ubc_   : (Ndat,db.sdim) array
            Filtered boundary condition vector around yield point
        sig_   : (Ndat,db.sdim) array
            Filtered stress tensor around yield point (princ. stresses for sdim=3)
        f_yld_ : (Ndat,) array
            Categorial yield function of data points around yield point ("-1": elastic, "+1": plastic)
        i_el_  : (Ndat,) array
            Filtered indeces of data points lying in elastic regime
        i_pl_  : (Ndat,) array
            Filtered indeces for data points lying in plastic regime
        syc    : (Nlc,3) array
            Yield strength: interpolated cyl. stress tensor at onset of yielding for individual load cases
        load_case : list
            List of lists with data set indices belonging to one single load case from full data 
            (index space: [0,N])
        lc_    : list
            List of lists with data set indices belonging to one single load case from filtered data 
            around yield point (index space: [0,Ndat])
    
        '''
        def __init__(self, db, name, plot=False):
            '''def prop_elastic():
                'calculate estimates for elastic properties E, nu associated to data set'
                ssig = 1.e-1
                seps = 1.e3
                a = seps*self.eps[ind,:]
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
                
                #self.nu = self.C12/(self.C11 + self.C12) # estimate for isotropic materials
                #self.E = self.C12*(1.+self.nu)*(1.-2.*self.nu)/self.nu
                
                print('Estimated elasic tensor C_ij (GPa): \n%8.2f, %8.2f, %8.2f' 
                       % (self.C11/1000, self.C12/1000, self.C13/1000))
                print('%8.2f, %8.2f, %8.2f' 
                       % (self.C21/1000, self.C22/1000, self.C23/1000))
                print('%8.2f, %8.2f, %8.2f' 
                       % (self.C31/1000, self.C32/1000, self.C33/1000))
                print('Residuals of fitting elastic parameters: ',x[1])'''
            
            self.name = name
            self.db = db
            eps_0 = 0.
            sig_0 = 0.
            epl_0 = 0.
            with open(db.pj+name+'.json') as f:
                self.meta = json.load(f)

            #import data from CSV files of micromechanical simulations
            sep   = self.meta['CSV_separator']
            file  = db.pd + self.meta['CSV_data']
            names = self.meta['CSV_format'].split(sep)
            AllData = pd.read_csv(file,header=1,names=names,sep=sep)
            dtype = self.meta['Class']
            if dtype == 'Flow_Stress':
                if db.flow_stress is None:
                    db.flow_stress = True
                elif not db.flow_stress:
                    raise ValueError('Incompatible data sets: Yield_Stress and Flow-Stress classes cannot be mixed.')
            elif dtype == 'Yield_Stress':
                if db.flow_stress is None:
                    db.flow_stress = False
                elif db.flow_stress:
                    raise ValueError('Incompatible data sets: Yield_Stress and Flow-Stress classes cannot be mixed.')
            elif dtype == 'Predeformed':
                if db.flow_stress is None:
                    db.flow_stress = True
                    db.predef = True
                elif not db.flow_stress:
                    raise ValueError('Incompatible data sets: Yield_Stress and Flow-Stress classes cannot be mixed.')
            else:
                raise ValueError('Unknown value for Class: '+str(dtype))
            self.sig = AllData[sig_names[0:db.sdim]].to_numpy()[::db.nth]
            self.N   = np.shape(self.sig)[0]
            if db.predef:
                sig_0 = deepcopy(self.sig[0])
                #self.sig -= sig_0
            if db.sdim == 3:
                self.sc_full = s_cyl(self.sig)   # transform pinc. stresses into cylindrical coordinates
                self.evec = None
            else:
                self.sc_full = np.zeros((self.N,3))
                self.evec    = np.zeros((self.N,3,3))
                for i, hsv in enumerate(self.sig):
                    hso = Stress(hsv)  # define object of stress tensor
                    self.sc_full[i,:] = hso.cyl() # transform princ. stress into cylindrical stress
                    self.evec[i,:,:]  = hso.evec # store eigenvectors in form of rotation matrix from reference frame to princ. stress frame
            #self.theta   = AllData[theta_name].to_numpy()[::db.nth]
            if db.flow_stress:
                #flow stresses are privided for various plastic strains
                self.eps = AllData[eps_names].to_numpy()[::db.nth]
                if db.predef:
                    eps_0=deepcopy(self.eps[0])
                    self.eps -= eps_0
                self.epl = AllData[epl_names].to_numpy()[::db.nth]
                if db.predef:
                    epl_0 = deepcopy(self.epl[0])
                    self.epl -= epl_0
                #self.eel = AllData[eel_names].to_numpy()[::db.nth]
                try:
                    self.ubc = AllData[ubc_names[0:db.sdim]].to_numpy()[::db.nth]
                except:
                    self.ubc = AllData[sbc_names[0:db.sdim]].to_numpy()[::db.nth]
                #calculate eqiv. stresses and strains and theta values
                self.peeq_full = eps_eq(self.epl)  # calculate equiv. plastic strain from data
            #print('***Shapes',self.eps.shape)
            print('\n*** Microstructure:',self.name,'***')
            print(self.N,' data points imported into database ',self.db.name)
            if db.flow_stress:
                print('Data for flow stresses at various plastic strains imported.')
            if db.predef:
                print('\n###Mechanical data obtained after predeformation.')
                print('Initial stress: ',sig_0, ' equiv. stress: ', seq_J2(sig_0))
                print('Initial total strain: ',eps_0, 'equiv. total strain', eps_eq(eps_0))
                print('Initial plastic strain: ',epl_0, 'equiv. plastic strain: ',eps_eq(epl_0))
                print('Initial values have been subtracted from stress-strain data.')
            
            #import texture parameters
            self.texture_param = self.meta['Texture_index']
            self.texture_name  = self.meta['Texture_name']
            self.texture_type  = self.meta['Texture_type']
            print('Texture ', self.texture_name, 'with texture parameter: ',self.texture_param)
            
            #Consistency checks
            if np.amax(np.abs(self.sc_full[:,2])) > 1.:
                print('*** Warning: Large hydrostatic stresses: minimum p=%5.2f MPa, maximum p=%5.2f MPa' 
                      %(np.amin(self.sc_full[:,2]),np.amax(self.sc_full[:,2])))
            '''hh = self.theta - self.sc_full[:,1]
            if np.amax(np.abs(hh)) > 1.e-6:
                print('*** WARNING: Inconsistency in theta!')
                print(self.theta[0:self.N:2500], self.sc_full[0:self.N:2500,1], hh[0:self.N:2500])'''
                
            if not db.flow_stress:
                #data provided only for stress tensor at yield point
                self.syc  = s_cyl(self.sig)   # critical cylindrical stress tensor at yield onset
                self.sy   = np.average(self.syc[:,0]) # average value for yield strength of data set
                self.Nlc  = self.N
                self.Ndat = self.N
                print('Estimated yield strength: %5.2f MPa, from %i data sets with PEEQ %5.3f' 
                      %(self.sy,self.Ndat,db.epc))
                if "Prop_E" in self.meta:
                    self.E = self.meta['Prop_E']
                    print("Young's modulus provided in meta data:",self.E)
                else:
                    self.E = None
                if "Prop_nu" in self.meta:
                    self.nu = self.meta['Prop_nu']
                    print("Poisson ratio provided in meta data:", self.nu)
                else:
                    self.nu = None
                
            else:
                #data is provided for flow stress at various plastic strains

                #filter load cases for complete stress-strain data
                self.load_case = []   # list of lists with data set indices belonging to one load case
                hh = []
                uvec = self.ubc[0,:]
                for i in range(self.N):
                    if np.linalg.norm(self.ubc[i,:]-uvec) < 1.e-4:
                        hh.append(i)
                    else:
                        self.load_case.append(hh)
                        hh = []
                        uvec = self.ubc[i,:]
                
                #select data points close to yield point => yield strength sy and raw data for ML flow rule
                ind = np.nonzero(np.logical_and(self.peeq_full>db.epc-db.dep, self.peeq_full<db.epc+db.dep))[0]
                if len(ind)<10:
                    warnings.warn('***Warning: too few data points around yield criterion:'+str(len(ind))+', '\
                                  +str(ind)+', '+str(db.epc)+', '+str(db.dep))
                self.Ndat = len(ind)
                scyl_raw = self.sc_full[ind]
                self.sy = np.average(scyl_raw[:,0])  # get first estimate of yield point, will be refined later
            
                if db.mirror:
                    #mirror stress data w.r.t. theta in deviatoric stress space
                    peeq_raw = self.peeq_full[ind]
                    sc2 = np.zeros((self.Ndat,2))
                    sc2[:,0] = scyl_raw[:,0]
                    sc2[:,1] = scyl_raw[:,1]-np.pi
                    ih = np.nonzero(sc2[:,1]<-np.pi)[0]
                    sc2[ih,1] += 2*np.pi
            
                    #calculate associated load vector for boundary conditions
                    hs1 = self.sig[ind,:]    # original stresses
                    hs2 = np.array(hs1)      # mirrored stresses
                    sign = np.sign(np.sum(hs1*hs2, axis=1))   # filter stress components where sign has changed by mirroring
                    ubc2 = self.ubc[ind]*sign[:,None] # change sign accordingly in BC vector

                    #add mirrored data to flow stresses and augment plastic strain arrays accordingly
                    self.Ndat *= 2    # number of data points in raw data
                    self.sfc_  = np.append(scyl_raw[:,0:2], sc2, axis=0)
                    self.peeq_ = np.append(peeq_raw, peeq_raw, axis=0)
                    self.ubc_  = np.append(self.ubc[ind], ubc2, axis=0)
                    if db.sdim == 3:
                        self.sig_  = sp_cart(self.sfc_) # transform back into 3D principle stress space
                    else:
                        self.sig_ = np.zeros((self.Ndat,6))
                        for i,sc in enumerate(self.sfc_):
                            self.sig_[i,:] = svoigt(sc, self.evec[i,:,:])
                else: 
                    self.sfc_  = self.sc_full[ind]
                    self.peeq_ = self.peeq_full[ind]
                    self.ubc_  = self.ubc[ind]
                    self.sig_ = self.sig[ind]
            
                #calculate yield function of raw data
                self.f_yld_ = np.sign(self.peeq_ - db.epc)
                self.i_el_  = np.nonzero(self.f_yld_<0.)[0]
                self.i_pl_  = np.nonzero(self.f_yld_>=0.)[0]
            
                #filter load cases for selected data around yield point
                self.lc_ = []   # list of lists with data set indices belonging to one load case
                hh = []
                uvec = self.ubc_[0,:]
                #print('First load case: ', uvec)
                for i in range(self.Ndat):
                    if np.linalg.norm(self.ubc_[i,:]-uvec) < 1.e-4:
                        hh.append(i)
                    else:
                        if len(hh) > 2:
                            self.lc_.append(hh)
                            uvec = self.ubc_[i,:]
                            #print('Load case selection', i, uvec, self.lc_[-1][0],self.lc_[-1][-1])
                        hh = []
                self.Nlc = len(self.lc_)
                print('Number of load cases: ',self.Nlc, '; with ',self.Ndat, ' data points around yield point')

                #for each load case: interpolate flow stress to fixed PEEQ
                hs = []  # help array for equiv stress (seq) at yielding
                ht = []  # help array for polar angle (theta) at yielding
                hv = []  # help array for Voigt stress at yielding
                for i in range(self.Nlc):
                    ind = self.lc_[i]
                    iel = np.nonzero(self.f_yld_[ind]<0.)[0]  # find elastic data sets in load case
                    ipl = np.nonzero(self.f_yld_[ind]>=0.)[0] # find plastic data sets in load case
                    if len(iel)==0 or len(ipl)==0:
                        print('**Load case', i, 'not enough data points around yield point. iel=',iel,'ipl=',ipl)
                        self.Nlc -= 1
                    else:
                        # difference in seq b/w first plastic and last elastic data point
                        dseq = self.sfc_[ind[ipl[0]],0] - self.sfc_[ind[iel[-1]],0]
                        # difference in full stress b/w first plastic and last elastic data point
                        dsig = self.sig_[ind[ipl[0]],0] - self.sig_[ind[iel[-1]],0]
                        # difference in peeq b/w first plastic and last elastic data point
                        de = self.peeq_[ind[ipl[0]]]  - self.peeq_[ind[iel[-1]]]
                        # difference in peeq b/w nominal yield strain and last elastic data point
                        dint = db.epc-self.peeq_[ind[iel[-1]]]
                        # linearly interpolated equiv. stress at yield onset
                        hs.append(self.sfc_[ind[iel[-1]],0] + dseq*dint/de)
                        ht.append(self.sfc_[ind[iel[-1]],1]) # polar angle of load case
                        hv.append(self.sig_[ind[iel[-1]],:] + dsig*dint/de)
                ind = np.argsort(ht)   # sort w.r.t. polar angle
                self.syc = np.zeros((self.Nlc,3))   # critical cylindrical stress tensor at yield onset
                self.syc[:,0] = np.array(hs)[ind]   # first component: seq
                self.syc[:,1] = np.array(ht)[ind]   # second component: polar angle
                self.syld = np.array(hv)[ind]       # full Voigt stress at yield onset
                self.sy = np.average(self.syc[:,0]) # refined value for yield strength of data set
            
                #select data points with eqiv. stress in range [0.1,0.4]sy => elastic constants
                seq = seq_J2(self.sig)
                ind1 = np.nonzero(np.logical_and(seq>0.1*self.sy, seq<0.4*self.sy))[0]
                seq = seq_J2(self.sig[ind1])
                eeq = eps_eq(self.eps[ind1])
                self.E = np.average(seq/eeq)
                self.nu = 0.3
            
                print('Estimated elasic constants: E=%5.2f GPa, nu=%4.2f' % (self.E/1000, self.nu))
                print('Estimated yield strength: %5.2f MPa, from %i data sets with PEEQ approx. %5.3f' 
                      %(self.sy,self.Ndat,db.epc))
                if "Prop_E" in self.meta:
                    print("Young's modulus provided in meta data:",self.meta['Prop_E'])
                if "Prop_nu" in self.meta:
                    print("Poisson ratio provided in meta data:", self.meta['Prop_nu'])
                if plot:
                    self.plot_set()

        def plot_set(self, file=None, nth=15, fontsize=18):
            '''Graphical output of equiv. stress vs. equic. total strain for selected load cases and 
            raw data in deviatoric cyl. stress space
            
            Parameters
            ----------
            file : str
                Write graph to pdf file (optional)
            nth  : int
                Plot every nth load case (optional, default: 18)
            fontsize : 20
                Fontsize for plot (optional, default: 20)
            '''
            cmap = plt.cm.get_cmap('viridis', 10)
            fig = plt.figure(figsize=(18,7))
            plt.subplots_adjust(wspace=0.2)
            N = len(self.load_case)
            ax = plt.subplot(1,2,1)
            for i in range(0,N,nth):
                ind = self.load_case[i]
                col = polar_ang(self.sig[ind[-1]])/np.pi
                plt.plot(eps_eq(self.eps[ind])*100, seq_J2(self.sig[ind]), color=cmap(col)) #'.k')
            plt.xlabel(r'$\epsilon_{eq}^\mathrm{tot}$ (%)', fontsize=fontsize)
            plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
            plt.title('Equiv. total strain vs. equiv. J2 stress', fontsize=fontsize)
            plt.tick_params(axis="x", labelsize=fontsize-4)
            plt.tick_params(axis="y", labelsize=fontsize-4)
            
            ax = plt.subplot(1, 2, 2) #, projection='polar')
            plt.plot(self.sfc_[self.i_pl_,1], self.sfc_[self.i_pl_,0], 'or')
            plt.plot(self.sfc_[self.i_el_,1], self.sfc_[self.i_el_,0], 'ob')
            plt.plot(self.syc[:,1], self.syc[:,0], '-k')
            plt.plot([-np.pi, np.pi], [self.sy, self.sy], '--k')
            plt.legend(['raw data above yield point', 'raw data below yield point', 
                           'interpolated yield strength', 'average yield strength'],loc=(1.04,0.8),fontsize=fontsize-2)
            plt.title('Raw data '+self.name, fontsize=fontsize)
            plt.xlabel(r'$\theta$ (rad)', fontsize=fontsize)
            plt.ylabel(r'$\sigma_{eq}$ (MPa)', fontsize=fontsize)
            plt.tick_params(axis="x", labelsize=fontsize-4)
            plt.tick_params(axis="y", labelsize=fontsize-4)
            if file is not None:
                plt.savefig(file+self.texture_name+'.pdf', format='pdf', dpi=300)
            plt.show()
    
    def plot_yield_locus(self, active, set_ind=0, scatter=False, data=None, data_label=None,
                         arrow=False, file=None, title=None, fontsize=18):
        '''Plot yield loci of imported microstructures in database.
        
        Parameters
        ----------
        active  : str
            'flow_stress', 'work_hard' or 'texture', selct which parameter to vary for set of plots
        set_ind : int
            select data set for work hardening plots (corresponds to level of plastic strain) (optional, default: 0)
        scatter : Boolean
            defines if raw data points are plotted (optional, default: False)
        data : (N,3) or (N,2) array
            additional data to plot in form of cylindrical stresses (optional, default: None)
        data_label : str
            label for legend of additional data (optional, default: None)
        arrow : Boolean
            indicate if arrows for principal stress directions are drawn (optional, default: False)
        file   : str
            filename for output (optional, default: None (no output))
        title  : str
            title for plot (optional, default: None)
        fontsize : int
            specifies fontsize used in plot (optional, default: 18)
        '''
        fig = plt.figure(figsize=(15, 8))
        cmap = plt.cm.get_cmap('viridis', 10)
        ms_max = np.amax(self.mat_param[active])
        Ndat = len(self.mat_param[active])
        v0 = self.mat_param[active][0]
        scale = self.mat_param[active][-1] - v0
        if np.abs(scale)<1.e-3:
            scale = 1.
            
        for i in range(Ndat):
            val = self.mat_param[active][i]
            hc = (val-v0)/scale
            if active=='work_hard':
                dset = self.set[set_ind]
                sc = self.mat_param['flow_stress'][set_ind,i,:,:]
                label = 'PEEQ: '+str(val.round(decimals=4))
                color = cmap(hc)
            elif active=='texture':
                dset = self.set[i]
                sc = self.mat_param['flow_stress'][i,0,:,:]
                label = dset.texture_name
                color = (hc,0,1-hc)
            elif active=='flow_stress':
                sc = self.mat_param['flow_stress'][0,0,:,:]
                plt.polar(sc[:,1], sc[:,0], '.r', label='shear')
                sc = data_RP.mat_param['flow_stress'][0,0,:,:]
            else:
                raise ValueError('Undefined value for active field in "plot_yield_locus"')
            if scatter:
                plt.polar(sc[:,1], sc[:,0], '.m', label=label)
            if data is not None:
                plt.polar(data[:,1], data[:,0], '.r', label=data_label)
            plt.polar(sc[:,1], sc[:,0], label=label, color=color)
            plt.legend(loc=(1.04,0.7),fontsize=fontsize-2)
        plt.tick_params(axis="x", labelsize=fontsize-4)
        plt.tick_params(axis="y", labelsize=fontsize-4)
        if arrow:
            dr = self.sy_av
            drh = 0.08*dr
            plt.arrow(0, 0, 0, dr, head_width=0.05, width=0.004, 
                     head_length=drh, color='r', length_includes_head=True)
            plt.text(-0.12, dr*0.87, '$\sigma_1$', color='r',fontsize=22)
            plt.arrow(2.0944, 0, 0, dr, head_width=0.05,
                     width=0.004, head_length=drh, color='r', length_includes_head=True)
            plt.text(2.26, dr*0.92, '$\sigma_2$', color='r',fontsize=22)
            plt.arrow(-2.0944, 0, 0, dr, head_width=0.05,
                     width=0.004, head_length=drh, color='r', length_includes_head=True)
            plt.text(-2.04, dr*0.95, '$\sigma_3$', color='r',fontsize=22)
        if title is not None:
            plt.title(title)
        if file is not None:
            plt.savefig(file+'.pdf', format='pdf', dpi=300)
        plt.show()