import numpy as np
from scipy.interpolate import interp1d
from scipy import integrate
from scipy.stats import norm
from sphericosmo.cosmocontainer import *
from sphericosmo.sphericalpower import *

def SetupPiTau(piOption,zLimits,cosmoCont):
    
    zCurve=cosmoCont.zCurve
    tauCurve=cosmoCont.taus
        
    withinRange=getIndicesInRedshiftRange(zLimits, cosmoCont)
    
    if piOption==1:
        
        ###Option 1
        #Here we have a uniform dN/dz
        dzCurve=zCurve[1:]-zCurve[:-1]

        dzCentral=np.zeros(len(zCurve))

        for j in range(len(zCurve)):

            if j-1>=0 and j<len(dzCurve):

                dzCentral[j]=(dzCurve[j-1]+dzCurve[j])/2.0

        zRange=np.sum(dzCentral[withinRange])

        Pi_z=np.zeros(len(tauCurve))

        for j in withinRange:

            Pi_z[j]=1.0/zRange

        Pi_tau=Pi_z*np.gradient(zCurve,edge_order=2)/np.gradient(tauCurve,edge_order=2)

    elif piOption==2:
        
        ###Option 2
        #Here Pi_tau is used in place of r^2 n_C, but normalized to be integrated over tau
        Pi_tau=np.zeros(len(tauCurve))

        Pi_tau[withinRange]=1.0

        Pi_tau/=integrate.trapz((tauCurve[-1]-tauCurve)**2*Pi_tau,tauCurve)

        Pi_tau*=(tauCurve[-1]-tauCurve)**2

    elif piOption==3:
        
        ###Option 3
        #Here we use a pre-set dN/dz
        Pi_tau=dN_dzNorm(zCurve,zCutIndex)*np.gradient(zCurve,edge_order=2)/np.gradient(tauCurve,edge_order=2)

    elif piOption==4:
        
        ###Option 4
        #Here we use a Gaussian dN/dz
        mu=(zLimits[1]-zLimits[0])/2.0
        sigma=mu/3.0

        Pi_z=np.zeros(len(zCurve))
        Pi_z[withinRange]=norm.pdf(zCurve[withinRange], loc=mu, scale=sigma)

        Pi_z/=integrate.trapz(Pi_z[withinRange],zCurve[withinRange])

        Pi_tau=Pi_z*np.gradient(zCurve,edge_order=2)/np.gradient(tauCurve,edge_order=2)
    
    else:
        
        raise ValueError('Invalid piOption value provided. Valid options: 1,2,3,4')
        
    return Pi_tau


def SetupPiTauFromZHist(zBinLimits, countHist, cosmoCont):

    zCurve=cosmoCont.zCurve
    tauCurve=cosmoCont.taus
        
    zBinWidths=zBinLimits[1:]-zBinLimits[:-1]
    #zBinCenters=(zBinLimits[1:]+zBinLimits[:-1])/2.0
    
    #dN_dz = interp1d(zBinCenters, countHist/zBinWidths, kind='cubic', fill_value='extrapolate')
    dN_dz = interp1d(zBinLimits, np.concatenate([countHist/zBinWidths,[0.0]]), kind='previous', 
                     bounds_error=False, fill_value=0.0)
    
    withinRange=getIndicesInRedshiftRange([zBinLimits[0],zBinLimits[-1]], cosmoCont)

    Pi_z=np.zeros(len(zCurve))

    Pi_z[withinRange]=dN_dz(zCurve[withinRange])
    
    Pi_z/=integrate.trapz(Pi_z[withinRange],zCurve[withinRange])
    
    Pi_tau=Pi_z*np.gradient(zCurve,edge_order=2)/np.gradient(tauCurve,edge_order=2)
    
    return Pi_tau
    
    
def SetupPiTauForBinnedSN(piOption,zLimitList,cosmoCont):
    #Setup a Pi_tau that is normalized to 1 integral (over tau) in each bin of zLimitList
    
    zCurve=cosmoCont.zCurve
    tauCurve=cosmoCont.taus
    
    if piOption==1:
        ###Option 1
        #Here we have a uniform dN/dz
    
        Pi_z=np.zeros(len(tauCurve))
    
        for i in range(len(zLimitList)):

            withinRangeCurrentBin=getIndicesInRedshiftRange(zLimitList[i], cosmoCont)

            Pi_z[withinRangeCurrentBin[1:-1]]=1.0

            Pi_z[withinRangeCurrentBin]/=integrate.trapz(Pi_z[withinRangeCurrentBin],zCurve[withinRangeCurrentBin])

        Pi_tau=Pi_z*np.gradient(zCurve,edge_order=2)/np.gradient(tauCurve,edge_order=2)
        
    else:
        
        raise ValueError('Invalid piOption value provided. Valid options: 1')
        
    return Pi_tau

