import numpy as np
import camb
from scipy import integrate
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from sphericosmo.constants import *

def growthFunc(zz,HH):
    # this is the factor to convert the scale-less analytical growth factor to real
    H_zMax_scaleLess = (2.0/3.0*(1.0+zz[-1])**(3.0/2.0))
    ff = 1.0/(HH[-1] / H_zMax_scaleLess )**3 #This is essentially t_0^3 in the EdS formula, fitted at the highest z

    arg_EdS = lambda x: 1.0/(1.0+x)**(7.0/2.0)
    D_zMax_inf_EdS_raw, err = integrate.quad(arg_EdS, zz[-1], np.inf)
    D_zMax_inf_EdS_raw = 27.0/8.0 * D_zMax_inf_EdS_raw

    #print(err)
    
    D_zMax_inf_EdS = ff*D_zMax_inf_EdS_raw

    D_0_z = integrate.cumtrapz(np.divide(1.0+zz,HH**3),zz,initial=0)
    D_0_9 = integrate.trapz(np.divide(1.0+zz,HH**3),zz)

    D1 = HH/HH[0] * \
        (D_0_9 + D_zMax_inf_EdS - D_0_z) \
        / (D_0_9 + D_zMax_inf_EdS)
        
    return D1

def growthFunc_EdS(zz,HH,zNorm,HNorm,DNorm):
    
    return HH/HNorm*(1+zz)**(-5.0/2.0)/((1+zNorm)**(-5.0/2.0))*DNorm


class CosmoContainer:
    
    
    def __init__(self, scaleFactorVect, tauVect, timeVect, HVect, D1, onePlusZD1_dtauVect, kVect, p_kVect, H0, omega_m, 
                 referenceRedshift=0.0, pointsToCMB=None):
        #referenceRedshift is the redshift at which 
        
        self.zCurve = 1.0/scaleFactorVect-1.0
        
        self.taus = tauVect #in Gyr
        
        self.times = timeVect #in Gyr
        
        self.D1 = D1
        self.onePlusZD1_dtauVect = onePlusZD1_dtauVect
        
        self.H=HVect
        
        h=H0/100.0
        self.H0_SI = H0*1000/MpcInMeters
        self.omega_m = omega_m

        
        z_CMB=1089.90
        C=scaleFactorVect[0]/self.times[0]**(2.0/3.0)
        t_CMB=(1.0/(z_CMB+1.0)/C)**(3.0/2.0)
        
        self.tau_CMB=3.0*t_CMB**(1.0/3.0)/C
        
        if pointsToCMB is not None:
            
            tauExtension=np.linspace(self.tau_CMB, self.taus[0], pointsToCMB+1)[:-1]
            
            timeExtension=(tauExtension*C/3.0)**3.0
            
            zExtension=1.0/(C*timeExtension**(2.0/3.0))-1.0

            HExtension=(1.0+zExtension)**(3.0/2.0)/((1.0+self.zCurve[0])**(3.0/2.0))*self.H[0]
            
            DExtension=growthFunc_EdS(zExtension, HExtension, self.zCurve[0], self.H[0], self.D1[0])
          
            DDerivExtension=np.zeros(len(DExtension))
            
            self.taus=np.concatenate([tauExtension,self.taus])  
            self.times=np.concatenate([timeExtension,self.times])
            self.zCurve=np.concatenate([zExtension,self.zCurve])
            self.H=np.concatenate([HExtension,self.H])
            self.D1=np.concatenate([DExtension,self.D1])
            self.onePlusZD1_dtauVect=np.concatenate([DDerivExtension,self.onePlusZD1_dtauVect])

        
        self.kLimits = [np.amin(kVect.astype(np.float64)*h),np.amax(kVect.astype(np.float64)*h)]

        referenceIndex=np.argmin(abs(self.zCurve-referenceRedshift))
        
        self.p_kFunc = interp1d(kVect.astype(np.float64)*h, 
                                p_kVect[0,:]*MpcInMeters**3*(self.D1[-1]/self.D1[referenceIndex])**2, 
                                kind='cubic', 
                                fill_value='extrapolate')

    def createFromCambResults(scaleFactorVect, cambResults, smoothingSigma, 
                              isLinear=True, referenceRedshift=0.0, pointsToCMB=None):
        #isLinear==False may require tuning of parameters
        
        zCurve=1.0/scaleFactorVect-1.0
        
        tauVect = cambResults.conformal_time(zCurve)/speedOfLightMpcGyr
        
        
        timeVect=np.empty(len(zCurve))
        for i in range(len(zCurve)):
            
            timeVect[i] = cambResults.physical_time(zCurve[i])
        
        
        hubbleFactors = cambResults.hubble_parameter(zCurve)

        D1_sim=np.flipud(growthFunc(np.flipud(zCurve),np.flipud(hubbleFactors)))

        onePlusZD1_dtau_sim=np.gradient((1.0/scaleFactorVect)*D1_sim,edge_order=2)/np.gradient(tauVect,edge_order=2)

        onePlusZD1_dtau_sim_smooth=gaussian_filter1d(onePlusZD1_dtau_sim,sigma=smoothingSigma, mode='nearest')
    
        if isLinear:
            
            kVect, zVect, p_kVect = cambResults.get_linear_matter_power_spectrum(hubble_units=False, have_power_spectra=True)
            
        else:
            
            kVect, zVect, p_kVect = cambResults.get_matter_power_spectrum(have_power_spectra=True, 
                                                                          minkh=2e-5, 
                                                                          maxkh=1500, 
                                                                          npoints = 500)
            
            p_kVect/=(cambResults.get_params().H0/100)**3
    
        return CosmoContainer(scaleFactorVect, tauVect, timeVect, hubbleFactors, D1_sim, onePlusZD1_dtau_sim_smooth, 
                              kVect, p_kVect,
                              cambResults.get_params().H0, 
                              cambResults.get_params().omegab+cambResults.get_params().omegac,
                              referenceRedshift, pointsToCMB)

