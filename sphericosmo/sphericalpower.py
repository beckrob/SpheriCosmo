
def getIndicesInRedshiftRange(zLimits, cosmoCont):
    
    boundaryIndices=np.argmin(abs(cosmoCont.zCurve.reshape(-1,1)-zLimits),axis=0)
    boundaryIndices.sort()

    return np.arange(boundaryIndices[0],boundaryIndices[1]+1)

def BesselCurve_tau(l,k,cosmoCont):

    tauVect=cosmoCont.taus
    
    return spherical_jn(l,(tauVect[-1]-tauVect)*speedOfLightMpcGyr*k)


def G_ISW_tau(k,cosmoCont,bessel_tau):
    
    innerIntegrand=cosmoCont.onePlusZD1_dtauVect/GyrInSeconds*bessel_tau
        
    return -(3*(cosmoCont.H0_SI**2)/(speedOfLightSI**2)*cosmoCont.omega_m)*CMBTemp/(k**2)*(MpcInMeters**2)*innerIntegrand


def G_Gal_tau(Pi,b,cosmoCont,bessel_tau):

    innerIntegrand=b*Pi/GyrInSeconds*cosmoCont.D1*bessel_tau
    
    return innerIntegrand


def G_k_tau(cosmoCont,bessel_tau):
    
    tauVect=cosmoCont.taus
    tau_0=tauVect[-1]
    
    chi_SI=(tau_0-tauVect)*speedOfLightMpcGyr*MpcInMeters
    
    chi_CMB_SI=(tau_0-cosmoCont.tau_CMB)*speedOfLightMpcGyr*MpcInMeters


    chiTerm=(chi_CMB_SI-chi_SI)/chi_CMB_SI*chi_SI
    
    innerIntegrand=cosmoCont.D1*(cosmoCont.zCurve+1.0)*chiTerm*bessel_tau
    
    return (3.0/2.0*(cosmoCont.H0_SI**2)/(speedOfLightSI)*cosmoCont.omega_m)*innerIntegrand


def C_lIntegrand(k,cosmoCont,G1,G2):
    
    return (2.0/math.pi)*cosmoCont.p_kFunc(k)*(k**2)/(MpcInMeters**2)*G1*G2


def C_l_Bessel(lVect, zLimits, kLimits, kRes, cosmoCont, corrType='TT', Pi=None, b=None, returnAsTauFunction=False):

    kVect=np.logspace(np.log10(kLimits[0]),np.log10(kLimits[1]),kRes,base=10)

    withinRange=getIndicesInRedshiftRange(zLimits, cosmoCont)
    
    C_l=[]
    
    for l in lVect:

        outerIntegrand=[]

        for k in kVect:

            bessel_tau=BesselCurve_tau(l,k,cosmoCont)

            if corrType=='TT':

                G2_tau=G_ISW_tau(k,cosmoCont,bessel_tau)

                G1_tau=G2_tau[withinRange]

            elif corrType=='GG':

                G2_tau=G_Gal_tau(Pi,b,cosmoCont,bessel_tau)

                G1_tau=G2_tau[withinRange]
                
            elif corrType=='kk':

                G2_tau=G_k_tau(cosmoCont,bessel_tau)

                G1_tau=G2_tau[withinRange]

            elif corrType=='GT':

                G2_tau=G_ISW_tau(k,cosmoCont,bessel_tau)

                G1_tau=G_Gal_tau(Pi,b,cosmoCont,bessel_tau)[withinRange]       

            elif corrType=='Gk':

                G2_tau=G_k_tau(cosmoCont,bessel_tau)

                G1_tau=G_Gal_tau(Pi,b,cosmoCont,bessel_tau)[withinRange]    
                
            elif corrType=='Tk':

                G2_tau=G_k_tau(cosmoCont,bessel_tau)

                G1_tau=G_ISW_tau(k,cosmoCont,bessel_tau)[withinRange]
                
            else:

                raise ValueError('Invalid corrType value provided. Valid options: TT, GG, kk, GT, Gk and Tk')

            if returnAsTauFunction:
                G1=G1_tau           
            else:
                G1=integrate.trapz(G1_tau,cosmoCont.taus[withinRange]*GyrInSeconds)

            G2=integrate.trapz(G2_tau,cosmoCont.taus*GyrInSeconds)

            outerIntegrand.append(C_lIntegrand(k,cosmoCont,G1,G2))

        C_l.append(integrate.trapz(np.array(outerIntegrand),kVect/MpcInMeters,axis=0))
        
    return np.array(C_l)


def G_ISW_Limber_tau(l,cosmoCont):

    tauVect=cosmoCont.taus
    
    tau_0=tauVect[-1]
    
    kInv=((tau_0-tauVect)*speedOfLightMpcGyr)/(l+0.5)
    
    #The derivative by tau takes the dimension of tauVect
    return -((3*(cosmoCont.H0_SI**2)/(speedOfLightSI**2)*cosmoCont.omega_m)*CMBTemp*(kInv**2)*(MpcInMeters**2)*
             cosmoCont.onePlusZD1_dtauVect/GyrInSeconds)


def G_Gal_Limber_tau(Pi,b,cosmoCont):

    return b*Pi/GyrInSeconds*cosmoCont.D1


def G_k_Limber_tau(cosmoCont):
    
    tauVect=cosmoCont.taus
    tau_0=tauVect[-1]
    
    
    chi_SI=(tau_0-tauVect)*speedOfLightMpcGyr*MpcInMeters
    
    chi_CMB_SI=(tau_0-cosmoCont.tau_CMB)*speedOfLightMpcGyr*MpcInMeters


    chiTerm=(chi_CMB_SI-chi_SI)/chi_CMB_SI*chi_SI
    
    return ((3.0/2.0*(cosmoCont.H0_SI**2)/(speedOfLightSI)*cosmoCont.omega_m)*
             cosmoCont.D1*(cosmoCont.zCurve+1.0)*chiTerm)


def C_lIntegrand_Limber(l,cosmoCont,G1,G2):
    
    tauVect=cosmoCont.taus
    
    tau_0=tauVect[-1]
    
    chi=(tau_0-tauVect)*speedOfLightMpcGyr
    chi[-1]=1.0
    #This is to prevent division by zero. Knocked out by P_k[-1]=0.0 below
    
    P_k=np.empty(len(tauVect))
    
    P_k[:-1]=cosmoCont.p_kFunc( (l+0.5)/((tau_0-tauVect[:-1])*speedOfLightMpcGyr) )
    P_k[-1]=0.0
    
    return G1*G2/(chi**2)/(MpcInMeters**2)*P_k/speedOfLightSI


def C_l_Limber(lVect, zLimits, cosmoCont, corrType='TT', Pi=None, b=None, returnAsTauFunction=False):

    withinRange=getIndicesInRedshiftRange(zLimits, cosmoCont)
    
    C_l=[]
    
    for l in lVect:

        if corrType=='TT':

            G2_tau=G_ISW_Limber_tau(l,cosmoCont)

            G1_tau=G2_tau

        elif corrType=='GG':

            G2_tau=G_Gal_Limber_tau(Pi,b,cosmoCont)

            G1_tau=G2_tau
            
        elif corrType=='kk':

            G2_tau=G_k_Limber_tau(cosmoCont)

            G1_tau=G2_tau

        elif corrType=='GT':

            G2_tau=G_ISW_Limber_tau(l,cosmoCont)

            G1_tau=G_Gal_Limber_tau(Pi,b,cosmoCont)

        elif corrType=='Gk':

            G2_tau=G_k_Limber_tau(cosmoCont)

            G1_tau=G_Gal_Limber_tau(Pi,b,cosmoCont)
            
        elif corrType=='Tk':

            G2_tau=G_k_Limber_tau(cosmoCont)

            G1_tau=G_ISW_Limber_tau(l,cosmoCont)
            
        else:

            raise ValueError('Invalid corrType value provided. Valid options: TT, GG, kk, GT, Gk and Tk')

        
        C_l_tau=C_lIntegrand_Limber(l,cosmoCont,G1_tau,G2_tau)[withinRange]
            
        if returnAsTauFunction:
            C_l.append(C_l_tau)
        else:
            C_l.append(integrate.trapz(C_l_tau,cosmoCont.taus[withinRange]*GyrInSeconds,axis=0)) 
        
    return np.array(C_l)


def C_l_Switched(lLimitForLimber,lVect, zLimits, kLimits, kRes, cosmoCont, corrType='TT', 
                 Pi=None, b=None, returnAsTauFunction=False):
    
    lVect_Bessel=[]
    lVect_Limber=[]
    
    for l in lVect:
        
        if l>=lLimitForLimber:
            lVect_Limber.append(l)
        else:
            lVect_Bessel.append(l)
    
    C_l=None
    
    if len(lVect_Bessel)>0:
        
        C_l=C_l_Bessel(lVect_Bessel, zLimits, kLimits, kRes, cosmoCont, corrType, Pi, b, returnAsTauFunction)
    
     
    if len(lVect_Limber)>0:
    
        C_l_2=C_l_Limber(lVect_Limber, zLimits, cosmoCont, corrType, Pi, b, returnAsTauFunction)
    
        if C_l is None:
            
            C_l=C_l_2
            
        else:
            
            C_l=np.concatenate((C_l,C_l_2))

    return C_l
