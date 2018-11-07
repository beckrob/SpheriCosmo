
class TheoryClComputer:
    
    
    def __init__(self, aCosmoContainer, aLimberSwitch=30, aKRes=2400, aPiOption=1):
        #aPiOption=1 means assuming uniform dN/dz, aPiOption=2 means assuming uniform density when galaxy histogram is missing
        
        self.cosmoContainer=aCosmoContainer
        
        self.kRes=aKRes
        self.limberSwitch=aLimberSwitch
        
        self.piOption=aPiOption

    
    def computeCls(self, aLValues, aCorrType=None, aZLimits=None, aZCountHist=None, aLimberSwitch=None, aKRes=None, aFSky=1.0):
        #If the correlation is GX and there is a galaxy histogram, aZLimits will be ignored and taken instead from aGalHist
        #If the correlation is GX and there is no galaxy histogram, assume uniform density within aZLimits
        
        if aLimberSwitch is None:
            
            aLimberSwitch=self.limberSwitch
        
        if aKRes is None:
            
            aKRes=self.kRes  

        Pi_tau=None
        b_tau=None
        
        if 'G' in aCorrType:
            
            if aZCountHist is not None:
        
                aZLimits=[aZCountHist.zBinEdges[0],aZCountHist.zBinEdges[-1]]
            
                Pi_tau=SetupPiTauFromZHist(aZCountHist.zBinEdges, aZCountHist.countInBin, self.cosmoContainer)
                b_tau=np.ones(len(Pi_tau))
            
            elif aZLimits is not None:
                
                Pi_tau=SetupPiTau(self.piOption, aZLimits, self.cosmoContainer)
                b_tau=np.ones(len(Pi_tau))
        
        
        if aZLimits is None:
            
            raise ValueError('Error - provide z range for integral directly, or through the galaxy histogram')
        
        
        
        clVect=C_l_Switched(lLimitForLimber=aLimberSwitch, 
                            lVect=aLValues,
                            zLimits=aZLimits,
                            kLimits=self.cosmoContainer.kLimits,
                            kRes=aKRes,
                            cosmoCont=self.cosmoContainer,
                            corrType=aCorrType,
                            Pi=Pi_tau,
                            b=b_tau,
                            returnAsTauFunction=False)
        
        
        return ClContainer(aLVect=aLValues,
                           aClVect=clVect,
                           aClCovar=None,
                           aFSky=aFSky, 
                           aIsAutoCorr=(aCorrType[0]==aCorrType[1]))
   
