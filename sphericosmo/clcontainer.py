import numpy as np
import math
import copy


class ClContainer:
    
    
    def __init__(self, aLVect, aClVect, aClCovar=None, aLBands=None, aWeightBinByLs=False, aFSky=1.0, aIsAutoCorr=False):
        
        self.l=copy.deepcopy(aLVect)
        self.cl=copy.deepcopy(aClVect)
        self.clCovar=copy.deepcopy(aClCovar)
        
        self.lBands=copy.deepcopy(aLBands)
        
        self.fSky=aFSky
        self.isAutoCorr=aIsAutoCorr
        
        if aLBands is not None:
            
            self.applyBinning(aLBands, aWeightBinByLs)
            
        else:
            
            self.l_band=None
            self.cl_band=None
            self.deltaCl_band=None
            
        
    def applyBinning(self, aLBands, aWeightBinByLs, aFSky=None):
        
        if aFSky is not None:
            
            self.fSky=aFSky
        
        self.lBands=copy.deepcopy(aLBands)
        
        self.l_band=np.empty(len(aLBands))
        self.cl_band=np.empty(len(aLBands))
        self.deltaCl_band=np.empty(len(aLBands))
        
        for i in range(len(aLBands)):
            
            binSlice=self.getBinSlice(aLBands[i])
                          
            if aWeightBinByLs:
                
                modeWeightVect=2*self.l[binSlice]+1
               
            else:
                
                modeWeightVect=np.ones(len(self.l[binSlice]))
            
            modeWeightNorm=np.sum(modeWeightVect)

            self.l_band[i]=np.sum(self.l[binSlice]*modeWeightVect)/modeWeightNorm
            
            self.cl_band[i]=np.sum(self.cl[binSlice]*modeWeightVect)/modeWeightNorm
          
            if self.clCovar is None:
                #No covariance matrix: the error is assumed to be cosmic variance limited
                #Thus, it is C_l/sqrt(N) where N is the number of multipoles in the bin

                if self.isAutoCorr:
                    multipoleFactor=0.5
                else:
                    multipoleFactor=1.0

                totalModeNum=self.fSky*multipoleFactor*np.sum(2*self.l[binSlice]+1)

                self.deltaCl_band[i]=self.cl_band[i]/math.sqrt(totalModeNum)

            else:
                #Computing the block average of the provided covariance matrix

                modeWeightMatrix=np.outer(modeWeightVect,modeWeightVect)

                self.deltaCl_band[i]=math.sqrt(np.sum(modeWeightMatrix*self.clCovar[binSlice,binSlice])
                                               /modeWeightNorm**2)

        return
            
        
    def getBinSlice(self, aLBand):
        #Get the slice object for this container that corresponds to a given l range
        #Will throw exception if none of the ls are within coverage
        
        sliceStart=np.where(self.l>=aLBand[0])[0][0]
        
        sliceEnd=np.where(self.l<=aLBand[1])[0][-1]
        
        if self.l[sliceEnd]<aLBand[1]:
            sliceEnd+=1
        
        return slice(sliceStart,sliceEnd)
    

    def getCl(self, aL):
        
        match=np.where(self.l==aL)[0]
        
        if len(match)>0:
            
            return self.cl[match[0]]
        
        else:
            
            return None

