
class FisherFitter:
    
    def getEmpiricalCovarianceMatrix(aClTTMeas,
                                     aClGGMeasForZBinPair, aClGGTheorForZBinPair, aBiasForZBin, 
                                     aLBandListForZBin, aFSkyForZBin, aObjectNumForZBin,
                                     aMinF1F2, aRescaleTheory):
        
        
        dim=SpiceCorrelator.getClVectLength(aLBandListForZBin)
        
        ClGGMatMeas=np.zeros([dim,dim])
        
        ClGGVectTheor=np.zeros([dim])
        ClGGVectMeas=np.zeros([dim])
        
        fSkyVect=np.zeros(dim)
        
        clIndexesForZBin=SpiceCorrelator.getClVectIndexList(aLBandListForZBin)
        
        sigma=np.zeros([dim,dim])
        
        for i in range(len(aBiasForZBin)):
        
            ClGGVectTheor[clIndexesForZBin[i]]=aClGGTheorForZBinPair[i,i].cl_band*aBiasForZBin[i]**2+ \
                                                4*math.pi*aFSkyForZBin[i]/aObjectNumForZBin[i]
                
            ClGGVectMeas[clIndexesForZBin[i]]=aClGGMeasForZBinPair[i,i].cl_band
            
            fSkyVect[clIndexesForZBin[i]]=aFSkyForZBin[i]
        
            for j in range(len(aBiasForZBin)):
                    
                ClGGMatMeas[clIndexesForZBin[i],clIndexesForZBin[j]]=aClGGMeasForZBinPair[i,j].cl_band
            
            
            
            
        clIndexesForBand=SpiceCorrelator.getClVectIndexesForBand(aLBandListForZBin)
        
        modeNumForBand=SpiceCorrelator.getModeNumForBand(aLBandListForZBin)
        
        for b in range(len(clIndexesForBand)):
            
            clIndexes=clIndexesForBand[b]
            
            if aMinF1F2:
                
                fMatrix=np.repeat(fSkyVect[clIndexes].reshape(-1,1), len(clIndexes), axis=1)
                
                fMatrix=np.minimum(fMatrix, fMatrix.T)
                
            else:
                
                fMatrix=np.sqrt(np.outer(fSkyVect[clIndexes], fSkyVect[clIndexes]))
            
            
            if aRescaleTheory:
            
                scaleMatrix=np.sqrt(np.outer(ClGGVectTheor[clIndexes], ClGGVectTheor[clIndexes])) \
                            /np.sqrt(np.outer(ClGGVectMeas[clIndexes], ClGGVectMeas[clIndexes]))
                    
            else:
                
                scaleMatrix=1.0          
                    
            sigma[np.ix_(clIndexes,clIndexes)]=aClTTMeas.cl_band[b]*ClGGMatMeas[np.ix_(clIndexes,clIndexes)]*scaleMatrix \
                                                /modeNumForBand[b]/fMatrix
        
        
        return sigma
        
        
    
    def fitAFromGTCorrelation(aClMeasForZBin, aClTheorForZBin, aBiasForZBin, aLBandListForZBin, aCovarMatrix):
        
        ClVectMeas=np.zeros(SpiceCorrelator.getClVectLength(aLBandListForZBin))
        ClVectTheor=np.zeros(len(ClVectMeas))
        
        clIndexesForZBin=SpiceCorrelator.getClVectIndexList(aLBandListForZBin)
        
        for i in range(len(aClMeasForZBin)):
        
            ClVectMeas[clIndexesForZBin[i]]=aClMeasForZBin[i].cl_band
            ClVectTheor[clIndexesForZBin[i]]=aClTheorForZBin[i].cl_band*aBiasForZBin[i]
        
        sigmaInv=np.linalg.inv(aCovarMatrix)
        
        Fisher=np.dot(np.dot(ClVectTheor,sigmaInv),ClVectTheor)
        
        A=np.dot(np.dot(ClVectMeas,sigmaInv),ClVectTheor)/Fisher
        
        AErr=1.0/sqrt(Fisher)
        
        return (A, AErr)

