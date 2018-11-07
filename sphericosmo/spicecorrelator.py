
def defaultApodization(fSky, multiplier=1.0):

    return sqrt(41252.96*fSky)*multiplier

class SpiceCorrelator:
    
    def __init__(self, aOutdir, aMaxL, aApodizeFunction=defaultApodization, aThetaMaxFunction=defaultApodization,
                 aSubAverage='YES', aSubDipole='YES'):

        self.apodizeFunction=aApodizeFunction
        self.thetaMaxFunction=aThetaMaxFunction
    
        self.outDir=aOutdir
        
        self.maxL=aMaxL
        
        self.subAverage=aSubAverage
        self.subDipole=aSubDipole
        
        
    def computeCls(self, aMapCont1, aMapCont2=None, aLabel='', 
                   aKeepInput=False, aKeepOutput=True, aGetCovar=True, aGetCor=False):
        #The correlator does not store maps and masks in memory, just writes them to files
        #The label can be used to separate
        #If two maps are given, they are assumed to be different maps (i.e. cross-correlation)
        
        hp.fitsfunc.write_map(self.outDir+'/'+aLabel+'_map1.fit', aMapCont1.map, 
                              dtype=np.float64, fits_IDL=False, coord='C', overwrite=True)

        hp.fitsfunc.write_map(self.outDir+'/'+aLabel+'_mask1.fit', aMapCont1.mask, 
                              dtype=np.float64, fits_IDL=False, coord='C', overwrite=True)
        
        
        if aGetCovar:
            
            covOutputPath=self.outDir+'/'+aLabel+'_Covar.fit'
            
        else:
            
            covOutputPath=''
            
            
        if aGetCor:
            
            corOutputPath=self.outDir+'/'+aLabel+'_Cor.fit'
            
        else:
            
            corOutputPath=''
        
        
        if aMapCont2 is not None:
            
            autoCorr=False
            fSky=min(aMapCont1.fSky, aMapCont2.fSky)
            
            hp.fitsfunc.write_map(self.outDir+'/'+aLabel+'_map2.fit', aMapCont2.map, 
                                  dtype=np.float64, fits_IDL=False, coord='C', overwrite=True)

            hp.fitsfunc.write_map(self.outDir+'/'+aLabel+'_mask2.fit', aMapCont2.mask, 
                                  dtype=np.float64, fits_IDL=False, coord='C', overwrite=True)
            
        
            ispice.ispice(mapin1=self.outDir+'/'+aLabel+'_map1.fit',
                          clout=self.outDir+'/'+aLabel+'_Cl.fit',
                          maskfile1=self.outDir+'/'+aLabel+'_mask1.fit',
                          mapfile2=self.outDir+'/'+aLabel+'_map2.fit',
                          maskfile2=self.outDir+'/'+aLabel+'_mask2.fit',
                          covfileout=covOutputPath,
                          apodizesigma=min(self.apodizeFunction(aMapCont1.fSky),self.apodizeFunction(aMapCont2.fSky)),
                          thetamax=min(self.thetaMaxFunction(aMapCont1.fSky),self.thetaMaxFunction(aMapCont2.fSky)),
                          beam1=aMapCont1.beam,
                          beam2=aMapCont2.beam,
                          corfile=corOutputPath,
                          nlmax=self.maxL,
                          subav=self.subAverage, 
                          subdipole=self.subDipole)
            
        else:
            
            autoCorr=True
            fSky=aMapCont1.fSky
            
            ispice.ispice(mapin1=self.outDir+'/'+aLabel+'_map1.fit',
                          clout=self.outDir+'/'+aLabel+'_Cl.fit',
                          maskfile1=self.outDir+'/'+aLabel+'_mask1.fit',
                          covfileout=covOutputPath,
                          apodizesigma=self.apodizeFunction(aMapCont1.fSky),
                          thetamax=self.thetaMaxFunction(aMapCont1.fSky),
                          beam1=aMapCont1.beam,
                          corfile=corOutputPath,
                          nlmax=self.maxL,
                          subav=self.subAverage, 
                          subdipole=self.subDipole)
        
        
        
        SpiceCorrelator.fileCleanUp(self.outDir, aLabel, aKeepInput=aKeepInput, aKeepOutput=True)

        
        return SpiceCorrelator.readClOutput(self.outDir, aLabel, aKeepOutput, fSky, autoCorr, aGetCovar)
    

    def computeClsRandomMap(self, aMapCont1, aRandomMapClCont, aRandomMapMask, aRandomMapBeam='NO', aLabel='', 
                            aKeepInput=False, aKeepOutput=True, aGetCovar=True, aGetCor=False):
        #Assuming that the random map should have the same resolution as the fixed map
        
        randomCl_padded=SpiceCorrelator.getPaddedClForRandomMap(aRandomMapClCont)
        
        randomMapInstance=hp.sphtfunc.synfast(cls=randomCl_padded, nside=aMapCont1.healpixRes, new=True, verbose=False)
    
        randomMapCont=MapContainer(randomMapInstance, aRandomMapMask, aRandomMapBeam)
    
        return self.computeCls(aMapCont1, randomMapCont, aLabel=aLabel, 
                               aKeepInput=aKeepInput, aKeepOutput=aKeepOutput, aGetCovar=aGetCovar, aGetCor=aGetCor)

    
    
    def covarianceMonteCarlo(self, aSampleNum, aRandomMapClCont, aRandomMapMask, 
                             aFixedMapList, aLBandListForFixedMap, 
                             aWeightBinByLs=False, aRandomMapBeam='NO'):
        
        
        randomCl_padded=SpiceCorrelator.getPaddedClForRandomMap(aRandomMapClCont)       
        
        maxHealpixRes=-1
        
        for fixedMap in aFixedMapList:
        
            maxHealpixRes=max(maxHealpixRes,fixedMap.healpixRes)
        

        clMatrix=np.zeros([aSampleNum, SpiceCorrelator.getClVectLength(aLBandListForFixedMap)])
            
        clIndexesForMap=SpiceCorrelator.getClVectIndexList(aLBandListForFixedMap)
                          
    
        for i in range(aSampleNum):

            randomMapInstance=hp.sphtfunc.synfast(cls=randomCl_padded, nside=maxHealpixRes, new=True, verbose=False)
            
            randomMapCont=MapContainer(randomMapInstance, aRandomMapMask, aRandomMapBeam)
            
            for j in range(len(aFixedMapList)):
                
                clRes=self.computeCls(randomMapCont, aFixedMapList[j], aLabel='MC', 
                                      aKeepInput=False, aKeepOutput=False, aGetCovar=False, aGetCor=False)
                
                clRes.applyBinning(aLBandListForFixedMap[j], aWeightBinByLs)
                
                clMatrix[i, clIndexesForMap[j]]=clRes.cl_band

        return np.cov(clMatrix, rowvar=False)
    
    
    def computeTemplateSubtractionCls(self, aTracerMap, aTargetMap, aTemplateMapList):
        
        clTracerTemplateList=[]
        
        clTemplateTemplateList=[]
        
        clTargetTemplateList=[]
        
        for templateMap in aTemplateMapList:
        
            clTracerTemplateList.append(self.computeCls(aTracerMap, templateMap, aLabel='template', 
                                        aKeepInput=False, aKeepOutput=False, aGetCovar=True, aGetCor=False))
            
            clTemplateTemplateList.append(self.computeCls(templateMap, None, aLabel='template', 
                                          aKeepInput=False, aKeepOutput=False, aGetCovar=True, aGetCor=False))
      
            clTargetTemplateList.append(self.computeCls(aTargetMap, templateMap, aLabel='template', 
                                        aKeepInput=False, aKeepOutput=False, aGetCovar=True, aGetCor=False))
    
        return (clTracerTemplateList, clTemplateTemplateList, clTargetTemplateList)
        
    def readClOutput(aOutdir, aLabel='', aKeepOutput=True, aFSky=1.0, aIsAutoCorr=False, aGetCovar=True):
        

        clFrame=Table.read(aOutdir+'/'+aLabel+'_Cl.fit').to_pandas()

        if aGetCovar:
            
            covarMatrix=fits.open(aOutdir+'/'+aLabel+'_Covar.fit')[0].data[0]
            
        else:
            
            covarMatrix=None
        
        SpiceCorrelator.fileCleanUp(aOutdir, aLabel, aKeepInput=True, aKeepOutput=aKeepOutput)
            
            
        return ClContainer(aLVect=clFrame.index.values, 
                           aClVect=clFrame.TT.values,
                           aClCovar=covarMatrix,
                           aFSky=aFSky, 
                           aIsAutoCorr=aIsAutoCorr)
        
        
        
    def fileCleanUp(aOutdir, aLabel='', aKeepInput=False, aKeepOutput=False):
        
        if not aKeepInput:
        
            inputTags=['map1','mask1','map2','mask2']
        
            for inputTag in inputTags:
            
                if os.path.exists(aOutdir+'/'+aLabel+'_'+inputTag+'.fit'):

                    os.remove(aOutdir+'/'+aLabel+'_'+inputTag+'.fit')
                
        
        if not aKeepOutput:
            
            outputTags=['Cl','Covar','Cor']
            
            for outputTag in outputTags:
            
                if os.path.exists(aOutdir+'/'+aLabel+'_'+outputTag+'.fit'):

                    os.remove(aOutdir+'/'+aLabel+'_'+outputTag+'.fit')
        
        return

    
    def getClVectLength(aLBandListForMap):
        
        return np.sum([len(lBandList) for lBandList in aLBandListForMap])
    
    
    def getIndexInClVect(aBinID, aMapID, aLBandListForMap):
        
        if aMapID>=len(aLBandListForMap):
            
            raise ValueError('Error - aMapID does not exist in list')

        if aBinID>=len(aLBandListForMap[aMapID]):
            
            raise ValueError('Error - aBinID does not exist for the given aMapID')
            
        #First, count how many total bins there are whose ID is smaller that aBinID
        numSmallerBins=np.sum([np.sum(np.arange(len(lBandList))<aBinID) for lBandList in aLBandListForMap])
        
        #Then, count how many maps with smaller aMapID have a bin with aBinID 
        numThisBin=np.sum([len(lBandList)>aBinID for lBandList in aLBandListForMap[0:aMapID]])
        
        return int(numSmallerBins+numThisBin)
    
           
    def getClVectIndexList(aLBandListForMap):
    
        clIndexesForMap=[]
        
        for j in range(len(aLBandListForMap)):
            
            ids=[]
            
            for i in range(len(aLBandListForMap[j])):
            
                ids.append(SpiceCorrelator.getIndexInClVect(i,j,aLBandListForMap))
            
            clIndexesForMap.append(ids)   
            
        return clIndexesForMap
    
    
    def getClVectIndexesForBand(aLBandListForMap):
            
        bandNum=int(np.amax([len(lBandList) for lBandList in aLBandListForMap]))
        
        clIndexesForBand=[]
        counter=0
        
        for i in range(bandNum):
            
            ids=[]
            
            for j in range(len(aLBandListForMap)):
            
                if i<len(aLBandListForMap[j]):
                    
                    ids.append(counter)
                    counter+=1
                    
            clIndexesForBand.append(ids)

        return clIndexesForBand
    
        
    def getModeNumForBand(aLBandListForMap):

        bandNum=int(np.amax([len(lBandList) for lBandList in aLBandListForMap]))
        
        modeNumForBand=[]
        
        for lBandList in aLBandListForMap:
        
            if len(lBandList)==bandNum:
            
                for lBand in lBandList:
        
                    lVect=np.arange(lBand[0],lBand[1])
            
                    modeNumForBand.append(np.sum(2*lVect+1))
            
                break
            
        return modeNumForBand
        
    def getPaddedClForRandomMap(aRandomMapClCont):
        
        randomCl_padded=[]

        for l in np.arange(0,np.amax(aRandomMapClCont.l)+1):

            Cl=aRandomMapClCont.getCl(l)

            if Cl is not None:

                randomCl_padded.append(Cl)

            else:

                randomCl_padded.append(0.0)

        randomCl_padded=np.array(randomCl_padded) 
        
        return randomCl_padded
        