import numpy as np
import math
import copy

class MapContainer:
    
    def __init__(self, aMap, aMask, aBeam='NO', aObjectNum=None):  
            
        self.map=copy.deepcopy(aMap)
        self.mask=copy.deepcopy(aMask)
        
        self.beam=aBeam
        self.objectNum=aObjectNum
        
        healPixArea=4.0*math.pi/len(self.mask)

        self.fSky=np.sum(self.mask)*healPixArea/(4.0*math.pi)
        
        self.healpixRes=int(round(math.sqrt(len(self.map)/12.0)))
        

