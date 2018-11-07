
class RedshiftCountHist:
    
    def __init__(self, aZBinEdges, aCountInBin):
        
        self.zBinEdges=copy.deepcopy(aZBinEdges)
        self.countInBin=copy.deepcopy(aCountInBin)
