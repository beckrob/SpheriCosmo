import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from functools import partial
import inspect
import copy
from sphericosmo.clcontainer import *

class ClFitter:


    
    def fitClBandsToTheory(aFitFunction, aClTheory, aClMeasured, **kwargs):
        #fitFunction must have the following first two positional arguments: l, Cl_theory - these are not fitted
        #The rest of the positional arguments are fitted
        #Additional keyword-type arguments can also be provided, these also won't be fitted
        #The binning should be the same in theory and measured - linear intepolation/extrapolation in l will be used otherwise
        
        def interpolationHelper(aL, *args, aClTheory, aFitFunction, **kwargs):

            interpFunc = interp1d(aClTheory.l_band, 
                                  aFitFunction(aClTheory.l_band, aClTheory.cl_band, *args, **kwargs), 
                                  kind='linear', fill_value='extrapolate')

            return interpFunc(aL)
        
        keyWordParams=set(kwargs.keys())
        allParams=set(list(inspect.signature(aFitFunction).parameters.keys())[2:])
             
        fittedParams=list(allParams.difference(keyWordParams))
        

        (params, paramCovar)=curve_fit(f=partial(interpolationHelper, aClTheory=aClTheory, aFitFunction=aFitFunction,
                                                 **kwargs),
                                       xdata=aClMeasured.l_band,
                                       ydata=aClMeasured.cl_band,
                                       sigma=aClMeasured.deltaCl_band,
                                       p0=np.ones(len(fittedParams)))

        paramErrors=np.sqrt(np.diag(paramCovar))

        return (params,paramErrors,paramCovar)


    def fitTemplateSubtractedCl(aClMeasured, aClMapTemplateList, aClTemplateTemplateList, aClTargetTemplateList,
                                aReturnLambdas=False):
        
        def constantFunc(x,C):

            return C

        lambdaValues=[]
        lambdaErrors=[]
        
        for i in range(len(aClMapTemplateList)):

            yDataVect=aClMapTemplateList[i].cl_band/aClTemplateTemplateList[i].cl_band

            sigmaVect=np.abs(yDataVect)*np.sqrt(
                                    (aClMapTemplateList[i].deltaCl_band/aClMapTemplateList[i].cl_band)**2+
                                    (aClTemplateTemplateList[i].deltaCl_band/aClTemplateTemplateList[i].cl_band)**2)


            (params, paramCovar)=curve_fit(f=constantFunc,
                                           xdata=aClMapTemplateList[i].l_band,
                                           ydata=yDataVect,
                                           sigma=sigmaVect)
            
            paramErrors=np.sqrt(np.diag(paramCovar))
            
            lambdaValues.append(params[0])
            lambdaErrors.append(paramErrors[0])
        
        ClFiltered=copy.deepcopy(aClMeasured)

        for i in range(len(aClMapTemplateList)):

            templateSignal=lambdaValues[i]*aClTargetTemplateList[i].cl_band

            templateSignalErr=np.abs(templateSignal)*np.sqrt(
                                    (aClTargetTemplateList[i].deltaCl_band/aClTargetTemplateList[i].cl_band)**2+
                                    (lambdaErrors[i]/lambdaValues[i])**2)


            ClFiltered.cl_band-=templateSignal

            ClFiltered.deltaCl_band=np.sqrt(ClFiltered.deltaCl_band**2+templateSignalErr**2)

        if aReturnLambdas:
            
            return (ClFiltered, lambdaValues, lambdaErrors)
        
        else:
            
            return ClFiltered

