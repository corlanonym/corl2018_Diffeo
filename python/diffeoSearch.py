from copy import deepcopy

from coreUtils import *
import diffeoUtils as du

from scipy.linalg import norm as deformationNorm

def getAddTransformations(thisTrans:du.pd.localPolyMultiTranslation, xSource:np.ndarray, xTarget:np.ndarray, xCurrentOld:np.ndarray,
                         maxOkErr:float=0.01, minBase:float=1e-3, divCoef:float=2., dValue:float=1./6., safeFac:float=0.3,
                         margins:float=[0.6,0.2], influenceFac:float=0.3, baseRegVal:float=1e-3, dir:bool=True, maxKernel=20, minTrans=-0.005, dist2Origin=0.01):
    assert dir, 'Inverse TBD'

    print(thisTrans.dist2nondiffeo(marginSingle=margins[0],marginOverlap=margins[1]))
    try:
        assert np.all(thisTrans.dist2nondiffeo(marginSingle=margins[0],marginOverlap=margins[1]) > 0)
    except:
        print('Fail')
        assert 0

    if minTrans < 0:
        minTrans = -np.max(cNorm(xSource, False))*minTrans

    dim,nPt = xSource.shape
    
    if thisTrans.nTrans==0:
        xCurrent = xSource.copy()
    else:
        xCurrent = thisTrans.forwardTransform(xSource)

    xErr = xTarget-xCurrent
    dx = zeros(xErr.shape)

    #indOk = ones((nPt,)).astype(np.bool_)
    indOk = 1-thisTrans.approxMaxDistortion(xSource, dir=True)-margins[1]>0.5*safeFac #Dir is False because thisTrans has already been applied to xCurrent
    
    divCoefO = deepcopy(divCoef)
    
    #Get the desired center and translation of the transformation
    while thisTrans.nTrans<maxKernel:
        dbgCounter = 0
        divCoef = deepcopy(divCoefO)
        # Now we have to test if with bestBase the whole trans is still a guaranteed to be a diffeo
        try:
            print(thisTrans.dist2nondiffeo(marginSingle=margins[0],marginOverlap=margins[1]))
            assert np.all(thisTrans.dist2nondiffeo(marginSingle=margins[0],marginOverlap=margins[1]) > 0)
        except:
            pass
        print(thisTrans.nTrans)
        print('A')
        #Loop over kernels in one multitrans
        xErrn = cNorm(xErr,kd=False)


        while True:
            print('B')
            
            xErrnOk = xErrn[indOk]
            xErrOk = xErr[:,indOk]
            xCurrentOk = xCurrent[:,indOk]
            xSourceOk = xSource[:,indOk]
            if (not np.any(indOk)) or np.all(xErrnOk <= maxOkErr):
                return thisTrans,dx
            
            if dbgCounter>10000:
                print('help')
            dbgCounter+=1
            #Search for the next kernel to be added by iterating over the possibilities
            indMax = np.argmax(xErrnOk)
            if callable(divCoef):
                trans = xErrOk[:,[indMax]]/divCoef(cNorm(xErrOk[:,[indMax]], False))
            else:
                trans = xErrOk[:,[indMax]]/divCoef
            #center = xCurrentOk[:,[indMax]]+0.01*trans
            #Get the preimage of center
            #center = thisTrans.inverseTransform(center)
            center = xSourceOk[:,[indMax]]+0.01*trans
            
            #Get the largest base so that xCurrentOld is untouched
            if xCurrentOld is None:
                maxBase = 2.*np.max(cNorm(xCurrent,False))
            else:
                maxBase = np.min(cNorm(xCurrentOld-center, kd=False))
            #Additional constraints
            maxBase = min(na([float(maxBase), float(cNorm(center,kd=False))-dist2Origin]))
            
            
            minBaseR = max(minBase, du.pd.c1PolyMinB(cNorm(trans,False), dValue, safeFac, e=0.))
            if minBaseR < maxBase:
                break
            else:
                #Increase divCoef
                if minBaseR < 1.25*minBase:
                    #and exclude  some points if really tiny
                    indOk = np.logical_and( indOk, cNorm(xSource-xSourceOk[:,[indMax]], kd=False)>=1.1*maxBase )
                try:
                    divCoef*=1.25
                except TypeError:
                    divCoef = max(divCoef(cNorm(xErrOk[:,[indMax]], False))*1.25, 1.25)
                if not np.any(indOk):
                    #No more sensible centers available
                    return thisTrans, dx
        # Check trans size
        if cNorm(trans,False) < minTrans:
            # and exclude  some points around
            indOk = np.logical_and(indOk,cNorm(xSource-xSourceOk[:,[indMax]], kd=False) >= 1.2*minBaseR)
            continue
        # We got a new candidate center/translation
        # Search for best base
        fCost = lambda newBase: deformationNorm(xErr - du.pd.c1PolyTrans(xSource, center, trans, float(newBase), float(dValue) ))+baseRegVal*newBase # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
        bestBase, error, ok, _dummy = fminbound(fCost, minBaseR, maxBase, full_output=True)
        
        thisTrans.addTransition(center, trans, bestBase, dValue, 0.)
        print("base {0}".format(bestBase))

        #Ensure the resulting transformation stays a diffeo
        isRemoved = False
        while True:
            
            currentDist, isOverlap = thisTrans.dist2nondiffeo(marginSingle=margins[0],marginOverlap=margins[1], fullOut=True)
            isOverlap = isOverlap
            isNotOverlap = np.logical_not(isOverlap)
            isOverlap = isOverlap.astype(np.float_)
            isNotOverlap = isNotOverlap.astype(np.float_)
            
            if np.all(currentDist>0.):
                #The transformation is diffeomorphic
                break
            
            if cNorm(trans,kd=False)<minTrans:
                #Erase the transformation if the translation is too small
                thisTrans.removeTransition(thisTrans.nTrans-1)
                isRemoved = True
                break
            #reduce the norm of the translation that cause the non-diffeomorphic behaviour
            
            for k in range(thisTrans.nTrans):
                isNOk = currentDist[k]<=0.
                if not isNOk:
                    continue
                # reduce the norm of the transitions that overlap and violate the constraints
                thisTrans._translations *= (0.95*isOverlap[[k],:] + isNotOverlap[[k],:])
                # Avoid multiple reduction in one shot
                for l in range(k, thisTrans.nTrans):
                    if isOverlap[k,l]:
                        currentDist[l] =1.

        #Exclude the points too close to the new kernel
        #and update dx
        if not isRemoved:
            thisDX, thisInfl = du.pd.c1PolyTrans(xSource,center,trans,float(bestBase),dValue, gammaOut=True)
            dx += thisDX
            xErr -= thisDX
            indOk = np.logical_and(indOk, thisInfl<influenceFac)
        else:
            indOk = np.logical_and(indOk,cNorm(xSource-xSourceOk[:,[indMax]],kd=False) >= 1.2*minBaseR)
    
    #assert np.all(cNorm(thisTrans._translations)>=minTrans)
    
    print("Attained max kernel number")
    return thisTrans

def matchInfeasible(xSource:np.ndarray, xTarget:np.ndarray,
                    xSourceOld: np.ndarray = None,xTargetOld: np.ndarray = None, currentDiffeo:du.diffeomorphism=None,
                    depthMax:int=30, maxErr:list=[5e-2,5e-3], isPars=[True,True,True,False,False], mseFac=6., diffeoOpts={}, optimizeOpts={}):

    diffeoOpts = deepcopy(diffeoOpts)
    optimizeOpts = deepcopy(optimizeOpts)

    # Options concerning optimization
    optimizeOpts.setdefault('doOpt', True) # Do a gradient descent minimization where the cost is the MSE
    optimizeOpts.setdefault('maxIter', 20) # Maximal number of iteration of the gradient descent
    optimizeOpts.setdefault('relTol', 0.001) # Relative error: when the improvement between two iterations is smaller (in a relative manner) than the value the optimization is terminated
    optimizeOpts.setdefault('absTol', -0.05) # Absolute error: When the MSE is smaller than the threshold, the optimization is terminated. If negative than the value is multiplied with the initial error
    optimizeOpts.setdefault('alphaConstraints',-.05)  # The margins to "non-diffeo" will be increased by epsConstraint
    optimizeOpts.setdefault('epsConstraints', 1e-2) # The margins to "non-diffeo" will be increased by epsConstraint
    optimizeOpts.setdefault('tau',.5) # Parameter of the backtrackstep search
    optimizeOpts.setdefault('c',1e-2) # Parameter of the backtrackstep search


    # Options concerning the diffeomorphism
    diffeoOpts.setdefault('minBase',float(0.1*np.max(cNorm(xSource,kd=False)))) # Float minimal size of the base of a transformation
    #diffeoOpts.setdefault('divCoef',2.5)
    diffeoOpts.setdefault('divCoef',lambda tn: 0.9+2./(1+exp(-1.0*(1.5*tn-6.)))) # Float or lambda-Func division coefficient -> Computes the translation based on initial error; Idea: Large error needs a large region of influence
    diffeoOpts.setdefault('dValue',1./6.) # Float ]0.,0.5[ Defines the "size" of region of constant acceleration
    diffeoOpts.setdefault('safeFac', 0.5) # Float safety coefficient. Enlarges base
    diffeoOpts.setdefault('margins', [0.35,0.05]) # List[Float] minimal distance to "non-diffeo" [0]: Distance for each transformation [1]: Distance for worst case cumullative
    diffeoOpts.setdefault('influenceFac',0.3) # Maximal cumulative influence on a point to be considered for next search
    diffeoOpts.setdefault('dir',True) # True -> Forward transition goes from demonstration to control space, False inverse
    diffeoOpts.setdefault('maxOkErr',float(0.05*np.max(cNorm(xSource-xTarget,kd=False)))) # Convergence criterion
    diffeoOpts.setdefault('baseRegVal',1e-3) # regularize search
    diffeoOpts.setdefault('maxKernel',20) # maximal number of kernels per multitransformation
    diffeoOpts.setdefault('minTrans',-0.005) # minimal norm of translation to be considered; If negative, multiplied with "trajectory size"
    diffeoOpts.setdefault('dist2Origin',0.1) # minimal distance between influenced region and 0
    
    # Internal
    diffeoOptsPure = deepcopy(diffeoOpts)

    thisDepth = 0
    xCurrent = xSource.copy()
    if xSourceOld is None:
        xCurrentOld = None
    else:
        xCurrentOld = xSourceOld.copy()
    initialError = compMSE(xTarget-xCurrent)
    initialMaxNormError = np.max(cNorm(xTarget-xCurrent, kd=True))
    
    if currentDiffeo is None:
        #Here old is None; First demonstration added
        currentDiffeo = du.diffeomorphism(dim=xSource.shape[0], isPars=isPars)
        
        while (thisDepth < depthMax) and (np.any(cNorm(xTarget-xCurrent, kd=False)>maxErr[0]*initialMaxNormError ) ) and ((compMSE(xTarget-xCurrent))>maxErr[1]*initialError):
            thisTrans = du.pd.localPolyMultiTranslation(dim=xSource.shape[0], isPars=isPars)
            getAddTransformations(thisTrans, xCurrent, xTarget, xCurrentOld, **getValidKWARGDict(getAddTransformations, diffeoOpts) )
            assert np.all(thisTrans.dist2nondiffeo(diffeoOpts['margins'][0], diffeoOpts['margins'][1])>0.)
            if thisTrans.nTrans==0:
                break
            currentDiffeo.addTransformation(thisTrans, True)
            #update
            xCurrent = thisTrans.forwardTransform(xCurrent)
            xCurrentOld = thisTrans.forwardTransformation(xCurrentOld) if xCurrentOld is not None else None
            
            thisDepth += 1
    else:
        #It is assumed that now that xSourceOld and xTargetOld exists
        # First check if current diffeo is good enough
        # This is the case if the errors of old and new are comparable
        xSourceOldTilde = currentDiffeo.forwardTransform(xSourceOld)
        mseOld = compMSE(xSourceOldTilde-xTargetOld)
        xSourceTilde = currentDiffeo.forwardTransform(xSource)
        if compMSE(xSourceTilde-xTarget) > mseOld*4.:
            #If not ok, check if kernels can be added to the existing transformations
            for k,(thisDir, thisTrans) in currentDiffeo.enum():
                try:
                    assert np.all(thisTrans.dist2nondiffeo(diffeoOpts['margins'][0],diffeoOpts['margins'][1]) > 0.)
                except:
                    assert 0
                xTargetInverse = currentDiffeo.inverseTransform(xTarget, kStart=0, kStop = currentDiffeo.nTrans-k-1)
                getAddTransformations(thisTrans, xCurrent.copy(), xTargetInverse, xCurrentOld, **getValidKWARGDict(getAddTransformations, diffeoOpts))
                assert np.all(thisTrans.dist2nondiffeo(diffeoOpts['margins'][0],diffeoOpts['margins'][1]) > 0.)
                xCurrent = thisTrans.forwardTransform(xCurrent)
                xCurrentOld = thisTrans.forwardTransform(xCurrentOld)
            
            #Check if the result is acceptable
            xSourceTilde = currentDiffeo.forwardTransform(xSource)
            if compMSE(xSourceTilde-xTarget) >= mseOld*mseFac:
                # If this is still insufficient we can prepend the diffeo with additional transformations
                preDiffeo = du.diffeomorphism(dim=xSource.shape[0], isPars=isPars)
                thisDepth = currentDiffeo.nTrans
                xCurrent = xSource.copy()
                xCurrentOld = xSourceOld.copy()
                xTargetTilde = currentDiffeo.inverseTransform(xTarget)
                initialError = compMSE(xTarget-xCurrent)
                initialMaxNormError = np.max(cNorm(xTarget-xCurrent,kd=True))
                
                while (thisDepth < depthMax) and (np.any(cNorm(xTarget-xCurrent, kd=False)>maxErr[0]*initialMaxNormError ) ) and ((compMSE(xTarget-xCurrent))>maxErr[1]*initialError) and compMSE(preDiffeo.forwardTransform(currentDiffeo.forwardTransform(xSource))-xTarget) > mseOld*mseFac:
                    thisTrans = du.pd.localPolyMultiTranslation(dim=xSource.shape[0], isPars=isPars)
                    getAddTransformations(thisTrans, xCurrent, xTargetTilde, xSourceOld, **getValidKWARGDict(getAddTransformations, diffeoOpts) )
                    if thisTrans.nTrans==0:
                        break
                    preDiffeo.addTransformation(thisTrans, True)
                    #update
                    xCurrent = thisTrans.forwardTransform(xCurrent)
                    xCurrentOld = thisTrans.forwardTransform(xCurrentOld)
                    assert np.allclose(xSourceOld, xCurrentOld), 'Prepended diffeo modified old source'
                    thisDepth += 1
                #Prepend the diffeo
                currentDiffeo.insertTransformation(preDiffeo._transformationList, preDiffeo._directionList, list(range(preDiffeo.nTrans)))

    # Do the optimization if demanded
    if optimizeOpts['doOpt']:
        if xSourceOld is not None:
            xSourceAll = np.hstack((xSource, xSourceOld))
            xTargetAll = np.hstack((xTarget, xTargetOld))
        else:
            xSourceAll = xSource
            xTargetAll = xTarget
        initialError = compMSE(xTargetAll - currentDiffeo.forwardTransform(xSourceAll)) #Compute MSE regardless of log barrier
        

        optimizeOpts['absTol']  = -optimizeOpts['absTol']*initialError if (optimizeOpts['absTol']<0.) else optimizeOpts['absTol']

        currentDiffeo.epsConstraints = optimizeOpts['epsConstraints']
        currentDiffeo.alphaConstraints = optimizeOpts['alphaConstraints']
        
        currentDiffeo.alphaConstraints = -currentDiffeo.alphaConstraints*initialError if (currentDiffeo.alphaConstraints<0.) else currentDiffeo.alphaConstraints
        
        currentDiffeo.margins = diffeoOptsPure['margins']
        
        #Recompute taking into account the barrier
        initialError = currentDiffeo.getMSE(sourcex=xSourceAll, targetx=xTargetAll, alphaConstraints=optimizeOpts['alphaConstraints'], epsConstraints=optimizeOpts['epsConstraints'], marginSingle=currentDiffeo.margins[0], marginOverlap=currentDiffeo.margins[1])
        
        # Other init
        lastMSE = 1e320
        newMSE = initialError
        thisIter = 0
        
        while ((thisIter < optimizeOpts['maxIter']) and (newMSE > optimizeOpts['absTol']) and ((lastMSE-newMSE)/newMSE > optimizeOpts['relTol'])):
            print("aaaa")
            thisIter += 1
            lastMSE = newMSE
            print("aaaabbbb")
            #Step
            newMSE,alphaApply = currentDiffeo.backTrackStep(xSourceAll,xTargetAll,alpha=optimizeOpts['alpha'], tau=optimizeOpts['tau'],c=optimizeOpts['c'], alphaConstraints=optimizeOpts['alphaConstraints'], epsConstraints=optimizeOpts['epsConstraints'], marginSingle=currentDiffeo.margins[0], marginOverlap=currentDiffeo.margins[1])
            optimizeOpts['alpha'] = optimizeOpts['alphaFac']*alphaApply
            print("step {0} : {1}".format(thisIter, newMSE/initialError))

    return currentDiffeo


def statisticMatching(xSource:np.ndarray, xTarget:np.ndarray, method:Union["GMM","KMeans"], baseDir:bool=True, diffeo:du.diffeomorphism=None):
    pass