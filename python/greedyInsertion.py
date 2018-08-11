from mainPars import *
from coreUtils import *
try:
    from cudaUtils import *
except:
    print("No cuda support")
from modifiedEM import *
try:
    from antisymmericUtils import aSymDynamicsGMM, varsInMat
except:
    aSymDynamicsGMM = varsInMat = None
    print('No antisymtools')

from copy import deepcopy as dp,deepcopy

import multiprocessing as mp
from multiprocessing import sharedctypes
import ctypes

import inspect

if mainDoParallel_:
    from multiprocessing import Pool
    from multiprocessing import sharedctypes as sctypes

    # Get a fairly large shared array
    _Xshared = sctypes.RawArray(ctypes.c_double,10000000)  # ten millions values are about 40MB
    _XKshared = sctypes.RawArray(ctypes.c_double,10000000)

def  partialInsertionCPU(GMMpars,xIn,xKIn,k,i,j,doOpt=True,add2Sigma=1e-3,iterMax=100,relTol=1e-3,absTol=1e-3, otherOpts={}):
    
    if isinstance(xIn, np.ndarray):
        dim, nPtK = xKIn.shape
        x=xIn
        xK = xKIn
    else:
        dim, nPt, nPtK = otherOpts['dimensions']
        x = np.frombuffer(xIn)[:dim*nPt]
        x.resize(dim, nPt)

        xK = np.frombuffer(xKIn)[:dim*nPtK]
        xK.resize(dim,nPtK)
        
    if isinstance(add2Sigma, float):
        add2Sigma = Id(dim)*add2Sigma
    else:
        assert add2Sigma.shape == (dim,dim), 'Wrong add2Sigma {0}; {1}'.format([dim,dim], add2Sigma.shape)

    xi = xK[:,[i]]
    xj = xK[:,[j]]
    indI = cNorm( xK - xi, False ) < cNorm( xK - xj, False )
    indJ = np.logical_not(indI)
    nPtI = int(sum(indI))
    nPtJ = int(sum(indJ))
    if (nPtI < 2*x.shape[0]) or (nPtJ < 2*x.shape[0]):
        warnings.warn("Discarding a test for {0} since too few points are associated to the new gaussian")
        return [-np.Inf, []]


    # Delete the k'th gaussian
    priorK = GMMpars['prior'][k,0]
    priorNew = np.delete(GMMpars['prior'], k, axis=0).squeeze()
    delKernel = GMMpars['kernelList'].pop(k)

    # Get mean and covariance of the new kernels
    mui = np.mean(xK[:,indI], keepdims=True, axis=1)
    muj = np.mean(xK[:,indJ], keepdims=True, axis=1)
    Sigmai = myCov(xK[:,indI]-mui, cpy=False, N=nPtI - 1)+add2Sigma
    Sigmaj = myCov(xK[:,indJ]-muj, cpy=False, N=nPtJ - 1)+add2Sigma

    # Add the new kernels
    Ki = dp(delKernel)
    Kj = dp(delKernel)
    Ki['mu'] = mui
    Ki['Sigma'] = Sigmai+add2Sigma
    Kj['mu'] = muj
    Kj['Sigma'] = Sigmaj+add2Sigma
    priorNew = np.hstack(( priorNew, priorK*float(nPtI)/float(nPtK), priorK*float(nPtJ)/float(nPtK) ))
    GMMpars['prior'] = priorNew
    
    try:
        if otherOpts['fixZeroKernel']:
            #Kj is the designated zero kernel -> modifiy mean and covariance accordingly
            Kj['mu'], Kj['Sigma'] = adjustZeroKernel(GMMpars['kernelList'][0]['nVarI'], Kj['mu'].copy(), Kj['Sigma'].copy())
            
    except KeyError:
        pass
    
    # Add; Possible zerokernel is last kernel
    GMMpars['kernelList'] += [Ki,Kj]
        

    # Finished setting initial guess, create GMM
    GMM = GaussianMixtureModel(parDict=GMMpars)

    # Do optimization if demanded
    if doOpt:
        EMAlgorithmCPU(GMM, x, add2Sigma=add2Sigma, iterMax=iterMax, relTol=relTol, absTol=absTol, otherOpts=otherOpts)

    return [ GMM.getLogLikelihood(x), GMM.toPars() ]

##################################
# Inherit for multiproc
if mainDoParallel_:
    def  partialInsertionCPUInherit(GMMpars,k,i,j,doOpt=True,add2Sigma=1e-3,iterMax=100,relTol=1e-3,absTol=1e-3, otherOpts={}):
        return partialInsertionCPU(GMMpars,_Xshared,_XKshared,k,i,j,doOpt=doOpt,add2Sigma=add2Sigma,iterMax=iterMax,relTol=relTol,absTol=absTol, otherOpts=otherOpts)


    ##################################
    # Pool
    insertionWorkers = Pool(countCPU_)

def addZeroKernel(GMM:GaussianMixtureModel, nVarI:int, x:np.ndarray, add2Sigma:np.ndarray, relTol:float, absTol:float):

    nPt = x.shape[1]
    nK = GMM.nK
    weights = GMM._getWeightsCPU(x)
    indKernel = np.argmax(weights, axis=0)
    bestLogLike = -np.inf
    bestPars = None
    
    # Kernel closest to zero
    k0 = np.argmin(cNormSquare(GMM.mu,kd=False))
    iK0 = indKernel==k0
    
    xk0 = x[:,iK0]
    #clost point to zero
    ixk0 = np.argmin(cNormSquare(xk0,kd=False))
    
    #Other ind
    iOther = np.random.choice(xk0.shape[1], 10)
    indReplace = ixk0==iOther
    while np.any(indReplace):
            iOther[indReplace] = np.random.choice(xk0.shape[1], sum(indReplace))
            indReplace = ixk0 == iOther

    #def  partialInsertionCPU(GMMpars,x,xK,k,i,j,doOpt=True,add2Sigma=1e-3,iterMax=100,relTol=1e-3,absTol=1e-3, otherOpts={}):
    for iJ in iOther:
        thisLogLike, thisPars =  partialInsertionCPU(GMM.toPars(),x,xk0,k0,iJ,ixk0,doOpt=True,add2Sigma=add2Sigma,iterMax=100,relTol=1e-3,absTol=1e-3,otherOpts={"fixZeroKernel":True})
        if thisLogLike>bestLogLike:
            bestLogLike = thisLogLike
            bestPars = thisPars
    
    #GMM.loadPars(bestPars)
    GMM = GaussianMixtureModel(parDict=bestPars)
    return GMM
    
    


def greedyInsertionCPU(GMM, x, nPartial=10, speedUp=False, doOpt=True, add2Sigma=1e-3, iterMax=100, relTol=1e-3, absTol=1e-3):
    dim,nPt = x.shape
    nK = GMM.nK
    weights = GMM._getWeightsCPU(x)
    indKernel = np.argmax(weights, axis=0)
    
    if mainDoParallel_:
        #xL = nPartial*[x]
        #Copy to buffer
        _Xshared[:dim*nPt] = x.ravel()
        doOptL = nPartial*[doOpt]
        add2SigmaL = nPartial*[add2Sigma]
        iterMaxL = nPartial*[iterMax]
        relTolL = nPartial*[relTol]
        absTolL = nPartial*[absTol]
    
    # Now split each of the kernels
    resultList = []
    for k in range(nK):
        indKernelK = indKernel == k
        xK = x[:, indKernelK]
        nPtK = xK.shape[1]
        if mainDoParallel_:
            _XKshared[:dim*nPtK] = xK.ravel()
            otherOptsL = nPartial*[{'dimensions':[dim,nPt, nPtK]}]
        if xK.shape[1] < GMM.nVarTot*4:
            warnings.warn("Skipping kernel {0} in update process due to a lack of points")
            continue
        if speedUp:
            # Check consistency if assumption that the base of each kernel is disjoint from the others
            parasitInfl = np.mean(weights[np.hstack((np.arange(0, k), np.arange(k + 1, nK))), indKernelK])
            if parasitInfl > 0.05:
                warnings.warn("Disjoint base assumption might not be valid for kernel {0}".format(k))
            del parasitInfl

        # Generate random samples
        randSampI = np.random.choice(nPtK, nPartial)
        randSampJ = np.random.choice(nPtK, nPartial)
        indReplace = randSampI == randSampJ
        while np.any(indReplace):
            randSampI[indReplace] = np.random.choice(nPtK, sum(indReplace))
            randSampJ[indReplace] = np.random.choice(nPtK, sum(indReplace))
            indReplace = randSampI == randSampJ

        if speedUp:
            assert 0, "TBD"
        else:
            if mainDoParallel_:
                #assert 0, "TBD cuda driver error"
                thisPars = GMM.toPars()
                GMMparsL = [ deepcopy(thisPars) for k in range(nPartial)]
                # partialInsertionCPU(GMMpars,x,xK,k,i,j,doOpt=True,add2Sigma=1e-3,iterMax=100,relTol=1e-3,absTol=1e-3)
                newList = insertionWorkers.starmap(partialInsertionCPUInherit, zip(GMMparsL, nPartial*[k], randSampI, randSampJ, doOptL, add2SigmaL, iterMaxL, relTolL, absTolL, otherOptsL))
                resultList += newList
            else:
                resultList += lmap( lambda ij : partialInsertionCPU(GMM.toPars(),x,xK,k,ij[0],ij[1],doOpt=doOpt,add2Sigma=add2Sigma, iterMax=iterMax, relTol=relTol,absTol=absTol), zip(randSampI, randSampJ) )

    #Get the best updated model among all tested ones
    bestVal = -np.Inf
    bestPars = None
    for newVal, newPars in resultList:
        if newVal > bestVal:
            bestVal = newVal
            bestPars = newPars

    # Load pars
    newGMM = GaussianMixtureModel(parDict=bestPars)
    return  newGMM

# Greedy algorithm to find GMM with "optimal" number of components
def greedyEMCPU(x, nVarI=None, nKMax=5, nPartial=10, add2Sigma=1.e-3, iterMax=100, relTol=1e-2, absTol=1e-2,
                doPlot=False, speedUp=False, interOpt=True, reOpt=None, convFac = 100., warmStartGMM=None, otherOpts={}):

    reOpt = not interOpt if reOpt is None else reOpt
    dim = x.shape[0]
    nVarI = dim-1 if nVarI is None else nVarI
    
    otherOptsBase = {"fixZeroKernel":False}
    otherOptsBase.update(otherOpts)

    # Start off with one component
    if warmStartGMM is None:
        GMM = GaussianMixtureModel()
        mu = np.mean(x,axis=1,keepdims=True)
        Sigma = myCov(x-mu,N=x.shape[1]-1,cpy=False)+add2Sigma
        GMM.addKernel(gaussianKernel(nVarI,Sigma,mu))
        GMM.prior = ones((1,))
    else:
        GMM = deepcopy(warmStartGMM)

    while True:
        GMMold = dp(GMM)
        lastLogLike = GMMold._getLogLikelihoodCPU(x)#GMMold.getLogLikelihood(x)
        print("old best has {0} components and a MSE of {1}".format(GMMold.nK, lastLogLike))
        # Stop if max components
        if GMM.nK == nKMax:
            print("Maximal number of components reached with logLike of {0}".format(lastLogLike))
            if otherOptsBase['fixZeroKernel']:
                return addZeroKernel(GMMold,nVarI,x,add2Sigma,relTol,absTol)
            else:
                return GMMold
        # Add a component
        GMM = greedyInsertionCPU(GMM, x, nPartial=nPartial, speedUp=speedUp, doOpt=interOpt, add2Sigma=add2Sigma, iterMax=iterMax, relTol=relTol/convFac,
                        absTol=absTol/convFac)
        # Reoptimise
        if reOpt:
            EMAlgorithm(GMM, x, add2Sigma=add2Sigma, doInit="warm_start", iterMax=iterMax, relTol=relTol/convFac,
                        absTol=absTol/convFac, doPlot=doPlot)
        # Chek result
        newLogLike = GMM._getLogLikelihoodCPU(x)#GMM.getLogLikelihood(x)
        print("new best has {0} components and a MSE of {1}".format(GMM.nK,newLogLike))
        if (abs((lastLogLike - newLogLike) / newLogLike) < relTol) or (abs(newLogLike-lastLogLike) < absTol):
            # Result already converged in last step
            print("greedy insertion convergenced with {0} components and MSE of{1}".format(GMMold.nK, lastLogLike))
            if otherOptsBase['fixZeroKernel']:
                return addZeroKernel(GMMold, nVarI, x, add2Sigma, relTol, absTol)
            else:
                return GMMold
            
    # Done



def modifiedPartialInsertionCPU(asdGMMpars,xTilde,xTildeK,y,k,i,j,doOpt=True,regVal=1e-2,add2Sigma=1e-3,
                                                  iterMax=100,relTolLogLike=1e-3,absTolLogLike=1e-3,relTolMSE=1e-3,absTolMSE=1e-3,mseConverged=0., convBounds=[0.5,0.3], regValStep=[0.08,4],addKWARGS={}, usedClass=aSymDynamicsGMM):
    
    addKWARGS.setdefault('JacSpace')
    
    nVarI = asdGMMpars['kernelList'][0]['nVarI']
    nVarTot, nPtK = xTildeK.shape
    nVarD = nVarTot-nVarI
    nPt = xTilde.shape[1]
    
    if isinstance(add2Sigma, float):
        add2Sigma = Id(nVarTot)*add2Sigma
    else:
        assert add2Sigma.shape == (nVarTot, nVarTot)

    xTildei = xTildeK[:,[i]]
    xTildej = xTildeK[:,[j]]

    indI = cNorm(xTildeK-xTildei,False) < cNorm(xTildeK-xTildej,False)
    indJ = np.logical_not(indI)
    nPtI = int(sum(indI))
    nPtJ = int(sum(indJ))
    if (nPtI < 2*nVarTot) or (nPtJ < 2*nVarTot):
        warnings.warn("Discarding a test for {0} since too few points are associated to the new gaussian")
        return [+np.Inf,[]] #We look to minimize MSE

    # Delete the k'th gaussian
    priorK = asdGMMpars['prior'][k,0]
    priorNew = np.delete(asdGMMpars['prior'],k,axis=0).squeeze()
    delKernel = asdGMMpars['kernelList'].pop(k)

    # Get mean and covariance of the new kernels
    mui = np.mean(xTildeK[:,indI],keepdims=True,axis=1)
    muj = np.mean(xTildeK[:,indJ],keepdims=True,axis=1)
    Sigmai = myCov(xTildeK[:,indI]-mui,cpy=False,N=nPtI-1)+add2Sigma
    Sigmaj = myCov(xTildeK[:,indJ]-muj,cpy=False,N=nPtJ-1)+add2Sigma

    # Add the new kernels
    Ki = dp(delKernel)
    Kj = dp(delKernel)
    Ki['mu'] = mui
    Ki['Sigma'] = Sigmai
    Kj['mu'] = muj
    Kj['Sigma'] = Sigmaj
    priorNew = np.hstack((priorNew,priorK*float(nPtI)/float(nPtK),priorK*float(nPtJ)/float(nPtK)))
    asdGMMpars['prior'] = priorNew
    asdGMMpars['kernelList'] += [Ki,Kj]

    # Finished setting initial guess, create GMM
    asdGMM = usedClass(parDict=asdGMMpars)

    # Do optimization if demanded
    if doOpt:
        thetaOpt = empty((nVarD,nPt))
        #modifiedEMAlgorithmCPU(asdGMM,xTilde[:nVarI,:],y,add2Sigma=add2Sigma,doInit="warm_start",iterMax=iterMax,regVal=regVal,relTolLogLike=relTolLogLike,absTolLogLike=absTolLogLike,relTolMSE=relTolMSE,absTolMSE=absTolMSE,doPlot=1,convBounds=convBounds,regValStep=regValStep)
        try:
            modifiedEMAlgorithmCPU(asdGMM,xTilde[:nVarI,:],y,thetaInit=xTilde[nVarI:,:],add2Sigma=add2Sigma,doInit="continue",iterMax=iterMax,regVal=regVal,
                                   relTolLogLike=relTolLogLike,absTolLogLike=absTolLogLike,relTolMSE=relTolMSE,absTolMSE=absTolMSE,mseConverged=mseConverged,doPlot=0,
                                   convBounds=convBounds,regValStep=regValStep,thetaOpt=thetaOpt,**addKWARGS)
        except:
            newAddKWARGS = getValidKWARGDict(modifiedEMAlgorithmCPU, addKWARGS)
            modifiedEMAlgorithmCPU(asdGMM,xTilde[:nVarI,:],y,thetaInit=xTilde[nVarI:,:],add2Sigma=add2Sigma,doInit="continue",iterMax=iterMax,
                                   regVal=regVal,relTolLogLike=relTolLogLike,absTolLogLike=absTolLogLike,relTolMSE=relTolMSE,absTolMSE=absTolMSE,mseConverged=mseConverged,doPlot=0,
                                   convBounds=convBounds,regValStep=regValStep,thetaOpt=thetaOpt,**newAddKWARGS)
    else:
        thetaOpt = xTilde[nVarI:,:]

    mse = asdGMM.getMSE(xTilde[:nVarI,:],y,**getValidKWARGDict(asdGMM.getMSE, addKWARGS))

    try:
        if mse < addKWARGS['bestOverall']:
            addKWARGS['bestOverall'] = mse
            oldTheta = np.frombuffer(addKWARGS['bestThetaOpt'].get_obj())
            oldTheta.resize((nVarD,xTilde.shape[1]))
            oldTheta[:] = thetaOpt
    except KeyError:
        pass

    if ((not (mse>-0.01)) and mse < 1e250):
        print('RR')

    return [mse,asdGMM.toPars()]


def modifiedGreedyInsertionCPU(asdGMM, xTilde, y, nPartial=10, speedUp=False, regVal=1e-2, doOpt=True, add2Sigma=1e-3, iterMax=100, relTolLogLike=1e-3,
                        absTolLogLike=1e-3, relTolMSE=1e-3, absTolMSE=1e-3,mseConverged=0., convBounds=[0.5,0.3], regValStep=[0.08, 4], addKWARGS={} ):
    
    addKWARGS.setdefault('JacSpace')
    
    nPt = xTilde.shape[1]
    nK = asdGMM.nK
    weights = asdGMM._getWeightsCPU(xTilde)# TBD or based on x only; I odn't think so
    indKernel = np.argmax(weights,axis=0)
    
    if mainDoParallel_:
        xL = nPartial*[x]
        doOptL = nPartial*[doOpt]
        add2SigmaL = nPartial*[add2Sigma]
        iterMaxL = nPartial*[iterMax]
        relTolL = nPartial*[relTol]
        absTolL = nPartial*[absTol]
        mseConvergedL = nPartial*[mseConverged]

    # Now split each of the kernels
    #This is somewhat dirty
    bestOverall = sharedctypes.Value(ctypes.c_double); bestOverall = 1e200
    bestThetaOpt = sharedctypes.Array(ctypes.c_double,varsInMat(y.shape[0])*y.shape[1])
    addKWARGS.update({'bestOverall':bestOverall,'bestThetaOpt':bestThetaOpt})
    resultList = []
    usedClass = asdGMM.__class__
    for k in range(nK):
        indKernelK = indKernel == k
        xTildeK = xTilde[:,indKernelK]
        nPtK = xTildeK.shape[1]
        if xTildeK.shape[1] < asdGMM.nVarTot*4:
            warnings.warn("Skipping kernel {0} in update process due to a lack of points".format(k))
            continue
        if speedUp:
            # Check consistency if assumption that the base of each kernel is disjoint from the others
            parasitInfl = np.mean(weights[np.hstack((np.arange(0,k),np.arange(k+1,nK))),indKernelK])
            if parasitInfl > 0.05:
                warnings.warn("Disjoint base assumption might not be valid for kernel {0}".format(k))
            del parasitInfl

        # Generate random samples
        randSampI = np.random.choice(nPtK,nPartial)
        randSampJ = np.random.choice(nPtK,nPartial)
        indReplace = randSampI == randSampJ
        while np.any(indReplace):
            randSampI[indReplace] = np.random.choice(nPtK,sum(indReplace))
            randSampJ[indReplace] = np.random.choice(nPtK,sum(indReplace))
            indReplace = randSampI == randSampJ

        if speedUp:
            assert 0,"TBD"
        else:
            if mainDoParallel_:
                assert 0, "TBD cuda driver error and other"
                GMMparsL = [asdGMM.toPars() for k in range(nPartial)]
                xTildeKL = nPartial*[xTildeK]
                # partialInsertionCPU(GMMpars,x,xK,k,i,j,doOpt=True,add2Sigma=1e-3,iterMax=100,relTol=1e-3,absTol=1e-3)
                with Pool(4) as p:
                    newList = p.starmap(modifiedPartialInsertionCPU,
                                        zip(asdGMMparsL,xTildeL,xTildeKL,y,nPartial*[k],randSampI,randSampJ,doOptL,regValL,add2SigmaL,
                                            iterMaxL,relTolL,absTolL,mseConvergedL))
                resultList += newList
            else:
                resultList += lmap(
                    lambda ij:modifiedPartialInsertionCPU(asdGMM.toPars(),xTilde,xTildeK,y,k,ij[0],ij[1],doOpt=doOpt,regVal=regVal,add2Sigma=add2Sigma,
                                                  iterMax=iterMax,relTolLogLike=relTolLogLike,absTolLogLike=absTolLogLike,relTolMSE=relTolMSE,absTolMSE=absTolMSE,mseConverged=mseConverged, convBounds=convBounds, regValStep=regValStep, addKWARGS=addKWARGS, usedClass=usedClass), zip(randSampI,randSampJ))
        
    #Get the best updated model among all tested ones
    #Here we care more about mse than loglike
    bestVal = np.Inf
    bestPars = None
    for newVal, newPars in resultList:
        if newVal < bestVal:
            bestVal = newVal
            bestPars = newPars

    # Load pars
    newasdGMM = aSymDynamicsGMM(parDict=bestPars)
    return  newasdGMM


def modifiedGreedyEMCPU(asdGMM, x, y, warm_start=False, thetaInit=None, nKMax=5, nPartial=10, regVal=1e-2, add2Sigma=1.e-3, iterMax=100, relTolLogLike=1e-2, absTolLogLike=1e-2, relTolMSE=1e-2, absTolMSE=1e-2, mseConverged=0,
                doPlot=False, speedUp=False, interOpt=True, reOpt=None, convFac = 100., convBounds=[0.2,0.1], regValStep=[0.08, 4.], plotCallback=None, addKWARGS={}):
    
    addKWARGS.setdefault('JacSpace')
    
    # Here we do not really care about loglikelihood, only mse
    asdGMMCurrent = asdGMM.__copy__()
    if isinstance(regValStep, float):
        regValStep = [regValStep, max(1., 20.*regValStep)]
    try:
        _ = regValStep[0]
        _ = regValStep[1]
    except:
        assert 0, "regValStep could not be processed"
    
    reOpt = not interOpt if reOpt is None else reOpt
    absTolMSE = absTolMSE if (absTolMSE>=0.) else -absTolMSE*np.mean(y**2, keepdims=False)
    
    
    if warm_start and (thetaInit is None):
        thetaInit = asdGMMCurrent._evalMapCPU(x)
    elif thetaInit is None:
        thetaInit = asdGMMCurrent.initOptimize(x,y,regVal=regVal)
    
    #Augment space
    xTilde = np.vstack((x,thetaInit))
    nVarI, nPt = x.shape
    nVarD = thetaInit.shape[0]
    nVarTot = nVarI+nVarD


    if isinstance(add2Sigma,float) or (isinstance(add2Sigma,np.ndarray) and add2Sigma.size == 1):
        add2Sigma = add2Sigma*Id(nVarTot)
    else:
        assert add2Sigma.shape == (nVarTot,nVarTot)
        try:
            assert np.all(add2Sigma == add2Sigma.T)
            chol(add2Sigma)
        except:
            assert 0,"Regularization covariance is not symmetric positive"

    thetaOpt=None
    if not warm_start:
        thetaOpt = empty(thetaInit.shape)
        asdGMMCurrent._gaussianList = []
        #Start off with a one component mixture
        mu = np.mean(xTilde, axis=1, keepdims=True)
        Sigma = Id(nVarTot)*np.median( eigh(myCov(xTilde-mu, N=nPt-1, cpy=False))[0] ) #Do not initialize with real cov
        asdGMMCurrent.addKernel( gaussianKernel(nVarI, Sigma, mu) )
        asdGMMCurrent.prior = ones((1,))
        
        # First call; ignore mseConverged
        modifiedEMAlgorithm(asdGMMCurrent,x,y,doInit="continue",thetaInit=thetaInit, iterMax=iterMax,relTolLogLike=relTolLogLike/convFac,absTolLogLike=absTolLogLike/convFac,
                            relTolMSE=relTolMSE/convFac,absTolMSE=absTolLogLike/convFac,mseConverged=0.,initStepSizeJac=0.001,add2Sigma=add2Sigma,doPlot=doPlot,
                            convBounds=convBounds,regValStep=regValStep, thetaOpt=thetaOpt, **addKWARGS)

    thetaOpt = thetaOpt if thetaOpt is not None else xTilde[nVarI:,:].copy()
    xTilde[nVarI:,:] = thetaOpt
    thisIter = 0
    if mseConverged<0:
        # If negative than use regularized current pars as errors + the mean of all y values
        thisMSE = sum(square(y))/y.shape[1]
        mseConverged = -mseConverged*thisMSE
        for k,aGaussian in asdGMMCurrent.enum():
            mseConverged += sum(square(aGaussian.mu[aGaussian.nVarI:,[0]]))*regVal
        del thisMSE
    while True:
        thisIter += 1
        print("Insert component {0}".format(thisIter+1))
        if callable(plotCallback):
            plotCallback(asdGMMCurrent, x, xTilde, y)
        
        asdGMMold = dp(asdGMMCurrent)
        lastLogLike = asdGMMCurrent.getLogLikelihood(xTilde)
        lastMSE = asdGMMCurrent.getMSE(x,y)
        
        if (asdGMMold.nK == nKMax) or (thisIter==iterMax):
            #return
            if thisIter==iterMax:
                warnings.warn('Maximum iterations ({0}) reached for modifiedEM'.format(iterMax))
            print("Maximal number of components reached with LogLike of {0} and MSE of {1}".format(lastLogLike, lastMSE))
            break
        
        #Add a component
        asdGMMCurrent = modifiedGreedyInsertionCPU(asdGMMCurrent, xTilde, y, nPartial=nPartial, speedUp=speedUp, regVal=regVal, doOpt=interOpt, add2Sigma=add2Sigma, iterMax=iterMax, relTolLogLike=relTolLogLike/convFac,
                        absTolLogLike=absTolLogLike/convFac, relTolMSE=relTolMSE/convFac, absTolMSE=absTolMSE/convFac, mseConverged=mseConverged/convFac, convBounds=convBounds, regValStep=regValStep, addKWARGS=addKWARGS)
        newBestTheta = np.frombuffer(addKWARGS['bestThetaOpt'].get_obj())
        newBestTheta.resize((nVarD,nPt))
        xTilde[nVarI:,:] = newBestTheta
        print("Found new best GMM")
        
        #Reoptimize
        if reOpt:
            try:
                modifiedEMAlgorithm(asdGMMCurrent, x,y,thetaInit=xTilde[nVarI:,:],add2Sigma=add2Sigma, doInit="continue", iterMax=iterMax, regVal=regVal, relTolLogLike=relTolLogLike, absTolLogLike=absTolLogLike, relTolMSE=relTolMSE, absTolMSE=absTolMSE, mseConverged=mseConverged, doPlot=doPlot, convBounds=convBounds, regValStep=regValStep, thetaOpt=thetaOpt, **addKWARGS)
            except:
                modifiedEMAlgorithm(asdGMMCurrent, x,y,thetaInit=xTilde[nVarI:,:],add2Sigma=add2Sigma, doInit="continue", iterMax=iterMax, regVal=regVal, relTolLogLike=relTolLogLike, absTolLogLike=absTolLogLike, relTolMSE=relTolMSE, absTolMSE=absTolMSE, mseConverged=mseConverged, doPlot=doPlot, convBounds=convBounds, regValStep=regValStep, thetaOpt=thetaOpt, **getValidKWARGDict(modifiedEMAlgorithm, addKWARGS) )
            xTilde[nVarI:,:] = thetaOpt
            
        #Update xTilde
        #asdGMMCurrent._evalDynCPU(x,v=xTilde[nVarI:,:])
        #dxTilde = asdGMMCurrent._evalMapCPU(x)-xTilde[nVarI:,:]
        #dxTildeHat = asdGMMCurrent.takeProjectedStepPen(x,y,xTilde[nVarI:,:],dxTilde,regVal=regValStep[0],penVal=regValStep[1])
        #xTilde[nVarI:,:] = dxTildeHat
        #dxTilde = asdGMMCurrent._evalMapCPU(x)-xTilde[nVarI:,:]
        #dxTildeHat,nsProjector = asdGMMCurrent.takeNSProjectedStep(dxTilde,x,nsProjector,returnProjector=True)
        #xTilde[nVarI:,:] += dxTildeHat
        # Replace the optimized values

        #Check result
        newLogLike = asdGMMCurrent.getLogLikelihood(xTilde)
        newMSE = asdGMMCurrent.getMSE(x,y)

        print("New best mixture of {0} components with LogLike of {1} and MSE of {2}".format(asdGMMCurrent.nK, newLogLike,newMSE))
        
        if (((newLogLike-lastLogLike)/abs(newLogLike) < relTolLogLike or (newLogLike-lastLogLike) < absTolLogLike) and ( (lastMSE-newMSE)/newMSE < relTolMSE or lastMSE < absTolMSE)) or ((lastMSE<mseConverged) and bool(mseConverged)) :
            # Result already converged in last step
            #return asdGMMold
            print("modified GMM converged with {0} components for a final logLike of {1} and a MSE of {2}".format(asdGMMold.nK, lastLogLike, lastMSE))
            break
    
    asdGMM.loadPars(asdGMMold.toPars())
    return asdGMM
    #Done