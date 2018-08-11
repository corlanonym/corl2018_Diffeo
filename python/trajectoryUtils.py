from copy import deepcopy

from coreUtils import *
import diffeoUtils as du
import polyDiffeo as pd

import mySpline as mySpline

try:
    import antisymmericUtils as au
except:
    print("Non antisymtools")

import ctypes

from interpolatePy import regularSpline
from modifiedEM import gaussianKernel

if mainDoParallel_:
    from multiprocessing import Pool
    from multiprocessing import sharedctypes as sctypes

    # Get a fairly large shared array
    _ErrShared = sctypes.RawArray(ctypes.c_double,10000000)
    _XSourceShared = sctypes.RawArray(ctypes.c_double,10000000)
    _XTargetShared = sctypes.RawArray(ctypes.c_double,10000000)


# Some pars
roundLenFacPar_ = 0.3

def localTrajectoryDistance(kernelW:gaussianKernel, kernelU:gaussianKernel, xEval:np.ndarray, xS0:np.ndarray, xS1:np.ndarray, dirEval:np.ndarray = None, dS0:np.ndarray=None, dS1:np.ndarray=None):
    
    dS0 = computeDirections(xS0) if (dS0 is None) else dS0
    dS1 = computeDirections(xS1) if (dS1 is None) else dS1
    
    dS0 = dS0/(cNorm(dS0,kd=True)+epsFloat)
    dS1 = dS1/(cNorm(dS1,kd=True)+epsFloat)
    
    
    
    xEval = xEval.reshape((xEval.shape[0],-1))
    dirEval = dirEval if (dirEval is None) else dirEval.reshape(xEval.shape)
    
    out = []
    
    for k in range(xEval.shape[1]):
        
        kernelW.mu = xEval[:,[k]]
        kernelU.mu = xEval[:,[k]]

        thisDirEval = np.sum(kernelW.getWeights(xS0)*dS0,axis=1,keepdims=True) if (dirEval is None) else dirEval[:,[k]]
        thisDirEval /= (cNorm(thisDirEval, kd=False)+epsFloat)
        out.append( np.mean(np.multiply(thisDirEval, kernelU.getWeights(xS1)) * dS1) )
    
    return out
    
#####################################
def repeatInit(xseries:List[np.ndarray], repeatOpt=None, vseries:List[np.ndarray]=None):
    out = []
    if repeatOpt is None:
        if vseries is not None:
            return np.hstack((xseries)), np.hstack((vseries))
        else:
            return np.hstack((xseries))
    else:
        xStack = []
        for ax in xseries:
            thisI = int(repeatOpt[0] if repeatOpt[0]>=0 else -repeatOpt[0]*ax.shape[1])
            xStack.append( np.hstack( repeatOpt[1]*[ax[:,:thisI]] + [ax] ) )
        
        if vseries is None:
            return np.hstack(xStack)
        else:
            vStack = []
            for av in vseries:
                thisI = repeatOpt[0] if repeatOpt[0]>=0 else -repeatOpt[0]*av.shape[1]
                vStack.append( np.hstack( repeatOpt[1]*[av[:,:thisI]] + [ax] ) )
            
            return np.hstack(xStack), np.hstack(vStack)
    
        
#####################################
# Cost computation
# fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
def fBaseMult(baseMin,baseMax,thisM,thisT,thisD,thisReg,dim,nPt):
    err = np.frombuffer(_ErrShared)[:dim*nPt]
    err.resize(dim,nPt)
    
    xsource = np.frombuffer(_XSourceShared)[:dim*nPt]
    xsource.resize(dim,nPt)
    
    fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM,thisT,float(newBase),thisD))-thisReg*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
    
    bestBase,error,ok,_dummy = fminbound(fCost,baseMin,baseMax,full_output=True)
    
    assert ok==0
    
    return [bestBase,error]


#####################################
# Cost computation
# fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
def fBaseMultDirections(baseMin,baseMax,thisM,thisT,thisD,thisReg,dim,nPt, allIndList):
    # Abuse
    xTarget = np.frombuffer(_XTargetShared)[:dim*nPt]
    xTarget.resize(dim,nPt)
    
    xCurrent = np.frombuffer(_ErrShared)[:dim*nPt]
    xCurrent.resize(dim,nPt)
    
    xsource = np.frombuffer(_XSourceShared)[:dim*nPt]
    xsource.resize(dim,nPt)
    
    fCost = lambda newBase:trajDist(xTarget-regularSpline(xCurrent+du.pd.c1PolyTrans(xsource,thisM,thisT,float(newBase),thisD), indList=allIndList)[0])-thisReg*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
    
    bestBase,error,ok,_dummy = fminbound(fCost,baseMin,baseMax,full_output=True)
    
    assert ok == 0
    
    return [bestBase,error]


if mainDoParallel_:
    minimizeWorkers = Pool(countCPU_)
    
def getBestAlphaVel(x0:"start Position", x1:"end Position", v0:"start velocity", v1:"end Velocity", mode:int=2, doPlot:bool=False, gammaCurv = .001)->"velocity coeff; float":
    #doPlot=True
    if doPlot:
        import plotUtils as pu
        ff,aa = pu.plt.subplots(2,1)
    
    if mode==1:
        tti = np.linspace(0,1,1000)
        alphaVel = 1.
        
        x0s,x1s = x0,x1
        x0sdn = v0/np.linalg.norm(v0)
        x1sdn = v0/np.linalg.norm(v1)
        
        while True:
            print(alphaVel)
            if alphaVel < 1e-3:
                alphaVel = -1
                break
            x0sd = alphaVel*v0
            x1sd = alphaVel*v1
            
            # Get the function
            iFund = mySpline.getSpline3(x0s,x0sd,x1s,x1sd,getDeriv=1)
            xxxdi = iFund(tti)
            xxxdi /= np.linalg.norm(xxxdi,axis=0,keepdims=True)
            # Get all dot prods
            dotv0 = np.sum(x0sdn*xxxdi,axis=0,keepdims=False)
            dotv1 = np.sum(x1sdn*xxxdi,axis=0,keepdims=False)
            dotv0v1 = np.sum(x0sdn*x1sdn)
            # No loop if worst not at end/beginning
            # old
            if 0:
                if (np.argmin(dotv0) < (dotv0.size-1)) or (np.argmin(dotv1) > 0):
                    # Loop
                    alphaVel *= 0.9
                else:
                    break
            elif 1:
                # The worst is only allowed to be so an so much worse than the two end velocities
                mindotv0 = np.min(dotv0)
                mindotv1 = np.min(dotv1)
                deltaDot = 1.-dotv0v1
                overshootFac = 0.1
                # If start and end velocity point into same halfspace allow no overshoot
                if dotv0v1 > 0.:
                    overshootFac = 0.15
                #overshootFac = 0.2 # todo check
                if (mindotv0 < dotv0v1-overshootFac*deltaDot) or (mindotv1 < dotv0v1-overshootFac*deltaDot):
                    # Loop
                    alphaVel *= 0.9
                else:
                    break
            
            else:
                assert 0,'TBD'
    elif mode == 2:
        
        allAlphaVel = np.linspace(0.001,1,50)**2
        tti = np.linspace(0,1,200)

        v0n = v0/np.linalg.norm(v0)
        v1n = v1/np.linalg.norm(v1)
        dotv0v1 = np.sum(v0n*v1n)
        deltaDot = 1.-dotv0v1
        overshootFac = .01
        # If start and end velocity point into same halfspace allow no overshoot
        if dotv0v1 > 0.:
            overshootFac = 2.5e-3
        
        maxCurv = np.empty_like(allAlphaVel)
        
        totCost = np.empty_like(allAlphaVel)
        indOk = np.empty_like(allAlphaVel).astype(np.bool_)
        
        for k,aAlpha in enumerate(allAlphaVel):
            # Check if loop
            v0s = v0*aAlpha
            v1s = v1*aAlpha

            # Get the function
            iFund = mySpline.getSpline3(x0,v0s,x1,v1s,getDeriv=1)
            xxxdi = iFund(tti)
            xxxdin = np.linalg.norm(xxxdi,axis=0,keepdims=True)
            xxxdiN = xxxdi/xxxdin
            # Get all dot prods
            mindotv0 = np.min(np.sum(v0n*xxxdiN,axis=0,keepdims=False))
            mindotv1 = np.min(np.sum(v1n*xxxdiN,axis=0,keepdims=False))

            if (mindotv0 < dotv0v1-overshootFac*deltaDot) or (mindotv1 < dotv0v1-overshootFac*deltaDot):
                # Loop -> infinite cost
                maxCurv[k] = np.infty
                indOk[k] = False
                if doPlot and True:
                    iFun = mySpline.getSpline3(x0,v0s,x1,v1s,getDeriv=[])
                    xxxi = iFun(tti)
                    aa[0].plot(xxxi[0,:], xxxi[1,:], 'r')
                    
            else:
                # Loop -> Ok, check "curvature" Simply use normal acceleration divided by tangential velocity
                # Get acc
                indOk[k] = True
                iFundd = mySpline.getSpline3(x0,v0s,x1,v1s,getDeriv=2)
                xxxddi = iFundd(tti)
                # Norm of normal acc
                xxxddinNorm = np.linalg.norm(xxxddi - xxxdiN*(np.sum(xxxdiN*xxxddi, axis=0, keepdims=False)), axis=0, keepdims=False)
                # Divide and get max
                maxCurv[k] = np.max(xxxddinNorm/xxxdin)
                
                # Get total cost
                totCost[k] = gammaCurv*maxCurv[k] + 1-aAlpha
                
                if doPlot and True:
                    iFun = mySpline.getSpline3(x0,v0s,x1,v1s,getDeriv=[])
                    xxxi = iFun(tti)
                    aa[0].plot(xxxi[0,:], xxxi[1,:], 'b')
                    #aa[1].plot((xxxddinNorm/xxxdin).squeeze())
                    #pu.myQuiver(aa[1], xxxi[:2,:], (xxxddi - xxxdiN*(np.sum(xxxdiN*xxxddi, axis=0, keepdims=False)))[:2,:], 'b')
                    
        allAlphaVel = allAlphaVel[indOk]
        maxCurv = maxCurv[indOk]
        totCost = totCost[indOk]
        
        # Get smallest non-looping curvature factor
        if not np.any(indOk):
            alphaVel = -1.
            warnings.warn('Attention all values seems to loop')
        else:
            alphaVel = allAlphaVel[np.argmin(totCost)]
            #print('Best ind is {0} with alpha {1}'.format(np.argmin(totCost), alphaVel))
    else:
        assert 0, "Mode {0} could not be parsed".format(mode)

    if doPlot:
        iFun = mySpline.getSpline3(x0,v0*alphaVel,x1,v1*alphaVel,getDeriv=[])
        xxxi = iFun(tti)
        aa[0].plot(xxxi[0,:],xxxi[1,:],'g',linewidth=2)
        aa[0].autoscale(True)
        aa[1].autoscale(True)
    
    if alphaVel < 1e-2:
        alphaVel = -1.
    #print('retVal is {0}'.format(alphaVel))
    return float(alphaVel)
                
    
def getFeasibleSourceTrajectory(xIn:np.ndarray, tIn:np.ndarray,vIn:np.ndarray=None,alpha:Union[float, np.ndarray]=-0.001, Ncut=0, fullout=False, doRound:float = 1, doRoundStart:float = 1, spiralMode:bool=False, matchDirections:bool=False):
    """input a basic trajectory -> return a feasible source trajectory
    Assumes that the end of the trajectory is feasible
    """

    interSpline = 0
    doSmooth = False

    # assert alpha<=0.

    if interSpline > 1:
        from scipy.interpolate import pchip_interpolate,interp1d
        xOrig = xIn.copy()
        tOrig = tIn.copy()
        vOrig = vIn.copy()
        dim,nPt = xOrig.shape
    
        fac1 = np.linspace(0.,1.,interSpline+1,endpoint=False)
        fac0 = 1.-fac1
        tIn = []
        for k in range(tOrig.size-1):
            tIn.append(tOrig[k]*fac0+tOrig[k+1]*fac1)
        tIn.append(tOrig[-1])
    
        tIn = np.hstack(tIn)
        xIn = pchip_interpolate(tOrig,xOrig,tIn,axis=1)
        vIn = pchip_interpolate(tOrig,vOrig,tIn,axis=1)
    
        if isinstance(alpha,np.ndarray):
            alphaOrig = alpha.copy()
            alpha = interp1d(tOrig,alphaOrig)(tIn)
    
        dim,nPt = xOrig.shape
    else:
        dim,nPt = xIn.shape

    if not matchDirections:
        xnorm = cNormSquare(xIn,kd=False,cpy=True)
    
        Ncut = Ncut if Ncut > 0 else int(-xIn.shape[1]*Ncut)
        if vIn is None:
            vIn = np.hstack((np.diff(xIn,axis=1)/np.diff(t.squeeze()),zeros((dim,1))))

        # Ignore the last couple of points
        xnorm = xnorm[:-Ncut].copy()
        x = xIn[:,:-Ncut].copy()
        t = tIn[:-Ncut].copy()
        v = vIn[:,:-Ncut].copy()
        if isinstance(alpha,np.ndarray):
            alphaO = alpha.copy()
            alpha = alphaO[:-Ncut]
    
        # vconv = sum(v*x/(cNorm(v)*cNorm(x), axis=0) # Velocity simply pointing inwards
        vconv = sum(2.*x*v/xnorm,axis=0,keepdims=False)  # Quadratic lyapunov-> V = x'.x; Vdot = 2x'.xdot
    
        feasible = np.hstack((1,vconv < alpha))
        fromFeas2Non = np.flatnonzero(np.logical_and(feasible[:-1],np.logical_not(feasible[1:])))
        fromNon2Feas = np.flatnonzero(np.logical_and(np.logical_not(feasible[:-1]),feasible[1:]))
    
        replaceStartInd = []
        replaceStopInd = []
    
        xFeas = x.copy()
        vFeas = v.copy()
    
        if spiralMode:
            iStartFeasible = None
            iStopFeasible = None
            # If spiralMode and the initial points are infeasible the backpropagate
            if not feasible[1]:
                fromFeas2Non = fromFeas2Non[1:]
                iStartFeasible = np.where(xnorm[fromNon2Feas[0]+1:] > xnorm[fromNon2Feas[0]:-1])[0][0]+fromNon2Feas[0]  # fromNon2Feas[1]
            # If spiralMode and the end points are infeasible forward propagate
            if not feasible[-1]:
                iStopFeasible = fromFeas2Non[-1]
                fromFeas2Non = fromFeas2Non[:-1]
    else:
    
        # some fixed constants that should become parameters later on
        radiusShrinkingVel_ = 0.01
    
        xnorm = cNormSquare(xIn,kd=False,cpy=True)
        xnormed = xIn/(cNorm(xIn,kd=True)+epsFloat)
    
        Ncut = Ncut if Ncut > 0 else int(-xIn.shape[1]*Ncut)
        if vIn is None:
            vIn = np.hstack((np.diff(xIn,axis=1)/np.diff(t.squeeze()),zeros((dim,1))))

        # Transform into directions
        vIn = vIn/(cNorm(vIn,kd=True)+epsFloat)
    
        # Ignore the last couple of points
        xnormed = xnormed[:,:-Ncut].copy()
        xnorm = xnorm[:-Ncut].copy()
        x = xIn[:,:-Ncut].copy()
        t = tIn[:-Ncut].copy()
        v = vIn[:,:-Ncut].copy()
        if isinstance(alpha,np.ndarray):
            alphaO = alpha.copy()
            alpha = alphaO[:-Ncut]

        # Compute direction convergence
        dirconv = np.sum(np.multiply(xnormed,v),axis=0,keepdims=False)
    
        feasible = np.hstack((1,dirconv < alpha))
        fromFeas2Non = np.flatnonzero(np.logical_and(feasible[:-1],np.logical_not(feasible[1:])))
        fromNon2Feas = np.flatnonzero(np.logical_and(np.logical_not(feasible[:-1]),feasible[1:]))
        
        if fromFeas2Non.size == fromNon2Feas.size+1:
            fromFeas2Non[-1] = min(fromFeas2Non[-1], x.shape[1]-3)
            fromNon2Feas = np.hstack((fromNon2Feas, x.shape[1]-2))
            
        
        try:
            assert fromFeas2Non.size == fromNon2Feas.size, "This should alwys come in pairs for movements converging to the origin"
        except:
            print("?")
            assert 0
    
        replaceStartInd = []
        replaceStopInd = []
    
        xFeas = x.copy()
        vFeas = v.copy()

    while fromFeas2Non.size:
        deltaInd = 0
        stopInd = x.shape[1]
        
        #Test
        startInd2 = fromNon2Feas[0]
        
        while stopInd > x.shape[1]-1:
            startInd = fromFeas2Non[0]-deltaInd
            try:
                assert startInd >= 0
            except:
                print("?")
                assert 0
            thisNorm = xnorm[startInd]
        
            # todo check if this ok for matchDirections
            if isinstance(alpha,float):
                # convergedNorm = thisNorm + alpha*(t[startInd:]-t[startInd]) # This is velocity pointing inwards
                #convergedNorm = thisNorm*np.exp(alpha*(t[startInd:]-t[startInd]))  # Quadratic convergence
                convergedNorm = thisNorm*np.exp(alpha*(t[startInd2:]-t[startInd]))  # Quadratic convergence only consider points after first feasible # test
            else:
                # convergedNorm = thisNorm+np.cumsum(alpha[startInd:]*(np.hstack((0.,np.diff(t[startInd:]))))*xnorm[startInd:]) #This is velocity pointing inwards
                #convergedNorm = thisNorm*np.cumprod(np.exp(alpha[startInd:]*np.hstack([0,np.diff(t[startInd:]-t[startInd])])))  # Approximate integration
                # All scaling factors
                thisAllScale = np.cumprod(np.exp(alpha[startInd:]*np.hstack([0,np.diff(t[startInd:]-t[startInd])])))  # Approximate integration only consider points after first feasible # test
                # Only from startind2 on
                thisAllScale = thisAllScale[startInd2-startInd:]
                convergedNorm = thisNorm*thisAllScale
        
            # Get the first "feasible" point
            try:
                # take the closest large
                #stopInd = np.flatnonzero(xnorm[startInd+2:] < convergedNorm[2:])[0]+startInd+1+2
                stopInd = np.flatnonzero(xnorm[startInd2:] < convergedNorm)[0]+startInd2+1 #test,.
            except IndexError:
                # If the above fails there is a very small loop around the origin
                stopInd = x.shape[1]+10
            deltaInd += 1
        
        if stopInd <= startInd+3:
            # Very small infeasability zone -> skip
            stopInd = startInd+3
            if stopInd>x.shape[1]-4:
                fromFeas2Non=np.empty(0,)
                fromNon2Feas=np.empty(0,)
                continue
            if np.all(feasible[startInd+1:stopInd] == False):
                fromFeas2Non[0] = stopInd+1
                fromNon2Feas[0] = min(max(fromNon2Feas[0], fromFeas2Non[0]+4), x.shape[1]-2)
                #fromFeas2Non = fromFeas2Non[1:]
                continue
    
        if stopInd < fromFeas2Non[0]:
            # If deltaInd is > 0 this can happen under unfortunate circumstances
            stopInd = min([xFeas.shape[1]-1,fromFeas2Non[0]+1])
    
        replaceStartInd.append(startInd)
        replaceStopInd.append(stopInd)
    
        xStart = x[:,[startInd]]
        xStop = x[:,[stopInd]]
        thisT = t[startInd:stopInd]-t[startInd]
        # Average convergence
        # alphaPrime = (xnorm[stopInd]**0.5-xnorm[startInd]**0.5)/thisT[-1] # This is velocity pointing inwards
        # Quadratic convergence
        # V(t) = Vo*exp(alpha*t)  -> alpha = ln(V(t)/Vo)/t
        alphaPrime = np.log(xnorm[stopInd]/xnorm[startInd])/thisT[-1]
    
        # Get the plane, turn space
        chi0 = xStart/cNorm(xStart,kd=False)
        try:
            chi1 = xStop/cNorm(xStop,kd=False)
        except FloatingPointError:
            assert 0,"xStop is zero"
    
        try:
            chi1 = chi1-chi0*np.sum(chi0*chi1)
            chi1 /= cNorm(chi1,kd=False)
        except FloatingPointError:
            # xStart and xStop are not linearly indep.
            # Any other direction is ok
            chi1 = xStop/cNorm(xStop,kd=False)
            d = np.sign(np.random.rand(*chi1.shape))
            d[d == 0] = 1
            chi1 += d
            chi1 = chi1-chi0*np.sum(chi0*chi1)
            chi1 /= cNorm(chi1,kd=False)
    
        R = np.vstack((chi0.T,chi1.T))
        R = np.vstack((R,(nullspace(R)[1]).T))
    
        if np.linalg.det(R) < 0.:
            R = np.vstack((chi0.T,-chi1.T))
            R = np.vstack((R,(nullspace(R)[1]).T))

        # Angle between them
        xStopP = dot(R,xStop)
        ang0 = np.arctan2(xStopP[1],xStopP[0])
        if ang0 < 0.:
            ang1 = 2.*np.pi+ang0
        else:
            ang1 = ang0-2.*np.pi
    
        allAng0 = thisT*(ang0/thisT[-1])
        allAng1 = thisT*(ang1/thisT[-1])
    
        # allNorms = thisNorm**0.5 + alphaPrime*thisT # Velocity pointing inwards
        allNorms = (thisNorm*np.exp(alphaPrime*thisT))**0.5  # Quadratic convergence
    
        xPatch0 = zeros((dim,stopInd-startInd))
        xPatch0[0,:] = np.cos(allAng0)*allNorms
        xPatch0[1,:] = np.sin(allAng0)*allNorms
        xPatch1 = zeros((dim,stopInd-startInd))
        xPatch1[0,:] = np.cos(allAng1)*allNorms
        xPatch1[1,:] = np.sin(allAng1)*allNorms
    
        xPatch0 = np.dot(R.T,xPatch0)
        xPatch1 = np.dot(R.T,xPatch1)
    
        err0 = np.mean(compMSE(xFeas[:,startInd:stopInd]-xPatch0,ax=1))
        err1 = np.mean(compMSE(xFeas[:,startInd:stopInd]-xPatch1,ax=1))

        # Patch the traj
        if err0 < err1:
            xPatch = xPatch0
        else:
            xPatch = xPatch1
    
        xFeas[:,startInd:stopInd] = xPatch
    
        if doRoundStart:
            if (xPatch.shape[1] > 8) and (startInd > 0):  # Very small patches will not be rounded; Cannot round if initial dir is not feasible
                # Check if "direction" of original and feasible velocity are more or less equal
                # dxOrig = np.diff(xFeas[:,startInd-3:startInd], axis=1)
                # dxPatch = np.diff(xFeas[:,startInd:startInd+dxOrig.shape[1]],axis=1)
                # dxOrig /= cNorm(dxOrig,kd=True)
                # dxPatch /= cNorm(dxPatch,kd=True)

                # if np.sum(dxOrig*dxPatch)/dxOrig.shape[-1] > 0.95:
                # The direction coincide, linearly ramp up velocity
            
                # else:
                # Directions do not match, do rounding as for stop
            
                roundLenFac = roundLenFacPar_  # todo change to parameter
                roundLenInd = int((roundLenFac*xPatch.shape[1]))+1
                stopInd3 = startInd+roundLenInd
                roundR = cNorm(xPatch[:,[roundLenInd]]-xPatch[:,[0]],kd=False)
            
                distNormSign = np.sign(cNorm(xFeas[:,:startInd]-xPatch[:,[0]],kd=False)-roundR).astype(np.int_)
                # Get last sign change (closest to startInd)
                startInd3 = np.nonzero(distNormSign[:-1] != distNormSign[1:])[0]
                if startInd3.size:
                    startInd3 = startInd3[-1]
                else:
                    startInd3 = max([0,startInd-1])
            
                # Now do the actual rounding
                try:
                    tTot = t[stopInd3]-t[startInd3]
                except IndexError:
                    assert 0
                # Get the position and velocities
                x0s = xFeas[:,startInd3]
                x1s = xFeas[:,stopInd3]
            
                # Velocities have to be normalized since "spline time" is between zero and one
                x0sd = (xFeas[:,[startInd3]]-xFeas[:,[startInd3-int(roundLenInd*0.1+1)-1]])/(t[startInd3]-t[startInd3-int(roundLenInd*0.1+1)-1])/tTot  # Get approximate velocity
                x1sd = (xFeas[:,[stopInd3+int(roundLenInd*0.1+1)+1]]-xFeas[:,[stopInd3]])/(t[stopInd3+int(roundLenInd*0.1+1)+1]-t[stopInd3])/tTot  # Get approximate velocity
            
                alphaVel = getBestAlphaVel(x0=x0s,x1=x1s,v0=x0sd,v1=x1sd,mode=2,doPlot=False)
            
                # interpolate
                if alphaVel > 0:
                    iFun = mySpline.getSpline3(x0s,x0sd*alphaVel,x1s,x1sd*alphaVel)
                    ti = (t[startInd3:stopInd3+1]-t[startInd3])/tTot
                    xFeas[:,startInd3:stopInd3+1] = iFun(ti)
                else:
                    try:
                        for kk in range(xFeas.shape[0]):
                            xFeas[kk,startInd3:stopInd3+1] = np.linspace(xFeas[kk,startInd3],xFeas[kk,stopInd3],stopInd3-startInd3+1)
                    except ValueError:
                        pass

            # todo error here rounding velocity start index wrong
    
        if doRound:
        
            # #cAngle = +1.-np.inner(xFeas[:,max(stopInd-5,0)],xFeas[:,max(stopInd+5,xFeas.shape[1]-1)])
            # #thisRoundSize = doRound if (doRound > 0) else int(-(stopInd-startInd)*cAngle*doRound)+1
            # thisRoundSize = doRound if (doRound > 0) else int(-(stopInd-startInd)*doRound)+1
            # #Ensure that the patch is smaller than the array
            # thisRoundSize = min( [thisRoundSize, startInd+0, xFeas.shape[1]-stopInd-1] )
            # roundSize2 = int(thisRoundSize/1.5)+1
            #
            # if roundSize2 > 2:
            #
            #     toRoundPatch = xFeas[:,stopInd-thisRoundSize:stopInd+thisRoundSize].copy()
            #     toRoundPatchE = xFeas[:,stopInd-2*thisRoundSize:stopInd+2*thisRoundSize].copy()#np.hstack((xFeas[:,stopInd-2*thisRoundSize:stopInd], xFeas[:,stopInd+roundSize2:stopInd+roundSize2+2*thisRoundSize])).copy()#
            #     convPatch = np.ones((roundSize2,))/roundSize2
            #
            #     roundedPatchE = np.empty_like(toRoundPatchE)
            #     roundedPatch = np.empty_like(toRoundPatch)
            #
            #     for rDim in range(toRoundPatchE.shape[0]):
            #         roundedPatchE[rDim,:] = np.convolve(toRoundPatchE[rDim,:], convPatch, 'same')
            #         roundedPatch[rDim,:] = roundedPatchE[rDim, thisRoundSize:-thisRoundSize]
            #
            #     xFeas[:,stopInd-thisRoundSize:stopInd+thisRoundSize] = roundedPatch
            #
            #     stopInd = stopInd+thisRoundSize
            #     for kk in range(fromFeas2Non.size):
            #         if kk==0:
            #             continue
            #         if fromFeas2Non[kk] < stopInd:
            #             fromFeas2Non[kk] = stopInd+1
            # Do a spline interpolation
            while (fromFeas2Non.size > 0) and (stopInd >= fromFeas2Non[0]):
                fromFeas2Non = fromFeas2Non[1:]
                # todo check
                fromNon2Feas = fromNon2Feas[1:]
            # Only round "large enough" patches
            if (int(0.8*(xFeas.shape[1]-stopInd)) > 3) and xPatch.shape[1] > 8:
                print(xPatch.shape)
                try:
                    roundLenInd = 2
                    roundLenFac = roundLenFacPar_  # todo change to par
                    while roundLenInd > 1:
                        roundLenInd = int((roundLenFac*xPatch.shape[1]))+1
                        startInd2 = stopInd-roundLenInd
                        roundR = np.linalg.norm(xPatch[:,[-1]]-xPatch[:,[-roundLenInd]])
                        if roundR > np.linalg.norm(xPatch[:,-2]):
                            roundR = np.linalg.norm(xPatch[:,-2])
                            stopInd2 = xFeas.shape[1]-2
                        else:
                            stopInd2 = stopInd+np.argmin(np.abs(np.linalg.norm(xFeas[:,stopInd:]-xPatch[:,[-1]],axis=0)-roundR))
                    
                        stopInd2 = min([stopInd2,xFeas.shape[1]-1])
                    
                        if np.all(feasible[stopInd:stopInd2+1]):
                            # All points in the additional rounding zone are feasible -> everything is ok
                            break
                        else:
                            # Reduce the rounding zone
                            roundLenFac *= 0.5
                
                    if stopInd2 > (startInd2+2):
                        try:
                            tTot = t[stopInd2]-t[startInd2]
                        except IndexError:
                            assert 0
                        # Get the position and velocities
                        x0s = xFeas[:,startInd2]
                        x1s = xFeas[:,stopInd2]
                    
                        # Velocities have to be normalized since "spline time" is between zero and one
                        x0sd = (xFeas[:,[startInd2+int(roundLenInd*0.1+1)+1]]-xFeas[:,[startInd2]])/(t[startInd2+int(roundLenInd*0.1+1)+1]-t[startInd2])/tTot  # Get approximate velocity
                        x1sd = (xFeas[:,[stopInd2]]-xFeas[:,[stopInd2-int(roundLenInd*0.1+1)-1]])/(t[stopInd2]-t[stopInd2-int(roundLenInd*0.1+1)-1])/tTot  # Get approximate velocity
                    
                        alphaVel = getBestAlphaVel(x0=x0s,x1=x1s,v0=x0sd,v1=x1sd,mode=2,doPlot=False)
                    
                        # interpolate
                        if alphaVel > 0:
                            iFun = mySpline.getSpline3(x0s,x0sd*alphaVel,x1s,x1sd*alphaVel)
                            ti = (t[startInd2:stopInd2+1]-t[startInd2])/tTot
                            xFeas[:,startInd2:stopInd2+1] = iFun(ti)
                        else:
                            try:
                                for kk in range(xFeas.shape[0]):
                                    xFeas[kk,startInd2:stopInd2+1] = np.linspace(xFeas[kk,startInd2],xFeas[kk,stopInd2],stopInd2-startInd2+1)
                            except ValueError:
                                pass
                        stopInd = stopInd2
                    else:
                        print('Final rounding zone is too small')
                except FloatingPointError:
                    print('??')
        try:
            vFeas[:,startInd3:stopInd-1] = np.diff(xFeas[:,startInd3:stopInd],axis=1)/np.diff(t[startInd3:stopInd])
        except NameError:
            vFeas[:,startInd:stopInd-1] = np.diff(xFeas[:,startInd:stopInd],axis=1)/np.diff(t[startInd:stopInd])
        # todo check because sth is wrong with velocity
        while (fromFeas2Non.size > 0) and (stopInd >= fromFeas2Non[0]):
            fromFeas2Non = fromFeas2Non[1:]
            #todo check
            fromNon2Feas = fromNon2Feas[1:]
    # Undo the cutting or not
    # xFeas = np.hstack((xFeas, xIn[:,-Ncut:]))
    # vFeas = np.hstack((vFeas, vIn[:,-Ncut:]))

    if spiralMode:
        # xFeas is now actually feasible except (the case given) the start and end
        if iStopFeasible is not None:
            N = 20
            nVars = au.varsInMat(dim)
            # skewInverseMultArray(const double[:,::1] x, const long[:,::1] indMat, const long N):
            indMat = -np.ones((dim,dim))
            indMat[np.triu_indices(dim,1)] = np.arange(nVars)
            indMat[np.tril_indices(dim,-1)] = np.arange(nVars)
            indMat = indMat.astype(np.int_)
            xTilde = au.skewInverseMultArray(np.require(xFeas[:,iStopFeasible-N:iStopFeasible],np.float_,'COA'),indMat,nVars)
            xTilde2 = np.zeros((N,dim,nVars+1))
            for k in range(N):
                xTilde2[k,:,:-1] = xTilde[k,:,:]
                xTilde2[k,:,-1] = xFeas[:,iStopFeasible-N+k]
            xTilde = np.vstack(list(xTilde2))
            vTilde = np.ravel(vFeas[:,iStopFeasible-N:iStopFeasible],'F')
            # Solve
            theta = np.linalg.lstsq(xTilde,vTilde)[0]
            A = au.vect2ASym(theta[:-1])
            A += np.diag(np.ones(dim))*theta[-1]
        
            # A represents the approximate dynamics
            tMiss = t[iStopFeasible-N:]-t[iStopFeasible-N]
            x0 = xFeas[:,[iStopFeasible-N]]
            for k in range(tMiss):
                xk = np.dot(sp.linalg.expm(A*tMiss[k]),x0)
                vk = np.dot(A,xk)
                xFeas[:,[iStopFeasible-N+k]] = xk
                vFeas[:,[iStopFeasible-N+k]] = vk

    if doSmooth:
        from scipy.signal import savgol_filter
        import plotUtils as pu
        from scipy import signal
    
        wLength = (interSpline if interSpline > 1 else 1)*71
        if wLength%2 == 0:
            wLength += 1
        # xFilt = savgol_filter(xFeas, wLength, polyorder=3, axis=1)
    
        window = signal.gaussian(wLength,(wLength/2.+1)/2.)
        window /= np.sum(window)
        xFilt = np.vstack([np.convolve(xFeas[ddim,:],window,'valid') for ddim in range(dim)])
        xFilt = np.hstack((xFeas[:,:int((wLength-1)/2)],xFilt,xFeas[:,int(-(wLength-1)/2):]))
        # xFilt = []
        # for ddim in range(dim):
        #    dx = [vFeas[ddim, 0]*(t[1]-t[0]), vFeas[ddim, -1]*(t[-1]-t[-2])]
        #    xFilt.append( np.convolve( np.hstack( [ [xFeas[ddim,0]-dx[0]*(wLength-kkk) for kkk in range(wLength)], xFeas[ddim,:], [xFeas[ddim,-1]+dx[1]*kkk for kkk in range(wLength)] ]), window,'same')[wLength:-wLength])
        # xFilt = np.vstack((xFilt))
    
        vFiltOO = np.hstack((np.diff(xFilt,axis=1)/np.diff(t),np.zeros((dim,1))))
        # Hack for time
        vFiltOO[:,-1] = vFiltOO[:,-2]
    
        vFilt = vFiltOO.copy()
    
        vFilt[:,int((wLength-1)/2-1)] = 0.25*vFilt[:,int((wLength-1)/2-1)-2]+0.25*vFilt[:,int((wLength-1)/2-1)-1]+0.25*vFilt[:,int((wLength-1)/2-1)+2]+0.25*vFilt[:,int((wLength-1)/2-1)+1]
        vFilt[:,-int((wLength-1)/2-1)-2] = 0.25*vFilt[:,-int((wLength-1)/2-1)-2-2]+0.25*vFilt[:,-int((wLength-1)/2-1)-2-1]+0.25*vFilt[:,-int((wLength-1)/2-1)-2+2]+0.25*vFilt[:,-int((wLength-1)/2-1)-2+1]
    
        # iiii = 3
        # nnnn = min(6, wLength/2-2-iiii)
        # wwww0 = 1./(2.*nnnn)
        # for jjjj in range(iiii):
        #     vFilt[:,int((wLength-1)/2-1)-jjjj] = 0.
        #     vFilt[:,-int((wLength-1)/2-1)-2-jjjj] = 0.
        #
        #     if jjjj!= 0:
        #         vFilt[:,int((wLength-1)/2-1)-jjjj] = 0.
        #         vFilt[:,-int((wLength-1)/2-1)-2-jjjj] = 0.
        #
        #     for kkkk in range(nnnn):
        #         vFilt[:, int((wLength-1)/2-1)-jjjj] += wwww*vFiltOO[:, int((wLength-1)/2-1)-1-jjjj-kkkk]+wwww*vFiltOO[:, int((wLength-1)/2-1)+1-jjjj+kkkk]
        #         vFilt[:,-int((wLength-1)/2-1)-2-jjjj] += wwww*vFiltOO[:,-int((wLength-1)/2-1-jjjj-kkkk)-2-1]+wwww*vFiltOO[:,-int((wLength-1)/2-1)-2+1-jjjj+kkkk]
        #         if jjjj!=0:
        #             vFilt[:,int((wLength-1)/2-1)+jjjj] += wwww*vFiltOO[:,int((wLength-1)/2-1)-1+jjjj-kkkk]+wwww*vFiltOO[:,int((wLength-1)/2-1)+1+jjjj+kkkk]
        #             vFilt[:,-int((wLength-1)/2-1)-2+jjjj] += wwww*vFiltOO[:,-int((wLength-1)/2-1+jjjj-kkkk)-2-1]+wwww*vFiltOO[:,-int((wLength-1)/2-1)-2+1+jjjj+kkkk]
    
        if 0:
            ff,aa = pu.plt.subplots(dim,2)
            for kk in range(dim):
                try:
                    aa[kk,0].plot(tOrig,xOrig[kk,:],'k')
                    aa[kk,1].plot(tOrig,vOrig[kk,:],'k')
                except NameError:
                    aa[kk,0].plot(tIn,xIn[kk,:],'k')
                    aa[kk,1].plot(tIn,vIn[kk,:],'k')
                aa[kk,0].plot(t,xFeas[kk,:],'g')
                aa[kk,1].plot(t,vFeas[kk,:],'g')
                aa[kk,0].plot(t,xFilt[kk,:],'r')
                aa[kk,1].plot(t,vFilt[kk,:],'r')
    
        xFeas = xFilt
        vFeas = vFilt

    if interSpline > 1:
        xFeas = interp1d(t,xFeas,axis=1)(tOrig)
        vFeas = interp1d(t,vFeas,axis=1)(tOrig)

    try:
        assert np.all([(x-y) == 0 for x,y in zip(xFeas.shape,vFeas.shape)]),'Shape error'
    except AssertionError:
        print('aaa')
    if fullout:
        return xFeas,vFeas,{"replaceStartInd":replaceStartInd,"replaceStopInd":replaceStopInd,'Ncut':Ncut}
    else:
        return xFeas,vFeas


from scipy.interpolate import interp1d
from scipy.integrate import odeint


class replayDemonstration:
    
    def __init__(self,x:np.ndarray,v:np.ndarray,t:np.ndarray, alpha:float=-0.01, beta:float=None, p:float=1., whichTraj:int = 1, doRound:bool=True, doRoundStart:bool=True, spiralMode:bool=False, matchDirections:bool=False):
        
        self._x = x.copy()
        self._v = v.copy()
        self._t = t.copy()
        
        self._xf = interp1d(t, x, copy=False,bounds_error=False,fill_value=(x[:,0], x[:,-1]))#linInterp1d(x,t)#interp1d(t, x, copy=False,bounds_error=False,fill_value=(x[:,0], x[:,-1]))
        self._vf = interp1d(t,v,copy=False,bounds_error=False,fill_value=(v[:,0], v[:,-1]))#linInterp1d(v,t)#interp1d(t,v,copy=False,bounds_error=False,fill_value=(v[:,0], v[:,-1]))
        
        self._p = p
        
        self._dim = self._x.shape[0]
        
        self._alpha = alpha
        self._beta = beta
        
        self._whichTraj = whichTraj
        
        self._doRound=doRound
        self._doRoundStart = doRoundStart
        self._spiralMode = spiralMode
        
        self._matchDirections = matchDirections
        
    def getVel(self,x:np.ndarray,t:np.ndarray, fullOut:bool=False)->np.ndarray:
        
        x.resize((self._dim, x.size//self._dim))
        
        xnSquared = cNormSquare(x, kd=False, cpy=True)
        
        vd = self._vf(t)
        vd.resize((self._dim, x.size//self._dim))
        xd = self._xf(t)
        xd.resize((self._dim, x.size//self._dim))
        
        #Add correction
        vd -= self._p*(x-xd)
        
        #Ensure convergence
        # Get radial convergence
        xtv = np.sum(x*vd,axis=0,keepdims=False)
        # Get convergence fac
        minConvFac = self._alpha+self._beta*(self._t[-1]-t)

        minConvFac = minConvFac.reshape((x.shape[1],))

        # Check
        ind = 2*xtv/(xnSquared+epsFloat) >= minConvFac
        ind.resize((x.shape[1],))

        # Adjust
        if np.any(ind):  # Nahh
            try:
                adjustFactor = -xtv[ind]+1./2.*minConvFac[ind]*xnSquared[ind]
                # Correct velocity
                vd[:,ind] += adjustFactor*x[:,ind]/(xnSquared[ind]+epsFloat)
            except:
                assert 0
        
        if fullOut:
            return vd, {'ind':ind}
        else:
            return vd
    
    def getTraj0(self, returnVel=False):
        
        def f(x,t):
            v = self.getVel(x,t)
            v.resize((v.size,))
            return v
        
        xSol = odeint(f, self._x[:,0], self._t).T

        if returnVel:
            vSol = self.getVel(xSol, self._t)

        if returnVel:
            return  xSol, vSol
        else:
            return xSol
    
    def getTraj1(self, returnVel=False):

        #Get the alphas
        alpha = self._alpha + self._beta*(self._t[-1]-self._t)

        xSol, vSol, fullOutDict = getFeasibleSourceTrajectory(self._x, self._t, self._v, alpha, Ncut=-0.01, fullout=True, doRound=self._doRound, doRoundStart=self._doRoundStart, spiralMode=self._spiralMode, matchDirections=self._matchDirections)
        
        try:
            if fullOutDict['Ncut']>0:
                xSol = np.hstack((xSol, self._x[:,-fullOutDict['Ncut']:]))
                vSol = np.hstack((vSol, self._v[:,-fullOutDict['Ncut']:]))
        except KeyError:
            pass
        if returnVel:
            return xSol, vSol
        else:
            return xSol
    
    def getTraj(self, returnVel=False):
        
        if self._whichTraj == 0:
            return self.getTraj0(returnVel)
        elif self._whichTraj == 1:
            return self.getTraj1(returnVel)
        
        else:
            assert 0


from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
#def c1PolyKer(x, bIn, dIn, eIn=None, deriv=0, cIn=None,out=None, fullOut=False, kerFun=c1KerFun, inPolarCoords=False):
from polyDiffeo import c1PolyKer, c1PolyMinB
from scipy.linalg import norm as deformationNorm

class stochasticMatching:
    def __init__(self, x:List[np.ndarray], v:List[np.ndarray], t:List[np.ndarray], optionsSeries={}, optionsMatching={}):
        
        #t = list(map(lambda at: at if (at[0]>at[-1]) else at[-1]-at, t))# This is a bad idea
        
        self._opt = {'errStep':-0.1,
                'alpha':-0.05, 'betaMax':None, 'cBetaMax':None, 'skipEnd':-0.05,
                'acceptTol':-0.025,'p':1., 'doRound':True, 'spiralMode':False,
                'matchDirections':False}
        self._opt.update(optionsSeries)

        self._optMatching = {'isPars':[True,True,True,False,False], 'maxIter':20, 'maxNKernel':6, 'assumePerfect':False, 'interOpt':False, 'finalOpt':False, 'relTol':.5e-2, 'absTol':-2e-2,
                             'nInterTrans':5, 'transFac':1.1, 'minTrans':-.5e-2, 'minBase':-5e-2, 'mode':'kmeans', 'overlapFac':0.7, 'dValues':1./10.,
                             'transCenterFac':0.35,
                             'marginSingle':0.6, 'marginOverlap':0.3, 'alphaConstraints':-.5e-2, 'epsConstraints':3.e-2,
                             'baseRegVal':0.1e-4, 'includeMaxFac':0.35,
                             'maxNKernelErrStruct':10,
                             'matchDirections':False
                             }
        self._optMatching.update(optionsMatching)

        self._x = deepcopy(x)
        self._v = deepcopy(v)
        self._t = deepcopy(t)
        
        self._dim = self._x[0].shape[0]
        
        if self._opt['matchDirections']:
            self._x, self._t, self._v = regularSpline(self._x, self._t, self._v) # regualrise
            self._v = [av/(cNorm(av, kd=True)+epsFloat) for av in self._v] # Direction

            # Compute beta max
            self._opt['betaMax'] = -np.Inf
            for ax,av,at in zip(self._x,self._v,self._t):
                thisSkipEnd = self._opt['skipEnd'] if (self._opt['skipEnd'] > 0) else int(-self._opt['skipEnd']*at.size)
                self._opt['betaMax'] = max( [self._opt['betaMax'], -np.min( np.sum(np.multiply(ax[:,:-thisSkipEnd], av[:,:-thisSkipEnd]), axis=0)/(cNorm(ax[:,:-thisSkipEnd], kd=False)+epsFloat) )] )
        else:
            #Compute beta max
            self._opt['betaMax'] = -np.Inf
            for ax,av,at in zip(x,v,t):
                thisSkipEnd = self._opt['skipEnd'] if (self._opt['skipEnd']>0) else int(-self._opt['skipEnd']*at.size)
                if at[0]<at[-1]:
                    ati = at[-1]-at #Inverse, we expect time to be the remaining go-to time
                else:
                    ati = at
                self._opt['betaMax'] = max([self._opt['betaMax'],np.max((2.*np.sum(av[:,:-thisSkipEnd]*ax[:,:-thisSkipEnd],axis=0)/np.sum(np.square(ax[:,:-thisSkipEnd]),axis=0)-self._opt['alpha'])/ati[:-thisSkipEnd])])
        
    
    def getSeries(self, returnVel:bool=False)->List[np.ndarray]:
        """

        :param returnVel: Also return new nominal velocity
        :return: List of sources and (eventually) velocities
        """
        self._opt['cBetaMax'] = self._opt['betaMax']
        
        #Get the interpolators
        trajList = []
        for ax,av,at in zip(self._x, self._v, self._t):
            trajList.append( replayDemonstration(ax, av, at, alpha = self._opt['alpha'], beta = 0, p=self._opt['p'], doRound=self._opt['doRound'], doRoundStart=self._opt['doRoundStart'], spiralMode=self._opt['spiralMode'], matchDirections=self._opt['matchDirections']))

        #Initialise list and define erorr function
        if returnVel:
            xc = [ [(ax.copy(), av.copy()) for ax, av in zip(self._x, self._v)]]
            errorFunc = lambda thisX, xc: sum(list(map(lambda xnew,xvold:compMSE(xnew-xvold[0]),thisX,xc[-1])))/len(thisX)
        else:
            xc = [deepcopy(self._x)]
            errorFunc = lambda thisX,xc:sum(list(map(lambda xnew,xold:compMSE(xnew-xold),thisX,xc[-1])))/len(thisX)
        
        while self._opt['cBetaMax'] > 0:
            
            print('Searching next intermediate')
            
            #Reset p
            for at in trajList:
                at._p = self._opt['p']
            
            if np.isfinite(self._opt['errStep']):
                #Get the currently allowed mse error
                if returnVel:
                    allCurrentX = np.hstack([ ax for ax,_ in xc[-1] ])
                else:
                    allCurrentX = np.hstack(xc[-1])
                maxErr = compMSE(allCurrentX*self._opt['errStep'])
                acceptTol = compMSE(allCurrentX*self._opt['acceptTol'])
    
                
                # Decrease beta until error too large
                betaU = self._opt['cBetaMax']
                betaL = 0.
            else:
                # Get the currently allowed mse error
                if returnVel:
                    allCurrentX = np.hstack([ax for ax,_ in xc[-1]])
                else:
                    allCurrentX = np.hstack(xc[-1])
                maxErr = 1e300
                acceptTol = 1e300
    
                # Decrease beta until error too large
                betaU = 0.
                betaL = 0.
                
            while True:#(betaU-betaL)>1e-4 and betaU>0.:
                # Decrease beta
                beta = 0.4*betaU+0.6*betaL
                beta = 0. if beta<1e-4 else beta
                #beta = 0.
                for at in trajList:
                    at._beta = beta
                while True:
                    # Ensure that all converge by increasing p
                    thisX = []
                    allConverged = True
                    for at in trajList:
                        thisX.append(at.getTraj())
                        try:
                            at._xf.reset()
                            at._vf.reset()
                        except AttributeError:
                            pass
                        allConverged = allConverged and (bool(cNorm(thisX[-1][:,[-1]],kd=False)<acceptTol) or bool(cNorm(thisX[-1][:,[-1]]-at._x[:,[-1]],kd=False)<acceptTol))
                    
                    if allConverged:
                        break
                    else:
                        print('Increase p')
                        for at in trajList:
                            at._p *= 2.
                
                #Check the error
                thisError = errorFunc(thisX, xc)
                errFac = 1.15
                print("BetaLU: {0};{1} Beta: {2}, error : {3}".format(betaL, betaU, beta, thisError/maxErr))
                if (((maxErr<=thisError) and (thisError<=errFac*maxErr))) or (betaU==0.) or (abs(betaU-betaL)<1.e-3):
                    # Recompute if velocity needed
                    if returnVel:
                        xc.append( [at.getTraj(returnVel) for at in trajList] )
                    else:
                        xc.append(thisX)
                    self._opt['cBetaMax'] = beta
                    break
                elif (thisError<maxErr):
                    # Error too small
                    if beta < 1e-3:
                        betaU = 0.
                    else:
                        betaU = beta
                elif (errFac*maxErr<thisError):
                    # Error too large
                    if beta < 1e-3:
                        betaL = 0.
                    else:
                        betaL = beta
                else:
                    assert 0
        
        return xc


    def doMatching(self, xSeries:List[np.ndarray]=None, options:dict={})->du.diffeomorphism:
        
        _opts = deepcopy(self._optMatching)
        _opts.update(options)

        repeatStart_ = _opts.get('repeatStart_')  # None

        if (_opts['matchDirections']):
            from interpolatePy import regularSpline
            
        additionalCenters = np.zeros((self._dim,0))
        additionalCenters2 = np.zeros((self._dim,0))
        
        if not ensureIdentityOrigin_ and mainParsDict_['convergeToZero_']: #Deactivate only if curves do not end at origin
            additionalCenters = np.hstack(( additionalCenters, np.zeros((self._dim,1)) ))
        if 'additionalCenters' in options.keys():
            additionalCenters = np.hstack((additionalCenters, options['additionalCenters']))
        if 'additionalCenters2' in options.keys():
            #['additionalCenters2'] -> first is source second is target
            additionalCenters2 = np.hstack((additionalCenters2, options['additionalCenters2'][0]))

        if xSeries is None:
            #Get series with current options
            xSeries = self.getSeries()
            # XSeries[0] is final target, Xseries[-1] is initial source -> inverse
            xSeries.reverse()
        if isinstance(xSeries[0][0], (list, tuple)):
            # The series also holds the velocities which are unnecessary here
            xSeries = [ [ax for ax,_ in thisSeries] for thisSeries in xSeries ]
            
        
        # If we are only interested in matching the directions without dynamics
        # we look at the equally splined curve
        if (_opts['matchDirections']):
            from modifiedEM import gaussianKernel
            
            xSeries = [ regularSpline(axS)[0] for axS in xSeries ]
            # All lines have the same length
            allIndList = np.hstack((0,np.cumsum(np.array([ax.shape[1] for ax in xSeries[0]]))))
            
            #Get a kernel
            xxxtmp = np.hstack(xSeries[-1])
            xlimtmp = np.mean(np.max(xxxtmp, axis=1)-np.min(xxxtmp, axis=1))

            #directionKernel = gaussianKernel(nVarI=1,Sigma=np.identity(self._dim)*xlimtmp/40.,mu=np.zeros((self._dim,1)),doCond=False)
            #influenceKernel = gaussianKernel(nVarI=1, Sigma=np.identity(self._dim)*xlimtmp/20., mu=np.zeros((self._dim,1)), doCond=False)
            # todo check if dimension factor is ok
            directionKernel = gaussianKernel(nVarI=1,Sigma=np.identity(self._dim)*(xlimtmp/100.**.5)**2*(2./self._dim),mu=np.zeros((self._dim,1)),doCond=False)
            #influenceKernel = gaussianKernel(nVarI=1,Sigma=np.identity(self._dim)*(xlimtmp/30.**.5)**2*(2./self._dim),mu=np.zeros((self._dim,1)),doCond=False)
            influenceKernel = gaussianKernel(nVarI=1,Sigma=np.identity(self._dim)*(xlimtmp/100.**.5)**2*(2./self._dim),mu=np.zeros((self._dim,1)),doCond=False)
            
            del xxxtmp, xlimtmp
            
            ## Additional stuff
            dirSeries = [ [computeDirections(ax, endOption='Zero') for ax in aS] for aS in xSeries ]
            
            finalX = np.hstack(xSeries[-1])
            finalDir = np.hstack(dirSeries[-1])

            checkStruct = KMeans(n_clusters=20,n_init=10,n_jobs=4)
            checkStruct.fit(finalX.T)
            
            checkCenters = checkStruct.cluster_centers_.T
            allCloseness = localTrajectoryDistance(directionKernel, influenceKernel, checkCenters, finalX, finalX, dS0=finalDir, dS1=finalDir)
            
            maxCloseness = 10.*np.mean(np.array(allCloseness))
            
            
        diffeo = du.diffeomorphism(self._dim, isPars=_opts['isPars'])

        errStruct = None

        for k in range(len(xSeries)-1):
            print("Computing sub-diffeo {0}".format(k))
            
            # Get the source
            if _opts['assumePerfect']:
                # For the next step it is assumed that the last diffeo was perfect so xtarget == diffeo.forwardTransformation(xsource)
                xsource = np.hstack(xSeries[k])
                xsourceL = deepcopy(xSeries[k])
            else:
                #xsource = diffeo.forwardTransform( repeatInit(xSeries[0], repeatStart_))
                xsourceL = [diffeo.forwardTransform(ax.copy()) for ax in xSeries[0]]
                xsource = repeatInit(xsourceL,repeatStart_)
            # Get the (intermediate) target
            xtargetL = deepcopy(xSeries[k+1])
            xtarget = repeatInit(xtargetL, repeatStart_)
            
            if _opts['matchDirections']:
                dirtargetL = deepcopy(dirSeries[k+1])
                dirtarget = repeatInit( dirtargetL, repeatStart_ )

            diffeok = du.diffeomorphism(self._dim,isPars=_opts['isPars'])

            xsourceLk = deepcopy(xsourceL)
            xsourcek = xsource.copy()
            
            dim, nPt = xsourcek.shape

            for i in range(_opts['nInterTrans']):
                print("Computing intertrans {0}".format(i))

                thisMinTrans = _opts['minTrans'] if (_opts['minTrans']>0) else -_opts['minTrans']*np.max(cNorm(xsource,kd=False))
                thisMinBase = _opts['minBase'] if (_opts['minBase']>0) else -_opts['minBase']*np.max(cNorm(xsource,kd=False))

                if (_opts['matchDirections']):
                    #xtarget = regularSpline(xtarget, indList=allIndList)[0]
                    #xsource = regularSpline(xsource, indList=allIndList)[0]
                    xtargetL = regularSpline(xtargetL)[0]
                    xsourceL = regularSpline(xsourceL)[0]
                    xtarget = repeatInit(xtargetL, repeatStart_)
                    xsource = repeatInit(xsourceL,repeatStart_)

                # Get current error
                err = xtarget-xsource
                if mainDoParallel_:
                    _XSourceShared[:nPt*dim] = xsource.ravel()

                # Quantify error
                if _opts['mode'].lower() in ('gmm', 'vargmm'):
                    if _opts['mode'].lower()=='gmm':
                        errStruct = GaussianMixture(n_components=_opts['maxNKernel'], covariance_type='spherical')
                    elif _opts['mode'].lower()=='vargmm':
                        errStruct = BayesianGaussianMixture(n_components=_opts['maxNKernel'], covariance_type='spherical')
                    else:
                        assert 0
    
                    # "learn"
                    errStruct.fit(np.vstack((xsource, err)).T)
    
                    #Get the centers and translations
                    centers = errStruct.means_.T[:self._dim,:]
                    translationsRaw = errStruct.means_.T[self._dim:,:]
                    prec = errStruct.precisions_
                    bases = np.sqrt(prec)
                elif (_opts['mode'].lower() in ('kmeans')) and False:
                    if _opts['mode'].lower() in ('kmeans'):
                        
                        # Get points with error
                        indErr = cNormSquare(err, kd=False, cpy=True)>thisMinTrans**2
                        
                        errStruct = KMeans(n_clusters=min(_opts['maxNKernel']*2, 4), n_jobs=1)
                        try:
                            errStruct.fit(np.vstack((xsource[:,indErr], err[:,indErr])).T)
                        except ValueError:
                            continue
                        
                        xMeans, tMeans = errStruct.cluster_centers_.T[:self._dim, :], errStruct.cluster_centers_.T[self._dim:,:]
                        tMeansN = cNorm(tMeans, kd=False)
                        tMeansNormed = tMeans/(tMeansN+epsFloat)
                        tMeansN = tMeansN.squeeze()
                        
                        while True and xMeans.shape[1]>3:
                            # Compute approximated sizes
                            thisDelaunay = Delaunay(xMeans.T)
                            bases = np.zeros((xMeans.shape[1]))
                            indices,indptr = thisDelaunay.vertex_neighbor_vertices
                            for l in range(xMeans.shape[1]):
                                # Distance to neighbors
                                bases[l] = np.min(cNorm(xMeans[:,indptr[indices[l]:indices[l+1]]]-xMeans[:,[l]],kd=False))/1.
                            
                            #Check of each point if its neighboors have the same error direction
                            for l in range(xMeans.shape[1]):
                                isClose = cNorm(xMeans-xMeans[:,[l]], kd=False) < bases+bases[l]
                                doBreak = False
                                for j in range(xMeans.shape[1]):
                                    if (l==j) or not isClose[j]:
                                        continue
                                    
                                    if (np.inner( tMeansNormed[:,l], tMeansNormed[:,j] ) > 0.9) and (tMeansN[j] < 1.25*tMeansN[l]) and (tMeansN[l] < 1.25*tMeansN[j]):
                                        newXMean = np.mean(xMeans[:,[j]],axis=1)
                                        newTMean = np.mean(tMeans[:,[j]],axis=1)
                                        xMeans = np.delete(xMeans, [j], axis=1)
                                        tMeans = np.delete(tMeans,[j],axis=1)
                                        xMeans[:,l] = newXMean
                                        tMeans[:,l] = newTMean
                                        tMeansN = cNorm(tMeans,kd=False)
                                        tMeansNormed = tMeans/(tMeansN+epsFloat)
                                        tMeansN = tMeansN.squeeze()
                                        doBreak = True
                                        break
                                if doBreak:
                                    break
                            if not doBreak:
                                break
                        thisDelaunay = Delaunay(xMeans.T)
                        bases = np.zeros((xMeans.shape[1]))
                        indices,indptr = thisDelaunay.vertex_neighbor_vertices
                        for l in range(xMeans.shape[1]):
                            # Distance to neighbors
                            bases[l] = np.min(cNorm(xMeans[:,indptr[indices[l]:indices[l+1]]]-xMeans[:,[l]],kd=False))/1.
                        #Get new translation directions
                        for l in range(xMeans.shape[1]):
                            thisFac = c1PolyKer(cNorm(xsource-xMeans[:,[l]], kd=False), bIn=bases[l], dIn=_opts['dValues'], eIn=0.)
                            thisFac /= np.sum(thisFac)
                            tMeans[:,l] = np.sum(err*thisFac,axis=1)
                            # ensure minimal base size
                            # c1PolyMinB = lambda tn,d,safeFac=0.5,e=0:-(tn/safeFac)*(2*e-2)/(1+2*d)
                            bases[l] = max([ bases[l], c1PolyMinB(cNorm(tMeans[:,[l]],kd=False), d=_opts['dValues'], safeFac=_opts['marginSingle']+2.*_opts['epsConstraints'],e=0.) ])
                        centers = xMeans
                        translationsRaw = tMeans
                        
                    else:
                        assert 0
                elif _opts['mode'].lower() in ('kmeans'):
                    if _opts['mode'].lower() in ('kmeans'):
                        #########
                        #   Actually used!
                        #########
                        # Get current error
                        xsourceCurrentL = deepcopy(xsourceL)
                        xsourceCurrent = xsource.copy()
                        maxBase = 2.*np.max(cNorm(xsource,kd=False)) if mainParsDict_['maxBaseVal_'] is None else mainParsDict_['maxBaseVal_']
                        
                        centers = []
                        translationsRaw = []
                        bases = []
                        
                        includeInd = np.ones((xsource.shape[1],)).astype(np.bool_)
                        includeIndAdd = np.ones((additionalCenters.shape[1],)).astype(np.bool_)
                        includeIndAdd2 = np.ones((additionalCenters2.shape[1],)).astype(np.bool_)
                        if additionalCenters.size:
                            additionalCentersSourceK = diffeok.forwardTransform(diffeo.forwardTransform(additionalCenters.copy()))
                        if additionalCenters2.size:
                            additionalCenters2SourceK = diffeok.forwardTransform(diffeo.forwardTransform(additionalCenters2.copy()))
                        
                        thisTrans = pd.localPolyMultiTranslation(self._dim, np.zeros((self._dim,0)), np.zeros((self._dim,0)),bases=np.zeros((0,)), dValues=np.zeros((0,)), eValues=np.zeros((0,)), isPars=_opts['isPars'])
                        
                        for blubTemp in range(_opts['maxNKernel']):
                            print("Checking kernel {0}".format(blubTemp))
                            err = xtarget-xsourceCurrent
                            # Get points with error
                            indErr = np.logical_and(includeInd, cNormSquare(err,kd=False,cpy=True) > thisMinTrans**2)

                            if np.sum(indErr)<(_opts['maxNKernelErrStruct']/2):
                                print('Not enough points remain')
                                break
                            
                            if mainDoParallel_:
                                if _opts['matchDirections']:
                                    ###_ErrShared[:nPt*dim] = xsourceCurrent.ravel()
                                    _ErrShared[:nPt*dim] = err.ravel()
                                else:
                                    _ErrShared[:nPt*dim] = err.ravel()
                            
                            if _opts['matchDirections']:
                                #dirsourceCurrent = computeDirections(xsourceCurrent)
                                dirsourceCurrentL = [computeDirections(ax) for ax in xsourceCurrentL]
                                dirsourceCurrent = repeatInit(dirsourceCurrentL, repeatStart_)
                            
                            if 0:
                                bestErr = -1e300
                                try:
                                    for thisKernelNumber in range(4, _opts['maxNKernelErrStruct']):
                                        errStruct = KMeans(n_clusters=thisKernelNumber,n_init=4,n_jobs=2)
                                        errStruct.fit(np.vstack((xsource[:,indErr],err[:,indErr])).T)
                                        thisErr = errStruct.score(np.vstack((xsource[:,indErr],err[:,indErr])).T)
                                        if (thisErr < 0.8*bestErr):
                                            break
                                        bestErr = thisErr
                                except ValueError:
                                    continue
                            else:
                                errStruct = KMeans(n_clusters=_opts['maxNKernelErrStruct'],n_init=4,n_jobs=2) if (errStruct is None) else errStruct
                                try:
                                    errStruct.fit(np.vstack((xsource[:,indErr],mainParsDict_['errorCoeff_']*err[:,indErr])).T)
                                except ValueError:
                                    continue
        
                            xMeans,tMeans = errStruct.cluster_centers_.T[:self._dim,:],errStruct.cluster_centers_.T[self._dim:,:]
                            tMeans /= mainParsDict_['errorCoeff_']
                            
                            #if not ensureIdentityOrigin_:
                            #    #Append the transformation to put origin on origin
                            #    originPrime = thisTrans.forwardTransform(diffeok.forwardTransform(diffeo.forwardTransform(np.zeros((self._dim,1)))))
                            #    xMeans = np.hstack((xMeans, originPrime))
                            #    tMeans = np.hstack((tMeans, -originPrime))
                            if additionalCenters.size:
                                additionalCentersPrime = thisTrans.forwardTransform(additionalCentersSourceK.copy())
                                xMeans = np.hstack((xMeans,additionalCentersSourceK[:,includeIndAdd]))
                                tMeans = np.hstack((tMeans, -additionalCentersPrime[:,includeIndAdd]+additionalCenters[:,includeIndAdd]))
                            if additionalCenters2.size:
                                additionalCenters2Prime = thisTrans.forwardTransform(additionalCenters2SourceK.copy())
                                xMeans = np.hstack((xMeans,additionalCenters2SourceK[:,includeIndAdd2]))
                                tMeans = np.hstack((tMeans, -additionalCenters2Prime[:,includeIndAdd2]+options['additionalCenters2'][1][:,includeIndAdd2]))
                            
                            tMeansN = cNorm(tMeans,kd=False) + epsFloat
                            
                            if _opts['matchDirections']:
                                # Using the modified glaunes distance on the found centers
                                xMeansPrime = thisTrans.forwardTransform(xMeans)
                                thisAllDist = np.array(localTrajectoryDistance(directionKernel,influenceKernel,xMeansPrime,xsourceCurrent,xtarget,dS0=dirsourceCurrent,dS1=dirtarget))
                                #thisAllDist = np.zeros((xMeans.shape[1],))
                            
                            # Check if conditions fullfilled
                            oktrans = tMeansN>thisMinTrans
                            
                            if _opts['matchDirections']:
                                # Check additional cond
                                # closeness with modified glaunes measure
                                tooClose = thisAllDist>maxCloseness
                                
                                # diretions are too close
                                sameDirecVal = np.empty(tooClose.shape)
                                sameDirec = np.empty_like(tooClose)
                                #orthDirec = np.empty_like(tooClose)
                                
                                for k in range(xMeans.shape[1]):
                                    directionKernel.mu = xMeansPrime[:,[k]]
                                    thisCurveDir = np.sum(directionKernel.getWeights(xsourceCurrent)*dirsourceCurrent,axis=1,keepdims=True)
                                    thisCurveDir /= (cNorm(thisCurveDir,kd=False)+epsFloat)
                                    sameDirecVal[k] = np.sum(thisCurveDir*(tMeans[:,[k]]/tMeansN[k]))
                                    sameDirec[k] = sameDirecVal[k] > 10.975
                                
                                # assemble all flags
                                print("tooClose {0} \n sameDirec {1}".format(tooClose, sameDirec))
                                #oktrans = np.logical_and( oktrans, np.logical_not( np.logical_and(tooClose, sameDirec) ) )
                                oktrans = np.logical_and(oktrans,np.logical_not(np.logical_or(tooClose,sameDirec)))
                            
                            # Only use ok transitions but first check if any left
                            if not np.any(oktrans):
                                #All possibilities do not match conditions
                                break
                            
                            xMeans = xMeans[:, oktrans]
                            tMeans = tMeans[:, oktrans]
                            tMeansN = tMeansN[oktrans]
                            
                            if (not _opts['matchDirections']):
                                thisI = np.argmax(tMeansN)
                            elif (True or (not _opts['matchDirections'])):
                            #if not _opts['matchDirections']:
                                #thisI = np.argmax(tMeansN)
                                #Weight with angle
                                sameDirecVal = sameDirecVal[oktrans]
                                sameDirecVal = np.abs(sameDirecVal) # 1-> point in same direc
                                thisI = np.argmax(tMeansN*(mainParsDict_['directionCostCoeff_']-sameDirecVal))
                            else:
                                thisAllDist = thisAllDist[oktrans]
                                thisI = np.argmin(thisAllDist)
                                
                            thisM = xMeans[:,[thisI]]
                            thisT = tMeans[:,[thisI]]
                            allThisT = []
                            allThisM = []
                            allBases = []
                            allError = []
                            startFac = thisMinTrans/tMeansN[thisI]
                            stopFac = max([_opts['transFac'], 1.1*startFac])
                            #allFac = np.hstack((0.,np.linspace(startFac, stopFac, 10)))
                            allFac = np.linspace(startFac,stopFac,10)
                            if mainDoParallel_:
                                # TBD
                                minBaseL = [float(max([1.05*thisMinBase, c1PolyMinB(cNorm(thisT*afac,kd=False), d=_opts['dValues'], safeFac=1.-(_opts['marginSingle']+5.*_opts['epsConstraints']),e=0.)])) for afac in allFac]
                                minBaseS = []
                                allFacS = []
                                parsForStar = []
                                
                                for aBase, aFac in zip(minBaseL, allFac):
                                    if ensureIdentityOrigin_:
                                        thisMaxBase = min([maxBase,0.99*cNorm(thisM+_opts['transCenterFac']*thisT*aFac,kd=False)]) # Make sure no kernel translates the origin
                                    else:
                                        # Just take some huge value
                                        thisMaxBase = 10.*maxBase  if mainParsDict_['maxBaseVal_'] is None else mainParsDict_['maxBaseVal_']
                                    
                                    if  thisMaxBase>aBase:
                                        allFacS.append(aFac)

                                        allThisT.append(thisT*aFac)
                                        allThisM.append(thisM+_opts['transCenterFac']*thisT*aFac)
                                        if _opts['matchDirections']:
                                            ###parsForStar.append( [aBase, thisMaxBase, allThisM[-1], allThisT[-1], float(_opts['dValues']), _opts['baseRegVal'], dim, nPt, allIndList] )
                                            parsForStar.append([aBase,thisMaxBase,allThisM[-1],allThisT[-1],float(_opts['dValues']),_opts['baseRegVal'],dim,nPt])
                                        else:
                                            parsForStar.append([aBase,thisMaxBase,allThisM[-1],allThisT[-1],float(_opts['dValues']),_opts['baseRegVal'],dim,nPt])

                                #def fBaseMult( baseMin, baseMax, thisM, thisT, thisD, thisReg, dim, nPt):#
                                if _opts['matchDirections']:
                                    ###allBasesNErrors = minimizeWorkers.starmap( fBaseMultDirections, parsForStar )
                                    allBasesNErrors = minimizeWorkers.starmap(fBaseMult,parsForStar)
                                else:
                                    allBasesNErrors = minimizeWorkers.starmap( fBaseMult, parsForStar )

                                allBases = [aa[0] for aa in allBasesNErrors]
                                allError = [aa[1] for aa in allBasesNErrors]
                                
                            else:
                                for fac in allFac:
                                    minBase = float(max([1.05*thisMinBase, c1PolyMinB(cNorm(thisT*fac,kd=False), d=_opts['dValues'], safeFac=1.-(_opts['marginSingle']+5.*_opts['epsConstraints']),e=0.)]))
                                    #here
                                    if _opts['matchDirections']:
                                        ###fCost = lambda newBase:trajDist(xtarget-
                                        ###                                regularSpline(xsourceCurrent+
                                        ###                                  du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])), indList=allIndList)[0])\
                                        ###                       -_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small onesfCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
                                        fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
                                    else:
                                        fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones

                                    if ensureIdentityOrigin_:
                                        thisMaxBase = min([maxBase, 0.99*cNorm(thisM+_opts['transCenterFac']*thisT*fac,kd=False)])
                                    else:
                                        thisMaxBase = 10.*maxBase if mainParsDict_['maxBaseVal_'] is None else mainParsDict_['maxBaseVal_']
                                    
                                    if thisMaxBase>minBase:
                                        bestBase,error,ok,_dummy = fminbound(fCost,minBase,thisMaxBase,full_output=True)
                                        allBases.append(bestBase)
                                        allThisT.append(thisT*fac)
                                        allThisM.append(thisM+_opts['transCenterFac']*thisT*fac)
                                        allError.append(error)
                            
                            if len(allThisT):
                                thisI = np.argmin(allError)
                                centers.append(allThisM[thisI])
                                translationsRaw.append(allThisT[thisI])
                                bases.append(allBases[thisI])
                                
                                #Add
                                thisTrans.addTransition(centers[-1], translationsRaw[-1], bases=bases[-1], dValues=_opts['dValues'], eValues=0.)
                                # Ensure
                                thisTrans.enforceMargins(marginSingle=_opts['marginSingle']+5.*_opts['epsConstraints'],marginOverlap=_opts['marginOverlap']+5.*_opts['epsConstraints'])
                                
                                #Update include ind
                                includeInd = np.logical_and(includeInd, c1PolyKer( cNorm(xsource-centers[-1], kd=False), bIn=float(bases[-1]), dIn=float(_opts['dValues']), eIn=0.) < _opts['includeMaxFac'])
                                if (includeIndAdd.size):
                                    includeIndAdd = np.logical_and(includeIndAdd, c1PolyKer( cNorm(additionalCentersSourceK-centers[-1], kd=False), bIn=float(bases[-1]), dIn=float(_opts['dValues']), eIn=0.) < _opts['includeMaxFac'])
                                if (includeIndAdd2.size):
                                    includeIndAdd2 = np.logical_and(includeIndAdd2, c1PolyKer( cNorm(additionalCenters2SourceK-centers[-1], kd=False), bIn=float(bases[-1]), dIn=float(_opts['dValues']), eIn=0.) < _opts['includeMaxFac'])
                                # Modify current source
                                #xsourceCurrent = thisTrans.forwardTransform(xsource)
                                xsourceCurrentL = [thisTrans.forwardTransform(ax) for ax in xsourceL]
                                
                                if _opts['matchDirections']:
                                    #xsourceCurrent = regularSpline(xsourceCurrent, indList=allIndList)[0]
                                    xsourceCurrentL = regularSpline(xsourceCurrentL)[0]

                                xsourceCurrent = repeatInit(xsourceCurrentL,repeatStart_)
                                
                                
                            else:
                                # No new kernel is added -> No changement -> break
                                break
                        try:
                            centers = np.hstack(centers)
                            translationsRaw = np.hstack(translationsRaw)
                            bases = np.hstack(bases)
                        except (ValueError, IndexError):
                            centers = np.zeros((self._dim,0))
                            translationsRaw = np.zeros((self._dim,0))
                            bases = np.zeros((0,))
                            
                    else:
                        assert 0
                else:
                    assert 0
                
                if centers.size == 0:
                    continue
                
                # Eliminate all bases that are too small
                indIn = bases >= thisMinBase
                centers = centers[:,indIn]
                translationsRaw = translationsRaw[:,indIn]
                bases = bases[indIn]
                
                if centers.size == 0:
                    continue

                # Check if overlap and do scaling if so
                if 0:
                    doOverlap = np.zeros((bases.size, bases.size)).astype(np.bool_)
                    for l in range(bases.size):
                        doOverlap[l,:] = cNorm(centers-centers[:,[l]], kd=False) > bases+bases[l]
                        doOverlap[l,l] = False
                    doOverlap = np.max(doOverlap, axis=0).reshape((1,-1))
                    translations = translationsRaw.copy()
                    translations *= doOverlap*_opts['overlapFac']+np.logical_not(doOverlap)*_opts['transFac']
                else:
                    translations = translationsRaw.copy()

                #Checking if translations large enough
                indIn = cNorm(translations, kd=False) > thisMinTrans
                centers = centers[:,indIn]
                translations = translations[:,indIn]
                bases = bases[indIn]
                if centers.size == 0:
                    continue

                #Get the corresponding transition
                thisTrans = pd.localPolyMultiTranslation(self._dim, centers, translations, bases, dValues=np.ones((bases.size,))*_opts['dValues'], eValues=np.zeros((bases.size,)), isPars=_opts['isPars'])
                _ = thisTrans.enforceMargins(marginSingle=_opts['marginSingle']+.5*_opts['epsConstraints'], marginOverlap=_opts['marginOverlap']+.5*_opts['epsConstraints'], minTransNorm = thisMinTrans)
                print("Enforce margins returned after {0} iterations".format(_))
                
                if not bool(thisTrans.nTrans):
                    # The final translations are too small skip rest
                    print("Final transformation respecting boundaries has zero kernels")

                #Add it to the k diffeo
                diffeok.addTransformation(thisTrans, True)

                #Apply it to source
                #xsource = diffeok.forwardTransform(xsourcek)
                xsourceL = [diffeok.forwardTransform(ax) for ax in xsourceLk]
                if _opts['matchDirections']:
                    xsourceL = regularSpline(xsourceL)[0]
                xsource = repeatInit(xsourceL, repeatStart_)

            # Do the optimization if demanded
            if _opts['interOpt'] and len(diffeok._transformationList):
                diffeok.optimize(xsourcek, xtarget, options=_opts)
            
            # Concatenate the general diffeo and diffeok
            if len(diffeok._transformationList):
                diffeo.addTransformation( diffeok._transformationList, diffeok._directionList )
        
        
        # Add an additional transformation if the origin is not preserved by default
        if not ensureIdentityOrigin_:
            thisCenter = diffeo.forwardTransform(np.zeros((self._dim,1)))
            thisT = -thisCenter
            
            minBase = float(max([1.05*thisMinBase,c1PolyMinB(cNorm(thisT,kd=False),d=_opts['dValues'],safeFac=1.-(_opts['marginSingle']+5.*_opts['epsConstraints']),e=0.)]))
            
            #Recompute source and target
            xsourceL = [diffeo.forwardTransform(ax) for ax in xSeries[0]]
            xtargetL = deepcopy(xSeries[-1])
            if _opts['matchDirections']:
                xsourceL = regularSpline(xsourceL)[0]
                xtargetL = regularSpline(xtargetL)[0]
            xsource = repeatInit(xsourceL, repeatStart_)
            xtarget = repeatInit(xtargetL, repeatStart_)
            
            err = xtarget - xsource
            
            # here
            if _opts['matchDirections']:
                ###fCost = lambda newBase:trajDist(xtarget-
                ###                                regularSpline(xsourceCurrent+
                ###                                  du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])), indList=allIndList)[0])\
                ###                       -_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small onesfCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM+_opts['transCenterFac']*thisT*fac,thisT*fac,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
                fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM,thisT,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones
            else:
                fCost = lambda newBase:trajDist(err-du.pd.c1PolyTrans(xsource,thisM,thisT,float(newBase),float(_opts['dValues'])))-_opts['baseRegVal']*newBase  # deformationNorm + baseRegVal*newBase since larger bases are more desirable than small ones

            thisMaxBase = max(10.*maxBase, 10.*minBase)
            bestBase,error,ok,_dummy = fminbound(fCost,minBase,thisMaxBase,full_output=True)
            assert ok==0
            thisTrans = pd.localPolyMultiTranslation(self._dim,thisCenter,thisT,np.array([bestBase]),dValues=np.array([_opts['dValues']]),eValues=np.array([0.]),isPars=_opts['isPars'])
            diffeo.addTransformation(thisTrans, True)
        
        # Do the final total optimization
        if _opts['finalOpt']:
            diffeo.optimize(np.hstack(xSeries[0]), np.hstack(xSeries[-1]), options=_opts)
        
        return diffeo








            
            
            
            
        
        
        
        
        
        
    
    
    
        
        
        























