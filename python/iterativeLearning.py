from coreUtils import *

import gmmDynamics as gmmDyn
import modifiedEM as mEM
import diffeoUtils as du
import greedyInsertion as greedyInsert

import antisymmetricDynamics as asymD

from otherUtils import learningMove

import plotUtils as pu

import collections

import interpolate as myIter

def myUpdate(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = myUpdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def iterativeGMMLearner(x:List[np.ndarray], v:List[np.ndarray], t:List[np.ndarray], mode:Union["asym","gmm"]="gmm", options={}, saveName=None ):
    """
    
    :param x: Positions, columns are sample points rows are dims
    :param v: Associated velocity
    :param t: cumulative time passed since start
    :param options:
    :return:
    """
    dim = x[0].shape[0]
    
    #Default options
    _options={  'relTolDist':1.e-2, 'distFunc':compMSE, 'skipStartSim':0, 'skipStartSimAll':[], 'skipEndConvCalc':-10, 'alpha':-0.2, 'beta':-np.Inf,
                "breakRadius":-.1,"radialMinConv":None,
                'acceptTol':-5, 'mseAccept':-0.2, 'componentPenalty':1.2,
                'doPlot':True,
                'nPartialGMM':10,
                'nBetaTest':10,
                'greedyOptions':{"nKMax":5, "nPartial":10, "add2Sigma":1., "iterMax":20, "relTol":1e-4, "absTol":1e-4,
                    "doPlot":False, "speedUp":False, "interOpt":True, "reOpt":False, "convFac":1.},
                'greedyAsymOptions':{"nPartial":10, "nKMax":8, "add2Sigma":5.e-1, "relTol":1e-1, "absTol":.5e-1,"convFac":1.,
                                     "relTolLogLike":1.e-2,"absTolLogLike":1.e-2,"relTolMSE":1.e-2,
                                      "interOpt":False,"reOpt":True,"iterMax":10, "addKWARGS":None}
              }
    #Update
    _options = myUpdate(_options, options)
    #_options.update(options)
    
    # Step 1: Find the "simplest" unconstrained dynamics that takes all demonstrations to 0
    
    xAll = np.hstack((x))
    vAll = np.hstack((v))
    tAll = np.hstack((t))
    tAlli = []
    xIn = np.zeros((dim,0))
    xOut = np.zeros((dim,0))

    if _options['doPlot'] or 1:
        Ngrid = 150
        gridSpacex = np.linspace(np.min(xAll[0,:])*1.1-5,np.max(xAll[0,:])*1.1+5,Ngrid)
        gridSpacey = np.linspace(np.min(xAll[1,:])*1.1-5,np.max(xAll[1,:])*1.1+5,Ngrid)
        xv,yv = np.meshgrid(gridSpacex,gridSpacey)
        Xgrid = np.vstack((xv.ravel(),yv.ravel()))
        del xv, yv
        
    
    
    #Do some updates
    _options['breakRadius'] = -np.max(cNorm(xAll, kd=False))*_options['breakRadius'] if (_options['breakRadius'] < 0.) else _options['breakRadius']
    _options['acceptTol'] = -np.max(cNorm(xAll, kd=False))*_options['acceptTol']/100. if (_options['acceptTol'] < 0.) else _options['acceptTol']
    _options['mseAccept'] = compMSE(xAll*_options['mseAccept']) if (_options['mseAccept'] < 0.) else _options['mseAccept']
    
    baseVel = 0.
    betaMax = -np.Inf
    betaMaxT = 0.
    alpha0 = _options['alpha']
    for k, [ax, at] in enumerate(zip(x, t)):
        thisT = at[0]-at #Inverse time direction
        tAlli.append((thisT))
        thisSkipStart = _options['skipStartSim'] if (_options['skipStartSim']>0) else -int(x[k].shape[1]*_options['skipStartSim']/100)
        _options['skipStartSimAll'].append(thisSkipStart)
        xIn = np.hstack((xIn, ax[:,[thisSkipStart]]))
        xOut = np.hstack((xOut,ax[:,[-1]]))
        baseVel += float(cNorm(ax[:,[thisSkipStart]],kd=False)/thisT[-1])
        #Compute the minimal convergence
        thisSkipEnd = _options['skipEndConvCalc'] if (_options['skipEndConvCalc'] > 0) else -int(x[k].shape[1]*_options['skipEndConvCalc']/100)
        thisBetaMax = (2.*np.sum(v[k][:,:-thisSkipEnd]*x[k][:,:-thisSkipEnd],axis=0)/np.sum(np.square(x[k][:,:-thisSkipEnd]),axis=0)-_options['alpha'])/at[:-thisSkipEnd]
        argMax = np.argmax(thisBetaMax)
        if betaMax<thisBetaMax[argMax]:
            betaMax = thisBetaMax[argMax]
            betaMaxT = at[argMax]
    _options['beta'] = betaMax
    
    baseVel /= xOut.shape[1]
    
    #Compute and set the base velocity
    
    
    # Increase number of kernels until all demonstrations converge in time for unconstrained learning
    maxNumKernel = 12
    if mode.lower()=="gmmsplit":
        bestDynRad = None
        bestDynTang = None
    else:
        bestDyn = None # Dynamics
    bestDiffeo = du.diffeomorphism(xAll.shape[0]) # Transformation
    bestDiffeoCtrl = None # Best control
    
    lastMSEOverall = 1e320
    
    # shorthands for clarity
    # Demonstration space
    xd = xAll
    vd = vAll
    # Control space
    xc = np.empty_like(xd)
    vc = np.empty_like(vd)
    
    while True:
        xc,vc = bestDiffeo.inverseTransformJac(xd, vd,outx=xc,outv=vc)
        if mode.lower() == "gmmsplit":
            xcnorm = xc/(cNorm(xc)+epsFloat)
            vcrad = sum(xcnorm*vc,axis=0,keepdims=True)
            vctang = vc-vcrad*xcnorm

    
        #Reset the dynamics when the transformation changed. Check if it can be reused
        bestDyn = None
        lastMSESim = 1e320
        isDone = False
        for thisnKMax in range(1,maxNumKernel):
            if isDone:
                break
            
            # Get the best gmm with current maximal number of kernels using the last best as warmstart
            # Learn in control space
            _options['greedyOptions']['nKMax'] = thisnKMax
            _options['greedyAsymOptions']['nKMax'] = thisnKMax
            if mode.lower() == "gmm":
                bestDyn = greedyInsert.greedyEMCPU(np.vstack((xc,vc)),nVarI=xAll.shape[0], warmStartGMM=bestDyn, **_options['greedyOptions'])
                bestDiffeoCtrl = gmmDyn.gmmDiffeoCtrl(bestDiffeo,bestDyn,True,breakRadius=_options['breakRadius'], radialMinConv=_options['radialMinConv'])
                # Set base velocity
                bestDiffeoCtrl._baseVelocity = baseVel

                # Get the convergence function
                thisLearner = learningMove(bestDiffeoCtrl,alpha=_options['alpha'],beta=_options['beta'],x=xAll,t=tAll)
                # Set
                bestDiffeoCtrl.radialMinConv = thisLearner
            elif mode.lower() == "gmmsplit":
                bestDynRad = greedyInsert.greedyEMCPU(np.vstack((xc,vcrad)),nVarI=xAll.shape[0],warmStartGMM=bestDynRad,**_options['greedyOptions'])
                bestDynTang = greedyInsert.greedyEMCPU(np.vstack((xc,vctang)),nVarI=xAll.shape[0],warmStartGMM=bestDynTang,**_options['greedyOptions'])

                bestDiffeoCtrl = gmmDyn.gmmDiffeoCtrlSplit(bestDiffeo, bestDynTang, bestDynRad, baseDir=True, breakRadius=_options['breakRadius'], radialMinConv=_options['radialMinConv'])
                # Set base velocity
                bestDiffeoCtrl._baseVelocity = -baseVel

                # Get the convergence function
                thisLearner = learningMove(bestDiffeoCtrl,alpha=_options['alpha'],beta=_options['beta'],x=xAll,t=tAll)
                # Set
                bestDiffeoCtrl.radialMinConv = thisLearner

            elif  mode.lower() == "asym":

                #findGreedyDynamicsDiffeo(x: np.ndarray,v: np.ndarray,learnRadial: bool = True,radialMinConv = -0.001,diffeo = None,options = {}):
                bestDiffeoCtrl = asymD.findGreedyDynamicsDiffeo(xc, vc, learnRadial=True, radialMinConv=0., diffeo=bestDiffeo, options=_options['greedyAsymOptions'], diffeoCtrl=bestDiffeoCtrl)

                # Get the convergence function
                thisLearner = learningMove(bestDiffeoCtrl,alpha=_options['alpha'],beta=_options['beta'],x=xAll,t=tAll)
                # Set
                bestDiffeoCtrl.radialMinConv = thisLearner
                bestDiffeoCtrl.breakVelocity = -baseVel
                bestDiffeoCtrl.breakRadius = _options['breakRadius']
            else:
                assert 0, "Mode unknown"
            

            # Check if all converge with reasonable error for some value from [betaMax, 0]
            isDone = False
            for kk, aAlpha in enumerate(np.linspace(alpha0, -1.5, 2)):
                if isDone:
                    break

                # Set this beta
                bestDiffeoCtrl.radialMinConv._alpha = aAlpha
                bestDiffeoCtrl.radialMinConv._beta = betaMax-aAlpha/betaMaxT


                # Replay all initial positions and increase convergence if necessary
                if _options['doPlot'] or kk==0:
                    fff,aaa = pu.plt.subplots(3,1)
                    aaa[0].plot(xd[0,:],xd[1,:],'.k')
                    aaa[1].plot(xc[0,:],xc[1,:],'.k')
                    Vd = bestDiffeoCtrl.getDemSpaceVeloctiy(Xgrid)
                    Vc = bestDiffeoCtrl.getCtrlSpaceVelocity(Xgrid)
                    Vdn = cNorm(Vd,kd=False)
                    Vcn = cNorm(Vc,kd=False)
                    Vdn *= 2./np.max(Vdn)
                    Vcn *= 2./np.max(Vcn)
                    aaa[0].streamplot(gridSpacex,gridSpacey,Vd[0,:].reshape((Ngrid,Ngrid)),
                                      Vd[1,:].reshape((Ngrid,Ngrid)),linewidth=Vdn.reshape((Ngrid,Ngrid)))
                    aaa[1].streamplot(gridSpacex,gridSpacey,Vc[0,:].reshape((Ngrid,Ngrid)),
                                      Vc[1,:].reshape((Ngrid,Ngrid)),linewidth=Vcn.reshape((Ngrid,Ngrid)))
                    del Vd,Vc
                
                
                #Do the computations for the given alpha
                allMSEBeta = np.empty((_options['nBetaTest'],))
                allConvergedList = np.empty((_options['nBetaTest'],)).astype(np.bool_)
                allBeta = np.hstack((np.linspace(betaMax-aAlpha/betaMaxT,.0,_options['nBetaTest']))) #Adjust for alpha
                for k, aBeta in enumerate(allBeta):
                    #Set this beta
                    bestDiffeoCtrl.radialMinConv._beta = aBeta
                    thisMSESim = []
                    allConverged = True
                    for m in range(xIn.shape[1]):
                        thisX, _ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],tAlli[m][_options['skipStartSimAll'][m]:],inInDem=True,outInCtrl=False,outInDem=True,returnVel=False)
                        # Get mse
                        thisMSESim.append( compMSE(thisX-x[m][:,_options['skipStartSimAll'][m]:]) )
                        # Converged?
                        allConverged = allConverged and cNorm(xOut[:,[m]]-thisX[:,[-1]])<_options['acceptTol']
                        if _options['doPlot'] or (kk==0 and k==0):
                            thisXd, thisXc, _ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],tAlli[m][_options['skipStartSimAll'][m]:], inInDem=True,outInCtrl=True,outInDem=True, returnVel=False)
                            aaa[0].plot(thisXd[0,:], thisXd[1,:], 'r')
                            if 0:
                                errr = x[m][:,_options['skipStartSimAll'][m]:]-thisXd
                                pu.myQuiver(aaa[0], thisXd[:,0:-1:20], errr[:,0:-1:20], 'g')
                            aaa[1].plot(thisXc[0,:],thisXc[1,:],'r')
                    
                    allMSEErr = sum(thisMSESim)/xIn.shape[1] # all(map(lambda aMSE: aMSE < _options['mseAccept'], thisMSESim)) and sum(thisMSESim)/xIn.shape[1] < .5*_options['mseAccept']
                    allMSEBeta[k] = allMSEErr
                    allConvergedList[k] = allConverged
                
                if not np.all(  np.logical_or(allMSEBeta > _options['mseAccept'], np.logical_not(allConvergedList) ) ): #not (allConverged and allMSEErr):
                    # The representation is not good enough with any of the tested beta values
                    isDone = True
                    continue
                
                if (k == _options['nBetaTest']-1) and (allMSEErr <= _options['mseAccept']):
                    # If the error is small but it does not converge -> increase enforced convergence rate
                    if _options['doPlot']:
                        fff,aaa = pu.plt.subplots(3,1)
                        aaa[0].plot(xd[0,:],xd[1,:],'.k')
                        aaa[1].plot(xc[0,:],xc[1,:],'.k')
                    while True:
                        bestDiffeoCtrl.radialMinConv._alpha -= 0.05
                        bestDiffeoCtrl.radialMinConv._alpha *= 1.5
    
                        thisMSESim = []
                        allConverged = True
                        for m in range(xIn.shape[1]):
                            thisX,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],tAlli[m][_options['skipStartSimAll'][m]:], inInDem=True,outInCtrl=False,outInDem=True, returnVel=False)
                            # Get mse
                            thisMSESim.append(compMSE(thisX-x[m][:,_options['skipStartSimAll'][m]:]))
                            # Converged?
                            allConverged = allConverged and cNorm(xOut[:,[m]]-thisX[:,[-1]]) < _options['acceptTol']
                            if _options['doPlot']:
                                thisXd,thisXc,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],
                                                                               tAlli[m][_options['skipStartSimAll'][m]:],
                                                                               inInDem=True,outInCtrl=True,outInDem=True,
                                                                               returnVel=False)
                                aaa[0].plot(thisXd[0,:],thisXd[1,:],'r')
                                if 0:
                                    errr = x[m][:,_options['skipStartSimAll'][m]:]-thisXd
                                    pu.myQuiver(aaa[0],thisXd[:,0:-1:20],errr[:,0:-1:20],'g')
                                aaa[1].plot(thisXc[0,:],thisXc[1,:],'r')
    
                        allMSEErr = sum(thisMSESim)/xIn.shape[1]
                        if (allMSEErr > _options['mseAccept']):
                            # Convergence could not be achieved for reasonable error
                            break
                        elif allConverged and (allMSEErr <= _options['mseAccept']):
                            allMSEBeta = np.array([[allMSEErr]])
                            allConvergedList = [allConverged]
                            break
            
                if np.all(  np.logical_or(allMSEBeta > _options['mseAccept'], np.logical_not(allConvergedList) ) ): #not (allConverged and allMSEErr):
                    # The representation is not good enough with any of the tested beta values
                    continue
                #Else -> Done
                isDone = True

        #Search highest convergence rate for which all points converge and error does not increase significantly
        betaL = -0.01
        allMSEBeta[np.logical_not(allConvergedList)] = np.Inf
        betaU = allBeta[np.argmin(allMSEBeta)]

        if _options['doPlot']:
            fff,aaa = pu.plt.subplots(2,1)
            ffff,aaaa = pu.plt.subplots(2,1)
        while (betaU-betaL>betaMax/5000) and ((betaU-betaL)>1e-4):
            #Do all init pos
            aBeta = 0.2*betaU+0.8*betaL
            bestDiffeoCtrl.radialMinConv._beta = aBeta

            thisMSESim = []
            allConverged = True
            for m in range(xIn.shape[1]):
                if not allConverged:
                    break
                thisX,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],tAlli[m][_options['skipStartSimAll'][m]:],
                                                       inInDem=True,outInCtrl=False,outInDem=True,returnVel=False)
                # Get mse
                thisMSESim.append(compMSE(thisX-x[m][:,_options['skipStartSimAll'][m]:]))
                # Converged?
                allConverged = allConverged and bool(cNorm(xOut[:,[m]]-thisX[:,[-1]]) < _options['acceptTol'])
                if _options['doPlot']:
                    thisXd,thisXc,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[m]],tAlli[m][_options['skipStartSimAll'][m]:],
                                                                   inInDem=True,outInCtrl=True,outInDem=True,
                                                                   returnVel=False)
                    aaa[0].plot(thisXd[0,:],thisXd[1,:],'r')
                    aaa[1].plot(thisXc[0,:],thisXc[1,:],'r')

            allMSEErr = sum(thisMSESim)/xIn.shape[1] < _options['mseAccept']  # all(map(lambda aMSE: aMSE < _options['mseAccept'], thisMSESim)) and sum(thisMSESim)/xIn.shape[1] < .5*_options['mseAccept']

            if _options['doPlot']:
                if allConverged:
                    aaaa[0].plot([aBeta], [sum(thisMSESim)/xIn.shape[1]], 'og')
                else:
                    aaaa[0].plot([aBeta],[sum(thisMSESim)/xIn.shape[1]],'xr')
            if (allConverged and allMSEErr):
                # The representation is still good enough
                betaU = aBeta
            else:
                betaL = aBeta
            
            
            #Redo trajectory generation to obtain points for matching
            bestDiffeoCtrl.radialMinConv._beta = betaU
            xSim = np.empty_like(xd)
            currentInd = 0
            fff,aaa = pu.plt.subplots(2,1)
            aaa[0].plot(xd[0,:],xd[1,:],'.k')
            aaa[1].plot(xc[0,:],xc[1,:],'.k')
            for l in range(xIn.shape[1]):
                thisX,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[l]],tAlli[l][_options['skipStartSimAll'][m]:], inInDem=True,outInCtrl=False,outInDem=True,returnVel=False)
                xSim[:, currentInd:currentInd+thisX.shape[1]] = thisX
                currentInd += thisX.shape[1]
                if _options['doPlot']:
                    thisXd,thisXc,_ = bestDiffeoCtrl.getTrajectory(xIn[:,[l]],tAlli[l][_options['skipStartSimAll'][l]:], inInDem=True,outInCtrl=True,outInDem=True, returnVel=False)
                    aaa[0].plot(thisXd[0,:],thisXd[1,:],'g')
                    aaa[1].plot(thisXc[0,:],thisXc[1,:],'g')
                    if 1:
                        errr = x[l][:,_options['skipStartSimAll'][l]:]-thisXd
                        pu.myQuiver(aaa[0], thisXd[:,0:-1:20], errr[:,0:-1:20], 'g')
            if _options['doPlot']:
                aaa[0].axis('equal')
                aaa[1].axis('equal')
            
            # Now get a matching
            
            if _options['doPlot'] and not (saveName is None):
                pu.plt.savefig(saveName, transparent=True, format='pdf')
                return
            
            
            print('Obtained matching curves for {0}-components and beta over betaMax {1}'.format(bestDyn.nK, bestDiffeoCtrl.radialMinConv._beta/betaMax))
            #pu.plt.show()
            
            
            
            
            
            
            