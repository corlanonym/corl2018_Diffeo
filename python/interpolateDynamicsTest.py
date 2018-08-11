from copy import deepcopy

from combinedDynamics import pointListToCombinedDirections,combinedLocallyWeightedDirections,convergingDirections,locallyWeightedDirections,minimallyConvergingDirection,getMagnitudeModel,combinedDiffeoCtrl
from coreUtils import *
import os

import plotUtils as pu

from interpolatePy import regularSpline
from modifiedEM import gaussianKernel
from distribution import cauchyKernel
from trajectoryUtils import stochasticMatching

import inspect
import time

from cppInterface import callCPP, batchedCallCPP


#def myRange(a,b,step):
#    if a==b:
#        return np.array([a,b]).astype(np.int_)
#    else:
#        x = np.arange(a,b,step)
#        if x[-1] != b:
#            x = np.hstack((x,b))
#        return x

def myRange(a,b,step):
    if a==b:
        return np.array([a,a+1]).astype(np.int_)
    else:
        n = max(int(round((b-a)/step))+1.,2)
        x = np.linspace(a,b,n,endpoint=True, dtype=np.int_)
        return x

def testInterpolate(name,inputFolder,outputFolder,resultFolder,cppExe=None,nDemos=-1,optionsSeries={},optionsMatching={},otheroptions={},optionsInterpolation={},parsDyn={},parsWeights={}):
    
    _optionsSeries = {'errStep':np.Inf,'p':5.,'alpha':-0.2,'doRound':True,'doRoundStart':True,'matchDirections':True, '_steps':np.array([0.,0.65, 1.])}
    _optionsMatching = {'baseRegVal':.5e-2,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':6,'maxNKernelErrStruct':12,'nInterTrans':8,'minTrans':-.5e-2,'minBase':-.5e-1,
                       'marginSingle':0.5,'marginOverlap':0.01,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-5,'epsConstraints':3.e-5}
    _optionsInterpolation = {}  # np.linspace(0.,1.,5)
    # otherOptions = {'Nstep':1, 'doPlot':True, 'nPtsDirDyn':15}
    _otherOptions = {'Nstep':1,'doPlot':True,'nPtsDirDyn':15,'boundingBoxSize':None,'demSpaceOffset':None}

    # parsDyn = {'alpha':-.2,'beta':-.05,'baseConv':-1.e-6, 'addFinalConv':True, '_finalConvBeta':-.5}
    _parsDyn = {'alpha':3.,'alpha0':3.,'beta':-.1,'beta0':-.1,'baseConv':-1.e-6,'addFinalConv':True,'_finalConvBeta':-1.,'_finalConvWeight':20, '_nptPerSeg':50}
    # parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':-1.,'doCond':False, 'finalCov':.5}
    _parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':2.,'doCond':False,'finalCov':2.,
                    'pi': 1.,'gamma': 1.}  #Cauchy stuff}
    
    _optionsSeries.update(optionsSeries)
    _optionsMatching.update(optionsMatching)
    _optionsInterpolation.update(optionsInterpolation)
    _otherOptions.update(otheroptions)
    
    _parsDyn.update(parsDyn)
    _parsWeights.update(parsWeights)
    
    try:
        cppExe = cppExe if (os.path.isfile(cppExe)) else None
    except:
        cppExe = None
        
    
    allFigList = []
    
    namedOutFolder = os.path.join(outputFolder,name)
    
    mkdir_p(outputFolder)
    mkdir_p(namedOutFolder)
    mkdir_p(resultFolder)
    
    # Load data
    allT = []
    allX = []
    allV = []
    sampleList = []
    Nstep = _otherOptions['Nstep']
    dim = None
    if nDemos>0:
        for k in range(nDemos):
            allT.append(np.loadtxt(os.path.join(inputFolder,name,"t_{0}.txt".format(k+1)))[0:-1:Nstep])
            allX.append(np.loadtxt(os.path.join(inputFolder,name,"pos_{0}.txt".format(k+1)))[:,0:-1:Nstep])
            allV.append(np.loadtxt(os.path.join(inputFolder,name,"vel_{0}.txt".format(k+1)))[:,0:-1:Nstep])
            sampleList.append(allT[-1].size)
            
            dim = allX[-1].shape[0] if (dim is None) else dim
            assert (dim == allX[-1].shape[0]),'Dim incompatible'
    else:
        k=0
        while True:
            try:
                allT.append(np.loadtxt(os.path.join(inputFolder,name,"t_{0}.txt".format(k+1)))[0:-1:Nstep])
                allX.append(np.loadtxt(os.path.join(inputFolder,name,"pos_{0}.txt".format(k+1)))[:,0:-1:Nstep])
                allV.append(np.loadtxt(os.path.join(inputFolder,name,"vel_{0}.txt".format(k+1)))[:,0:-1:Nstep])
                sampleList.append(allT[-1].size)
    
                dim = allX[-1].shape[0] if (dim is None) else dim
                assert (dim == allX[-1].shape[0]),'Dim incompatible'
                k+=1
            except OSError:
                nDemos = len(allX)
                print("Found {0} valid demos".format(nDemos))
                break
                
    
    
    del Nstep
    
    
    # Compute mean init
    xMean = np.zeros((dim,1))
    for ax in allX:
        xMean += ax[:,[0]]
    xMean /= len(allX)
    Array2TXT(os.path.join(resultFolder,'meanInitPos.txt'),xMean)
    
    # Create the matching object
    thisMatching = stochasticMatching(allX,allV,allT,optionsSeries=_optionsSeries,optionsMatching=_optionsMatching)
    
    # Get the approximatively feasible series
    xSeriesInversed = thisMatching.getSeries(returnVel=False)
    
    allXStartIrreg = deepcopy(xSeriesInversed[-1])
    allXStopIrreg = deepcopy(xSeriesInversed[0])
    
    #Check where it is infeasible
    feasInd = [ np.hstack((1,cNormSquare(axStop-axStart,kd=False)<1e-3)) for axStart, axStop in zip(allXStartIrreg, allXStopIrreg) ]
    fromFeas2NonIrreg = [list(np.flatnonzero(np.logical_and(afeas[:-1],np.logical_not(afeas[1:])))) for afeas in feasInd]
    fromNon2FeasIrreg = [list(np.flatnonzero(np.logical_and(np.logical_not(afeas[:-1]),afeas[1:]))) for afeas in feasInd]

    # Additional out for plots
    allOrigNregList = deepcopy(allX)
    # allUnchangedNregList = [ax[:,afeas[1:]].copy() for ax, afeas in zip(allX, feasInd)]
    # allPatchNregList = [ax[:,np.logical_not(afeas[1:])].copy() for ax, afeas in zip(allX, feasInd)]
    allUnchangedNregList = []
    allPatchNregList = []
    allUnfeasNregList = []
    
    # Deduce the direction dynamics
    allYfeasible = xSeriesInversed[-1]
    # regular, same step length interpolation
    allXStart = regularSpline(allYfeasible)[0]
    #regular final
    allXStop = regularSpline(allX)[0]
    
    fromFeas2NonStart = []
    fromFeas2NonStop = []
    fromNon2FeasStart = []
    fromNon2FeasStop = []
    for axStartIrreg,axStart,axStop,aFeas2Non,aNon2Feas in zip(allXStartIrreg, allXStart, allXStop, fromFeas2NonIrreg, fromNon2FeasIrreg):
        
        fromFeas2NonStart.append( [ np.argmin( cNormSquare(axStart-axStartIrreg[:,[aInd]], kd=False) ) for aInd in aFeas2Non ] )
        fromFeas2NonStop.append([np.argmin(cNormSquare(axStop-axStartIrreg[:,[aInd]],kd=False)) for aInd in aFeas2Non])

        fromNon2FeasStart.append([np.argmin(cNormSquare(axStart-axStartIrreg[:,[aInd]],kd=False)) for aInd in aNon2Feas])
        fromNon2FeasStop.append([np.argmin(cNormSquare(axStop-axStartIrreg[:,[aInd]],kd=False)) for aInd in aNon2Feas])

    if _otherOptions['doPlot']:
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)

        for k in range(len(allXStopIrreg)):
            aa.plot(allXStartIrreg[k][0,:], allXStartIrreg[k][1,:], '.g')
            aa.plot(allXStopIrreg[k][0,:],allXStopIrreg[k][1,:],'.k')

            aa.plot(allXStart[k][0,:],allXStart[k][1,:],'.r')
            aa.plot(allXStop[k][0,:],allXStop[k][1,:],'.b')

            aa.plot(allXStartIrreg[k][0,fromFeas2NonIrreg[k]],allXStartIrreg[k][1,fromFeas2NonIrreg[k]],'og')
            aa.plot(allXStopIrreg[k][0,fromFeas2NonIrreg[k]],allXStopIrreg[k][1,fromFeas2NonIrreg[k]],'ok')
            aa.plot(allXStartIrreg[k][0,fromNon2FeasIrreg[k]],allXStartIrreg[k][1,fromNon2FeasIrreg[k]],'sg')
            aa.plot(allXStopIrreg[k][0,fromNon2FeasIrreg[k]],allXStopIrreg[k][1,fromNon2FeasIrreg[k]],'sk')

            aa.plot(allXStart[k][0,fromFeas2NonStart[k]],allXStart[k][1,fromFeas2NonStart[k]],'*r')
            aa.plot(allXStop[k][0,fromFeas2NonStop[k]],allXStop[k][1,fromFeas2NonStop[k]],'db')
            aa.plot(allXStart[k][0,fromNon2FeasStart[k]],allXStart[k][1,fromNon2FeasStart[k]],'*r')
            aa.plot(allXStop[k][0,fromNon2FeasStop[k]],allXStop[k][1,fromNon2FeasStop[k]],'db')
    
    
    #Split up
    #First prepend 0 to Non2Feas
    fromFeas2NonStart = [ [0]+aList+[allX[k].shape[1]-1] for k,aList in enumerate(fromFeas2NonStart) ]
    fromFeas2NonStop = [[0]+aList+[allX[k].shape[1]-1] for k,aList in enumerate(fromFeas2NonStop)]
    allIndUnchanged = [] #Taken from start
    allIndChangedStart = []
    allIndChangedStop = []
    
    allXunchanged = []
    allXChangedStart = []
    allXChangedStop = []
    
    
    ##nPtperSeg = 50
    nPtperSeg = _parsDyn['_nptPerSeg']
    
    for k in range(len(allXStopIrreg)):
        #It is assumed that the movement is feasible at the end
        
        thisUnchanged = []
        thisChangedStart = []
        thisChangedStop = []
        thisXUnchanged = []
        thisXChangedStart = []
        thisXChangedStop = []

        allUnchangedNregList.append([]) # Plot
        allPatchNregList.append([]) # Plot
        allUnfeasNregList = []
        
        if ((fromFeas2NonStart[k][1]-fromFeas2NonStart[k][0])>nPtperSeg/4.):
            thisUnchanged.append( myRange(fromFeas2NonStart[k][0], fromFeas2NonStart[k][1], nPtperSeg) )
            thisXUnchanged.append( allXStart[k][:,thisUnchanged[-1]].copy() )
            allUnchangedNregList[-1].append(allXStart[k][:,fromFeas2NonStart[k][0]:fromFeas2NonStart[k][1]]) # Plot
        else:
            print("skipped initial unchanged {0} to {1} ".format(fromFeas2NonStart[k][1],fromFeas2NonStart[k][0]))
        
        if len(fromFeas2NonStart)==2:
            break
        
        for i in range(1,len(fromFeas2NonStart[k])-1):
            if ((fromNon2FeasStop[k][i-1]-fromFeas2NonStop[k][i])>nPtperSeg/4.):
                thisChangedStop.append(myRange(fromFeas2NonStop[k][i],fromNon2FeasStop[k][i-1],nPtperSeg))
                thisXChangedStop.append(allXStop[k][:,thisChangedStop[-1]])
            
                thisChangedStart.append( np.linspace(fromFeas2NonStart[k][i], fromNon2FeasStart[k][i-1], len(thisChangedStop[-1]), dtype=np.int_) )
                thisXChangedStart.append(allXStart[k][:,thisChangedStart[-1]])
                
                allPatchNregList[-1].append( allXStart[k][:,fromFeas2NonStart[k][i]:fromNon2FeasStart[k][i-1]].copy() ) # Plot
            else:
                print("skipped changed {0} to {1} ".format(fromNon2FeasStop[k][i-1],fromFeas2NonStop[k][i]))
            
            if ((fromFeas2NonStart[k][i+1]-fromNon2FeasStart[k][i-1])>nPtperSeg/4.):
                thisUnchanged.append(myRange(fromNon2FeasStart[k][i-1],fromFeas2NonStart[k][i+1],nPtperSeg))
                thisXUnchanged.append(allXStart[k][:,thisUnchanged[-1]].copy())
                
                allUnchangedNregList[-1].append(allXStart[k][:,fromNon2FeasStart[k][i-1]:fromFeas2NonStart[k][i+1]].copy()) # Plot
            else:
                print("skipped unchanged {0} to {1} ".format(fromFeas2NonStart[k][i+1],fromNon2FeasStart[k][i-1]))
        
        allIndUnchanged.append(thisUnchanged)
        allIndChangedStart.append(thisChangedStart)
        allIndChangedStop.append(thisChangedStop)
        allXunchanged.append(thisXUnchanged)
        allXChangedStart.append(thisXChangedStart)
        allXChangedStop.append(thisXChangedStop)

    if _otherOptions['doPlot']:
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)
        for aUn, aCStart, aCStop in zip(allXunchanged, allXChangedStart, allXChangedStop):

            for aaUn in aUn:
                aa.plot(aaUn[0,:], aaUn[1,:], '.-r')
                aa.plot(aaUn[0,[0,-1]],aaUn[1,[0,-1]],'*r')

            for aaCStart in aCStart:
                aa.plot(aaCStart[0,:], aaCStart[1,:], '.-b')
                aa.plot(aaCStart[0,[0,-1]],aaCStart[1,[0,-1]],'*b')

            for aaCStop in aCStop:
                aa.plot(aaCStop[0,:],aaCStop[1,:],'.-k')
                aa.plot(aaCStop[0,[0,-1]],aaCStop[1,[0,-1]],'*k')

    allDirDynStart = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
    allDirDynStartAsList = []
    allDirDynStop = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
    allDirDynStopAsList = []
    allDirDynConstant = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
    allDirDynConstantAsList = []

    allDirDynComplete = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
    
    for aX,aI in zip(allXunchanged, allIndUnchanged):
        for aSeg, aSegI in zip(aX,aI):
            thisDirDyn,thisDirDynList = pointListToCombinedDirections(aSeg.copy(),parsWeights=_parsWeights,parsDyn=_parsDyn,fullOut=True)
            
            aW = (np.diff(aSegI)/nPtperSeg)
            for kk, thisRelWeight in enumerate(aW):
                thisDirDynList[kk]._weight = thisRelWeight
                thisDirDyn._dirList[kk]._weight=thisRelWeight

            allDirDynConstant._dirList += thisDirDyn._dirList
            allDirDynConstantAsList.append(thisDirDynList)
    
    for aX,aI in zip(allXChangedStop, allIndChangedStop):
        for aSeg, aSegI in zip(aX,aI):
            thisDirDyn,thisDirDynList = pointListToCombinedDirections(aSeg.copy(),parsWeights=_parsWeights,parsDyn=_parsDyn,fullOut=True)

            aW = (np.diff(aSegI)/nPtperSeg)
            for kk,thisRelWeight in enumerate(aW):
                thisDirDynList[kk]._weight = thisRelWeight
                thisDirDyn._dirList[kk]._weight = thisRelWeight

            allDirDynStop._dirList += thisDirDyn._dirList
            allDirDynStopAsList.append(thisDirDynList)
    
    for aX,aI in zip(allXChangedStart, allIndChangedStart):
        for aSeg,aSegI in zip(aX,aI):
            thisDirDyn,thisDirDynList = pointListToCombinedDirections(aSeg.copy(),parsWeights=_parsWeights,parsDyn=_parsDyn,fullOut=True)

            aW = (np.diff(aSegI)/nPtperSeg)
            for kk,thisRelWeight in enumerate(aW):
                thisDirDynList[kk]._weight = thisRelWeight
                thisDirDyn._dirList[kk]._weight = thisRelWeight

            allDirDynStart._dirList += thisDirDyn._dirList
            allDirDynStartAsList.append(thisDirDynList)

    if _parsDyn['addFinalConv']:
        finalConv = convergingDirections(x0=np.zeros((dim,1)),vp=np.zeros((dim,1)),alpha=0.,beta=_parsDyn['_finalConvBeta'])
        if locallyConvDirecKernel_==0:
            finalWeight = gaussianKernel(1,_parsWeights['finalCov']*np.identity(dim),np.zeros((dim,1)),doCond=_parsWeights['doCond'])
        elif locallyConvDirecKernel_==1:
            finalWeight = cauchyKernel(None,_parsWeights['finalCov']*np.identity(dim),np.zeros((dim,1)),doCond=_parsWeights['doCond'], gamma=_parsWeights['gamma'], pi=_parsWeights['pi'])
        else:
            assert 0
        finalDir = locallyWeightedDirections(finalWeight,finalConv,weight=_parsDyn['_finalConvWeight'])
    
    #Actually constrain only the start dynamics others are free
    #allDirDynStart._ensureConv = lambda x,v:v
    fConv = lambda x,v:minimallyConvergingDirection(x,v,minAng=minConvAngle_,minAngIsAng=False)
    allDirDynStart._ensureConv = fConv
    allDirDynStop._ensureConv = lambda x,v:v
    allDirDynConstant._ensureConv = lambda x,v:v
    allDirDynComplete._ensureConv = lambda x,v:v
    
    if _otherOptions['doPlot']:
        NN = 100
        XXStop = np.hstack(allXStop)
        XXStart = np.hstack(allXStart)

        lims = [np.min(XXStop[0,:]),np.max(XXStop[0,:]),np.min(XXStop[1,:]),np.max(XXStop[1,:])]
        xGrid,yGrid = np.meshgrid(np.linspace(lims[0],lims[1],NN),np.linspace(lims[2],lims[3],NN))
        XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))

    allXInterpolated = []

    for alpha in _optionsSeries['_steps']:
        # Limit points can not be interpolated
        if alpha < 1e-6:
            # Start dynamics
            allDirDynComplete._dirList = allDirDynStart._dirList + allDirDynConstant._dirList
            if _parsDyn['addFinalConv']:
                allDirDynComplete.addDyn(finalDir)
            #Set convergence
            with open(os.path.join(namedOutFolder,"minConv.txt"),'w+') as file:
                file.write(double2Str(.0125))
        elif alpha>(1-1e-6):
            # Stop dynamics
            allDirDynComplete._dirList = allDirDynStop._dirList + allDirDynConstant._dirList
            if _parsDyn['addFinalConv']:
                allDirDynComplete.addDyn(finalDir)
            with open(os.path.join(namedOutFolder,"minConv.txt"),'w+') as file:
                file.write(double2Str(-10000.0))
        else:
            # mixed
            allDirDynComplete._dirList = (((1. - alpha) * allDirDynStart) + (alpha * allDirDynStop))._dirList + allDirDynConstant._dirList
            if _parsDyn['addFinalConv']:
                allDirDynComplete.addDyn(finalDir)
            with open(os.path.join(namedOutFolder,"minConv.txt"),'w+') as file:
                file.write(double2Str(-10000.0))

        allDirDynComplete.toText(os.path.join(namedOutFolder, "dirModel.txt"))
        if cppExe is not None:
            thisAllRes = batchedCallCPP(what='dirTraj', dims=(2, 0), definitionFolder=namedOutFolder, inputFolder=resultFolder, resultFolder=resultFolder, cppExe=cppExe, xIn=[ax[:,[0]] for ax in allX])
            thisXinterpolated = [aRes[3] for aRes in thisAllRes]
        else:
            thisAllRes = [ allDirDynComplete.getDirTraj(ax[:,[0]], stopCond=lambda x: cNorm(x,False) < 1.) for ax in allX ]
            thisXinterpolated = [ares[0][1] for ares in thisAllRes]
        
        thisXinterpolated, _, _ = regularSpline(thisXinterpolated, sampleList=[ax.shape[1] for ax in allXStart])
        
        allXInterpolated.append(thisXinterpolated)

        if _otherOptions['doPlot']:
            ff, aa = pu.plt.subplots(1, 1)
            aa.set_title("{0}".format(alpha))
            allFigList.append(ff)
            aa.plot(XXStop[0, :], XXStop[1, :], '.k')
            aa.plot(XXStart[0, :], XXStart[1, :], '.b')
            VV = allDirDynComplete.getDir(XGrid.copy())
            aa.streamplot(xGrid, yGrid, VV[0, :].reshape((NN, NN)), VV[1, :].reshape((NN, NN)), color='c')
            for k,ax in enumerate(thisXinterpolated):
                aa.plot(ax[0,:], ax[1,:], 'g', linewidth=2)
            
                pu.myQuiver(aa, allXStart[k][:,0::40], ax[:,0::40]-allXStart[k][:,0::40], 'g')
                pu.myQuiver(aa,ax[:,0::40],allXStop[k][:,0::40]-ax[:,0::40],'b')

    return allFigList, allXInterpolated, {'const':allDirDynConstant,'start':allDirDynStart, 'stop':allDirDynStop, 'final':finalDir, 'fConv':fConv, 'feasTrajList':allXStart, 'plot':{'allOrigNregList':allOrigNregList, 'allUnchangedNregList':allUnchangedNregList, 'allPatchNregList':allPatchNregList}}
    
def namedTest(name):
    import os
    import pickle
    
    osabspath = os.path.abspath
    
    
    
    # Test
    # searchNStoreVectorfield(name, inputFolder, outputFolder, resultFolder, cppExe=None, nDemos=7, optionsSeries={}, optionsMatching={}, otheroptions={}, parsDyn={}, parsWeights={}, optionsMagMod={}):
    inputFolder = osabspath('./data')
    outputFolder = osabspath('./cpp/tmpFiles')
    resultFolder = os.path.join(osabspath('./results/new2/'), name)
    
    optionsSeries = {'errStep':np.Inf,'p':5.,'alpha':-0.2,'doRound':True,'doRoundStart':True,'matchDirections':True}
    optionsMatching = {'baseRegVal':.5e-2,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':6,'maxNKernelErrStruct':12,'nInterTrans':8,'minTrans':-.5e-2,'minBase':-.5e-1,
                       'marginSingle':0.5,'marginOverlap':0.01,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-5,'epsConstraints':3.e-5}
    # otherOptions = {'Nstep':1, 'doPlot':True, 'nPtsDirDyn':15}
    otherOptions = {'Nstep':1,'doPlot':2,'nPtsDirDyn':20}
    
    # parsDyn = {'alpha':-.2,'beta':-.05,'baseConv':-1.e-6, 'addFinalConv':True, '_finalConvBeta':-.5}
    parsDyn = {'alpha':3.,'alpha0':3.,'beta':-.1,'beta0':-.1,'baseConv':-1.e-6,'addFinalConv':True,'_finalConvBeta':-1.,'_finalConvWeight':20, '_nptPerSeg':50}
    # parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':-1.,'doCond':False, 'finalCov':.5}
    parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':2.,'doCond':False,'finalCov':2.}
    
    #cppExe = os.path.abspath("/home/elfuius/temp/myBuild/diffeoInterface")#os.path.abspath("./cpp/mybuild/diffeoInterface")#"/home/elfuius/ros_ws/devel/lib/baxter_diffeo/diffeoInterface"  # None#'./cpp/cmake-build-debug/combinedControl'
    cppExe = os.path.abspath("/home/elfuius/ros_ws/devel/lib/baxter_diffeo/diffeoInterface")
    # Baxter specific to adjust size
    # otherOptions['boundingBoxSize'] = np.array([.25,.25])
    # otherOptions['demSpaceOffset'] = np.array([.8,-.1])
    
    allFig, allX, dirDynDict = testInterpolate(name,inputFolder,outputFolder,resultFolder,cppExe=cppExe,optionsSeries=optionsSeries,optionsMatching=optionsMatching,otheroptions=otherOptions,parsDyn=parsDyn,parsWeights=parsWeights)
    
    for kk,aFig in enumerate(allFig):
        aFig.savefig(os.path.join(resultFolder, "{0}.png".format(kk)), format='png')
    return allFig

    
if __name__ == '__main__':
    
    allFig = []

    all = ['GShape','Angle','BendedLine','DoubleBendedLine','Khamesh','Leaf_1','Leaf_2','Sharpc','Snake']
    #all += ['Multi_Models_1','Multi_Models_2','Multi_Models_3','Multi_Models_4']
    
    for name in all:
        allFig += namedTest(name)
    
    pu.plt.show()