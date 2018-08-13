import subprocess
from copy import deepcopy

from combinedDynamics import pointListToCombinedDirections,combinedLocallyWeightedDirections,convergingDirections,locallyWeightedDirections,minimallyConvergingDirection,getMagnitudeModel,combinedDiffeoCtrl
from coreUtils import *
import os

import plotUtils as pu

from interpolatePy import regularSpline
from modifiedEM import gaussianKernel
from trajectoryUtils import stochasticMatching

import inspect
import time

def fit2bBoxSize(allX,allV,bBoxDesired):
    xxxTemp = np.hstack(allX)
    bBoxSize = np.max(xxxTemp,axis=1)-np.min(xxxTemp,axis=1)
    bBoxSize = bBoxSize.squeeze()
    sizeFac = bBoxSize/bBoxDesired
    sizeFacMax = max(list(sizeFac))
    addScaleFac = 1./sizeFacMax
    
    # Directly scale velocities and size leave time unchanged
    for ax,av in zip(allX,allV):
        ax *= addScaleFac
        av *= addScaleFac


def searchNStoreVectorfield(name,inputFolder,outputFolder,resultFolder,cppExe=None,nDemos=-1,optionsSeries={},optionsMatching={},otheroptions={},parsDyn={},parsWeights={},optionsMagMod={}):
    from cppInterface import batchedCallCPP,callCPP
    _optionsSeries = {'errStep':np.Inf,'p':5.,'alpha':-0.25,'doRound':True,'doRoundStart':True,'matchDirections':True, '_steps':[0.,0.6], '_direct':2}
    _optionsMatching = {'baseRegVal':1.e-3,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':12,'maxNKernelErrStruct':12,'nInterTrans':15,'minTrans':-1.e-2,'minBase':-5.e-2,
                        'marginSingle':0.5,'marginOverlap':0.01,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-5,'epsConstraints':3.e-5}
    _otherOptions = {'Nstep':1,'doPlot':False,'nPtsDirDyn':15,
                     'boundingBoxSize':None,'demSpaceOffset':None, 'computeSEA_':True}
    
    _parsDyn = {'alpha':-.1,'alpha0':-.1,'beta':-.1,'beta0':-.5,'baseConv':-1.e-6,'addFinalConv':True,'_finalConvBeta':-.5,'_finalConvWeight':10}
    _parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':-.4,'doCond':False,'finalCov':2.}
    
    _optionsMagMod = {'add2Sigma':2.5,'iterMax':100,'relTol':2e-2,'absTol':1.e-2,'interOpt':True,'reOpt':False,'convFac':1.,'nPartial':10}
    
    _optionsSeries.update(optionsSeries)
    _optionsMatching.update(optionsMatching)
    _otherOptions.update(otheroptions)
    
    _parsDyn.update(parsDyn)
    _parsWeights.update(parsWeights)
    
    _optionsMagMod.update(optionsMagMod)
    
    cppExe = cppExe if (os.path.isfile(cppExe)) else None
    
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
    
    # Adapt given demonstrations to bounding box
    # Only necessairy if trained device is not replay device
    if _otherOptions['boundingBoxSize'] is not None:
        fit2bBoxSize(allX,allV,_otherOptions['boundingBoxSize'])
    
    # Compute mean init
    xMean = np.zeros((dim,1))
    for ax in allX:
        xMean += ax[:,[0]]
    xMean /= len(allX)
    Array2TXT(os.path.join(namedOutFolder,'meanInitPos.txt'),xMean)
    
    # Create the matching object
    thisMatching = stochasticMatching(allX,allV,allT,optionsSeries=_optionsSeries,optionsMatching=_optionsMatching)
    
    # Get the appreciatively feasible series
    # xSeriesInversed = thisMatching.getSeries(returnVel=False)

    # Get the series based on dynamic interpolation
    from interpolateDynamicsTest import testInterpolate
    # testInterpolate(name,inputFolder,outputFolder,resultFolder,cppExe=None,nDemos=7,optionsSeries={},optionsMatching={},otheroptions={},optionsInterpolation={},parsDyn={},parsWeights={}):
    #    return allFigList, allXInterpolated, {'const':allDirDynConstant,'start':allDirDynStart, 'stop':allDirDynStop, 'final':finalDir, 'fConv':fConv}

    # addFigList, xSeriesInversed, dynDict = testInterpolate(name, inputFolder, outputFolder, resultFolder, cppExe=cppExe, optionsSeries=_optionsSeries, optionsMatching=_optionsMatching, otheroptions=_otherOptions)
    addFigList,xSeriesInversed,dynDict = testInterpolate(name,inputFolder,outputFolder,resultFolder,cppExe=cppExe,optionsSeries=_optionsSeries,optionsMatching=_optionsMatching,otheroptions=_otherOptions,parsDyn=parsDyn,parsWeights=parsWeights)
    
    allFigList += addFigList
    
    # Deduce the direction dynamics
    allYfeasible = xSeriesInversed[-1]
    allYfeasibleUsedToGenerateDyn = dynDict['feasTrajList']
    allY = regularSpline(allYfeasible)[0]
    # Save for later
    allYForDemCtrl = deepcopy(allY)
    
    # regular non inversed series
    xSeriesReg = [regularSpline(ax)[0] for ax in xSeriesInversed]
    # regular with final real data
    if _optionsSeries['_direct']==1:
        xSeriesReg.append( deepcopy(allX) )
    elif _optionsSeries['_direct']==2:
        xSeriesReg = [xSeriesReg[0],xSeriesReg[-1]]
    elif _optionsSeries['_direct'] == 3:
        xSeriesReg = [xSeriesReg[0],xSeriesReg[-1]]
        xSeriesReg.append(deepcopy(allX))
    
    try:
        # Use the dynamics used for trajectory generation
        allDirDyn = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
        allDirDyn._dirList = dynDict['start']._dirList+dynDict['const']._dirList
        allDirDyn.addDyn(dynDict['final'])
    
    except:
        # Use new dynamics
        # pointListToCombinedDirections(points: np.ndarray,parsWeights: dict = {},parsDyn:dict = {} )->combinedLocallyWeightedDirections:
        nPtsDirDyn = _otherOptions['nPtsDirDyn']
        allDirDyn = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
        if _otherOptions['doPlot']:
            ptLists2Plt = []
        for ay in allY:
            thisI = np.linspace(0,ay.shape[1]-1,nPtsDirDyn,dtype=np.int_)
            thisDirDyn = pointListToCombinedDirections(ay[:,thisI].copy(),parsWeights=_parsWeights,parsDyn=_parsDyn)
            
            allDirDyn._dirList += thisDirDyn._dirList
            
            if _otherOptions['doPlot']:
                ptLists2Plt.append(ay[:,thisI].copy())
        del nPtsDirDyn
        
        # Add a final converging if demanded
        if _parsDyn['addFinalConv']:
            finalConv = convergingDirections(x0=np.zeros((dim,1)),vp=np.zeros((dim,1)),alpha=0.,beta=_parsDyn['_finalConvBeta'])
            finalWeight = gaussianKernel(1,_parsWeights['finalCov']*np.identity(dim),np.zeros((dim,1)),doCond=_parsWeights['doCond'])
            finalDir = locallyWeightedDirections(finalWeight,finalConv,weight=_parsDyn['_finalConvWeight'])
            
            allDirDyn.addDyn(finalDir)

    # Add minimal convergence
    fConv = lambda x,v:minimallyConvergingDirection(x,v,minAng=minConvAngle_,minAngIsAng=False)
    allDirDyn._ensureConv = fConv

    # Get the direction trajectories for each demonstration
    xInit = np.hstack([ax[:,[0]] for ax in allX])
    if cppExe is not None:
        thisAllCppRes = batchedCallCPP(what='dirTraj',dims=(dim,0),definitionFolder=namedOutFolder,inputFolder=resultFolder,resultFolder=resultFolder,cppExe=cppExe,xIn=[ax[:,[0]] for ax in allX])
        allSol = [aRes[3] for aRes in thisAllCppRes]
        for ax in allSol:
            ax[:,-1] = 0.
        # Again use regularspline
        newX = regularSpline(allSol,sampleList=sampleList)[0]
    else:
        allSol = allDirDyn.getDirTraj(xInit)

        # set last to zero (should be almost zero)
        for _,ax in allSol:
            ax[:,-1] = 0.
        # Again use regularspline
        newX = regularSpline([ax for at,ax in allSol],sampleList=sampleList)[0]
    
    xSeriesInversed = deepcopy([allX,newX])
    
    if (_otherOptions['doPlot'] > 2 and (dim == 2)):
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        for ay in allY:
            aa.plot(ay[0,:],ay[1,:],'k')
        
        for ay in ptLists2Plt:
            aa.plot(ay[0,:],ay[1,:],'.-r')
        
        xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,200),np.linspace(aa.get_ylim()[0]-5,aa.get_ylim()[1]+5,200))
        XX = np.vstack((xx.flatten(),yy.flatten()))
        VV = allDirDyn.getDir(XX)
        
        aa.streamplot(xx,yy,VV[0,:].reshape((200,200)),VV[1,:].reshape((200,200)))

        # Also plot the obtained traj
        for ax in newX:
            aa.plot(ax[0,:],ax[1,:],'b')
    
    # Search for the diffeo
    newOpts = {'additionalCenters':np.hstack([ax[:,[0]] for ax in allX])}
    
    import time
    TT = time.time()
    try:
        xxxxx = deepcopy(xSeriesReg)
        # Update feasible with actually obtained trajectories
        thisDiffeo = thisMatching.doMatching(xSeriesReg,options=newOpts)
    except NameError:
        xxxxx = deepcopy(list(reversed(deepcopy(xSeriesInversed))))
        thisDiffeo = thisMatching.doMatching(list(reversed(deepcopy(xSeriesInversed))),options=newOpts)
    TT = time.time()-TT
    print("Construction time was {0}".format(TT))
    allY = deepcopy(xxxxx[0])
    
    
    if (_otherOptions['doPlot'] >= 1 and (dim == 2)):
        ff,aa = pu.plt.subplots(2,2)
        allFigList.append(ff)
        if (False and (cppExe is not None)):
            
            # Save to text - temporary
            # Can not be done yet
            from cppInterface import batchedCallCPP
            forwardSol = batchedCallCPP('forward',dims=(dim,0),definitionFolder=namedOutFolder,inputFolder=resultFolder,resultFolder=resultFolder,cppExe=cppExe,xIn=xxxxx[0])
            inverseSol = batchedCallCPP('inverse',dims=(dim,0),definitionFolder=namedOutFolder,inputFolder=resultFolder,resultFolder=resultFolder,cppExe=cppExe,xIn=xxxxx[-1])

            # Dem
            actualDemSpaceTraj = batchedCallCPP('demSpaceTraj',dims=(dim,0),definitionFolder=namedOutFolder,inputFolder=resultFolder,resultFolder=resultFolder,cppExe=cppExe,xIn=[ax[:,[0]] for ax in allX])
            
            forwardSol = [aSol[1] for aSol in forwardSol]
            inverseSol = [aSol[1] for aSol in inverseSol]
            actualDemSpaceTraj = [adst[3] for adst in actualDemSpaceTraj]
            for ax,ay,axf,ayi,axx,axxSol in zip(xxxxx[0],xxxxx[-1],forwardSol,inverseSol,allX,actualDemSpaceTraj):
                aa[0,0].plot(ax[0,:],ax[1,:],'--b')
                aa[0,1].plot(ax[0,:],ax[1,:],'.-k')
                aa[0,0].plot(ay[0,:],ay[1,:],'.-k')
                aa[0,1].plot(ay[0,:],ay[1,:],'--b')
                aa[0,0].plot(axf[0,:],axf[1,:],'g')
                aa[0,1].plot(ayi[0,:],ayi[1,:],'g')
                aa[1,0].plot(ax[0,:],ax[1,:],'-k')
                aa[1,0].plot(ay[0,:],ay[1,:],'-b')
                
                aa[1,1].plot(axx[0,:],axx[1,:],'-k')
                aa[1,1].plot(axxSol[0,:],axxSol[1,:],'-g')
        else:
            for axr,ax,ay in zip(allX,xxxxx[0],xxxxx[-1]):
                aa[0,0].plot(ax[0,:],ax[1,:],'--b')
                aa[0,0].plot(axr[0,:],axr[1,:],'-c')
                aa[0,1].plot(ax[0,:],ax[1,:],'.-k')
                aa[0,0].plot(ay[0,:],ay[1,:],'.-k')
                aa[0,1].plot(ay[0,:],ay[1,:],'--b')
                aa[0,1].plot(axr[0,:],axr[1,:],'-c')
                ccc = thisDiffeo.forwardTransform(ax.copy())
                aa[0,0].plot(ccc[0,:],ccc[1,:],'g',linewidth=2)
                aa[1,0].plot(ax[0,:],ax[1,:],'-k')
                aa[1,0].plot(ay[0,:],ay[1,:],'-b')
        for TT in thisDiffeo._transformationList:
            aa[0,0].plot(TT._centers[0,:],TT._centers[1,:],'o')
            pu.myQuiver(aa[0,0],TT.centers,TT.translations,'k')
        
        XXgrid = pu.getGrid([aa[1,0].get_xlim()[0]-5,aa[1,0].get_xlim()[1]+5,aa[1,0].get_ylim()[0]-5,aa[1,0].get_ylim()[1]+5],[20,200])
        XXgridp = thisDiffeo.forwardTransform(XXgrid.copy())
        pu.plotGrid(aa[1,0],XXgridp,[20,200],'c')

    # return None, allFigList
    
    if (_otherOptions['doPlot'] > 2 and (dim == 2)):
        NNN = 250
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        for ax in allX:
            aa.plot(ax[0,:],ax[1,:],'k')
        
        xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,NNN),np.linspace(aa[0,0].get_ylim()[0]-5,aa[0,0].get_ylim()[1]+5,NNN))
        XX = np.vstack((xx.flatten(),yy.flatten()))
        VV = allDirDyn.getDir(XX)
        tmpLims = [aa.get_xlim(),aa.get_ylim()]
        
        # PlotLyap
        vMax = 1.1*np.max(cNorm(np.hstack(allX),kd=False))
        circleList = pu.plotLyap(vMax,nCirc=8,ax=aa,c='k')
        
        aa.streamplot(xx,yy,VV[0,:].reshape((NNN,NNN)),VV[1,:].reshape((NNN,NNN)),color='c')
        
        # Also plot the obtained traj
        for ax in newX:
            aa.plot(ax[0,:],ax[1,:],'b')
        
        aa.set_xlim(tmpLims[0])
        aa.set_ylim(tmpLims[1])
        
        aa.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        aa.set(adjustable='box-forced',aspect='equal')
    
    if (_otherOptions['doPlot'] > 1 and (dim == 3)):
        ff = pu.plt.figure()
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        aa = ff.add_subplot(111,projectoin='3d')
        
        for ay in allY:
            aa.plot(ay[0,:],ay[1,:],ay[2,:],'k')
        # Also plot the obtained traj
        for ax in newX:
            aa.plot(ax[0,:],ax[1,:],ax[2,:],'b')
    
    if (_otherOptions['doPlot'] > 2 and (dim == 2)):
        XXgrid = pu.getGrid([aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,aa.get_ylim()[0]-5,aa.get_ylim()[1]+5],[20,200])
        
        XXgridp = thisDiffeo.forwardTransform(XXgrid.copy())
        
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        pu.plotGrid(aa,XXgridp,[20,200],'m')
        
        for ax in allX:
            aa.plot(ax[0,:],ax[1,:],'k')
        
        xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,50),np.linspace(aa.get_ylim()[0]-5,aa.get_ylim()[1]+5,50))
        XX = np.vstack((xx.flatten(),yy.flatten()))
        
        XXp = thisDiffeo.inverseTransform(XX.copy())
        VVp = allDirDyn.getDir(XXp)
        
        XXt,VVt = thisDiffeo.forwardTransformJac(x=XXp.copy(),v=VVp.copy())
        
        assert np.max(np.abs(XX-XXt)) < 1e-4,'Huge fucking error'
        
        aa.streamplot(xx,yy,VVt[0,:].reshape((50,50)),VVt[1,:].reshape((50,50)))
        aa.axis('equal')
    
    if (_otherOptions['doPlot'] > 2 and (dim == 2)):
        ff,aa = pu.plt.subplots(1,2)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        for k in range(len(allX)):
            print("current line is {0}".format(inspect.currentframe().f_lineno))
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            aaX = thisDiffeo.forwardTransform(allY[k])
            initX = allX[k][:,[0]]
            initY = thisDiffeo.inverseTransform(initX)
            trajY = allDirDyn.getDirTraj(initY)[0][1]
            trajX = thisDiffeo.forwardTransform(trajY)
            aa[0].plot(allX[k][0,:],allX[k][1,:],'r')
            aa[1].plot(allYfeasible[k][0,:],allYfeasible[k][1,:],'g')
            aa[1].plot(allY[k][0,:],allY[k][1,:],'r')
            aa[0].plot(aaX[0,:],aaX[1,:],'g')
            aa[0].plot(trajX[0,:],trajX[1,:],'b')
            aa[1].plot(trajY[0,:],trajY[1,:],'b')
        aa[0].axis('equal')
        aa[1].axis('equal')
    
    if (_otherOptions['doPlot'] > 1 and (dim == 3)):
        ff = pu.plt.figure()
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        
        aa = [ff.add_subplot(112,projection='3d'),ff.add_subplot(212,projection='3d')]
        
        for k in range(len(allX)):
            print("current line is {0}".format(inspect.currentframe().f_lineno))
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            aaX = thisDiffeo.forwardTransform(allY[k])
            initX = allX[k][:,[0]]
            initY = thisDiffeo.inverseTransform(initX)
            trajY = allDirDyn.getDirTraj(initY)[0][1]
            trajX = thisDiffeo.forwardTransform(trajY)
            aa[0].plot(allX[k][0,:],allX[k][1,:],allX[k][2,:],'r')
            aa[1].plot(allYfeasible[k][0,:],allYfeasible[k][1,:],allYfeasible[k][2,:],'g')
            aa[1].plot(allY[k][0,:],allY[k][1,:],allY[k][2,:],'r')
            aa[0].plot(aaX[0,:],aaX[1,:],aaX[2,:],'g')
            aa[0].plot(trajX[0,:],trajX[1,:],trajX[2,:],'b')
            aa[1].plot(trajY[0,:],trajY[1,:],trajY[2,:],'b')
        aa[0].axis('equal')
        aa[1].axis('equal')
    
    # Learn magnitude model in demonstration space
    allXstacked = np.hstack(allX)
    allVstacked = np.hstack(allV)
    allVn = cNorm(allVstacked,kd=True)
    thisMagModel = getMagnitudeModel(allXstacked,allVn,opts=_optionsMagMod)
    
    if _otherOptions['doPlot']:
        # Check quality
        allVnPred = thisMagModel.evalMap(allXstacked)
        ff,aa = pu.plt.subplots(1,1)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        aa.plot(allVn.squeeze(),'b')
        aa.plot(allVnPred.squeeze(),'r')
    
    # Test actual integration
    # def getMagnitudeModel(x:np.ndarray,v:np.ndarray,nKMax:int=12, opts={}):
    
    # assemble combined control
    thisControl = combinedDiffeoCtrl(allDirDyn,thisMagModel,thisDiffeo,False)
    
    if _otherOptions['doPlot'] and False:
        # fairly time consumming
        # Get all velocities
        allVtest = thisControl.getDemSpaceVelocity(allXstacked)
        ff,aa = pu.plt.subplots(dim,1)
        allFigList.append(ff)
        print("current line is {0}".format(inspect.currentframe().f_lineno))
        print(time.strftime("%Y-%m-%d %H:%M:%S"))
        for k in range(allXstacked.shape[0]):
            aa[k].plot(allVstacked[k,:],'b')
            aa[k].plot(allVtest[k,:],'r')

    # Explicitely set dimInt
    thisControl._dimInt = 0
    # and scaling #todo
    thisControl.overallScaling = np.ones((dim,1))
    if _otherOptions['demSpaceOffset'] is None:
        thisControl._demSpaceOffset = np.zeros((dim,1))
    else:
        thisControl._demSpaceOffset = _otherOptions['demSpaceOffset'].reshape((dim,1))
    
    # Save to text
    thisDiffeo.toText(os.path.join(namedOutFolder,"diffeo.txt"))
    thisMagModel.toText(os.path.join(namedOutFolder,"magModel.txt"))
    allDirDyn.toText(os.path.join(namedOutFolder,"dirModel.txt"))
    thisControl.toText(os.path.join(namedOutFolder,"combinedControl.txt"))
    with open(os.path.join(namedOutFolder,"minConv.txt"),'w+') as file:
        file.write(double2Str(minConvAngle_))

    #ff,aa = pu.plt.subplots(2,1)
    #ff.set_size_inches(4,10)
    ff = pu.plt.figure(figsize=(5,10))
    aa = np.array([ ff.add_subplot(211), ff.add_subplot(212) ])
    
    allFigList.append(ff)

    for aOrig in dynDict['plot']['allOrigNregList'][:-1]:
        aa[0].plot(aOrig[0,:],aOrig[1,:],'--k', linewidth=3.)
    for aPatch in dynDict['plot']['allPatchNregList'][:-1]:
        for aaPatch in aPatch:
            aa[0].plot(aaPatch[0,:],aaPatch[1,:],'--r', linewidth=3.)
    aa0Lims = [aa[0].get_xlim(), aa[0].get_ylim()]
    N = 100
    xGrid,yGrid = np.meshgrid(np.linspace(aa[0].get_xlim()[0],aa[0].get_xlim()[1],N),np.linspace(aa[0].get_ylim()[0],aa[0].get_ylim()[1],N))
    XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))
    VGrid = thisControl.getCtrlSpaceVelocity(XGrid)
    aa[0].streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')

    allCtrlTraj = [ares[3] for ares in batchedCallCPP('dirTraj',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,xIn=[ax[:,[0]].copy() for ax in allX])]

    for actrltraj in allCtrlTraj:
        aa[0].plot(actrltraj[0,:],actrltraj[1,:],'b', linewidth=3.)
    
    aa[0].set_xlim(aa0Lims[0])
    aa[0].set_ylim(aa0Lims[1])

    allCtrlTrajPrime = [thisControl.inverseTransform(ax.copy()) for ax in allCtrlTraj]
    XGrid = pu.getGrid(aa[0],[10,1000])
    XGridp = thisControl.inverseTransform(XGrid)

    pu.plotGrid(aa[1],XGridp,(10,1000),c='m')

    N = 100
    xGrid,yGrid = np.meshgrid(np.linspace(aa[1].get_xlim()[0],aa[1].get_xlim()[1],N),np.linspace(aa[1].get_ylim()[0],aa[1].get_ylim()[1],N))
    XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))
    VGrid = callCPP('demSpaceVel',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,XGrid,addInfo=['dem'])[2]
    aa[1].streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')

    allDemTraj = [ares[3] for ares in batchedCallCPP('demSpaceTraj',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,xIn=[ax[:,[0]].copy() for ax in allX],addInfo=len(allX)*[['dem']])]

    for actrltraj,ademtraj,aOrig in zip(allCtrlTraj,allDemTraj,dynDict['plot']['allOrigNregList']):
        aa[1].plot(actrltraj[0,:],actrltraj[1,:],'--b', linewidth=3.)
        aa[1].plot(aOrig[0,:],aOrig[1,:],'--k', linewidth=3.)
        aa[1].plot(ademtraj[0,:],ademtraj[1,:],'-g', linewidth=3.)

    aa[1].set_xlim(aa[0].get_xlim())
    aa[1].set_ylim(aa[0].get_ylim())

    aa[0].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    aa[1].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)

    pu.plt.tight_layout()

    ff,aa = pu.plt.subplots(2,2)
    allFigList.append(ff)

    for aOrig in dynDict['plot']['allOrigNregList'][:-1]:
        aa[0,0].plot(aOrig[0,:],aOrig[1,:],'--k')
    for aPatch in dynDict['plot']['allPatchNregList'][:-1]:
        for aaPatch in aPatch:
            aa[0,0].plot(aaPatch[0,:],aaPatch[1,:],'--r')
        

    aOrig = dynDict['plot']['allOrigNregList'][-1]
    aa[0,0].plot(aOrig[0,:],aOrig[1,:],'-k',linewidth=2.)
    for aaPatch in dynDict['plot']['allPatchNregList'][-1]:
        aa[0,0].plot(aaPatch[0,:],aaPatch[1,:],'-r',linewidth=2.)

    for aOrig in dynDict['plot']['allOrigNregList'][:-1]:
        aa[1,0].plot(aOrig[0,:],aOrig[1,:],'--k')
    for aPatch in dynDict['plot']['allPatchNregList'][:-1]:
        for aaPatch in aPatch:
            aa[1,0].plot(aaPatch[0,:],aaPatch[1,:],'--r')

            
    N=100
    xGrid, yGrid = np.meshgrid( np.linspace(aa[1,0].get_xlim()[0], aa[1,0].get_xlim()[1], N), np.linspace(aa[1,0].get_ylim()[0], aa[1,0].get_ylim()[1], N)  )
    XGrid = np.vstack((xGrid.flatten(), yGrid.flatten()))
    VGrid = thisControl.getCtrlSpaceVelocity(XGrid)
    aa[1,0].streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')

    allCtrlTraj = [ares[3] for ares in batchedCallCPP('dirTraj',(2,0),namedOutFolder, resultFolder, resultFolder, cppExe, xIn=[ax[:,[0]].copy() for ax in allX]) ]
    
    for actrltraj in allCtrlTraj:
        aa[1,0].plot(actrltraj[0,:], actrltraj[1,:], 'b')
    
    allCtrlTrajPrime = [thisControl.inverseTransform(ax.copy()) for ax in allCtrlTraj]
    XGrid = pu.getGrid(aa[1,0], [15,500])
    XGridp = thisControl.inverseTransform(XGrid)
    
    pu.plotGrid(aa[0,1], XGridp, (15,500), c='c')
    
    for actrltraj, actrltrajp, aOrig in zip(allCtrlTraj, allCtrlTrajPrime, dynDict['plot']['allOrigNregList']):
        aa[0,1].plot(actrltraj[0,:], actrltraj[1,:], '--b')
        aa[0,1].plot(aOrig[0,:],aOrig[1,:],'--k')
        aa[0,1].plot(actrltrajp[0,:],actrltrajp[1,:],'-g')

    N = 100
    xGrid,yGrid = np.meshgrid(np.linspace(aa[1,0].get_xlim()[0],aa[1,0].get_xlim()[1],N),np.linspace(aa[1,0].get_ylim()[0],aa[1,0].get_ylim()[1],N))
    XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))
    VGrid = callCPP('demSpaceVel', (2,0), namedOutFolder, resultFolder, resultFolder, cppExe, XGrid, addInfo=['dem'])[2]
    aa[1,1].streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')

    allDemTraj = [ares[3] for ares in batchedCallCPP('demSpaceTraj',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,xIn=[ax[:,[0]].copy() for ax in allX],addInfo=len(allX)*[['dem']])]
    
    for ademtraj, aOrig in zip(allDemTraj, dynDict['plot']['allOrigNregList']):
        aa[1,1].plot(aOrig[0,:],aOrig[1,:],'--k')
        aa[1,1].plot(ademtraj[0,:],ademtraj[1,:],'-g')
    
    aa[1,0].set_xlim(aa[0,0].get_xlim())
    aa[1,0].set_ylim(aa[0,0].get_ylim())
    aa[1,1].set_xlim(aa[0,0].get_xlim())
    aa[1,1].set_ylim(aa[0,0].get_ylim())

    aa[0,0].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    aa[1,0].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    aa[0,1].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
    aa[1,1].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)

    fff,aaa = pu.plt.subplots(1,1)
    allFigList.append(fff)

    # Start off plotting the deformed grid
    XGrid = pu.getGrid(aa[0,0],[8,1000])
    XGridp = thisControl.inverseTransform(XGrid)
    pu.plotGrid(aaa,XGridp,(8,1000),c='m')
    # streamlines
    N = 100
    xGrid,yGrid = np.meshgrid(np.linspace(aa[0,0].get_xlim()[0],aa[0,0].get_xlim()[1],N),np.linspace(aa[0,0].get_ylim()[0],aa[0,0].get_ylim()[1],N))
    XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))
    VGrid = callCPP('demSpaceVel',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,XGrid,addInfo=['dem'])[2]
    aaa.streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')

    # plot demonstration and ctrl space traj
    for ax in allX:
        aaa.plot(ax[0,:],ax[1,:],'-k',linewidth=2.)
    for ax in allCtrlTraj:
        aaa.plot(ax[0,:],ax[1,:],'-b',linewidth=2.)
    for ax in allDemTraj:
        aaa.plot(ax[0,:],ax[1,:],'-g',linewidth=2.)
        
    aaa.set_xlim(aa[0,0].get_xlim())
    aaa.set_ylim(aa[0,0].get_ylim())

    aaa.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)

    return thisControl,allFigList
    
    ff,aa = pu.plt.subplots(1,1)
    cc = pu.plt.get_cmap('jet')(np.linspace(0,1,len(thisDiffeo._transformationList)))
    for ax in allCtrlTraj:
        aa.plot(*ax, 'k')
    
    xxx = deepcopy(allCtrlTraj)
    
    for k,aTrans in enumerate(thisDiffeo._transformationList):
        xxx = [aTrans.forwardTransform(ax.copy()) for ax in xxx]
        for ax in xxx:
            aa.plot(*ax,color=cc[k,:])
    
    
    # One more beautiful plot
    if (_otherOptions['doPlot'] >= 3 and (dim == 2)):
        # Plot the resulting replay left
        # plot source target grid right
        ff,aa = pu.plt.subplots(1,2)
        allFigList.append(ff)
        
        for ax in allX:
            aa[0].plot(ax[0,:],ax[1,:],'k',linewidth=1.)
        
        for ax,at in zip(allX,allT):
            xInit = ax[:,[0]].copy()
            xTraj = thisControl.getTrajectory(xInit,at)[0]
            
            aa[0].plot(xTraj[0,:],xTraj[1,:],'g',linewidth=2.)
        
        # Plot learned
        for ay in allY:
            aa[1].plot(ay[0,:],ay[1,:],'k',linewidth=2.)

        # Plot streamlines
        N = 200
        xGrid,yGrid = np.meshgrid(np.linspace(aa[1].get_xlim()[0],aa[1].get_xlim()[1],N),np.linspace(aa[1].get_ylim()[0],aa[1].get_ylim()[1],N))
        
        XGrid = np.vstack((xGrid.flatten(),yGrid.flatten()))
        
        # Get the ctrl space
        VGrid = thisControl.getCtrlSpaceVelocity(XGrid.copy())
        
        aa[1].streamplot(xGrid,yGrid,VGrid[0,:].reshape((N,N)),VGrid[1,:].reshape((N,N)),color='c')
        
        aa[0].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        aa[0].set(adjustable='box-forced',aspect='equal')
        aa[1].tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        aa[1].set(adjustable='box-forced',aspect='equal')
    
    if (_otherOptions['doPlot'] > 0 and (dim == 2)):
        
        nGrid = (10,200)
        xxTemp = np.hstack(allX)
        xGrid = pu.getGrid([np.min(xxTemp[0,:]),np.max(xxTemp[0,:]),np.min(xxTemp[1,:]),np.max(xxTemp[1,:])],nGrid)
        
        xCurrent = [deepcopy(allY)]
        xGridCurrent = [xGrid.copy()]
        
        for aTrans in thisDiffeo._transformationList:
            
            ff,aa = pu.plt.subplots(1,1)
            allFigList.append(ff)
            # Update and plot grid
            xGridCurrent.append(aTrans.forwardTransform(xGridCurrent[-1].copy()))
            
            pu.plotGrid(aa,xGridCurrent[-1],nGrid,'c')

            # source target last
            for ax,ay,lastx in zip(allX,allY,xCurrent[-1]):
                aa.plot(ax[0,:],ax[1,:],'b',linewidth=2)
                aa.plot(ay[0,:],ay[1,:],'k',linewidth=2)
                aa.plot(lastx[0,:],lastx[1,:],'--r')
            xlims,ylims = aa.get_xlim(),aa.get_ylim()
            pu.plotTransformation(aa,aTrans)

            # update current
            xCurrent.append([aTrans.forwardTransform(aax.copy()) for aax in xCurrent[-1]])

            # current
            for currentx in xCurrent[-1]:
                aa.plot(currentx[0,:],currentx[1,:],'g',linewidth=2)
            
            aa.tick_params(axis='both',which='both',bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
            aa.set(adjustable='box-forced',aspect='equal')
            aa.set_xlim(xlims)
            aa.set_ylim(ylims)
    
    if _otherOptions['computeSEA_']:
        
        #Compute all traj
        allDemTrajTimed = [ares[3] for ares in batchedCallCPP('demSpaceTraj',(2,0),namedOutFolder,resultFolder,resultFolder,cppExe,xIn=[ax[:,[0]].copy() for ax in allX], tIn=allT,addInfo=len(allX)*[['dem']])]
        
        averageSEA = sum( np.sum(SEAerror(axD, axT)) for axD,axT in zip(allX,allDemTrajTimed) )/len(allX)
        
        ff,aa = pu.plt.subplots(2,1)
        fff,aaa = pu.plt.subplots(1,1)
        aa[0].set_title("SEA error is {0}".format(averageSEA))
        cc = pu.plt.get_cmap('jet')(np.linspace(0,1,len(allX)))
        for k,(at,axD,axT) in enumerate(zip(allT, allX, allDemTrajTimed)):
            aa[0].plot(at, axD[0,:], '--', color=cc[k,:])
            aa[0].plot(at,axT[0,:],'-',color=cc[k,:])
            aa[1].plot(at,axD[1,:],'--',color=cc[k,:])
            aa[1].plot(at,axT[1,:],'-',color=cc[k,:])
            aaa.plot(axD[0,:], axD[1,:], '--', color=cc[k,:])
            aaa.plot(axT[0,:],axT[1,:],'-',color=cc[k,:])
            pu.myQuiver(aaa, axT[:,0::50], axD[:,0::50]-axT[:,0::50], c=list(cc[k,:]))
            
    print("Constructed diffeo has {0} multitransformations with a total of {1} kernels".format(len(thisDiffeo._transformationList),sum([aTrans.nTrans for aTrans in thisDiffeo._transformationList])))
    for ii,aTrans in enumerate(thisDiffeo._transformationList):
        print("Multitrans {0} has {1}".format(ii,aTrans.nTrans))
    
    return thisControl,allFigList


def simpleNamedSearch(name):
    import os
    import pickle
    osabspath = os.path.abspath
    # Test
    # searchNStoreVectorfield(name, inputFolder, outputFolder, resultFolder, cppExe=None, nDemos=7, optionsSeries={}, optionsMatching={}, otheroptions={}, parsDyn={}, parsWeights={}, optionsMagMod={}):
    inputFolder = osabspath('../data')
    outputFolder = osabspath('../tmpFiles')
    resultFolder = osabspath('../tempResults')
    mkdir_p(outputFolder)
    mkdir_p(resultFolder)
    
    optionsSeries = {'errStep':np.Inf,'p':5.,'alpha':-0.125,'doRound':True,'doRoundStart':False,'matchDirections':True, '_steps':[0.,1.], '_direct':1}
    otherOptions = {'Nstep':1,'doPlot':False,'nPtsDirDyn':30, '_nSteps':[0.,1.], '_direct':1, 'computeSEA_':True}

    _optionsMagMod = {'add2Sigma':2.,'iterMax':100,'relTol':.5e-2,'absTol':.25e-2,'interOpt':True,'reOpt':False,'convFac':1.,'nPartial':10}
    
    optionsMatching = {'baseRegVal':.1e-3*.05*20.,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':6,'maxNKernelErrStruct':12,'nInterTrans':6,'minTrans':-.1e-3,'minBase':-.1e-1,
                       'marginSingle':0.5,'marginOverlap':0.01,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-5,'epsConstraints':3.e-5,
                       "repeatStart_":[-.1,5]}
    ffac = 1.5
    parsDyn = {'alpha':-.5/ffac,'alpha0':-.25/ffac,'alphaF':0.,'beta':1./ffac,'beta0':2./ffac,'betaF':.75/ffac,'baseConv':-1.e-300,'addFinalConv':True,'_finalConvBeta':-1.,'_finalConvWeight':25,'_nptPerSeg':int(50*ffac)}
    parsWeights = {'centerPoint':0.5,'centerPoint0':0.5,'centerPointF':0.5,'maxEndInf':0.1, 'maxEndInf0':0.3,'maxEndInfF':0.05,'orthCovScale':-3./ffac,'orthCovScale0':-3./ffac,'orthCovScaleF':-3./ffac,'doCond':False,'finalCov':2.}
    mainParsDict_['directionCostCoeff_'] = 1.1
    

    cppExe = os.path.abspath("../cpp/cmake-build-debug/diffeoInterface")
    
    thisControl,thisFigList = searchNStoreVectorfield(name,inputFolder,outputFolder,resultFolder,cppExe=cppExe,optionsSeries=optionsSeries,optionsMatching=optionsMatching,
                                                       otheroptions=otherOptions,parsDyn=parsDyn,parsWeights=parsWeights, optionsMagMod=_optionsMagMod)

    picPath = '../results/'
    mkdir_p(picPath)
    mkdir_p(picPath+"{0}".format(name))
    
    for k,aFig in enumerate(thisFigList):
        with open(os.path.join(picPath+"{0}".format(name),"{0}.pickle".format(k)),'wb+') as file:
            pickle.dump(aFig,file)
        aFig.savefig(os.path.join(picPath+"{0}".format(name),"{0}.png".format(k)),format='png')
        aFig.savefig(os.path.join(picPath+"{0}".format(name),"{0}.pdf".format(k)),format='pdf')
    
    import json
    with open(os.path.join(picPath+"optsSeries.json"),'w+') as fp:
        json.dump(optionsSeries,fp)
    with open(os.path.join(picPath+"optsMatching.json"),'w+') as fp:
        json.dump(optionsMatching,fp)
    with open(os.path.join(picPath+"otherOptions.json"),'w+') as fp:
        json.dump(otherOptions,fp)
    with open(os.path.join(picPath+"parsDyn.json"),'w+') as fp:
        json.dump(parsDyn,fp)
    with open(os.path.join(picPath+"parsWeights.json"),'w+') as fp:
        json.dump(parsWeights,fp)
    subprocess.call(["cp","./mainPars.py",picPath],shell=False)
    subprocess.call(["cp","./ctrlSpaceDyn.py",picPath],shell=False)

    # pu.plt.show()


if __name__ == '__main__':
    
    import os
    
    all = ['Leaf_2','GShape','BendedLine','DoubleBendedLine','Leaf_1','Sharpc','Snake', 'NShape']
    all += ['doubleSharpC','heeWorm']
    succ = []
    fail = []
    for aName in all:
        if 1:
            simpleNamedSearch(aName)
        else:
            try:
                simpleNamedSearch(aName)
                succ.append(aName)
            except:
                print('FAIL '+aName)
                fail.append(aName)
                pass
    print("Success")
    print(succ)
    print("Fail")
    print(fail)
    pu.plt.show()





