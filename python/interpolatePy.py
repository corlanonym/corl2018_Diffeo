from copy import deepcopy

from combinedDynamics import combinedLocallyWeightedDirections,pointListToCombinedDirections,minimallyConvergingDirection,convergingDirections,locallyWeightedDirections
from coreUtils import *

from scipy.interpolate import interp1d

from modifiedEM import gaussianKernel


def regularSpline_old(allX:List[np.ndarray], allT:List[np.ndarray], allV:List[np.ndarray]=None, doPlot:bool=True) -> List[List[np.ndarray]]:
    """Interpolate linear equal spaced points"""
    allV = len(allT)*[None] if allV is None else allV
    if doPlot:
        import plotUtils as pu
        ff,aa = pu.plt.subplots(3,1)
        
        for aX, aV in zip(allX, allV):
            aa[0].plot(aX[0,:], aX[1,:], '-k')
            aa[1].plot(aX[0,:],aX[1,:],'-k')
            aa[2].plot(aX[0,0:-1:20],aX[1,0:-1:20],'.k')
            if aV is not None:
                pu.myQuiver(aa[1], aX[:,0:-1:50], aV[:,0:-1:50], 'r')
    
    allXnew = []
    allVnew = []
    allTnew = []
    for aX, aT in zip(allX, allT):
        dim,nPt = aX.shape
        dxNorm = cNorm(np.diff(aX, 1), kd=False)
        l = np.hstack((0, np.cumsum(dxNorm)))
        li = np.linspace(l[0], l[-1], nPt)
        
        dt = abs(aT[0]-aT[-1])/nPt
        aTnew = np.linspace(aT[0], aT[-1], nPt)
        
        aXnewI = interp1d(l, aX, copy=False)
        aXnew = aXnewI(li)
        
        # get vel
        aVnew = np.hstack(( np.diff(aXnew, 1), -1e-3*aXnew[:,[-1]] ))/dt
        
        allXnew.append(np.require(aXnew, dtype=np.float_, requirements=['C','O','W']))
        allVnew.append(np.require(aVnew, dtype=np.float_, requirements=['C','O','W']))
        allTnew.append(np.require(aTnew, dtype=np.float_, requirements=['C','O','W']))
    
    if doPlot:
        for aX, aV in zip(allXnew, allVnew):
            aa[0].plot(aX[0,:], aX[1,:], '-g')
            aa[1].plot(aX[0,:],aX[1,:],'-g')
            aa[2].plot(aX[0,0:-1:20],aX[1,0:-1:20],'.g')
            pu.myQuiver(aa[1], aX[:,0:-1:50], aV[:,0:-1:50], 'b')
    
    return allXnew, allVnew, allTnew

def regularSpline2_wrong(allX: List[np.ndarray],allT:List[np.ndarray],allV: List[np.ndarray],doPlot:bool = True) -> List[List[np.ndarray]]:
    """Interpolate linear equal spaced points"""
    if doPlot:
        import plotUtils as pu
        dim,nPt = allX[0].shape
        ff,aa = pu.plt.subplots(3,1)
        fff,aaa = pu.plt.subplots(dim,2)
        for aX,aV,aT in zip(allX,allV,allT):
            aa[0].plot(aX[0,:],aX[1,:],'-k')
            aa[1].plot(aX[0,:],aX[1,:],'-k')
            aa[2].plot(aX[0,0:-1:20],aX[1,0:-1:20],'.k')
            pu.myQuiver(aa[1],aX[:,0:-1:50],aV[:,0:-1:50],'r')
            for kk in range(dim):
                aaa[kk,0].plot(aT, aX[kk,:],'-k')
                aaa[kk,1].plot(aT, aV[kk,:],'-k')
                
    
    allXnew = []
    allVnew = []
    allTnew = []
    for aX,aV,aT in zip(allX,allV,allT):
        dim,nPt = aX.shape
        dxNorm = cNorm(np.diff(aX,1),kd=False)
        l = np.hstack((0,np.cumsum(dxNorm)))
        li = np.linspace(l[0],l[-1],nPt)
        
        dt = abs(aT[0]-aT[-1])/nPt
        aTnew = np.linspace(aT[0],aT[-1],nPt)
        
        #get pos
        aXnewI = interp1d(l,aX,copy=False)
        aXnew = aXnewI(li)
        
        # get vel
        aVnewI = interp1d(l,aV,copy=False)
        aVnew = aVnewI(li)
        
        #get time
        aTnewI = interp1d(l,aT,copy=False)
        aTnew = aTnewI(li)
        
        allXnew.append(np.require(aXnew,dtype=np.float_,requirements=['C','O','W']))
        allVnew.append(np.require(aVnew,dtype=np.float_,requirements=['C','O','W']))
        allTnew.append(np.require(aTnew,dtype=np.float_,requirements=['C','O','W']))
    
    if doPlot:
        for aX,aV,aT in zip(allXnew,allVnew,allTnew):
            aa[0].plot(aX[0,:],aX[1,:],'-g')
            aa[1].plot(aX[0,:],aX[1,:],'-g')
            aa[2].plot(aX[0,0:-1:20],aX[1,0:-1:20],'.g')
            pu.myQuiver(aa[1],aX[:,0:-1:50],aV[:,0:-1:50],'b')
            for kk in range(dim):
                aaa[kk,0].plot(aT, aX[kk,:],'-g')
                aaa[kk,1].plot(aT, aV[kk,:],'-g')
    
    return allXnew,allVnew,allTnew

def regularSplineAA(allX: List[np.ndarray],allT:List[np.ndarray],allV: List[np.ndarray],doPlot:bool = True) -> List[List[np.ndarray]]:
    
    from scipy.interpolate import pchip
    if doPlot:
        import plotUtils as pu
    
    allXnew = []
    allVnew = []
    allTnew = []
    for aX,aV,aT in zip(allX,allV,allT):
        dim,nPt = aX.shape
        
        ti = np.linspace(aT[0], aT[-1], 5*nPt)
        xi = interppp(aT, aX, axis=1)(ti)
        vi = interppp(aT,aV, axis=1)(ti)
        
        
        dxNorm = cNorm(np.diff(xi,1),kd=False)
        l = np.hstack((0,np.cumsum(dxNorm)))
        li = np.linspace(l[0],l[-1],nPt)
        
        idx = 1
        tNew = np.zeros((2*nPt))
        xNew = np.zeros((dim,2*nPt))
        vNew = np.zeros((dim,2*nPt))
        
        tNew[0] = aT[0]
        xNew[:,0] = aX[:,0]
        vNew[:,0] = aV[:,0]
        
        for k in range(5*nPt):
            if l[k] < li[idx]:
                continue
            
            tNew[idx] = ti[k]
            xNew[:,idx] = xi[:,k]
            vNew[:,idx] = vi[:,k]
            idx+=1
        
        tNew = tNew[:idx]
        xNew = xNew[:,:idx]
        vNew = vNew[:,:idx]

        allXnew.append(np.require(xNew,dtype=np.float_,requirements=['C','O','W']))
        allVnew.append(np.require(vNew,dtype=np.float_,requirements=['C','O','W']))
        allTnew.append(np.require(tNew,dtype=np.float_,requirements=['C','O','W']))
        
        if doPlot:
            ff,aa = pu.plt.subplots(2,1)
            for k in range(dim):
                aa[0].plot(aT, aX[k,:], '--')
                aa[0].plot(tNew,xNew[k,:],'-')

                aa[1].plot(aT,aV[k,:],'--')
                aa[1].plot(tNew,vNew[k,:],'-')
                aa[1].plot(tNew[:-1],np.diff(xNew[k,:])/np.diff(tNew),'.-')
                
                

    return allXnew,allVnew,allTnew
        
    
def regularSpline(allX: List[np.ndarray], allT:List[np.ndarray]=None,allV: List[np.ndarray]=None, indList:np.ndarray=None, sampleList=None,doPlot:bool = True) -> List[List[np.ndarray]]:
    
    #from scipy.interpolate import pchip as interp1ddd
    from scipy.interpolate import interp1d
    
    interp1ddd = lambda *args, **kwargs: interp1d(*args, kind='cubic', **kwargs)
    
    if doPlot:
        import plotUtils as pu
    
    returnList = True
    
    if indList is not None:
        assert isinstance(allX, np.ndarray)
        returnList = False
        allX = [allX[:,indList[ii]:indList[ii+1]] for ii in range(len(indList)-1)]
        if isinstance(allT, np.ndarray):
            allT = [allT[:,indList[ii]:indList[ii+1]] for ii in range(len(indList)-1)]
        if isinstance(allV,np.ndarray):
            allV = [allV[:,indList[ii]:indList[ii+1]] for ii in range(len(indList)-1)]
        
    
    allXnew = []
    allVnew = []
    allTnew = []
    
    if allT == None:
        allT = [ np.linspace(0.,1.,ax.shape[1]) for ax in list(allX)]
    
    if allV is None:
        allV = len(allX)*[None]
        
    
    
    for nCurve, (aX,aV,aT) in enumerate(zip(allX,allV,allT)):
        dim,nPt = aX.shape
        
        nPt2 = nPt if sampleList is None else sampleList[nCurve]
        
        ti = np.linspace(aT[0],aT[-1],5*nPt)
        xi = interp1ddd(aT,aX,axis=1)(ti)
        vi = interp1ddd(aT,aV,axis=1)(ti) if (aV is not None) else None
        
        dxNorm = cNorm(np.diff(xi,1),kd=False)
        l = np.hstack((0,np.cumsum(dxNorm)))
        li = np.linspace(l[0],l[-1],nPt2)
        
        tiNew = interp1ddd(l, ti)(li)
        xiNew = interp1ddd(l, xi, axis=1)(li)
        viNew = interp1ddd(l,vi,axis=1)(li) if (aV is not None) else None
        
        allTnew.append(np.require(tiNew,dtype=np.float_,requirements=['C','O','W']))
        allXnew.append(np.require(xiNew,dtype=np.float_,requirements=['C','O','W']))
        allVnew.append(np.require(viNew,dtype=np.float_,requirements=['C','O','W']) if (aV is not None) else None)
    
    if returnList:
        return allXnew, allTnew, allVnew
    else:
        try:
            allVnew = np.hstack(allVnew)
        except TypeError:
            allVnew = None
        return np.hstack(allXnew), np.hstack(allTnew),allVnew



if __name__ == '__main__':
    
    from coreUtils import *
    import plotUtils as pu
    import time
    from combinedDynamics import getMagnitudeModel, combinedDiffeoCtrl
    
    N=200
    
    t = np.linspace(0.,1.,N)
    x = np.vstack(( np.sin(2.*np.pi*t), t**3 ))
    
    xi,ti,_ = regularSpline([x])
    
    ti = ti[0]
    xi = xi[0]
    
    ff,aa = pu.plt.subplots(1,1)
    
    aa.plot(x[0,:], x[1,:], '.-r')
    aa.plot(xi[0,:],xi[1,:],'.-b')

    N = 20

    t = np.linspace(0.,1.,N)
    x = np.vstack((np.sin(2.*np.pi*t),t**3))

    tStart = time.time()
    xi,ti,_ = regularSpline(10*[x])
    tTot = time.time()-tStart
    
    print("10 time 2000 points took {0} s".format(tTot))
    
    # Testing match
    
    from trajectoryUtils import stochasticMatching
    if 0:
        N = 200
    
        t = np.linspace(1.,2.,N)
        x = np.vstack((np.sin(2.*np.pi*t),t**3))
        x = np.fliplr(x)
        x -= x[:,[-1]]
        y = np.vstack( [np.linspace(x[i,0], x[i,-1], N) for i in range(2) ] )
        
        xSeries = [[y.copy()], [x.copy()]]
        
        optionsMatching = {'baseRegVal':.1e-3,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':10,'maxNKernelErrStruct':12,'nInterTrans':15,'minTrans':-.1e-3,'minBase':-5.e-2,
                           'marginSingle':0.5,'marginOverlap':0.05,'transCenterFac':0.15,'matchDirections':True, 'alphaConstraints':-.5e-3, 'epsConstraints':3.e-3}
        
        thisMatching = stochasticMatching([x], [np.zeros_like(x)], [t], optionsMatching=optionsMatching)
        
        thisDiffeo = thisMatching.doMatching(xSeries)
        
        ytilde = thisDiffeo.forwardTransform(y)
        
        ff,aa = pu.plt.subplots(1,1)
        
        aa.plot(x[0,:],x[1,:],'r')
        aa.plot(y[0,:],y[1,:],'b')
        aa.plot(ytilde[0,:],ytilde[1,:],'g')
    
    
    # Try more realistic
    if 0:
        allX = []
        for k in range(7):
            allX.append(np.loadtxt("./data/NShape"+"/pos_{0}.txt".format(k+1)))
    
        allY = [np.vstack([np.linspace(ax[i,0],ax[i,-1],ax.shape[1]) for i in range(2)]) for ax in allX]
    
        xSeries = [deepcopy(allY),deepcopy(allX)]
    
        optionsMatching = {'baseRegVal':5.e-3,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':10,'maxNKernelErrStruct':12,'nInterTrans':15,'minTrans':-.1e-3,'minBase':-1.e-2,
                           'marginSingle':0.5,'marginOverlap':0.05,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-3,'epsConstraints':3.e-3}
    
        thisMatching = stochasticMatching([np.vstack(2*[np.linspace(1.,0.,10)])],[-np.ones((2,10))],[np.linspace(0.,1.,10)],optionsMatching=optionsMatching)
    
        thisDiffeo = thisMatching.doMatching(xSeries)
    
        ff,aa = pu.plt.subplots(1,1)
        
        for k in range(7):
            aaX = thisDiffeo.forwardTransform(allY[k])
            aa.plot(allX[k][0,:],allX[k][1,:],'r')
            aa.plot(allY[k][0,:],allY[k][1,:],'b')
            aa.plot(aaX[0,:],aaX[1,:],'g')
    
    #Complete
    if 1:
        for dataName in ['Snake', 'Leaf_2', 'GShape', 'NShape', 'Leaf_1', 'BendedLine', 'Angle', 'DoubleBendedLine', 'Sharpc', 'Snake', 'Spoon', 'Multi_Models_1', 'Multi_Models_2', 'Multi_Models_3', 'Multi_Models_4']:
            
            allFigList = []
            
            Nstep = 2
            allT = []
            allX = []
            allV = []
            sampleList = []
            for k in range(7):
                allT.append(np.loadtxt(("./data/{1}"+"/t_{0}.txt").format(k+1, dataName))[0:-1:Nstep]  )
                allX.append(np.loadtxt(("./data/{1}"+"/pos_{0}.txt").format(k+1, dataName))[:, 0:-1:Nstep]  )
                allV.append(np.loadtxt(("./data/{1}"+"/vel_{0}.txt").format(k+1, dataName))[:, 0:-1:Nstep]  )
                sampleList.append(allT[-1].size)
            
            optionsSeries = {'errStep':np.Inf,'p':5.,'alpha':-0.3,'doRound':True,'doRoundStart':True, 'matchDirections':True}
            optionsMatching = {'baseRegVal':1.e-3,'dValues':.2,'includeMaxFac':0.8,'maxNKernel':12,'maxNKernelErrStruct':12,'nInterTrans':12,'minTrans':-.1e-2,'minBase':-.1e-2,
                               'marginSingle':0.5,'marginOverlap':0.01,'transCenterFac':0.15,'matchDirections':True,'alphaConstraints':-.5e-5,'epsConstraints':3.e-5}
            
            thisMatching = stochasticMatching(allX, allV, allT, optionsSeries=optionsSeries, optionsMatching=optionsMatching)
            
            xSeriesInversed = thisMatching.getSeries(returnVel=False)
            
            # Deduce the direction dynamics
            allYfeasible = xSeriesInversed[-1]
            # regular
            allY = regularSpline(allYfeasible)[0]
    
            #pointListToCombinedDirections(points: np.ndarray,parsWeights: dict = {},parsDyn:dict = {} )->combinedLocallyWeightedDirections:
            nPts = 15
            #parsDyn = {'alpha':-0.05,'beta':-.1,'baseConv':-1.e-6}
            #parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':-0.75,'doCond':False}
            parsDyn = {'alpha':.5,'beta':-.1,'baseConv':-1.e-6}
            parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':4.,'doCond':False}
            allDirDyn = combinedLocallyWeightedDirections(parsDyn['baseConv'])
            for ay in allY:
                thisI = np.linspace(0,ay.shape[1]-1, nPts, dtype=np.int_)
                thisDirDyn = pointListToCombinedDirections(ay[:,thisI].copy(),parsWeights=parsWeights ,parsDyn=parsDyn )
                
                allDirDyn._dirList += thisDirDyn._dirList
            
            # Add a final converging
            finalConv = convergingDirections(x0=np.zeros((2,1)), vp=np.zeros((2,1)), alpha=0., beta=-5.)
            finalWeight = gaussianKernel(1, 3.*np.identity(2), np.zeros((2,1)), doCond=parsWeights['doCond'])
            finalDir = locallyWeightedDirections(finalWeight, finalConv)
            
            allDirDyn.addDyn(finalDir)
            
            fConv = lambda x,v: minimallyConvergingDirection(x,v,minAng=0.025, minAngIsAng=False)
            allDirDyn._ensureConv = fConv
            
            ff,aa = pu.plt.subplots(1,1)
            allFigList.append(ff)
            
            for ay in allY:
                aa.plot(ay[0,:], ay[1,:], 'k')
            
            xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,200), np.linspace(aa.get_ylim()[0]-5,aa.get_ylim()[1]+5,200))
            XX = np.vstack((xx.flatten(),yy.flatten()))
            VV = allDirDyn.getDir(XX)
            
            aa.streamplot(xx,yy,VV[0,:].reshape((200,200)),VV[1,:].reshape((200,200)))
            
            # Now with the direction dynamics established get trajectories
            xInit = np.hstack( [ax[:,[0]] for ax in allX] )
            allSol = allDirDyn.getDirTraj(xInit)
            
            #Set to zero
            for _,ax in allSol:
                ax[:,-1] = 0.

            newX = regularSpline([ax for at,ax in allSol],sampleList=sampleList)[0]

            xSeriesInversed = deepcopy([allX,newX])

            for at,ax in allSol:
                aa.plot(ax[0,:],ax[1,:],'r')
            aa.axis('equal')

            allY = xSeriesInversed[-1]
            
            #xSeriesMatched = list(reversed(deepcopy(xSeriesInversed)))

            thisDiffeo = thisMatching.doMatching(list(reversed(deepcopy(xSeriesInversed))))
            
            XXgrid = pu.getGrid([aa.get_xlim()[0]-5,aa.get_xlim()[1]+5, aa.get_ylim()[0]-5,aa.get_ylim()[1]+5], [20,200])
            
            XXgridp = thisDiffeo.forwardTransform(XXgrid.copy())
            
            ff,aa = pu.plt.subplots(1,1)
            allFigList.append(ff)
            
            pu.plotGrid(aa, XXgridp, [20,200], 'm')
            
            for ax in allX:
                aa.plot(ax[0,:],ax[1,:],'k')

            xx,yy = np.meshgrid(np.linspace(aa.get_xlim()[0]-5,aa.get_xlim()[1]+5,50),np.linspace(aa.get_ylim()[0]-5,aa.get_ylim()[1]+5,50))
            XX = np.vstack((xx.flatten(),yy.flatten()))
            
            XXp = thisDiffeo.inverseTransform(XX.copy())
            VVp = allDirDyn.getDir(XXp)
        
            XXt,VVt = thisDiffeo.forwardTransformJac(x=XXp.copy(), v=VVp.copy())
        
            print(np.max(np.abs(XX-XXt)))
            
            aa.streamplot(xx,yy,VVt[0,:].reshape((50,50)),VVt[1,:].reshape((50,50)))
            aa.axis('equal')
            
    
            ff,aa = pu.plt.subplots(2,1)
            allFigList.append(ff)
    
            for k in range(len(allX)):
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
            
            #Test actual integration
            #def getMagnitudeModel(x:np.ndarray,v:np.ndarray,nKMax:int=12, opts={}):
            optsMagMod = {'add2Sigma':5.,
                          'iterMax':100,
                          'relTol':2e-2,'absTol':1.e-2,
                          'interOpt':True,'reOpt':False,'convFac':1.,
                          'nPartial':10
                          }
            allXstacked = np.hstack(allX)
            allVstacked = np.hstack(allV)
            allVn = cNorm(allVstacked,kd=True)
            thisMagModel = getMagnitudeModel(allXstacked, allVn, opts=optsMagMod)
            
            # Check quality
            allVnPred = thisMagModel.evalMap(allXstacked)
            ff,aa = pu.plt.subplots(1,1)
            allFigList.append(ff)
            aa.plot(allVn.squeeze(), 'b')
            aa.plot(allVnPred.squeeze(), 'r')
            
            # assemble combined control
            thisControl = combinedDiffeoCtrl(allDirDyn, thisMagModel, thisDiffeo, False)
            
            #Get all velocities
            allVtest = thisControl.getDemSpaceVelocity(allXstacked)
            ff,aa = pu.plt.subplots(allXstacked.shape[0],1)
            for k in range(allXstacked.shape[0]):
                aa[k].plot(allVstacked[k,:], 'b')
                aa[k].plot(allVtest[k,:], 'r')
            
            
            #diffeoCombinedDyn
            
            ff,aa = pu.plt.subplots(1,1)
            allFigList.append(ff)

            for k in range(len(allX)):
                aa.plot(allX[k][0,:],allX[k][1,:],'k')

            import sys
            sys.exit()
            
            ii = np.random.randint(0, len(allX))
            xIn = allX[ii][:,[0]]
            xInCtrl = thisControl.forwardTransform(xIn.copy())
            thisTraj = thisControl.getTrajectory(xIn, allT[ii]*1.1)[0] #This takes a insanely long time
            thisTrajDirCtrl = allDirDyn.getDirTraj(xInCtrl)[0][1]
            thisTrajDirDem = thisControl.inverseTransform(thisTrajDirCtrl)
            
            aa.plot(thisTraj[0,:], thisTraj[1,:], 'r')
            aa.plot(thisTrajDirCtrl[0,:],thisTrajDirCtrl[1,:],'c')
            aa.plot(thisTrajDirDem[0,:],thisTrajDirDem[1,:],'m')

            aa.axis('equal')

            print(dataName)
            
            for atrans in thisDiffeo._transformationList:
                print(atrans.nTrans)
            
            print(dataName)
            
            for k, afig in enumerate(allFigList):
                afig.savefig('./images/{0}_{1}.png'.format(dataName, k), format='png')
            
            import sys
            sys.exit()
            
            
        
    
    
    
    pu.plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        