from copy import deepcopy

from coreUtils import *
from diffeoUtils import diffeomorphism
from modifiedEM import GaussianMixtureModel
from mainPars import *


def indicatorFunction(positiveWeights:"GMM or comvinedDirDyn", negativeWeights:"GMM or comvinedDirDyn", positiveRelWeights:"List array or int"=None, negativeRelWeights:"List array or int"=None):
    from functools import reduce
    if isinstance(positiveWeights, GaussianMixtureModel):
        listPos = positiveWeights._gaussianList
        if positiveRelWeights is None:
            positiveRelWeights = positiveWeights._prior
    else:
        listPos = [aDir._prob for aDir in positiveWeights._dirList]
        if positiveRelWeights is None:
            positiveRelWeights = 1.

    if isinstance(negativeWeights, GaussianMixtureModel):
        listNeg = negativeWeights._gaussianList
        if negativeRelWeights is None:
            negativeRelWeights = negativeWeights._prior
    else:
        listNeg = [aDir._prob for aDir in negativeWeights._dirList]
        if positiveRelWeights is None:
            positiveRelWeights = 1.
    
    # Sum up
    #sum = reduce(lambda asum, aGK)
    
        
    
    

def minimallyConvergingDirection(xIn:np.ndarray, dirIn:np.ndarray, minAng:bool=0., xIsNorm:bool=False, minAngIsAng:bool=True):
    
    if xIsNorm:
        x = xIn
    else:
        x = xIn/(cNorm(xIn)+epsFloat)
    
    if minAngIsAng:
        minAng = 1.-np.cos(minAng)
    
    #compute dot prod
    coefFac = 1.1
    while True:
        acosxdirDiff = np.sum(x*dirIn, axis=0, keepdims=False)+minAng
    
        ind = acosxdirDiff>0
        
        if not np.any(ind):
            break

        dirIn[:,ind] -= coefFac*x[:,ind]*acosxdirDiff[ind]

        dirIn[:,ind] /= (cNorm(dirIn[:,ind], kd=True)+epsFloat)

        coefFac *= 1.5
    
    return dirIn
    

###################################################################################################################################################################################
class convergingDirections:
    def __init__(self, x0:np.ndarray, vp:np.ndarray, alpha:float, beta:float):
        """
        
        :param x0: Target point of the Directions
        :param vp: principal direction
        :param alpha: zero convergence offset
        :param beta: convergence rate
        """
        
        vp = vp/(cNorm(vp)+epsFloat)
        
        self._x0 = x0.copy()
        
        self._vp = vp.copy()
        self._Vnull = nullspace(self._vp.T)[1]
        self._Pnull = self._Vnull.T.copy()
        self._R = np.vstack((self._vp.T, self._Pnull))
        
        self._alpha = alpha
        self._beta = beta
    
    @property
    def dim(self):
        return self._x0.size
    
    def __add__(self, other):
        assert isinstance(other, convergingDirections), 'Only defined for same type'
        return convergingDirections(self._x0+other._x0, self._vp+other._vp, self._alpha+other._alpha, self._beta+other._beta)
    
    def __mul__(self, other):
        assert isinstance(other, (float, int)), 'Can only be multiplied by scalar'
        other = float(other)
        new = convergingDirections(other*self._x0, self._vp, other*self._alpha, other*self._beta)
        #trickery
        new._vp *= other
        return new
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def toStringList(self):
        totStringList = []
        
        totStringList.append(int2Str(self._x0.size))
        totStringList += vec2List(self._x0.squeeze())
        totStringList += vec2List(self._vp.squeeze())
        totStringList.append(double2Str(self._alpha))
        totStringList.append(double2Str(self._beta))
        
        return totStringList
    
    def getDir(self, xIn:np.ndarray, cpy:bool=True):
        """
        
        :param xIn:
        :param cpy:
        :return:
        """
        
        out = np.zeros_like(xIn)
        out += self._vp
        
        if cpy:
            x = xIn-self._x0
        else:
            x=xIn
            x-=self._x0
        
        #Get orthorgonal projections
        h = np.dot(self._Pnull, x)
        
        #Establish non-convergent zone
        if self._alpha > 0.:
            hsign = np.sign(h)
            habs = np.abs(h)
            
            habs -= self._alpha
            habs = np.maximum(habs, 0.)
            
            h = habs*hsign
        
        # Multiply with factor
        h *= self._beta
        
        #Add up the orthogonal convergence
        for k in range(self._Vnull.shape[1]):
            out += self._Vnull[:,[k]]*h[[k],:]
        #einsum?
        
        return out

###################################################################################################################################################################################
class locallyWeightedDirections:
    def __init__(self, thisProb, thisDirections, weight=1.):
        self._prob = thisProb
        self._dir = thisDirections
        self._weight = weight
    
    def __add__(self, other):
        assert isinstance(other, locallyWeightedDirections)
        return locallyWeightedDirections(self._prob+other._prob, self._dir+other._dir, self._weight+other._weight)
    
    def __mul__(self, other):
        assert isinstance(other, (float, int))
        other = float(other)
        return locallyWeightedDirections(other*self._prob, other*self._dir, other*self._weight)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    @property
    def dim(self):
        return self._prob.dim
    
    def toStringList(self):
        totStringList = []
        # parameters for this locallyWeightedDirections
        totStringList.append(int2Str(self._dir.dim))
        totStringList.append(double2Str(self._weight))
        
        # parameters for the sub-objects
        totStringList += self._dir.toStringList()
        totStringList += self._prob.toStringList()
        
        return totStringList
    
    def getDir(self, xIn):
        return (self._weight*self._prob.getWeights(xIn, cpy=True, kd=True))*self._dir.getDir(xIn, cpy=True)

###################################################################################################################################################################################
class combinedLocallyWeightedDirections:
    def __init__(self, baseConv = 0., ensureConv=None):
        self._dirList = [] # type: List[locallyWeightedDirections]
        self._baseConv = baseConv
        self._ensureConv = ensureConv if (ensureConv is not None) else lambda x,v: v
    
    def __add__(self, other):
        assert isinstance(other, combinedLocallyWeightedDirections)
        assert len(self._dirList)==len(other._dirList)
        
        new = combinedLocallyWeightedDirections(self._baseConv+other._baseConv, self._ensureConv)
        new._dirList = [selfDir+otherDir for selfDir,otherDir in zip(self._dirList, other._dirList)]
        return new
    
    def __mul__(self, other):
        assert isinstance(other, (int, float))
        
        new = combinedLocallyWeightedDirections(other*self._baseConv, self._ensureConv)
        new._dirList = [other*aDir for aDir in self._dirList]
        return new
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
        
    
    @property
    def dim(self):
        return self._dirList[0].dim if self._dirList else None
    @property
    def nK(self):
        return len(self._dirList)
    
    def toStringList(self):
        
        totStringList = []

        #Excpected string style
        #dim
        #baseConv
        #nComponents
        #[for each localDir]
        #nParsLocalLines
        
        totStringList.append(int2Str(self.dim))
        totStringList.append(double2Str(self._baseConv))
        totStringList.append(int2Str(self.nK))
        
        for aDir in self._dirList:
            totStringList += aDir.toStringList()
        
        return totStringList
    
    def toText(self, fileName:str):
        with open(fileName, 'w+') as file:
            file.writelines(self.toStringList())
        return True
    
    def addDyn(self, newDyn:locallyWeightedDirections):
        self._dirList.append(newDyn)
    
    def getDir(self, xIn):
        
        xnorm = cNorm(xIn, kd=True)
        
        out = (self._baseConv/(xnorm+epsFloat))*xIn
        
        for adir in self._dirList:
            out += adir.getDir(xIn)
        
        # Scale -> Normalize vectors
        out /= (cNorm(out, kd=True)+epsFloat)
        
        out = self._ensureConv(xIn, out)
        
        out *= np.minimum(1e4*xnorm,1.)
        
        return out
    
    
    def getDirTraj(self, x0:np.ndarray, t=None, stopCond=None):
        
        if ((t is None) and (stopCond is None)):
            stopCond = lambda x: cNormSquare(x,kd=False, cpy=True)<(1e-2)**2
        
        assert ((t is None) != (stopCond is None))
        
        from scipy.integrate import odeint
        
        
        
        if t is not None:
            
            xshape = x0.shape
            fInt = lambda x, t:self.getDir(x.reshape(xshape)).ravel()
            
            if isinstance(t, (float, int)):
                t = np.linspace(0., t, 100)
            xout = odeint(fInt, x0.ravel(), t, rtol=1e-4, atol=1e-6)
            
            xout = xout.T.reshape((xshape[0],xshape[1]*t.size))
            
            return xout
        else:
    
            xshape = [x0.shape[0], 1]
            fInt = lambda x,t:self.getDir(x.reshape(xshape)).ravel()
            
            tStep = np.min(cNorm(x0, kd=False))/3.
            
            thisTStep = np.linspace(0., tStep, 100)
            
            allSol = []
            
            for k in range(x0.shape[1]):
                print('main {0}'.format(k))
                xi = x0[:,[k]].copy()
                thisX = []
                thisT = []
                firstRun = True
                while not stopCond(xi):
                    print('sub with \n {0}'.format(xi))
                    newX = odeint(fInt, xi.ravel(), thisTStep, rtol=1e-4, atol=1e-6)
                    if firstRun:
                        thisT.append(thisTStep)
                        thisX.append( newX.T.reshape((xshape[0],-1)) )
                        firstRun = False
                    else:
                        thisT.append(thisTStep[1:])
                        thisX.append((newX.T)[:,1:].reshape((xshape[0],-1)))
                    
                    xi = thisX[-1][:,[-1]].copy()
                
                ind = np.argmax(stopCond(thisX[-1]))
                thisX[-1] = thisX[-1][:,:ind]
                thisT[-1] = thisT[-1][:ind]
                    
                allSol.append( [np.cumsum(np.hstack(thisT)), np.hstack(thisX)] )
            
            return allSol
            
            
class combinedDiffeoCtrl:
    def __init__(self, dirDyn:combinedLocallyWeightedDirections, magnitudeModel:GaussianMixtureModel, diffeo:diffeomorphism, baseDir:bool=False, demSpaceOffset:np.ndarray=None, overallScaling:np.ndarray=None):
        
        self._dirDyn = dirDyn
        self._magModel = magnitudeModel
        self._diffeo = diffeo
        self._baseDir = baseDir
        self._demSpaceOffset = demSpaceOffset
        self._overallScaling = overallScaling
        self._overallInverseScaling = 1./overallScaling if (overallScaling is not None) else None
        
        self._minMagnitude = 0.001
        
        self._dimInt=None
    
    @property
    def dim(self):
        return self._dirDyn.dim
    @property
    def dimTot(self):
        return self._dirDyn.dim
    @property
    def dimInt(self):
        return self._dimInt
    
    def toStringList(self):
        totStringList = []

        # dimTot
        # dimInt
        # direction
        # scaling
        # demspace offset
        
        totStringList.append(int2Str(self.dim))
        totStringList.append(int2Str(self.dimInt))
        totStringList.append(int2Str(self._baseDir))
        totStringList += vec2List(self._overallScaling if (self._overallScaling is not None) else np.ones((self.dim,1)))
        totStringList += vec2List(self._demSpaceOffset if (self._demSpaceOffset is not None) else np.ones((self.dim,1)))
        
        return totStringList
    
    def toText(self, fileName:str):
        with open(fileName, 'w+') as file:
            file.writelines(self.toStringList())
        return True

    @property
    def overallScaling(self):
        return deepcopy(self._overallScaling)

    @overallScaling.setter
    def overallScaling(self,newScaling: np.ndarray):
        try:
            newScaling = np.array(newScaling).reshape((self._diffeo._dim,1)) if (newScaling is not None) else None
        except:
            assert 0,'new scaling could not be transformed into column-vector'
    
        self._overallScaling = newScaling
        self._overallInverseScaling = 1./newScaling if (newScaling is not None) else None
        return 0

    def forwardTransform(self,x):
        # dem 2 ctrl
        if self._demSpaceOffset is not None:
            x -= self._demSpaceOffset
        if self._overallScaling is not None:
            x *= self._overallScaling
        if self._baseDir:
            x = self._diffeo.forwardTransform(x)
        else:
            x = self._diffeo.inverseTransform(x)
    
        return x

    def forwardTransformV(self,x,v):
        # dem 2 ctrl
        if self._demSpaceOffset is not None:
            x -= self._demSpaceOffset
        if self._overallScaling is not None:
            x *= self._overallScaling
            v *= self._overallScaling
        if self._baseDir:
            x,v = self._diffeo.forwardTransformJac(x,v=v)
        else:
            x,v = self._diffeo.inverseTransformJac(x,vp=v)
    
        return x,v

    def forwardTransformV(self,x):
        """Attention, the jacobian does not take into account the scaling!
           Attention, the jacobian is such that it always takes vCtrl to vDem, so
           vDem / dirDem= J * vCtrl / dirCtrl"""
        # dem 2 ctrl
        
        if self._baseDir:
            x,J = self._diffeo.forwardTransformJac(x,Jac=True,outInvJac=False)
        else:
            x,J = self._diffeo.inverseTransformJac(x,Jacp=True,outInvJac=False)
    
        return x,J

    def inverseTransform(self,x):
        # ctrl 2 dem
        if not self._baseDir:
            x = self._diffeo.forwardTransform(x)
        else:
            x = self._diffeo.inverseTransform(x)
        if self._overallInverseScaling is not None:
            x *= self._overallInverseScaling
        if self._demSpaceOffset is not None:
            x += self._demSpaceOffset
        return x

    def inverseTransformV(self,x,v):
        # ctrl 2 dem
        if not self._baseDir:
            x,v = self._diffeo.forwardTransformJac(x,v=v)
        else:
            x,v = self._diffeo.inverseTransformJac(x,vp=v)
        if self._overallInverseScaling is not None:
            x *= self._overallInverseScaling
            v *= self._overallInverseScaling
        if self._demSpaceOffset is not None:
            x += self._demSpaceOffset
        return x,v

    def inverseTransformJ(self,x):
        """Attention, the jacobian does not take into account the scaling!
           Attention, the jacobian is such that it always takes vCtrl to vDem, so
           vDem / dirDem= J * vCtrl / dirCtrl"""
        # ctrl 2 dem
        
        if not self._baseDir:
            x,J = self._diffeo.forwardTransformJac(x,Jac=True,outInvJac=True)
        else:
            x,J = self._diffeo.inverseTransformJac(x,Jacp=True,outInvJac=True)
        
        return x,J
    
    def getCtrlSpaceVelocity(self,x,*args, **kwargs):
        """
        # Attention this function only returns the directions not the actual velocity
        :param x: control space points to be evaluated
        :param args:
        :param kwargs:
        :return:
        """
        return self._dirDyn.getDir(x) #Direction
    
    def getDemSpaceVelocity(self, x:np.ndarray)->np.ndarray:
        
        """
        Return the velocity (with magnitude) associated to the demonstration space points
        :param x: demonstration space points
        :return:
        """
        
        xp = self.forwardTransform(x.copy())
        #Get direction
        vp = self.getCtrlSpaceVelocity(xp)
        #Retransform to demonstration space
        _, vpp = self.inverseTransformV(xp,vp)
        #Renormalize
        vpp /= (cNorm(vpp, kd=True)+epsFloat)
        #get magnitude -> assure positivity
        m = np.minimum( np.maximum(self._magModel.evalMap(x), self._minMagnitude), cNorm(xp,kd=True)*1e3) #Limit velocity when close to control space origin
        m.resize((1,x.shape[1]))
        
        return vpp*m

    def getTrajectory(self,xIn: np.ndarray,t: Union[np.ndarray,float],inInDem: bool = True,outInDem: bool = True,
                      outInCtrl: bool = False,returnVel: bool = False):
        
        """
        
        :param xIn:
        :param t:
        :param inInDem:
        :param outInDem:
        :param outInCtrl:
        :param returnVel:
        :return:
        """
        from scipy.integrate import odeint

        xIn = xIn.copy()

        dim,nPt = xIn.shape

        if isinstance(t,(float,int)):
            t = np.arange(0.,t,0.01)
        
        if not inInDem:
            xIn = self.inverseTransform(xIn)
        
        fInt = lambda x,t: self.getDemSpaceVelocity(x.reshape((dim,nPt))).ravel()
        
        #integrate Attention integration is really performed in the demonstration space
        Xd = odeint(fInt,xIn.ravel(),t,rtol=1e-4,atol=1e-6)

        Xd = Xd.T.reshape((dim,nPt*t.size))  # Format [x_0_t0,x_0_t1,...,x_0_tN,x_1_t0,x_1_t1,...,x_1_t

        out = []

        if outInDem:
            out.append(Xd)
            if returnVel:
                out.append(self.getDemSpaceVelocity(Xd))

        if outInCtrl:
            Xc = self.forwardTransform(Xd)
            out.append(Xc)
            if returnVel:
                out.append(self.getCtrlSpaceVelocity(Xc)) #Attention this is actually the direction not the true velocity

        out.append([t.size,dim,nPt])
        return out

###################################################################################################################################################################################
def pointListToCombinedDirections(points:np.ndarray, parsWeights:dict={}, parsDyn:dict={}, fullOut:bool=False )->combinedLocallyWeightedDirections:
    """
    
    :param points: List of points stored as matrix from points[:,[0]] to points[:,[-1]]
    :param parsWeights:
    :param parsDyn:
    :return:
    """
    #Chose
    _kernel = locallyConvDirecKernel_
    if _kernel==0:
        from modifiedEM import gaussianKernel as kernel
    elif _kernel==1:
        from distribution import cauchyKernel as kernel
    
    _parsDyn = {'alpha':0., 'beta':-1., 'baseConv':-1.e-3}
    _parsWeights = {'centerPoint':0.5, 'maxEndInf':0.35, 'orthCovScale':-0.2, 'doCond':False
                    ,'pi': 1., 'gamma':1.}#Cauchy stuff

    for aName in ['alpha', 'beta']:
        for aExt in ['0','F']:
            _parsDyn.setdefault(aName+aExt, _parsDyn[aName])
    for aName in ['centerPoint', 'maxEndInf', 'orthCovScale']:
        for aExt in ['0','F']:
            _parsWeights.setdefault(aName+aExt, _parsWeights[aName])

    _parsDyn.update(parsDyn)
    _parsWeights.update(parsWeights)

    thisCombinedDyn = combinedLocallyWeightedDirections(_parsDyn['baseConv'])
    if fullOut:
        asSimpleList = []
    
    #Loop
    for i in range(0,points.shape[1]-1):
        # Get the Directions
        if i==0:
            alpha = _parsDyn['alpha0']
            beta = _parsDyn['beta0']
            orthScale = _parsWeights['orthCovScale0']
            endInf = _parsWeights['maxEndInf0']
            cPoint = _parsWeights['centerPoint0']
            
        if i == points.shape[1]-2:
            alpha = _parsDyn['alphaF']
            beta = _parsDyn['betaF']
            orthScale = _parsWeights['orthCovScaleF']
            endInf = _parsWeights['maxEndInfF']
            cPoint = _parsWeights['centerPointF']
        else:
            alpha = _parsDyn['alpha']
            beta = _parsDyn['beta']
            orthScale = _parsWeights['orthCovScale']
            endInf = _parsWeights['maxEndInf']
            cPoint = _parsWeights['centerPoint']

        # Get the weighting
        xCenter = points[:, [i]] * (1. - cPoint) + points[:, [i + 1]] * cPoint
        dx = min(float(cNorm(points[:, [i]] - xCenter, kd=False)), float(cNorm(points[:, [i + 1]] - xCenter, kd=False)))
        if _kernel==0:
            #gaussianKernel
            S00 = 1./(np.log(endInf)/(-.5*dx**2))
        elif _kernel == 1:
            #modififed multivariate cauchy
            S00 = dx**2/(_parsWeights['gamma']/(endInf*_parsWeights['pi']) - _parsWeights['gamma']**2.)
            assert S00>0., "Parameters for cauchy inconsistent"
        convScaleDX = orthScale if (orthScale > 0.) else -dx * orthScale

        thisDyn = convergingDirections(points[:,[i+1]].copy(),
                                       points[:,[i+1]]-points[:,[i]],
                                       alpha=alpha if (alpha>=0.) else (-alpha*cNorm(points[:,[i+1]]-points[:,[i]],kd=False)),
                                       beta=beta if (beta<=0.) else -beta/convScaleDX)
        
        if orthScale>0.:
            S11 = orthScale
        else:
            S11 = -orthScale*S00
        
        S = np.identity(points.shape[0])*S11
        S[0,0] = S00
        
        S = ndot(thisDyn._R.T, S, thisDyn._R)

        if _kernel==0:
            thisProb = kernel(1, Sigma=S, mu=xCenter, doCond=_parsWeights['doCond'])
        elif _kernel==1:
            thisProb = kernel(None, Sigma=S, mu=xCenter, doCond=_parsWeights['doCond'],gamma=_parsWeights['gamma'],pi=_parsWeights['pi'])

        #Assemble the local Directions and append
        thisCombinedDyn.addDyn(locallyWeightedDirections(thisProb, thisDyn))
        if fullOut:
            asSimpleList.append(locallyWeightedDirections(thisProb, thisDyn))
    
    if fullOut:
        return thisCombinedDyn, asSimpleList
    else:
        return thisCombinedDyn
        

def getMagnitudeModel(x:np.ndarray,v:np.ndarray,nKMax:int=12, opts={}):
    
    from greedyInsertion import greedyEMCPU
    
    _opts = {'add2Sigma':1e-3,
            'iterMax':100,
            'relTol':1e-2, 'absTol':1.e-2,
            'interOpt':True, 'reOpt':False, 'convFac':1.,
            'nPartial':10
            }
    
    _opts.update(opts)
    
    if x.shape == v.shape:
        xtilde = np.vstack((x, cNorm(v, kd=True)))
    else:
        # v already contains magnitude
        xtilde = np.vstack((x,v.reshape((1,-1))))

    #greedyEMCPU(x,nVarI=None,nKMax=5,nPartial=10,add2Sigma=1.e-3,iterMax=100,relTol=1e-2,absTol=1e-2,
    #            doPlot=False,speedUp=False,interOpt=True,reOpt=None,convFac=100.,warmStartGMM=None,otherOpts={}):
    magnitudeModel = greedyEMCPU(xtilde, xtilde.shape[0]-1, nKMax=nKMax, nPartial=_opts['nPartial'],add2Sigma=_opts['add2Sigma'],iterMax=_opts['iterMax'],relTol=_opts['relTol'],absTol=_opts['absTol'],
                                    doPlot=False,speedUp=False,interOpt=_opts['interOpt'],reOpt=_opts['reOpt'],convFac=_opts['convFac'],warmStartGMM=None)
    
    return magnitudeModel
    
    

if __name__ == '__main__':
    
    import plotUtils as pu
    
    N = 300
    
    points = np.array([[1.,0.],[1.,1.]])
    parsDyn = {'alpha':0., 'beta':-1., 'baseConv':-1.e-3}
    parsWeights = {'centerPoint':0.5, 'maxEndInf':0.35, 'orthCovScale':-0.2, 'doCond':False}
    thisDyn = pointListToCombinedDirections(points, parsDyn, parsWeights)
    
    xx,yy = np.meshgrid(np.linspace(-2,2,N), np.linspace(-2,2,N))
    X = np.vstack((xx.flatten(), yy.flatten()))
    
    V = thisDyn.getDir(X)
    
    V = minimallyConvergingDirection(X, V, minAng=5./180.*np.pi)
    
    vx = V[0,:].reshape((N,N))
    vy = V[1,:].reshape((N,N))
    
    ff,aa = pu.plt.subplots(1,1)
    
    aa.plot(points[0,:], points[1,:], '--.')
    aa.streamplot(xx,yy,vx,vy)
    aa.axis('equal')

    points = np.array([[0.,0.,1.,2.,1.,1.,0.],[4.,2.,3.,3.,2.,1.,0.]])
    parsDyn = {'alpha':0., 'beta':-2., 'baseConv':-1.e-3}
    parsWeights = {'centerPoint':0.5, 'maxEndInf':0.35, 'orthCovScale':-0.1, 'doCond':False}
    thisDyn = pointListToCombinedDirections(points, parsDyn, parsWeights)

    xx,yy = np.meshgrid(np.linspace(-1,3,N),np.linspace(-1,5,N))
    X = np.vstack((xx.flatten(),yy.flatten()))

    V = thisDyn.getDir(X)

    V = minimallyConvergingDirection(X,V,minAng=5./180.*np.pi)

    vx = V[0,:].reshape((N,N))
    vy = V[1,:].reshape((N,N))

    ff,aa = pu.plt.subplots(1,1)

    aa.plot(points[0,:],points[1,:],'--.')
    aa.streamplot(xx,yy,vx,vy)
    aa.axis('equal')

    points = np.array([[-1., 1., -1., 1., -1., 1., 0.], [6., 5., 4., 3., 2., 1., 0.]])
    parsDyn = {'alpha': 0., 'beta': -5., 'baseConv': -1.e-3}
    parsWeights = {'centerPoint': 0.5, 'maxEndInf': 0.1, 'orthCovScale': -0.2, 'doCond': False}
    thisDyn = pointListToCombinedDirections(points, parsDyn, parsWeights)

    xx, yy = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-1, 7, N))
    X = np.vstack((xx.flatten(), yy.flatten()))

    V = thisDyn.getDir(X)

    V = minimallyConvergingDirection(X, V, minAng=5. / 180. * np.pi)

    vx = V[0, :].reshape((N, N))
    vy = V[1, :].reshape((N, N))

    ff, aa = pu.plt.subplots(1, 1)

    aa.plot(points[0, :], points[1, :], '--.')
    aa.streamplot(xx, yy, vx, vy)
    aa.axis('equal')

    points = np.array([[-1., 1., -1., 1., -1., 1., 0.], [6., 5., 4., 3., 2., 1., 0.]])
    parsDyn = {'alpha': .2, 'beta': -5., 'baseConv': -1.e-3}
    parsWeights = {'centerPoint': 0.5, 'maxEndInf': 0.1, 'orthCovScale': -0.2, 'doCond': False}
    thisDyn = pointListToCombinedDirections(points, parsDyn, parsWeights)

    xx, yy = np.meshgrid(np.linspace(-2, 2, N), np.linspace(-1, 7, N))
    X = np.vstack((xx.flatten(), yy.flatten()))

    V = thisDyn.getDir(X)

    V = minimallyConvergingDirection(X, V, minAng=5./180.*np.pi)

    vx = V[0, :].reshape((N, N))
    vy = V[1, :].reshape((N, N))

    ff, aa = pu.plt.subplots(1, 1)

    aa.plot(points[0, :], points[1, :], '--.')
    aa.streamplot(xx, yy, vx, vy)
    aa.axis('equal')

    points = np.array([[-1.,1.,-1.,1.,-1.,1.,0.],[6.,5.,4.,3.,2.,1.,0.],[6.,5.5,4.,3.5,2.,1.,0.]])

    parsDyn = {'alpha':.2,'beta':-5.,'baseConv':-1.e-3}
    parsWeights = {'centerPoint':0.5,'maxEndInf':0.1,'orthCovScale':-0.2,'doCond':False}
    thisDyn = pointListToCombinedDirections(points,parsDyn,parsWeights)
    thisDyn._ensureConv = lambda x,v: minimallyConvergingDirection(x, v, minAng=5./180.*np.pi)
    
    Xin = np.vstack(( (np.random.rand(1,10)-.5)*3, np.random.rand(1,10)*6+4., np.random.rand(1,10)*6+4. ))
    
    ff = pu.plt.figure()
    aa = ff.add_subplot(111, projection='3d')
    
    aa.plot(points[0,:], points[1,:], points[2,:], '.-r')
    
    for kk in range(Xin.shape[1]):
        XXX = thisDyn.getDirTraj(Xin[:,[kk]])[0][1]
        aa.plot(XXX[0,:], XXX[1,:], XXX[2,:], 'b')
    

    pu.plt.show()
    
    
        
    
    