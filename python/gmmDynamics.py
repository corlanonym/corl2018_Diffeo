#Forward declare
from copy import deepcopy


class gmmDiffeoCtrl:
    pass

from dynamics import *
import multiprocessingTools as mt
from scipy.integrate import odeint
import greedyInsertion as gi
import modifiedEM as mEM
import diffeoUtils as du

from typing import Union,List,Callable

from otherUtils import learningMove

class gmmDiffeoCtrl:
    def __init__(self, aDiffeo, gmmDyn:mEM.GaussianMixtureModel, baseDir:bool = False,breakRadius:float = 0.1,radialMinConv:Union[float,Callable,learningMove] = -0.001, contractionFac:float = None, contractionZone:float=.6827):
        self._diffeo = aDiffeo
        self._diffeoDyn = None # type: du.diffeomorphism
        self._dyn = gmmDyn
        self._Id = Idn(1,aDiffeo._dim)
        self._Id.setflags(write=False)
        self.baseDir = baseDir  # Transformation direction from demonstration space to control space. True->Forward , False:Inverse
        self.breakRadius = breakRadius
        self.radialMinConv = radialMinConv
        self._overallScaling = None
        self._overallInverseScaling = None
        self._baseVelocity = 0.
        self._contractionFac = contractionFac
        self._contractionZone = contractionZone
        self._allDims = [aDiffeo._dim, 1]

        self._demSpaceOffset = None

    def toText(self, fileName, double2String = double2Str):

        with open(fileName, 'w+') as file:
            file.write("{0:d}\n".format(self._allDims[0]))
            file.write("{0:d}\n".format(self._allDims[1]))
            file.write("{0:d}\n".format(self.baseDir));
            file.writelines( vec2List(self._overallScaling.squeeze()) )
            file.writelines(vec2List(self._demSpaceOffset.squeeze()))

    def convergenceToText(self, fileName, double2String = double2Str):

        with open(fileName,'w+') as file:
            file.write( "{0:d}\n".format(self.breakRadius.size+1 ) )
            file.write( "1\n" )
            file.write( double2String(self.radialMinConv._radianMinConv) )
            file.writelines( vec2List(self.breakRadius.squeeze()) )

    @property
    def overallScaling(self):
        return deepcopy(self._overallScaling)
    
    @overallScaling.setter
    def overallScaling(self,newScaling: np.ndarray):
        try:
            newScaling = np.array(newScaling).reshape((self._diffeo._dim,1))
        except:
            assert 0, 'new scaling could not be transformed into column-vector'
        
        self._overallScaling = newScaling
        self._overallInverseScaling = 1./newScaling
        return 0
    
    def forwardTransform(self,x):
        # dem 2 ctrl
        if self._demSpaceOffset is not None:
            x-=self._demSpaceOffset
        if self._overallScaling is not None:
            x*=self._overallScaling
        if self.baseDir:
            x = self._diffeo.forwardTransform(x)
        else:
            x = self._diffeo.inverseTransform(x)

        return x
    
    def forwardTransformV(self,x,v):
        # dem 2 ctrl
        if self._demSpaceOffset is not None:
            x-=self._demSpaceOffset
        if self._overallScaling is not None:
            x*=self._overallScaling
            v*= self._overallScaling
        if self.baseDir:
            x,v = self._diffeo.forwardTransformJac(x, v=v)
        else:
            x,v= self._diffeo.inverseTransformJac(x,vp=v)

        return x,v

    def inverseTransform(self,x):
        # ctrl 2 dem
        if not self.baseDir:
            x = self._diffeo.forwardTransform(x)
        else:
            x = self._diffeo.inverseTransform(x)
        if self._overallInverseScaling is not None:
            x*=self._overallInverseScaling
        if self._demSpaceOffset is not None:
            x+=self._demSpaceOffset
        return x
    
    def inverseTransformV(self,x,v):
        # ctrl 2 dem
        if not self.baseDir:
            x,v = self._diffeo.forwardTransformJac(x,v=v)
        else:
            x,v = self._diffeo.inverseTransformJac(x,vp=v)
        if self._overallInverseScaling is not None:
            x*=self._overallInverseScaling
            v*= self._overallInverseScaling
        if self._demSpaceOffset is not None:
            x+=self._demSpaceOffset
        return x,v
    
    def getCtrlSpaceRadialVelocity(self,x: np.ndarray) -> np.ndarray:
        
        v = self.getCtrlSpaceVelocity(x)

        xnorm = x/(cNorm(x)+epsFloat)
        vrad = sum(xnorm*v,axis=0,keepdims=True)
        
        return vrad
    
    def getCtrlSpaceTangentialVelocity(self,x):
        
        v = self.getCtrlSpaceVelocity(x)
        
        xnorm = x/(cNorm(x)+epsFloat)
        vrad = sum(xnorm*v,axis=0,keepdims=True)
        vtang = v-vrad*xnorm
        
        return vtang

    def getCtrlSpaceVelocity(self,x, fullOut=False):
        # Assuming that self.radialMonConv is a learningMove

        if self._diffeoDyn is None:
            xEval = x
        else:
            xEval = self._diffeoDyn.forwardTransform(x)

        if self._contractionFac is None:
            if (x.shape[1] == 1) and isinstance(x,np.ndarray):
                v = self._dyn._evalMapCPU(xEval)
            else:
                v = self._dyn.evalMap(xEval)
        else:
            if (x.shape[1] == 1) and isinstance(x,np.ndarray):
                v, weights = self._dyn._evalMapCPU(xEval, returnWeights=True)
            else:
                v, weights = self._dyn.evalMap(xEval, returnWeights=True)

        
        xnSquared = np.sum(np.square(x), axis=0, keepdims=True)
        xn = np.sqrt(xnSquared)
        xnormed = x/(xn+epsFloat)
        
        #If close to the center, then xd = -x
        if isinstance(self.breakRadius, float):
            ind = xn < self.breakRadius
            ind.resize((x.shape[1],))
        
            vni = np.maximum(cNorm(v[:,ind],kd=True), self._baseVelocity)
            v[:,ind] = -xnormed[:,ind]*vni*xn[[0],ind]/self.breakRadius
        else:
            breakDist = np.sum(np.square(np.multiply(x, 1./self.breakRadius)),axis=0, keepdims=True)
            facBreak = np.exp(-breakDist)
            vn = np.maximum(cNorm(v,kd=True),self._baseVelocity)
            v = (1.-facBreak)*v - facBreak*vn*xnormed*minimum(breakDist*5.,1.)**.5

        # Add perpendicular contraction
        if not (self._contractionFac is None):
            assert 0
            nind = np.logical_not(ind)
            wo = weights[:,nind]
            xo = x[:, nind]
            vo = v[:,nind]
            vNormO = self._contractionFac*cNorm(vo)/(1.-self._contractionZone)
            if vo.shape[1]:
                # Loop over all kernels and contract
                for k, aGaus in self._dyn.enum():
                    # Compute direction to supposed orthogonal direction
                    # project point onto mux + lambda*muy
                    muvn = aGaus.muy
                    muvn /= cNorm(muvn)

                    dxOrth = aGaus.mux + muvn*(mDotProd(muvn, xo-aGaus.mux))-xo

                    thisCond = aGaus.doCond
                    aGaus.doCond = False
                    thisW = aGaus._getWeightsCPU(aGaus.mux+dxOrth)
                    thisW.resize((1,xo.shape[1]))
                    aGaus._doCond = thisCond

                    # This is not diff-bar -> change if nice
                    #thisW = (thisW<self._contractionZone).astype(np.float_)
                    thisW[thisW > self._contractionZone] = 0.

                    # Normalize and scale
                    dirOrth = np.multiply(dxOrth, np.multiply(np.divide(vNormO,cNorm(dxOrth)), thisW))
                    vo -= np.multiply(wo[[k],:], dirOrth) #Minus because contractionFac is already positive
            v[:, np.logical_not(ind)] = vo

        #Assure convergence via the learnedMove object
        #ind = np.logical_not(ind)
        if fullOut:
            v,fDict = self.radialMinConv.ensureMinConv(x,v,xnSquared,fullOut=fullOut)#v[:,ind],fDict = self.radialMinConv.ensureMinConv(x[:,ind],v[:,ind],xnSquared[:,ind],fullOut=fullOut)
            #ind1 = ind.copy()
            #ind1[ind] = fDict['ind']
            ind1 = fDict['ind']
        else:
            v = self.radialMinConv.ensureMinConv(x,v,xnSquared)#v[:,ind] = self.radialMinConv.ensureMinConv(x[:,ind],v[:,ind],xnSquared[:,ind])
        
        if fullOut:
            return v, {"ind0":ind, "ind1":ind1}
        else:
            return v
    
    def getCtrlSpaceVelocityOld(self,x):
        assert 0
        if (x.shape[1] == 1) and isinstance(x,np.ndarray):
            v = self._dyn._evalMapCPU(x)
        else:
            v = self._dyn.evalMap(x)
        
        # Now v should be in the tangent space; Add radial convergence
        xn = cNorm(x,kd=True)
        
        ind = (xn < self.breakRadius)[0,:]
        
        xni = np.ones_like(xn)

        xni[[0],ind] = xn[[0],ind]/self.breakRadius
        
        v *= xni
        
        #Ensure convergence
        #attention singularity
        if isinstance(self.radialMinConv, float):
            ind = xn[0,:]<100.*epsFloat
            xn[[0],ind] = 1.
            xnormed = x/xn

            #Check radial velocity
            vrad = sum(xnormed*v,axis=0,keepdims=True)

            ind2 = (vrad > self.radialMinConv*xn)[0,:]

            # Treat actual velocity
            v[:, ind2] += (-vrad[[0],ind2]+self.radialMinConv*xn[[0],ind2])*xnormed[:,ind2]

        elif isinstance(self.radialMinConv, learningMove):
            minConv = self.radialMinConv.getMinConv(x)
            minConv = minConv.reshape((1,x.shape[1]))

            ind = xn[0,:] < 100.*epsFloat
            xn[[0],ind] = 1.
            xnormed = x/xn

            # Check radial velocity
            vrad = sum(xnormed*v,axis=0,keepdims=True)

            ind2 = (vrad > minConv*xn)[0,:]

            # Treat actual velocity
            try:
                v[:,ind2] += (-vrad[[0],ind2]+minConv[[0],ind2]*xn[[0],ind2])*xnormed[:,ind2]
            except:
                print("??")

        elif callable(self.radialMinConv):
            convCorrFac = self.radialMinConv(x, xn, v)
            v += convCorrFac*x
        else:
            assert 0, "Convergence needs to be callable or float"
        
        #Set to zero if very close
        v[:,ind] = 0.
        
        return v
    
    def getDemSpaceVelocity(self,x,fullOut=False,cpy=True):
        dim,nPt = x.shape
        
        x = x.copy() if cpy else x
        
        # Do the initial scaling
        x-=self._demSpaceOffset
        if self._overallScaling is not None:
            x *= self._overallScaling

        if x.shape[1] == 1:
            # Transform into cartesian control space
            if self.baseDir:
                xprime,Jac = self._diffeo.forwardTransformJac(x,Jacp=self._Id)
                # Inverse the jacobian
                Jac = inv(Jac[0,:,:],overwrite_a=True,check_finite=False)
            else:
                # This function already returns the jacobian of the inverse transformation
                xprime,Jac = self._diffeo.inverseTransformJac(x,Jacp=self._Id,outInvJac=True, whichSide='right')
                Jac = Jac[0,:,:]
            # Get control space velocity
            if fullOut:
                vprime, isConvDict = self.getCtrlSpaceVelocity(xprime, fullOut=fullOut)
            else:
                vprime = self.getCtrlSpaceVelocity(xprime, fullOut=fullOut)
            # Invert the jacobian and compute the velocity in the demonstration space
            v = dot(Jac,vprime)
        else:
            # Transform into cartesian control space
            JacRA = mt.sctypes.RawArray(mt.c.c_double,nPt*dim*dim)
            Jac = np.frombuffer(JacRA)
            Jac.resize((x.shape[1],x.shape[0],x.shape[0]))
            if self.baseDir:
                xprime,_ = self._diffeo.forwardTransformJac(x,Jac=IdnR(nPt,dim),outJac=Jac)
                # Inverse the jacobian
                mt.slicedInversion(JacRA,cpy=False,NnDim=[x.shape[1],x.shape[0]],returnNp=False)
            else:
                # This function already returns the jacobian of the inverse transformation
                xprime,_ = self._diffeo.inverseTransformJac(x,Jacp=IdnR(nPt,dim),outJac=Jac,outInvJac=True, whichSide='right')
            # Get control space velocity
            vprime = self.getCtrlSpaceVelocity(xprime)
            # Transform velocity
            v = np.einsum('ijk,ki->ji',Jac,vprime)  # Jac holds the inverse since values are overridden
        # Reverse the initial scaling
        if self._overallInverseScaling is not None:
            v *= self._overallInverseScaling
            
        if fullOut:
            return v, isConvDict
        else:
            return v
    
    def getTrajectory(self,xIn: np.ndarray,t:Union[np.ndarray,float],inInDem: bool = True,outInDem: bool = True,
                      outInCtrl: bool = False,returnVel: bool = False):
        
        xIn = xIn.copy()
        
        dim,nPt = xIn.shape
        
        if isinstance(t,(float,int)):
            t = np.arange(0.,t,0.01)
        
        # if xIn is in demo-space, transform
        if inInDem:
            # Do the initial scaling
            if self._demSpaceOffset is not None:
                xIn-=self._demSpaceOffset
            if self._overallScaling is not None:
                xIn *= self._overallScaling
            if self.baseDir:
                xInPrime = self._diffeo.forwardTransform(xIn)
            else:
                xInPrime = self._diffeo.inverseTransform(xIn)
        else:
            xInPrime = xIn
        
        # Integrate
        def fInt(x,t):
            x.resize((dim,nPt))
            v = self.getCtrlSpaceVelocity(x)
            v.resize((dim*nPt,))
            return v
        
        Xc = odeint(fInt,xInPrime.ravel(),t,rtol=1e-6,atol=1e-7)
        Xc = Xc.T.reshape((dim,nPt*t.size))  # Format [x_0_t0,x_0_t1,...,x_0_tN,x_1_t0,x_1_t1,...,x_1_t
        
        out = []
        
        if outInDem:
            if self.baseDir:
                Xd = self._diffeo.inverseTransform(Xc)  # Format [x_0_t0,x_0_t1,...,x_0_tN,x_1_t0,x_1_t1,...,x_1_t
            else:
                Xd = self._diffeo.forwardTransform(Xc)
            # Reverse the scaling
            if self._overallInverseScaling is not None:
                Xd *= self._overallInverseScaling
            if self._demSpaceOffset is not None:
                Xd += self._demSpaceOffset
            out.append(Xd)
            if returnVel:
                out.append(self.getDemSpaceVelocity(Xd))
        
        if outInCtrl:
            out.append(Xc)
            if returnVel:
                out.append(self.getCtrlSpaceVelocity(Xc))
        
        out.append([t.size,dim,nPt])
        return out


class gmmDiffeoCtrlSplit:
    def __init__(self,aDiffeo,gmmDynTang:mEM.GaussianMixtureModel,gmmDynRad:mEM.GaussianMixtureModel,baseDir: bool = False,breakRadius: float = 0.1,radialMinConv: Union[float,Callable,learningMove] = -0.001):
        self._diffeo = aDiffeo
        self._dynTang = gmmDynTang
        self._dynRad = gmmDynRad
        self._Id = Idn(1,aDiffeo._dim)
        self._Id.setflags(write=False)
        self.baseDir = baseDir  # Transformation direction from demonstration space to control space. True->Forward , False:Inverse
        self.breakRadius = breakRadius
        self.radialMinConv = radialMinConv
        self._overallScaling = None
        self._overallInverseScaling = None
        self._baseVelocity = 0.

    @property
    def overallScaling(self):
        return deepcopy(self._overallScaling)

    @overallScaling.setter
    def overallScaling(self,newScaling: np.ndarray):
        try:
            newScaling = np.array(newScaling).reshape((self._diffeo._dim,1))
        except:
            assert 0,'new scaling could not be transformed into column-vector'
    
        self._overallScaling = newScaling
        self._overallInverseScaling = 1./newScaling
        return 0

    def getCtrlSpaceRadialVelocity(self,x: np.ndarray, xn:Union[None,np.ndarray]=None, xnormed:Union[None,np.ndarray]=None) -> np.ndarray:

        if (x.shape[1] == 1) and isinstance(x,np.ndarray):
            vRadn = self._dynRad._evalMapCPU(x)
        else:
            vRadn = self._dynRad.evalMap(x)

        # Get the norms
        xn = cNorm(x,kd=True) if (xn is None) else xn
        xn.resize((1,x.shape[1]))
        xnormed = x/(xn+epsFloat) if (xnormed is None) else xnormed

        # Take care of breakRadius
        #ind = xn[0,:] < 5.*self.breakRadius
        #if np.any(ind):
        #    vRadn[0,ind] = self._baseVelocity*xn[0,ind]/self.breakRadius

        breakCoef0 = 1.-np.exp(-xn/self.breakRadius)
        breakCoef1 = 1.-np.exp(-xn/(2.*self.breakRadius))
        vRadn = self._baseVelocity*breakCoef0 + vRadn*breakCoef1
        vRad = xnormed*vRadn

        return vRad

    def getCtrlSpaceTangentialVelocity(self,x):

        if (x.shape[1] == 1) and isinstance(x,np.ndarray):
            vtang = self._dynTang._evalMapCPU(x)
        else:
            vtang = self._dynTang.evalMap(x)

        return vtang

    def getCtrlSpaceVelocity(self,x,fullOut=False):
        # Assuming that self.radialMonConv is a learningMove

        v =  self.getCtrlSpaceRadialVelocity(x)
        v += self.getCtrlSpaceTangentialVelocity(x)

        # Assure convergence via the learnedMove object
        # ind = np.logical_not(ind)
        if fullOut:
            v,fDict = self.radialMinConv.ensureMinConv(x,v,fullOut=fullOut)  # v[:,ind],fDict = self.radialMinConv.ensureMinConv(x[:,ind],v[:,ind],xnSquared[:,ind],fullOut=fullOut)
            # ind1 = ind.copy()
            # ind1[ind] = fDict['ind']
            ind1 = fDict['ind']
        else:
            v = self.radialMinConv.ensureMinConv(x,v)  # v[:,ind] = self.radialMinConv.ensureMinConv(x[:,ind],v[:,ind],xnSquared[:,ind])

        if fullOut:
            return v,{"ind0":ind,"ind1":ind1}
        else:
            return v

    def getDemSpaceVeloctiy(self,x):
        dim,nPt = x.shape

        # Do the initial scaling
        if self._overallScaling is not None:
            x *= self._overallScaling

        if x.shape[1] == 1:
            # Transform into cartesian control space
            if self.baseDir:
                xprime,Jac = self._diffeo.forwardTransformJac(x,Jacp=self._Id)
                # Inverse the jacobian
                Jac = inv(Jac[0,:,:],overwrite_a=True,check_finite=False)
            else:
                # This function already returns the jacobian of the inverse transformation
                xprime,Jac = self._diffeo.inverseTransformJac(x,Jacp=self._Id,outInvJac=True)
                Jac = Jac[0,:,:]
            # Get control space velocity
            vprime = self.getCtrlSpaceVelocity(xprime)
            # Invert the jacobian and compute the velocity in the demonstration space
            v = dot(Jac,vprime)
        else:
            # Transform into cartesian control space
            JacRA = mt.sctypes.RawArray(mt.c.c_double,nPt*dim*dim)
            Jac = np.frombuffer(JacRA)
            Jac.resize((x.shape[1],x.shape[0],x.shape[0]))
            if self.baseDir:
                xprime,_ = self._diffeo.forwardTransformJac(x,Jac=IdnR(nPt,dim),outJac=Jac)
                # Inverse the jacobian
                mt.slicedInversion(JacRA,cpy=False,NnDim=[x.shape[1],x.shape[0]],returnNp=False)
            else:
                # This function already returns the jacobian of the inverse transformation
                xprime,_ = self._diffeo.inverseTransformJac(x,Jacp=IdnR(nPt,dim),outJac=Jac,outInvJac=True)
            # Get control space velocity
            vprime = self.getCtrlSpaceVelocity(xprime)
            # Transform velocity
            v = np.einsum('ijk,ki->ji',Jac,vprime)  # Jac holds the inverse since values are overridden
        # Reverse the initial scaling
        if self._overallInverseScaling is not None:
            v *= self._overallInverseScaling
        return v

    def getTrajectory(self,xIn: np.ndarray,t: Union[np.ndarray,float],inInDem: bool = True,outInDem: bool = True,
                      outInCtrl: bool = False,returnVel: bool = False):

        dim,nPt = xIn.shape

        if isinstance(t,(float,int)):
            t = np.arange(0.,t,0.01)

        # if xIn is in demo-space, transform
        if inInDem:
            # Do the initial scaling
            if self._overallScaling is not None:
                xIn *= self._overallScaling
            if self.baseDir:
                xInPrime = self._diffeo.forwardTransform(xIn)
            else:
                xInPrime = self._diffeo.inverseTransform(xIn)
        else:
            xInPrime = xIn

        # Integrate
        def fInt(x,t):
            x.resize((dim,nPt))
            v = self.getCtrlSpaceVelocity(x)
            v.resize((dim*nPt,))
            return v

        Xc = odeint(fInt,xInPrime.ravel(),t,rtol=1e-6,atol=1e-7)
        Xc = Xc.T.reshape((dim,nPt*t.size))  # Format [x_0_t0,x_0_t1,...,x_0_tN,x_1_t0,x_1_t1,...,x_1_t

        out = []

        if outInDem:
            if self.baseDir:
                Xd = self._diffeo.inverseTransform(Xc)  # Format [x_0_t0,x_0_t1,...,x_0_tN,x_1_t0,x_1_t1,...,x_1_t
            else:
                Xd = self._diffeo.forwardTransform(Xc)
            # Reverse the scaling
            if self._overallInverseScaling is not None:
                Xd *= self._overallInverseScaling
            out.append(Xd)
            if returnVel:
                out.append(self.getDemSpaceVeloctiy(Xd))

        if outInCtrl:
            out.append(Xc)
            if returnVel:
                out.append(self.getCtrlSpaceVelocity(Xc))

        out.append([t.size,dim,nPt])
        return out






def findGreedyDynamics(x:np.ndarray,v:np.ndarray,breakRadius = 1., radialMinConv=-0.001,nPartial=10, nKMax=8, add2Sigma=30.e-1, relTol=5.e-3, absTol=.25e-2, convFac=1., reOpt=False, otherOpts={}):
    
    assert np.all(np.array(breakRadius)>0.)
    
    #Learn all dynamics
    dynGMM = gi.greedyEMCPU(np.vstack((x,v)),nVarI= x.shape[0],nPartial=nPartial, nKMax=nKMax, add2Sigma=add2Sigma, relTol=relTol, absTol=absTol, convFac=convFac, reOpt=reOpt, otherOpts=otherOpts)

    thisDiffeo = du.diffeomorphism(dim=x.shape[0])

    gmmDiffeoDyn = gmmDiffeoCtrl(thisDiffeo, dynGMM, baseDir=True, breakRadius=breakRadius, radialMinConv=radialMinConv)

    return gmmDiffeoDyn

def findGreedyDynamicsSplit(x:np.ndarray,v:np.ndarray,breakRadius = 1., radialMinConv=-0.001,nPartial=10, nKMax=8, add2Sigma=30.e-1, relTol=1e-2, absTol=.5e-2, convFac=1., reOpt=False):

    assert breakRadius > 0.

    xnorm = x/(cNorm(x)+epsFloat)
    vrad = sum(xnorm*v,axis=0,keepdims=True)
    vtang = v-vrad*xnorm

    #Learn dynamics
    dynGMMrad = gi.greedyEMCPU(np.vstack((x,vrad)),nVarI= x.shape[0],nPartial=nPartial, nKMax=nKMax, add2Sigma=add2Sigma, relTol=relTol, absTol=absTol, convFac=convFac, reOpt=reOpt)

