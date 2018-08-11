#Forward declare
class learningMove:
    pass


from coreUtils import *

import gmmDynamics
import antisymmetricDynamics
import modifiedEM

class convScaling:
    def __init__(self, alpha, beta):
        self._alpha = alpha
        self._beta = beta

    def __call__(self, x:np.ndarray, xn:np.ndarray, v:np.ndarray):

        allowedRadVel = self._alpha + self._beta*xn

        ind = xn[0,:] < 100.*epsFloat
        xn[[0],ind] = 1.
        xnormed = x/xn

        # Check radial velocity
        vrad = sum(xnormed*v,axis=0,keepdims=True)

        ind2 = (vrad > allowedRadVel)[0,:]

        convCorrFac = np.zeros((1,x.shape[1]))

        convCorrFac[[0],ind2] += -vrad[[0],ind2]+allowedRadVel[[0],ind2]

        return convCorrFac

class learningMove:
    def __init__(self, diffeoCtrl:Union["gmmDynamics.gmmDiffeoCtrl","antisymmetricDynamics.asymDiffeoCtrl"], alpha, beta, x, t):
        """This only works for antisym if radial vel is learned"""

        self._dyn = diffeoCtrl
        self._alpha = alpha
        self._beta = beta

        self._t = []

        if isinstance(self._dyn, gmmDynamics.gmmDiffeoCtrl):
            thisGMM = self._dyn._dyn
        elif isinstance(self._dyn, gmmDynamics.gmmDiffeoCtrlSplit):
            if self._dyn._dynRad.nK > self._dyn._dynTang.nK:
                thisGMM = self._dyn._dynRad
            else:
                thisGMM = self._dyn._dynTang
        elif isinstance(self._dyn, antisymmetricDynamics.asymDiffeoCtrl):
            assert isinstance(self._dyn._baseVel, modifiedEM.GaussianMixtureModel)
            thisGMM = self._dyn._baseVel

        self._GMM = thisGMM
        assert self.compTime(x,t)


    def ensureMinConv(self,x,v,xnSquared=None, fullOut=False):
        
        xnSquared = np.sum(np.square(x),axis=0, keepdims=True) if xnSquared is None else xnSquared
        
        # Get radial convergence
        xtv = np.sum(x*v,axis=0, keepdims=True)
        
        # Get minimal convergence demanded for each point
        # Get the weights of each gaussian
        # thisW = self._GMM.getWeights(x)
        # Get the most probable time
        # thisT = np.sum(thisW*self._t,0,keepdims=True)  # This is always positive since it is a sum of positive terms
        thisT = self.getTime(x)
        thisT.resize((1,thisT.size))
        # get the corresponding minimal convergence
        minConvFac = self._alpha+self._beta*thisT
        #minConvFac.resize((1,x.shape[1]))
        
        #Check
        ind = 2*xtv/(xnSquared+epsFloat) >= minConvFac
        ind = ind.reshape((x.shape[1],))
        
        #Adjust
        if np.any(ind): #Nahh
            try:
                adjustFactor = -xtv[[0],ind] +1./2.*minConvFac[[0],ind]*xnSquared[[0],ind]
                #Correct velocity
                v[:,ind] += adjustFactor*x[:,ind]/(xnSquared[:,ind]+epsFloat)
            except:
                assert 0
                pass
        if fullOut:
            return v, {'ind':ind}
        else:
            return v
    
    def getTime(self, x):
        # Get the weights of each gaussian
        thisW = self._GMM.getWeights(x)
        thisT = np.sum(thisW*self._t,0).squeeze()  # This is always positive since it is a sum of positive terms
        
        return thisT
    
    def getMinConv(self, x):

        thisT = self.getTime(x)

        minConvFac = self._alpha + self._beta*thisT

        return minConvFac

    def compTime(self,x,t):
        for aGaussian in self._GMM:
            #Get the weights
            thisW = aGaussian.getWeights(x, kd=False)
            #Normalize weights
            thisW /= np.sum(thisW)
            thisMean = np.sum(t*thisW)
            self._t.append(thisMean)
        self._t = np.array(self._t).reshape((-1,1))
        return True


class constMinConv:
    
    def __init__(self, radialMinConv:float=0.):
        self._radianMinConv = radialMinConv

    def ensureMinConv(self,x,v,xnSquared=None,fullOut=False):
    
        xnSquared = np.sum(np.square(x),axis=0,keepdims=True) if xnSquared is None else xnSquared
    
        # Get radial convergence
        xtv = np.sum(x*v,axis=0,keepdims=True)
    
        # Check
        ind = 2*xtv/(xnSquared+epsFloat) >= self._radianMinConv
        ind = ind.reshape((x.shape[1],))

        # Adjust
        if np.any(ind):  # Nahh
            try:
                adjustFactor = -xtv[[0],ind]+1./2.*self._radianMinConv*xnSquared[[0],ind]
                # Correct velocity
                v[:,ind] += adjustFactor*x[:,ind]/(xnSquared[:,ind]+epsFloat)
            except:
                assert 0
                pass
        if fullOut:
            return v,{'ind':ind}
        else:
            return v

def discreteLineMvt(x, dx, kt, kr, kd=.5, xd=None):
    # x : Demonstration points interpreted as mass points
    # t : time vector
    # kt : stiffness of time spring
    # kr : rotational stiffness
    # xd : Last velocity of each point

    dim, nPt = x.shape
    if xd is None:
        xdd = zeros((dim, nPt))
    else:
        xdd = -kd*xd

    dx = x[:,1:]-x[:,:-1]
    dxn = cNorm(dx, False)
    dxnorm = sp.divide(dx,dxn)

    return


