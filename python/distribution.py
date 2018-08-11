from coreUtils import *

from copy import deepcopy as dp
import warnings

__debug = False
# Currently only for usage in locally weighted directions
# -> we only need getWeights
class cauchyKernel:
    """Modified multivariate cauchy distribution"""
    #It is actually no correct multivariate cauchy distrib
    # but it is heavy tailed
    def __init__(self, nVarI=None, Sigma=None, mu=None, doCond=False, gamma=1., pi=1.):
        
        
        self.doCond = doCond
        self._Sigma = Sigma
        self._SigmaI = inv(Sigma)
        self._SigmaCI = inv(chol(Sigma).T)
        self._mu = mu
        self._gamma = gamma
        self._pi = pi

    def __add__(self,other):
        assert (isinstance(other,cauchyKernel) and (self.dim == other.dim) and (self.doCond == other.doCond))
        return cauchyKernel(None,self.Sigma+other.Sigma,self.mu+other.mu,self.doCond, self._gamma+other._gamma,self._pi+other._pi)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self,other):
        assert (isinstance(other,(int,float)))
        other = float(other)
        assert (other>0.), "factor must be positive"
        return cauchyKernel(None,other*self.Sigma,other*self.mu,self.doCond, other*self._gamma, other*self._pi)

    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __copy__(self):
        return self.__deepcopy__({})
    def __deepcopy__(self, memo):
        pars = self.toPars
        new = cauchyKernel()
        new.loadPars(pars)
        return new
    
    @property
    def dim(self):
        return self._mu.size
    @property
    def Sigma(self):
        return np.copy(self._Sigma)
    @property
    def mu(self):
        return np.copy(self._mu)
    @mu.setter
    def mu(self,newMu):
        self._mu = newMu.reshape(self._mu.shape)
    @Sigma.setter
    def Sigma(self, newSigma):
        assert newSigma.shape == self._Sigma.shape
        try:
            self._SigmaI = inv(newSigma)
            self._SigmaCI = inv(chol(newSigma).T)
            self._Sigma = newSigma
        except np.linalg.LinAlgError:
            print("New covariance not pos def -> not applied")
    
    def toPars(self):
        return {"doCond":self.doCond, "mu":self.mu, "Sigma":self.Sigma, "gamma":self._gamma, "pi":self._pi}
    def loadPars(self, pars):
        self.doCond = pars["doCcond"]
        self.mu = pars["mu"]
        self.Sigma = pars["Sigma"]
        self._gamma = pars['gamma']
        self._pi = pars['pi']

    def toStringList(self):
        totStringList = []

        # Size definitions
        totStringList.append(int2Str(self.dim))
        # Mean vector
        totStringList += vec2List(self.mu.squeeze())
        # Covariance matrix in c-style
        totStringList += vec2List(self.Sigma.flatten('C'))
        #cond
        totStringList.append(bool2Str(self.doCond))
        #gamma
        totStringList.append(double2Str(self._gamma))
        #pi
        totStringList.append(double2Str(self._pi))
        return totStringList

    def getWeights(self, x, out=None, kd=True, cpy=True):

        out = np.empty((1,x.shape[1])) if kd else np.empty((x.shape[1],))

        if cpy:
            x = x-self._mu
        else:
            x -= self._mu

        cNormSquare(np.dot(self._SigmaCI, x), out=out, kd=kd, cpy=False) # get the weighted distance
        out[:] = (self._gamma / ((self._gamma ** 2) + out))

        if self.doCond:
            out *= (1./self._pi)

        return out






