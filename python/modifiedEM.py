#! /usr/bin/env python3
### Module implementing the modified EM-algorithm used to find the underlying process

from coreUtils import *
try:
    from cudaUtils import *
    from modifiedEMGPUUtils import *
except:
    cpuFloat = np.float_
    modEMdefStreams=None
    epsMin = np.nextafter(0.,1.,dtype=cpuFloat)
    epsCPU = 3.*np.nextafter(0.,1.,dtype=cpuFloat)
    print("No cuda support")

from copy import deepcopy as dp
import warnings

__debug = False

import plotUtils as pu


def _logsumexpCPU(x,out=None,kd=True,cpy=False):
    # Use the max to normalize, as with the log this is what accumulates
    # the less errors
    assert (out is None) or ((out.shape == (1,x.shape[1])) and kd) or ((out.shape == (x.shape[1],)) and (not kd))
    if out is None:
        if kd:
            out = empty((1,x.shape[1]))
        else:
            out = empty((x.shape[1],))
    vmax = x.max(axis=0)
    if cpy:
        np.log(np.sum(np.exp(x-vmax),axis=0,keepdims=kd),out=out)
    else:
        x -= vmax
        np.exp(x,out=x)
        np.sum(x,axis=0,out=out,keepdims=kd)
        np.log(out,out=out)
    out += vmax
    return out

logsumexpCPU = _logsumexpCPU


def adjustZeroKernel(nVarI, mu, Sigma):
    #mu = np.zeros_like(mu)
    #thisSigma = Sigma.copy()
    #mainDiag = np.diag(thisSigma).copy()
    #sideDiag = np.diag(thisSigma,nVarI).copy()
    #sideDiag = np.minimum(sideDiag,-0.01)
    #thisSigma[:] = 0.
    #thisSigma += np.diag(mainDiag)
    #thisSigma += np.diag(sideDiag,nVarI)
    #thisSigma += np.diag(sideDiag,-nVarI)
    #
    #while True:
    #    try:
    #        chol(thisSigma-1e-3*np.identity(thisSigma.shape[0]))
    #        break
    #    except np.linalg.LinAlgError:
    #        pass
    #    thisSigma += 1e-3*np.identity(thisSigma.shape[0])
    #
    #Sigma = thisSigma
    
    mux = mu[:nVarI,[0]].copy()
    muy = mu[nVarI:,[0]].copy()
    
    Sxx = Sigma[:nVarI,:nVarI].copy()
    SxxI = inv(Sxx);
    Syx = Sigma[nVarI:,:nVarI].copy()


    if 0:
        while True:
            SyxSxxI = np.dot(Syx, SxxI)
            SyxSxxI += SyxSxxI.T
            
            try:
                chol(-SyxSxxI)
                break
            except np.linalg.LinAlgError:
                pass
            Syx -= 1e-3*np.identity(Syx.shape[0])
    
    mux *= 0.9
    muy = np.dot(np.dot(Syx, SxxI),mux)
    
    mu = np.vstack((mux,muy))
    if 0:
        Sigma[nVarI:,:nVarI] = Syx.copy()
        Sigma[:nVarI,nVarI:] = Syx.copy().T
    return mu,Sigma

def myCov(x, w=None, N=None, isRoot=False, isCentered=True, out=None, cpy=True, forceCPU=False, streams=modEMdefStreams, doSync=False):
    # Computes the covariance of the data x, with the possibility of adjusting the weight
    # of each data point
    # So given the data is centered (if not it will be centered)
    # This function computes
    # sum_k w_k.x[:,k].x[:,k]'
    # If w is None it will be treated as all ones
    # Since internally we compute
    # sum_k (sqrt(w_k).x[:,k]).(sqrt(w_k).x[:,k])'
    # w_k can already be given as sqrt(w_k), the isRoot=True
    
    if mainUseGPU_ and ((isinstance(x,gpuarray.GPUArray) or (x.shape[1] > nMinGPU)) and not forceCPU):
        if not isinstance(x,gpuarray.GPUArray):
            x = gpuarray.to_gpu_async(toGPUData(x), stream=streams[0])
            cpyx=False
        else:
            cpyx=cpy
        
        if not isCentered:
            # Center date
            xMeanN = -cudaSum(x,1,keepdims=True)/float(x.shape[1])#This is sync in default stream :( !!TBDGPUPERF
            if cpyx:
                x = cudaMisc.add_matvec(x,xMeanN,axis=0,stream=streams[0])
            else:
                cudaMisc.add_matvec(x,xMeanN,axis=0,out=x,stream=streams[0])
            #x -= (cudaSum(x,1,keepdims=True)/x.shape[1])
            
        if w is not None:
            if not isinstance(w,gpuarray.GPUArray):
                w = gpuarray.to_gpu_async(toGPUData(w),stream=streams[1])
                streams[1].synchronize()

            if not isRoot:
                N = float(cudaSum(w).get()) if N is None else float(N)
                if cpy:
                    w=cudaMath.sqrt(w, stream=streams[1])
                else:
                    cudaMath.sqrt(w,out=w, stream=streams[1])
            else:
                N = float(cudaSum(w**2).get()) if (N is None) else float(N)# !!TBDGPUPERF
            # Apply root of weight
            streams[0].synchronize()
            streams[1].synchronize()
            x = cudaMisc.multiply(x,w)#This is sync # !!TBDGPUPERF
        else:
            N = x.shape[1]-1

            

        # Compute actual covarianz
        assert (out is None) or isinstance(out, gpuarray.GPUArray)
        out = gpuarray.empty((x.shape[0],x.shape[0]), dtype=gpuFloat) if out is None else out
        myCovGPU(x,out,N,stream=streams[0],doSync=doSync)
    else:
        x = x.copy() if cpy else x
        w = w.copy() if (cpy and (w is not None)) else w

        if not isCentered:
            # Center date
            x -= sum(x,1,keepdims=True)/x.shape[1]

        if (w is None) and (N is None):
            N = x.shape[1]-1
        elif (w is not None):
            if not isRoot:
                N = sum(w) if N is None else N
                np.sqrt(w,out=w)
            else:
                N = sum(square(w)) if N is None else N
            np.multiply(x,w,out=x)
        #Compute actual covarianz
        out = empty((x.shape[0],x.shape[0])) if out is None else out
        #Only compute upper triang
        for i in range(x.shape[0]):
            for j in range(i, x.shape[0]):
                out[i,j] = sum(np.multiply(x[i,:], x[j,:]))
                out[j,i] = out[i,j]
        out /= N
    return out
    

########################################################################################################################
class gaussianKernel:
    def __init__(self, nVarI = None, Sigma=None, mu=None, doCond=False):

        if isinstance(nVarI, dict):
            pars = nVarI
            nVarI = pars["nVarI"]
            Sigma = pars["Sigma"]
            mu = pars["mu"]
            doCond = pars["doCond"]

        #For the moment we will assume that there is not a ridiculous amount of kernels,
        #So we will assign two threads/cublas handle to each kernel
        if mainUseGPU_:
            self._streams = [cuda.Stream(),cuda.Stream()]
            self._cuBlasHandles = [skcuda.cublas.cublasCreate(), skcuda.cublas.cublasCreate()]
    
            skcuda.cublas.cublasSetStream(self._cuBlasHandles[0], self._streams[0].handle)
            skcuda.cublas.cublasSetStream(self._cuBlasHandles[1], self._streams[1].handle)

        self.doCond = doCond
        self._nVarI = 1 if nVarI is None else nVarI #Number of independent variables
        self._nVarTot = None

        self._shape = None
        self._Sigma = na([])
        self._SigmaC = na([])
        self._SigmaI = na([])
        self._SigmaCI = na([])
        self._lnSigmaCIdsqrt2 = na([]) #This value is prescaled to incorporate 1/2 factor
        self._scaling = 0
        self._lnscaling = 0
        self._mu = na([])
        
        if mainUseGPU_:
            self._Sigma_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SigmaC_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SigmaI_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SigmaCI_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._lnSigmaCIdsqrt2_gpu = na([]) #This value is prescaled to incorporate 1/2 factor
            self._scaling_gpu = 0#gpuarray.to_gpu(toGPUData(na([1])))
            self._lnscaling_gpu = 0  # gpuarray.to_gpu(toGPUData(na([1])))
            self._mu_gpu = gpuarray.to_gpu(toGPUData(na([1])))
        
        #The same for the independent / dependent variables for MAP
        self._Sxx = na([])
        self._SxxI = na([])
        self._SxxCI = na([])
        self._lnSxxCIdsqrt2 = na([])
        self._Syx = na([])
        self._SyxSxxI = na([])
        self._scalingx = 0
        self._lnscalingx = 0
        self._mux = na([])
        self._muy = na([])
        
        if mainUseGPU_:
            self._Sxx_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SxxI_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SxxCI_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._lnSxxCIdsqrt2_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._Syx_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._SyxSxxI_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._scalingx_gpu = 0#gpuarray.to_gpu(toGPUData(na([1])))
            self._lnscalingx_gpu = 0  # gpuarray.to_gpu(toGPUData(na([1])))
            self._mux_gpu = gpuarray.to_gpu(toGPUData(na([1])))
            self._muy_gpu = gpuarray.to_gpu(toGPUData(na([1])))
        

        if Sigma is not None:
            self.Sigma = Sigma
        if mu is not None:
            self.mu = mu

        return None
    
    ###############################################
    #def __init__(self, nVarI = None, Sigma=None, mu=None, doCond=False):
    def __add__(self, other):
        assert (isinstance(other, gaussianKernel) and (self.nVarI==other.nVarI) and (self.nVarD==other.nVarD) and (self.doCond==other.doCond))
        return gaussianKernel(self.nVarI, self.Sigma+other.Sigma, self.mu+other.mu, self.doCond)
    def __radd__(self, other):
        return self.__add__(other)
    
    def __mul__(self, other):
        assert (isinstance(other, (int,float)))
        other = float(other)
        return gaussianKernel(self.nVarI, other*self.Sigma, other*self.mu, self.doCond)
    def __rmul__(self, other):
        return self.__mul__(other)
    ###############################################
    @property
    def dim(self):
        return self._mu.size
    @property
    def nVarTot(self):
        return dp(self._mu.size)
    @nVarTot.setter
    def nVarTot(self, newVal):
        if not(  (newVal==self._Sigma.shape[0]) and (newVal==self._mu.size) ):
            warnings.warn('Kernel not consistent, proceed with caution')
        self._nVarTot = newVal
    @property
    def nVarI(self):
        return dp(self._nVarI)
    @nVarI.setter
    def nVarI(self, newValue):
        if (self._Sigma.size == 0) or (self._mu.size==0):
            #Do nothing if sigma or mu not yet set
            self._nVarI = newValue
        else:
            #Actually do stuff
            self._nVarI = n = newValue
            self._Sxx = self._Sigma[:n,:n]
            self._SxxI = inv(self._Sxx)
            self._SxxCI = inv(chol(self._Sxx).T)
            self._lnSxxCIdsqrt2 = self._SxxCI/sq2
            self._Syx = self._Sigma[n:,:n]
            self._SyxSxxI = dot(self._Syx, self._SxxI)
            self._scalingx = na([1./sqrt( (2.*pi)**(self._nVarI) * det(self._Sxx) )])
            self._lnscalingx = np.log(self._scalingx)
            self._mux = self._mu[:n,[0]]
            self._muy = self._mu[n:,[0]]
            #Let the driver handle the scope
            if mainUseGPU_:
                self._Sxx_gpu = gpuarray.to_gpu_async(toGPUData(self._Sxx),stream=self._streams[0])
                self._SxxI_gpu = gpuarray.to_gpu_async(toGPUData(self._SxxI),stream=self._streams[1])
                self._SxxCI_gpu = gpuarray.to_gpu_async(toGPUData(self._SxxCI),stream=self._streams[0])
                self._lnSxxCIdsqrt2_gpu = gpuarray.to_gpu_async(toGPUData(self._lnSxxCIdsqrt2),stream=self._streams[1])
                self._Syx_gpu = gpuarray.to_gpu_async(toGPUData(self._Syx),stream=self._streams[0])
                self._SyxSxxI_gpu = gpuarray.to_gpu_async(toGPUData(self._SyxSxxI),stream=self._streams[1])
                #TBD
                self._scalingx_gpu = float(self._scalingx)#gpuarray.to_gpu_async(toGPUData(self._scalingx))
                self._lnscalingx_gpu = float(self._lnscalingx)  # gpuarray.to_gpu_async(toGPUData(self._lnscalingx))
                self._mux_gpu = gpuarray.to_gpu_async(toGPUData(self._mux),stream=self._streams[0])
                self._muy_gpu = gpuarray.to_gpu_async(toGPUData(self._muy),stream=self._streams[1])
                self._streams[0].synchronize()
                self._streams[1].synchronize()
        return 0
    @property
    def nVarD(self):
        return self._nVarTot-self._nVarI
        
    ###############################################
    #Sigma and mu can only be accessed through getter and setter methods to keep data consistent across CPU/GPU
    #ATTENTION! NO cross-validation is performed between mu and Sigma
    @property
    def Sigma(self):
        return np.copy(self._Sigma)
    @Sigma.setter
    def Sigma(self, newSigma):

        shapeChange = newSigma.shape != self._shape
        self.nVarTot = newSigma.shape[0]
        
        #Perform calculations
        self._shape = newSigma.shape
        self._Sigma = newSigma.astype(cpuFloat)
        self._SigmaC = chol(self._Sigma)
        self._SigmaI = inv(self._Sigma)
        self._SigmaCI = inv(self._SigmaC.T)
        self._lnSigmaCIdsqrt2 = self._SigmaCI/sq2
        self._scaling = na([1./sqrt( (2.*pi)**(self.nVarTot) * det(self._Sigma) )])
        self._lnscaling = np.log(self._scaling)

        #Push to device
        # Push the new matrices to preallocated on device
        # or create a new gpuarray if size changed
        if mainUseGPU_:
            if shapeChange:
                self._Sigma_gpu = gpuarray.to_gpu_async(toGPUData(self._Sigma),stream=self._streams[0])
                self._SigmaC_gpu = gpuarray.to_gpu_async(toGPUData(self._SigmaC),stream=self._streams[1])
                self._SigmaI_gpu = gpuarray.to_gpu_async(toGPUData(self._SigmaI),stream=self._streams[0])
                self._SigmaCI_gpu = gpuarray.to_gpu_async(toGPUData(self._SigmaCI),stream=self._streams[1])
                self._lnSigmaCIdsqrt2_gpu = gpuarray.to_gpu_async(toGPUData(self._lnSigmaCIdsqrt2),stream=self._streams[0])
                self._scaling_gpu = float(self._scaling)#gpuarray.to_gpu_async(toGPUData(self._scaling))
                self._lnscaling_gpu = float(self._lnscaling)  # gpuarray.to_gpu_async(toGPUData(self._scaling))
            else:
                self._Sigma_gpu.set_async(toGPUData(self._Sigma),stream=self._streams[0])
                self._SigmaC_gpu.set_async(toGPUData(self._SigmaC),stream=self._streams[1])
                self._SigmaI_gpu.set_async(toGPUData(self._SigmaI),stream=self._streams[0])
                self._SigmaCI_gpu.set_async(toGPUData(self._SigmaCI),stream=self._streams[1])
                self._lnSigmaCIdsqrt2_gpu.set_async(toGPUData(self._lnSigmaCIdsqrt2),stream=self._streams[0])
                #TBD
                self._scaling_gpu = float(self._scaling)#gpuarray.set_async(toGPUData(self._scaling))
                self._lnscaling_gpu = float(self._lnscaling)  # gpuarray.set_async(toGPUData(self._scaling))
            self._streams[0].synchronize()
            self._streams[1].synchronize()
        
        #Update other
        self.nVarI = self._nVarI
        

        return 0

    ###############################################
    @property
    def mu(self):
        return np.copy(self._mu)
    @property
    def mux(self):
        return np.copy(self._mux)
    @property
    def muy(self):
        return np.copy(self._muy)
    @mu.setter
    def mu(self, newMu):
        self._mu = newMu.reshape((-1,1)).astype(cpuFloat)
        if mainUseGPU_:
            self._mu_gpu = gpuarray.to_gpu_async(toGPUData(self._mu),stream=self._streams[0])#Let the old one go out of scope, will be handled automatically
        # Update other
        self.nVarI = self._nVarI
        self.nVarTot = len(newMu)
        if mainUseGPU_:
            self._streams[0].synchronize()
        return 0

    ###############################################
    def __copy__(self):
        return self.__deepcopy__({})
    def __deepcopy__(self, memo):
         return gaussianKernel(self.nVarI, self.Sigma, self.mu)
    ###############################################
    
    def toStringList(self):
        
        totStringList = []
        
        #Size definitions
        totStringList.append(int2Str(self.nVarI))
        totStringList.append(int2Str(self.nVarD))
        #Mean vector
        totStringList += vec2List(self.mu.squeeze())
        #Covariance matrix in c
        totStringList += vec2List(self.Sigma.flatten('C'))
        
        #Did i miss this??
        totStringList.append(bool2Str(self.doCond))
        
        return totStringList
        
    
    def toPars(self):
        return {"doCond":self.doCond, "nVarI":self.nVarI, "mu":self.mu, "Sigma":self.Sigma}
    def loadPars(self, pars):
        self.doCond = pars["doCcond"]
        self.nVarI = pars["nVarI"]
        self.mu = pars["mu"]
        self.Sigma = pars["Sigma"]
    ###############################################
    def _getLogProbCPU(self, x, out=None, cpy=True, kd=True, lnprioir=0.):
        # Get the log probability of each point in x to belong to this gaussian
        #
        if out is None:
            if kd:
                out = empty((1,x.shape[1]))
            else:
                out = empty(x.shape[1],)
        else:
            kd = out.shape == (1,x.shape[1])
    
        # Check if input is full-dimensional, if not split mean and covariance
        assert x.shape[0] in [self._nVarTot,self._nVarI]
        if x.shape[0] == self._shape[0]:
            tempSCI = self._lnSigmaCIdsqrt2
            tempScaling = self._lnscaling
            tempMu = self._mu
        else:
            tempSCI = self._lnSxxCIdsqrt2
            tempScaling = self._lnscalingx
            tempMu = self._mux

        if cpy:
            x = x - tempMu
        else:
            x -= tempMu
        
        # todo replace with lapack strmv
        cNormSquare(np.dot(tempSCI,x),out=out,kd=kd,cpy=False)
        np.negative(out, out=out)
        if self.doCond:
            out += (tempScaling+lnprioir)
        
        return out
    ###############################################
    def _getLogProbGPU(self, x, outGPU = None, outCPU=None, cpy=True, kd=True, lnprioir=0.):
        
        assert mainUseGPU_, 'GPU deactivated'

        if isinstance(x, np.ndarray):
            xGPU = gpuarray.to_gpu_async(toGPUData(x), stream=self._streams[0])
        else:
            assert isinstance(x, gpuarray.GPUArray)
            if cpy:
                xGPU = x.copy()
            else:
                xGPU = x
        
        if (outGPU is None):
            if kd:
                outGPU = gpuarray.empty((1,x.shape[1]), dtype=gpuFloat)
            else:
                outGPU = gpuarray.empty((x.shape[1],),dtype=gpuFloat)
        else:
            assert isinstance(outGPU, gpuarray.GPUArray)

        # Check if input is full-dimensional, if not split mean and covariance
        assert x.shape[0] in [self._nVarTot,self._nVarI]
        if x.shape[0] == self._shape[0]:
            tempSCI = self._lnSigmaCIdsqrt2_gpu
            tempScaling = self._lnscaling_gpu
            tempMu = self._mu_gpu
        else:
            tempSCI = self._lnSxxCIdsqrt2_gpu
            tempScaling = self._lnscalingx_gpu
            tempMu = self._mux_gpu

        cudaMisc.add_matvec(xGPU,-tempMu,out=xGPU, axis=0, stream=self._streams[0])
        tempGPU = cudaDot(tempSCI, xGPU, handle=self._cuBlasHandles[0]) #handle[0] is linked to stream[0];Check if synchronization is really not necessary
        self._streams[0].synchronize()
        tempGPU **= 2 # !!TBDGPUPERF There is no stream option for __pow__
        #This is probably faster than returning a temporary object
        cudaSum(tempGPU,0,out=outGPU,keepdims=kd)  # !!TBDGPUPERF
        #negateGPU(outGPU, stream=self._streams[0]) #TBD
        self._streams[0].synchronize()
        outGPU *= -1.
        
        if self.doCond:
            outGPU += (tempScaling+lnprioir)# !!TBDGPUPERF

        if outCPU is not None:
            #cuda.memcpy_dtoh(outCPU, tempGPU) #outCPU = tempGPU.get()
            getNStore(outGPU,outCPU)

        return outGPU
    ###############################################
    def getLogProb(self,x,out=None,kd=True,cpy=True, lnprior=0.):
    
        if out is None:
            if kd:
                out = empty((1,x.shape[1]))
            else:
                out = empty((x.shape[1],))
    
        if mainUseGPU_ and (x.shape[1] > nMinGPU):
            self._getLogProbGPU(x,outGPU=None,outCPU=out,cpy=cpy,kd=kd,lnprior=lnprior)
        else:
            self._getLogProbCPU(x,out=out,cpy=cpy,lnprior=lnprior)
    
        if kd:
            out.resize((1,x.shape[1]))
        return out
    ###############################################
    #Get the kernel values for comlumnwise stored points
    #CPU version
    #computes v = 1/((2*pi)^d*det(Sigma)
    def _getWeightsCPU(self, x, out=None, cpy=True, kd=True):
        #get loglikelihood
        out = self._getLogProbCPU(x,out=out,cpy=cpy,kd=kd)
        #take the exp
        np.exp(out, out=out)
        return out

    ###############################################
    #GPU
    def _getWeightsGPU(self, x, outGPU=None, outCPU=None, cpy=True, kd=True):
        assert mainUseGPU_,'GPU deactivated'
        # get loglikelihood on gpu
        outGPU = self._getLogProbGPU(x,outGPU=outGPU,outCPU=None,cpy=cpy,kd=kd)
        # take the exp on gpu
        cudaMath.exp(outGPU, out=outGPU, stream=self._streams[0])
        self._streams[0].synchronize()

        if outCPU is not None:
            #cuda.memcpy_dtoh(outCPU, tempGPU) #outCPU = tempGPU.get()
            getNStore(outGPU,outCPU)

        return outGPU

    ###############################################
    #Generic call
    def getWeights(self, x, out=None, kd=True, cpy=True):

        if out is None:
            out = empty((x.shape[1],))

        if mainUseGPU_ and (x.shape[1] > nMinGPU):
            self._getWeightsGPU(x, outCPU=out, cpy=cpy, kd=kd)
        else:
            self._getWeightsCPU(x, out, cpy=cpy)

        if kd:
            out.resize((1,x.shape[1]))
        return out
    ###############################################
    #Function evaluating the gaussian using maximum a posteriori estimation
    #Currently only works for sorted input
    #so given the first n entries the others will be predicted
    def _evalMapCPU(self, x, y=None, cpy=True):
        #The map is given as
        #mu_y + Syx*SxxI*(x-mu_x)
        if y is None:
            y = zeros((self._shape[0]-self._nVarI, x.shape[1]))
        else:
            assert  y.shape == (self._shape[0]-self._nVarI, x.shape[1]), 'Output has wrong size'
        # Get difference
        if cpy:
            x = x - self._mux
        else:
            x -= self._mux

        np.dot(self._SyxSxxI, x, out=y)
        y += self._muy

        return y
    ###############################################
    #GPU version
    def _evalMapGPU(self, x, yGPU=None, yCPU=None, cpy=True):
        assert mainUseGPU_, 'GPU deactivated'
        if isinstance(x, np.ndarray):
            xGPU = gpuarray.to_gpu_async(toGPUData(np.require(x)), stream=self._streams[0])
            cpy = False #Only the gpu array will be modified. Since non-existing before no problem
        else:
            assert isinstance(x, gpuarray.GPUArray)
            if cpy:
                xGPU = x.copy()
            else:
                xGPU = x

        assert isinstance(yGPU,gpuarray.GPUArray) or (yGPU is None)
        if yGPU is None:
            yGPU = gpuarray.empty((self._nVarTot-self._nVarI, x.shape[1]),dtype=gpuFloat)#This is blocking

        #Compute delta
        cudaMisc.add_matvec(xGPU, -self._mux_gpu, out=xGPU, axis=0, stream=self._streams[0])
        #Get correlation influence
        cudaDot( self._SyxSxxI_gpu, xGPU, out=yGPU, handle=self._cuBlasHandles[0] ) #Check if no sync necessary
        #Add mean
        cudaMisc.add_matvec(yGPU, self._muy_gpu, out=yGPU, axis=0, stream=self._streams[0])

        self._streams[0].synchronize()

        if yCPU is not None:
            getNStore(yGPU, yCPU)
        
        return yGPU
    ###############################################
    #generic call to evaluate
    def evalMap(self, x, yOut=None, cpy=True):
        
        if yOut is None:
            yOut = empty((self._shape[0]-self._nVarI, x.shape[1]))
        else:
            assert yOut.shape == (self._shape[0]-self._nVarI, x.shape[1]), 'Wrong output shape'
        
        if mainUseGPU_ and ((x.shape[1] > nMinGPU) or isinstance(x, gpuarray.GPUArray)):
            self._evalMapGPU(x, yCPU=yOut, cpy=cpy)
        else:
            self._evalMapCPU(x, yOut, cpy=cpy)
        
        return yOut

########################################################################################################################
class GaussianMixtureModel:
    def __init__(self, parDict=None):
        self._gaussianList = [] #type : List[gaussianKernel]
        self._prior = zeros((0,1))
        self._lnprioir = None
        self._priorGPU = None
        self._lnprioirGPU = None
        self._doCond=True

        #Let us not be to stingy with the streams
        if mainUseGPU_:
            self._streams = [cuda.Stream(),cuda.Stream()]
            self._cuBlasHandles = [skcuda.cublas.cublasCreate(),skcuda.cublas.cublasCreate()]
    
            skcuda.cublas.cublasSetStream(self._cuBlasHandles[0],self._streams[0].handle)
            skcuda.cublas.cublasSetStream(self._cuBlasHandles[1],self._streams[1].handle)

        self._nK = None
        self._nVarI = None
        self._nVarD = None
        self._nVarTot = None
        

        if parDict is not None:
            self.loadPars(parDict)

        return None
    
    @property
    def dim(self):
        return self._nVarI+self._nVarD
    
    def toStringList(self):
    
        totStringList = []
        
        totStringList.append(int2Str(self.nVarI))
        totStringList.append(int2Str(self.nVarD))
        totStringList.append(int2Str(self.doCond))
        totStringList.append(int2Str(self.nK))
        
        totStringList += vec2List(self.prior.squeeze())
        
        for aGK in self._gaussianList:
            totStringList += aGK.toStringList()
        
        return totStringList
    
    def toText(self, fileName):
        
        with open(fileName, 'w+') as file:
            file.writelines(self.toStringList())
        
        return True
                
                
    
    def addKernel(self, aK, aP=None):
        self._gaussianList.append(aK)
        if aP is None:
            self._prior = np.vstack((self._prior, 1))
            self._prior /= sum(self._prior)
            if mainUseGPU_:
                self._priorGPU = gpuarray.to_gpu_async(toGPUData(self._prior),stream=self._streams[0])
        else:
            aP = np.array(aP).squeeze()
            if aP.size == 1:
                self.prior = np.hstack( [self.prior.squeeze()]+aP )
            elif aP.size == len(self._gaussianList):
                self.prior = aP
            else:
                assert 0, 'wrong prior size'
        self._gaussianList[-1].doCond = self._doCond
        if mainUseGPU_:
            self._streams[0].synchronize()
        return 0

    @property
    def nK(self):
        return len(self._gaussianList)
    @nK.setter
    def nK(self,*args,**kwargs):
        print("nVarI can not be set")

    @property
    def nVarI(self):
        return self._gaussianList[0].nVarI
    @nVarI.setter
    def nVarI(self,*args,**kwargs):
        print("nVarI can not be set")
    @property
    def nVarD(self):
        return self._gaussianList[0].nVarTot-self._gaussianList[0].nVarI
    @nVarD.setter
    def nVarD(self,*args,**kwargs):
        print("nVarD can not be set")
    @property
    def nVarTot(self):
        return self._gaussianList[0].nVarTot
    @nVarTot.setter
    def nVarTot(self,*args,**kwargs):
        print("nVarTot can not be set")
        
    @property
    def doCond(self):
        return dp(self._doCond)
    @doCond.setter
    def doCond(self, newDoCond):
        self._doCond = bool(newDoCond)
        for aGaussian in self._gaussianList:
            aGaussian.doCond = self._doCond
        return 0
    @property
    def prior(self):
        return self._prior.copy()
    @prior.setter
    def prior(self, newPrior):
        assert (newPrior.size == self._prior.size) or (newPrior.size == self.nK)
        if newPrior.size:
            newPrior = na(newPrior)
            newPrior.resize((self.nK,1))
            newPrior /= sum(newPrior)
            self._prior = newPrior
            self._lnprioir = np.log(self._prior)
            if mainUseGPU_:
                self._lnprioirGPU = toGPUData(self._lnprioir)
                if newPrior.shape == self._priorGPU.shape:
                    self._priorGPU.set_async(toGPUData(self._prior),stream=self._streams[0])
                else:
                    self._priorGPU = gpuarray.to_gpu_async(toGPUData(self._prior),stream=self._streams[0])
                self._streams[0].synchronize()
        return 0
    @property
    def mu(self):
        return np.hstack( [aGaus._mu for aGaus in self._gaussianList] )
    @mu.setter
    def mu(self, newMu:Union[List[np.ndarray], np.ndarray]):
        if isinstance(newMu, (list, tuple)):
            for aGaus, aMu in zip(self._gaussianList, newMu):
                aGaus.mu = aMu
        else:
            for k,aGaus in self.enum():
                aGaus.mu = newMu[:,[k]]
        return 0
    ########################################################################################
    def __copy__(self):
        return self.__deepcopy__({})
    def __deepcopy__(self, memodict={}):
        new = self.__class__()
        for k,aGaussian in self.enum():
            new.addKernel( aGaussian.__deepcopy__(memodict) )
        new.doCond = self.doCond
        if self.prior.size:
            new.prior = self.prior
        else:
            new.prior = np.zeros((0,))
        return new
    ########################################################################################
    #Convinience not necessarily fast
    def __getitem__(self, item):
        return self._gaussianList[item]
    def enum(self):
        return enumerate(self._gaussianList)

    ########################################################################################
    def toPars(self):
        return {"doCond":self.doCond, "prior":self.prior, "kernelList":lmap( lambda aGaussian : aGaussian.toPars(), self._gaussianList )}
    def loadPars(self, pars):
        self.doCond = pars["doCond"]
        self._gaussianList = []
        for aKernelPars in pars["kernelList"]:
            self.addKernel( gaussianKernel(nVarI=aKernelPars) )
        self.prior = pars["prior"]
        return 0

    #These functions should be parallelized, especially the gpu version
    def _evalMapCPU(self, x, weights=None, yOut=None, scaled=True, returnWeights=False):
        if self._prior.size == 0:
            #all kernel equal
            self.prior = ones((self.nK, 1))
        #No check-up is performed regarding consistency among gaussians and their definitions
        nPt = x.shape[1]
        nX = x.shape[0]
        nY = self[0].nVarTot-self[0].nVarI
        assert self[0].nVarI == nX,"Input has wrong dimension"

        if yOut is None:
            yOut = zeros((nY,x.shape[1]))
        else:
            assert yOut.shape == (nY,x.shape[1]),'Output has wrong dimension'
            yOut[:] = 0.

        # First get the weights
        if weights is None:
            weights = self._getWeightsCPU(x, scaled=scaled)
        else:
            assert weights.shape == (self.nK,nPt)
        
        #Now get the map estimates for each kernel, scale them and add to y
        yCurrent = empty(yOut.shape)
        xCopy = empty(x.shape)
        for k, aGaus in self.enum():
            np.copyto(xCopy,x)  # Do copy to allocated space, avoiding reallocation
            #_evalMapCPU(self, x, y=None, cpy=True):
            aGaus._evalMapCPU(xCopy, y=yCurrent, cpy=False)
            #Scale and add
            np.multiply(yCurrent,weights[[k],:], out=yCurrent)
            yOut += yCurrent
        #Done
        if returnWeights:
            return yOut, weights
        else:
            return yOut

    def _evalMapGPU(self, x, weights=None, yOutCPU=None, scaled=True, returnWeights=False):
        assert mainUseGPU_, 'GPU deactivated'
        if self._prior.size == 0:
            #all kernel equal
            self.prior = ones((self.nK, 1))
        
        nPt = x.shape[1]
        nX = x.shape[0]
        nY = self[0].nVarTot-self._gaussianList[0].nVarI
        assert self[0].nVarI == nX,"Input has wrong dimension"
        assert (yOutCPU is None) or (yOutCPU.shape == (nY,nPt)), "CPU output has to be none or have correct shape"
        
        assert isinstance(x, (np.ndarray, gpuarray.GPUArray))
        if isinstance(x, np.ndarray):
            xGPU = gpuarray.to_gpu_async(toGPUData(x), stream=self._streams[0])
        else:
            xGPU = x
        self._streams[0].synchronize()

        #First get the weights
        if weights is None:
            weights = self._getWeightsGPU(x, scaled=scaled)
        elif isinstance(weights, np.ndarray):
            weights = gpuarray.to_gpu(toGPUData(weights))
        else:
            assert isinstance(weights, gpuarray.GPUArray) and weights.shape == (self.nK, nPt)

        #Compute the y value for each kernel and its influence
        yGPUOut = gpuarray.zeros((nY, nPt), dtype=gpuFloat)
        yGPUCurrent = [None for k in range(self.nK)]
        xGPUCopy = gpuarray.empty_like(xGPU)
        for k, aGaus in self.enum():
            # Copy the fresh x data to copy
            cuda.memcpy_dtod_async(xGPUCopy.ptr,xGPU.ptr,xGPU.nbytes,stream=aGaus._streams[0])
            #_evalMapGPU(self, x, yCPU=None, cpy=True):
            yGPUCurrent[k] = aGaus._evalMapGPU(xGPUCopy, yCPU=None, cpy=False)
            #Till here it is async, rest needs to be changed for performance
            yGPUCurrent[k] = cudaMisc.multiply(yGPUCurrent[k], weights[k,:]) # !!TBDGPUEFF
        #Once parallel is done sum
        for ayOut in yGPUCurrent:
            yGPUOut += ayOut
        
        if yOutCPU is not None:
            getNStore(yGPUOut, yOutCPU)

        if returnWeights:
            return yGPUOut, weights
        else:
            return yGPUOut
    
    #Generic call
    def evalMap(self, x, yOut=None, cpy=True, weights=None, scaled=True, returnWeights=False):
        #Evaluate the map estimator for the gmm
        if yOut is None:
            yOut = zeros((self[0]._nVarTot-self[0]._nVarI,x.shape[1]))
        else:
            assert yOut.shape == (self[0]._nVarTot-self[0]._nVarI,x.shape[1]),'Wrong output shape'

        if returnWeights:
            if mainUseGPU_ and ((x.shape[1] > nMinGPU) or isinstance(x, gpuarray.GPUArray)):
                _, weightsOut = self._evalMapGPU(x,yOutCPU=yOut,weights=weights, scaled=scaled, returnWeights=True)
            else:
                _,weightsOut = self._evalMapCPU(x,yOut=yOut,weights=weights, scaled=scaled, returnWeights=True)
            return yOut, weightsOut
        else:
            if mainUseGPU_ and ((x.shape[1] > nMinGPU) or isinstance(x, gpuarray.GPUArray)):
                self._evalMapGPU(x,yOutCPU=yOut,weights=weights, scaled=scaled)
            else:
                self._evalMapCPU(x,yOut=yOut,weights=weights, scaled=scaled)
            return yOut
    
    #################################
    def _getLogProbCPU(self,x,out=None,reduce=False,kd=True):
        # Compute loglikelihood of all points in x given the GMM
        # if reduce, it will be summed over kernels
        nPt = x.shape[1]
        if out is not None:
            if reduce:
                assert out.size==nPt
                kd = out.shape == (1,nPt)
                outT = empty((self.nK,x.shape[1]))
            else:
                assert out.shape == (self.nK,nPt)
                outT = out
        else:
            outT = empty((self.nK,x.shape[1]))
            if reduce:
                if kd:
                    out = empty((1,nPt))
                else:
                    out = empty((nPt,))
            else:
                out = outT


        xTemp = empty(x.shape)
        for k, aGaus in self.enum():
            np.copyto(xTemp,x)
            aGaus._getLogProbCPU(xTemp, out=outT[k,:], cpy=False, kd=False, lnprioir=self._lnprioir[k])

        if reduce:
            #sum(outT, axis=0, out=out, keepdims=kd)
            logsumexpCPU(outT, out=out, cpy=False, kd=kd)

        return out
    ####################
    def _getLogProbGPU(self,x,outGPU=None,outCPU=None,reduce=False,kd=True):
        assert mainUseGPU_, 'GPU deactivated'
        assert isinstance(x, (np.ndarray, gpuarray.GPUArray))
        nPt = x.shape[1]
        if isinstance(x, np.ndarray):
            x = gpuarray.to_gpu_async(toGPUData(x), stream=self._streams[0])

        if outGPU is not None:
            if reduce:
                assert outGPU.size==nPt
                kd = outGPU.shape == (1,nPt)
                outTGPU = gpuarray.empty((self.nK,x.shape[1]), dtype=gpuFloat)
            else:
                assert outGPU.shape == (self.nK,nPt)
                out = outTGPU = outGPU
        else:
            outTGPU = gpuarray.empty((self.nK,x.shape[1]), dtype=gpuFloat)
            if reduce:
                if kd:
                    outGPU = gpuarray.empty((1,nPt),dtype=gpuFloat)
                else:
                    outGPU = gpuarray.empty((nPt,),dtype=gpuFloat)
            else:
                outGPU = outTGPU

        for k,aGaus in self.enum():
            aGaus._getLogProbGPU(x,outGPU=outTGPU[k,:],cpy=True,kd=False, lnprioir=self._lnprioir[k]) #This is ineffective since no streamed minus

        if reduce:
            #cudaSum(outTGPU, axis=0, out=outGPU, keepdims=kd)
            logsumexpGPU(outTGPU, out=outGPU, cpy=False, kd=kd)

        if outCPU is not None:
            getNStore(outGPU, outCPU)

        return outGPU

    ########
    #Generic call
    def getLogProb(self, x, out=None, kd=True, reduce=False):

        if out is None:
            if reduce is False:
                out = empty((self.nK, x.shape[1]))
            else:
                if kd:
                    out=empty((1,x.shape[1]))
                else:
                    out = empty((x.shape[1],1))

        if mainUseGPU_ and ((x.shape[1] > nMinGPU) or isinstance(x, gpuarray.GPUArray)):
            self._getLogProbGPU(x, outCPU=out, reduce=reduce, kd=kd)
        else:
            self._getLogProbCPU(x, out=out, reduce=reduce, kd=kd)

        return out

    #################################
    def _getLogLikelihoodCPU(self,x):
        if not isinstance(x, np.ndarray):
            # Implies gpu array
            x = x.get()
        meanLogLike = float(np.mean(self._getLogProbCPU(x,reduce=True)))

        if not np.isfinite(meanLogLike):
            warnings.warn("loglike is not finite")
        
        return meanLogLike
    #################################
    def _getLogLikelihoodGPU(self,x):
        assert mainUseGPU_, 'GPU deactivated'

        if isinstance(x, np.ndarray):
            x = gpuarray.to_gpu(x)
        meanLogLike = float(cudaMisc.mean(self._getLogProbGPU(x,reduce=True)).get())
        
        if not np.isfinite(meanLogLike):
            warnings.warn("loglike is not finite")

        return meanLogLike
        
    
    #################################
    def getLogLikelihood(self,x):
        if mainUseGPU_ and ((x.shape[1] > nMinGPU) or isinstance(x, gpuarray.GPUArray)):
            #meanLogLike = float(cudaMisc.mean(self._getLogProbGPU(x,reduce=True)).get())
            meanLogLike = self._getLogLikelihoodGPU(x)
        else:
            #meanLogLike = float(np.mean(self._getLogProbCPU(x,reduce=True)))
            meanLogLike = self._getLogLikelihoodCPU(x)
        
        return meanLogLike
    #################################
    def _getWeightsGPU(self, x, weightsOutGPU=None, weightsOutCPU=None, scaled = True):
        assert mainUseGPU_, 'GPU deactivated'

        nPt = x.shape[1]
        
        assert (weightsOutCPU is None) or (weightsOutCPU.shape == (self.nK,nPt)),"CPU output has to be none or have correct shape"
    
        assert isinstance(x,(np.ndarray,gpuarray.GPUArray))
        
        if weightsOutGPU is None:
            weightsOutGPU = gpuarray.empty((self.nK, nPt), dtype=gpuFloat)
        else:
            assert weightsOutGPU.shape == (self.nK, nPt)

        #Get logproba
        self._getLogProbGPU(x, outGPU=weightsOutGPU, reduce=False)
        #Take exp
        cudaMath.exp(weightsOutGPU, out=weightsOutGPU)
        #Apply prior
        #weights = cudaMisc.multiply(self._priorGPU, weights)
        #Prior applied in logproba now

        # Ensure numerical stability
        weightsOutGPU += epsGPU
        if scaled:
            weightsOutGPU = cudaMisc.divide(weightsOutGPU, cudaSum(weightsOutGPU, 0, keepdims=True))#Mhh does as copy...
        
        if weightsOutCPU is not None:
            getNStore(weightsOutGPU, weightsOutCPU)
        
        return weightsOutGPU
    
    def _getWeightsCPU(self, x, weights=None, scaled=True):
        # No check-up is performed regarding consistency among gaussians and their definitions
        nPt = x.shape[1]

        if weights is None:
            weights = zeros((self.nK,x.shape[1]))
        else:
            assert weights.shape == (self.nK,x.shape[1]),'Output has wrong dimension'
    
        # First get the logPob
        self._getLogProbCPU(x,out=weights,reduce=False)
        #Take exp
        np.exp(weights, out=weights)

        # Scale with prior
        # np.multiply(weights,self._prior,out=weights)
        # Prior applied in logproba now

        # Scale the weights to have equal 1 per point
        # Always regularize
        weights += epsCPU
        if scaled:
            np.divide(weights,np.sum(weights,0,keepdims=True),out=weights)

        return weights
    
    def getWeights(self, x, weights=None, scaled=True):
        
        assert (weights is None) or (weights.shape == (self.nK, x.shape[1]))
        if weights is None:
            weights = empty((len(self._gaussianList), x.shape[1]))
        
        if mainUseGPU_ and (isinstance(x, gpuarray.GPUArray) or (x.shape[1]>nMinGPU)):
            self._getWeightsGPU(x, weightsOutCPU=weights, scaled=scaled)
        else:
            self._getWeightsCPU(x, weights=weights, scaled=scaled)
        
        return weights

    ###############################################
    def simpleEval(self,x,yOut=None):
        # This functions simply computes the weighted sum of means
        if isinstance(x,gpuarray.GPUArray):
            yOutGPU = gpuarray.zeros((self._gaussianList[0]._muy.size, x.shape[1]), dtype=gpuFloat)
            thisMu = gpuarray.empty((self._gaussianList[0]._muy.size, x.shape[1]), dtype=gpuFloat)
            allWeights = self._getWeightsGPU(x)
            for k,aGaussian in enumerate(self._gaussianList):
                #Multiply mu and weight
                #Here dot is the outerproduct
                cudaDot(aGaussian._muy_gpu, allWeights[[k],:], thisMu)
                yOutGPU += thisMu
            
            if yOut is not None:
                assert yOut.shape == yOutGPU.shape
                getNStore(yOutGPU, yOut)
            return yOutGPU
        else:
            yOut = zeros((self._gaussianList[0]._muy.size, x.shape[1])) if yOut is None else yOut
            assert yOut.shape == (self._gaussianList[0]._muy.size, x.shape[1])
            thisMu = empty(yOut.shape)
            allWeights = self._getWeightsCPU(x)
            for k,aGaussian in enumerate(self._gaussianList):
                dot(aGaussian._muy, allWeights[[k],:], out=thisMu)
                yOut += thisMu
            return yOut
################################################

def EMAlgorithmGPU(GMM,x,add2Sigma=1e-6, iterMax=1000, relTol=1e-3, absTol=1e-3, doPlot=False):
    assert mainUseGPU_, 'GPU deactivated'
    nK = GMM.nK
    dim, nPt = x.shape
    if isinstance(add2Sigma, float) or (isinstance(add2Sigma, np.ndarray) and add2Sigma.size==1):
        add2Sigma = add2Sigma*Id(dim)
    else:
        assert add2Sigma.shape == (dim,dim)
        try:
            assert np.all( add2Sigma==add2Sigma.T )
            chol(add2Sigma)
        except:
            assert 0, "Regularization covariance is not symmetric positive"

    thisIter = 0
    weights = gpuarray.empty((nK,nPt), dtype=gpuFloat)
    sumWeights = gpuarray.empty((nPt,), dtype=gpuFloat)
    rootweights = gpuarray.empty((nK,nPt), dtype=gpuFloat)
    scaledPrior = gpuarray.empty((nK,1), dtype=gpuFloat)
    lastLogLike = -1e20
    newLogLike = -1e19

    newSigma = []
    for k in range(nK):
        newSigma.append( gpuarray.empty((dim,dim), dtype=gpuFloat) )

    if doPlot:
        allLogLike = empty((iterMax,))
        fig,ax = pu.plt.subplots(1,1)

    while (thisIter < iterMax) and ((newLogLike-lastLogLike) > relTol) and ((newLogLike-lastLogLike) > absTol):

        GMM._getWeightsGPU(x,weightsOutGPU=weights,scaled=False)
        #Compute new overall likelihood
        
        cudaSum(weights,axis=0,out=sumWeights,keepdims=False)
        cudaMath.log(sumWeights, out=sumWeights)
        lastLogLike = newLogLike
        newLogLike = float(cudaMisc.mean(sumWeights).get())#GMM.getLogLikelihood(x)

        # Now scale so that each point has summed weight 1
        # avoid numeric problems
        weights += epsGPU
        cudaSum(weights,axis=0,out=sumWeights,keepdims=False)
        cudaMisc.div_matvec(weights, sumWeights, axis=1, out=weights)#weights = cudaMisc.divide(weights,sumWeights,axis=1)

        # Get the new prior
        cudaSum(weights,axis=1,keepdims=True,out=scaledPrior)
        scaledPriorCPU = toCPUData(scaledPrior.get()).squeeze()
        GMM.prior = scaledPriorCPU#toCPUData(scaledPrior.get())  # Scaled internally
        

        # Square root of weights
        cudaMath.sqrt(weights,out=rootweights)
        #Compute the new means, parallelize covariance computations
        #This could run entirely in parallel however many functions do not propose a streamed async method
        for k,aGaussian in GMM.enum():
            # Get the new mean
            aGaussian.mu = toCPUData(cudaSum(cudaMisc.multiply(x,weights[k,:]),axis=1,keepdims=True).get())/scaledPriorCPU[k]
            # Get new covariance
            myCov(cudaMisc.add_matvec(x, -aGaussian._mu_gpu, axis=0, stream=aGaussian._streams[0]),w=rootweights[k,:],N=scaledPriorCPU[k],out=newSigma[k],isRoot=True,isCentered=True,cpy=False,streams=aGaussian._streams,doSync=False)

        for k,aGaussian in GMM.enum():
            # Get new covariance
            aGaussian._streams[0].synchronize()
            aGaussian._streams[1].synchronize()
            aGaussian.Sigma = toCPUData(newSigma[k].get()) + add2Sigma

        if doPlot:
            allLogLike[thisIter] = newLogLike

        thisIter += 1

    if doPlot:
        ax.plot(allLogLike, '.-')
        return newLogLike, allLogLike
    else:
        return newLogLike


def EMAlgorithmCPU(GMM,x,add2Sigma=1e-6, iterMax=1000, relTol=1e-3, absTol=1e-3, doPlot=False, otherOpts={}):
    
    otherOptsBase={'fixZeroKernel':False}
    otherOptsBase.update(otherOpts)

    nK = GMM.nK
    dim, nPt = x.shape
    if isinstance(add2Sigma, float) or (isinstance(add2Sigma, np.ndarray) and add2Sigma.size==1):
        add2Sigma = add2Sigma*Id(dim)
    else:
        assert add2Sigma.shape == (dim,dim)
        try:
            assert np.all( add2Sigma==add2Sigma.T )
            chol(add2Sigma)
        except:
            assert 0, "Regularization covariance is not symmetric positive"

    thisIter = 0
    weights = empty((nK,nPt))
    sumWeights = empty((1,nPt))
    rootWeights = empty((nK,nPt))
    scaledPrior = empty((nK,1))
    lastLogLike = -1e20
    newLogLike = -1e19
    
    if doPlot:
        allLogLike = empty((iterMax,))
        fig,ax = pu.plt.subplots(1,1)

    while (thisIter < iterMax) and ((newLogLike-lastLogLike) > relTol) and ((newLogLike-lastLogLike) > absTol):

        GMM._getWeightsCPU(x,weights=weights,scaled=False)
        
        #Compute new oveall likelihood
        lastLogLike = newLogLike
        try:
            newLogLike = np.mean(log(sum(weights, axis=0)))#GMM.getLogLikelihood(x)
        except Exception as e:
            print(e)
            assert 0, 'failed'
        if newLogLike < lastLogLike:
            print('EM step did not converge new \n {0:.16e} \n old \n {1:.16e} \n diff \n {2:.16e}'.format(newLogLike, lastLogLike,newLogLike-lastLogLike))
            if doPlot:
                ax.plot([thisIter], [newLogLike], 'sr')
            warnings.warn('EM step did not converge')
        
        #Now scale so that each point has summed weight 1
        #avoid numeric problems
        weights += epsCPU
        sum(weights, axis=0, keepdims=True,out=sumWeights)
        divide(weights, sumWeights, out=weights)
        
        # Get the new prior
        sum(weights,axis=1,keepdims=True,out=scaledPrior)
        GMM.prior = scaledPrior  # Scaled internally

        # Square root of weights
        sqrt(weights,out=rootWeights)
        for k,aGaussian in GMM.enum():
            # Get the new mean
            aGaussian.mu = sum(np.multiply(x,weights[[k],:]),1,keepdims=True)/scaledPrior[k]
            # Get new covariance
            aGaussian.Sigma = myCov(x-aGaussian._mu,w=rootWeights[[k],:],N=scaledPrior[k],isRoot=True,isCentered=True,cpy=False)+add2Sigma
            
            if (k==GMM.nK-1) and (otherOptsBase['fixZeroKernel']):
                # Fix the last kernel onto zero and ensure diag negative behaviour
                aGaussian.mu,aGaussian.Sigma = adjustZeroKernel(aGaussian.nVarI,aGaussian.mu,aGaussian.Sigma)

        if doPlot:
            allLogLike[thisIter] = newLogLike

        thisIter += 1

    if doPlot:
        ax.plot(range(thisIter), allLogLike[:thisIter], '.-')
        return newLogLike, allLogLike
    else:
        return newLogLike

def EMAlgorithm(GMM,x,add2Sigma=1.e-6,doInit="KMeans", iterMax=100, relTol=1e-3, absTol=1e-3, doPlot=False):

    #Regular expectation maximization algorithm
    assert isinstance(GMM,GaussianMixtureModel)
    nK = GMM.nK
    nPt = x.shape[1]
    assert doInit in ('warm_start', 'KMeans')

    isGPU = isinstance(x, gpuarray.GPUArray)

    if doInit == "KMeans":
        from sklearn.cluster import KMeans

        if isGPU:
            xInit = toCPUData(x.get()).T
            isGPU = True
        else:
            xInit = x.T

        initGMM = KMeans(nK,precompute_distances=True)
        initGMM.fit_predict(xInit)
        del xInit
        # Get the information
        thisPrior = zeros((nK,1))
        for k,aGaussian in enumerate(GMM._gaussianList):
            # Get the mean
            aGaussian.mu = initGMM.cluster_centers_[[k],:].T
            # Estimate the covariance
            thisInd = initGMM.labels_ == k
            thisPrior[k] = sum(thisInd)/nPt
            # thisDx = x[:,thisInd]-aGaussian.mu
            # aGaussian.Sigma = 1./float(thisDx.shape[1])*dot(thisDx, thisDx.T)
            if isGPU:
                #GPU version
                aGaussian.Sigma = toCPUData(myCov(x-aGaussian._mu_gpu, w=toGPUData(thisInd), isRoot=True, isCentered=True, cpy=False, streams=aGaussian._streams).get())
            else:
                aGaussian.Sigma = myCov(x[:,thisInd]-aGaussian.mu,isCentered=True,cpy=False)  # cov(x[:,thisInd]-aGaussian.mu)

        GMM.prior = thisPrior

    if mainUseGPU_ and (isGPU or x.shape[1] > nMinGPU):
        x = gpuarray.to_gpu(toGPUData(x)) if isinstance(x, np.ndarray) else x
        return EMAlgorithmGPU(GMM, x, add2Sigma=add2Sigma, iterMax=iterMax, relTol=relTol, absTol=absTol, doPlot=doPlot)
    else:
        return EMAlgorithmCPU(GMM,x,add2Sigma=add2Sigma,iterMax=iterMax,relTol=relTol,absTol=absTol, doPlot=doPlot)

########################################################################################################################

def modifiedEMAlgorithmCPU(GMM:GaussianMixtureModel,x,y,add2Sigma=0.,doInit="KMeans", thetaInit=None, iterMax=500, regVal = 1e-5, relTolLogLike=1e-3, absTolLogLike=1e-3, relTolMSE=1e-3, absTolMSE=1e-3, mseConverged=0., initStepSizeJac=0.005, convBounds=[0.2,0.1], regValStep=[0.1, 4], doPlot=1, plotCallBack=None, thetaOpt=None, nsProjector=None, JacSpace=None):
    
    nK = GMM.nK
    nPt = x.shape[1]
    
    nVarI = GMM[0].nVarI
    nVarTot = GMM[0].nVarTot
    nVarD = nVarTot-nVarI
    
    assert (nVarI == GMM[0]._nVarI) and (nVarTot==GMM[0]._nVarTot), "Data not consistent with kernel sizes"
    assert (thetaOpt is None) or (thetaOpt.shape == (nVarD, nPt))
    
    if isinstance(regValStep, (float, int)):
        regValStep = [float(regValStep), min(20.*float(regValStep), 1.)]
    try:
        dummy = regValStep[0]
        dummy = regValStep[1]
        del dummy
    except:
        print("regValStep could not be handled")
        
    
    if (doInit is not "warm_start") and not callable(doInit):
        if thetaInit is None:
            # Do initial opt
            # initOptimizeParsCVX(self,x,vReal,initVars=None,regVal=1e-2)
            thetaInit = GMM.initOptimize(x,y,regVal=regVal)
        else:
            thetaInit = thetaInit if isinstance(thetaInit,np.ndarray) else toCPUData(thetaInit.get())
    elif callable(doInit):
        thetaInit = doInit(x,y)
    elif thetaInit:
        thetaInit = GMM._evalMapCPU(x)
        
    # Augment space
    xTilde = np.vstack((x,thetaInit))
    
    # The initialization does not take into account the weights
    # todo
    
    if doInit == "KMeans":
        from sklearn.cluster import KMeans
        
        initGMM = KMeans(nK,precompute_distances=True)
        initGMM.fit_predict(xTilde.T)
        # Get the information
        thisPrior = zeros((nK,1))
        for k,aGaussian in GMM.enum():
            # Get the mean
            aGaussian.mu = initGMM.cluster_centers_[[k],:].T
            # Estimate the covariance
            thisInd = initGMM.labels_ == k
            thisPrior[k] = sum(thisInd)/nPt
            # thisDx = x[:,thisInd]-aGaussian.mu
            # aGaussian.Sigma = 1./float(thisDx.shape[1])*dot(thisDx, thisDx.T)
            aGaussian.Sigma = myCov(xTilde[:,thisInd]-aGaussian.mu,isCentered=True, cpy=False)  # cov(x[:,thisInd]-aGaussian.mu)
        GMM.prior = thisPrior
        #Use this GMM to predict y
        GMM._evalMapCPU(x, yOut=xTilde[nVarI:,:])
    elif doInit == "EM":
        from sklearn.mixture import GaussianMixture
        thisRegCovar = add2Sigma if isinstance(add2Sigma, float) else float(add2Sigma[0,0])
        initGMM = GaussianMixture(n_components=GMM.nK, reg_covar=thisRegCovar)
        initGMM.fit(xTilde.T)
        for k, aGaussian in GMM.enum():
            aGaussian.mu = initGMM.means_[[k],:].T
            aGaussian.Sigma = initGMM.covariances_[k,:,:]
        GMM.prior = initGMM.weights_
        #evalSc = GMM._evalMapCPU(x, scaled=True)
        #evalUsc = GMM._evalMapCPU(x,scaled=False)
        #vSc = GMM._evalDynCPU(x, scaled=True)
        #vUsc = GMM._evalDynCPU(x, scaled=False)
        #xTilde1 = xTilde.copy();xTilde2 = xTilde.copy()
        #v1 = np.random.rand(*y.shape);v2 = np.random.rand(*y.shape)
        #GMM._evalMapCPU(x,yOut=xTilde1[nVarI:,:],scaled=True)
        #GMM._evalMapCPU(x,yOut=xTilde2[nVarI:,:],scaled=False)
        #GMM._evalDynCPU(x,v=v1,scaled=True)
        #GMM._evalDynCPU(x,v=v2,scaled=False)
        
        GMM._evalMapCPU(x,yOut=xTilde[nVarI:,:])
    elif doInit == "continue":
        pass
    
    thisIter = 0
    weights = empty((nK,nPt))
    sumWeights = empty((1,nPt))
    rootWeights = empty((nK,nPt))
    scaledPrior = empty((nK,1))
    lastLogLike = -1e20
    newLogLike = GMM.getLogLikelihood(xTilde)
    lastMSE = 1e20
    newMSE = GMM.getMSE(x,y)
    oldMu = []
    for _, aGaussian in GMM.enum():
        oldMu.append( aGaussian.mu )
    
    if doPlot:
        allMSE = empty((iterMax,))
        allLogLike = empty((iterMax,))
        # Init plots
        aang = np.linspace(0,2*pi,nPt)
        ffig,aax,lline0 = pu.errorPlot(GMM.getErr(x,y, JacSpace=JacSpace),ang=aang,color='k')
        _,aax,lline1 = pu.errorPlot(GMM.getErr(x,y, JacSpace=JacSpace),ang=aang,fig=ffig,ax=aax,color='r')
        ffig1,aax1 = pu.plt.subplots(2,1,sharex=True)

        aax1[0].plot([thisIter],[GMM.getLogLikelihood(xTilde)],'.b')
        aax1[1].plot([thisIter],[GMM.getMSE(x,y, JacSpace=JacSpace)],'.b')
        aax1[1].plot([thisIter],[GMM.getMSE(x,y, JacSpace=JacSpace,regVal=regVal)],'+b')
        aax1[1].plot([thisIter],[GMM.getMSE(x,y, JacSpace=JacSpace,regVal=0)],'+r')
    
    # We accept that the either -loglike or mse increases if the other decreases.
    # We will internally adapt the stepsize in order to "ensure" convergence
    jacFacOld = initStepSizeJac

    while (thisIter < iterMax) and \
            ( ((newLogLike-lastLogLike)/lastLogLike > relTolLogLike and (newLogLike-lastLogLike) > absTolLogLike) or \
                (((lastMSE-newMSE) > absTolMSE) and (lastMSE-newMSE)/lastMSE > relTolMSE) ):
        
        print("Iteration {0} of modified EM".format(thisIter))
        # debug
        GMMold = GMM.__copy__()
        if ((newMSE < mseConverged) and bool(mseConverged)):
            break
        
        if doPlot:
            # Update intermediate
            pu.errorPlot(GMM.getErr(x,y),ang=aang,fig=ffig,ax=aax,thisLine=lline1,color='r')
        if callable(plotCallBack):
            plotCallBack(GMM, x, xTilde, y)

        if 0:
            #First fo regular EM
            GMM._getWeightsCPU(xTilde,weights=weights,scaled=False)
            
            lastLogLike = newLogLike
            lastMSE = newMSE
            newLogLike = GMM.getLogLikelihood(xTilde) # Error too large #np.mean(log(sum(weights, axis=0)))# <-> GMM.getLogLikelihood(xTilde)
            if doPlot:
                newMSEplot = GMM.getMSE(x,y,regVal=regVal, JacSpace=JacSpace)
            # Now scale so that each point has summed weight 1
            # avoid numeric problems
            weights += epsCPU
            sum(weights,axis=0,keepdims=True,out=sumWeights)
            divide(weights,sumWeights,out=weights)
            # Get the new prior
            sum(weights,axis=1,keepdims=True,out=scaledPrior)
        else:
            # Not that efficient
            GMM._getWeightsCPU(xTilde,weights=weights,scaled=True)

            lastLogLike = newLogLike
            lastMSE = newMSE
            if doPlot:
                newMSEplot = GMM.getMSE(x,y,regVal=regVal,JacSpace=JacSpace)
            
            # Get the new prior
            sum(weights,axis=1,keepdims=True,out=scaledPrior)
        
        GMM.prior = scaledPrior  # Scaled internally

        # Square root of weights
        sqrt(weights,out=rootWeights)
        for k,aGaussian in GMM.enum():
            # Get the new mean
            aGaussian.mu = sum(np.multiply(xTilde,weights[[k],:]),1,keepdims=True)/scaledPrior[k]
            # Get new covariance
            try:
                aGaussian.Sigma = myCov(xTilde-aGaussian._mu,w=rootWeights[[k],:],N=scaledPrior[k],isRoot=True,isCentered=True,cpy=False)+add2Sigma
            except:
                #print("Cov not sym pos")
                print(rootWeights[[k],:].T)
                print("sum of weights: {0}".format(np.sum(rootWeights[k,:])))
                print(myCov(xTilde-aGaussian._mu,w=rootWeights[[k],:],N=scaledPrior[k],isRoot=True,isCentered=True,cpy=False))
                assert 0, "Cov not sym pos"

        newLogLike = GMM.getLogLikelihood(xTilde)  # Error too large #np.mean(log(sum(weights, axis=0)))# <-> GMM.getLogLikelihood(xTilde)
        # debug check up
        if __debug:
            if GMM.getLogLikelihood(xTilde)<=GMMold.getLogLikelihood(xTilde):
                print('no improvement')

        # Optimization step
        jacScaled = zeros((nVarTot,nK))
        if 1:
            # This might be done more efficiently
            if newLogLike<=lastLogLike:
                print('Delta loglike is {0}'.format(newLogLike-lastLogLike))
                warnings.warn('Non convergent step; Skipping modification')
            else:
                deltaLogLike = newLogLike-lastLogLike
                minLogLike = lastLogLike + (1.-convBounds[0])*deltaLogLike
                GMM._getWeightsCPU(x,weights=weights,scaled=True)  # Get weights in x (NOT xTilde)
                # Save current mu
                oldMu = GMM.mu
                # Get Current MSE and Jac
                newMSE,jacTilde = GMM._getMSEnJacCPU(x,y,weights,regVal=regVal,JacSpace=JacSpace)  # newMSEInter,jac = GMM._getMSEnJacCPU(x,y,weights,thetaCurrentAll=xTilde[nVarI:,:],regVal=regVal)
                jac = np.zeros_like(oldMu)
                jac[nVarI:,:] = jacTilde
                
                
                minMSE = newMSE
                jacFac = newMSE/20.
                jacFacOld = jacFac
                #Test this step
                GMM.mu = oldMu-jacFac*jac
                cMSE = GMM.getMSE(x,y,regVal=regVal,weights=weights, JacSpace=JacSpace)
                cLogLike = GMM.getLogLikelihood(xTilde)
                
                #Check
                if ((cMSE<minMSE) and (cLogLike>minLogLike)):
                    # For this step, mse and loglike converge, so we can try to increase it
                    while True:
                        print("increase step")
                        jacFacOld = jacFac
                        jacFac = jacFac*1.5 # Increase
                        GMM.mu = oldMu-jacFac*jac
                        cMSE = GMM.getMSE(x,y,regVal=regVal,weights=weights, JacSpace=JacSpace)
                        cLogLike = GMM.getLogLikelihood(xTilde)
                        
                        if ((cMSE<minMSE) and (cLogLike>minLogLike)):
                            minMSE = cMSE
                        else:
                            break
                else:
                    #Either mse or loglike does not converge -> decrease step
                    while True:
                        print("decrease step")
                        jacFacOld = jacFac
                        jacFac = jacFac*0.75  # Increase
                        if jacFacOld < 1e-4:
                            warnings.warn("Setting very small mu-update to zero")
                            jacFac=jacFacOld=0.
                            break
                        GMM.mu = oldMu-jacFac*jac
                        cMSE = GMM.getMSE(x,y,regVal=regVal,weights=weights, JacSpace=JacSpace)
                        cLogLike = GMM.getLogLikelihood(xTilde)
    
                        if ((cMSE < minMSE) and (cLogLike > minLogLike)):
                            jacFacOld = jacFac
                            break
                
                # Set the final result
                GMM.mu = oldMu-jacFacOld*jac

        elif 0:
            if 0:
                #First get the upper bound for the loglike
                upperBoundLogLike = GMM.getLogLikelihood(xTilde) #np.mean(log(sum(weights, axis=0))) #This is the best we can achieve without changing the underlying data
                if ((newLogLike-upperBoundLogLike)/newLogLike > 5.*relTolLogLike) or ((upperBoundLogLike-newLogLike) > 5.*absTolLogLike):
                    # This is the case if the EM has not yet converged. In this case we target the mean between the last logLike and the upper bound
                    #delta = (upperBoundLogLike-newLogLike)/2.
                    #mean = (upperBoundLogLike+newLogLike)/2.
                    #targetLogLike = [mean-0.2*delta,mean+0.2*delta]
                    targetLogLike = [ upperBoundLogLike-convBounds[0]*(upperBoundLogLike-newLogLike), upperBoundLogLike-convBounds[1]*(upperBoundLogLike-newLogLike) ]
                else:
                    #This is the case if the em has already converged, but not the variable distribution
                    targetLogLike = [upperBoundLogLike - 0.1*abs(upperBoundLogLike) - 0.1, upperBoundLogLike - 0.01*abs(upperBoundLogLike) - 0.01]
                try:
                    assert targetLogLike[0]<targetLogLike[1]
                    print('aabbvv')
                except AssertionError:
                    print('noo')
                    pass
                # Update weights to take into account only x for map
                GMM._getWeightsCPU(x,weights=weights,scaled=True)#Get weights in x (NOT xTilde)

                # Get MSE and jac
                # def getMSEnJac(self,x,vReal,weights,thetaCurrentAll=None,vErr=None,regVal=1e-2):
                newMSE,jac = GMM._getMSEnJacCPU(x,y,weights,regVal=regVal, JacSpace=JacSpace)#newMSEInter,jac = GMM._getMSEnJacCPU(x,y,weights,thetaCurrentAll=xTilde[nVarI:,:],regVal=regVal)
                # Normalize jac
                maxJacFac = newMSE/20.
                jacFac = min(jacFacOld, maxJacFac)#min([(stepSizeJac/sqrt(sum(square(jac)))),newMSEInter/6.])
                # jacFac = stepSizeJac/sqrt(sum(square(jac)))
                # Store current mu
                for k, aGaussian in GMM.enum():
                    np.copyto(oldMu[k], aGaussian._mu, casting='no')
                #Test if last jacFac is ok
                jacScaled[nVarI:,:] = jacFac*jac
                for k,aGaussian in GMM.enum():
                    aGaussian.mu = oldMu[k]-jacScaled[:,[k]]  # Go into decreasing direction; Do not do deepest stew since this will disturb convergence of EM
                interMediateLogLike = GMM.getLogLikelihood(xTilde)
                isDone = False
                if (interMediateLogLike <= targetLogLike[1]) and (interMediateLogLike >= targetLogLike[0]):
                    # The old or maximal step was ok
                    isDone = True
                elif (interMediateLogLike >= targetLogLike[1]) and (jacFac == maxJacFac):
                    #The largest allowed step still yields good results
                    isDone = True
                elif (interMediateLogLike >= targetLogLike[1]) and (jacFac < maxJacFac):
                    # We can increase the step
                    while (interMediateLogLike >= targetLogLike[1]) and (jacFac < maxJacFac):
                        print("Modifying step since too large: {0}".format(jacFac))
                        lowerJacFac = jacFac
                        jacFac = min(2.*jacFac, maxJacFac)
                        jacScaled[nVarI:,:] = jacFac*jac
                        for k,aGaussian in GMM.enum():
                            aGaussian.mu = oldMu[k]-jacScaled[:,[k]]  # Go into decreasing direction; Do not do deepest stew since this will disturb convergence of EM
                        interMediateLogLike = GMM.getLogLikelihood(xTilde)
                    upperJacFac = jacFac
                    if jacFac >= maxJacFac:
                        #The largest step yields good results
                        jacFac = maxJacFac
                        isDone = True
                    else:
                        if (interMediateLogLike < targetLogLike[1]) and (interMediateLogLike > targetLogLike[0]):
                            #The last step was ok
                            isDone = True
                        else:
                            isDone = False
                elif (interMediateLogLike <= targetLogLike[0]):
                    # The current step is too large
                    while (interMediateLogLike <= targetLogLike[0]):
                        print("Modifying step since too small: {0}".format(jacFac))
                        upperJacFac = jacFac
                        jacFac /= 2.
                        jacScaled[nVarI:,:] = jacFac*jac
                        for k,aGaussian in GMM.enum():
                            aGaussian.mu = oldMu[k]-jacScaled[:,[k]]  # Go into decreasing direction; Do not do deepest stew since this will disturb convergence of EM
                        interMediateLogLike = GMM.getLogLikelihood(xTilde)
                    lowerJacFac = jacFac
                    if (interMediateLogLike <= targetLogLike[1]) and (interMediateLogLike >= targetLogLike[0]):
                        # The last step was ok
                        isDone = True
                    else:
                        isDone = False
                else:
                    assert 0, "The program should not have landed here"

                #if not done, test the boundaries first
                if not isDone:
                    jacFac = upperJacFac
                    jacScaled[nVarI:,:] = jacFac*jac
                    for k,aGaussian in GMM.enum():
                        aGaussian.mu = oldMu[k]-jacScaled[:,[k]]
                    interMediateLogLike = GMM.getLogLikelihood(xTilde)
                    if (interMediateLogLike <= targetLogLike[1]) and (interMediateLogLike >= targetLogLike[0]):
                        isDone = True
                if not isDone:
                    jacFac = lowerJacFac
                    jacScaled[nVarI:,:] = jacFac*jac
                    for k,aGaussian in GMM.enum():
                        aGaussian.mu = oldMu[k]-jacScaled[:,[k]]
                    interMediateLogLike = GMM.getLogLikelihood(xTilde)
                    if (interMediateLogLike <= targetLogLike[1]) and (interMediateLogLike >= targetLogLike[0]):
                        isDone = True

                # Do dichotomic search if necessary
                while not isDone:
                    print('Final dichotomic search')
                    #TBD check if deadlock is here
                    jacFac = (lowerJacFac+upperJacFac)/2.
                    jacScaled[nVarI:,:] = jacFac*jac
                    for k,aGaussian in GMM.enum():
                        aGaussian.mu = oldMu[k]-jacScaled[:,[k]]
                    interMediateLogLike = GMM.getLogLikelihood(xTilde)
                    if (interMediateLogLike <= targetLogLike[1]) and (interMediateLogLike >= targetLogLike[0]):
                            isDone = True
                    elif (interMediateLogLike < targetLogLike[0]):
                        #Step too large
                        upperJacFac = jacFac
                    elif (interMediateLogLike > targetLogLike[1]):
                        #Step too small
                        lowerJacFac = jacFac


                #Final update of the gmm
                for k,aGaussian in GMM.enum():
                    aGaussian.mu = oldMu[k]-jacScaled[:,[k]]  # Go into decreasing direction; Do not do deepest stew since this will disturb convergence of EM

            for k,aGaussian in GMM.enum():
                aGaussian.mu = oldMu[k]-jacScaled[:,[k]]*0.  # Go into decreasing direction; Do not do deepest stew since this will disturb convergence of EM
                #Update the associated parameters
        else:
            GMM._getWeightsCPU(x,weights=weights,scaled=True)
            newMSEInter,jac = GMM._getMSEnJacCPU(x,y,weights,regVal=regVal, JacSpace=JacSpace)
            maxJacFac = jacFac = min(initStepSizeJac/sp.linalg.norm(jac), newMSEInter/40.)
            jacScaled[nVarI:,:] = jacFac*jac
            for k,aGaussian in GMM.enum():
                aGaussian.mu = oldMu[k]-jacScaled[:,[k]]

        #xTilde[nVarI:,:] = 0.
        #GMM._evalMapCPU(x, weights=weights, yOut = xTilde[nVarI:,:])
        #dxTilde = GMM._evalMapCPU(x,weights=weights)-xTilde[nVarI:,:]
        #dxTildeHat = GMM.takeProjectedStepPen(x,y,xTilde[nVarI:,:],dxTilde,regVal=regValStep[0],penVal=regValStep[1])
        #xTilde[nVarI:,:] = dxTildeHat
        
        #thetaInit fullfills A.x=y
        #Now we take a step into the dxTilde direction however projected into the nullspace of the linear application
        dxTilde = GMM._evalMapCPU(x,weights=weights)-xTilde[nVarI:,:]
        dxTildeHat,nsProjector = GMM.takeNSProjectedStep(dxTilde,x,nsProjector,returnProjector=True)
        
        tildeFacApply = GMM.applyMaxStep(xTilde, dxTildeHat)
        
        print("theta update step was {0}".format(tildeFacApply))
        
        #Update loglike for new xtilde
        newLogLike = GMM.getLogLikelihood(xTilde)
        
        thisIter += 1
        if doPlot:
            try:
                print("step size of taken step: {0} max step size {1}".format(jacFac, maxJacFac))
            except UnboundLocalError:
                print("step of size {0} was taken".format(jacFacOld))
            aax1[0].plot([thisIter],[newLogLike],'.b')
            aax1[0].plot([thisIter],[GMM._getLogLikelihoodCPU(xTilde)],'+b')
            aax1[1].plot([thisIter],[newMSEplot],'.b')
            aax1[1].plot([thisIter],[GMM.getMSE(x,y,regVal=regVal)],'+b')
            aax1[1].plot([thisIter],[GMM.getMSE(x,y,regVal=0)],'+r')
    
    if doPlot:
        pu.errorPlot(GMM.getErr(x,y, JacSpace=JacSpace),ang=aang,fig=ffig,ax=aax,color='g')

    if thetaOpt is not None:
        np.copyto(thetaOpt, xTilde[nVarI:,:], 'no')
    
    return GMM


def modifiedEMAlgorithm(GMM,x,y,add2Sigma=0.,doInit="KMeans", thetaInit=None, iterMax=500, regVal = 1e-5, relTolLogLike=1e-3, absTolLogLike=1e-3, relTolMSE=1e-3, absTolMSE=1e-3, mseConverged=0., initStepSizeJac=0.005, convBounds=[0.2,0.1], regValStep=[0.08,4.], doPlot=1, plotCallBack=None, thetaOpt=None, nsProjector=None, JacSpace=None):

    #Regular expectation maximization algorithm
    assert isinstance(GMM,GaussianMixtureModel)
    assert x.shape[1]==y.shape[1]
    assert (thetaInit is None) or (x.shape[1] == thetaInit.shape[1])
    assert callable(doInit) or (doInit in ('warm_start', 'KMeans', 'EM', 'continue'))
    
    nVarTot = GMM[0].nVarTot
    
    if isinstance(add2Sigma,float) or (isinstance(add2Sigma,np.ndarray) and add2Sigma.size == 1):
        add2Sigma = add2Sigma*Id(nVarTot)
    else:
        assert add2Sigma.shape == (nVarTot,nVarTot)
        try:
            assert np.all(add2Sigma == add2Sigma.T)
            chol(add2Sigma)
        except:
            assert 0,"Regularization covariance is not symmetric positive"

    if mainUseGPU_ and (isinstance(x, gpuarray.GPUArray) or (x.shape[1] > nMinGPU)):
        assert 0, 'TBD'
        return modifiedEMAlgorithmGPU(GMM,x,y,add2Sigma=add2Sigma,doInit=doInit, thetaInit=thetaInit, iterMax=iterMax, regVal=regVal, relTol=relTol, absTol=absTol, mseConverged=mseConverged, initStepSizeJac=initStepSizeJac, regValStep=regValStep, doPlot=doPlot, thetaOpt=thetaOpt, nsProjector=nsProjector)
    else:
        return modifiedEMAlgorithmCPU(GMM,x,y,add2Sigma=add2Sigma,doInit=doInit,thetaInit=thetaInit,iterMax=iterMax, regVal=regVal,relTolLogLike=relTolLogLike, absTolLogLike=absTolLogLike,relTolMSE=relTolMSE, absTolMSE=absTolMSE, mseConverged=mseConverged,initStepSizeJac=initStepSizeJac,convBounds=convBounds, regValStep=regValStep,doPlot=doPlot, plotCallBack=plotCallBack, thetaOpt=thetaOpt, nsProjector=nsProjector, JacSpace=JacSpace)

if __name__ == '__main__':
    import sys
    import modifiedEMUnitTest as unitT
    
    A = np.random.rand(3,3)
    B = np.random.rand(3,3)
    A = A+B
    A += B
    
    unitT.gaussianTest(dim=6)
    sys.exit()


    
    print("Doing some testing")
    A = np.random.rand(3,1024*500);
    AGPU = gpuarray.to_gpu(toGPUData(A))
    BGPU = AGPU.copy()
    negateGPU(BGPU)
    AGPU+BGPU
    w=np.random.rand(A.shape[1],)
    wGPU = gpuarray.to_gpu(toGPUData(w))
    print(myCov(A,w=w,isCentered=False,forceCPU=True))
    print(myCov(A,w=w,isCentered=False))
    print(myCov(AGPU,w=wGPU,isCentered=False))
    import plotUtils as pu

    # Get a random 2d kernel
    Sigma = np.random.rand(2,2)
    Sigma = 0.1*Id(2)+dot(Sigma, Sigma.T)
    mu = np.random.rand(2,1)-.5
    #Create a gaussian
    aGaussian = gaussianKernel(1, Sigma, mu)
    aGaussian.doCond=True
    
    #Create evaluation points
    x,y = np.meshgrid(linspace(-5,5,100), linspace(-5,5,100))
    X = np.vstack((x.flatten(), y.flatten()))
    del x,y
    outCPU = empty((X.shape[1],))
    outGPU = empty((X.shape[1],))
    yCPU = empty((1,X.shape[1]))
    yGPU = empty((1,X.shape[1]))
    aGaussian._getWeightsCPU(X, out=outCPU, cpy=True)
    aGaussian._getWeightsGPU(X, outCPU=outGPU, cpy=True)
    aGaussian._evalMapCPU(X[[0],:], yCPU)
    aGaussian._evalMapGPU(X[[0],:], yCPU=yGPU)
    
    fig = pu.plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0,:], X[1,:], outCPU, '.r')
    ax.plot(X[0,:], X[1,:], outGPU, '.b')
    
    fig = pu.plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[0,:],yCPU.squeeze(),'.r')
    ax.plot(X[0,:],yGPU.squeeze(),'.b')
    print(aGaussian.Sigma)
    print(inv(aGaussian.Sigma))
    print("Gaussian evaluation correct on GPU/CPU: {0}".format(np.allclose(outCPU, outGPU)))
    print("Gaussian MAP estimation correct on GPU/CPU: {0}".format(np.allclose(yCPU,yGPU)))
    
    #Get a toy GMM
    Sigma = np.random.rand(2,2)
    Sigma = 0.1*Id(2)+dot(Sigma,Sigma.T)
    mu = np.random.rand(2,1)-.5
    bGaussian = gaussianKernel(1,Sigma,mu)
    
    thisGMM = GaussianMixtureModel()
    thisGMM._gaussianList.append(aGaussian)
    thisGMM._gaussianList.append(bGaussian)
    
    yOutCPU = thisGMM._evalMapCPU(X[[0],:])
    yOutGPU = empty(yOutCPU.shape)
    thisGMM._evalMapGPU(X[[0],:],yOutCPU=yOutGPU)

    print("GMM evaluation correct on GPU/CPU: {0}".format(np.allclose(yOutCPU,yOutGPU)))
    fig = pu.plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(X[0,:],yOutCPU.squeeze(),'.r')
    ax.plot(X[0,:],yOutGPU.squeeze(),'.b')
    
    #Higher dimension
    newGMM = GaussianMixtureModel()
    NNI = 2
    NN = NNI+2
    for k in range(5):
        S = np.random.rand(NN,NN)
        S = dot(S, S.T)+.5*Id(NN)
        mu = np.random.rand(NN,1)-.5
        thisKernel = gaussianKernel(NNI, S, mu)
        newGMM.addKernel(thisKernel)
    X2 = 1.*(np.random.rand(NNI,1000)-.5)

    yOutCPU2 = newGMM._evalMapCPU(X2)
    yOutGPU2 = empty(yOutCPU2.shape)
    newGMM._evalMapGPU(X2,yOutCPU=yOutGPU2)
    print("GMM evaluation correct on GPU/CPU in higher dim: {0}".format(np.allclose(yOutCPU2,yOutGPU2, rtol=1e-4, atol=1e-6)))

    #Another test
    newGMM.prior = np.ones((len(newGMM._gaussianList),1))
    weightsCPU = newGMM._getWeightsCPU(X2)
    weightsGPU = zeros((weightsCPU.shape))
    newGMM._getWeightsGPU(X2, weightsOutCPU=weightsGPU)
    print("GMM weights computation correct on GPU/CPU in higher dim: {0}".format(np.allclose(weightsCPU,weightsGPU,rtol=1e-4,atol=1e-6)))
    
    
    #Test loglike
    for aGaussian in newGMM._gaussianList:
        aGaussian.doCond = True
    print(newGMM.prior)
    print(newGMM._priorGPU)
    X2 = 1.*(np.random.rand(NNI,1000)-.5)
    logLikeCPU = newGMM.getLogLikelihood(X2)
    logLikeGPU = newGMM.getLogLikelihood(gpuarray.to_gpu(toGPUData(X2)))
    print("GMM loglikelihood succeeded for CPU/GPU: {0}".format( bool(abs(logLikeCPU-logLikeGPU)<1e-4) ))
    
    X2 = gpuarray.to_gpu(toGPUData(X2))
    xx = X2[0,:]
    
    pu.plt.show()
    cuda.stop_profiler()
    





