# Package only concerned with diffeomorphic transformations based on polynomial kernels
from mainPars import *

from copy import deepcopy

from coreUtils import *
from angleUtils import *
from multiprocessing import Pool
import multiprocessingTools as mt


if whichKernel_ == 1:
    # True original c1 kernel
    from c1PolyKerMod import c1PolyMaxDeform, c1PolyMinB, c1KerFun,c1KerFunPartial, c1PolyKer, c1PolyKerPartialDeriv, nonDiffeoCheckPointList, nonDiffeoCheckPointListLen
elif whichKernel_ == 2:
    #Helper c2 kernel or minimal jerk
    from polyKerMod import c1PolyMaxDeform, c1PolyMinB, c1KerFun,c1KerFunPartial, c1PolyKer, c1PolyKerPartialDeriv, nonDiffeoCheckPointList, nonDiffeoCheckPointListLen
elif whichKernel_ >= 3:
    #Minimal jerk
    from polyKerMod import c1PolyMaxDeform, c1PolyMinB, c1KerFun,c1KerFunPartial, c1PolyKer, c1PolyKerPartialDeriv, nonDiffeoCheckPointList, nonDiffeoCheckPointListLen

def c1PolyTrans(x, c, t, b, d, e=0., deriv=0, out=None, gammaOut=False, inPolarCoords=False):
    out = empty(x.shape) if out is None else out
    if inPolarCoords:
        norm = cNormWrapped
    else:
        norm = cNorm
    if gammaOut:
        kVal = c1PolyKer(norm(x-c,kd=False),bIn=b,dIn=d,eIn=e,deriv=deriv,out=None, inPolarCoords=inPolarCoords)
        out[:] = multiply(t,kVal)
        return out, kVal
    else:
        out[:] = multiply(t,c1PolyKer(norm(x-c,kd=False),bIn=b,dIn=d,eIn=e,deriv=deriv,out=None,inPolarCoords=inPolarCoords))
        return out

def c1PolyTransJac( x, c, t, b, d, e=0., outx=None, outJac=None, gammaOut=False, inPolarCoords=False):
    """"""
    
    if inPolarCoords:
        norm = cNormWrapped
    else:
        norm = cNorm
    
    dim, nPt = x.shape

    outx = zeros(x.shape) if outx is None else outx
    outJac = Idn(nPt, dim) if outJac is None else outJac
    
    dx = x-c # Distance center point
    dxn = norm(dx,kd=False)+epsFloat # Norm distance center point; Ensure that norm is never zero since we will divide by it
    
    #Get the kernel value and derivative
    kval, dkval = c1PolyKer(dxn,b,d,e,deriv=[0,1],inPolarCoords=False)
    
    #Update x
    outx += multiply(t, kval)
    
    #Scale dkval with norm and multiply with distance
    dx = multiply(divide(dkval, dxn), dx)
    # Compute the jacobian
    outJac += np.einsum("m,ij->jmi",t.squeeze(), dx)
    
    if gammaOut:
        return outx, outJac, gammaOut
    else:
        return outx, outJac

def c1PolyInvTrans(xp, centers, translations, bases, dValues, eValues, epsInv=inversionEps, out=None, gammaOut=False, inPolarCoords=False):
    
    assert out is None, "TBD checkup there seems to be a problem with copyto sliced"
    
    nTrans = centers.shape[1]
    convScale = .8
    
    gammaCurrent = eValues.copy()
    errCurrent = ones((nTrans,))

    epsDeriv = np.nextafter(0.,1.)
    
    while (np.any(np.abs(errCurrent) >= epsInv)):
        x = xp-np.sum(np.multiply(translations,gammaCurrent),axis=1,keepdims=1)  # Current estimate of the source point
        dxk = x-centers  # Distance between current estimate and kernel points
        kerVal,kerValD = c1PolyKer(x,bIn=bases,dIn=dValues,eIn=eValues,cIn=centers,deriv=[0,1],inPolarCoords=inPolarCoords)
        kerVal=kerVal.squeeze(); kerValD = kerValD.squeeze()
        kerValD[kerValD==0.] += epsDeriv
        errCurrent = gammaCurrent-kerVal  # Current error
        dxkn = cNorm(dxk,kd=False)
        # The derivative of the error with respect to gamma_k is
        # 1+p'(||xp-sum_k ( v_k.gamma_k ) - c_k||)-1/sqrt(()'.()).||xp-sum_k ( v_k.gamma_k ) - c_k||'.v_k
        deriv = 1+kerValD/dxkn*sum(dxk*translations,axis=0)
        gammaCurrent -= errCurrent/deriv*convScale
        gammaCurrent = np.minimum(np.maximum(gammaCurrent,eValues),1.)
    
    #Actually perform the step
    if out is None:
        out=xp-np.sum(np.multiply(translations,gammaCurrent),axis=1,keepdims=1)
    else:
        np.copyto(out, xp-np.sum(np.multiply(translations,gammaCurrent),axis=1,keepdims=1))
    if inPolarCoords:
        #In order to make sure that the new pos is correctly within ]-pi,pi]
        wrapAngles(out[0,:],out[0,:])
    if gammaOut:
        out = [out, gammaCurrent]
    return out

def c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues, epsInv=inversionEps, out=None, gammaOut=False, inPolarCoords=False):
    
    if gammaOut:
        out = [empty(xp.shape), empty((len(bases),xp.shape[1]))] if out is None else out
    else:
        out = empty(xp.shape) if out is None else out
    
    if gammaOut:
        for k in range(xp.shape[1]):
            out[0][:,[k]],out[1][:,k] = c1PolyInvTrans(xp[:,[k]].copy(),centers,translations,bases,dValues,eValues,epsInv=epsInv,out=None, gammaOut=True, inPolarCoords=inPolarCoords)
    else:
        for k in range(xp.shape[1]):
            out[:,[k]] = c1PolyInvTrans(xp[:,[k]].copy(), centers, translations, bases, dValues, eValues, epsInv=epsInv, out=None, gammaOut=False, inPolarCoords=inPolarCoords)
    
    return out

def c1PolyKerInv(xp, centers, translations, bases, dValues, eValues, deriv=0,epsInv=inversionEps,fullOut=False,inPolarCoords=False):
    
    #First compute original points
    if deriv==0 and not fullOut:
        _, out = c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues,epsInv=Inv, gammaOut=True,inPolarCoords=inPolarCoords)
    else:
        x = c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues,epsInv=epsInv,inPolarCoords=inPolarCoords)
        out = c1PolyKer(x,bIn=bases,dIn=dValues,eIn=eValues,deriv=deriv,cIn=centers,fullOut=fullOut,inPolarCoords=inPolarCoords)
    return out
    
workerPool = Pool(4)


class localPolyMultiTranslation:
    """locally weighted polynomial multitransformation"""
    def __init__(self, dim, centers=None, translations=None, bases=None, dValues=None, eValues=None, isPars=5*[True]):
        
        self._dim = dim
        self._isPars = isPars
        self._nTrans = 0
        #self._nVars = None
        
        self._alphaConstraints = 0.05 #
        self._epsConstraints = None
        self._margins=[0.6,0.2]
        
        self._centers = zeros((dim,0))
        self._translations = zeros((dim,0))
        self._bases = []
        self._dValues = []
        self._eValues = []

        self._basesNA = None
        self._dValuesNA = None
        self._eValuesNA = None
        
        if centers is not None:
            self._nTrans = centers.shape[1]
            assert (centers.shape == translations.shape == (self._dim, self._nTrans)) and (bases.size == self._nTrans) and (len(eValues) == self._nTrans) and (len(dValues) == self._nTrans)
            self._centers = centers.copy()
            self._translations = translations.copy()
            self._bases = list(bases.flatten().copy())
            self._dValues = list(dValues.flatten().copy())
            self._eValues = list(eValues.flatten().copy())
            self._basesNA = na(self._bases)
            self._dValuesNA = na(self._dValues)
            self._eValuesNA = na(self._eValues)

    @property
    def margins(self):
        return deepcopy(self._margins)
    @margins.setter
    def margins(self, newMargins):
        newMargins = list(newMargins)
        assert len(newMargins)==2
        self._margins = newMargins

    @property
    def nTrans(self):
        return self._centers.shape[1]

    @property
    def centers(self):
        return self._centers.copy()
    @centers.setter
    def centers(self,newCenters):
        np.copyto(self._centers, newCenters)

    @property
    def translations(self):
        return self._translations.copy()
    @translations.setter
    def translations(self,newTranslations):
        np.copyto(self._translations,newTranslations)
    
    @property
    def alphaConstraints(self):
        return self._alphaConstraints

    @alphaConstraints.setter
    def alphaConstraints(self,newAlpha):
        self._alphaConstraints = newAlpha

    @property
    def bases(self):
        return self._basesNA.copy()
    @bases.setter
    def bases(self,newBases):
        newBases=na(newBases).squeeze()
        newBases=np.maximum(newBases,0.)
        np.copyto(self._basesNA,newBases)
        self._bases = list(self._basesNA)

    @property
    def dValues(self):
        return self._dValuesNA.copy()
    @dValues.setter
    def dValues(self,newdValues):
        newdValues = na(newdValues).squeeze()
        np.copyto(self._dValuesNA,newdValues)
        self._dValues = list(self._dValuesNA)

    @property
    def eValues(self):
        return self._eValuesNA.copy()
    @eValues.setter
    def eValues(self,neweValues):
        neweValues = na(neweValues).squeeze()
        neweValues = np.maximum(np.minimum(neweValues,1.),0.)
        np.copyto(self._eValuesNA,neweValues)
        self._eValues = list(self._eValuesNA)

    
    def range(self):
        return range(self._nTrans)
    
    def addTransition(self, centers, translations, bases, dValues, eValues):
        """Function adding new translations to the existing multitransformation"""
        nAddTrans = centers.shape[1]
        
        if isinstance(bases, float):
            bases = [bases]
        if isinstance(dValues,float):
            dValues = [dValues]
        if isinstance(eValues,float):
            eValues = [eValues]
        
        assert (centers.shape == translations.shape == (self._dim,nAddTrans)) and (len(bases) == nAddTrans) and (len(eValues) == nAddTrans)
        
        self._nTrans += nAddTrans
        
        self._centers = np.hstack((self._centers, centers))
        self._translations = np.hstack((self._translations,translations))
        self._bases += list(bases)
        self._dValues += list(dValues)
        self._eValues += list(eValues)
        self._basesNA = na(self._bases)
        self._dValuesNA = na(self._dValues)
        self._eValuesNA = na(self._eValues)
        
        return 0
    
    def removeTransition(self, removeList = []):
        """Removes the translations with the indices in removeList from the multitransformation"""
        assert isinstance(removeList, (list, tuple, np.ndarray, np.int64, np.int32, int))
        
        try:
            removeList = list(removeList)
        except:
            try:
                removeList = [int(removeList)]
            except:
                assert 0, "Fail"
        
        removeList.sort()

        self._centers = np.delete(self._centers, removeList, axis = 1)
        self._translations = np.delete(self._translations,removeList,axis=1)
        for r in reversed(removeList):
            del self._bases[r]
            del self._dValues[r]
            del self._eValues[r]
        self._basesNA = na(self._bases)
        self._dValuesNA = na(self._dValues)
        self._eValuesNA = na(self._eValues)
        self._nTrans -= len(removeList)
        
        return 0
    
    def getTransition(self, ind:int):
        assert ind < self.nTrans,"out of range"
        return self._centers[:,[ind]].copy(), self._translations[:,[ind]].copy(), self._bases[ind], self._dValues[ind], self._eValues[ind]
    
    def modifyTransition(self, ind, newCenter=None, newTrans=None, newBase=None, newDValue=None, newEValue=None ):
        assert ind < self.nTrans, "out of range"
        if newCenter is not None:
            newCenter = np.array((newCenter)).reshape((self._dim,1))
            self._centers[:,[ind]] = newCenter
        if newTrans is not None:
            newTrans = np.array((newTrans)).reshape((self._dim,1))
            self._translations[:,[ind]] = newTrans
        if newBase is not None:
            newBase = float(newBase)
            self._bases[ind] = newBase
            self._basesNA[ind] = newBase
        if newDValue is not None:
            newDValue = float(newDValue)
            self._dValues[ind] = newDValue
            self._dValuesNA[ind] = newDValue
        if newEValue is not None:
            newEValue = float(newEValue)
            self._eValues[ind] = newEValue
            self._eValuesNA[ind] = newEValue

    
    def __copy__(self):
        return self.__class__(self._dim, self._centers.copy(), self._translations.copy(), self._basesNA.copy(), self._dValuesNA.copy(), self._eValuesNA.copy(), self._isPars)
    
    def __deepcopy__(self, memodict={}):
        return self.__copy__()
    
    
    def toStringList(self):
        
        """Returns string to be used with c++"""
        totStringList = []
        
        if whichKernel_ == 3:
            # structure is
            # dim
            # nr of kernels
            # ##for each kernel
                # direction
                # dim
                # center
                # translation
                # base
                # evalue
            # ##d value is useless here
            totStringList.append( int2Str(self._centers.shape[0]))
            totStringList.append( int2Str(self._centers.shape[1]))
            for k in range(self._centers.shape[1]):
                #dim
                totStringList.append(int2Str(self._centers.shape[0]))
                #center
                totStringList += vec2List(self._centers[:,k])
                #translation
                totStringList += vec2List(self._translations[:,k])
                #base
                totStringList.append(double2Str(self._bases[k]))
                #evalue
                totStringList.append(double2Str(self._eValues[k]))
        else:
            assert False, "todo" # todo
            
        return totStringList
        
    
    def dist2nondiffeoSimple(self, marginSingle:float=0.5, marginOverlap:float=0.2, norm:"A vector norm"=cNorm, fullOut:bool=False) -> Union[np.ndarray, List[np.ndarray]]:
        """Compute a conservative approximation of the highest occuring deformation"""
        #Highest deformation of each kernel
        if self.nTrans == 0:
            return ones((1,))
        maxDistortion = c1PolyMaxDeform(self._basesNA,self._dValuesNA,cNorm(self._translations,kd=False),e=self._eValuesNA)

        distVec = empty((2,self._nTrans))
        distVec[1,:] = 1.-marginOverlap  # All that overlap
        
        # Store the maximal distortion induced by each kernel
        distVec[0,:] = maxDistortion #c1PolyMaxDeform(b=self._basesNA, d=self._dValuesNA,tn=cNorm(self._translations, kd=False),e=self._eValuesNA)
 
        # Stores for each kernel whether or not it overlaps with the others
        if fullOut:
            isOverlap = np.empty((self.nTrans, self.nTrans)).astype(np.bool_)
        

        # For each kernel
        for k in self.range():
            # Check the distance between this kernel and the others
            dXki = norm(self._centers-self._centers[:,[k]],False)
            # Check if overlap
            iXki = dXki < self._basesNA+self._bases[k]
            if fullOut:
                isOverlap[k,:] = iXki
            # Sum the worst case for all overlapping
            distVec[1,k] -= sum(distVec[0,iXki])
        
        #Now adjust for each kernel
        distVec[0,:] = 1-marginSingle-distVec[0,:]
        
        if fullOut:
            return np.min(distVec,0), isOverlap
        else:
            return np.min(distVec,0)

    def dist2nondiffeo(self, marginSingle: float = 0.5, marginOverlap: float = 0.2, norm: "A vector norm" = cNorm, fullOut: bool = False) -> Union[np.ndarray,
                                                                                                                                                       List[np.ndarray]]:
        """Compute a conservative approximation of the highest occuring deformation"""
        # Highest deformation of each kernel
        if self.nTrans == 0:
            if fullOut:
                return ones((1,)), ones((1,1))
            else:
                return ones((1,))
        maxDistortion = c1PolyMaxDeform(self._basesNA, self._dValuesNA, cNorm(self._translations, kd=False), e=self._eValuesNA)

        distVec = empty((2, self._nTrans))

        # Store the maximal distortion induced by each kernel
        distVec[0, :] = 1. - marginSingle -maxDistortion  # c1PolyMaxDeform(b=self._basesNA, d=self._dValuesNA,tn=cNorm(self._translations, kd=False),e=self._eValuesNA)

        #Create helper array
        checkPoints = np.empty((self.nTrans, nonDiffeoCheckPointListLen))
        maxDistCheckPoint = np.empty_like(checkPoints)
        # Fill
        for k in self.range():
            tn = cNorm(self._translations[:,[k]], False)
            # influence
            checkPoints[k,:] = xn = nonDiffeoCheckPointList(self._bases[k], self._dValues[k])
            # c1PolyKer(x,bIn,dIn,eIn=None,deriv=0,cIn=None,out=None,fullOut=False,inPolarCoords=False)
            for i in range(nonDiffeoCheckPointListLen):
                maxDistCheckPoint[k,i] = np.max(np.abs(tn*c1PolyKer(np.linspace(xn[i],xn[-1],201), self._bases[k], self._dValues[k], self._eValues[k],deriv=1))) #Discretization

        maxDistCombined = maxDistCheckPoint.copy()
        
        #Create a second helper array storing the dotproduct of the normalised translations
        mutualDotProd = np.zeros((self.nTrans,self.nTrans))
        normalisedTranslations = self._translations.copy()
        normalisedTranslations /= (cNorm(normalisedTranslations,kd=True)+epsFloat)
        
        for k in self.range():
            mutualDotProd[k,:] = sum( normalisedTranslations[:,[k]]*normalisedTranslations, axis=0 )
        
        if experimentalInfluence_:
            #if negative -> zero; if positive the maximal influence is bounded by the dotproduct
            mutualDotProd = np.maximum(mutualDotProd,0.)
        else:
            # If the dot product is negative, it can be discarded, else take plus one
            mutualDotProd = np.sign(mutualDotProd)
            mutualDotProd = np.maximum(mutualDotProd,0.)
        
        # Extend the meaning of the mutualDotProd to check if the halfspace compressing space intersect
        if checkHalfSpaceIntersect_:
            for m in self.range():
                #distances centers k from center m
                distM2K = self._centers-self._centers[:,[m]]
                projOnTransM = np.sum( distM2K*normalisedTranslations[:,[m]], axis=0 )
                for k in self.range():
                    #Do not check for self
                    if m==k:
                        continue
                    #if center k lies in the compressing halfspace of m
                    if(projOnTransM[k])>0.:
                        # The compression regions of m and k only overlap if base of m is large than the projected distance / distance to the hyperplane of kernel k
                        # Note that mutualDotProd is already zero if <trans_m,trans_k> < 0
                        if(self._bases[m] < np.dot(distM2K[:,[k]].T, normalisedTranslations[:,[k]])):
                            mutualDotProd[m,k] = 0.
                    #if center k lies in the decompressing halfspace of m -> m lies in the compressing half-space of k
                    else:
                        # The compressing region of k can only intersect with the compressing intersection of m
                        # if the base of k is larger than the distance from the center of k to the hyperplane of m
                        if (self._bases[k] < -projOnTransM[k]):
                            mutualDotProd[m,k] = 0.

        # Stores for each kernel whether or not it overlaps with the others
        if fullOut:
            isOverlap = np.empty((self.nTrans, self.nTrans)).astype(np.bool_)

        # For each kernel
        for k in self.range():
            # Check the distance between this kernel and the others
            dXki = norm(self._centers - self._centers[:, [k]], False)
            # Check if any overlap
            iXki = dXki < checkPoints[:,-1] + checkPoints[k,-1] #The last row of checkpoints corresponds to the outside base zone
            if fullOut:
                isOverlap[k, :] = iXki

            #Check each slice of kernel k for worst overlap with slices of kernel m
            for i in range(1, nonDiffeoCheckPointListLen):
                for m in self.range():
                    if k==m:
                        continue #same kernel
                    for j in range(1,nonDiffeoCheckPointListLen):
                        #Start with smallest influence region -> break if overlap found
                        if dXki[m]<(checkPoints[k,i]+checkPoints[m,j]):
                            #maxDistCombined[k,i-1] += maxDistCheckPoint[m,j-1]
                            # NEW take into account the dotproduct between the translation of k and m todo find a proper proof for this in the case of experimentalInfluence_
                            maxDistCombined[k,i-1] += maxDistCheckPoint[m,j-1]*mutualDotProd[k,m]
                            break
        # Get worst for each kernel
        maxDistCombined = np.max(maxDistCombined, axis=1)

        # Now adjust for each kernel
        distVec[1, :] = 1. - marginOverlap - maxDistCombined

        if fullOut:
            return np.min(distVec, 0), isOverlap
        else:
            return np.min(distVec, 0)
    
    def dist2nondiffeo2(self, marginSingle:float=0.5, marginOverlap:float=0.2, norm:"A vector norm"=cNorm, fullOut:bool=False):
        #The first part is the distance to include zero; Second is dist 2 nondiffeo concerning space deformation
        if fullOut:
            try:
                dist, overlap = self.dist2nondiffeo(marginSingle=marginSingle, marginOverlap=marginOverlap, norm=cNorm, fullOut=True)
            except ValueError:
                print(self.dist2nondiffeo(marginSingle=marginSingle, marginOverlap=marginOverlap, norm=cNorm, fullOut=True))
                assert 0
            if ensureIdentityOrigin_:
                return np.hstack(( cNorm(self._centers, kd=False)-self._basesNA, dist )), overlap
            else:
                # Do not take into account the zero constraint
                return np.hstack((np.ones((self.nTrans,)),dist)),overlap
        else:
            if ensureIdentityOrigin_:
                return np.hstack(( cNorm(self._centers, kd=False)-self._basesNA, self.dist2nondiffeo(marginSingle=marginSingle, marginOverlap=marginOverlap, norm=cNorm) ))
            else:
                return np.hstack((np.ones((self.nTrans,)),self.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap,norm=cNorm)))

    def enforceMargins(self, marginSingle:float,marginOverlap:float, minTransNorm:float=0., minBase:float=0., dist2Zero:float=0.)->bool:
        
        """Ensure that zero is not affected by the translation and that it the diffeomorphic property is ensured"""
        #TBD replace local code with this member function
        counter = 0
        while True:
            assert counter <100000

            thisTransNorm = cNorm(self._translations,kd=False)
            removeInd = np.array(range(self.nTrans)).astype(np.int_)[thisTransNorm<minTransNorm]
            if removeInd.size:
                print('removing {0}; too small transformation'.format(removeInd))
                self.removeTransition(removeInd)
            removeInd = np.array(range(self.nTrans)).astype(np.int_)[self._basesNA < minBase]
            if removeInd.size:
                print('removing {0}; too small base'.format(removeInd))
                self.removeTransition(removeInd)

            allDist,isOverlap = self.dist2nondiffeo2(marginSingle=marginSingle,marginOverlap=marginOverlap, fullOut=True)
            
            if np.all(allDist>0.):
                # All constraints are satisfied
                return counter
                
            dist2Zero,dist2NonDiffeo = allDist[:self.nTrans],allDist[self.nTrans:]
            isNoOverlap = np.logical_not(isOverlap)
            isOverlap = isOverlap.astype(np.float_)
            isNoOverlap = isNoOverlap.astype(np.float_)
            
            for k in self.range():
                # Push center out if too close to zero
                # and decrease base size
                if dist2Zero[k]<=0.:
                    self._centers[:,k] *= 1.025
                    if dist2NonDiffeo[k]>0.:
                        self._bases[k] *= 0.975
                        self._basesNA[k] *= 0.975
                
                if dist2NonDiffeo[k] <=0.:
                    self._translations *= 0.975*isOverlap[[k],:]+isNoOverlap[[k],:]
                    #Ignore those already diminished
                    for l in self.range():
                        if isOverlap[k,l]:
                            dist2NonDiffeo[l] = 1.
            counter += 1
        
        return False
    
    def pointInAnyBase(self,x:np.ndarray,dir:bool=True):
        """Returns a boolean array indicating whether the point is influenced by any of the translations"""
        if not dir:
            x = self.inverseTransform(x)

        if self._nTrans==0:
            return ones((x.shape[1],)).astype(np.bool_)
        xRep = np.repeat(x, repeats=self._nTrans, axis=1)
        centersTile = np.tile(self._centers, (1, x.shape[1]))
        basesTile = np.tile(self._basesNA, x.shape[1])
        allDist = cNorm(xRep-centersTile,kd=False)
        indFlat = allDist<basesTile
        indFlat.resize((x.shape[1],self._nTrans))
        
        return np.any(indFlat, axis=1,keepdims=False)

    def approxMaxDistortion(self, x:np.ndarray, dir:bool=True, norm=cNorm):

        if self.nTrans == 0:
            return zeros((x.shape[1],))

        if not dir:
            x = self.inverseTransform(x)
        # Highest deformation of each kernel
        maxDistortion = c1PolyMaxDeform(self._basesNA,self._dValuesNA,cNorm(self._translations,kd=False),e=self._eValuesNA)

        distPerPoint = zeros((x.shape[1]))
        for k in self.range():
            ind = norm(x-self._centers[:,[k]], kd=False)<self._bases[k]
            distPerPoint[ind] += maxDistortion[k]

        return distPerPoint

    def kVal(self,x,dir=True,deriv=0,fullOut=False,epsInv=inversionEps, inPolarCoords=False):
        
        dim,nPt = x.shape
        assert self._dim == dim
        if dir:
            return c1PolyKer(x,bIn=self._basesNA, dIn=self._dValuesNA, eIn=self._eValuesNA,cIn=self._centers,deriv=deriv,fullOut=fullOut, inPolarCoords=inPolarCoords)
        else:
            if nPt>100:
                #(xp, centers, translations, bases, dValues, eValues, deriv=0,epsInv=1e-6)
                separations = np.linspace(0,nPt,5,dtype=np.int_)
                xList = [x[:,separations[i]:separations[i+1]] for i in range(4)]
                allOut = workerPool.starmap( c1PolyKerInv, zip(xList,4*[self._centers],4*[self._translations],4*[self._basesNA],4*[self._dValuesNA],4*[self._eValuesNA],4*[deriv],4*[epsInv], 4*[fullOut],4*[inPolarCoords] ) )
                if not fullOut:
                    if deriv in ((0,1), [0,1]):
                        out = [ np.hstack( [a for a,b in allOut] ), np.hstack( [b for a,b in allOut] ) ]
                    else:
                        out = np.hstack((allOut))
                else:
                    if deriv in ((0,1), [0,1]):
                        out = [ np.hstack( [a for a,b,[i0,i1,i2,i3] in allOut] ), np.hstack( [b for a,b,[i0,i1,i2,i3] in allOut] ), [np.hstack( [i0 for a,b,[i0,i1,i2,i3] in allOut]),np.hstack( [i1 for a,b,[i0,i1,i2,i3] in allOut]),np.hstack( [i2 for a,b,[i0,i1,i2,i3] in allOut]),np.hstack( [i3 for a,b,[i0,i1,i2,i3] in allOut] )]]
                    else:
                        out = [ np.hstack( [a for a,[i0,i1,i2,i3] in allOut] ), [np.hstack( [i0 for a,[i0,i1,i2,i3] in allOut]),np.hstack( [i1 for a,[i0,i1,i2,i3] in allOut]),np.hstack( [i2 for a,[i0,i1,i2,i3] in allOut]),np.hstack( [i3 for a,[i0,i1,i2,i3] in allOut] )]]
                return out
            else:
                return c1PolyKerInv(x,self._centers,self._translations,self._basesNA,self._dValuesNA,self._eValuesNA,deriv=deriv,epsInv=epsInv,fullOut=fullOut, inPolarCoords=inPolarCoords)
    
    def  kValnPdiff(self, x, dir=True,epsInv=inversionEps):
        
        assert dir, "Inverse not done yet"
        
        return c1PolyKerPartialDeriv(x,self._basesNA,self._dValuesNA,self._eValuesNA,cIn=self._centers,fullOut=True)
    
    def forwardTransform(self, x, out=None, addDelta=False):
        assert (x.shape[0] == self._dim) and ((out is None) or (out.shape == x.shape))
        
        if out is None:
            out = x.copy() if not addDelta else zeros(x.shape)
        else:
            np.copyto(out, x, 'no')

        if self.nTrans == 0:
            return out
        
        if x.shape[1]*self._nTrans > 30000:
            xList = self._nTrans*[x]
            cList = [self._centers[:,[k]] for k in self.range()]
            tList = [self._translations[:,[k]] for k in self.range()]
            outList = workerPool.starmap( c1PolyTrans, zip(xList, cList, tList, self._bases, self._dValues, self.eValues) )
            for aOut in outList:
                out += aOut
        else:
            for k in self.range():
                out += c1PolyTrans(x, c=self._centers[:,[k]], t=self._translations[:,[k]], b=self._bases[k], d=self._dValues[k], e=self._eValues[k])
        
        return out
    
    def forwardTransformJac(self,x,v=None,Jac=None,cpy=True,outx=None,outv=None,outJac=None,outInvJac:bool=False, whichSide:"left or right"='left'):
        assert (x.shape[0] == self._dim) and ((outx is None) or (outx.shape == x.shape))
        assert (v is None) or (x.shape == v.shape)
        assert (Jac is None) or (Jac.shape == (x.shape[1],x.shape[0],x.shape[0]))
        assert (outx is None) or (x.shape == outx.shape)
        assert (outv is None) or (x.shape == outv.shape)
        assert (outJac is None) or (outJac.shape == (x.shape[1],x.shape[0],x.shape[0]))
        assert whichSide in ('left', 'right')
        
        if not outInvJac == {'left':False, 'right':True}[whichSide]:
            warnings.warn('Except special cases whichSide should be left if outInvJac False or right and true')
        
        nPt = x.shape[1]
        
        doJac = Jac is not None
        doV = v is not None
        
        outv = empty(v.shape) if (doV and (outv is None)) else outv
        outJac = empty(Jac.shape) if (doJac and (outJac is None)) else outJac
        if cpy:
            if outx is None:
                outx = x.copy()
            else:
                np.copyto(outx,x)
        else:
            outx = x
        
        # Do the computations
        tempJac = Idn(nPt,self._dim)
        tempdx = zeros(x.shape)
        for k in self.range():
            #Add up jacobian and translation
            c1PolyTransJac(x,c=self._centers[:,[k]],t=self._translations[:,[k]],b=self._bases[k],d=self._dValues[k],e=self._eValues[k],outx=tempdx,outJac=tempJac)
        
        # Add the translation
        outx += tempdx
        #Transform veloctiy
        if doV:
            #We have to compute v'[:,i] = tempJac[i,:,:].v[:,i]
            np.einsum("ijk,ki->ji",tempJac,v,out=outv) #Independent of whichSide and outInvJac
        # Compute overall jacobian
        # Phi = Phi_N(Phi_N-1(...(Phi_0(x)))
        # jac = Jac_N.Jac_N-1. ... Jac_0.v
        # We have to left multiply the new jacobian
        if doJac:
            if outInvJac:
                #Compute the inverse of the actual transformation
                tempJac = mt.slicedInversion(tempJac,cpy=True,NnDim=[x.shape[1],x.shape[0]])
            if whichSide=='left':
                # left multiply for jacobian construction
                np.einsum("ijk,ikl->ijl",tempJac,Jac,out=outJac)
            else:
                # right multiply for inverse jacobian construction
                np.einsum("ijk,ikl->ijl",Jac,tempJac,out=outJac)
                
            
        if doV and doJac:
            return outx, outv, outJac
        elif doV:
            return outx,outv
        elif doJac:
            return outx,outJac
        else:
            return outx
    
    def inverseTransform(self,xp,out=None, epsInv = 1e-6, addDelta=False):
        assert (xp.shape[0] == self._dim) and ((out is None) or (out.shape == xp.shape))
        
        assert addDelta==False, 'TBD'
        
        if out is None:
            out = empty(xp.shape)

        if self.nTrans == 0:
            out[:] = xp[:]
            return out
        
        nPt = xp.shape[1]
        
        #c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues, epsInv=1e-6, out=None, gammaOut=False)
        if nPt > 100:
            separations = np.linspace(0,nPt,5,dtype=np.int_)
            xList = [xp[:,separations[i]:separations[i+1]] for i in range(4)]
            out[:] = np.hstack( workerPool.starmap( c1PolyInvTransMulti, zip(xList, 4*[self._centers], 4*[self._translations], 4*[self._basesNA], 4*[self._dValuesNA], 4*[self._eValuesNA], 4*[epsInv]) ) )
        else:
            out[:] = c1PolyInvTransMulti(xp, centers=self._centers, translations=self._translations, bases=self._basesNA, dValues=self._dValuesNA, eValues=self._eValuesNA,epsInv=epsInv)
        
        return out
            
    def inverseTransformJac(self, xp, vp=None, Jacp=None, cpy=True, outx=None, outv=None, outJac=None, outInvJac:bool=False, epsInv = inversionEps, whichSide:"left or right"='left'):
        """
        
        :param xp:
        :param vp:
        :param Jacp:
        :param cpy:
        :param outx:
        :param outv:
        :param outJac:
        :param outInvJac: If True, the jacobian of the corresponding forward transformation is handed back; So its the inverse of the actual jacobian (the jacobian of the inverse transformation)
        :param epsInv:
        :return:
        """
        assert (xp.shape[0] == self._dim) and ((outx is None) or (outx.shape == xp.shape))
        assert (vp is None) or (xp.shape == vp.shape)
        assert (Jacp is None) or (Jacp.shape == (xp.shape[1],xp.shape[0],xp.shape[0]))
        assert (outx is None) or (xp.shape == outx.shape)
        assert (outv is None) or (xp.shape == outv.shape)
        assert (outJac is None) or (outJac.shape == (xp.shape[1],xp.shape[0],xp.shape[0]))
        assert whichSide in ('left','right')

        if not outInvJac == {'left':False,'right':True}[whichSide]:
            warnings.warn('Except special cases whichSide should be left if outInvJac False or right and true')
        
        nPt = xp.shape[1]
        
        doJac = Jacp is not None
        doV = vp is not None
        
        outv = empty(vp.shape) if (doV and (outv is None)) else outv
        outJac = empty(Jacp.shape) if (doJac and (outJac is None)) else outJac
        if cpy:
            if outx is None:
                outx = xp.copy()
            else:
                np.copyto(outx,xp)
        else:
            outx = xp
        
        #First get the inverse and the gamma's
        #c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues, epsInv=1e-6, out=None, gammaOut=False, inPolarCoords=False)
        if nPt>100:
            separations = np.linspace(0,nPt,5,dtype=np.int_)
            xList = [xp[:,separations[i]:separations[i+1]] for i in range(4)]
            outx[:] = np.hstack(workerPool.starmap(c1PolyInvTransMulti,zip(xList,4*[self._centers],4*[self._translations],4*[self._basesNA],4*[self._dValuesNA],4*[self._eValuesNA],4*[epsInv], 4*[None], 4*[False], 4*[False])))
        else:
            outx[:] = c1PolyInvTransMulti(xp,centers=self._centers,translations=self._translations,bases=self._basesNA,dValues=self._dValuesNA,eValues=self._eValuesNA,epsInv=epsInv)

        # Once the original points obtained we have to compute the forward transform to get the jacobian
        if (nPt>100) and (not outInvJac) or doV:
            #The inversed jacobian is needed
            tempJacShared = mt.sctypes.RawArray(mt.c.c_double, nPt*self._dim*self._dim)
            tempJac = np.frombuffer(tempJacShared)
            tempJac.resize((nPt,self._dim,self._dim))
            np.copyto(tempJac, Idn(nPt,self._dim), 'safe')
        else:
            tempJac = Idn(nPt,self._dim)
        tempdx = zeros(xp.shape)
        for k in self.range():
            # Add up jacobian and translation
            c1PolyTransJac(outx,c=self._centers[:,[k]],t=self._translations[:,[k]],b=self._bases[k],d=self._dValues[k],
                           e=self._eValues[k],outx=tempdx,outJac=tempJac)
        #Unfortunately we have to inverse if outInvJac==False or if we have to propagate a velocity
        if doV or (doJac and (not outInvJac)):
            try:
                tempJacI = mt.slicedInversion(tempJacShared, cpy=True, NnDim=[nPt, self._dim], returnNp=True)
            except NameError:
                tempJacI = mt.slicedInversion(tempJac,cpy=True,NnDim=[nPt,self._dim],returnNp=True)
        # Attention tempJacI is the jacobian of this transformation (the inverse transformation)
        # tempJac is the jacobian of the corresponding forward transformation (so the inverse jacobian of this transformation)
        #Modify velocity
        if doV:
            # We have to compute v[:,i] = tempJacI[i,:,:].v'[:,i]
            np.einsum("ijk,ki->ji",tempJacI,vp,out=outv)
        if doJac:
            # Decide which one to use
            thisTempJac = tempJac if outInvJac else tempJacI
            # left or right multiply
            if whichSide == 'left':
                np.einsum("ijk,ikl->ijl",thisTempJac,Jacp,out=outJac)
            else:
                np.einsum("ijk,ikl->ijl",Jacp,thisTempJac,out=outJac)
        
        if doV and doJac:
            return outx, outv, outJac
        elif doV:
            return outx,outv
        elif doJac:
            return outx,outJac
        else:
            return outx

    def getAllVarsRaw(self):
        return [self._centers, self._translations,self._basesNA,self._dValuesNA,self._eValuesNA]
    
    def getAllVars(self):
        return [self.centers, self.translations,self.bases,self.dValues,self.eValues]

    def parsAsVec(self, isPars=None):
        
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list, tuple)) and isPars:
            isPars = 5*[True]
        
        allPars = []
        for do,pars in zip(isPars, self.getAllVars()):
            if do:
                allPars.append( pars.flatten(order='F') ) # Columnwise storage
        
        return np.hstack(allPars)

    def nVars(self, isPars=None):
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 5*[True]
        return int(sum(lmap(lambda a: a[1].size if a[0] else 0, zip(isPars, self.getAllVarsRaw()))))
        
    
    def applyStep(self, deltaPars, isPars=None):
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 5*[True]
        
        #pars = ['centers', 'translations', 'bases', 'dValues','eValues']
        #execstring = """old=self.{0}; nEnd=nStart+old.size; self.{0} = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        #for k,do,par in enumerate(zip(isPars,pars)):
        #    if do:
        #        old=getattr(self, par); nEnd=nStart+old.size; setattr(self, par) = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        nStart = 0
        if isPars[0]:
            old=self.centers; nEnd=nStart+old.size; self.centers = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[1]:
            old=self.translations; nEnd=nStart+old.size; self.translations = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[2]:
            old=self.bases; nEnd=nStart+old.size; self.bases = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[3]:
            old=self.dValues; nEnd=nStart+old.size; self.dValues = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[4]:
            old=self.eValues; nEnd=nStart+old.size; self.eValues = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            
        
        #
        #    if do:
                #exec(execstring.format(pars[k]), {}, locals())
        
        return 0

    def applyConstrainedStep(self,deltaPars,isPars=None,marginSingle=.5,marginOverlap=.2):
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 5*[True]
        
        alphaL=0.
        alphaU=1.0
        
        copied = self.__copy__()
        assert np.all(copied.dist2nondiffeo(marginSingle=marginSingle, marginOverlap=marginOverlap)), 'Initial transformation not a diffeo'
        
        pars = ['centers','translations','bases','dValues','eValues']
        execstring = """old=self.{0}; nEnd=nStart+old.size; self.{0} = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        execstringAL = """old=self.{0}; nEnd=nStart+old.size; self.{0} = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        execstringC = """old=copied.{0}; nEnd=nStart+old.size; copied.{0} = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        execstringCA = """old=copied.{0}; nEnd=nStart+old.size; copied.{0} = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        try:
            alphaL=1.
            #nStart = 0
            #for k,do in enumerate(isPars):
            #    if do:
            #        exec(execstringC.format(pars[k]),{},locals())
            nStart = 0
            if isPars[0]:
                old=self.centers; nEnd=nStart+old.size; copied.centers = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            if isPars[1]:
                old=self.translations; nEnd=nStart+old.size; copied.translations = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            if isPars[2]:
                old=self.bases; nEnd=nStart+old.size; copied.bases = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            if isPars[3]:
                old=self.dValues; nEnd=nStart+old.size; copied.dValues = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            if isPars[4]:
                old=self.eValues; nEnd=nStart+old.size; copied.eValues = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
            
            assert np.all(copied.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap)>0.)
        except:
            alphaL=0.
            alpha = 0.5
            while ((alphaU-alphaL>0.05*alpha) or alphaL==0):
                if alphaU<1e-6:
                    print("Found no allowed step in the given direction")
                    alphaL = alpha = alphaU = 0.
                    break
                alpha = (alphaU+alphaL)/2.
                print("current inner step alpha is: {0}".format(alpha))
                #nStart = 0
                #for k,do in enumerate(isPars):
                #    if do:
                #        exec(execstringCA.format(pars[k]),{},locals())
                nStart = 0
                if isPars[0]:
                    old=self.centers; nEnd=nStart+old.size; copied.centers = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
                if isPars[1]:
                    old=self.translations; nEnd=nStart+old.size; copied.translations = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
                if isPars[2]:
                    old=self.bases; nEnd=nStart+old.size; copied.bases = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
                if isPars[3]:
                    old=self.dValues; nEnd=nStart+old.size; copied.dValues = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
                if isPars[4]:
                    old=self.eValues; nEnd=nStart+old.size; copied.eValues = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
                
                if np.all(copied.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap)>0.):
                    alphaL=alpha
                else:
                    alphaU=alpha
        
        #nStart = 0
        #for k,do in enumerate(isPars):
        #    if do:
        #        exec(execstringAL.format(pars[k]),{},locals())
        nStart = 0
        if isPars[0]:
            old=self.centers; nEnd=nStart+old.size; self.centers = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[1]:
            old=self.translations; nEnd=nStart+old.size; self.translations = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[2]:
            old=self.bases; nEnd=nStart+old.size; self.bases = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[3]:
            old=self.dValues; nEnd=nStart+old.size; self.dValues = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        if isPars[4]:
            old=self.eValues; nEnd=nStart+old.size; self.eValues = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;

        #Final test
        
        if not np.all(self.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap) >= 0.):
            assert 0
        return alphaL
    
    def getMSE(self, sourcex, targetx, dir=True, sourcev=None, targetv=None, xvtradeoff=None,epsInv=1e-6, alphaConstraints=None, epsConstraints=None, marginSingle=None, marginOverlap=None):
        
        if sourcev is None:
            if dir:
                targetxtilde = self.forwardTransform(sourcex)
            else:
                targetxtilde = self.inverseTransform(sourcex,epsInv=epsInv)
    
            targetxtilde -= targetx
            mse = compMSE(targetxtilde)
        else:
            if dir:
                targetxtilde, targetvtilde = self.forwardTransformJac(sourcex, sourcev)
            else:
                targetxtilde, targetvtilde = self.inverseTransformJac(sourcex, sourcev, epsInv=epsInv)
            targetxtilde -= targetx
            targetvtilde -= targetvtilde
            mse = np.mean(sum(square(targetxtilde),0)) + xvtradeoff*np.mean(sum(square(targetvtilde),0))
        
        epsConstraints = self._epsConstraints if epsConstraints is None else epsConstraints
        if epsConstraints is not None:
            alphaConstraints = self._alphaConstraints if alphaConstraints is None else alphaConstraints
            marginSingle = self._margin[0] if marginSingle is None else marginSingle
            marginOverlap = self._margin[1] if marginSingle is None else marginOverlap
            #mse += np.sum( np.exp(-self.dist2nondiffeo2(marginSingle=marginSingle, marginOverlap=marginOverlap)/epsConstraints) )
            mse += -alphaConstraints*np.sum(np.log(self.dist2nondiffeo2(marginSingle=marginSingle,marginOverlap=marginOverlap)/epsConstraints))
        
        return mse
    
    def getMSEnJac(self, sourcex,targetx=None,dir=True,isPars=None, err=None, mse=None, alphaConstraints=None, epsConstraints=None, marginSingle=None, marginOverlap=None):
        #TBD: sourcev/targetv and dir==False
        
        dim,nPt = sourcex.shape
        assert self._dim==dim
        assert (targetx is None) or (sourcex.shape==targetx.shape)
        assert (err is None) or (sourcex.shape == err.shape)
        assert (targetx is None) or (err is None)

        epsConstraints = self._epsConstraints if epsConstraints is None else epsConstraints
        alphaConstraints = self._alphaConstraints if alphaConstraints is None else alphaConstraints
        
        if epsConstraints is not None:
            marginSingle = self._margin[0] if marginSingle is None else marginSingle
            marginOverlap = self._margin[1] if marginSingle is None else marginOverlap
            # The barrier is active between 0 and 3*eps
            # barrier to keep zero zero
            dist2Zero = cNorm(self._centers, kd=False)-self._basesNA
            ind2Zero = dist2Zero < 3*epsConstraints
            dist2Zero[np.logical_not(ind2Zero)] = 3*epsConstraints
            dist2Zero /= epsConstraints
            barrier2Zero = -alphaConstraints*np.log(dist2Zero)
            derivBarrier2Zero = -alphaConstraints*1./(dist2Zero)*ind2Zero
            # barrier to ensure diffeo
            dist2NonDiffeo, isOverlapNondiffeo = self.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap, fullOut=True)
            ind2NonDiffeo = dist2NonDiffeo < 3*epsConstraints
            dist2NonDiffeo[np.logical_not(ind2NonDiffeo)] = 3*epsConstraints
            dist2NonDiffeo /= epsConstraints
            barrier2NonDiffeo = -alphaConstraints*np.log(dist2NonDiffeo)
            derivBarrier2NonDiffeo = -alphaConstraints*1./(dist2NonDiffeo)*ind2NonDiffeo
        
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 5*[True]
        #isPars: [centers, translations,bases,dValues,eValues]
        if dir:
            if err is None:
                targetxtilde = self.forwardTransform(sourcex)
                err = targetx-targetxtilde
            if mse is None:
                mse = compMSE(err)
            valD = self.kValnPdiff(sourcex,dir=True)
            
            allJac = []
            if isPars[0]:
                jacCenters = empty(self._centers.shape)
                for k in self.range():
                    dxcknorm = sourcex-self._centers[:,[k]]
                    dxcknorm /= (cNorm(dxcknorm)+epsFloat)
                    jacCenters[:,k] = 2*np.mean( (sum(err*self._translations[:,[k]],axis=0)*valD['kdVal'][k,:])*dxcknorm, axis=1 )
                if epsConstraints is not None:
                    # Add the "get away from boundary"
                    # In this case move the center away from the zero (main component)
                    # and decrease the base (minor component)
                    centersNormed = -self._centers/cNorm(self._centers,kd=True)
                    jacCenters += derivBarrier2Zero*centersNormed
                    
                allJac.append( jacCenters.flatten(order='F') )
            
            if isPars[1]:
                jacTranslation = empty(self._translations.shape)
                #Get the current translation
                try:
                    translationsNormed = self._translations/cNorm(self._translations,kd=True)
                except:
                    assert 0
                
                for k in self.range():
                    jacTranslation[:,k] = -2.*np.mean(valD['kVal'][k,:]*err, axis=1)
                    if epsConstraints is not None:
                        # Add the "get away from boundary"
                        # In this case reducing the norm of the translation vector is most effective
                        # increasing d-value would also help however it would change the "appearance" of the kernel
                        # Increasing b-Value with reduce max deformation for one transition but would also result in more overlapping
                        #Reduce the norm of all transformation that overlap and are therefore implicated in this propblem.
                        jacTranslation += derivBarrier2NonDiffeo[k]*isOverlapNondiffeo[[k],:]*translationsNormed
                    
                allJac.append( jacTranslation.flatten(order='F') )

            if np.any(isPars[2:]):
                subJac = [empty(self._basesNA.shape),empty(self._dValuesNA.shape),empty(self._eValuesNA.shape)]
                for k in self.range():
                    errdottk = -2*sum(err*self._translations[:,[k]],axis=0)
                    for apar,do,aJac in zip(['b','d','e'],isPars[2:],subJac):
                        if do:
                            aJac[k] = np.mean(errdottk*valD[apar][k,:])
                            if (epsConstraints is not None) and (apar == 'b'):
                                # Add the "get away from boundary"
                                # In this case move the center away from the zero (main component)
                                # and decrease the base (minor component)
                                aJac[k] += derivBarrier2Zero[k]/2.
                                
                for do,aJac in zip(isPars[2:],subJac):
                    if do:
                        allJac.append(aJac.flatten(order='F'))
            
            allJac = np.hstack(allJac)
        else:
            assert 0, "dir==False TBD"
        
        return mse, allJac
    
    def backTrackStep(self, sourcex, targetx, dir=True,alpha=50., tau=.5, c=1e-1, p='grad', marginSingle=0.5, marginOverlap=0.2):
        
        assert dir, 'Inverse direction TBD'
        
        fx,grad = self.getMSEnJac(sourcex,targetx,dir=True)
        
        if p == 'grad':
            p = -1.*grad #Search direction
            
        t=c*np.inner(p,grad)

        copied = self.__copy__()
        alphaApply = copied.applyConstrainedStep(alpha*p, marginSingle=marginSingle, marginOverlap=marginOverlap)
        alpha*=alphaApply
        #while (copied.getMSE(sourcex,targetx)-fx < alpha*alphaApply*t) and (alphaApply>.99) and (alpha < 10000.):
        #    copied = self.__copy__()
        #    alpha*=10.
        #    alphaApply = copied.applyConstrainedStep(alpha*p)
        fx1 = copied.getMSE(sourcex,targetx,dir=dir)
        while fx1 >= fx + alpha*t:
            copied = self.__copy__()
            alpha = alpha*tau
            alphaApply = copied.applyConstrainedStep(alpha*p)
            alpha *= alphaApply
            fx1 = copied.getMSE(sourcex,targetx)
        
        alphaApplyFinal = self.applyConstrainedStep(alpha*p)
        try:
            assert alphaApplyFinal > 0.95
        except:
            print('aa')
        
        return fx1, alpha*alphaApply
        
        


class localPolyMultiRotation(localPolyMultiTranslation):

    def __init__(self, dim, rotation, centers=None, translations=None, bases=None, dValues=None, eValues=None, isPars=6*[True], scaled=None):

        wrapAngles(centers[0,:], centers[0,:])

        super().__init__(dim, centers, translations, bases, dValues, eValues, isPars)

        if rotation is not None:
            assert rotation.shape == (dim, dim)
            self._rotation = rotation.copy()

        self._scaled = ones((self._dim, 1))
        if scaled is not None:
            assert len(scaled) in (3, self._dim)
            self.scaledRaw = scaled.squeeze()
    
    @property
    def rotation(self):
        return self._rotation.copy()
    @rotation.setter
    def rotation(self, newRotation):
        assert newRotation.shape == (self._dim,self._dim)
        if self._rotation is None:
            self._rotation = newRotation.copy()
        else:
            np.copyto(self._rotation, newRotation, 'no')

    @property
    def translations(self):
        return self._translations.copy()
    @translations.setter
    def translations(self, newTranslations):
        assert np.allclose(newTranslations[1:,:], 0.), "only angle is allowed to change"
        assert newTranslations.shape == self._translations.shape
        np.copyto(self._translations, newTranslations, 'no')

    @property
    def bases(self):
        return self._basesNA.copy()
    @bases.setter
    def bases(self,newBases):
        newBases = na(newBases).squeeze()
        newBases = np.minimum(newBases,0.5*np.pi)
        newBases = np.maximum(newBases,0.)
        np.copyto(self._basesNA,newBases)
        self._bases = list(self._basesNA)
    
    @property
    def centers(self):
        return None if self._centers is None else self._centers.copy()
    @centers.setter
    def centers(self, newCenters):
        wrapAngles(newCenters[0,:],newCenters[0,:]) #Assure angle in [-pi , pi]
        np.maximum(newCenters[1,:], 0., out=newCenters[1,:]) #Assure positive radius
        np.copyto(self._centers, newCenters)

    @property
    def centersCart(self):
        return self.toCartesian(self._centers/self._scaled)
    @centersCart.setter
    def centersCart(self, newCentersCart):
        self.centers = self.toPolar(newCentersCart)*self._scaled

    @property
    def scaled(self):
        return self._scaled.copy() if self._scaled is not None else None
    @scaled.setter
    def scaled(self, newScaled):
        newScaled /= newScaled[0] #Scale for angle has to be one always
        #Remove old scaled from values
        if self._scaled is not None:
            self._centers /= self._scaled
        if newScaled.size == self._dim:
            self._scaled = newScaled.reshape((self._dim,1))
        elif newScaled.size == 3:
            newScaled = newScaled.flatten()
            self._scaled[:3,0] = newScaled
            self._scaled[3:,0] = newScaled[2]
        self._centers *= self._scaled

    @property
    def scaledRaw(self):
        return self._scaled[:min(self._dim, 3),[0]].copy() if (self._scaled is not None) else None
    @scaledRaw.setter
    def scaledRaw(self, newScaled):
        if newScaled.size==self._dim:
            self._scaled = newScaled.reshape((self._dim,1))
        elif newScaled.size==3:
            newScaled=newScaled.flatten()
            self._scaled=empty((self._dim,1))
            self._scaled[:3,0]=newScaled
            self._scaled[3:,0] = newScaled[2]


    def addTransition(self, centers, translations, bases, dValues, eValues):
        assert np.allclose(translations[1:,:], 0.)
        centers.resize((self._dim, bases.size))
        centers *= self._scaled
        assert translations[0] < pi/2. #Theoretically pi should work too
        super(self.__class__, self).addTransition(centers, translations, bases, dValues, eValues)

    def __copy__(self):
        return self.__class__(self._dim, self._rotation.copy(), self._centers.copy(), self._translations.copy(), self._basesNA.copy(),
                              self._dValuesNA.copy(), self._eValuesNA.copy(), self._isPars, self._scaled)

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def dist2nondiffeo(self,marginSingle=0.5,marginOverlap=0.2,norm=cNormWrapped):
        
        allMinDist = empty((5,self._nTrans))
        allMinDist[0,:] = super().dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap,norm=cNormWrapped)
        allMinDist[1,:] = self._basesNA
        allMinDist[2,:] = np.pi*1./2. - self._basesNA
        allMinDist[3,:] = self._eValuesNA
        allMinDist[4,:] = 1-self._eValuesNA
        
        return np.min(allMinDist, axis=0)
        
    
    def Cart2Rot(self,x):
        return dot(self._rotation, x)
    
    def toPolar(self, x, isRot=False):
        if not isRot:
            x=self.Cart2Rot(x)
        else:
            x=x.copy()
        helper = np.arctan2(x[1,:],x[0,:])
        x[1,:] = cNorm(x[:2,:],kd=False)
        x[0,:] = helper
        return x
    
    def Pol2Rot(self,x,copy=True):
        x = x.copy() if copy else x
        ang = x[0,:]
        r = x[1,:]
        allSr = np.sin(ang)*r
        allCr = np.cos(ang)*r
        x[0,:]=allCr
        x[1,:]=allSr
        return x
    
    def toCartesian(self,x, isPol=True):
        if isPol:
            x = self.Pol2Rot(x)
        return dot(self._rotation.T,x)
    
    def kVal(self,x,dir=True,deriv=0,fullOut=False,epsInv=inversionEps, isPolar=False):
        
        if not isPolar:
            x = self.toPolar(x)
        
        if (self._scaled is not None):
            x = self._scaled*x
        
        return super(self.__class__, self).kVal(x,dir=True,deriv=0,fullOut=False,epsInv=1e-6,inPolarCoords=True)
    
    def kValnPdiff(self, x, dir=True,epsInv=1e-6, isPolar=False, isScaled=False):
        """Attention this function returns the """
        assert dir, "Inverse not done yet"

        if not isPolar:
            x = self.toPolar(x)

        if (self._scaled is not None) and not isScaled:
            x *= self._scaled

        return c1PolyKerPartialDeriv(x,self._basesNA,self._dValuesNA,self._eValuesNA,cIn=self._centers,fullOut=True, inPolarCoords=True)
    
    def forwardTransform(self, x, out=None, isPolar=False, outPolar=False):
    
        assert (x.shape[0] == self._dim) and ((out is None) or (out.shape == x.shape))
    
        if not isPolar:
            x = self.toPolar(x)
        
        if out is None:
            out = x.copy()
        else:
            np.copyto(out,x,'no')
        
        if self._scaled is not None:
            x *= self._scaled
        
    
        if x.shape[1]*self._nTrans > 30000:
            xList = self._nTrans*[x]
            cList = [self._centers[:,[k]] for k in self.range()]
            tList = [self._translations[:,[k]] for k in self.range()]
            outList = workerPool.starmap(c1PolyTrans,zip(xList,cList,tList,self._bases,self._dValues,self.eValues))
            for aOut in outList:
                out += aOut
        else:
            for k in self.range():
                out += c1PolyTrans(x,c=self._centers[:,[k]],t=self._translations[:,[k]],b=self._bases[k],
                                   d=self._dValues[k],e=self._eValues[k], inPolarCoords=True)
        
        if not outPolar:
            out = self.toCartesian(out)
        
        return out
    
    def intermediateJacRot2Polar(self,x):
        """x has to be in rotated coords"""
        r = cNorm(x[:2,:],kd=False)
        x0overr = x[0,:]/r
        x1overr = x[1,:]/r

        Jac = Idn(x.shape[1],x.shape[0])
        Jac[:,0,0] = -x1overr
        Jac[:,0,1] = x0overr
        Jac[:,1,0] = x0overr
        Jac[:,1,1] = x1overr
        
        return Jac
    
    def intermediateJac(self, x, isRot=False):
        """Computes the intermediate jacobians for the
        transformations from
        X_cart -> X_Rot -> X_polar -> X_scaled (direction = True)
        or
        X_scaled -> X_polar -> X_rot -> X_cart (direction = False)
        So the jacobians are
        J_polar2scaled.J_rot2polar.J_cart2rot (direction = True)
        and
        (J_polar2scaled.J_rot2polar.J_cart2rot)^-1 (direction = False)
        (J_cart2rot^-1).(J_rot2polar^-1).(J_polar2scaled^-1)
        Attention x has to be in cartesian coords or rotated coords !!!
        """
        if not isRot:
            x = dot(self._rotation,x)

        #Due to how it is constructed,
        #J_rot2polar is its own inverse (and symmetric)
        
        Jac = self.intermediateJacRot2Polar(x)
        
        #Jac = S.J.R
        
        s=self._scaled.reshape((1,self._dim,1))
        R = self._rotation
        
        #Compute J.R
        Jac = np.einsum("ijk,kl->ijl",Jac,R)
        #Left multiply with S
        #S.(J.R)
        Jac *= s #A left multiplication with a diagonal matrix is like scaling the rows
        
        return Jac
    
    def intermediateJacPol2Rot(self,x):
        """x has to be in NONSCALED POLAR COORDS"""
        allS = np.sin(x[0,:])
        allC = np.cos(x[0,:])
        allR = x[1,:]
        
        Jac = Idn(x.shape[1],self._dim)
        Jac[:,0,0] = -allS*allR
        Jac[:,0,1] = allC
        Jac[:,1,0] = allC*allR
        Jac[:,1,1] = allS
        return Jac
    
    def inverseIntermediateJac(self,x):
        """x has to be polar, non-scaled"""
        
        Ri = self._rotation.T
        si = (1./self._scaled).reshape((1,1,self._dim))
        
        Jac = self.intermediateJacPol2Rot(x)
        
        #Ri.J
        Jac = np.einsum("jk,ikl->ijl",Ri,Jac)
        #(Ri.J).diag(si)
        Jac *= si
        
        return Jac
        

    def forwardTransformJac(self, x, v=None, Jac=None, cpy=True, outx=None, outv=None, outJac=None,alsoReturnTrafoJac=False):
        """Everything is expected in CARTESIAN COORDS"""
        
        assert outx is None, "TBD, no efficient implemented for this yet available"
    
        assert (x.shape[0] == self._dim) and ((outx is None) or (outx.shape == x.shape))
        assert (v is None) or (x.shape == v.shape)
        assert (Jac is None) or (Jac.shape == (x.shape[1],x.shape[0],x.shape[0]))
        assert (outx is None) or (x.shape == outx.shape)
        assert (outv is None) or (x.shape == outv.shape)
        assert (outJac is None) or (outJac.shape == (x.shape[1],x.shape[0],x.shape[0]))
    
        nPt = x.shape[1]
    
        doJac = Jac is not None
        doV = v is not None
        
        xRot = self.Cart2Rot(x)
        x = self.toPolar(x)
    
        outv = empty(v.shape) if (doV and (outv is None)) else outv
        outJac = empty(Jac.shape) if doJac else outJac
        if cpy:
            if outx is None:
                outx = x.copy()
            else:
                np.copyto(outx,x)
        else:
            outx = x
    
        # Do the computations in scaled poar coords
        x*=self._scaled
        tempJac = Idn(nPt,self._dim)
        for k in self.range():
            # Add up jacobian and translation
            c1PolyTransJac(x,c=self._centers[:,[k]],t=self._translations[:,[k]],b=self._bases[k],d=self._dValues[k],
                           e=self._eValues[k],outx=outx,outJac=tempJac, inPolarCoords=True)
        #This jacobian is in the SCALED POLAR COORDS, So we have to right multiply the intermediteJac in dir=True with orig xrot
        # and left multiply with intermediteJac in dir=False with new xrot
        #Get new rot
        #AA
        #outx = self.Pol2Rot(outx,copy=False)
        #tempJac = np.einsum("ijk,ikl->ijl",tempJac,self.intermediateJac(xRot, isRot=True,dir=True))
        #tempJac = np.einsum("ijk,ikl->ijl", self.intermediateJac(outx,isRot=True,dir=False),tempJac)
        #outx = self.toCartesian(outx,isPol=False)
        #BB I have no clue why AA is wrong
        #Only angle is update the others are unscaled
        if alsoReturnTrafoJac:
            rawJac = tempJac.copy()
        tempJac = np.einsum("ijk,ikl->ijl",tempJac,self.intermediateJac(xRot, isRot=True))
        tempJac = np.einsum("ijk,ikl->ijl", self.inverseIntermediateJac(outx),tempJac)
        
        #Get the output in the cartesian frame from rotated frame
        outx = self.toCartesian(outx,isPol=True)
        
        
        # Transform veloctiy
        if doV:
            # We have to compute v'[:,i] = tempJac[i,:,:].v[:,i]
            np.einsum("ijk,ki->ji",tempJac,v,out=outv)
        # Compute overall jacobian
        # Phi = Phi_N(Phi_N-1(...(Phi_0(x)))
        # jac = Jac_N.Jac_N-1. ... Jac_0.v
        # We have to left multiply the new jacobian
        if doJac:
            # We have to compute J'[:,i] = tempJac[i,:,:].J[i,:,:]
            np.einsum("ijk,ikl->ijl",tempJac,Jac,out=outJac)
    
        if not alsoReturnTrafoJac:
            if doV and doJac:
                return outx,outv,outJac
            elif doV:
                return outx,outv
            elif doJac:
                return outx,outJac
            else:
                return outx
        else:
            if doV and doJac:
                return outx,outv,outJac,rawJac
            elif doV:
                return outx,outv,rawJac
            elif doJac:
                return outx,outJac,rawJac
            else:
                return outx,rawJac
        
    def inverseTransform(self,xp,out=None, epsInv = 1e-6, isPolar=False, outPolar = False):
        assert out is None, "TBD"
        
        
        assert (xp.shape[0] == self._dim) and ((out is None) or (out.shape == xp.shape))
        
        if out is None:
            out = empty(xp.shape)
        
        nPt = xp.shape[1]
        
        if not isPolar:
            xp = self.toPolar(xp)
            
        if self._scaled is not None:
            xp *= self._scaled
        
        # c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues, epsInv=1e-6, out=None, gammaOut=False, inPolarCoords=False)
        if nPt > 100:
            separations = np.linspace(0,nPt,5,dtype=np.int_)
            xList = [xp[:,separations[i]:separations[i+1]] for i in range(4)]
            out[:] = np.hstack(workerPool.starmap(c1PolyInvTransMulti,
                                                  zip(xList,4*[self._centers],4*[self._translations],4*[self._basesNA],
                                                      4*[self._dValuesNA],4*[self._eValuesNA],4*[epsInv],4*[None],4*[False],4*[True])))
        else:
            out[:] = c1PolyInvTransMulti(xp,centers=self._centers,translations=self._translations,bases=self._basesNA,
                                         dValues=self._dValuesNA,eValues=self._eValuesNA,epsInv=epsInv,inPolarCoords=True)
        #The above functions compute the inverse transformation in SCALED POLAR COORDS
        
        out /= self._scaled
        
        if not outPolar:
            out = self.toCartesian(out)
    
        return out

    def inverseTransformJac(self,xp,vp=None,Jacp=None,cpy=True,outx=None,outv=None,outJac=None, outInvJac:bool=False,epsInv=1e-6):
        """All arguments are expected in cartesian coords"""
        assert outx is None, "TBD, not efficient at the moment"
        assert (xp.shape[0] == self._dim) and ((outx is None) or (outx.shape == xp.shape))
        assert (vp is None) or (xp.shape == vp.shape)
        assert (Jacp is None) or (Jacp.shape == (xp.shape[1],xp.shape[0],xp.shape[0]))
        assert (outx is None) or (xp.shape == outx.shape)
        assert (outv is None) or (xp.shape == outv.shape)
        assert (outJac is None) or (outJac.shape == (xp.shape[1],xp.shape[0],xp.shape[0]))
    
        nPt = xp.shape[1]
    
        doJac = Jacp is not None
        doV = vp is not None
    
        outv = empty(vp.shape) if doV else outv
        outJac = empty(Jacp.shape) if doJac else outJac
        
        #Transform from cartesian coords prime to polar coords prime
        xp = self.toPolar(xp, isRot=False)
        #Compute the jacobian from scaled polar coords prime to cartesian prime
        tempJacPolarPrimeCartPrime = self.inverseIntermediateJac(xp)
        
        if cpy:
            if outx is None:
                outx = xp.copy()
            else:
                np.copyto(outx,xp)
        else:
            outx = xp

        # First get the inverse and the gamma's
        # Scale
        xInPolar = xp.copy()
        xp *= self._scaled
        # c1PolyInvTransMulti(xp, centers, translations, bases, dValues, eValues, epsInv=1e-6, out=None, gammaOut=False, inPolarCoords=False)
        #TBD check if partial function works and is quicker
        if nPt > 100:
            separations = np.linspace(0,nPt,5,dtype=np.int_)
            xList = [xp[:,separations[i]:separations[i+1]] for i in range(4)]
            outx[:] = np.hstack(workerPool.starmap(c1PolyInvTransMulti,
                                                   zip(xList,4*[self._centers],4*[self._translations],4*[self._basesNA],
                                                       4*[self._dValuesNA],4*[self._eValuesNA],4*[epsInv],4*[None],4*[False],4*[True])))
        else:
            outx[:] = c1PolyInvTransMulti(xp,centers=self._centers,translations=self._translations,bases=self._basesNA,
                                          dValues=self._dValuesNA,eValues=self._eValuesNA,epsInv=epsInv,inPolarCoords=True)
        #outx is in SCALED POLAR
        # Once the original points obtained we have to compute the forward transform to get the jacobian
        if (nPt > 100) and (not outInvJac) or doV:
            # The inversed jacobian is needed
            tempJacShared = mt.sctypes.RawArray(mt.c.c_double,nPt*self._dim*self._dim)
            tempJac = np.frombuffer(tempJacShared)
            tempJac.resize((nPt,self._dim,self._dim))
            np.copyto(tempJac,Idn(nPt,self._dim),'safe')
        else:
            tempJac = Idn(nPt,self._dim)
        tempdx = empty(xp.shape)
        for k in self.range():
            # Add up jacobian and translation
            c1PolyTransJac(outx,c=self._centers[:,[k]],t=self._translations[:,[k]],b=self._bases[k],d=self._dValues[k],
                           e=self._eValues[k],outx=tempdx,outJac=tempJac,inPolarCoords=True)
        del tempdx #Dummy
        
        #Transform outx (polar coords) into the rotated coords (No descaling needed)
        outx[1:,:] = xInPolar[1:,:] #outx /= self._scaled
        outx = self.Pol2Rot(outx,False)
        #Now compute the complete jac (in the forward direction) and then inverse it
        intermediateJac = np.einsum("ijk,ikl->ijl",tempJac,self.intermediateJac(outx,isRot=True))
        np.einsum("ijk,ikl->ijl",tempJacPolarPrimeCartPrime,intermediateJac,out=tempJac)
        
        outx = self.toCartesian(outx, isPol=False)
        
        # Unfortunately we have to inverse
        if doV or (doJac and (not outInvJac)):
            tempJacI = mt.slicedInversion(tempJac,cpy=True,NnDim=[nPt,self._dim],returnNp=True)
        # Modify velocity
        if doV:
            # We have to compute v[:,i] = tempJacI[i,:,:].v'[:,i]
            np.einsum("ijk,ki->ji",tempJacI,vp,out=outv)
        if doJac:
            # We have to compute inv(J)[:,i] = inv(tempJac[i,:,:].J'[i,:,:])=inv(J'[i,:,:])).inv(tempJac[i,:,:])
            if outInvJac:
                # This computes the jacobian of the forward transformation (So the inverse of the currently computed trafo)
                np.einsum("ijk,ikl->ijl",tempJac,Jacp,out=outJac)
            else:
                # Jacobian of this trafo
                np.einsum("ijk,ikl->ijl",tempJacI,Jacp,out=outJac)
    
        if doV and doJac:
            return outx,outv,outJac
        elif doV:
            return outx,outv
        elif doJac:
            return outx,outJac
        else:
            return outx
        
    def getAllVarsRaw(self):
        return [self._centers,self._translations[0,:],self._basesNA,self._dValuesNA,self._eValuesNA,self.scaledRaw,self._rotation]

    def getAllVars(self):
        return [self.centers,self.translations[0,:],self.bases,self.dValues,self.eValues, self.scaled[:min(self._dim, 3),0], self.rotation]

    def parsAsVec(self,isPars=None):
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 7*[True]
        
        return super().parsAsVec(isPars=isPars)

    def nVars(self,isPars=None):
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 7*[True]
        return super().nVars(isPars)

    def applyStep(self,deltaPars,isPars=None):
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 7*[True]

        # pars = ['centers', 'translations', 'bases', 'dValues','eValues']
        # execstring = """old=self.{0}; nEnd=nStart+old.size; self.{0} = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;"""
        # for k,do,par in enumerate(zip(isPars,pars)):
        #    if do:
        #        old=getattr(self, par); nEnd=nStart+old.size; setattr(self, par) = old+deltaPars[nStart:nEnd].reshape(old.shape, order='F');nStart=nEnd;
        nStart = 0
        if isPars[0]:
            old = self.centers;
            nEnd = nStart+old.size;
            self.centers = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F');
            nStart = nEnd;
        if isPars[1]:
            old = self.translations[0,:]
            nEnd = nStart+old.size
            old = self.translations
            old[0,:] += deltaPars[nStart:nEnd].reshape((nEnd-nStart,),order='F')
            self.translations = old
            nStart = nEnd
        if isPars[2]:
            old = self.bases;
            nEnd = nStart+old.size;
            self.bases = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F');
            nStart = nEnd;
        if isPars[3]:
            old = self.dValues;
            nEnd = nStart+old.size;
            self.dValues = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F');
            nStart = nEnd;
        if isPars[4]:
            old = self.eValues;
            nEnd = nStart+old.size;
            self.eValues = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F');
            nStart = nEnd;
        if isPars[5]:
            old = self.scaledRaw
            nEnd = nStart+old.size
            self.scaledRaw = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F');
        if isPars[6]:
            assert 0, 'TBD'
        
        return 0

    def applyConstrainedStep(self,deltaPars,isPars=None,marginSingle=.5,marginOverlap=.2):
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 7*[True]
    
        alphaU = 1.0
    
        copied = self.__copy__()
        
        try:
            assert np.all(self.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap) > 0.)
        except:
            print('BBB')
    
        try:
            alphaL = 1.
            # nStart = 0
            # for k,do in enumerate(isPars):
            #    if do:
            #        exec(execstringC.format(pars[k]),{},locals())
            nStart = 0
            if isPars[0]:
                old = self.centers
                nEnd = nStart+old.size
                copied.centers = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                nStart = nEnd
            if isPars[1]:
                old = self.translations[0,:]
                nEnd = nStart+old.size
                old = self.translations
                old[0,:] += deltaPars[nStart:nEnd].reshape((nEnd-nStart,),order='F')
                copied.translations = old
                nStart = nEnd
            if isPars[2]:
                old = self.bases
                nEnd = nStart+old.size
                copied.bases = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                nStart = nEnd
            if isPars[3]:
                old = self.dValues
                nEnd = nStart+old.size
                copied.dValues = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                nStart = nEnd
            if isPars[4]:
                old = self.eValues
                nEnd = nStart+old.size
                copied.eValues = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                nStart = nEnd
            if isPars[5]:
                old = self.scaledRaw
                nEnd = nStart+old.size
                copied.scaledRaw = old+deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                nStart = nEnd
            if isPars[6]:
                assert 0, 'TBD'
                
        
            assert np.all(copied.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap) > 0.)
        except:
            alphaL = 0.
            while ((alphaU-alphaL>0.05) or alphaL==0) and (alphaU>1e-16):
                alpha = (alphaU+alphaL)/2.
                # nStart = 0
                # for k,do in enumerate(isPars):
                #    if do:
                #        exec(execstringCA.format(pars[k]),{},locals())
                nStart = 0
                if isPars[0]:
                    old = self.centers
                    nEnd = nStart+old.size
                    copied.centers = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                    nStart = nEnd
                if isPars[1]:
                    old = self.translations[0,:]
                    nEnd = nStart+old.size
                    old = self.translations
                    old[0,:] += alpha*deltaPars[nStart:nEnd].reshape((nEnd-nStart,),order='F')
                    copied.translations = old
                    nStart = nEnd
                if isPars[2]:
                    old = self.bases
                    nEnd = nStart+old.size
                    copied.bases = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                    nStart = nEnd
                if isPars[3]:
                    old = self.dValues
                    nEnd = nStart+old.size
                    copied.dValues = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                    nStart = nEnd
                if isPars[4]:
                    old = self.eValues
                    nEnd = nStart+old.size
                    copied.eValues = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                    nStart = nEnd
                if isPars[5]:
                    old = self.scaledRaw
                    nEnd = nStart+old.size
                    copied.scaledRaw = old+alpha*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
                    nStart = nEnd
                if isPars[6]:
                    assert 0,'TBD'
            
                if np.all(copied.dist2nondiffeo(marginSingle=marginSingle,marginOverlap=marginOverlap) > 0.):
                    alphaL = alpha
                else:
                    alphaU = alpha
        # nStart = 0
        # for k,do in enumerate(isPars):
        #    if do:
        #        exec(execstringAL.format(pars[k]),{},locals())
        try:
            assert alphaL>0.
        except:
            print("AAA")
        nStart = 0
        if isPars[0]:
            old = self.centers
            nEnd = nStart+old.size
            self.centers = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
            nStart = nEnd
        if isPars[1]:
            old = self.translations[0,:]
            nEnd = nStart+old.size
            old = self.translations
            old[0,:] += alphaL*deltaPars[nStart:nEnd].reshape((nEnd-nStart,),order='F')
            self.translations = old
            nStart = nEnd
        if isPars[2]:
            old = self.bases
            nEnd = nStart+old.size
            self.bases = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
            nStart = nEnd
        if isPars[3]:
            old = self.dValues
            nEnd = nStart+old.size
            self.dValues = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
            nStart = nEnd
        if isPars[4]:
            old = self.eValues
            nEnd = nStart+old.size
            self.eValues = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
            nStart = nEnd
        if isPars[5]:
            old = self.scaledRaw
            nEnd = nStart+old.size
            self.scaledRaw = old+alphaL*deltaPars[nStart:nEnd].reshape(old.shape,order='F')
            nStart = nEnd
        if isPars[6]:
            assert 0,'TBD'
        return alphaL

    def getMSE(self,sourcex,targetx,dir=True,sourcev=None,targetv=None,xvtradeoff=None,epsInv=1e-6,mseOfPolarCoords=False, targetIsPolar=False):
        """source and target are assumed to be in cartesian coords"""

        if mseOfPolarCoords and not targetIsPolar:
            targetx = self.toPolar(targetx)
        elif targetIsPolar and not mseOfPolarCoords:
            targetx = self.toCartesian(targetx)
    
        if sourcev is None:
            if dir:
                targetxtilde = self.forwardTransform(sourcex,outPolar=mseOfPolarCoords)
            else:
                targetxtilde = self.inverseTransform(sourcex,epsInv=epsInv,outPolar=mseOfPolarCoords)
            #TBD this must be changed if one wants to consider spirals or similar
            if mseOfPolarCoords:
                err = targetxtilde - self.toPolar(targetx)
                wrapAngles(err[0,:],err[0,:])
                mse = compMSE(err)
            else:
                targetxtilde -= targetx
                mse = compMSE(targetxtilde)
                
        else:
            assert 0, "STH smart needs to be done with the displacement in this case"
            if dir:
                targetxtilde,targetvtilde = self.forwardTransformJac(sourcex,sourcev)
            else:
                targetxtilde,targetvtilde = self.inverseTransformJac(sourcex,sourcev,epsInv=epsInv)
            targetxtilde -= targetx
            targetvtilde -= targetvtilde
            mse = np.mean(sum(square(targetxtilde),0))+xvtradeoff*np.mean(sum(square(targetvtilde),0))
    
        return mse

    def getMSEnJac(self,sourcex,targetx=None,targetxtilde=None,dir=True,isPars=None,err=None,mse=None, alphaConstraints=None, epsConstraints=None):
        # TBD: sourcev/targetv and dir==False
        """source and target/err are assumed to be in cartesian coords"""
        """MSE will always be computed for the cartesian coords"""
        #source is in cartesian coords
    
        dim,nPt = sourcex.shape
        assert self._dim == dim
        assert (targetx is None) or (sourcex.shape == targetx.shape)
        assert (err is None) or (sourcex.shape == err.shape)
        assert (targetx is None) or (err is None)
    
        isPars = self._isPars if isPars is None else isPars
        if not isinstance(isPars,(list,tuple)) and isPars:
            isPars = 7*[True]
        # isPars: [centers, translations,bases,dValues,eValues]
        if dir:
            if isPars[5]:
                #If we modify scale we need the jac
                targetxtilde, JacScaledTransform = self.forwardTransformJac(sourcex,alsoReturnTrafoJac=True) #JacScaledTransform is the jacobian of the transformation in scaled polar coords
            else:
                targetxtilde = self.forwardTransform(sourcex) if targetxtilde is None else targetxtilde
            if err is None:
                err = targetx-targetxtilde #Error in cartesian prime coords
            if mse is None:
                mse = compMSE(err)

            # Bring the error from cartesian prime coords to scaled polar prime coords
            # To compute the jacobian of the intermediate transformation we need
            # the prime polar coords
            targetxtildePolarPrime = self.toPolar(targetxtilde)
            JintermediateBackwardPrime = self.inverseIntermediateJac(targetxtildePolarPrime)
            #We have to compute (err.T*Jac).T so Jac.T.err
            JintermediateBackwardPrime = np.transpose(JintermediateBackwardPrime,axes=(0,2,1))
            errScaledPolarPrime = np.einsum("ijk,ki->ji",JintermediateBackwardPrime,err)

            # Transform sourcex into scaled polar
            if isPars[5]:
                sourcexpolar = self.toPolar(sourcex)
                sourcexscaled = sourcexpolar*self._scaled
                # We also need the prime scaled polar coords
                #targetxtildeScaledPrime = targetxtildePolarPrime*self._scaled
            else:
                sourcexscaled = self.toPolar(sourcex)*self._scaled
            
            #Get the partial derivatives of the transformation in the scaled polar space
            valD = self.kValnPdiff(sourcexscaled,dir=True, isPolar=True, isScaled=True)
        
            allJac = []
            if isPars[0]:
                jacCenters = empty(self._centers.shape)
                for k in self.range():
                    dxcknorm = sourcexscaled-self._centers[:,[k]]
                    dxcknorm /= cNorm(dxcknorm)
                    jacCenters[:,k] = 2*np.mean((sum(errScaledPolarPrime*self._translations[:,[k]],axis=0)*valD['kdVal'][k,:])*dxcknorm,
                                                axis=1)
                allJac.append(jacCenters.flatten(order='F'))
        
            if isPars[1]:
                jacTranslation = empty(self._translations.shape)
                for k in self.range():
                    jacTranslation[:,k] = -2.*np.mean(valD['kVal'][k,:]*errScaledPolarPrime,axis=1)
                # TBD improve eff
                # The above computes the jacobian for each element of translation; However we keep only the angle dimension 0
                allJac.append(jacTranslation[0,:].flatten(order='F'))
        
            if np.any(isPars[2:5]):
                subJac = [empty(self._basesNA.shape),empty(self._dValuesNA.shape),empty(self._eValuesNA.shape)]
                for k in self.range():
                    #TBD increase eff
                    errdottk = -2*sum(errScaledPolarPrime*self._translations[:,[k]],axis=0)
                    for apar,do,aJac in zip(['b','d','e'],isPars[2:],subJac):
                        if do:
                            aJac[k] = np.mean(errdottk*valD[apar][k,:])
                for do,aJac in zip(isPars[2:],subJac):
                    if do:
                        allJac.append(aJac.flatten(order='F'))
            if isPars[5]:
                #assert 0, "Sth is wrong"
                # This one is special though
                # Get the two Jacobians needed and directly transpose them
                Jintermediate1 = np.transpose(np.einsum("jk,ikl->ijl", self._rotation.T, self.intermediateJacPol2Rot(targetxtildePolarPrime)),(0,2,1))
                #The second jacobian includes the entire inverseintermediate Jacobian plus (so right multiplied) with the
                #jacobian of the transformation itself
                Jintermediate0 = np.transpose(JacScaledTransform,(0,2,1))
                #Get the corresponding error
                error1 = np.einsum("ijk,ki->ji", Jintermediate1, err)
                error0 = np.einsum("ijk,ki->ji", Jintermediate0, errScaledPolarPrime)

                #Compute the Jacobian
                scaleJac = zeros((self.scaledRaw.size,))

                scaleJac[1] += np.mean(error0[1,:]*sourcexpolar[1,:])
                # Attention normally it is error.T * primed scaled polar coords / scale^2, but scaled polar coords is polar coords * scale -> simplify
                scaleJac[1] -= np.mean(error1[1,:]*targetxtildePolarPrime[1,:])/self._scaled[1]
                if self._dim > 2:
                    # If data is 2d than there is only radius and angle
                    # So no additional perpendicular direction exists -> nothing to do here
                    scaleJac[2] += np.mean(sum(error0[2:,:]*sourcexpolar[2:,:],axis=0,keepdims=False))
                    scaleJac[2] -= np.mean(sum(error1[2:,:]*targetxtildePolarPrime[2:,:],axis=0,keepdims=False))/self._scaled[2]



                #allJac.append(scaleJac.flatten(order='F'))
                # Inverse sign; should not be correct but seems to work
                allJac.append(-scaleJac.flatten(order='F'))
            if isPars[6]:
                assert 0, "rotation is TBD"
        
            allJac = np.hstack(allJac)
        else:
            assert 0,"dir==False TBD"
    
        return mse,allJac
    







        
        
        
