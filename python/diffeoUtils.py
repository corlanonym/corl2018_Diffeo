from coreUtils import *
import polyDiffeo as pd
from copy import deepcopy
import multiprocessingTools as mt

import sys

def pprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class diffeomorphism():
    def __init__(self, dim, transList = [], dirList=[], isPars=None):
        self._dim = dim # type : int
        self._isPars = isPars # type : List[bool]
        self._transformationList = [] # type : List[pd.localPolyMultiTranslation]
        self._directionList = [] # type : List[bool]
        self.addTransformation(transList, dirList)
        
    
    def __copy__(self):
        new = self.__class__(self._dim, lmap(lambda aTrans: aTrans.__copy__(), self._transformationList), deepcopy(self._directionList), self._isPars)
        return new
    def __deepcopy__(self, memodict={}):
        return self.__copy__()
    def __getitem__(self, item):
        return self._transformationList[item]
    
    def toStringList(self):
        
        totStringList = []

        # expected format
        # dimension
        # number of multitrans
        # ##for each multitrans
            # direction ##Inserted here
            # number of kernels ##given from multitrans
            # nPars multitrans+number of kernels*parsPerKernel
        
        totStringList.append(int2Str(self._dim))
        totStringList.append(int2Str(len(self._transformationList)))
        
        for aDir, aTrans in zip(self._directionList, self._transformationList):
            totStringList.append(int2Str(aDir))
            totStringList += aTrans.toStringList()
        
        return totStringList
    
    def toText(self, fileName):
        
        with open(fileName, 'w+') as file:
            file.writelines(self.toStringList())
        return True

    def range(self):
        return range(len(self._transformationList))
    
    def addTransformation(self, aTransList, dirList):
        
        if not isinstance(aTransList, (list, tuple)):
            aTransList = [aTransList]
            dirList = [dirList]
        
        for aTrans, aDir in zip(aTransList, dirList):
            assert self._dim == aTrans._dim
            assert aDir in (True,False)
        
        self._transformationList += aTransList
        self._directionList += dirList
    
    def addDiffeo(self, other:'diffeomorphism'):
        assert self._dim == other._dim
        
        self.addTransformation(other._transformationList, other._directionList)
    
    def removeTransformation(self, removeList):
        
        if isinstance(removeList, int):
            removeList = [removeList]
        
        for arl in removeList:
            self._transformationList.pop(arl)
            self._directionList.pop(arl)
    
    def insertTransformation(self, aTransList, dirList, indList):
        if not isinstance(aTransList, (tuple,list)):
            aTransList = [aTransList]
            dirList = [dirList]
            indList = [indList]
        
        for aTrans, aDir, aInd in zip(aTransList, dirList, indList):
            self._transformationList.insert(aInd, aTrans)
            self._directionList.insert(aInd, aDir)
    
    @property
    def nTrans(self):
        return len(self._transformationList)

    @property
    def alphaConstraints(self):
        return self._transformationList[0]._alphaConstraints

    @alphaConstraints.setter
    def alphaConstraints(self,newAlpha):
        for aTrans in self._transformationList:
            aTrans._alphaConstraints = newAlpha

    @property
    def epsConstraints(self):
        return self._transformationList[0]._epsConstraints

    @epsConstraints.setter
    def epsConstraints(self, newEps):
        assert newEps>0.
        for aTrans in self._transformationList:
            aTrans._epsConstraints = newEps

    @property
    def margins(self):
        return deepcopy(self._transformationList[0]._margins)

    @margins.setter
    def margins(self,newMargins):
        newMargins = list(newMargins)
        assert len(newMargins) == 2
        for aTrans in self._transformationList:
            aTrans.margins = newMargins
            
    def margins(self, newMargins):
        newMargins = list(newMargins)
        assert len(newMargins)==2
        for aTrans in self._transformationList:
            aTrans._margins = newMargins
    
    def enum(self):
        return enumerate(zip(self._directionList, self._transformationList))
    def zip(self):
        return zip(self._directionList,self._transformationList)
    def list(self):
        return list(self.zip())
    
    def forwardTransform(self, x, out=None, kStart=0, kStop=-1):
        """Apply all forward transformations"""
        if out is None:
            out = x.copy()
        else:
            np.copyto(out, x, 'no')
        
        kStop = self.nTrans if (kStop==-1) else kStop
        
        for k, [aDir, aTrans] in self.enum():
            if (k<kStart) or (k>kStop):
                continue
            if aDir:
                out = aTrans.forwardTransform(out)
            else:
                out = aTrans.inverseTransform(out)
        
        return out

    def inverseTransform(self,xp,out=None,kStart=0,kStop=-1):
        """Apply all inverse transformations"""
        if out is None:
            out = xp.copy()
        else:
            np.copyto(out,xp,'no')

        kStop = self.nTrans if (kStop == -1) else kStop
 
        for k, (aDir,aTrans) in enumerate(reversed(self.list())):
            if (k<kStart) or (k>=kStop):
                continue
            if not aDir:
                out = aTrans.forwardTransform(out)
            else:
                out = aTrans.inverseTransform(out)
    
        return out
    
    def forwardTransformJac(self, x, v=None, Jac=None, outx=None, outv=None, outJac=None, kStart=0, kStop=-1, outInvJac=False, whichSide='left'):
        
        """
        
        :param x:
        :param v:
        :param Jac:
        :param outx:
        :param outv:
        :param outJac:
        :param kStart:
        :param kStop:
        :param outInvJac: If True, the jacobian of this forward transformation is returned, if False the jacobian of the corresponding inverse transformation is handed back
        :param whichSide: left or right multiplication of the jacobian of this diffeo with given jac
        :return:
        """
        
        assert x.shape[0] == self._dim
        assert whichSide in ('left', 'right')

        if not outInvJac == {'left':False,'right':True}[whichSide]:
            warnings.warn('Except special cases whichSide should be left if outInvJac False or right and true')

        kStop = self.nTrans if (kStop == -1) else kStop
        
        dim, nPt = x.shape
        
        outx = x.copy()
        doV = v is not None
        doJac = Jac is not None
        
        if doJac:
            try:
                assert np.all( Jac.shape == [nPt, dim, dim] )
            except:
                Jac = Idn(nPt,dim)

        tempJac = Idn(nPt,dim)

        for k, [aDir, aTrans] in self.enum():
            if (k<kStart) or (k>kStop):
                continue
            if aDir:
                outx, tempJac = aTrans.forwardTransformJac(x=outx, Jac=tempJac, outInvJac=False, whichSide='left')
            else:
                outx, tempJac = aTrans.inverseTransformJac(xp=outx, Jacp=tempJac, outInvJac=False, whichSide='left')

        if doV:
            outv = empty(v.shape) if outv is None else outv
            # We have to compute vp[:,i] = tempJac[i,:,:].v[:,i]
            np.einsum("ijk,ki->ji", tempJac, v, out=outv)
        if doJac:
            outJac = empty(Jac.shape) if outJac is None else outJac
            # We have to compute J'[:,i] = tempJac[i,:,:].J[i,:,:]
            if outInvJac:
                # Invert the forward jacobian of this diffeo in order to get the jacobian of the inverse transformation
                # Attention : result will still be leftmultiplied with input Jac
                tempJac = mt.slicedInversion(tempJac)
            if whichSide=='left':
                np.einsum("ijk,ikl->ijl", tempJac, Jac, out=outJac)
            else:
                np.einsum("ijk,ikl->ijl",Jac,tempJac,out=outJac)

        if doV and doJac:
            return outx, outv, outJac
        elif doV:
            return outx, outv
        elif doJac:
            return outx, outJac
        else:
            return outx
    
    def inverseTransformJac(self, xp, vp=None, Jacp=None, outx=None, outv=None, outJac=None, outInvJac:bool=False, kStart=0, kStop=-1, whichSide:"left or right"='left'):
        """
        
        :param xp:
        :param vp:
        :param Jacp:
        :param outx:
        :param outv:
        :param outJac:
        :param outInvJac: If True, the jacobian of the corresponding forward transformation is handed back; So its the inverse of the actual jacobian (the jacobian of the inverse transformation)
        :param kStart:
        :param kStop:
        :return:
        """
        assert whichSide in ('left', 'right')
        assert xp.shape[0] == self._dim

        if not (outInvJac == {'left':False,'right':True}[whichSide]):
            warnings.warn('Except special cases whichSide should be left if outInvJac False or right and true')

        kStop = self.nTrans if (kStop == -1) else kStop
    
        dim,nPt = xp.shape
    
        outx = xp.copy()
        doV = vp is not None
        doJac = Jacp is not None
    
        tempJac = Idn(nPt,dim)
        
        # Decide on multiplication side
        # If we build the inverse jacobian (outInvJac==True), so the jacobian of the forward transformation we have to right multiply
        # otherwise we can as usual left multiply
        # Attention on velocity transformation
        if outInvJac:
            whichSideInter='right'
        else:
            whichSideInter = 'left'
        
        for k, [aDir,aTrans] in enumerate(reversed(self.list())):
            if (k<kStart) or (k>kStop):
                continue
            if not aDir:
                outx,tempJac = aTrans.forwardTransformJac(outx,Jac=tempJac,outInvJac=outInvJac, whichSide=whichSideInter)
            else:
                outx,tempJac = aTrans.inverseTransformJac(outx,Jacp=tempJac,outInvJac=outInvJac, whichSide=whichSideInter)
        
        if doV:
            if outInvJac:
                #The jacobian stored in tempJac is the jacobian of the inverse of the called transformation
                tempJacI = mt.slicedInversion(tempJac,NnDim=[nPt,dim])
            else:
                #The jacobian in tempJac is the jacobian of this transformation
                tempJacI = tempJac
            # We have to compute v[:,i] = tempJacI[i,:,:].v'[:,i]
            outv = empty(vp.shape) if outv is None else outv
            np.einsum("ijk,ki->ji",tempJacI,vp,out=outv)
        if doJac:
            # We have to compute inv(J)[:,i] = inv(tempJac[i,:,:].J'[i,:,:])=inv(J'[i,:,:])).inv(tempJac[i,:,:])
            outJac = empty(Jacp.shape) if outJac is None else outJac
            if whichSide == 'left':
                np.einsum("ijk,ikl->ijl",tempJac,Jacp,out=outJac)
            else:
                np.einsum("ijk,ikl->ijl",Jacp,tempJac,out=outJac)
        
        if doV and doJac:
            return outx, outv, outJac
        elif doV:
            return outx, outv
        elif doJac:
            return outx, outJac
            return outx, outJac
        else:
            return outx
    
    def parsAsVec(self):
        allPars = []
        nPars = [0]
        
        for aTrans in self._transformationList:
            allPars.append( aTrans.parsAsVec(self._isPars) )
            nPars.append( allPars[-1].size )
        
        return allPars, nPars
    
    def applyStep(self, deltaVars, isPars=None):
        
        isPars = self._isPars if isPars is None else isPars
        
        nStart = 0
        for aTrans in self._transformationList:
            nEnd = nStart+aTrans.nVars(isPars)
            aTrans.applyStep(deltaPars=deltaVars[nStart:nEnd], isPars=isPars)
            nStart = nEnd
        
        return 0
    
    def applyStepConstrainedStep(self,deltaVars,eachAlpha = None, isPars=None,marginSingle=.5,marginOverlap=.2):
        isPars = self._isPars if isPars is None else isPars
        
        if isinstance(marginSingle, float):
            marginSingle = self.nTrans*[marginSingle]
        if isinstance(marginOverlap,float):
            marginOverlap = self.nTrans*[marginOverlap]
           
        nStart=0
        alphaApply = []
        eachAlpha = ones((self.nTrans,)) if (eachAlpha is None) else eachAlpha
        for k,(_,aTrans) in self.enum():
            nEnd = nStart+aTrans.nVars(isPars)
            #TBD straighten the use of pars and vars
            alphaApply.append( aTrans.applyConstrainedStep(deltaPars=deltaVars[nStart:nEnd]*eachAlpha[k], marginSingle=marginSingle[k], marginOverlap=marginOverlap[k], isPars=isPars) )
            nStart = nEnd
            
        return alphaApply
    
    def enforceMargins(self, marginSingle:Union[float, List[float], None]=None, marginOverlap:Union[float, List[float], None]=None, dist2Zero:Union[float, List[float]]=0.)->bool:
        
        marginSingle = self.margins[0] if (marginSingle is None) else marginSingle
        marginOverlap = self.margins[1] if (marginOverlap is None) else marginOverlap
        
        if not isinstance(marginSingle, (list, tuple)):
            marginSingle = self.nTrans*[marginSingle]
        if not isinstance(marginSingle,(list,tuple)):
            marginSingle = self.nTrans*[marginSingle]
        if not isinstance(dist2Zero,(list,tuple)):
            dist2Zero = self.nTrans*[dist2Zero]
        
        res = True
        for k, (_,aTrans) in self.enum():
            res = res and aTrans.enforceMargins(marginSingle[k], marginOverlap[k], dist2Zero[k])
        return res
        
    
    def getMSE(self, sourcex=None, targetx=None, dir=True, epsInv=inversionEps, epsConstraints=None, alphaConstraints=None, marginSingle=None, marginOverlap=None, err=None, useBarrier=False):
        
        assert (err is None) or ((sourcex is None) and (targetx is None)), "Either source and target or error has to be given, not both"
        
        print('getMSE')
        
        if err is None:
            if dir:
                targetxtilde = self.forwardTransform(sourcex)
            else:
                targetxtilde = self.inverseTransform(sourcex, epsInv=epsInv)
            err = targetx-targetxtilde
        
        addMSE = 0.
        if useBarrier:
            if isinstance(epsConstraints, (float, int)):
                epsConstraints = self.nTrans*[epsConstraints]
            if isinstance(marginSingle,(float,int)):
                marginSingle = self.nTrans*[marginSingle]
            if isinstance(marginOverlap, (float, int)):
                marginOverlap = self.nTrans*[marginOverlap]
            for k, [_,aTrans] in self.enum():
                thisEpsConstraints = aTrans._epsConstraints if (epsConstraints is None) else epsConstraints[k]
                thisAlphaConstraints = aTrans._alphaConstraints if (alphaConstraints is None) else alphaConstraints
                if thisEpsConstraints is not None:
                    thisMarginSingle = aTrans._margin[0] if marginSingle is None else marginSingle[k]
                    thisMarginOverlap = aTrans._margin[1] if marginSingle is None else marginOverlap[k]
                    # Use truncated log barrier
                    # The barrier is active between 0 and 3*eps
                    # barrier to keep zero zero
                    dist2Zero = cNorm(aTrans._centers,kd=False)-aTrans._basesNA
                    ind2Zero = dist2Zero < 3*thisEpsConstraints
                    dist2Zero[np.logical_not(ind2Zero)] = 3*thisEpsConstraints
                    dist2Zero /= thisEpsConstraints
                    barrier2Zero = -thisAlphaConstraints*np.log(dist2Zero)
                    # barrier to ensure diffeo
                    dist2NonDiffeo,isOverlapNondiffeo = aTrans.dist2nondiffeo(marginSingle=thisMarginSingle, marginOverlap=thisMarginOverlap,fullOut=True)
                    ind2NonDiffeo = dist2NonDiffeo < 3*thisEpsConstraints
                    dist2NonDiffeo[np.logical_not(ind2NonDiffeo)] = 3*thisEpsConstraints
                    dist2NonDiffeo /= thisEpsConstraints
                    barrier2NonDiffeo = -thisAlphaConstraints*np.log(dist2NonDiffeo)
                    
                    addMSE += np.sum(barrier2Zero) + np.sum(barrier2NonDiffeo)
                
        return compMSE(err)+addMSE
    
    def getMSEnJac(self, sourcex, targetx, dir=True, epsInv=inversionEps, alphaConstraints=None, epsConstraints=None, marginSingle=None, marginOverlap=None, useBarrier=False):
        
        # Attention in order to be efficient this function we need "a lot of" memory
        dim,nPt = sourcex.shape
        assert  sourcex.shape==targetx.shape
        assert dim==self._dim
        assert dir, 'Inverse direction TBD'

        if isinstance(epsConstraints,(float,int)):
            epsConstraints = self.nTrans*[epsConstraints]
        
        if dir:
            allxJac = [[sourcex,Idn(nPt, dim)]]
            
            #Compute ALL intermediate x and Jac
            for aDir,aTrans in self.zip():
                if aDir:
                    allxJac.append( list(aTrans.forwardTransformJac(allxJac[-1][0],Jac=Idn(nPt, dim))) )
                else:
                    allxJac.append( list(aTrans.inverseTransformJac(allxJac[-1][0],Jacp=Idn(nPt,dim))) )
            
            #Get error and MSE:
            err = targetx-allxJac[-1][0]
            mse = self.getMSE(dir=dir, epsInv=epsInv, alphaConstraints=alphaConstraints, epsConstraints=epsConstraints, marginSingle=marginSingle, marginOverlap=marginOverlap, err=err, useBarrier=useBarrier)
            
            # Now backpropagate and compute gradient
            allxJac.append([0.,Idn(nPt,dim)])
            # We need the transposed jacobian
            allposGrad = [] #We will build it in reversed order and than reverse
            for k,(ax,aJ) in enumerate(allxJac):
                allxJac[k][1] = np.transpose(aJ, axes=(0,2,1))
            for k, (aDir, aTrans) in reversed(list(self.enum())):
                epsConstraintsk = aTrans._epsConstraints if epsConstraints is None else epsConstraints[k]
                if epsConstraintsk is not None:
                    alphaConstraintsk = aTrans._alphaConstraints if alphaConstraints is None else alphaConstraints
                    marginSinglek = aTrans._margin[0] if marginSingle is None else marginSingle[k]
                    marginOverlapk = aTrans._margin[1] if marginSingle is None else marginOverlap[k]

                #Transform the error
                #Compute err_k = J_k+1^T . err_k+1
                err = np.einsum("ijk,ki->ji",allxJac[k+2][1],err) #k+2 since source is index 0
                #Get this jacobian / positive gradient
                #getMSEnJac(self, sourcex,targetx,dir=True,isPars=None, err=None, mse=None):
                allposGrad.append( aTrans.getMSEnJac( allxJac[k][0], dir=aDir, err=err, mse=mse, isPars=self._isPars, alphaConstraints=alphaConstraintsk, epsConstraints=epsConstraintsk, marginSingle=marginSinglek, marginOverlap=marginOverlapk )[1] )#Only take the jacobian not the mse
            #Reverse
            allposGrad.reverse()
            
            allGrad = np.hstack(allposGrad)
        else:
            pass
        
        return mse, allGrad

    def backTrackStep(self,sourcex,targetx,dir=True,alpha=50.,tau=.5,c=1e-2,p='grad', alphaConstraints=None, epsConstraints=None, marginSingle=None,marginOverlap=None):
    
        pprint("bstep a")
        
        marginSingle = self.margins[0] if (marginSingle is None) else marginSingle
        marginOverlap = self.margins[1] if (marginOverlap is None) else marginOverlap
        alphaConstraints = self.alphaConstraints if (alphaConstraints is None) else alphaConstraints
        epsConstraints = self.epsConstraints if (epsConstraints is None) else epsConstraints

        pprint("bstep b")

        if isinstance(marginSingle, float):
            marginSingle = self.nTrans*[marginSingle]
        if isinstance(marginOverlap, float):
            marginOverlap = self.nTrans*[marginOverlap]
        if isinstance(epsConstraints,float):
            epsConstraints = self.nTrans*[epsConstraints]
        assert isinstance(alphaConstraints, float)

        pprint("bstep c")
        
        assert dir, "Inverse direction TBD"

        fx,grad = self.getMSEnJac(sourcex,targetx,dir=True,alphaConstraints=alphaConstraints,epsConstraints=epsConstraints,marginSingle=marginSingle,marginOverlap=marginOverlap, useBarrier=True)

        pprint("bstep d")

        if p == 'grad':
            p = -grad  # Search direction

        t = c*np.inner(p,grad)
        
        applyFunc = np.max

        pprint("bstep e")
        
        copied = self.__copy__()
        pprint("bstep 1")
        alphaApply = na(copied.applyStepConstrainedStep(alpha*p, marginSingle=marginSingle,marginOverlap=marginOverlap))
        pprint("bstep 11")
        alphaApplyOld = deepcopy(alphaApply)
        fx1 = copied.getMSE(sourcex,targetx,dir=dir,alphaConstraints=alphaConstraints,epsConstraints=epsConstraints,marginSingle=marginSingle,marginOverlap=marginOverlap,useBarrier=True)
        pprint("bstep 12")
        cZero=False
        while (fx1 >= fx+alpha*t): # Use mean or min or max ??
            assert alpha*t < 0.
            if alpha<1e-6:
                cZero = True
                break
            copied = self.__copy__()
            pprint("bstep 13")
            alpha = alpha*applyFunc(alphaApply)*tau
            pprint("current alpha :{0}".format(alpha))
            alphaApplyOld = deepcopy(alphaApply)
            alphaApply = na(copied.applyStepConstrainedStep(p,eachAlpha=alpha*alphaApply,marginSingle=marginSingle,marginOverlap=marginOverlap))
            pprint("bstep 14")
            fx1 = copied.getMSE(sourcex,targetx,alphaConstraints=alphaConstraints,epsConstraints=epsConstraints,marginSingle=marginSingle,marginOverlap=marginOverlap, useBarrier=True)
            pprint("bstep 15")
        
        if cZero:
            alpha=0
            alphaApplyOld=0
            fx1=fx

        alphaApplyFinal = self.applyStepConstrainedStep(p,eachAlpha=alpha*alphaApplyOld,marginSingle=marginSingle,marginOverlap=marginOverlap)
        fxFinal = self.getMSE(sourcex,targetx,alphaConstraints=alphaConstraints,epsConstraints=epsConstraints,marginSingle=marginSingle,marginOverlap=marginOverlap, useBarrier=True)
        if  np.any(na(alphaApplyFinal)<0.95*alphaApply) or fxFinal > 1.1*fx1 or fxFinal< 0.9*fx1:
            assert 0
            pprint("shit")

        # Test
        if not self.getMSE(sourcex,targetx,alphaConstraints=alphaConstraints,epsConstraints=epsConstraints,
                           marginSingle=marginSingle,marginOverlap=marginOverlap, useBarrier=True) < fx+.99*alpha*t:
            assert 0
            pprint("double shit")
        
        # Ensure robustness of boundaries
        if (epsConstraints is not None)  and (alphaConstraints is not None):
            assert self.enforceMargins(marginSingle=list(na(marginSingle)+na(epsConstraints)), marginOverlap=list(na(marginOverlap)+na(epsConstraints)), dist2Zero=epsConstraints)

        return fx1,applyFunc(alpha*alphaApply)
    
    
    def optimize(self, xSource:np.ndarray, xTarget:np.ndarray, options:dict={}):

        _options = {'maxIter':20, 'relTol':1e-2, 'absTol':-5e-2, 'tau':0.5, 'c':1e-2,
                    'alpha':50.,
                    'alphaConstraints':-5.e-2, 'epsConstraints':3.e-2,
                    'marginSingle':.5, 'marginOverlap':.2}
        
        _options.update(options)

        # Get the initial error
        initMSE = compMSE( self.forwardTransform(xSource)-xTarget )

        limAbsMSE = _options['absTol'] if (_options['absTol'] > 0.)  else -_options['absTol']*initMSE

        lastMSE = 2e300
        newMSE = 1e300
        thisIter = 0

        thisAlphaConstraint = -_options['alphaConstraints']*initMSE if (_options['alphaConstraints'] < 0.) else _options['alphaConstraints']
        
        self.getMSE(xSource, xTarget, marginSingle=_options['marginSingle'], marginOverlap=_options['marginOverlap'], alphaConstraints=thisAlphaConstraint, epsConstraints=_options['epsConstraints'], useBarrier=True)
        
        alpha = _options['alpha']
        while (thisIter < _options['maxIter']) and ((lastMSE-newMSE)>limAbsMSE) and ((lastMSE-newMSE)/lastMSE > _options['relTol']):
            #Do a step
            lastMSE = newMSE
            newMSE,alphaApply = self.backTrackStep(xSource,xTarget, alpha=alpha,tau=_options['tau'],c=_options['c'], marginSingle=_options['marginSingle'], marginOverlap=_options['marginOverlap'], alphaConstraints=thisAlphaConstraint, epsConstraints=_options['epsConstraints'])
            if newMSE > lastMSE:
                pprint("Problem with convergence")
                try:
                    assert 0
                except:
                    pprint('AAA')
                break
            alpha = 4.*alphaApply
            thisIter += 1
        
        return self


        
        
        
        
                
            
                    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    