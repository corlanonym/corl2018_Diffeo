from coreUtils import *
import multiprocessing as mp
from multiprocessing import sharedctypes as sctypes
import ctypes as c

#Get a fairly large shared array
_Ashared = sctypes.RawArray(c.c_double,10000000) #ten millions values are about 40MB

def slicedInverseWorker(Ain, N, dim, nStart=0, nEnd=None):
    
    nEnd = nEnd if nEnd is not None else N
    if isinstance(Ain, np.ndarray):
        AinAsNp=Ain
    else:
        if len(Ain)==N*dim*dim:
            AinAsNp = np.frombuffer(Ain)
        else:
            AinAsNp = np.frombuffer(Ain)[:N*dim*dim]
        AinAsNp.resize(N,dim,dim)
    for n in range(nStart, nEnd):
        AinAsNp[n,:,:] = inv(AinAsNp[n,:,:], overwrite_a=True, check_finite=False)
    
    return 0

#slicedInverseWorkerInherit = lambda N,dim,nStart=0,nEnd=None : slicedInverseWorker(_Ashared,N,dim,nStart,nEnd)
def slicedInverseWorkerInherit(N,dim,nStart=0,nEnd=None):
    return slicedInverseWorker(_Ashared,N,dim,nStart,nEnd)

multiProcWorkers = mp.Pool(4)


def slicedInversion(Ain, cpy=True, NnDim=None, returnNp=True):
    NnDim = Ain.shape[:2] if NnDim is None else NnDim
    doParallel = NnDim[0]*NnDim[1] > 1000
    
    if doParallel:
        if cpy:
            if isinstance(Ain,np.ndarray):
                Ain2 = Ain.copy()
            else:
                try:
                    Ain2 = sctypes.copy(Ain)
                except:
                    Ain2 = sctypes.copy(Ain.get_obj())
        else:
            Ain2 = Ain
        if isinstance(Ain, np.ndarray):
            Ain2 = Ain2.ravel()
        NperSlice = NnDim[1]**2
        slicePerLoop = 10000000//NperSlice
        for k in range(NnDim[0]//slicePerLoop+1):
            thisN = min(slicePerLoop, NnDim[0]-k*slicePerLoop)
            #copy the data
            _Ashared[:thisN*NperSlice] = Ain2[k*slicePerLoop*NperSlice:(k*slicePerLoop+thisN)*NperSlice]
            #Distribute work
            indList = np.linspace(0,thisN,5,dtype=np.int_)
            LL = lmap(lambda i:[thisN,NnDim[1],indList[i],indList[i+1]],range(4))
            #Do the work
            multiProcWorkers.starmap(slicedInverseWorkerInherit, LL)
            #copy back
            Ain2[k*slicePerLoop*NperSlice:(k*slicePerLoop+thisN)*NperSlice] = _Ashared[:thisN*NperSlice]
            if isinstance(Ain,np.ndarray) and (not cpy):
                #This is some extra work to keep consistency
                Atemp = np.frombuffer(_Ashared)[:thisN*NperSlice]
                Atemp.resize((thisN,NnDim[1],NnDim[1]))
                Ain[k*slicePerLoop:k*slicePerLoop+thisN,:,:] = Atemp
            #Done this loop
    else:
        if cpy:
            try:
                Ain2 = Ain.copy() if isinstance(Ain, np.ndarray) else sctypes.copy(Ain)
            except AttributeError:
                Ain2 = sctypes.copy(Ain.get_obj)
        else:
            Ain2 = Ain
        
        slicedInverseWorker(Ain2, NnDim[0], NnDim[1])
        
    
    if returnNp and not isinstance(Ain2, np.ndarray):
        try:
            out = np.frombuffer(Ain2)
        except:
            out = np.frombuffer(Ain2.get_obj())
        out.resize((NnDim[0],NnDim[1],NnDim[1]))
    elif returnNp:
        out = Ain2
        out.resize((NnDim[0],NnDim[1],NnDim[1]))
    if not returnNp and isinstance(Ain2, np.ndarray):
        out = sctypes.RawArray(c.c_double,Ain2.size)
        out[:] = Ain2.ravel()
    elif not returnNp:
        out=Ain2
    
    return out
    



