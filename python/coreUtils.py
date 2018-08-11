try:
    from mainPars import *
except:
    print('No mainPars found')

from typing import Union, List

import numpy as np
import scipy as sp
import sympy as sy

try:
    from cythonUtils import *
except:
    print('No cythonUtils found')

from numpy.linalg import multi_dot
na = np.array
Id = np.identity
ndot = lambda *args: multi_dot(args)

np.set_printoptions(precision=4, linewidth=150)

import errno
import os

try:
    from numba import jit
except:
    print("No numba found")
    

from typing import List, Union, Callable

from numpy import zeros, ones, dot, empty
from numpy import identity as Id
from numpy import multiply, divide
from numpy import sqrt
from numpy import linspace, arange
from scipy.linalg import expm, inv, det,pinv, lstsq, svd
from scipy.optimize import fminbound
from scipy.optimize import fmin_l_bfgs_b as fminsearch
from scipy import sin,cos,tan,arcsin,arccos,arctan, arctan2
from scipy import sum,dot,square,exp,log
from scipy import absolute as abs
from scipy import pi
from scipy import  minimum, maximum

from scipy.linalg import cholesky as chol
from scipy.linalg import eig, eigh

import inspect
import warnings

import collections

def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

#np.seterr(all='raise')
np.seterr(divide='raise',invalid='raise')

epsFloat = np.nextafter(0.,1.)

def Idn(n=None,dim=None, out=None):
    if out is not None:
        n,dim,_ = out.shape
        assert out.shape[0] == out.shape[1]
        out[:] = IdnCyYes(int(n),int(dim),out)
        return out
    else:
        return IdnCyNo(int(n),int(dim))

def IdnR(n=None,dim=None):
    return np.broadcast_to(Id(dim), (n,dim,dim))
#def Idn2dim(n=None,dim0=None,dim1=None, out=None):
#    if out is not None:
#        n,dim0,dim1 = out.shape
#        Idn2dimCyYes(int(n),int(dim0),int(dim1),out)
#        return out
#    else:
#        return Idn2dimCyNo(int(n),int(dim0),int(dim1))

def Idn1(n=None,dim=None, out=None):
    assert ((n is None) and (dim is None)) or (out is None)
    if out is not None:
        n,dim,_ = out.shape
        assert out.shape[0] == out.shape[1]
    else:
        out = empty((n,dim,dim))
    for k in range(n):
        out[k,:,:] = Id(dim)
    return out
def Idn2(n=None,dim=None, out=None):
    assert ((n is None) and (dim is None)) or (out is None)
    ii = Id(dim)
    if out is not None:
        n,dim,_ = out.shape
        assert out.shape[0] == out.shape[1]
        out[:] = np.stack( n*[ii] )
    else:
        out = np.stack(n*[ii])
    return out

sq2 = 2.**0.5

mDotProd = lambda x,y,axis=0,kd=True: np.sum(np.multiply(x,y),axis=axis,keepdims=kd)

cNorm = lambda x,kd=True: np.linalg.norm(x, 2, 0, kd)
rNorm = lambda x,kd=True: np.linalg.norm(x, 2, 1, kd)
cov = lambda x, aweights = None, out=None: np.cov(x, aweights=aweights, rowvar=True, bias=False, out=out)
compMSE = lambda x,ax=0 : np.mean(sum(square(x),axis=ax))
trajDist = compMSE
#trajDist = lambda x,ax=0 : np.mean(np.sqrt(sum(square(x),axis=ax)))

#trajDist = lambda x,ax=0 : np.linalg.norm(x)/x.shape[1]

inversionEps = 1e-10


def nullspace(a, rtol=1e-8):
    u, s, v = svd(a)
    rank = (s > rtol*s[0]).sum()
    return rank, v[rank:,:].T.copy()


#Compute the squared colwise 2 norm efficiently
def cNormSquare(x, out=None, kd=True, cpy=True):
    if (out is None):
        if kd:
            out = zeros((1,x.shape[1]))
        else:
            out = zeros((x.shape[1],))
    else:
        if kd:
            assert out.shape==(1,x.shape[1]), "out has wrong dimension"
        else:
            assert out.shape==(x.shape[1],), "out has wrong dimension"
    if cpy:
        x = np.square(x)
    else:
        np.square(x,x)
    
    np.sum(x,0,keepdims=kd,out=out)
    return out

#Multiply a series of matrices element-wise (There can be scalars in the mix)
def nMult(*args):
    out = np.multiply(args[0], args[1])
    for aMat in args[2:]:
        try:
            np.multiply(out, aMat, out)
        except ValueError:
            #If internally broadcasting
            out = np.multiply(out,aMat)
    return out
#Divcide a series of matrices element-wise (There can be scalars in the mix)
def nDivide(*args):
    out = np.divide(args[0], args[1])
    for aMat in args[2:]:
        try:
            np.divide(out, aMat, out)
        except ValueError:
            #If internally broadcasting
            out = np.divide(out,aMat)
    return out


#Normalize data
def normalizeData(x, type=1):
    if type==1:
        np.divide( x, sqrt(sp.var(x,1,keepdims=True)),out=x)
        x -= x[:,[-1]]
    else:
        assert 0, 'false or TBD'
    return 0

#Other
lmap = lambda *args: list(map(*args))

#Wrap [-2pi, 4pi] into [0,2pi]
def wrapAngles2(x):
    """Wraps angles from [-2pi, 4pi] into [0,2pi]"""
    x[x < 0] += 2*pi
    x[x>4*pi] -= 2*pi
    return x
#Get spirals if angles are in [0,2pi]
def accumulateAngles(x, tol=.1):
    x = x.squeeze()
    dx = np.diff(x)
    #Get upward jumps; so value first small then large
    #All following values sould be decreased by 2pi
    uJ = np.logical_and( dx>(2*pi-2*tol), np.logical_and( x[:-1]<tol, x[1:]>2*pi-tol ) )
    #Get downward jumps; so value first large then small
    #All following values should be incresed by 2pi
    dJ = np.logical_and( dx<-(2*pi-2*tol), np.logical_and( x[:-1]>2*pi-tol, x[1:]<tol ) )
    
    uV = np.cumsum(uJ*(-2*pi))
    dV = np.cumsum(dJ*(2*pi))
    
    x[1:] += uV+dV
    
    return x
    

def rowIndependentSet(A, relTol=5e-4, absTol=5e-5):
    nRows,nCols = A.shape
    deleteInd = zeros((nRows,)).astype(np.bool_)
    
    #Get all row norms
    rNormA = rNorm(A, kd=False)
    
    for i in range(nRows):
        #Inner product of the i'th row with all i+1 Rows
        innerIOther = sum(np.multiply( A[i+1:,:], np.tile(A[[i],:], (nRows-(i+1),1)) ), axis=1)
        #Multiplications of norms
        prodIOther = rNormA[i+1:]*rNormA[i]
        #Test linear independence with cauchy-schwarz
        deleteInd[i+1:] = np.logical_or(deleteInd[i+1:], np.isclose(innerIOther-prodIOther, 0., rtol=relTol, atol=absTol))
    
    return np.delete(A, deleteInd, axis=0)

def getValidKWARGDict(aCallable, addKWARGS:dict):
    argList = inspect.signature(aCallable).parameters.values()
    newAddKWARGS = {}
    for aArg in argList:
        try:
            newAddKWARGS[aArg.name] = addKWARGS[aArg.name]
        except KeyError:
            pass
    return newAddKWARGS


##############################################################
def TXT2Matrix(fileName, fileType="cpp"):
    aArray = np.loadtxt(fileName)
    if fileType == "python":
        return aArray
    elif fileType == "cpp":
        dim = int(aArray[0])
        nPoints = int(aArray[1])
        outArray = np.zeros((dim, nPoints))
        aArray = aArray[2::]
        for k in range(nPoints):
            outArray[:,k] = aArray[dim*k:dim*(k+1)]
        return outArray
    else:
        return False
##############################################################
def TXT2Vector(fileName, fileType="cpp"):
    aArray = np.loadtxt(fileName)
    if fileType == "python":
        return aArray
    elif fileType == "cpp":
        return np.resize( aArray[1::], int(aArray[0]) )
    else:
        return False
##############################################################
def Array2TXT(fileName, aArray, fileType="cpp", format="%.18e"):
    if fileType == "cpp":
        np.savetxt(fileName, np.hstack(( aArray.shape, aArray.T.reshape(aArray.size,) )), fmt=format)
        with open(fileName,'r') as f:
            data = f.readlines()
        for k,i in enumerate(aArray.shape):
            data[k] = "{0:d}\n".format(i)
        with open(fileName,'w') as f:
            f.writelines(data)
    elif fileType == "python":
        np.savetxt(fileName, aArray, fmt=format)
    else:
        return False
    return True
##############################################################

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def computeVelocities(x:np.ndarray, t=None, endOption='Zero'):
    
    assert endOption in ('Zero', 'Same', None)
    
    if t is None:
        v = np.diff(x,axis=1)
    else:
        v = np.diff(x,axis=1)/np.diff(t.reshape)
    
    if endOption == 'Zero':
        v = np.hstack((v, np.zeros((x.shape[0],1), dtype=x.dtype)))
    if endOption == 'Same':
        v = np.hstack((v,v[:,[-1]]))
    return v

def computeDirections(x:np.ndarray, endOption='Zero'):
    direc = computeVelocities(x, endOption=endOption)
    direc /= (cNorm(direc, kd=True)+epsFloat)
    return direc


bool2Str = lambda x: "{0:d}\n".format(x)
int2Str = lambda x: "{0:d}\n".format(x)
double2Str = lambda x: "{0:.16e}\n".format(float(x))
vec2List = lambda X: list(map(double2Str, X.reshape((-1,))))


###############################################
@jit
def SEAerror(x:np.ndarray, xhat:np.ndarray):
    
    assert x.shape == xhat.shape, "Array need to have same dimension"
    assert xhat.shape[0] == 2, "Polygons have to be simple, only trivially guaranteed in 2d"
    #https://math.blogoverflow.com/2014/06/04/greens-theorem-and-area-of-polygons/
    
    N = x.shape[1]
    A=0.0
    SEA = np.empty((N-1,))
    
    for i in range(N-1):
        #The ith polyhedron is composed of the vertices
        #xhat_i, xhat_i+1, x_i+1, x_i, xhat_i
        #So
        # Normally we have to integrate ccw with normal outwards. Since we do not know the direction sign can be inversed
        A = (xhat[0,i+1]+xhat[0,i])*(xhat[1,i+1]-xhat[1,i])
        A += (x[0,i+1]+xhat[0,i+1])*(x[1,i+1]-xhat[1,i+1])
        A += (x[0,i]+x[0,i+1])*(x[1,i]-x[1,i+1])
        A += (xhat[0,i]+x[0,i])*(xhat[1,i]-x[1,i])
        
        SEA[i] = abs(A)/2.
    
    return SEA


