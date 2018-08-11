import numpy
import numpy as np
#from numba import jit

from numpy import sqrt

__thisDBG = True

# For sliced dist2nondiffeo
# -> point list
def nonDiffeoCheckPointList(b:float,d:float):
    # No need for d
    d = None
    return np.hstack((0, np.linspace(__p2__,  __p4__, 12)))#np.array([0., __p2__, (__p2__ + __p3__)/2., __p3__, __p4__])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

#@jit
def c1PolyMaxDeform (b:float,d:float=None,tn:float=1.,e:float=0):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
        d = None
    return -tn*(__dyMax__)

#@jit
def c1PolyMinB(tn:float,d:float=None,safeFac:float=0.5,e:float=0.):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
        d = None
    return -(tn/safeFac)*(__minBaseExpr__)

#@jit
def c1KerFun(b:float,d:float,e:float):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
        d=None
    return __c0__, __c1__,__c2__, __c3__, __c4__, __c5__, __p1__, __p2__, __p3__, __p4__

def c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None):
    assert 0, "TBD"


def evalKerVal(xIn:np.ndarray, b:float, d:float, e:float, out:np.ndarray=None):

    if __thisDBG:
        assert np.all(np.array(xIn)>=0.), 'Only non-negative values allowed for kernel evaluation'
        d=None
    
    out = np.empty_like(xIn) if (out is None) else out

    c0,c1,c2,c3,c4,c5,p1,p2,p3,p4 = c1KerFun(b, None, e)

    if xIn.size>1:

        ind = xIn>b
        #Add up
        if np.any(ind):
            out[ind] = e
        ind = np.logical_not(ind)
        if np.any(ind):
            x = xIn[ind]
            out[ind] = __f__
    else:
        x=xIn
        if x>b:
            out[0] = e
        else:
            out[0] = __f__

    return out


def evalKerDeriv(xIn:np.ndarray, b:float, d:float, e:float, out=None):
    
    if __thisDBG:
        d=None
        assert np.all(np.array(xIn)>=0.)

    out = np.empty_like(xIn) if (out is None) else out

    c0,c1,c2,c3,c4,c5,p1,p2,p3,p4 = c1KerFun(b,None,e)

    if xIn.size>1:
        ind = xIn > b
        # Add up
        if np.any(ind):
            out[ind] = 0.
        ind = np.logical_not(ind)
        if np.any(ind):
            x = xIn[ind]
            out[ind] = __fp__
    else:
        x=xIn
        if x>b:
            out[0] = 0.
        else:
            out[0] = __fp__



    return out

def evalKerValnDeriv(xIn:np.ndarray, b:float, d:float, e:float, out=None):
    
    if __thisDBG:
        d=None
        assert np.all(np.array(xIn)>=0.)

    out = [np.empty_like(xIn), np.empty_like(xIn)] if (out is None) else out

    c0,c1,c2,c3,c4,c5,p1,p2,p3,p4 = c1KerFun(b,None,e)

    if xIn.size>1:
        ind = xIn > b
        # Add up
        if np.any(ind):
            out[0][ind] = e
            out[1][ind] = 0.
        ind = np.logical_not(ind)
        if np.any(ind):
            x = xIn[ind]
            out[0][ind] = __f__
            out[1][ind] = __fp__
    else:
        x=xIn
        if x>b:
            out[0][0] = e
            out[1][0] = 0.
        else:
            out[0][0] = __f__
            out[1][0] = __fp__

    return out




    
    