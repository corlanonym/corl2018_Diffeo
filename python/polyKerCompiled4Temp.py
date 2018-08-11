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
    return np.hstack((0, np.linspace(__p2__,  __p4__, 8)))#np.array([0., __p2__, (__p2__ + __p3__)/2., __p3__, __p4__])

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
    return __ca0__, __ca1__,__ca2__, __ca3__, __cb0__, __cb1__,__cb2__, __cb3__, __cc0__, __cc1__,__cc2__, __cc3__, __p1__, __p2__, __p3__, __p4__

def c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None):
    assert 0, "TBD"


def evalKerVal(xIn:np.ndarray, b:float, d:float, e:float, out:np.ndarray=None):

    if __thisDBG:
        assert np.all(np.array(xIn)>=0.), 'Only non-negative values allowed for kernel evaluation'
        d=None
    
    out = np.empty_like(xIn) if (out is None) else out

    ca0,ca1,ca2,ca3,cb0,cb1,cb2,cb3,cc0,cc1,cc2,cc3,p1,p2,p3,p4 = c1KerFun(b, None, e)

    if xIn.size>1:

        inde = xIn>b
        indc = np.logical_and(xIn > 3./4.*b,np.logical_not(inde))
        indb = np.logical_and(xIn > 1./4.*b,np.logical_not(np.logical_or(indc,inde)))
        inda = np.logical_not(np.logical_or(inde, np.logical_or(indc,indb)))
        
        #Add up
        if np.any(inde):
            out[inde] = e
        if np.any(indc):
            x = xIn[indc]
            out[indc] = __fc__
        if np.any(indb):
            x = xIn[indb]
            out[indb] = __fb__
        if np.any(inda):
            x = xIn[inda]
            out[inda] = __fa__
    else:
        x=xIn
        if x>b:
            out[0] = e
        elif x>3./4.*b:
            out[0] = __fc__
        elif x>1./4.*b:
            out[0] = __fb__
        else:
            out[0] = __fa__

    return out


def evalKerDeriv(xIn:np.ndarray, b:float, d:float, e:float, out=None):
    
    if __thisDBG:
        d=None
        assert np.all(np.array(xIn)>=0.)

    out = np.empty_like(xIn) if (out is None) else out

    ca0,ca1,ca2,ca3,cb0,cb1,cb2,cb3,cc0,cc1,cc2,cc3,p1,p2,p3,p4 = c1KerFun(b,None,e)

    if xIn.size > 1:
    
        inde = xIn > b
        indc = np.logical_and(xIn > 3./4.*b,np.logical_not(inde))
        indb = np.logical_and(xIn > 1./4.*b,np.logical_not(np.logical_or(indc,inde)))
        inda = np.logical_not(np.logical_or(inde,np.logical_or(indc,indb)))

        # Add up
        if np.any(inde):
            out[inde] = 0.
        if np.any(indc):
            x = xIn[indc]
            out[indc] = __fcp__
        if np.any(indb):
            x = xIn[indb]
            out[indb] = __fbp__
        if np.any(inda):
            x = xIn[inda]
            out[inda] = __fap__
    else:
        x = xIn
        if x > b:
            out[0] = e
        elif x > 3./4.*b:
            out[0] = __fcp__
        elif x > 1./4.*b:
            out[0] = __fbp__
        else:
            out[0] = __fap__


    return out

def evalKerValnDeriv(xIn:np.ndarray, b:float, d:float, e:float, out=None):
    
    if __thisDBG:
        d=None
        assert np.all(np.array(xIn)>=0.)

    out = [np.empty_like(xIn), np.empty_like(xIn)] if (out is None) else out

    ca0,ca1,ca2,ca3,cb0,cb1,cb2,cb3,cc0,cc1,cc2,cc3,p1,p2,p3,p4 = c1KerFun(b,None,e)

    if xIn.size > 1:
    
        inde = xIn > b
        indc = np.logical_and(xIn > 3./4.*b,np.logical_not(inde))
        indb = np.logical_and(xIn > 1./4.*b,np.logical_not(np.logical_or(indc,inde)))
        inda = np.logical_not(np.logical_or(inde,np.logical_or(indc,indb)))

        # Add up
        if np.any(inde):
            out[0][inde] = e
        if np.any(indc):
            x = xIn[indc]
            out[0][indc] = __fc__
            out[1][indc] = __fcp__
        if np.any(indb):
            x = xIn[indb]
            out[0][indb] = __fb__
            out[1][indb] = __fbp__
        if np.any(inda):
            x = xIn[inda]
            out[0][inda] = __fa__
            out[1][inda] = __fap__
    else:
        x = xIn
        if x > b:
            out[0][0] = e
            out[1][0] = 0.
        elif x > 3./4.*b:
            out[0][0] = __fc__
            out[1][0] = __fcp__
        elif x > 1./4.*b:
            out[0][0] = __fb__
            out[1][0] = __fbp__
        else:
            out[0][0] = __fa__
            out[1][0] = __fap__

    return out




    
    