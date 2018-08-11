import numpy
import numpy as np
##from numba import jit

from numpy import sqrt

__thisDBG = True

# For sliced dist2nondiffeo
# -> point list
def nonDiffeoCheckPointList(b:float,d:float):
    # No need for d
    d = None
    return np.hstack((0, np.linspace(0.5*b,  1.0*b, 12)))#np.array([0., 0.5*b, (0.5*b + 0.7*b)/2., 0.7*b, 1.0*b])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

#@jit
def c1PolyMaxDeform (b:float,d:float=None,tn:float=1.,e:float=0):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d = None
    return -tn*(20*(b/16 + sqrt(33)*b/16)*(e - 1)/(3*b**2) + 20*(b/16 + sqrt(33)*b/16)**3*(-e + 1)/b**4 + 40*(b/16 + sqrt(33)*b/16)**4*(e - 1)/(3*b**5))

#@jit
def c1PolyMinB(tn:float,d:float=None,safeFac:float=0.5,e:float=0.):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d = None
    return -(tn/safeFac)*(5*(39 + 55*sqrt(33))*(e - 1)/1024)

#@jit
def c1KerFun(b:float,d:float,e:float):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d=None
    return 1, 0,10*(e - 1)/(3*b**2), 0, 5*(-e + 1)/b**4, 8*(e - 1)/(3*b**5), 0.3*b, 0.5*b, 0.7*b, 1.0*b

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
            out[ind] = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
    else:
        x=xIn
        if x>b:
            out[0] = e
        else:
            out[0] = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5

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
            out[ind] = c1 + 2*c2*x + 3*c3*x**2 + 4*c4*x**3 + 5*c5*x**4
    else:
        x=xIn
        if x>b:
            out[0] = 0.
        else:
            out[0] = c1 + 2*c2*x + 3*c3*x**2 + 4*c4*x**3 + 5*c5*x**4



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
            out[0][ind] = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
            out[1][ind] = c1 + 2*c2*x + 3*c3*x**2 + 4*c4*x**3 + 5*c5*x**4
    else:
        x=xIn
        if x>b:
            out[0][0] = e
            out[1][0] = 0.
        else:
            out[0][0] = c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5
            out[1][0] = c1 + 2*c2*x + 3*c3*x**2 + 4*c4*x**3 + 5*c5*x**4

    return out




    
    