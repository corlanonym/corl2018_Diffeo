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
    return np.hstack((0, np.linspace(0.5*b,  1.0*b, 8)))#np.array([0., 0.5*b, (0.5*b + 0.7*b)/2., 0.7*b, 1.0*b])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

#@jit
def c1PolyMaxDeform (b:float,d:float=None,tn:float=1.,e:float=0):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d = None
    return -tn*(6.0*(-e + 1.0)/b + 8.0*(e - 1.0)/b)

#@jit
def c1PolyMinB(tn:float,d:float=None,safeFac:float=0.5,e:float=0.):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d = None
    return -(tn/safeFac)*(2.0*e - 2.0)

#@jit
def c1KerFun(b:float,d:float,e:float):
    if __thisDBG:
        assert np.all(0. < (0.3))
        assert np.all((0.3) < (0.5))
        assert np.all((0.5) < (0.7))
        assert np.all((0.7) < (1.0))
        d=None
    return 5.33333333333333*(e - 1.0)/b**3, 0.0,0.0, 1.00000000000000, 5.33333333333333*(-e + 1.0)/b**3, 8.0*(e - 1.0)/b**2,2.0*(-e + 1.0)/b, 0.166666666666667*e + 0.833333333333333, 5.33333333333333*(e - 1.0)/b**3, 16.0*(-e + 1.0)/b**2,16.0*(e - 1.0)/b, -4.33333333333333*e + 5.33333333333333, 0.3*b, 0.5*b, 0.7*b, 1.0*b

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
            out[indc] = cc0*x**3 + cc1*x**2 + cc2*x + cc3
        if np.any(indb):
            x = xIn[indb]
            out[indb] = cb0*x**3 + cb1*x**2 + cb2*x + cb3
        if np.any(inda):
            x = xIn[inda]
            out[inda] = ca0*x**3 + ca1*x**2 + ca2*x + ca3
    else:
        x=xIn
        if x>b:
            out[0] = e
        elif x>3./4.*b:
            out[0] = cc0*x**3 + cc1*x**2 + cc2*x + cc3
        elif x>1./4.*b:
            out[0] = cb0*x**3 + cb1*x**2 + cb2*x + cb3
        else:
            out[0] = ca0*x**3 + ca1*x**2 + ca2*x + ca3

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
            out[indc] = 3*cc0*x**2 + 2*cc1*x + cc2
        if np.any(indb):
            x = xIn[indb]
            out[indb] = 3*cb0*x**2 + 2*cb1*x + cb2
        if np.any(inda):
            x = xIn[inda]
            out[inda] = 3*ca0*x**2 + 2*ca1*x + ca2
    else:
        x = xIn
        if x > b:
            out[0] = e
        elif x > 3./4.*b:
            out[0] = 3*cc0*x**2 + 2*cc1*x + cc2
        elif x > 1./4.*b:
            out[0] = 3*cb0*x**2 + 2*cb1*x + cb2
        else:
            out[0] = 3*ca0*x**2 + 2*ca1*x + ca2


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
            out[0][indc] = cc0*x**3 + cc1*x**2 + cc2*x + cc3
            out[1][indc] = 3*cc0*x**2 + 2*cc1*x + cc2
        if np.any(indb):
            x = xIn[indb]
            out[0][indb] = cb0*x**3 + cb1*x**2 + cb2*x + cb3
            out[1][indb] = 3*cb0*x**2 + 2*cb1*x + cb2
        if np.any(inda):
            x = xIn[inda]
            out[0][inda] = ca0*x**3 + ca1*x**2 + ca2*x + ca3
            out[1][inda] = 3*ca0*x**2 + 2*ca1*x + ca2
    else:
        x = xIn
        if x > b:
            out[0][0] = e
            out[1][0] = 0.
        elif x > 3./4.*b:
            out[0][0] = cc0*x**3 + cc1*x**2 + cc2*x + cc3
            out[1][0] = 3*cc0*x**2 + 2*cc1*x + cc2
        elif x > 1./4.*b:
            out[0][0] = cb0*x**3 + cb1*x**2 + cb2*x + cb3
            out[1][0] = 3*cb0*x**2 + 2*cb1*x + cb2
        else:
            out[0][0] = ca0*x**3 + ca1*x**2 + ca2*x + ca3
            out[1][0] = 3*ca0*x**2 + 2*ca1*x + ca2

    return out




    
    