import numpy
import numpy as np
#from numba import jit

__d2BaseVal = 0.2
__thisDBG = True

# For sliced dist2nondiffeo
# -> point list
def nonDiffeoCheckPointList(b:float,d:float,d2:float=__d2BaseVal):
    return np.hstack((0, np.linspace(__p2__,  __p4__, 8)))#np.array([0., __p2__, (__p2__ + __p3__)/2., __p3__, __p4__])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

#@jit
def c1PolyMaxDeform (b,d,tn=1.,e=0,d2=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
    return -tn*(__dyMax__)

#@jit
def c1PolyMinB(tn,d,safeFac=0.5,e=0., d2=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
    return -(tn/safeFac)*(__minBaseExpr__)

#@jit
def c1KerFun(b:float,d:float,e:float,d2:float=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (__p1noB__))
        assert np.all((__p1noB__) < (__p2noB__))
        assert np.all((__p2noB__) < (__p3noB__))
        assert np.all((__p3noB__) < (__p4noB__))
    return __ddyFirst__, __ddySecond__,__dddyFirst__, __dddySecond__, __y00__, __dy00__, __y01__, __dy01__, __y02__, __dy02__, __y03__, __dy03__, __p1__, __p2__, __p3__, __p4__

def c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None):
    assert 0, "TBD"


def evalKerVal(xIn, b, d, e, d2=__d2BaseVal, out=None):

    out = np.empty_like(xIn) if (out is None) else out

    ddyFirst, ddySecond, dddyFirst, dddySecond, y00, dy00, y01, dy01, y02, dy02, y03, dy03, p1, p2, p3, p4 = c1KerFun(b, d, e, d2)

    if xIn.size>1:

        ind4 = xIn>p4
        ind3 = np.logical_and( xIn>p3, np.logical_not(ind4))
        ind2 = np.logical_and( xIn>p2, np.logical_not(np.logical_or(ind3, ind4)))
        ind1 = np.logical_and( xIn>p1, np.logical_not(np.logical_or(ind2, np.logical_or(ind3, ind4))))
        ind0 = np.logical_not(np.logical_or(ind1, np.logical_or(ind2, np.logical_or(ind3, ind4))))

        #Add up
        if np.any(ind4):
            x = xIn[ind4]
            out[ind4] = __f4__
        if np.any(ind3):
            x = xIn[ind3]
            out[ind3] = __f3__
        if np.any(ind2):
            x = xIn[ind2]
            out[ind2] = __f2__
        if np.any(ind1):
            x = xIn[ind1]
            out[ind1] = __f1__
        if np.any(ind0):
            x = xIn[ind0]
            out[ind0] = __f0__
    else:
        x=xIn
        if x>p4:
            out = __f4__
        elif x>p3:
            out = __f3__
        elif x>p2:
            out = __f2__
        elif x>p1:
            out = __f1__
        else:
            out = __f0__

    return out


def evalKerDeriv(xIn, b, d, e, d2=__d2BaseVal, out=None):

    out = np.empty_like(xIn) if (out is None) else out

    ddyFirst, ddySecond, dddyFirst, dddySecond, y00, dy00, y01, dy01, y02, dy02, y03, dy03, p1, p2, p3, p4 = c1KerFun(b, d, e, d2)

    if xIn.size>1:
        ind4 = xIn > p4
        ind3 = np.logical_and(xIn > p3, np.logical_not(ind4))
        ind2 = np.logical_and(xIn > p2, np.logical_not(np.logical_or(ind3, ind4)))
        ind1 = np.logical_and(xIn > p1, np.logical_not(np.logical_or(ind2, np.logical_or(ind3, ind4))))
        ind0 = np.logical_not(np.logical_or(ind1, np.logical_or(ind2, np.logical_or(ind3, ind4))))

        # Add up
        if np.any(ind4):
            x = xIn[ind4]
            out[ind4] = __f4p__
        if np.any(ind3):
            x = xIn[ind3]
            out[ind3] = __f3p__
        if np.any(ind2):
            x = xIn[ind2]
            out[ind2] = __f2p__
        if np.any(ind1):
            x = xIn[ind1]
            out[ind1] = __f1p__
        if np.any(ind0):
            x = xIn[ind0]
            out[ind0] = __f0p__
    else:
        x=xIn
        if x>p4:
            out = __f4p__
        elif x>p3:
            out = __f3p__
        elif x>p2:
            out = __f2p__
        elif x>p1:
            out = __f1p__
        else:
            out = __f0p__


    return out

def evalKerValnDeriv(xIn, b, d, e, d2=__d2BaseVal, out=None):

    out = [np.empty_like(xIn), np.empty_like(xIn)] if (out is None) else out

    ddyFirst, ddySecond, dddyFirst, dddySecond, y00, dy00, y01, dy01, y02, dy02, y03, dy03, p1, p2, p3, p4 = c1KerFun(b, d, e, d2)

    if xIn.size>1:
        ind4 = xIn > p4
        ind3 = np.logical_and(xIn > p3, np.logical_not(ind4))
        ind2 = np.logical_and(xIn > p2, np.logical_not(np.logical_or(ind3, ind4)))
        ind1 = np.logical_and(xIn > p1, np.logical_not(np.logical_or(ind2, np.logical_or(ind3, ind4))))
        ind0 = np.logical_not(np.logical_or(ind1, np.logical_or(ind2, np.logical_or(ind3, ind4))))

        # Add up
        if np.any(ind4):
            x = xIn[ind4]
            out[0][ind4] = __f4__
            out[1][ind4] = __f4p__
        if np.any(ind3):
            x = xIn[ind3]
            out[0][ind3] = __f3__
            out[1][ind3] = __f3p__
        if np.any(ind2):
            x = xIn[ind2]
            out[0][ind2] = __f2__
            out[1][ind2] = __f2p__
        if np.any(ind1):
            x = xIn[ind1]
            out[0][ind1] = __f1__
            out[1][ind1] = __f1p__
        if np.any(ind0):
            x = xIn[ind0]
            out[0][ind0] = __f0__
            out[1][ind0] = __f0p__
    else:
        x=xIn
        if x>p4:
            out[0] = __f4__
            out[1] = __f4p__
        elif x>p3:
            out[0] = __f3__
            out[1] = __f3p__
        elif x>p2:
            out[0] = __f2__
            out[1] = __f2p__
        elif x>p1:
            out[0] = __f1__
            out[1] = __f1p__
        else:
            out[0] = __f0__
            out[1] = __f0p__

    return out




    
    