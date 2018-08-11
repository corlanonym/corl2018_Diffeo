import numpy
import numpy as np
#from numba import jit

__d2BaseVal = 0.1
__thisDBG = True

# For sliced dist2nondiffeo
# -> point list
def nonDiffeoCheckPointList(b:float,d:float,d2:float=__d2BaseVal):
    return np.hstack((0, np.linspace(b*(d + 0.5),  1.0*b, 8)))#np.array([0., b*(d + 0.5), (b*(d + 0.5) + b*(-d2 + 1.0))/2., b*(-d2 + 1.0), 1.0*b])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

#@jit
def c1PolyMaxDeform (b,d,tn=1.,e=0,d2=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (-d + 0.5))
        assert np.all((-d + 0.5) < (d + 0.5))
        assert np.all((d + 0.5) < (-d2 + 1.0))
        assert np.all((-d2 + 1.0) < (1.0))
    return -tn*((b*(-d + 0.5) - 24.0*(-b*(-d + 0.5) + b*(d + 0.5))*(-d2*e + d2 + e - 1)/(b**2*(24.0*(-e + 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)) - 24.0*(-d2*e + d2 + e - 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))*(24.0*(-d2*e + d2 + e - 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)) + 1.0*(12.0*d2*e - 12.0*d2 - 24.0*e + 24.0)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)) - 0.5*(12.0*d2*e - 12.0*d2 - 24.0*e + 24.0)/(b**2*d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0))) + (6.0*d**2*d2*e - 6.0*d**2*d2 - 12.0*d**2*e + 12.0*d**2 - 6.0*d*d2*e + 6.0*d*d2 + 12.0*d*e - 12.0*d + 1.5*d2*e - 1.5*d2 - 3.0*e + 3.0)/(b*d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)) + 0.5*(b*(-d + 0.5) - 24.0*(-b*(-d + 0.5) + b*(d + 0.5))*(-d2*e + d2 + e - 1)/(b**2*(24.0*(-e + 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)) - 24.0*(-d2*e + d2 + e - 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))**2*(12.0*d2*e - 12.0*d2 - 24.0*e + 24.0)/(b**3*d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))

#@jit
def c1PolyMinB(tn,d,safeFac=0.5,e=0., d2=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (-d + 0.5))
        assert np.all((-d + 0.5) < (d + 0.5))
        assert np.all((d + 0.5) < (-d2 + 1.0))
        assert np.all((-d2 + 1.0) < (1.0))
    return -(tn/safeFac)*(-12.0*(d2 - 1.0)*(e - 1.0)*(2.0*d + d2 - 2.0)/((d2 - 2.0)*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)))

#@jit
def c1KerFun(b:float,d:float,e:float,d2:float=__d2BaseVal):
    if __thisDBG:
        assert np.all(0. < (-d + 0.5))
        assert np.all((-d + 0.5) < (d + 0.5))
        assert np.all((d + 0.5) < (-d2 + 1.0))
        assert np.all((-d2 + 1.0) < (1.0))
    return 24.0*(-d2*e + d2 + e - 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), 24.0*(-e + 1)/(b**2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)),(12.0*d2*e - 12.0*d2 - 24.0*e + 24.0)/(b**3*d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), 24.0*(e - 1)/(b**3*d2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), 1.00000000000000, 0.0, (2.0*d**3*d2*e + 2.0*d**3*d2 - 4.0*d**3*e - 4.0*d**3 - 3.0*d**2*d2*e + 3.0*d**2*d2 + 6.0*d**2*e - 6.0*d**2 + 4.0*d*d2**2 + 1.5*d*d2*e - 10.5*d*d2 - 3.0*d*e + 9.0*d - 0.25*d2*e + 0.25*d2 + 0.5*e - 0.5)/(d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), (6.0*d**2*d2*e - 6.0*d**2*d2 - 12.0*d**2*e + 12.0*d**2 - 6.0*d*d2*e + 6.0*d*d2 + 12.0*d*e - 12.0*d + 1.5*d2*e - 1.5*d2 - 3.0*e + 3.0)/(b*d*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), (4.0*d**2*d2*e - 8.0*d**2*e + 4.0*d2**2 + 3.0*d2*e - 12.0*d2 - 6.0*e + 12.0)/(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0), (-12.0*d2*e + 12.0*d2 + 24.0*e - 24.0)/(b*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), (4.0*d**2*d2**2*e - 8.0*d**2*d2*e + 4.0*d2**3*e - 9.0*d2**2*e + 6.0*d2*e - 4.0*e + 4.0)/(d2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), 12.0*(e - 1)/(b*d2*(4.0*d**2*d2 - 8.0*d**2 + 4.0*d2**2 - 9.0*d2 + 6.0)), b*(-d + 0.5), b*(d + 0.5), b*(-d2 + 1.0), 1.0*b

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
            out[ind4] = e
        if np.any(ind3):
            x = xIn[ind3]
            out[ind3] = 0.166666666666667*dddySecond*x**3 + 1.0*dy03*x + x**2*(0.5*b*d2*dddySecond - 0.5*b*dddySecond + 0.5*ddySecond) + y03
        if np.any(ind2):
            x = xIn[ind2]
            out[ind2] = ddySecond*x**2/2 + dy02*x + y02
        if np.any(ind1):
            x = xIn[ind1]
            out[ind1] = 0.166666666666667*dddyFirst*x**3 + 1.0*dy01*x + x**2*(0.5*b*d*dddyFirst - 0.25*b*dddyFirst + 0.5*ddyFirst) + y01
        if np.any(ind0):
            x = xIn[ind0]
            out[ind0] = ddyFirst*x**2/2 + dy00*x + y00
    else:
        x=xIn
        if x>p4:
            out = e
        elif x>p3:
            out = 0.166666666666667*dddySecond*x**3 + 1.0*dy03*x + x**2*(0.5*b*d2*dddySecond - 0.5*b*dddySecond + 0.5*ddySecond) + y03
        elif x>p2:
            out = ddySecond*x**2/2 + dy02*x + y02
        elif x>p1:
            out = 0.166666666666667*dddyFirst*x**3 + 1.0*dy01*x + x**2*(0.5*b*d*dddyFirst - 0.25*b*dddyFirst + 0.5*ddyFirst) + y01
        else:
            out = ddyFirst*x**2/2 + dy00*x + y00

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
            out[ind4] = 0.0
        if np.any(ind3):
            x = xIn[ind3]
            out[ind3] = 0.5*dddySecond*x**2 + dy03 + x*(1.0*b*d2*dddySecond - 1.0*b*dddySecond + 1.0*ddySecond)
        if np.any(ind2):
            x = xIn[ind2]
            out[ind2] = ddySecond*x + dy02
        if np.any(ind1):
            x = xIn[ind1]
            out[ind1] = 0.5*dddyFirst*x**2 + dy01 + x*(1.0*b*d*dddyFirst - 0.5*b*dddyFirst + 1.0*ddyFirst)
        if np.any(ind0):
            x = xIn[ind0]
            out[ind0] = ddyFirst*x + dy00
    else:
        x=xIn
        if x>p4:
            out = 0.0
        elif x>p3:
            out = 0.5*dddySecond*x**2 + dy03 + x*(1.0*b*d2*dddySecond - 1.0*b*dddySecond + 1.0*ddySecond)
        elif x>p2:
            out = ddySecond*x + dy02
        elif x>p1:
            out = 0.5*dddyFirst*x**2 + dy01 + x*(1.0*b*d*dddyFirst - 0.5*b*dddyFirst + 1.0*ddyFirst)
        else:
            out = ddyFirst*x + dy00


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
            out[0][ind4] = e
            out[1][ind4] = 0.0
        if np.any(ind3):
            x = xIn[ind3]
            out[0][ind3] = 0.166666666666667*dddySecond*x**3 + 1.0*dy03*x + x**2*(0.5*b*d2*dddySecond - 0.5*b*dddySecond + 0.5*ddySecond) + y03
            out[1][ind3] = 0.5*dddySecond*x**2 + dy03 + x*(1.0*b*d2*dddySecond - 1.0*b*dddySecond + 1.0*ddySecond)
        if np.any(ind2):
            x = xIn[ind2]
            out[0][ind2] = ddySecond*x**2/2 + dy02*x + y02
            out[1][ind2] = ddySecond*x + dy02
        if np.any(ind1):
            x = xIn[ind1]
            out[0][ind1] = 0.166666666666667*dddyFirst*x**3 + 1.0*dy01*x + x**2*(0.5*b*d*dddyFirst - 0.25*b*dddyFirst + 0.5*ddyFirst) + y01
            out[1][ind1] = 0.5*dddyFirst*x**2 + dy01 + x*(1.0*b*d*dddyFirst - 0.5*b*dddyFirst + 1.0*ddyFirst)
        if np.any(ind0):
            x = xIn[ind0]
            out[0][ind0] = ddyFirst*x**2/2 + dy00*x + y00
            out[1][ind0] = ddyFirst*x + dy00
    else:
        x=xIn
        if x>p4:
            out[0] = e
            out[1] = 0.0
        elif x>p3:
            out[0] = 0.166666666666667*dddySecond*x**3 + 1.0*dy03*x + x**2*(0.5*b*d2*dddySecond - 0.5*b*dddySecond + 0.5*ddySecond) + y03
            out[1] = 0.5*dddySecond*x**2 + dy03 + x*(1.0*b*d2*dddySecond - 1.0*b*dddySecond + 1.0*ddySecond)
        elif x>p2:
            out[0] = ddySecond*x**2/2 + dy02*x + y02
            out[1] = ddySecond*x + dy02
        elif x>p1:
            out[0] = 0.166666666666667*dddyFirst*x**3 + 1.0*dy01*x + x**2*(0.5*b*d*dddyFirst - 0.25*b*dddyFirst + 0.5*ddyFirst) + y01
            out[1] = 0.5*dddyFirst*x**2 + dy01 + x*(1.0*b*d*dddyFirst - 0.5*b*dddyFirst + 1.0*ddyFirst)
        else:
            out[0] = ddyFirst*x**2/2 + dy00*x + y00
            out[1] = ddyFirst*x + dy00

    return out




    
    