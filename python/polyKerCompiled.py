import sympy as sy
import numpy
import numpy as np
x,xS,b,d,e = sy.symbols('x xS b d e')

f0b = sy.lambdify( (x,xS,b,d,e), 4.0*xS*(e - 1.0)/(b**3*(4.0*d**2 - 1.0)), numpy )
f1b = sy.lambdify( (x,xS,b,d,e), 2.0*(d - 0.5)*(e - 1.0)/(b*(2.0*d + 1.0)) - 2.0*(e - 1.0)*(-b*(-d + 0.5) + x)/(b**2*(2.0*d + 1.0)), numpy )
f2b = sy.lambdify( (x,xS,b,d,e), 2.0*(-d - 0.5)*(e - 1.0)/(b*(2.0*d + 1.0)) + 2.0*(-2*d - 1.0)*(e - 1.0)*(-b*(d + 0.5) + x)/(b**2*(4.0*d**2 - 1.0)) - 2.0*(e - 1.0)*(-b*(d + 0.5) + x)/(b**2*(2.0*d + 1.0)) - 4.0*(e - 1.0)*(-b*(d + 0.5) + x)**2/(b**3*(4.0*d**2 - 1.0)), numpy )
f3b = sy.lambdify( (x,xS,b,d,e), 0, numpy )

f0d = sy.lambdify( (x,xS,b,d,e), 16.0*d*xS*(e - 1.0)/(b**2*(4.0*d**2 - 1.0)**2), numpy )
f1d = sy.lambdify( (x,xS,b,d,e), 0.5*(-2.0*e + 6.0)/(2.0*d + 1.0) + 2.0*(e - 1.0)/(2.0*d + 1.0) - 1.0*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0)**2 - 4.0*(e - 1.0)*(-b*(-d + 0.5) + x)/(b*(2.0*d + 1.0)**2), numpy )
f2d = sy.lambdify( (x,xS,b,d,e), -2.0*(e - 1.0)/(2.0*d + 1.0) + 0.5*(6.0*e - 2.0)/(2.0*d + 1.0) - 1.0*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0)**2 - 4.0*(e - 1.0)*(-b*(d + 0.5) + x)/(b*(4.0*d**2 - 1.0)) - 4.0*(e - 1.0)*(-b*(d + 0.5) + x)/(b*(2.0*d + 1.0)**2) - 16.0*d*(e - 1.0)*(-b*(d + 0.5) + x)**2/(b**2*(4.0*d**2 - 1.0)**2), numpy )
f3d = sy.lambdify( (x,xS,b,d,e), 0, numpy )

f0e = sy.lambdify( (x,xS,b,d,e), -2.0*xS/(b**2*(4.0*d**2 - 1.0)), numpy )
f1e = sy.lambdify( (x,xS,b,d,e), 0.5*(-2.0*d + 1.0)/(2.0*d + 1.0) + 2.0*(-b*(-d + 0.5) + x)/(b*(2.0*d + 1.0)), numpy )
f2e = sy.lambdify( (x,xS,b,d,e), 0.5*(6.0*d + 1)/(2.0*d + 1.0) + 2.0*(-b*(d + 0.5) + x)/(b*(2.0*d + 1.0)) + 2.0*(-b*(d + 0.5) + x)**2/(b**2*(4.0*d**2 - 1.0)), numpy )
f3e = sy.lambdify( (x,xS,b,d,e), 1, numpy )


def c1KerFun(b,d,e):
    return 4.0*(e - 1.0)/(b**2*(4.0*d**2 - 1.0)),1.00000000000000,0.0,0.5*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0),2.0*(e - 1.0)/(b*(2.0*d + 1.0)),0.5*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0),2.0*(e - 1.0)/(b*(2.0*d + 1.0)),e

def c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None):
    vald=np.zeros((i0.size,))
    if diffVar=='b':
        if x is not None:
            vald[i0] += f0b(x[i0],xS[i0],b,d,e)
            vald[i1] += f1b(x[i1],xS[i1],b,d,e)
            vald[i2] += f2b(x[i2],xS[i2],b,d,e)
            vald[i3] += f3b(0.,0.,b,d,e)#Independent of x, out of base
        else:
            vald[i0] += f0b(x0,xS0,b,d,e)
            vald[i1] += f1b(x1,xS1,b,d,e)
            vald[i2] += f2b(x2,xS2,b,d,e)
            vald[i3] += f3b(0.,0.,b,d,e)#Independent of x, out of base
    elif diffVar=='d':
        if x is not None:
            vald[i0] += f0d(x[i0],xS[i0],b,d,e)
            vald[i1] += f1d(x[i1],xS[i1],b,d,e)
            vald[i2] += f2d(x[i2],xS[i2],b,d,e)
            vald[i3] += f3d(0.,0.,b,d,e)#Independent of x, out of base
        else:
            vald[i0] += f0d(x0,xS0,b,d,e)
            vald[i1] += f1d(x1,xS1,b,d,e)
            vald[i2] += f2d(x2,xS2,b,d,e)
            vald[i3] += f3d(0.,0.,b,d,e) #Independent of x, out of base
    elif diffVar=='e':
        if x is not None:
            vald[i0] += f0e(x[i0],xS[i0],b,d,e)
            vald[i1] += f1e(x[i1],xS[i1],b,d,e)
            vald[i2] += f2e(x[i2],xS[i2],b,d,e)
            vald[i3] += f3e(0.,0.,b,d,e)#Independent of x, out of base
        else:
            vald[i0] += f0e(x0,xS0,b,d,e)
            vald[i1] += f1e(x1,xS1,b,d,e)
            vald[i2] += f2e(x2,xS2,b,d,e)
            vald[i3] += f3e(0.,0.,b,d,e)#Independent of x, out of base
    else:
        assert 0,'diffVar not recognised'
    return vald
    
    