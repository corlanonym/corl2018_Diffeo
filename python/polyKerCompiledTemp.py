import sympy as sy
import numpy
import numpy as np
x,xS,b,d,e = sy.symbols('x xS b d e')

f0b = sy.lambdify( (x,xS,b,d,e), __f0b__, numpy )
f1b = sy.lambdify( (x,xS,b,d,e), __f1b__, numpy )
f2b = sy.lambdify( (x,xS,b,d,e), __f2b__, numpy )
f3b = sy.lambdify( (x,xS,b,d,e), __f3b__, numpy )

f0d = sy.lambdify( (x,xS,b,d,e), __f0d__, numpy )
f1d = sy.lambdify( (x,xS,b,d,e), __f1d__, numpy )
f2d = sy.lambdify( (x,xS,b,d,e), __f2d__, numpy )
f3d = sy.lambdify( (x,xS,b,d,e), __f3d__, numpy )

f0e = sy.lambdify( (x,xS,b,d,e), __f0e__, numpy )
f1e = sy.lambdify( (x,xS,b,d,e), __f1e__, numpy )
f2e = sy.lambdify( (x,xS,b,d,e), __f2e__, numpy )
f3e = sy.lambdify( (x,xS,b,d,e), __f3e__, numpy )


def c1KerFun(b,d,e):
    return __ddy__,__y00__,__dy00__,__y01__,__dy01__,__y02__,__dy02__,__y03__

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
    
    