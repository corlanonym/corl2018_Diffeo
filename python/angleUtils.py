import numpy as np
from coreUtils import *

def wrapAngles(ang, out=None):
    """wraps angles from (-3pi to 3pi] to (-pi to pi]"""
    from numpy import pi
    
    if out is None:
        out=ang.copy()
    else:
        np.copyto(out, ang)
    
    out[ang>pi] -= 2.*pi
    out[ang<-pi] += 2.*pi
    return out

def cNormWrapped(x,kd=True):
    if x.size>10000:
        aWrap = wrapAngles(x[0,:])
        
        out = sqrt( square(aWrap)+sum(square(x[1:,:]),axis=0) )
        if kd:
            out.resize((1,x.shape[1]))
        else:
            out.resize((x.shape[1],))
        return out
    else:
        x=x.copy()
        wrapAngles(x[0,:],x[0,:])
        return cNorm(x,kd=False)
        
    