from coreUtils import *

from mainPars import whichKernel_

# c1KerFun = compile('ddy = 4.0*(e - 1.0)/(b**2*(4.0*d**2 - 1.0));y00 = 1.00000000000000;dy00 = 0.0;y01 = 0.5*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0);dy01 = 2.0*(e - 1.0)/(b*(2.0*d + 1.0));y02 = 0.5*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0);dy02 = 2.0*(e - 1.0)/(b*(2.0*d + 1.0));', '<string>', 'exec')
# c1KerFunPartial = {'b':compile('ddy = -8.0*(e - 1.0)/(b**3*(4.0*d**2 - 1.0));y00 = 0;dy00 = 0;y01 = 0;dy01 = -2.0*(e - 1.0)/(b**2*(2.0*d + 1.0));y02 = 0;dy02 = -2.0*(e - 1.0)/(b**2*(2.0*d + 1.0));y03=0.;', '<string>', 'exec'),
#                   'd':compile('ddy = -32.0*d*(e - 1.0)/(b**2*(4.0*d**2 - 1.0)**2);y00 = 0;dy00 = 0;y01 = 0.5*(-2.0*e + 6.0)/(2.0*d + 1.0) - 1.0*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0)**2;dy01 = -4.0*(e - 1.0)/(b*(2.0*d + 1.0)**2);y02 = 0.5*(6.0*e - 2.0)/(2.0*d + 1.0) - 1.0*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0)**2;dy02 = -4.0*(e - 1.0)/(b*(2.0*d + 1.0)**2);y03=0.;', '<string>', 'exec'),
#                   'e':compile('ddy = 4.0/(b**2*(4.0*d**2 - 1.0));y00 = 0;dy00 = 0;y01 = 0.5*(-2.0*d + 1.0)/(2.0*d + 1.0);dy01 = 2.0/(b*(2.0*d + 1.0));y02 = 0.5*(6.0*d + 1)/(2.0*d + 1.0);dy02 = 2.0/(b*(2.0*d + 1.0));y03=1.;', '<string>', 'exec')}


if whichKernel_==2:
    # C2 diffeo
    from polyKerCompiled2 import c1KerFun,c1KerFunPartial,c1PolyMaxDeform,c1PolyMinB,evalKerVal,evalKerDeriv,evalKerValnDeriv,nonDiffeoCheckPointList,nonDiffeoCheckPointListLen
elif whichKernel_ == 3:
    #Minimal jerk but continuous
    from polyKerCompiled3 import c1KerFun,c1KerFunPartial,c1PolyMaxDeform,c1PolyMinB,evalKerVal,evalKerDeriv,evalKerValnDeriv,nonDiffeoCheckPointList,nonDiffeoCheckPointListLen
elif whichKernel_ == 4:
    #Minimal jerk
    from polyKerCompiled4 import c1KerFun,c1KerFunPartial,c1PolyMaxDeform,c1PolyMinB,evalKerVal,evalKerDeriv,evalKerValnDeriv,nonDiffeoCheckPointList,nonDiffeoCheckPointListLen

def c1PolyKer(x,bIn,dIn,eIn=None,deriv=0,cIn=None,out=None,fullOut=False,inPolarCoords=False):
    """Returns the coefficients of a polynomial kernel
    x : numpy array with x values
    bIn : base of the kernel float or array
    dIn : zone of constant change
    eIn : kernel value outside of base
    deriv : 0 : kernel value, 1 : derivative of kernel value, [0,1] : both
    out : optional array where results are added
    """

    if inPolarCoords:
        norm = cNormWrapped
    else:
        norm = cNorm
    
    # Transition points
    # y00, ... These will be filled in exec
    # vValues = {"ddy":0.,"y00":0.,"dy00":0.,"y01":0.,"dy01":0.,"y02":0.,"dy02":0.,"y03":0.}
    # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['dy02'];y02=vValues['dy02'];y03=vValues['y03']
    if isinstance(bIn,float):
        eIn = 0. if eIn is None else eIn

        # If c is given x is vector else scalar
        if cIn is not None:
            x = norm(x - cIn, kd=False)

        if deriv == 0:
            out = zeros((x.size,)) if out is None else out
            evalKerVal(x, bIn, dIn, eIn, out=out)

        elif deriv == 1:
            out = zeros((x.size,)) if out is None else out
            evalKerDeriv(x, bIn, dIn, eIn, out=out)

        elif deriv in ([0,1],(0,1)):
            out = [zeros((x.size,)),zeros((x.size,))] if out is None else out
            evalKerValnDeriv(x, bIn, dIn, eIn, out=out)
        else:
            assert 0,"Input deriv could not be interpreted"
        
        if fullOut:
            assert 0, 'TBD'
    else:
        nK = len(bIn)
        nPt = x.shape[1]
        if deriv in (0,1):
            out = zeros((nK,nPt))
        else:
            out = [zeros((nK,nPt)),zeros((nK,nPt))]
        
        eIn = nK*[0.] if eIn is None else eIn
        
        if fullOut:
            assert 0, 'TBD'
        
        for k,(b,d,e) in enumerate(zip(bIn,dIn,eIn)):
            
            # if c is not NOne, then x is vector
            if cIn is not None:
                xn = norm(x-cIn[:,[k]],kd=False)
            else:
                xn = x
            
            if deriv == 0:
                evalKerVal(xn, b, d, e, out=out[k, :])
            elif deriv == 1:
                evalKerDeriv(xn, b, d, e, out=out[k, :])
            elif deriv in ([0,1],(0,1)):
                evalKerValnDeriv(xn, b, d, e, out=[out[0][k,:],out[1][k,:]])
            else:
                assert 0,"Input deriv could not be interpreted"
        if fullOut:
            assert 0, 'TBD'
    
    return out


def c1PolyKerPartialDeriv(x,bIn,dIn,eIn=None,cIn=None,fullOut=False,kerFun=c1KerFun,pdiffKerFun=c1KerFunPartial,inPolarCoords=False):
    """Returns the partialderivates of a polynomial kernel
        x : numpy array with x values
        bIn : base of the kernel float or array
        dIn : zone of constant change
        eIn : kernel value outside of base
        deriv : 0 : kernel value, 1 : derivative of kernel value, [0,1] : both
        """
    # b = base of the kernel
    # d = zone of constant velocity b*(.5-d) = p1 <-> b'(.5+d) = p2
    
    # Function values
    # Within first constant acceleration
    # f0 = y00+dy00*x-1/2*ddy*x**2
    # d/dx f0 = dy00-ddy*x
    # Within zero acceleration phase
    # f1 = y01+dy01*(x-p1)
    # d/dx f1 = dy01
    # Within second acceleration phase
    # f2 = y02+dy02*(x-p2)+1/2*ddy*(x-p2)**2
    # d/dx f2 = dy02+ddy*(x-p2)
    assert 0, 'TBD'