from coreUtils import *

c1PolyMaxDeform = lambda b,d,tn=1.,e=0:-tn*((2*(e-1))/(b+2*b*d))
c1PolyMinB = lambda tn,d,safeFac=0.5,e=0:-(tn/safeFac)*(2*e-2)/(1+2*d)

# c1KerFun = compile('ddy = 4.0*(e - 1.0)/(b**2*(4.0*d**2 - 1.0));y00 = 1.00000000000000;dy00 = 0.0;y01 = 0.5*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0);dy01 = 2.0*(e - 1.0)/(b*(2.0*d + 1.0));y02 = 0.5*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0);dy02 = 2.0*(e - 1.0)/(b*(2.0*d + 1.0));', '<string>', 'exec')
# c1KerFunPartial = {'b':compile('ddy = -8.0*(e - 1.0)/(b**3*(4.0*d**2 - 1.0));y00 = 0;dy00 = 0;y01 = 0;dy01 = -2.0*(e - 1.0)/(b**2*(2.0*d + 1.0));y02 = 0;dy02 = -2.0*(e - 1.0)/(b**2*(2.0*d + 1.0));y03=0.;', '<string>', 'exec'),
#                   'd':compile('ddy = -32.0*d*(e - 1.0)/(b**2*(4.0*d**2 - 1.0)**2);y00 = 0;dy00 = 0;y01 = 0.5*(-2.0*e + 6.0)/(2.0*d + 1.0) - 1.0*(4.0*d - (2.0*d - 1.0)*(e - 1.0) + 2.0)/(2.0*d + 1.0)**2;dy01 = -4.0*(e - 1.0)/(b*(2.0*d + 1.0)**2);y02 = 0.5*(6.0*e - 2.0)/(2.0*d + 1.0) - 1.0*(6.0*d*e - 2.0*d + e + 1.0)/(2.0*d + 1.0)**2;dy02 = -4.0*(e - 1.0)/(b*(2.0*d + 1.0)**2);y03=0.;', '<string>', 'exec'),
#                   'e':compile('ddy = 4.0/(b**2*(4.0*d**2 - 1.0));y00 = 0;dy00 = 0;y01 = 0.5*(-2.0*d + 1.0)/(2.0*d + 1.0);dy01 = 2.0/(b*(2.0*d + 1.0));y02 = 0.5*(6.0*d + 1)/(2.0*d + 1.0);dy02 = 2.0/(b*(2.0*d + 1.0));y03=1.;', '<string>', 'exec')}

from polyKerCompiled import c1KerFun,c1KerFunPartial

# For sliced dist2nondiffeo
# -> point list
def nonDiffeoCheckPointList(b:float,d:float):
    return np.array([0., b*(d + 0.5), b*(-d + 0.5), b])

nonDiffeoCheckPointListLen = len(nonDiffeoCheckPointList(1.,.1))

def c1PolyKer(x,bIn,dIn,eIn=None,deriv=0,cIn=None,out=None,fullOut=False,kerFun=c1KerFun,inPolarCoords=False):
    """Returns the coefficients of a polynomial kernel
    x : numpy array with x values
    bIn : base of the kernel float or array
    dIn : zone of constant change
    eIn : kernel value outside of base
    deriv : 0 : kernel value, 1 : derivative of kernel value, [0,1] : both
    out : optional array where results are added
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
    
    # debug
    # if not np.all(np.isfinite(x)):
    #    print(x)
    #    print('Problem')
    
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
        b,d,e = bIn,dIn,eIn

        # If c is given x is vector else scalar
        if cIn is not None:
            x = norm(x-cIn,kd=False)
        
        p1 = (.5-d)*b
        p2 = (.5+d)*b

        # Compute kernel values and the first and second deriv at p1/p2
        # ddy = 4.0*(e-1.0)/(b**2*(4.0*d**2-1.0))
        # y00 = 1.0
        # dy00 = 0.0
        # y01 = 0.5*(4.0*d-(2.0*d-1.0)*(e-1.0)+2.0)/(2.0*d+1.0)
        # dy01 = 2.0*(e-1.0)/(b*(2.0*d+1.0))
        # y02 = 0.5*(6.0*d*e-2.0*d+e+1.0)/(2.0*d+1.0)
        # dy02 = 2.0*(e-1.0)/(b*(2.0*d+1.0))
        # TBD change this; exec does not work within functions as locals() is tricky
        # vValues['b'] = b; vValues['d'] = d; vValues['e'] = e
        # exec(kerFunc, {}, vValues)
        # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
        ddy,y00,dy00,y01,dy01,y02,dy02,y03 = kerFun(b,d,e)
        
        # Compute which point is in which section of the piece-wise defined function
        i0 = x <= p1
        i1 = np.logical_and(np.logical_not(i0),x <= p2)
        i2 = np.logical_and(np.logical_not(np.logical_or(i0,i1)),x <= b)
        i3 = x > b
        
        xTemp0 = x[i0]  # This performs an actual copy
        xTemp1 = x[i1]-p1
        xTemp2 = x[i2]-p2  # This performs an actual copy
        
        if deriv == 0:
            out = zeros((x.size,)) if out is None else out
            
            out[i0] += y00+dy00*xTemp0-(1/2*ddy)*square(xTemp0)
            out[i1] += y01+dy01*xTemp1
            out[i2] += y02+dy02*xTemp2+(1/2*ddy)*square(xTemp2)
            out[i3] += e

            # debug
            # if not np.all(np.isfinite(out)):
            #    print('outProb')
        elif deriv == 1:
            out = zeros((x.size,)) if out is None else out
            
            out[i0] += dy00-ddy*xTemp0
            out[i1] += dy01
            out[i2] += dy02+ddy*xTemp2
            # debug
            # if not np.all(np.isfinite(out)):
            #    print('outProb')
        elif deriv in ([0,1],(0,1)):
            out = [zeros((x.size,)),zeros((x.size,))] if out is None else out
            
            out[0][i0] += y00+dy00*xTemp0-(1/2*ddy)*square(xTemp0)
            out[1][i0] += dy00-ddy*xTemp0
            out[0][i1] += y01+dy01*xTemp1
            out[1][i1] += dy01
            out[0][i2] += y02+dy02*xTemp2+(1/2*ddy)*square(xTemp2)
            out[1][i2] += dy02+ddy*xTemp2
            out[0][i3] += e
            # debug
            # if not np.all(np.isfinite(out[0])):
            #    print('outProb')
            # if not np.all(np.isfinite(out[1])):
            #    print('outProb')
        else:
            assert 0,"Input deriv could not be interpreted"
        
        if fullOut:
            out.append([i0,i1,i2,i3])
    else:
        nK = len(bIn)
        nPt = x.shape[1]
        if deriv in (0,1):
            out = zeros((nK,nPt))
        else:
            out = [zeros((nK,nPt)),zeros((nK,nPt))]
        
        eIn = nK*[0.] if eIn is None else eIn
        
        if fullOut:
            i0All = np.zeros((nK,nPt))
            i1All = np.zeros((nK,nPt))
            i2All = np.zeros((nK,nPt))
            i3All = np.zeros((nK,nPt))
        
        for k,(b,d,e) in enumerate(zip(bIn,dIn,eIn)):
            
            # if c is not NOne, then x is vector
            if cIn is not None:
                xn = norm(x-cIn[:,[k]],kd=False)
            else:
                xn = x
            
            p1 = (.5-d)*b
            p2 = (.5+d)*b
            
            # Compute kernel values and the first and second deriv at p1/p2
            # ddy = 4.0 * (e - 1.0) / (b ** 2 * (4.0 * d ** 2 - 1.0))
            # y00 = 1.0
            # dy00 = 0.0
            # y01 = 0.5 * (4.0 * d - (2.0 * d - 1.0) * (e - 1.0) + 2.0) / (2.0 * d + 1.0)
            # dy01 = 2.0 * (e - 1.0) / (b * (2.0 * d + 1.0))
            # y02 = 0.5 * (6.0 * d * e - 2.0 * d + e + 1.0) / (2.0 * d + 1.0)
            # dy02 = 2.0 * (e - 1.0) / (b * (2.0 * d + 1.0))
            # vValues['b'] = b; vValues['d'] = d; vValues['e'] = e
            # exec(kerFunc,{},locals())
            # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
            ddy,y00,dy00,y01,dy01,y02,dy02,y03 = kerFun(b,d,e)
            
            # Compute which point is in which section of the piece-wise defined function
            i0 = xn <= p1
            i1 = np.logical_and(np.logical_not(i0),xn <= p2)
            i2 = np.logical_and(np.logical_not(np.logical_or(i0,i1)),xn <= b)
            i3 = xn > b
            
            if fullOut:
                i0All[k,:] = i0
                i1All[k,:] = i1
                i2All[k,:] = i2
                i3All[k,:] = i3
            
            xTemp0 = xn[i0]  # This performs an actual copy
            xTemp1 = xn[i1]-p1
            xTemp2 = xn[i2]-p2
            
            if deriv == 0:
                out[k,i0] += y00+dy00*xTemp0-(1/2*ddy)*square(xTemp0)
                out[k,i1] += y01+dy01*xTemp1
                out[k,i2] += y02+dy02*xTemp2+(1/2*ddy)*square(xTemp2)
                out[k,i3] += e
                # debug
                # if not np.all(np.isfinite(out[k,:])):
                #    print('outProb')
            elif deriv == 1:
                out[k,i0] += dy00-ddy*xTemp0
                out[k,i1] += dy01
                out[k,i2] += dy02+ddy*xTemp2
                # debug
                # if not np.all(np.isfinite(out[k,:])):
                #    print('outProb')
            elif deriv in ([0,1],(0,1)):
                out[0][k,i0] += y00+dy00*xTemp0-(1/2*ddy)*square(xTemp0)
                out[1][k,i0] += dy00-ddy*xTemp0
                out[0][k,i1] += y01+dy01*xTemp1
                out[1][k,i1] += dy01
                out[0][k,i2] += y02+dy02*xTemp2+(1/2*ddy)*square(xTemp2)
                out[1][k,i2] += dy02+ddy*xTemp2
                out[0][k,i3] += e
                # debug
                # if not np.all(np.isfinite(out[0][k,:])):
                #    print('outProb')
                # debug
                # if not np.all(np.isfinite(out[1][k,:])):
                #    print('outProb')
            else:
                assert 0,"Input deriv could not be interpreted"
        if fullOut:
            out.append([i0All,i1All,i2All,i3All])
    
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
    
    if inPolarCoords:
        norm = cNormWrapped
    else:
        norm = cNorm
    
    # Transition points
    # vValues = {"ddy":0.,"y00":0.,"dy00":0.,"y01":0.,"dy01":0.,"y02":0.,"dy02":0.,"y03":0.}
    if isinstance(bIn,float):
        eIn = 0. if eIn is None else eIn
        b,d,e = bIn,dIn,eIn
        
        # If c is given x is vector else scalar
        if cIn is not None:
            x = norm(x-cIn,kd=False)
        
        p1 = (.5-d)*b
        p2 = (.5+d)*b
        
        # Compute kernel values and the first and second deriv at p1/p2
        # vValues['b'] = b; vValues['d'] = d; vValues['e'] = e
        # exec(kerFun,{},vValues)
        # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
        ddy,y00,dy00,y01,dy01,y02,dy02,y03 = kerFun(b,d,e)
        
        # Compute which point is in which section of the piece-wise defined function
        i0 = x <= p1
        i1 = np.logical_and(np.logical_not(i0),x <= p2)
        i2 = np.logical_and(np.logical_not(np.logical_or(i0,i1)),x <= b)
        i3 = x > b
        
        xTemp0 = x[i0]  # This performs an actual copy
        xTemp1 = x[i1]-p1
        xTemp2 = x[i2]-p2
        
        xTemp0s = square(xTemp0)
        xTemp1s = square(xTemp1)
        xTemp2s = square(xTemp2)
        
        out = {'kVal':zeros((x.size,)),'kdVal':zeros((x.size,)),'b':zeros((x.size,)),'d':zeros((x.size,)),'e':zeros((x.size,))}
        
        out['kVal'][i0] += y00+dy00*xTemp0-(1/2*ddy)*xTemp0s
        out['kdVal'][i0] += dy00-ddy*xTemp0
        out['kVal'][i1] += y01+dy01*xTemp1
        out['kdVal'][i1] += dy01
        out['kVal'][i2] += y02+dy02*xTemp2+(1/2*ddy)*xTemp2s
        out['kdVal'][i2] += dy02+ddy*xTemp2
        out['kVal'][i3] += e
        
        for apdiff in ('b','d','e'):
            # exec(pdiffKerFun[apdiff],{},vValues)
            # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
            # ddy,y00,dy00,y01,dy01,y02,dy02,y03 = c1KerFunPartial(apdiff,b,d,e)
            # Attention the above call replaces the values of y00 etc with the values of the pdiff
            # out[apdiff][i0] += y00+dy00*xTemp0-(1/2*ddy)*xTemp0s
            # out[apdiff][i1] += y01+dy01*xTemp1
            # out[apdiff][i2] += y02+dy02*xTemp2+(1/2*ddy)*xTemp2s
            # out[apdiff][i3] += y03
            # c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None,x3=None,xS3=None):
            xTemp0 = x[i0]  # This performs an actual copy
            xTemp1 = x[i1]
            xTemp2 = x[i2]
            
            xTemp0s = square(xTemp0)
            xTemp1s = square(xTemp1)
            xTemp2s = square(xTemp2)
            out[apdiff] = pdiffKerFun(apdiff,i0,i1,i2,i3,b,d,e,x0=xTemp0,xS0=xTemp0s,x1=xTemp1,xS1=xTemp1s,x2=xTemp2,xS2=xTemp2s)
        
        if fullOut:
            out.update({'i0':i0,'i1':i1,'i2':i2,'i3':i3})
    else:
        nK = len(bIn)
        nPt = x.shape[1]
        out = {'kVal':zeros((nK,nPt)),'kdVal':zeros((nK,nPt)),'b':zeros((nK,nPt)),'d':zeros((nK,nPt)),'e':zeros((nK,nPt))}
        
        eIn = nK*[0.] if eIn is None else eIn
        
        if fullOut:
            i0All = np.zeros((nK,nPt))
            i1All = np.zeros((nK,nPt))
            i2All = np.zeros((nK,nPt))
            i3All = np.zeros((nK,nPt))
        
        for k,(b,d,e) in enumerate(zip(bIn,dIn,eIn)):
            
            # if c is not NOne, then x is vector
            if cIn is not None:
                xn = norm(x-cIn[:,[k]],kd=False)
            else:
                xn = x
            
            p1 = (.5-d)*b
            p2 = (.5+d)*b
            
            # Compute which point is in which section of the piece-wise defined function
            i0 = xn <= p1
            i1 = np.logical_and(np.logical_not(i0),xn <= p2)
            i2 = np.logical_and(np.logical_not(np.logical_or(i0,i1)),xn <= b)
            i3 = xn > b
            
            if fullOut:
                i0All[k,:] = i0
                i1All[k,:] = i1
                i2All[k,:] = i2
                i3All[k,:] = i3
            
            xTemp0 = xn[i0]  # This performs an actual copy
            xTemp1 = xn[i1]-p1
            xTemp2 = xn[i2]-p2
            
            xTemp0s = square(xTemp0)
            xTemp1s = square(xTemp1)
            xTemp2s = square(xTemp2)
            
            # Compute kernel values and the first and second deriv at p1/p2
            # vValues['b'] = b; vValues['d'] = d; vValues['e'] = e
            # exec(kerFun,{},vValues)
            # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
            ddy,y00,dy00,y01,dy01,y02,dy02,y03 = kerFun(b,d,e)
            ddyhalf = 1/2*ddy
            
            out['kVal'][k,i0] += y00+dy00*xTemp0-ddyhalf*xTemp0s
            out['kdVal'][k,i0] += dy00-ddy*xTemp0
            out['kVal'][k,i1] += y01+dy01*xTemp1
            out['kdVal'][k,i1] += dy01
            out['kVal'][k,i2] += y02+dy02*xTemp2+ddyhalf*xTemp2s
            out['kdVal'][k,i2] += dy02+ddy*xTemp2
            out['kVal'][k,i3] += e
            
            for apdiff in ('b','d','e'):
                # exec(pdiffKerFun[apdiff],{},vValues)
                # ddy=vValues['ddy'];y00=vValues['y00'];dy00=vValues['dy00'];y01=vValues['y01'];dy01=vValues['dy01'];y02=vValues['y02'];dy02=vValues['dy02'];y03=vValues['y03']
                # ddy,y00,dy00,y01,dy01,y02,dy02,y03 = c1KerFunPartial(apdiff,b,d,e)
                # Attention the above call replaces the values of y00 etc with the values of the pdiff
                # out[apdiff][k,i0] += y00+dy00*xTemp0-ddyhalf*xTemp0s
                # out[apdiff][k,i1] += y01+dy01*xTemp1
                # out[apdiff][k,i2] += y02+dy02*xTemp2+ddyhalf*xTemp2s
                # out[apdiff][k,i3] += y03
                xTemp0 = xn[i0]  # This performs an actual copy
                xTemp1 = xn[i1]
                xTemp2 = xn[i2]
                
                xTemp0s = square(xTemp0)
                xTemp1s = square(xTemp1)
                xTemp2s = square(xTemp2)
                # def c1KerFunPartial(diffVar,i0,i1,i2,i3,b,d,e,x=None,xS=None,x0=None,xS0=None,x1=None,xS1=None,x2=None,xS2=None)
                out[apdiff][k,:] = pdiffKerFun(apdiff,i0,i1,i2,i3,b,d,e,x0=xTemp0,xS0=xTemp0s,x1=xTemp1,xS1=xTemp1s,x2=xTemp2,xS2=xTemp2s)
        if fullOut:
            out.update({'i0':i0All,'i1':i0All,'i2':i0All,'i3':i0All})
    return out