import re

from copy import deepcopy

from coreUtils import *

from sympy.solvers import solve as sysolve

from sympy.abc import x

from collections import defaultdict
recursivedict = lambda: defaultdict(recursivedict)

splineDict = recursivedict()

presolveDict = {}
presolveDict[3] = {'None':{'0101':None}, 1:{'0101':None}, 2:{'0101':None}}
presolveDict[3]['None']['0101'] = '(x - 0.5)**3*(2.0*c0*t1 - 2.0*c0*t3 - c1*t0**2 + 2.0*c1*t0*t3 + c1*t2**2 - 2.0*c1*t2*t3 - 2.0*c2*t1 + 2.0*c2*t3 + c3*t0**2 - 2.0*c3*t0*t1 + 2.0*c3*t1*t2 - c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + 0.5*(x - 0.5)**2*(-6.0*c0*t1**2 + 6.0*c0*t1 + 6.0*c0*t3**2 - 6.0*c0*t3 + 2.0*c1*t0**3 - 3.0*c1*t0**2 - 6.0*c1*t0*t3**2 + 6.0*c1*t0*t3 - 2.0*c1*t2**3 + 3.0*c1*t2**2 + 6.0*c1*t2*t3**2 - 6.0*c1*t2*t3 + 6.0*c2*t1**2 - 6.0*c2*t1 - 6.0*c2*t3**2 + 6.0*c2*t3 - 2.0*c3*t0**3 + 3.0*c3*t0**2 + 6.0*c3*t0*t1**2 - 6.0*c3*t0*t1 - 6.0*c3*t1**2*t2 + 6.0*c3*t1*t2 + 2.0*c3*t2**3 - 3.0*c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + 0.25*(x - 0.5)*(24.0*c0*t1**2*t3 - 12.0*c0*t1**2 - 24.0*c0*t1*t3**2 + 6.0*c0*t1 + 12.0*c0*t3**2 - 6.0*c0*t3 - 8.0*c1*t0**3*t3 + 4.0*c1*t0**3 + 12.0*c1*t0**2*t3**2 - 3.0*c1*t0**2 - 12.0*c1*t0*t3**2 + 6.0*c1*t0*t3 + 8.0*c1*t2**3*t3 - 4.0*c1*t2**3 - 12.0*c1*t2**2*t3**2 + 3.0*c1*t2**2 + 12.0*c1*t2*t3**2 - 6.0*c1*t2*t3 - 24.0*c2*t1**2*t3 + 12.0*c2*t1**2 + 24.0*c2*t1*t3**2 - 6.0*c2*t1 - 12.0*c2*t3**2 + 6.0*c2*t3 + 8.0*c3*t0**3*t1 - 4.0*c3*t0**3 - 12.0*c3*t0**2*t1**2 + 3.0*c3*t0**2 + 12.0*c3*t0*t1**2 - 6.0*c3*t0*t1 + 12.0*c3*t1**2*t2**2 - 12.0*c3*t1**2*t2 - 8.0*c3*t1*t2**3 + 6.0*c3*t1*t2 + 4.0*c3*t2**3 - 3.0*c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + (24.0*c0*t1**2*t2**2 - 48.0*c0*t1**2*t2*t3 + 24.0*c0*t1**2*t3 - 6.0*c0*t1**2 - 16.0*c0*t1*t2**3 + 48.0*c0*t1*t2*t3**2 - 24.0*c0*t1*t3**2 + 2.0*c0*t1 + 16.0*c0*t2**3*t3 - 24.0*c0*t2**2*t3**2 + 6.0*c0*t3**2 - 2.0*c0*t3 - 8.0*c1*t0**3*t2**2 + 16.0*c1*t0**3*t2*t3 - 8.0*c1*t0**3*t3 + 2.0*c1*t0**3 + 8.0*c1*t0**2*t2**3 - 24.0*c1*t0**2*t2*t3**2 + 12.0*c1*t0**2*t3**2 - c1*t0**2 - 16.0*c1*t0*t2**3*t3 + 24.0*c1*t0*t2**2*t3**2 - 6.0*c1*t0*t3**2 + 2.0*c1*t0*t3 + 8.0*c1*t2**3*t3 - 2.0*c1*t2**3 - 12.0*c1*t2**2*t3**2 + c1*t2**2 + 6.0*c1*t2*t3**2 - 2.0*c1*t2*t3 + 16.0*c2*t0**3*t1 - 16.0*c2*t0**3*t3 - 24.0*c2*t0**2*t1**2 + 24.0*c2*t0**2*t3**2 + 48.0*c2*t0*t1**2*t3 - 48.0*c2*t0*t1*t3**2 - 24.0*c2*t1**2*t3 + 6.0*c2*t1**2 + 24.0*c2*t1*t3**2 - 2.0*c2*t1 - 6.0*c2*t3**2 + 2.0*c2*t3 - 16.0*c3*t0**3*t1*t2 + 8.0*c3*t0**3*t1 + 8.0*c3*t0**3*t2**2 - 2.0*c3*t0**3 + 24.0*c3*t0**2*t1**2*t2 - 12.0*c3*t0**2*t1**2 - 8.0*c3*t0**2*t2**3 + c3*t0**2 - 24.0*c3*t0*t1**2*t2**2 + 6.0*c3*t0*t1**2 + 16.0*c3*t0*t1*t2**3 - 2.0*c3*t0*t1 + 12.0*c3*t1**2*t2**2 - 6.0*c3*t1**2*t2 - 8.0*c3*t1*t2**3 + 2.0*c3*t1*t2 + 2.0*c3*t2**3 - c3*t2**2)/(16.0*t0**3*t1 - 16.0*t0**3*t3 - 24.0*t0**2*t1**2 + 24.0*t0**2*t3**2 + 48.0*t0*t1**2*t3 - 48.0*t0*t1*t3**2 + 24.0*t1**2*t2**2 - 48.0*t1**2*t2*t3 - 16.0*t1*t2**3 + 48.0*t1*t2*t3**2 + 16.0*t2**3*t3 - 24.0*t2**2*t3**2)'
presolveDict[3][1]['0101'] = '3*(x - 0.5)**2*(2.0*c0*t1 - 2.0*c0*t3 - c1*t0**2 + 2.0*c1*t0*t3 + c1*t2**2 - 2.0*c1*t2*t3 - 2.0*c2*t1 + 2.0*c2*t3 + c3*t0**2 - 2.0*c3*t0*t1 + 2.0*c3*t1*t2 - c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + 0.5*(2*x - 1.0)*(-6.0*c0*t1**2 + 6.0*c0*t1 + 6.0*c0*t3**2 - 6.0*c0*t3 + 2.0*c1*t0**3 - 3.0*c1*t0**2 - 6.0*c1*t0*t3**2 + 6.0*c1*t0*t3 - 2.0*c1*t2**3 + 3.0*c1*t2**2 + 6.0*c1*t2*t3**2 - 6.0*c1*t2*t3 + 6.0*c2*t1**2 - 6.0*c2*t1 - 6.0*c2*t3**2 + 6.0*c2*t3 - 2.0*c3*t0**3 + 3.0*c3*t0**2 + 6.0*c3*t0*t1**2 - 6.0*c3*t0*t1 - 6.0*c3*t1**2*t2 + 6.0*c3*t1*t2 + 2.0*c3*t2**3 - 3.0*c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + 0.25*(24.0*c0*t1**2*t3 - 12.0*c0*t1**2 - 24.0*c0*t1*t3**2 + 6.0*c0*t1 + 12.0*c0*t3**2 - 6.0*c0*t3 - 8.0*c1*t0**3*t3 + 4.0*c1*t0**3 + 12.0*c1*t0**2*t3**2 - 3.0*c1*t0**2 - 12.0*c1*t0*t3**2 + 6.0*c1*t0*t3 + 8.0*c1*t2**3*t3 - 4.0*c1*t2**3 - 12.0*c1*t2**2*t3**2 + 3.0*c1*t2**2 + 12.0*c1*t2*t3**2 - 6.0*c1*t2*t3 - 24.0*c2*t1**2*t3 + 12.0*c2*t1**2 + 24.0*c2*t1*t3**2 - 6.0*c2*t1 - 12.0*c2*t3**2 + 6.0*c2*t3 + 8.0*c3*t0**3*t1 - 4.0*c3*t0**3 - 12.0*c3*t0**2*t1**2 + 3.0*c3*t0**2 + 12.0*c3*t0*t1**2 - 6.0*c3*t0*t1 + 12.0*c3*t1**2*t2**2 - 12.0*c3*t1**2*t2 - 8.0*c3*t1*t2**3 + 6.0*c3*t1*t2 + 4.0*c3*t2**3 - 3.0*c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2)'
presolveDict[3][2]['0101'] = '3*(2*x - 1.0)*(2.0*c0*t1 - 2.0*c0*t3 - c1*t0**2 + 2.0*c1*t0*t3 + c1*t2**2 - 2.0*c1*t2*t3 - 2.0*c2*t1 + 2.0*c2*t3 + c3*t0**2 - 2.0*c3*t0*t1 + 2.0*c3*t1*t2 - c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2) + 1.0*(-6.0*c0*t1**2 + 6.0*c0*t1 + 6.0*c0*t3**2 - 6.0*c0*t3 + 2.0*c1*t0**3 - 3.0*c1*t0**2 - 6.0*c1*t0*t3**2 + 6.0*c1*t0*t3 - 2.0*c1*t2**3 + 3.0*c1*t2**2 + 6.0*c1*t2*t3**2 - 6.0*c1*t2*t3 + 6.0*c2*t1**2 - 6.0*c2*t1 - 6.0*c2*t3**2 + 6.0*c2*t3 - 2.0*c3*t0**3 + 3.0*c3*t0**2 + 6.0*c3*t0*t1**2 - 6.0*c3*t0*t1 - 6.0*c3*t1**2*t2 + 6.0*c3*t1*t2 + 2.0*c3*t2**3 - 3.0*c3*t2**2)/(2.0*t0**3*t1 - 2.0*t0**3*t3 - 3.0*t0**2*t1**2 + 3.0*t0**2*t3**2 + 6.0*t0*t1**2*t3 - 6.0*t0*t1*t3**2 + 3.0*t1**2*t2**2 - 6.0*t1**2*t2*t3 - 2.0*t1*t2**3 + 6.0*t1*t2*t3**2 + 2.0*t2**3*t3 - 3.0*t2**2*t3**2)'

def cstrDerivHash(cstrList):
    hashStr = ''
    for aCstr in cstrList:
        hashStr += '{0:d}'.format(aCstr[0])
    return hashStr

def myStringSub(syStr, tStr, cStr, cstrList):
    
    for tt,cc,aCstr in zip(tStr, cStr, cstrList):
        syStr = re.sub(tt,'{0:.12e}'.format(aCstr[1]), syStr)
        syStr = re.sub(cc,'{0:.12e}'.format(aCstr[2]), syStr)
    
    return syStr

def myStringSubCmp(syStr, tStrCmp, cStrCmp, cstrList):
    
    for tt,cc,aCstr in zip(tStrCmp,cStrCmp,cstrList):
        syStr = tt.sub('{0:.12e}'.format(aCstr[1]),syStr)
        syStr = cc.sub('{0:.12e}'.format(aCstr[2]),syStr)
    
    return syStr

def myStringSubCmp2(syStr, cstrList):
    #return syStr.format(*[ a for tup in cstrList for a in tup])
    return syStr.format(*[ a for tup in map(lambda a: a[1:], cstrList) for a in tup])


def lambdaExecution(fLambda, x, cstrList):
    
    #n = x.size
    
    #t = []
    #c = []
    #for _,at,ac in cstrList:
    #    t.append(np.tile(at,(n,)))
    #    c.append(np.tile(ac,(n,)))
    
    #return fLambda(*t, *c, x)
    n = len(cstrList)
    tc = (2*n+1)*[None]
    for k,aCstr in enumerate(cstrList):
        tc[k] = aCstr[1]
        tc[n+k] = aCstr[2]
    # Needed for pytohn3.4
    tc[-1] = x
    return fLambda(*tc)

def getSpline(deg, cstrList:List[List], getDeriv:int=[], doSave:bool=True)->"lambdaFunc":
    
    gotProb=False # type:bool
    presolved=False # type:bool
    if doSave:
        try:
            thisDict = splineDict[deg][getDeriv if getDeriv else "None"][ cstrDerivHash(cstrList) ]
            assert thisDict
            #print('reuse')
            #aStr = thisDict['aStr']
            #tStrCmp = thisDict['tStrCmp']
            #cStrCmp = thisDict['cStrCmp']
            #fStr = thisDict['fStr']
            #fStrRep = thisDict['fStrRep']
            fStrLamda = thisDict['fStrLamda']
            #sol = thisDict['sol']
            gotProb = True
        except (KeyError, AssertionError) as thisExcept:
            #Check presolve
            try:
                fStr = presolveDict[deg][getDeriv if getDeriv else "None"][cstrDerivHash(cstrList)]
                presolved = True
            except (KeyError,AssertionError) as thisExcept:
                pass
                
            
    if not gotProb:
        print('new')
        a = sy.symbols("a0:{0:d}".format(deg+1))
        t = sy.symbols("t0:{0:d}".format(len(cstrList)))
        c = sy.symbols("c0:{0:d}".format(len(cstrList)))
        f = a[0]
        
        if not presolved:
            for k in range(1,1+deg):
                f += a[k]*(x-0.5)**k
        
            eq = []
            for aCstr,tCstrSym,aCstrSym in zip(cstrList,t,c):
                if aCstr[0]>0:
                    fd = sy.diff(f, *aCstr[0]*[x])
                else:
                    fd = f
                eq.append( fd.subs(x,tCstrSym)-aCstrSym )#eq.append( fd.subs(x,aCstr[1])-aCstr[2] )
    
            sol = sysolve(eq, a)
        
            assert sol, 'Result dict empty'
    
            if getDeriv:
                getDeriv=int(getDeriv)
                assert getDeriv>0
                f = sy.diff(f,*getDeriv*[x])
        
            # Create entire function
            #fi = sy.lambdify([x,t,c], f.subs(sol), 'numpy')
            fStr = str(f.subs(sol))

            aStr = [str(aa) for aa in a]
            tStr = [str(tt) for tt in t]
            cStr = [str(cc) for cc in c]
            aStrCmp = [re.compile(str(aa)) for aa in a]
            tStrCmp = [re.compile(str(tt)) for tt in t]
            cStrCmp = [re.compile(str(cc)) for cc in c]

            idx = 0
            fStrRep = deepcopy(fStr)

            for aT,aC in zip(tStr,cStr):
                fStrRep = re.sub(aT,"{{{0}:.12e}}".format(idx),fStrRep)
                idx += 1
                fStrRep = re.sub(aC,"{{{0}:.12e}}".format(idx),fStrRep)
                idx += 1

            if doSave:
                splineDict[deg][getDeriv if getDeriv else 'None'][cstrDerivHash(cstrList)] = {'a':a,'t':t,'c':c,'f':f,'sol':sol,'aStr':aStr,'tStr':tStr,'cStr':cStr,'aStrCmp':aStrCmp,'tStrCmp':tStrCmp,'cStrCmp':cStrCmp,'fStr':fStr,'fStrRep':fStrRep,'fStrLamda':fStrLamda}
            
        else:
            for k,aa in enumerate(a):
                exec("{0}=a[{1}]".format(aa, k))
            for k,tt in enumerate(t):
                exec("{0}=t[{1}]".format(tt,k))
            for k,cc in enumerate(c):
                exec("{0}=c[{1}]".format(cc,k))
            f = eval(fStr)

            # Not valid in python3.4?
            #fStrLamda = sy.lambdify([*t,*c,x], f, 'numpy')
            fStrLamda = sy.lambdify(list(t)+list(c)+[x],f,'numpy')
        
            if doSave:
                splineDict[deg][getDeriv if getDeriv else 'None'][ cstrDerivHash(cstrList) ] = {'a':a, 't':t, 'c':c, 'f':f, 'fStr':fStr, 'fStrLamda':fStrLamda}
        
    
    ##solNCstr = deepcopy(sol)
    # Evaluation time
    ##solNCstr.update(dict(zip(t, map(lambda aCstr: aCstr[1], cstrList))))
    # Function value
    ##solNCstr.update(dict(zip(c,map(lambda aCstr:aCstr[2], cstrList))))
    #aaa = f.subs(solNCstr)
    ##fn = sy.lambdify([x], f.subs(solNCstr), 'numpy')
    # Create lambda funtion
    #fStr = myStringSub(fStr, tStr, cStr, cstrList)
    #fStr = myStringSubCmp(fStr,tStrCmp,cStrCmp,cstrList)
    #fStr = myStringSubCmp2(fStrRep, cstrList)
    #ff = eval(fStr)
    #fn = sy.lambdify([x], ff, 'numpy')
    
    fn = lambda x: lambdaExecution(fStrLamda, x, cstrList)

    return fn

def getSpline3(x0,x0d,x1,x1d,getDeriv=[], doSave=True):


    x0 = np.array(x0).flatten()
    x0d = np.array(x0d).flatten()
    x1 = np.array(x1).flatten()
    x1d = np.array(x1d).flatten()

    if len(x0)==1:
        return getSpline(3, [ [0,0.,x0[0]], [1,0.,x0d[0]], [0,1.,x1[0]], [1,1.,x1d[0]] ],getDeriv=getDeriv, doSave=doSave)
    else:
        allF = []
        for ax0,ax0d,ax1,ax1d in zip(x0,x0d,x1,x1d):
            allF.append(getSpline(3, [ [0,0.,ax0], [1,0.,ax0d], [0,1.,ax1], [1,1.,ax1d] ],getDeriv=getDeriv, doSave=doSave))

        return lambda x: np.vstack( list(map(lambda aF: aF(x),allF)) )
