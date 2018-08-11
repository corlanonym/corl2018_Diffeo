# fourth order polynomial minimizing l_infinity norm jerk

# forward transformation
# Variables for continuity
from coreUtils import *
from copy import deepcopy

import sympy as sy

from sympy.printing import cxxcode

import re

ca0,ca1,ca2,ca3 = ca = toReplaceVars = sy.symbols("ca:4")
cb0,cb1,cb2,cb3 = cb = sy.symbols("cb:4")
cc0,cc1,cc2,cc3 = cc = sy.symbols("cc:4")
toReplaceVars = list(toReplaceVars)
toReplaceVars += cb
toReplaceVars += cc

# Coefficients
b, e, d, x = sy.symbols("b, e, d, x")
# a corresponds to the constant acceleration
# b corresponds to the total size of the base (actual half only one direction is considered)
# d offset in the middle; the velocity is constant in [(.5-d)*b, (.5+d)*b[
# e is the function value at b and more


# Transition points
p1noB = .3
p2noB = .5
p3noB = .7
p4noB = 1.

p1 = p1noB*b
p2 = p2noB*b
p3 = p3noB*b
p4 = p4noB*b

# Function values
fa = 0.
fb = 0.
fc = 0.
for k,(aca, acb, acc) in enumerate(zip(reversed(ca),reversed(cb),reversed(cc))):
    fa += aca*x**k
    fb += acb*x**k
    fc += acc*x**k

fap = sy.diff(fa,x) #velocity
fapp = sy.diff(fa,x,2) #acceleration
fappp = sy.diff(fa,x,3) #jerk
fbp = sy.diff(fb,x) #velocity
fbpp = sy.diff(fb,x,2) #acceleration
fbppp = sy.diff(fb,x,3) #jerk
fcp = sy.diff(fc,x) #velocity
fcpp = sy.diff(fc,x,2) #acceleration
fcppp = sy.diff(fc,x,3) #jerk

funcNameList = ['fa', 'fap', 'fb', 'fbp', 'fc', 'fcp', 'p1', 'p2', 'p3', 'p4', 'p1noB', 'p2noB', 'p3noB', 'p4noB']
funcNameList += lmap(lambda a: str(a), toReplaceVars)

funcDict = dict(lmap( lambda aName: (aName, eval(aName)), funcNameList ))
funcDictStr = dict(lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), funcDict.items()))

# Set up equation system
eq = []
eq.append( fappp + fbppp ) #Jerk is equal in absolute
eq.append( fa.subs(x,0.)-1 ) # At zero the function has to equal 1
eq.append( fap.subs(x,0.) ) # with zero change rate
eq.append( fapp.subs(x,0.) ) # with zero acceleration

eq.append(fa.subs(x,b/4.)-fb.subs(x,b/4.))
eq.append(fap.subs(x,b/4.)-fbp.subs(x,b/4.))
eq.append(fapp.subs(x,b/4.)-fbpp.subs(x,b/4.))

eq.append(fb.subs(x,b*3./4.)-fc.subs(x,b*3./4.))
eq.append(fbp.subs(x,b*3./4.)-fcp.subs(x,b*3./4.))
eq.append(fbpp.subs(x,b*3./4.)-fcpp.subs(x,b*3./4.))

eq.append( fc.subs(x,b)-e ) # At b the function has to equal e
eq.append( fcp.subs(x,b) ) # with zero change rate
eq.append( fcpp.subs(x,b) ) # with zero 'acc'

# Solve the equation system to get a
sol1 = sy.solve(eq, *(list(ca)+list(cb)+list(cc)))

for kk, dd in sol1.items():
    print(kk)
    print(dd)

solExpr = dict( lmap( lambda aVar: (str(aVar), sol1[aVar]), toReplaceVars ) )
solExprStr = dict( lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), solExpr.items()))

# The highest change rate occurs at half base
dyMax = fbp.subs(x, b/2.)
dyMax = dyMax.subs(sol1)

solExpr['dyMax'] = dyMax
solExprStr['dyMax'] = str(dyMax)

minBaseExpr = sy.solve(dyMax-1, b)
assert len(minBaseExpr)==1., 'Multiple solutions?!?'

minBaseExpr = minBaseExpr[0]

solExpr['minBaseExpr'] = minBaseExpr
solExprStr['minBaseExpr'] = str(minBaseExpr)


allExpStr = deepcopy(funcDictStr)
allExpStr.update(solExprStr)

# Replace
with open('polyKerCompiled4Temp.py', 'r') as f:
    allLines = f.readlines()
for k,aLine in enumerate(allLines):
    for aKey, aVal in allExpStr.items():
        thisPattern = "__{0}__".format(aKey)
        aLine = re.sub(thisPattern, aVal, aLine)
    allLines[k] = aLine
with open('polyKerCompiled4.py', 'w+') as f:
    f.writelines(allLines)


for kk, dd in sol1.items():
    print(kk)
    print(dd)

    
    
    
    
