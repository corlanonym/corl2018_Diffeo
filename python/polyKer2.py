# Computes coefficients for the kernels induced by constant but non differentiable "accelerations"

# forward transformation
# Variables for continuity
from copy import deepcopy

from coreUtils import lmap
import sympy as sy

from sympy.printing import cxxcode

import re

y00, dy00, y01, dy01, y02, dy02, y03, dy03 = toReplaceVars = sy.symbols("y00, dy00, y01, dy01, y02, dy02, y03, dy03")
# Coefficients
ddyFirst, ddySecond,dddyFirst, dddySecond,b, e, d, s, x, d2 = sy.symbols("ddyFirst, ddySecond, dddyFirst, dddySecond, b, e, d, s, x, d2")
# a corresponds to the constant acceleration
# b corresponds to the total size of the base (actual half only one direction is considered)
# d offset in the middle; the velocity is constant in [(.5-d)*b, (.5+d)*b[
# e is the function value at b and more
toReplaceVars = list(toReplaceVars)
toReplaceVars += [ddyFirst, ddySecond,dddyFirst, dddySecond]

# Transition points
p1noB = (.5-d)
p2noB = (.5+d)
p3noB = (1.-d2)
p4noB = (1.)

p1 = p1noB*b
p2 = p2noB*b
p3 = p3noB*b
p4 = p4noB*b

# Function values
# Within first constant acceleration
f0pp = ddyFirst
f0p = sy.integrate(f0pp, x) + dy00
f0 = sy.integrate(f0p, x) + y00
# Within linear change acceleration phase
f1pp = ddyFirst + dddyFirst*(x-p1)
f1p = sy.integrate(f1pp, x) + dy01
f1 = sy.integrate(f1p, x) + y01
# Within second acceleration phase
f2pp = ddySecond
f2p = sy.integrate(f2pp, x) + dy02
f2 = sy.integrate(f2p, x) + y02
# with in second linear change
f3pp = ddySecond + dddySecond*(x-p3)
f3p = sy.integrate(f3pp, x) + dy03
f3 = sy.integrate(f3p, x) + y03

#outside
f4 = e
f4p = 0.
f4pp = 0.

funcNameList = ['f0', 'f0p', 'f0pp', 'f1', 'f1p', 'f1pp', 'f2', 'f2p', 'f2pp', 'f3', 'f3p', 'f3pp', 'f4', 'f4p', 'f4pp', 'p1', 'p2', 'p3', 'p4', 'p1noB', 'p2noB', 'p3noB', 'p4noB']
funcNameList += lmap(lambda a: str(a), toReplaceVars)

funcDict = dict(lmap( lambda aName: (aName, eval(aName)), funcNameList ))
funcDictStr = dict(lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), funcDict.items()))

# Set up equation system
eq = []
eq.append( f0.subs(x,0)-1 ) # At zero the function has to equal 1
eq.append( f0p.subs(x,0) ) # with zero change rate
eq.append( f3.subs(x,p4)-e ) # At p4 (the end of the compact influence region) the influence is e
eq.append( f3p.subs(x,p4) ) # with zero change rate
eq.append( f3pp.subs(x,p4) ) # with zero change of change rate

eq.append( f0.subs(x, p1) - f1.subs(x, p1) ) # Continuity of position at p1
eq.append( f0p.subs(x, p1) - f1p.subs(x, p1) ) # Continuity of change rate at p1
#eq.append( f0pp.subs(x, p1) - f1pp.subs(x, p1) ) # Continuity of change of change rate at p1

eq.append( f1.subs(x, p2) - f2.subs(x, p2) ) # Continuity of position at p2
eq.append( f1p.subs(x, p2) - f2p.subs(x, p2) ) # Continuity of change rate at p2
eq.append( f1pp.subs(x, p2) - f2pp.subs(x, p2) ) # Continuity of change of change rate at p2

eq.append( f2.subs(x, p3) - f3.subs(x, p3) ) # Continuity of position at p3
eq.append( f2p.subs(x, p3) - f3p.subs(x, p3) ) # Continuity of change rate at p3
#eq.append( f2pp.subs(x, p2) - f3pp.subs(x, p2) ) # Continuity of change of change rate at p3


# Solve the equation system to get a
sol1 = sy.solve(eq, [ddyFirst, ddySecond,dddyFirst, dddySecond, y00, dy00, y01, dy01, y02, dy02, y03, dy03])
for aKey,aVal in sol1.items():
    sol1[aKey] = sy.simplify(aVal)


solExpr = dict( lmap( lambda aVar: (str(aVar), sol1[aVar]), toReplaceVars ) )
solExprStr = dict( lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), solExpr.items()))

# Compute the highest change rate

#xMax = p1 + (p2-p1)*(-sol1[dddySecond]/sol1[dddyFirst])
xMax = p1 + (p2-p1)*((-ddyFirst)/(-ddyFirst+ddySecond))
dyMax = f1p.subs(x,xMax)
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
with open('polyKerCompiled2Temp.py', 'r') as f:
    allLines = f.readlines()
for k,aLine in enumerate(allLines):
    for aKey, aVal in allExpStr.items():
        thisPattern = "__{0}__".format(aKey)
        aLine = re.sub(thisPattern, aVal, aLine)
    allLines[k] = aLine
with open('polyKerCompiled2.py', 'w+') as f:
    f.writelines(allLines)


for kk, dd in sol1.items():
    print(kk)
    print(dd)

    
    
    
    
