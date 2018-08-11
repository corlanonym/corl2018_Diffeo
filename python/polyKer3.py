# Computes coefficients for the kernels induced by constant but non differentiable "accelerations"

# forward transformation
# Variables for continuity
from coreUtils import *
from copy import deepcopy

import sympy as sy

from sympy.printing import cxxcode

import re

c0,c1,c2,c3,c4,c5 = c = toReplaceVars = sy.symbols("c:6")

# Coefficients
b, e, d, x = sy.symbols("b, e, d, x")
# a corresponds to the constant acceleration
# b corresponds to the total size of the base (actual half only one direction is considered)
# d offset in the middle; the velocity is constant in [(.5-d)*b, (.5+d)*b[
# e is the function value at b and more
toReplaceVars = list(toReplaceVars)

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
f = 0.
for k,ac in enumerate(c):
    f += ac*x**k

fp = sy.diff(f,x)
fpp = sy.diff(f,x,2)
fppp = sy.diff(f,x,3)

funcNameList = ['f', 'fp', 'p1', 'p2', 'p3', 'p4', 'p1noB', 'p2noB', 'p3noB', 'p4noB']
funcNameList += lmap(lambda a: str(a), toReplaceVars)

funcDict = dict(lmap( lambda aName: (aName, eval(aName)), funcNameList ))
funcDictStr = dict(lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), funcDict.items()))

# Set up equation system
eq = []
eq.append( f.subs(x,0.)-1 ) # At zero the function has to equal 1
eq.append( fp.subs(x,0.) ) # with zero change rate
eq.append( fppp.subs(x,0.) ) # with zero 'jerk'

eq.append( f.subs(x,b)-e ) # At b the function has to equal 1
eq.append( fp.subs(x,b) ) # with zero change rate
eq.append( fpp.subs(x,b) ) # with zero 'acc'


# Solve the equation system to get a
sol1 = sy.solve(eq, c)

solExpr = dict( lmap( lambda aVar: (str(aVar), sol1[aVar]), toReplaceVars ) )
solExprStr = dict( lmap( lambda aKeyVal: (aKeyVal[0], str(aKeyVal[1])), solExpr.items()))

# Compute the highest change rate

solDeriv = sy.solve(fpp.subs(sol1), x)
bTest = float(2.+np.random.rand(1))
solDerivEval = [(aSol.subs(b,bTest)).evalf() for aSol in solDeriv]
solDerivReal = []
for k,aSolDeriv in enumerate(solDerivEval):
    if (0.<aSolDeriv) and (aSolDeriv<bTest):
        solDerivReal.append(solDeriv[k])

assert len(solDerivReal) == 1, '?!?'

xMax = solDerivReal[0]
dyMax = fp.subs(x,xMax)
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
with open('polyKerCompiled3Temp.py', 'r') as file:
    allLines = file.readlines()
for k,aLine in enumerate(allLines):
    for aKey, aVal in allExpStr.items():
        thisPattern = "__{0}__".format(aKey)
        aLine = re.sub(thisPattern, aVal, aLine)
    allLines[k] = aLine
with open('polyKerCompiled3.py', 'w+') as file:
    file.writelines(allLines)


for kk, dd in sol1.items():
    print(kk)
    print(dd)


#Convert to c++
solExprCpp = deepcopy(solExpr)

cppReplaceSymbols = [e, b]
cppSymbols = [ sy.symbols("_{0}".format(aSyms)) for aSyms in cppReplaceSymbols ]
cppReplaceDict = dict(zip(cppReplaceSymbols, cppSymbols))

solExprCppStr = {}
for key,val in solExprCpp.items():
    solExprCppStr[key] = cxxcode(val.subs(cppReplaceDict), standard="C++11")


fIn = open("./cpp/src/minContJerkKernel_temp.h", 'r')
fOut = open("./cpp/src/minContJerkKernel.h", 'w+')

import re

for line in fIn:
    for key,value in solExprCppStr.items():
        line = re.sub("__{0}_cpp".format(key), value, line)
    fOut.write(line)

fIn.close()
fOut.close()

print(solExprCppStr)






    
    
    
