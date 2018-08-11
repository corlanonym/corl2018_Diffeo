# Computes coefficients for the kernels induced by constant but non differentiable "accelerations"

# forward transformation
# Variables for continuity
from coreUtils import lmap
import sympy as sy

from sympy.printing import cxxcode

y00, dy00, y01, dy01, y02, dy02 = sy.symbols("y00, dy00, y01, dy01, y02, dy02")
# Coefficients
ddy, b, e, d, s, x = sy.symbols("ddy, b, e, d, s, x")
# a corresponds to the constant acceleration
# b corresponds to the total size of the base (actual half only one direction is considered)
# d offset in the middle; the velocity is constant in [(.5-d)*b, (.5+d)*b[
# e is the function value at b and more

# Transition points
p1 = (.5-d)*b
p2 = (.5+d)*b

# Function values
# Within first constant acceleration
f0 = y00 + dy00*x - 1/2*ddy*x**2
f0p = sy.diff(f0, x)
# Within zero acceleration phase
f1 = y01 + dy01*(x-p1)
f1p = sy.diff(f1, x)
# Within second acceleration phase
f2 = y02 + dy02*(x-p2) + 1/2*ddy*(x-p2)**2
f2p = sy.diff(f2, x)

# Set up equation system
eq = []
eq.append( f0.subs(x,0)-1 ) # At zero the function has to equal 1
eq.append( f0p.subs(x,0) ) # with zero change rate
eq.append( f2.subs(x,b)-e ) # At b (the end of the compact influence region) the influence is e
eq.append( f2p.subs(x,b) ) # with zero change rate
eq.append( f0.subs(x, p1) - f1.subs(x, p1) ) # Continuity of position at p1
eq.append( f0p.subs(x, p1) - f1p.subs(x, p1) ) # Continuity of change rate at p1
eq.append( f1.subs(x, p2) - f2.subs(x, p2) ) # Continuity of position at p2
eq.append( f1p.subs(x, p2) - f2p.subs(x, p2) ) # Continuity of change rate at p2

# Solve the equation system to get a
sol1 = sy.solve(eq, [ddy, y00, dy00, y01, dy01, y02, dy02])


# Additional vars for actual sol
v, s = sy.symbols("v, s")
# First part
f0Sol = f0.subs( lmap( lambda aVar: (aVar, sol1[aVar]), [y00, dy00, ddy] ) )
f1Sol = f1.subs( lmap( lambda aVar: (aVar, sol1[aVar]), [y01, dy01, ddy] ) )
f2Sol = f2.subs( lmap( lambda aVar: (aVar, sol1[aVar]), [y02, dy02, ddy] ) )
f3Sol=e

print("f0Sol: \n{0}".format(f0Sol))
print("f1Sol: \n{0}".format(f1Sol))
print("f2Sol: \n{0}".format(f2Sol))

print("ddy = {0}".format(ddy.subs(sol1)))
print("y00 = {0}".format(y00.subs(sol1)))
print("dy00 = {0}".format(dy00.subs(sol1)))
print("y01 = {0}".format(y01.subs(sol1)))
print("dy01 = {0}".format(dy01.subs(sol1)))
print("y02 = {0}".format(y02.subs(sol1)))
print("dy02 = {0}".format(dy02.subs(sol1)))
print("y03 = {0}".format(e))

cppdict = {}
cppdict['__ddy_cpp'] = cxxcode( ddy.subs(sol1), standard="C++11")
cppdict['__y00_cpp'] = cxxcode( y00.subs(sol1), standard="C++11")
cppdict['__dy00_cpp'] = cxxcode( dy00.subs(sol1), standard="C++11")
cppdict['__y01_cpp'] = cxxcode( y01.subs(sol1), standard="C++11")
cppdict['__dy01_cpp'] = cxxcode( dy01.subs(sol1), standard="C++11")
cppdict['__y02_cpp'] = cxxcode( y02.subs(sol1), standard="C++11")
cppdict['__dy02_cpp'] = cxxcode( dy02.subs(sol1), standard="C++11")
cppdict['_y03_cpp'] = cxxcode( e, standard="C++11")

fIn = open("./cpp/src/c1Kernel_temp.h", 'r')
fOut = open("./cpp/src/c1Kernel.h", 'w+')

import re

for line in fIn:
    for key,value in cppdict.items():
        line = re.sub(key, value, line)
    fOut.write(line)

assert 0

all=["ddy = {0}".format(ddy.subs(sol1)), "y00 = {0}".format(y00.subs(sol1)), "dy00 = {0}".format(dy00.subs(sol1)), "y01 = {0}".format(y01.subs(sol1)), "dy01 = {0}".format(dy01.subs(sol1)), "y02 = {0}".format(y02.subs(sol1)),"dy02 = {0}".format(dy02.subs(sol1))]
sstr=""
for a in all:
    sstr += a+";"
print("\n")
print(sstr)
print("\n\n")

print("ddyb = {0}".format( sy.diff(ddy.subs(sol1),b) ))
print("y00b = {0}".format( sy.diff(y00.subs(sol1),b) ))
print("dy00b = {0}".format( sy.diff(dy00.subs(sol1),b) ))
print("y01b = {0}".format( sy.diff(y01.subs(sol1),b) ))
print("dy01b = {0}".format( sy.diff(dy01.subs(sol1),b) ))
print("y02b = {0}".format( sy.diff(y02.subs(sol1),b) ))
print("dy02b = {0}".format( sy.diff(dy02.subs(sol1),b) ))

all = ["ddy = {0}".format( sy.diff(ddy.subs(sol1),b) ),"y00 = {0}".format( sy.diff(y00.subs(sol1),b) ),"dy00 = {0}".format( sy.diff(dy00.subs(sol1),b) ),"y01 = {0}".format( sy.diff(y01.subs(sol1),b) ),"dy01 = {0}".format( sy.diff(dy01.subs(sol1),b) ),"y02 = {0}".format( sy.diff(y02.subs(sol1),b) ),"dy02 = {0}".format( sy.diff(dy02.subs(sol1),b) ), "y03=0."]
sstr=""
for a in all:
    sstr += a+";"
print("\n")
print(sstr)
print("\n\n")

print("ddyd = {0}".format( sy.diff(ddy.subs(sol1),d) ))
print("y00d = {0}".format( sy.diff(y00.subs(sol1),d) ))
print("dy00d = {0}".format( sy.diff(dy00.subs(sol1),d) ))
print("y01d = {0}".format( sy.diff(y01.subs(sol1),d) ))
print("dy01d = {0}".format( sy.diff(dy01.subs(sol1),d) ))
print("y02d = {0}".format( sy.diff(y02.subs(sol1),d) ))
print("dy02d = {0}".format( sy.diff(dy02.subs(sol1),d) ))

all=["ddy = {0}".format( sy.diff(ddy.subs(sol1),d) ),"y00 = {0}".format( sy.diff(y00.subs(sol1),d) ),"dy00 = {0}".format( sy.diff(dy00.subs(sol1),d) ),"y01 = {0}".format( sy.diff(y01.subs(sol1),d) ),"dy01 = {0}".format( sy.diff(dy01.subs(sol1),d) ),"y02 = {0}".format( sy.diff(y02.subs(sol1),d) ),"dy02 = {0}".format( sy.diff(dy02.subs(sol1),d) ), "y03=0."]
sstr=""
for a in all:
    sstr += a+";"
print("\n")
print(sstr)
print("\n\n")

print("ddye = {0}".format( sy.diff(ddy.subs(sol1),e) ))
print("y00e = {0}".format( sy.diff(y00.subs(sol1),e) ))
print("dy00e = {0}".format( sy.diff(dy00.subs(sol1),e) ))
print("y01e = {0}".format( sy.diff(y01.subs(sol1),e) ))
print("dy01e = {0}".format( sy.diff(dy01.subs(sol1),e) ))
print("y02e = {0}".format( sy.diff(y02.subs(sol1),e) ))
print("dy02e = {0}".format( sy.diff(dy02.subs(sol1),e) ))

all=["ddy = {0}".format( sy.diff(ddy.subs(sol1),e) ),"y00 = {0}".format( sy.diff(y00.subs(sol1),e) ),"dy00 = {0}".format( sy.diff(dy00.subs(sol1),e) ),"y01 = {0}".format( sy.diff(y01.subs(sol1),e) ),"dy01 = {0}".format( sy.diff(dy01.subs(sol1),e) ),"y02 = {0}".format( sy.diff(y02.subs(sol1),e) ),"dy02 = {0}".format( sy.diff(dy02.subs(sol1),e) ), "y03=1."]
sstr=""
for a in all:
    sstr += a+";"
print("\n")
print(sstr)

allDict = { "__ddy__":str(ddy.subs(sol1)), "__y00__":str(y00.subs(sol1)), "__dy00__":str(dy00.subs(sol1)), "__y01__":str(y01.subs(sol1)), "__dy01__":str(dy01.subs(sol1)), "__y02__":str(y02.subs(sol1)), "__dy02__":str(dy02.subs(sol1)), "__y03__":"e"}
allDict.update( { "__ddyb__":str(sy.diff(ddy.subs(sol1),b)), "__y00b__":str(sy.diff(y00.subs(sol1),b)), "__dy00b__":str(sy.diff(dy00.subs(sol1),b)), "__y01b__":str(sy.diff(y01.subs(sol1),b)), "__dy01b__":str(sy.diff(dy01.subs(sol1),b)), "__y02b__":str(sy.diff(y02.subs(sol1),b)), "__dy02b__":str(sy.diff(dy02.subs(sol1),b)), "__y03b__":"0.0" } )
allDict.update( { "__ddyd__":str(sy.diff(ddy.subs(sol1),d)), "__y00d__":str(sy.diff(y00.subs(sol1),d)), "__dy00d__":str(sy.diff(dy00.subs(sol1),d)), "__y01d__":str(sy.diff(y01.subs(sol1),d)), "__dy01d__":str(sy.diff(dy01.subs(sol1),d)), "__y02d__":str(sy.diff(y02.subs(sol1),d)), "__dy02d__":str(sy.diff(dy02.subs(sol1),d)), "__y03d__":"0.0" } )
allDict.update( { "__ddye__":str(sy.diff(ddy.subs(sol1),e)), "__y00e__":str(sy.diff(y00.subs(sol1),e)), "__dy00e__":str(sy.diff(dy00.subs(sol1),e)), "__y01e__":str(sy.diff(y01.subs(sol1),e)), "__dy01e__":str(sy.diff(dy01.subs(sol1),e)), "__y02e__":str(sy.diff(y02.subs(sol1),e)), "__dy02e__":str(sy.diff(dy02.subs(sol1),e)), "__y03e__":"1.0" } )

allDict.update( { "__f0b__":str(sy.diff(f0Sol,b)),"__f1b__":str(sy.diff(f1Sol,b)),"__f2b__":str(sy.diff(f2Sol,b)),"__f3b__":str(sy.diff(f3Sol,b)) } )
allDict.update( { "__f0d__":str(sy.diff(f0Sol,d)),"__f1d__":str(sy.diff(f1Sol,d)),"__f2d__":str(sy.diff(f2Sol,d)),"__f3d__":str(sy.diff(f3Sol,d)) } )
allDict.update( { "__f0e__":str(sy.diff(f0Sol,e)),"__f1e__":str(sy.diff(f1Sol,e)),"__f2e__":str(sy.diff(f2Sol,e)),"__f3e__":str(sy.diff(f3Sol,e)) } )

performanceReplace = {'x\*\*2':'xS'}

import re


for key,val in allDict.items():
    for keyRep,valRep in performanceReplace.items():
        allDict[key] = re.sub(keyRep, valRep, val )

        
fIn = open("polyKerCompiledTemp.py", 'r')
fOut = open("polyKerCompiled.py", 'w+')

for line in fIn:
    for key,value in allDict.items():
        line = re.sub(key, value, line)
    fOut.write(line)

fIn.close()
fOut.close()
    








