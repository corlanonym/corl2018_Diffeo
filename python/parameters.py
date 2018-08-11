#Numerical parameters for diffeomorphismes

#Convergence tolerance for non-analitic function inversions
epsInvGlob = 1.e-7
#Safety margins for a multitransform to be difffeomorphic
#First value is the minimal influence of one trans considered
#Second is the maximal value for a single transform
#Third is the highest value accepted
diffeoSafeMarginsGlob = [0.01, 0.8, 0.8]
nCheckUnsafeGlob = 5
bMaxGlob = 5.
maxInfGuessGlob = 0.5

cstSpeedZoneGlob = 0.0

costSwitchGlob = 3

#Debugging
dbgGlob = 1
dbgPrintGlob = 5

def dbgPrint(lvl,str):
    if lvl <=dbgPrintGlob:
        print(str)
    return 0
    