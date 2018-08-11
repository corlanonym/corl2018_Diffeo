mainDoParallel_ = True
mainUseGPU_ = False
countCPU_ = None
if mainDoParallel_:
    from multiprocessing import  cpu_count
    countCPU_ = cpu_count()

whichKernel_ = 3

ensureIdentityOrigin_ = False

experimentalInfluence_ = True
checkHalfSpaceIntersect_ = True

minConvAngle_ = 0.001

locallyConvDirecKernel_ = 0

mainParsDict_ = {'convergeToZero_':True, 'directionCostCoeff_':1.1, 'maxBaseVal_':None, 'errorCoeff_':2}