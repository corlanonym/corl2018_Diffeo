from coreUtils import *
from parameters import *

if costSwitchGlob == 1:
    circleNormGlob = lambda x: sum(square(x))
elif costSwitchGlob == 2:
    # Cost function resembling more the l2 (in the matrix sense) norm minimization
    circleNormGlob = lambda x:sp.linalg.norm(x)
elif costSwitchGlob == 3:
    # Yet another cost
    circleNormGlob = lambda x:sum(abs(x))
else:
    assert 0,'Undefined cost'