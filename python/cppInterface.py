from coreUtils import *

import subprocess

from multiprocessing import Pool
from itertools import starmap

# Option
doMult_ = True

taskDict = {'forward':1, 'forwardV':2, 'inverse':3, 'inverseV':4, 'demSpaceVel':5, 'ctrlSpaceVel':6, 'demSpaceTraj':7, 'ctrlSpaceTraj':8, 'dirTraj':9}

resultNames = ['tVec.txt', 'pos.txt', 'vel.txt', 'traj.txt']
whichForm = [TXT2Vector, TXT2Matrix, TXT2Matrix, TXT2Matrix]


def callCPP(what, dims, definitionFolder:str, inputFolder:str, resultFolder:str, cppExe:str, xIn:np.ndarray=None, vIn:np.ndarray=None, tIn:np.ndarray=None, addInfo=[]):
    
    #dim:dimTot,dimInt
    
    from shutil import rmtree
    
    assert (what in taskDict.keys())
    
    if not isinstance(addInfo, list):
        addInfo = list(addInfo)
    
    #Attention deletes first
    rmtree(inputFolder, True)
    
    mkdir_p(inputFolder)
    mkdir_p(resultFolder)
    
    results = 4*[None]
    
    if xIn is not None:
        Array2TXT(os.path.join(inputFolder,'pos.txt'),xIn)
    if vIn is not None:
        Array2TXT(os.path.join(inputFolder,'vel.txt'),vIn)
    if tIn is not None:
        Array2TXT(os.path.join(inputFolder,'t.txt'),tIn)
    try:
        subprocess.call([cppExe,"{0:d}{1:d}{2:d}".format(taskDict[what],dims[0],dims[1]),definitionFolder,inputFolder,resultFolder]+addInfo,shell=False, timeout=20.)
    except:
        return results
        #pass
    
    for k,[aFun, aName] in enumerate(zip(whichForm, resultNames)):
        try:
            results[k]=aFun(os.path.join(resultFolder, aName))
        except:
            pass
    
    return results


# Create worker pool
cppPool = Pool(4)

def batchedCallCPP(what, dims, definitionFolder:str, inputFolder:str, resultFolder:str, cppExe:str, xIn:np.ndarray=[None], vIn:np.ndarray=[None], tIn:np.ndarray=[None], addInfo=[[]]):

        maxLen = max(len(xIn), len(vIn), len(tIn))

        xIn = xIn if len(xIn)==maxLen else maxLen*[None]
        vIn = vIn if len(vIn) == maxLen else maxLen * [None]
        tIn = tIn if len(tIn) == maxLen else maxLen * [None]
        addInfo = addInfo if len(addInfo) == maxLen else maxLen*[[]]

        whatL = maxLen*[what]
        dimsL = maxLen*[dims]

        definitionFolderL = maxLen*[definitionFolder]
        inputFolderL = [os.path.join(inputFolder, "{0}".format(k)) for k in range(maxLen)]
        resultFolderL = [os.path.join(resultFolder, "{0}".format(k)) for k in range(maxLen)]

        if doMult_:
            resultList = cppPool.starmap(callCPP, zip(whatL, dimsL, definitionFolderL, inputFolderL, resultFolderL, maxLen*[cppExe], xIn, vIn, tIn, addInfo))
        else:
            resultList = list(starmap(callCPP, zip(whatL, dimsL, definitionFolderL, inputFolderL, resultFolderL, maxLen*[cppExe], xIn, vIn, tIn, addInfo)))

        return resultList
    
    
    
    
    