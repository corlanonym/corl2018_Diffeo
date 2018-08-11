from coreUtils import *

def combineDemos(folder:str, nameList:List[str], resultName:str, tfPos:List['lambdaFun'], tfVel:List['lambdaFun']):
    
    mkdir_p(os.path.join(folder,resultName))
    
    kk=0
    for aName, afPos, afVel in zip(nameList, tfPos, tfVel):
        k=0
        while True:
            try:
                np.savetxt(os.path.join(folder,resultName,"t_{0}.txt".format(kk+1)), np.loadtxt(os.path.join(folder,aName,"t_{0}.txt".format(k+1))))
                np.savetxt(os.path.join(folder,resultName,"pos_{0}.txt".format(kk+1)), afPos(np.loadtxt(os.path.join(folder,aName,"pos_{0}.txt".format(k+1)))))
                np.savetxt(os.path.join(folder,resultName,"vel_{0}.txt".format(kk+1)), afVel(np.loadtxt(os.path.join(folder,aName,"vel_{0}.txt".format(k+1)))))
                k+=1
                kk+=1
            except OSError:
                break
                

if __name__ =="__main__":

    folder = "./data"
    names = 2*["Sharpc"]
    outName = "doubleSharpC"
    tfPos = [ lambda x:x, lambda x:1.5*x]
    tfVel = [lambda v:v,lambda v:1.5*v]
    combineDemos(folder, names, outName, tfPos, tfVel)
    
    names = ["heee", "Worm"]
    outName = "heeWorm"
    tfPos = [lambda x:x,lambda x:x*np.array([[-1.],[2.5]])]
    tfVel = [lambda v:v,lambda v:v*np.array([[-1.],[2.5]])]
    combineDemos(folder,names,outName,tfPos,tfVel)