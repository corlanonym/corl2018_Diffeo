from coreUtils import *

def folderPickel2Format(folder:str, spec:dict={}, outFolder:str=None):

    import os
    import sys
    import pickle
    
    _opts = {'format':'png'}
    _opts.update(spec)
    
    outFolder = folder if outFolder is None else outFolder
    
    mkdir_p(outFolder)
    
    for file in os.listdir(folder):
        try:
            with open(os.path.join(folder, file), 'rb') as fileRead:
                fig = pickle.load(fileRead)
                fig.savefig( os.path.join( outFolder, os.path.splitext(os.path.basename(file))[0]+_opts['format'] ), **_opts )
            print("Converted {0}".format(file))
        except:
            print("Unexpected error: {0}".format(sys.exc_info()[0]))
            print("Failed converting {0}".format(file))
    
    return 0

if __name__ == '__main__':
    
    name = ''
    spec={}
    name2 = None
    print('...')
    folderPickel2Format(name, spec)
    