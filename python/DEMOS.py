if __name__ == '__main__':
    import os
    from ctrlSpaceDyn import simpleNamedSearch
    import plotUtils as pu
    
    all = ['doubleSharpC','heeWorm', 'Leaf_2','GShape','BendedLine','DoubleBendedLine','Leaf_1','Sharpc','Snake','NShape']
    succ = []
    fail = []
    for aName in all:
        try:
            simpleNamedSearch(aName)
            succ.append(aName)
        except:
            print('FAIL '+aName)
            fail.append(aName)
            pass
    print("Success")
    print(succ)
    print("Fail")
    print(fail)
    pu.plt.show()