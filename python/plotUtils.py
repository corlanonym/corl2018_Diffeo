#####Package for plotting 2d examples
from coreUtils import *


import diffeoUtils as du

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.patches import Ellipse

from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D

from copy import deepcopy as dp,deepcopy


from matplotlib.collections import LineCollection

# 2D rotation
def R2(angle, R=None):
    if R is None:
        R = Id(2)
    R[[0,1],[0,1]] = np.cos(angle)
    R[[0,1],[1,0]] = np.sin(angle)
    R[0,1] = -R[0,1]
    return R

#Create a grid
def getGrid(lims=[-1,1,-1,1], n12=[10,100]):
    
    if isinstance( lims, matplotlib.axes.Axes ):
        lims = list(lims.get_xlim())+list(lims.get_ylim())
    
    nPt = 2*n12[0]*n12[1]
    n1,n2 = n12
    
    X = zeros([2, nPt])
    
    xi0 = np.linspace(lims[0], lims[1], n12[0])
    xi1 = np.linspace(lims[0], lims[1], n12[1])
    yi0 = np.linspace(lims[2], lims[3], n12[0])
    yi1 = np.linspace(lims[2], lims[3], n12[1])
    aOnes = ones(n12[1])
    
    for i in range(n1):
        X[0,i*n2:(i+1)*n2] = aOnes*xi0[i]
        X[1,i*n2:(i+1)*n2] = yi1
        X[0, n1*n2+i*n2:n1*n2+(i+1)*n2] = xi1
        X[1, n1*n2+i*n2:n1*n2+(i+1)*n2] = aOnes*yi0[i]
    
    return X


def plotGrid(ax, X, n12=[10,100], c=None):
    if isinstance(X, (list,tuple)):
        for aL in X:
            if isinstance(aL, np.ndarray):
                plotGrid(ax,aL, n12, c)
            elif isinstance(aL, (tuple,list)):
                if len(aL) == 2:
                    plotGrid(ax,aL[0], aL[1], c)
                elif len(aL) == 3:
                    plotGrid(ax,aL[0], aL[1], aL[2])
                else:
                    assert 0
            else:
                assert 0
        return 0
    n1,n2 = n12
    for i in range(n1):
        ax.plot(X[0,i*n2:(i+1)*n2],X[1,i*n2:(i+1)*n2],color=c, linewidth=1)
        ax.plot(X[0,n1*n2+i*n2:n1*n2+(i+1)*n2],X[1,n1*n2+i*n2:n1*n2+(i+1)*n2],color=c, linewidth=1)
    return 0

def myQuiver(ax, X, V, c=None, otherPlotOptDict={}):
    dim = X.shape[0]
    assert dim in (2,3)
    
    XV = X+V
    
    if (c in ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']) or isinstance(c, (list, tuple)):
        cl = lambda x: c
    elif c=='arctan':
        cmap = plt.get_cmap('viridis')
        cl = lambda x: cmap( (np.arctan2(x[0],x[1])+np.pi)/(2.*np.pi) )
    else:
        assert 0
    
    if dim==2:
        XX = np.vstack((X[0,:], XV[0,:]))
        YY = np.vstack((X[1,:],XV[1,:]))
        for k in range(X.shape[1]):
            ax.plot(XX[:,k], YY[:,k], color=cl(V[:,k]), **otherPlotOptDict)
    else:
        XX = np.vstack((X[0,:],XV[0,:]))
        YY = np.vstack((X[1,:],XV[1,:]))
        ZZ = np.vstack((X[2,:],XV[2,:]))
        for k in range(X.shape[1]):
            ax.plot(XX[:,k],YY[:,k],ZZ[:,k],color=cl(V[:,k]), **otherPlotOptDict)
    
    return 0
    
    
    


def plotGridDiffeo( aDiffeo, baseDir = 1, ax = None, c=['b', 'r'], plotBoth=False, n12=[10,100], lims=None ):
    
    if (lims is None) and (ax is None):
        lims = [-1,1,-1,1]
    elif (lims is None):
        lims = list(ax.get_xlim())+list(ax.get_ylim())
    else:
        if len(lims) == 2:
            lims += lims
    
    n1,n2=n12
    
    if ax is None:
        fig,ax = plt.subplots(1,1)
    else:
        fig = None
    
    X = getGrid(lims, n12)
    
    if plotBoth:
        for i in range(n1):
            ax.plot(X[0,i*n2:(i+1)*n2], X[1,i*n2:(i+1)*n2], c[0])
            ax.plot(X[0,n1*n2+i*n2:n1*n2+(i+1)*n2], X[1,n1*n2+i*n2:n1*n2+(i+1)*n2], c[0])
    
    #Apply
    X,_dummy = du.applyDiffeo(X, aDiffeo, baseDir=baseDir, doCopy=False)
    for i in range(n1):
        ax.plot(X[0,i*n2:(i+1)*n2],X[1,i*n2:(i+1)*n2],c[1])
        ax.plot(X[0,n1*n2+i*n2:n1*n2+(i+1)*n2],X[1,n1*n2+i*n2:n1*n2+(i+1)*n2],c[1])
    
    return fig,ax

def errorPlot(xErr, ang=None, fig=None, ax=None, thisLine=None, color=None):
    if (fig is None) and (ax is None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
    
    xErr = xErr.squeeze()
    if xErr.ndim > 1:
        xErr = cNorm(xErr,kd=False)
    
    ang = np.linspace(0,2*pi,xErr.size) if ang is None else ang
    
    if thisLine is not None:
        thisLine.set_xdata(ang)
        thisLine.set_ydata(xErr)
        thisLine.set_color(color)
    else:
        thisLine = ax.plot(ang, xErr, color=color)[0]
    
    return fig, ax, thisLine
    
        
    
    ang = np.arange(0, )
    
def getEllipse(pos,P,alpha):
    v, E = sp.linalg.eigh(P)
    orient = np.arctan2(E[1,0],E[0,0])*180.0/np.pi
    return Ellipse(xy=pos, height=2.0*np.sqrt(alpha) * 1.0/np.sqrt(v[1]), width=2.0*np.sqrt(alpha) * 1.0/np.sqrt(v[0]), angle=orient)

def plotEllipse(ax, pos, P, alpha, color = [0.0,0.0,1.0,1.0], faceAlpha=0.5):
    color=np.array(dp(color)); color[-1]=color[-1]*faceAlpha; color=list(color)
    e = getEllipse(pos, P, alpha)
    ax.add_patch(e)
    e.set_facecolor( color )
    return e
    

def plotTransformation(aTrans, dims=[0,1], aColor=[1.0,1.0,1.0,1.], ax=None, marginSingle=0., marginOverlap=0, distFac = 5.):
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
        xlim = [np.min(aTrans._centers[0,:]), np.max(aTrans._centers[0,:])]
        ylim = [np.min(aTrans._centers[1,:]),np.max(aTrans._centers[1,:])]
    else:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
    
    # Plot centers
    ax.plot(aTrans._centers[dims[0],:], aTrans._centers[dims[1],:], 'o', color=deepcopy(aColor))
    # Plot translation
    myQuiver(ax, aTrans._centers, aTrans._translations, c=deepcopy(aColor))
    # Plot bases
    cLevel = aTrans.dist2nondiffeo(marginSingle=marginSingle, marginOverlap=marginOverlap)*distFac
    cLevel = 1.-np.minimum(np.maximum(cLevel, 0.),.8)
    #patches = []
    for k in aTrans.range():
        aColor[-1] = cLevel[k]
        thisCircle = Circle(aTrans._centers[dims, k], aTrans._bases[k], facecolor=aColor, edgecolor=deepcopy(aColor[:-1]), linestyle='-', linewidth=1.)
        #patches.append(thisCircle)
        ax.add_patch(thisCircle)
    #ax.add_collection(patches)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax

def plotDiffeo(aDiffeo:du.diffeomorphism, source=None, target=None, cMap='jet', dims=[0,1], allFigAx=None, margins=[None,None]):
    
    
    allFigAx = list(map( lambda k: plt.subplots(1,1), range(aDiffeo.nTrans) )) if allFigAx is None else allFigAx

    cMap = plt.get_cmap(cMap)
    allColors = cMap(np.linspace(0,1,aDiffeo.nTrans))
    
    for k,aTrans in enumerate(aDiffeo._transformationList):
        
        fig,ax = allFigAx[k]

        if not target is None:
            ax.plot(target[0,:],target[1,:],'--',color=allColors[-1,:])
        if not source is None:
            ax.plot(source[0,:],source[1,:],'--',color=allColors[0,:])
    
            xTildeK = aDiffeo.forwardTransform(source,kStop=k)
            ax.plot(xTildeK[0,:],xTildeK[1,:],'-',color=allColors[k,:])
    
        plotTransformation(aTrans, dims=dims, aColor=list(allColors[k,:]), ax = ax, marginSingle=aTrans.margins[0] if (margins[0] is None) else margins[0], marginOverlap=aTrans.margins[1] if (margins[1] is None) else margins[1])
    
    return allFigAx


def getColoredLine(data: np.ndarray,time: np.ndarray = None,cmap='viridis',lw=2., ax=None) -> "lineCollection":
    # https://stackoverflow.com/questions/10252412/matplotlib-varying-color-of-line-to-capture-natural-time-parameterization-in -da/10253183
    
    dim,nPts = data.shape
    assert dim in (2,3),"Dim has to be 2 or 3"
    
    if time is None:
        time = np.linspace(0.,1.,nPts)
    else:
        time = time.copy()
        time.squeeze()
        assert len(time) == nPts,'Time not compatible with data'
        time -= np.min(time)
        time /= np.max(time)
    
    # set up a list of (x,y) points
    points = data.transpose().reshape(-1,1,dim)
    
    # set up a list of segments
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    
    # make the collection of segments
    lc = LineCollection(segs,cmap=plt.get_cmap(cmap))
    lc.set_array(time)  # color the segments by our parameter
    lc.set_linewidth(lw)
    
    if ax is not None:
        ax.add_collection(lc)
    
    return lc


def plotLyap(vMax:'List or float', nCirc:int, ax=None, c:'color'='k', origin=[0,0]):
    
    if not isinstance(vMax, (list, tuple)):
        vMax = np.linspace(0.05*vMax, vMax, nCirc)

    if not isinstance(c,(list,tuple)):
        c = nCirc*[c]
    
    
    cList = []
    
    for k,avmax in enumerate(vMax):
        
        thisCirc = Circle(origin, avmax, facecolor='none', edgecolor=c[k])
        
        if ax is not None:
            ax.add_patch(thisCirc)
        
        cList.append(thisCirc)
    
    return cList
    
def plotTransformation(ax, aTrans, opts={}, whichTrans=None):
    
    _opts = {'centers':True, 'bases':True, 'translations':True, 'centerColor':'*r'}
    _opts.update(opts)
    whichTrans = [0,aTrans.nTrans] if whichTrans is None else whichTrans
    # Centers
    if _opts['centers']:
        ax.plot(aTrans._centers[0,range(whichTrans[0], whichTrans[1])],aTrans._centers[1,range(whichTrans[0], whichTrans[1])],_opts['centerColor'])
    # Influence region
    for ii in range(whichTrans[0], whichTrans[1]):
        if _opts['bases']:
            thisCircle = Circle(aTrans._centers[:,ii],aTrans._bases[ii],facecolor='none',edgecolor='m')
            ax.add_patch(thisCircle)
        if _opts['translations']:
            ax.arrow(aTrans._centers[0,ii],aTrans._centers[1,ii],aTrans._translations[0,ii],aTrans._translations[1,ii])
        
    return 0

def plotAxes(ax, orig, rot, l=1., c=['b', 'r', 'g'], otherOpts={}):
    import pyquaternion as pyq
    if isinstance(rot, pyq.Quaternion) or (len(list(rot)) == 4):
        rot = pyq.Quaternion(list(rot)).rotation_matrix
    rot *= l
    
    x = np.hstack((orig,orig+rot[:,[0]]))
    y = np.hstack((orig,orig+rot[:,[1]]))
    z = np.hstack((orig,orig+rot[:,[2]]))
    #rotation matrix -> columns are new base vectors
    ax.plot(*x,color=c[0],**otherOpts)
    ax.plot(*y,color=c[1],**otherOpts)
    ax.plot(*z,color=c[2],**otherOpts)
    
    return 0
    