from projectionUtils import *
from plotUtils import *

def getPointsOnNSphere(dim=2, Ne=10):
    
    N = Ne**dim
    x = zeros((dim, N))
    if dim == 2:
        ang = np.linspace(0,2*pi,N,endpoint=False)
        x[0,:] = cos(ang)
        x[1,:] = sin(ang)
    elif dim == 3:
        allR = linspace(0, 1, Ne, endpoint=True)
        allx = zeros((1,))
        ally = zeros((1,))
        allAng = np.linspace(0,2*pi,Ne*Ne,endpoint=False)
        allSin = sin(allAng)
        allCos = cos(allAng)
        for k in range(Ne):
            allx = np.hstack((allx,allCos*allR[k]))
            ally = np.hstack((ally,allSin*allR[k]))
        allz = sqrt(1-square(allx)-square(ally))
        x = np.hstack(( np.vstack((allx, ally, allz)), np.vstack((allx, ally, -allz)), np.vstack((allz, ally, allx)), np.vstack((-allz, ally, allx)), np.vstack((allx, allz, ally)), np.vstack((allx, -allz, ally)) ))
    else:
        x = np.random.rand(dim,N)
        np.divide(x, cNorm(x), x)

    return x

def plotStereoProj(x, n, alpha=1, alphaIsRel=True, beta=1, dir=1, v=None):

    dim, nPt = x.shape
    assert (0<=n) and n<dim, 'wrong projection dimension'
    assert dim in [2,3], 'Can only plot 2 or 3 dimensional data'
    if (v is None):
        y = stereographicProj(x, n, alpha, alphaIsRel=alphaIsRel, beta=beta, dir=dir)
    else:
        y,vy = stereographicProj(x, n, alpha, alphaIsRel=alphaIsRel, beta=beta, dir=dir, v=v)

    if alphaIsRel:
        alpha = nMult(alpha, cNorm(x,False))
    else:
        if not (isinstance(alpha, np.ndarray) and (alpha.size == nPt)):
            alpha = ones(nPt,)*alpha

    if dim == 2:
        fig, ax = plt.subplots(1,1)
        ax.plot(x[0,:], x[1,:], '.b')
        if not(v is None):
            ax.quiver(x[0,:], x[1,:], v[0,:], v[1,:],color='b')
        if n == 0:
            ax.plot(dir*alpha,y[0,:],'.r')
            if not (v is None):
                ax.quiver(dir*alpha,y[0,:], zeros(nPt), vy[0,:], color='r')
        else:
            ax.plot(y[0,:],dir*alpha,'.r')
            if not (v is None):
                ax.quiver(y[0,:],dir*alpha, vy[0,:], zeros(nPt), color='r')
        ax.axis('equal')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.plot(x[0,:], x[1,:], x[2,:], '.b')
        xProj = zeros(x.shape)
        xProj[arange(dim)!=n,:] = y
        xProj[n,:] = dir*alpha
        ax.plot(xProj[0,:],xProj[1,:],xProj[2,:],'.r')
        if not(v is None):
            vProj = zeros(v.shape)
            vProj[arange(dim) != n,:] = vy
            vProj[n,:] = 0
            ax.quiver(xProj[0,:],xProj[1,:],xProj[2,:],vProj[0,:],vProj[1,:],vProj[2,:],color='r')

    return fig, ax




def plotStereoProjOfTraj(x=None, r=None, alpha=1, alphaIsRel=True, beta=1, v=None):
    
    #accordType: One to align with r, 2 to align with alpha, 3 enumerate

    if r is None:
        r = cNorm(x,False)

    #Determine all necessary projections and the id's of the corresponding points
    dim,nPt = x.shape

    allFig = []
    allAx = []

    for thisDim in range(dim):
        xTop = x[:,x[thisDim]>=0]
        xBottom = x[:,x[thisDim] <= 0]
        if not(v is None):
            vTop = v[:,x[thisDim] >= 0]
            vBottom = v[:,x[thisDim] <= 0]
        else:
            vTop=vBottom=None

        if xTop.size > 0:
            thisFig, thisAx = plotStereoProj(xTop, n=thisDim, alpha=alpha, alphaIsRel=alphaIsRel, beta=beta, dir=1,v=vTop)
            allFig.append(thisFig)
            allAx.append(thisAx)
        if xBottom.size > 0:
            thisFig,thisAx = plotStereoProj(xBottom,n=thisDim,alpha=alpha,alphaIsRel=alphaIsRel,beta=beta,dir=-1,v=vBottom)
            allFig.append(thisFig)
            allAx.append(thisAx)

    return allFig, allAx



    
    
    
    
    
    
    
    # signListL = [[1],[-1]]
    # for k in range(dim-1):
    #     nL = len(signListL):
    #     for l in range(nL):
    #         signListL += [signListL[l].append(1), signListL[l].append(-1)]
    # signList = lmap(lambda aL: np.sign(aL).reshape((dim,1)))
    # #Check if any points on this chart
    # xSign = np.sign(x)
    # pointsOnCharts = []
    # for aSL in signList:
    #     thisInd = np.all( xSign==aSL, 0 )
    #     if thisInd.size:
    #         pointsOnCharts.append({'signs':aSL.copy(), 'points':x[:,thisInd], 'vel':v[:,thisInd] if not(v is None) else None })

    
    
    
    
    
    
    
    
    
    
    
