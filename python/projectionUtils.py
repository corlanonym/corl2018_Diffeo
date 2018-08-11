from coreUtils import *

def stereographicProj(x, n, alpha=1., alphaIsRel = True, beta=1., dir = +1, v=None, vIsTangential=False):
    """Performs stereographic projection of the data x (points as columns) for the n-th coordinate onto the hyperplane
    x[n] == alpha;
    if v is given
    If dir = +1 then the positive polar is the projection source else the negative pole is used
    """
    
    dim,nPt = x.shape
    assert (0<=n) and (n<dim), 'Invalid dimension'
    assert dir in [-1,1], 'Direction has to be [-1,1]'
    assert (v is None) or (np.all(x.shape == v.shape)), 'Dimension of positions and velocities does not match'

    if dir == +1:
        xn = x[n,:]
    else:
        xn = -x[n,:]

    r = cNorm(x,False)

    if alphaIsRel:
        alphaScaled = nMult(alpha, r)
    else:
        alphaScaled = alpha

    betaScaled = nMult(beta, r)
    fac = nDivide((alphaScaled+betaScaled), (xn+betaScaled))
    
    xtilde = np.delete(x, n, 0)
    y = nMult(xtilde, fac)
    
    if v is None:
        return y

    #Play safe
    v = v.copy()
    if not vIsTangential:
        vn = cNorm(v,False)
        ind = vn>1e-5

        xdotv = np.divide(sum(nMult(x[:,ind], v[:,ind]),0), r[ind], vn[ind])
        v[:,ind] = v[:,ind]-nMult(xdotv,nDivide(x[:,ind],r[ind]))#Here a implicit copy is done to preserve orinal data integrity
        #Now v lies in the tangentbundle
    #Do the jacoabians
    #The jacobians
    vn = v[n,:]
    vtilde = np.delete(v,n,0)
    vy = nMult(vtilde, fac) - nMult(xtilde, nDivide(fac, (xn+betaScaled)), vn)
    
    return y, vy
    
    
    
    
    
    
    
