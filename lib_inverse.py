import numpy as np
import matplotlib.pyplot as plt

import scipy.linalg as la
import scipy.special as special
from scipy import stats

def dispGik(G, ii, kk):
    nx, ny = G.shape
    if ii < 1 or ii > nx or kk < 1 or kk > ny:
        print(f"Error: Indices ({ii},{kk}) are out of bounds for a {nx}x{ny} matrix.")
    else:
        print(f"G({ii}, {kk}) = {G[ii-1,kk-1]}")    

def rot2d(xdeg):
    # return 2D rotation matrix
    cosx = np.cos(np.deg2rad(xdeg))
    sinx = np.sin(np.deg2rad(xdeg))
    R = np.array([[cosx,-sinx],[sinx,cosx]])
    return R


def svdmat(G):
    # equivalent of matlab command [U,S,V] = svd(G)
    [U,s,VH] = la.svd(G)
    S = la.diagsvd(s,*G.shape)
    V = VH.T
    #svec = np.reshape(s,(len(s),1))
    return U,S,V


def svdall(G):
    [U,S,V] = svdmat(G)
    p  = np.linalg.matrix_rank(G)
    Sp = S[:p,:p]   # square matrix of positive singular values
    Vp = V[:,:p]    # compact version of orthogonal model space basis vectors
    Up = U[:,:p]    # compact version of orthogonal data space basis vectors
    V0 = V[:,p:]    # model null space (could be empty)
    U0 = U[:,p:]    # data null space (could be empty)
    Rm = Vp@Vp.T    # model resolution matrix
    Rd = Up@Up.T    # data resolution matrix

    sarray = np.diag(Sp)  # array of singular values

    # generalized inverse (see also la.pinv)
    Gdagger = Vp @ la.inv(Sp) @ Up.T

    ndata,nparm = G.shape
    print('G is %i x %i, rank(G) = %i' % (ndata,nparm,p))

    return Up,Sp,Vp,U0,V0,Rm,Rd,Gdagger,p,sarray


def covC(id,parms):
#COVC evaluate covariance function C(d) at an array of distances d
#
# INPUT
#   id      array of distance values (scalar, vector, matrix, etc)
#   parms:
#   icov    type of covariance function (=1 Gaussian; =2 exponential)
#   iLprime length scale (same units as id)
#   sigma   amplitude factor
#   nu      OPTIONAL: parameter for Matern covariance (icov=4 only)
#
# OPTIONS FOR SPECIFYING LENGTH SCALE
#   (1) iLprime and id are indices for a spatial grid
#   (2) iLprime and id are actual lengths for a spatial grid
#
# See examples in covC_plot.ipynb
#
    nparm   = len(parms)
    icov    = parms[0]
    iLprime = parms[1]
    sigma   = parms[2]
    nu      = []
    if nparm==4:
        nu = parms[3]
    
    # see Appendix A of hw_cov.pdf
    LFACTOR = 2
    #LFACTOR = 1
    iL = iLprime / LFACTOR
    
    if icov==1:
        # Gaussian covariance (Tarantola, 2005, Eq. 5.28)
        # --> The factor of 2 in (2*iL^2) leads to smoother models
        Cd = sigma**2 * np.exp(-id**2 / (2*iL**2) )
    if icov==2:
        # exponential covariance (Tarantola, 2005, Eq. 5.27)
        Cd = sigma**2 * np.exp(-id / iL )
    if icov== 3:
        # circular covariance (Tarantola, 2005, Eq. 5.29)
        # here iL represents the diameter of the two intersecting discs
        iL = iLprime
        icirc     = (id <= iL)
        Cd        = np.zeros_like(id) 
        beta      = 2*np.arcsin(id[icirc] / iL)
        Cd[icirc] = sigma**2 * (1 - (beta + np.sin(beta))/np.pi )
    if icov==4:
        # Matern covariance
        # http://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        # note this uses the built-in functions gamma and besselk
        b  = special.kv(nu, np.sqrt(2*nu)*id/iL)
        Cd = sigma**2 * (1/(special.gamma(nu)*2**(nu-1))) * (np.sqrt(2*nu)*id/iL)**nu * b

    return Cd


def plot_matrix(M,gridlines=False,colormap=None):
    
    plt.imshow(M,cmap=colormap)
    plt.xticks(ticks=range(np.shape(M)[1]),labels=[str(val) for val in range(1,np.shape(M)[1]+1)])
    plt.yticks(ticks=range(np.shape(M)[0]),labels=[str(val) for val in range(1,np.shape(M)[0]+1)])
    if gridlines:
        xgrid = np.array(range(np.shape(M)[1] + 1)) - 0.5
        ygrid = np.array(range(np.shape(M)[0] + 1)) - 0.5
        for gridline in xgrid:
            plt.axvline(x=gridline,color='k',linewidth=1)
        for gridline in ygrid:
            plt.axhline(y=gridline,color='k',linewidth=1)


def plot_ellipse(DELTA2,C,m):
    # DELTA2 controls the size of the ellipse (see chi2inv in lib_peip.py)
    # C      2 x 2 input covariance matrix
    # m      2 x 1 (xs,ys) defining the center of the ellipse

    # construct a vector of n equally-spaced angles from (0,2*pi)
    n = 1000
    theta = np.linspace(0,2*np.pi,n).T
    # points defining unit circle
    xhat = np.array([np.cos(theta),np.sin(theta)]).T
    Cinv = np.linalg.inv(C)

    r = np.zeros((n,2))
    for i in range(n):
        #store each (x,y) pair on the confidence ellipse in the corresponding row of r
        #r(i,:) = sqrt(DELTA2/(xhat(i,:)*Cinv*xhat(i,:)'))*xhat(i,:)
        #r[i,:] = np.dot(np.sqrt(DELTA2/(xhat[i,:]@Cinv@xhat[i,:].T)),xhat[i,:])
        rlen   = np.sqrt(DELTA2 / (xhat[i,:] @ Cinv @ xhat[i,:].T))
        r[i,:] = rlen * xhat[i,:]
    
    # shift ellipse based on centerpoint m = (xs,ys)
    x = m[0] + r[:,0]
    y = m[1] + r[:,1]
    #plt.plot(x,y)
    
    return x,y

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

    
def tsvd(g, X, rvec):
# %TSVD regularization using truncated singular value decomposition
# %
# % INPUT
# %   g       n x 1 data vector
# %   X       n x p design matrix
# %   rvec    r x 1 vector of truncation parameters (integers between 1 and p)
# %
# % OUTPUT
# %   f_r     p x r matrix of TSVD model vectors
# %   rss     r x 1 vector of residual sum of squares
# %   f_r_ss  r x 1 vector of norm-squared of each f_r vector
# %
# % Given a vector g, a design matrix X, and a truncation parameter r, 
# %       [f_r, rss, f_r_ss] = tsvd(g, X, r) 
# % this returns the truncated SVD estimate of the vector f in the linear
# % regression model
# %        g = X*f + noise
# % 
# % If r is a vector of truncation parameters, then the ith column
# % f_r(:,i) is the truncated SVD estimate for the truncation
# % parameter r(i); the ith elements of rss and f_r_ss are the
# % associated residual sum of squares and estimate sum of squares.
# % 
# % Adapted from TSVD routine in Per Christian Hansen's Regularization Toolbox. 

    # allow rvec and g to be vectors
    rvec   = np.squeeze(rvec)
    g      = np.squeeze(g)

    # % size of inputs (n is number of data; p is number of parameters)
    (n,p)  = X.shape
    #print(n,p)
    q      = np.min([n, p])
    nr     = len(rvec)

    #initialize outputs
    f_r    = np.zeros((p,nr))       # set of r models
    rss    = np.zeros((nr,1))       # RSS for each model
    f_r_ss = np.zeros((nr,1))       # norm of each model

    #compute SVD of X
    [U,s,VH] = la.svd(X)
    V = VH.T
    S = la.diagsvd(s,*X.shape)

    # 'Fourier' coefficients (fc) in expansion of solution in terms of right singular vectors
    # note: these are also referred to as Picard ratios
    beta   = U[:,:q].T@g            # note data g
    fc     = beta / s

    # treat each truncation parameter separately
    f_r = V[:, :rvec[0]] @ fc[:rvec[0]]
    #print((V[:, :rvec[0]] @ fc[:rvec[0]]).shape)
    for j in range(nr):
        k = rvec[j]   # current truncation parameter
        if j > 0:
            f_r = np.vstack((f_r, V[:,:k] @ fc[:k])) # truncated SVD estimated model vector
        f_r_ss[j] = np.sum(fc[:k]**2)                # the squared norm of f_r
        rss[j]    = np.sum(beta[k:q]**2)             # residual sum of squares
    f_r = f_r.T

    # in overdetermined case, add rss of least-squares problem
    if (n > p):
        rss = rss + np.sum((g - U[:,:q]@beta)**2)   # note data g
    return f_r, rss, f_r_ss
    
    
def randomvec(a,b,n):
    # this allows a or b to be the larger of the two
    xmin = np.min([a,b])
    xmax = np.max([a,b])
    x    = (xmax - xmin)*np.random.rand(int(n),1) + xmin
    return x


def vm_F_chi(chi,F0,icprior,u,v):
    # VM_F_CHI variable metric algorithm without matrices

    # This function is based on Tarantola (2005), Section 6.22, Eq. 6.347.
    # It computes the operation of F on an arbitrary vector chi by storing a
    # set of vectors (u_k) and scalars (v_k).

    # EVERYTHING HERE IS ASSUMED TO BE IN THE NONHAT-NOTATION.

    # INPUT:
    #    chi     nparm x 1 vector
    #    F0      nparm x nparm initial preconditioner
    #    icprior nparm x nparm prior covariance matrix
    #    u       nparm x niter matrix of nparm x 1 vectors
    #    v       niter x 1 vector of stored values

    # OUTPUT:
    #    Fchi   nparm x 1 vector of F*chi

    # CARL TAPE, 05-June-2007
    # Amanda McPherson, 15 Feb 2022

    _,niter = u.shape

    # compute F*chi
    Fchi = F0@chi
    for jj in range(niter):
        vtmp = chi.T@icprior@u[:,jj]
        Fchi = Fchi + (vtmp/v[jj] * u[:,jj].reshape((len(u[:,jj]),1)))
        
    return Fchi


def vm_F(F0,icprior,u,v):
    # VM_F construct F by repeatedly calling vm_F_chi

    # See vm_F_chi for details.

    # INPUT:
    #    F0      nparm x nparm initial preconditioner
    #    icprior nparm x nparm prior covariance matrix
    #    u       nparm x niter matrix of nparm x 1 vectors
    #    v       niter x 1 vector of stored values
    
    # OUTPUT:
    #    F      nparm x nparm preconditioner that estimates the curvature H^(-1)
    #
    # calls vm_F_chi.m
    #
    # Carl Tape, 05-June-2007
    # Amanda McPherson, 15 Feb 2022
    
    _,nparm = F0.shape

    F = np.zeros((nparm,nparm))
    for ii in range(nparm):
        chi = np.zeros((nparm,1))
        chi[ii] = 1
        Fchi = vm_F_chi(chi,F0,icprior,u,v)
        F[:,ii] = Fchi.flatten()

    return F

# Python's np.corrcoef does not give p values and confidence intervals
# The function below does, but needs to be fed 2 1D arrays (doesn't work for matrix input)
# Source: https://zhiyzuo.github.io/Pearson-Correlation-CI-in-Python/
def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''
    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi
