# Library for GEOS 626 and GEOS627 courses at UAF
# Contributers: Carl Tape, Yuan Tian*, Amanda McPherson*
import numpy as np
import scipy.linalg as la
import scipy.special as special
import matplotlib.pyplot as plt


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
    return U,S,V


def svdall(G):
    [U, svec, VH] = np.linalg.svd(G) 
    S  = la.diagsvd(svec,*G.shape)
    V  = VH.T
    p  = np.linalg.matrix_rank(G)
    Sp = S[:p,:p]
    Vp = V[:,:p]
    V0 = V[:,p:]
    Up = U[:,:p]
    U0 = U[:,p:]
    Rm = Vp@Vp.T
    Rd = Up@Up.T
    ndata,nparm = G.shape
    print('G is %i x %i, rank(G) = %i' % (ndata,nparm,p))
    return Up,Sp,Vp,U0,V0,Rm,Rd,p


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
    # corresponding unit vector
    xhat = np.array([np.cos(theta),np.sin(theta)]).T
    Cinv = np.linalg.inv(C)
    # preallocate output array
    r = np.zeros((n,2))
    for i in range(n):
        #store each (x,y) pair on the confidence ellipse in the corresponding row of r
        #r(i,:) = sqrt(DELTA2/(xhat(i,:)*Cinv*xhat(i,:)'))*xhat(i,:)
        #r[i,:] = np.dot(np.sqrt(DELTA2/(xhat[i,:]@Cinv@xhat[i,:].T)),xhat[i,:])
        r[i,:] = np.sqrt(DELTA2/(xhat[i,:]@Cinv@xhat[i,:].T)) * xhat[i,:]
    
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


def plotconst_mod(x,l,r,color,lw):
    n = len(x)
    delta = (r-l)/n
    myx = np.array([0])
    myy = np.array([0])
    for i in range(n):
        myx = np.concatenate((myx,np.arange(i*delta+l,(i+1)*delta+l,(delta/20))))
        myy = np.concatenate((myy,np.arange(i*delta+l,(i+1)*delta+l,(delta/20))))
    l2 = len(myx)
    myx = myx[1:l2+1]
    myy = myy[1:l2+1]
    plt.plot(myx,myy,color,lw=lw)

    
def tsvd(g, X, rvec):
#     %TSVD regularization using truncated singular value decomposition
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
# %

    # % size of inputs (n is number of data; p is number of parameters)
    (n, p)      = X.shape
    #print(n,p)
    q           = np.min([n, p])
    nr          = len(rvec)

    #initialize outputs
    f_r         = np.zeros((p, nr))       # set of r models
    rss         = np.zeros((nr, 1))       # RSS for each model
    f_r_ss      = np.zeros((nr, 1))       # norm of each model

    #compute SVD of X
    [U, s, VH]   = np.linalg.svd(X) 
    S=scipy.linalg.diagsvd(s,*X.shape)    # vector of singular values

    # 'Fourier' coefficients (fc) in expansion of solution in terms of right singular vectors
    # note: these are also referred to as Picard ratios
    beta        = U[:, :q].T@g            # note data g
    fc          = beta / s
    V = VH.T
    # treat each truncation parameter separately
    f_r = V[:, :rvec[0]] @ fc[:rvec[0]]
    #print((V[:, :rvec[0]] @ fc[:rvec[0]]).shape)
    for j in range(nr):
        k         = rvec[j]               # current truncation parameter
        if j>0:
            f_r = np.vstack((f_r, V[:, :k] @ fc[:k]))    # truncated SVD estimated model vector
        f_r_ss[j] = np.sum(fc[:k]**2);    # the squared norm of f_r
        rss[j]    = np.sum(beta[k:q]**2)  # residual sum of squares
    f_r = f_r.T
    # in overdetermined case, add rss of least-squares problem
    if (n > p):
        rss = rss + np.sum((g - U[:, :q]@beta)**2)   # note data g
    return f_r, rss, f_r_ss


def fftvec(t):
    # Python version of fftvec.m made by Carl tape
    # FFTVEC provides frequency vector for Matlab's fft convention
    # 
    # This is the bare-bones FFT example for teaching purposes in the sense
    # that there is no padding with zeros, no odd-number time series, no
    # all-positive frequencies, and no use of fftshift.
    #
    # EXAMPLE:
    #   t = np.arange(2,9.0,0.4); n=len(t); f = fftvec(t); ix = 4; 
    #   dt = t[1]-t[0]; df = 1/(2*dt)/(n/2) 
    #   fig=plt.figure();plt.plot(f,'-'); 
    #   fig2=plt.figure(); plt.plot(np.fft.fftshift(f),'-');
    #
    # NOTE: I chose to set fNyq as the negative frequency such that
    #       fftshift(fftvec(t)) will give a vector increasing uniformly from
    #       -fNyq to just less than fNyq, with f=0 at index n/2+1
    #
    # NOTE: If you want to consider all-positive frequencies, then use abs(f)
    # 
    n = len(t)
    if n % 2 == 1:
       print('error(time series must have an even number of points)')

    dt = t[2] - t[1];       # sample rate
    fNyq = 1/(2*dt);        # Nyquist frequency
    
    # first half of frequency vector (note: first entry is f=0)
    f1 = np.transpose(np.linspace(0, float(fNyq), int((n/2)+1)));
    # full frequency vector
    #f = [f1 ; -f1(end-1:-1:2)];        % fNyq > 0
    
    f = np.concatenate([f1[0:int(n/2)] , -f1[:0:-1]])    # fNyq < 0
    return f
    
    # alternatively (for fNyq < 0)
    #df = 1/(n*dt);          % =2*fNyq/n
    #f1 = linspace(-fNyq,fNyq-df,n)'
    #f = [f1(n/2+1:n) ; f1(1:n/2)];

    
def gridvec(xmin,xmax,numx,ymin,ymax):
    """  This function inputs specifications for creating a grid 
         of uniformly spaced points, reshaped into column vectors
         of the x- and y-coordinates.  Note that dx = dy."""
    
    xvec0 = np.linspace(xmin,xmax,numx)
    dx = xvec0[1] - xvec0[0]
    yvec0 = np.arange(ymin, ymax+dx, dx)
    
    X, Y = np.meshgrid(xvec0,yvec0)
    a,b = X.shape
    xvec = np.reshape(X,(a*b,1))
    yvec = np.reshape(Y,(a*b,1))
    numy = len(yvec0)
    return xvec, yvec, numy, X, Y

#def gridvec(xmin,xmax,numx,ymin,ymax,returnXY=False):
#    """  This function inputs specifications for creating a grid 
#         of uniformly spaced points, reshaped into column vectors
#         of the x- and y-coordinates.  Note that dx = dy."""
#    
#    xvec0 = np.linspace(xmin,xmax,numx)
#    dx = xvec0[1] - xvec0[0]
#    yvec0 = np.arange(ymin, ymax+dx, dx)
#    
#    X, Y = np.meshgrid(xvec0,yvec0)
#    a,b = X.shape
#    xvec = np.reshape(X,(a*b,1))
#    yvec = np.reshape(Y,(a*b,1))
#    
#    numy = len(yvec0)
#    
#    if returnXY:
#        return xvec, yvec, numy, X, Y
#    else:
#        return xvec, yvec, numy
    
    
def randomvec(a,b,n):
    # this allows a or b to be the larger of the two
    xmin = np.min([a,b])
    xmax = np.max([a,b])
    x    = (xmax - xmin)*np.random.rand(int(n),1) + xmin
    return x


def plot_histo(hdat,edges,itype=2,make_plot=True):
    #PLOT_HISTO plot a histogram with cyan bars and black boundaries
    #
    # INPUT
    #   hdat        input data to bin
    #   edges       vector defining the edges of the bins (for hdat)
    #   itype       optional: type of histogram (=1,2,3) [default = 2]
    #   make_plot   optional: plot histogram [default = true]
    #hdat = hdat.flatten();
    #barcolor = [1, 1, 1]*0.8;
    
    # bin width (only relevant if bins are the same width)
    dbin   = edges[1] - edges[0]
    hedges = np.append(edges,edges[-1]+dbin)
    Ntotal = len(hdat);
    # key command
    N,b = np.histogram(hdat,hedges);

    if itype ==1:
        Nplot = N; xlab = 'Count'
    if itype ==2: 
        Nplot = np.divide(N,Ntotal); xlab = 'Fraction'
    if itype ==3: 
        Nplot = np.divide(np.divide(N,Ntotal),dbin); xlab = 'PDF'
        #if len(unique(edges)) > 1:
        if np.std(np.diff(edges))/np.mean(np.diff(edges)) > 1e-4:       # ad hoc criterion
            print(np.unique(np.diff(edges)))
            print('PDF is not implemented to allow bins with varying widths')
            
    elif itype!=1 and itype!=2 and itype!=3: 
        print('itype = %i -- it must be 1,2, or 3'%(itype)) 

    if make_plot==True:
        plt.bar(edges, Nplot, width=0.8*dbin, align='edge');
        plt.xlim([min(edges), max(edges)]);
        plt.ylabel('%s (N=%i)'% (xlab,Ntotal))
        
        if len(hdat) != np.sum(N):
            print('(plot_histo): You may want to extend the histogram edges -->');
            print(' there are %i/%i input that are outside the specified range'%
                (len(hdat)-np.sum(N),len(hdat)))
            #disp(sprintf(' the number of input (%i) does not equal the sum of bin counts (%i).',length(hdat),sum(N)));
    
    plt.tight_layout()


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