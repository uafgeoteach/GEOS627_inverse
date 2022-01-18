# Tools library of GEOS627 inverse course
# Coded by Yuan Tian at UAF 2021.01
import numpy as np
import scipy
import matplotlib.pyplot as plt


def covC(id,parms):
#COVC evaluate covariance function C(d) at an array of distances d
#
# INPUT
#   id      array of distance values (scalar, vector, matrix, etc)
#   parms:
#   icov    type of covariance function (=1 Gaussian; =2 exponential)
#   iL      length scale (same units as id)
#   sigma   amplitude factor
#   nu      OPTIONAL: parameter for Matern covariance (icov=4 only)
#
# OPTIONS FOR SPECIFYING LENGTH SCALE
#   (1) iL and id are indices for a spatial grid
#   (2) iL and id are actual lengths for a spatial grid
#
    nparm=len(parms)
    icov=parms[0]
    iL    = parms[1]
    sigma = parms[2]
    nu=[]
    if nparm==4:
        nu=parms[3]
    LFACTOR=2
    if icov==1:
        # Gaussian covariance
        # --> The factor of 2 in (2*iL^2) leads to smoother models
        iL = iL / LFACTOR
        Cd = sigma**2 * np.exp(-id**2 / (2*iL**2) )
    if icov==2:
        # exponential covariance
        iL = iL / LFACTOR;
        Cd = sigma**2 * np.exp(-id / iL )
    if icov== 3:
        # circular covariance
        # here iL represents the diameter of the two intersecting discs
        icirc = (id <= iL)
        Cd =np.zeros_like(id) 
        beta = 2*np.arcsin(id[icirc] / iL)
        Cd[icirc] = sigma**2 * (1 - (beta + np.sin(beta))/np.pi )
    if icov==4:
        # Matern covariance
        # http://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        # note this uses the built-in functions gamma and besselk
        iL = iL / LFACTOR
        b = scipy.special.kv(nu, np.sqrt(2*nu)*id/iL)
        Cd = sigma**2 * (1/(scipy.special.gamma(nu)*2**(nu-1))) * (np.sqrt(2*nu)*id/iL)**nu * b


    return Cd




def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def plotconst_mod(x,l,r,color,lw):
    n=len(x)
    delta =(r-l)/n
    myx=np.array([0])
    myy=np.array([0])
    for i in range(n):
        myx=np.concatenate((myx,np.arange(i*delta+l,(i+1)*delta+l,(delta/20))))
        myy=np.concatenate((myy,np.arange(i*delta+l,(i+1)*delta+l,(delta/20))))
    l2=len(myx)
    myx=myx[1:l2+1]
    myy=myy[1:l2+1]
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
    f_r         = np.zeros((p, nr))            # set of r models
    rss         = np.zeros((nr, 1))            # RSS for each model
    f_r_ss      = np.zeros((nr, 1))             # norm of each model

    #compute SVD of X
    [U, s, VH]   = np.linalg.svd(X) 
    S=scipy.linalg.diagsvd(s,*X.shape)                  # vector of singular values

    # 'Fourier' coefficients (fc) in expansion of solution in terms of right singular vectors
    # note: these are also referred to as Picard ratios
    beta        = U[:, :q].T@g            # note data g
    fc          = beta / s
    V=VH.T
    # treat each truncation parameter separately
    f_r=V[:, :rvec[0]] @ fc[:rvec[0]]
    #print((V[:, :rvec[0]] @ fc[:rvec[0]]).shape)
    for j in range(nr):
        k         = rvec[j]               # current truncation parameter
        if j>0:
            f_r =np.vstack((f_r, V[:, :k] @ fc[:k]))    # truncated SVD estimated model vector
        f_r_ss[j] = np.sum(fc[:k]**2);        # the squared norm of f_r
        rss[j]    = np.sum(beta[k:q]**2)    # residual sum of squares
    f_r=f_r.T
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

    #==========================================================================
    



def randomvec(xmin0,xmax0,n):
    xmin=np.min([xmin0,xmax0])
    xmax = np.max([xmin0,xmax0])
    x = (xmax - xmin)*np.random.rand(int(n),1) + xmin
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
    dbin = edges[1] - edges[0]
    hedges=np.append(edges,edges[-1]+5)
    Ntotal = len(hdat);
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
        plt.bar(edges,Nplot, width=0.8*dbin);
        plt.xlim([min(edges), max(edges)]);
        plt.ylabel('%s (N=%i)'% (xlab,Ntotal))
        
        if len(hdat) != np.sum(N):
            print('(plot_histo.m): You may want to extend the histogram edges -->');
            print(' there are %i/%i input that are outside the specified range'%
                (len(hdat)-np.sum(N),len(hdat)))
            #disp(sprintf(' the number of input (%i) does not equal the sum of bin counts (%i).',length(hdat),sum(N)));
    
    plt.tight_layout()

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

    #==========================================================================
    
