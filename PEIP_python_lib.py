# Python version for the PEIP book ex and lap lib
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

    
def phi(x):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # z=phi(x)
    #
    # Calculates the normal distribution and returns the value of the
    # integral
    #
    #       z=int((1/sqrt(2*pi))*exp(-t^2/2),t=-infinity..x)
    #
    # Input Parameters:
    #   x - endpoint of integration (scalar)
    3
    # Output Parameters:
    #   z - value of integral
    #Python version coded by Yuan Tian @UAF 2021
    if (x >= 0):
        z = 0.5 + 0.5 * scipy.special.erf(x/np.sqrt(2))
    else:
        z = 1 - phi(-x)
    return z
def chi2cdf(x,m):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # p=chi2cdf(x,m)
    #
    # Computes the Chi^2 CDF, using a transformation to N(0,1) on page
    # 333 of Thistead, Elements of Statistical Computing.
    #
    # Input Parameters:
    #   x - end value of chi^2 pdf to integrate to. (scalar)
    #   m - degrees of freedom (scalar)
    #
    # Output Parameters:
    #   p - probability that Chi^2 random variable is less than or
    #       equal to x (scalar).
    #
    # Note that x and m must be scalars.

    #Python version coded by Yuan Tian @UAF 2021
    if x==(m-1):
        p=0.5
    else:
        z = (x - m + 2 / 3 - 0.08 / m) * np.sqrt((m - 1)*np.log((m - 1)/x)+x-(m - 1)) / np.abs(x-m+1);
        p = phi(z)
    return p
def chi2inv(p,nu):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # x=chi2inv(p,nu)
    #
    # Computes the inverse Chi^2 distribution corresponding to a given
    # probability that a Chi^2 random variable with the given degrees
    # of freedom is less than or equal to x.  Uses chi2cdf.m.
    #
    # Input Parameters:
    #   p - probability that Chi^2 random variable is less than or
    #       equal to x (scalar).
    #   nu - degrees of freedom (scalar)
    #
    # Output Parameters:
    #   x - corresponding value of x for given probability.
    #
    # Note that x and m must be scalars.
    # Special cases.
    #Python version coded by Yuan Tian @UAF 2021
    if (p >= 1.0):
        return np.inf
    elif (p == 0.0):
        return 0
    elif (p < 0):
        return -np.inf

    # find a window with a cdf containing p
    l = 0.0
    r = 1.0
    
    while (chi2cdf(r, nu) < p):
        l = r
        r = r * 2
    # do a binary search until we have a sufficiently small interval around x
    while (((r - l) / r) > 1.0e-5):
        m = (l + r) / 2
        if (chi2cdf(m, nu) > p):
            r = m
        else:
            l = m
    return (l+r)/2

def xy2distance_row1(nx,ny):
    ix0 = np.arange(nx)
    iy0 = np.arange(ny)

    # generate integer ix and iy vectors
    [iX,iY] = np.meshgrid(ix0,iy0)
    ix = iX.flatten(order='F')
    iy = iY.flatten(order='F')

    n = nx*ny
    xref = ix[0]
    yref = iy[0]
    iD2row1 = (xref-ix)**2 + (yref-iy)**2
    return np.sqrt(iD2row1),ix0,iy0

def xy2distance(nx,ny):
    # integer index vectors
    # NOTE: these start with 0 for convenience in the FFT algorithm
    ix0 = np.arange(nx)
    iy0 = np.arange(ny)

    # generate integer ix and iy vectors
    [iX,iY] = np.meshgrid(ix0,iy0)
    ix = iX.T.flatten(order='F')
    iy = iY.T.flatten(order='F')
    MX,MY=np.meshgrid(ix,iy)
    iD=np.sqrt((MX-MX.T)**2+(MY-MY.T)**2)

    return iD,ix0,iy0

def k_of_x(x):
    N      = np.max(x.shape)
    dx     = x[1]-x[0]
    dk     = (2*np.pi)/(N*dx)
    inull  = N/2
    k      = dk*(np.linspace(1,N,N)-inull)
    return k

def x_of_k(k):
    N      = np.max(k.shape)
    dk     = k[1]-k[0]
    dx     = (2*np.pi)/(N*dk)
    x      = dx*(np.linspace(1,N,N)-1)
    return x


def mhfft2(x,y,f):
    # 2D Fast Fourier Transform of (x,y,f) into (k,l,ft).  The length of 
    # x,y  and f must be an even number, preferably a power of two.  The index of
    # the zero mode is inull=jnull=N/2.

    # Everything is assumed to have been generated by meshgrid, so that
    # f is indexed f(y,x)

    Nx         = np.max(x.shape)
    Ny         = np.max(y.shape)
    k          = k_of_x(x)
    l          = k_of_x(y)
    Periodx    = Nx*(x[1]-x[0])
    Periody    = Ny*(y[1]-y[0])
    inull      = Nx/2
    jnull      = Ny/2
    ft         = (Periodx/Nx)*(Periody/Ny)*np.roll(np.roll(np.fft.fft2(f),int(jnull-1),axis=0),int(inull-1),axis=1)
    return k,l,ft
def grf2(k,m,C,n,*argv):
    Nx         = np.max(k.shape)
    Ny         = np.max(m.shape)
    dk=k[1]-k[0]
    dm=m[1]-m[0]
    Periodx    =2*np.pi/dk
    Periody    =2*np.pi/dm
    Cmtx=np.repeat(C[:, :, np.newaxis], n, axis=2)
    
    # if nargin==6:
    #     disp('grf2mod.m: using the provided Gaussian random vectors (A and B)')
    # else:
    if argv:
        A=argv[0]
        B=argv[1]
    else:
        A = np.random.randn(Ny,Nx,n)   # N(0,1) random variables
        B = np.random.randn(Ny,Nx,n); 
    phi=np.sqrt(Periodx*Periody*Cmtx/2)*(A+B*1j)
    phi[np.isnan(phi)]=0
    return phi, A,B
def mhifft2(k,l,ft,rflag):
    # 2D Fast Fourier Transform of (x,y,f) into (k,l,ft).  The length of 
    # x,y  and f must be an even number, preferably a power of two.  The index of
    # the zero mode is inull=jnull=N/2.

    # Everything is assumed to have been generated by meshgrid, so that
    # f is indexed f(y,x)

    Nx         = np.max(k.shape)
    Ny         = np.max(l.shape)
    x          = x_of_k(k)
    y          = x_of_k(l)
    Periodx    = Nx*(x[1]-x[0])
    Periody    = Ny*(y[1]-y[0])
    inull      = Nx/2
    jnull      = Ny/2
    f         = (Nx/Periodx)*(Ny/Periody)*np.fft.ifftn(np.roll(np.roll(ft,-int(jnull-1),axis=0),-int(inull-1),axis=1),axes=(0,1))
    if rflag==1:
        f=np.real(f)
    return x,y,f


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def shaw(n):
    # Initialization.
    h = np.pi / n
    A = np.zeros((n, n))

    # Compute the matrix A.
    co = np.cos(-np.pi/2+np.arange(0.5,n+0.5)*h)
    psi = np.pi * np.sin(-np.pi/2+np.arange(0.5,n+0.5)*h)
    for i in range(int(n/2)):
        for j in range(i,n-i):
            ss = psi[i] + psi[j];
            A[i, j] = ((co[i] + co[j]) * np.sin(ss) / ss)**2;
            A[n-j-1, n-i-1] = A[i, j]

        A[i, n-i-1] = (2 * co[i])**2

    A = A + np.triu(A, 1).T
    A = A * h

    # Compute the vectors x and b.
    a1 = 2
    c1 = 6
    t1 = 0.8
    a2 = 1
    c2 = 2
    t2 = -0.5
    
    x = a1 * np.exp(-c1*(-np.pi / 2 + np.arange(0.5,n+0.5) * h - t1)**2)+ a2 * np.exp(-c2*(-np.pi / 2 + np.arange(0.5,n+0.5) * h - t2)**2)
    b = A * x
    return A,b,x


def ex_getG(iex,*argv):
# %EX_GETG return G for several examples in Aster
# %
# % EXAMPLES:
# %    G = ex_getG(1);           % Example 1.12
# %    G = ex_getG(2,100,100);   % Exercise 1-3-a
# %    G = ex_getG(2,4,4);       % Exercise 1-3-e
# %    G = ex_getG(2,4,20);      % Example 4.4
# %    G = ex_getG(2,20,4);
# %    G = ex_getG(3,210,210);   % Example 3.2
# %    G = ex_getG(4,20,20);     % Example 1.6, 3.3
# %    G = ex_getG(4,100,100);   % Example 1.6, 3.3
# %
# % all default examples at once:
# %    for iex=1:4, G = ex_getG[iex); end
# %
# % This function is featured in lab_svd.pdf
# %
    
    bfigure=True
    if iex==1:
        stlab = 'tomography ray tracing (Ex 1.12)'
        print('Aster Example 1.12 (and 3.1) for tomography ray tracing')
        t=np.sqrt(2)
        G=np.array([[1,0,0,1,0,0,1,0,0],
                    [0,1,0,0,1,0,0,1,0],
                    [0,0,1,0,0,1,0,0,1],
                    [1,1,1,0,0,0,0,0,0],
                    [0,0,0,1,1,1,0,0,0],
                    [0,0,0,0,0,0,1,1,1],
                    [t,0,0,0,t,0,0,0,t],
                    [0,0,0,0,0,0,0,0,t]],dtype='float')
    elif iex==2:
        stlab = 'vertical seismic profiling (Ex 1.3)'
        print('Aster Example 1.3, 1.9, 4.4 for the vertical seismic profiling')
        if argv:
            m=argv[0]
            n=argv[1]
        else:
            m=100
            n=m
        G=np.zeros((m,n))
        if n>=m:
            f=n/m
            for i in range(m):
                ncut=int(np.ceil(f*(i+1)))
                G[i,:ncut]=1
        else:
            f=m/n
            fint=int(np.floor(f))
            for k in range(n):
                for nf in np.arange(1,int(fint+1)):
                    inds=int(k*f+nf)
                    G[inds-1,k]=nf
                    for kk in range(k,0,-1):
                        #print(kk)
                        G[inds-1,kk-1]=fint
        # multiply by the depth increment
        zmin = 0
        zmax = 20    # Exercise 1-3-a
        zran = zmax-zmin
        dz = zran / m
        G = G*dz
    elif iex==3:
        if argv:
            m=argv[0]
            n=argv[1]
        else:
            m=210
            n=m
        stlab = 'deconvolution (Ex 3.2)'
        print('Aster Example 3.2 for deconvolution of instrument response (m=n)')
        # Discretizing values for M & N (210 data points)
        t=np.linspace(-5,100,n+1)
        sigi = 10;
        gmax = 3.6788

        G = np.zeros((m,n))
        for i in range(2,m+2):
            for j in range(n):
                tp = t[j]-t[i-1]
                if (tp > 0):
                    G[i-2,j] = 0
                else:
                    G[i-2,j] = -tp*np.exp(tp/sigi)
        
        # now divide everything by the denominator
        deltat = t[2]-t[1]
        G = G/gmax * deltat
    elif iex==4:
        stlab = 'Shaw slit (Ex 1.6)'
        print('Aster Example 1.6 and Example 3.3 for the Shaw slit diffraction problem (m=n)')
        if argv:
            m=argv[0]
            n=argv[1]
        else:
            n=20
        m=n
        G,b,x = shaw(n)
    if bfigure:
        plt.figure(figsize=(7,7))
        plt.imshow(G)
        plt.xlabel('column index k')
        plt.ylabel('row index i')
        plt.axis('equal')
        plt.colorbar()
        plt.title('G matrix [%i x %i] for %s' %(G.shape[0],G.shape[1],stlab))
    return G

def collocate(xmin,xmax,n):
    #COLLOCATE simple collocation discretization
    # Example: x = collocate(0,1,20)

    # Aster Eqs. 1.34, 1.35
    dx = (xmax-xmin)/n
    x = xmin + dx/2 + (np.arange(1,n+1)-1)*dx
    return x

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
    



