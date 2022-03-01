# FFT related tools library of GEOS627 inverse course
# Coded by Yuan Tian at UAF 2021.01
# Contributers: Amanda McPherson
import numpy as np
import matplotlib.pyplot as plt


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


def xy2distance(nx,ny,bdisplay=False):
    # integer index vectors
    # NOTE: these start with 0 for convenience in the FFT algorithm
    ix0 = np.arange(nx)
    iy0 = np.arange(ny)

    # generate integer ix and iy vectors
    [iX,iY] = np.meshgrid(ix0,iy0)
    ix = iX.flatten(order='F')
    iy = iY.flatten(order='F')
    
    n  = nx*ny              # number of points in 2D grid
    nd = 0.5*(n**2 - n)     # number of unique distances
    
    # indexing matrices
    [PA,PB] = np.meshgrid(np.arange(n),np.arange(n))
    
    MX,MY = np.meshgrid(ix,iy)
    iD    = np.sqrt((MX-MX.T)**2 + (MY-MY.T)**2)
    
    if bdisplay:
        id = iD.flatten(order='F')
        pA = PA.flatten(order='F')
        pB = PB.flatten(order='F')
        print('---------------------------')
        print('%i (x) by %i (y) = %i gridpoints'% (nx,ny,n))
        print('%i total number of distances, %i of which are unique pairs'% (n**2,nd))
        
        for ii in range(n**2):
            print('%2i-%2i (%i, %i)-(%i, %i) = %6.2f'% (pA[ii],pB[ii],ix[pA[ii]],iy[pA[ii]],ix[pB[ii]],iy[pB[ii]],id[ii]))
        
        print('---------------------------')
        
        # Plot figures
        ind = np.arange(n)
        ax0 = [-1, nx, -1, ny]
        ax1 = [-1, n, -1, n]
    
        # print some output
        print('ind:')
        print(ind)
        print('ix:')
        print(ix)
        print('iy:')
        print(iy)
        print('PA:')
        print(PA)
        print('PB:')
        print(PB)
        print('iD:')
        print(iD)
        
        ud = np.unique(id)
        print('%i unique nonzero entries:'% (len(ud)-1))
        print(ud[1:])
        
        plt.figure(figsize=(8,5.5))
        plt.plot(ix,iy,'.',ms='16')
        for kk in range(len(ix)):
            plt.text(ix[kk],iy[kk],str(ind[kk]))
            
        plt.axis(ax0)
        plt.grid()
        plt.xlabel('x (unshifted and unscaled)')
        plt.ylabel('y (unshifted and unscaled)')
        plt.title('Indexing of points in the mesh')
    
        plt.figure(figsize=(8,14))
        nr=2
        nc=1
        plt.subplot(nr,nc,1)
        plt.imshow(PA,vmin=1,vmax=n)
        #plt.axis(ax1)
        plt.colorbar()
        plt.title('Point A index')
    
        plt.subplot(nr,nc,2)
        plt.imshow(PB,vmin=1,vmax=n)
        #plt.axis(ax1)
        plt.colorbar()
        plt.title('Point B index')
    
        plt.figure(figsize=(8,14))
        nr=2
        nc=1
        plt.subplot(nr,nc,1)
        plt.imshow(iD)
        #plt.axis(ax1)
        plt.colorbar()
        plt.xlabel('Index of point B')
        plt.ylabel('Index of point A')
        plt.title('Index distance between points A and B, max(iD) = %.2f' % (np.amax(iD.flatten(order='F'))))
        
        # the next figures are related to a prescribed covariance function
    
        # compute covariance matrix
        iL = 1
        sigma = 1
        R = sigma**2 * np.exp(-iD**2 / (2*iL**2) )
    
        # plot
        stit = '(nx,ny,n) = (%i,%i,%i), iL = %i, sigma = %i' % (nx,ny,n,iL,sigma)
        stit2 = '%i x %i block Toeplitz with %i x %i (%i) blocks, each %i x %i' % (n,n,nx,nx,nx*nx,ny,ny)
    
        plt.subplot(nr,nc,2)
        plt.imshow(R)
        #plt.axis(ax1)
        plt.colorbar()
        plt.xlabel('Index of point A')
        plt.ylabel('Index of point B')
        plt.title('Gaussian covariance between points A and B\n' + stit2)
    
        # Gaussian sample
        A = np.linalg.cholesky(R)
        g = A@np.random.randn(n,1)
        
        plt.figure(figsize=(10,8))
        plt.imshow(g.reshape((ny,nx)),vmin=-3*sigma,vmax=3*sigma) 
        #set(gca,'ydir','normal')
        
        plt.plot(ix,iy,'o',mfc='w',mec='b')
        for kk in range(len(ix)):
            plt.text(ix[kk],iy[kk],str(ind[kk]))
            
        plt.axis([ax0[0],ax0[1],ax0[2],ax0[3]])
        plt.colorbar()
        plt.xlabel('x (unshifted and unscaled)')
        plt.ylabel('y (unshifted and unscaled)')
        plt.title(' Cm sample: %s' % (stit))
        
    return iD,ix0,iy0,ix,iy,PA,PB


def x2distance(xmin,xmax,nx):
    ix0     = np.arange(nx)
    Dx      = xmax-xmin
    [X1,X2] = np.meshgrid(ix0,ix0)
    iD      = np.abs(X1-X2)
    dx      = Dx/(nx-1)
    return iD,dx,ix0


def k_of_x(x):
    N     = np.max(x.shape)
    dx    = x[1]-x[0]
    dk    = (2*np.pi)/(N*dx)
    inull = N/2
    k     = dk*(np.linspace(1,N,N)-inull)
    return k


def x_of_k(k):
    N     = np.max(k.shape)
    dk    = k[1]-k[0]
    dx    = (2*np.pi)/(N*dk)
    x     = dx*(np.linspace(1,N,N)-1)
    return x


def mhfft(x,f):
    Nx      = np.max(x.shape)
    k       = k_of_x(x)
    Periodx = Nx*(x[1]-x[0])
    inull   = Nx/2
    ft      = (Periodx/Nx)*np.roll(np.fft.fft(f),int(inull-1))
    return k,ft


def mhfft2(x,y,f):
    # 2D Fast Fourier Transform of (x,y,f) into (k,l,ft).  The length of 
    # x,y  and f must be an even number, preferably a power of two.  The index of
    # the zero mode is inull=jnull=N/2.

    # Everything is assumed to have been generated by meshgrid, so that
    # f is indexed f(y,x)

    Nx      = np.max(x.shape)
    Ny      = np.max(y.shape)
    k       = k_of_x(x)
    l       = k_of_x(y)
    Periodx = Nx*(x[1]-x[0])
    Periody = Ny*(y[1]-y[0])
    inull   = Nx/2
    jnull   = Ny/2
    ft      = (Periodx/Nx)*(Periody/Ny)*np.roll(np.roll(np.fft.fft2(f),int(jnull-1),axis=0),int(inull-1),axis=1)
    return k,l,ft


def grf2(k,m,C,n,*argv):
    Nx      = np.max(k.shape)
    Ny      = np.max(m.shape)
    dk      = k[1]-k[0]
    dm      = m[1]-m[0]
    Periodx = 2*np.pi/dk
    Periody = 2*np.pi/dm
    Cmtx    = np.repeat(C[:, :, np.newaxis], n, axis=2)
    
    # if nargin==6:
    #     disp('grf2mod.m: using the provided Gaussian random vectors (A and B)')
    # else:
    if argv:
        A = argv[0]
        B = argv[1]
    else:
        A = np.random.randn(Ny,Nx,n)   # N(0,1) random variables
        B = np.random.randn(Ny,Nx,n) 
    phi = np.lib.scimath.sqrt(Periodx*Periody*Cmtx/2)*(A+B*1j)
    #phi[np.isnan(phi)]=0
    return phi,A,B


def mhifft2(k,l,ft,rflag):
    # 2D Fast Fourier Transform of (x,y,f) into (k,l,ft).  The length of 
    # x,y  and f must be an even number, preferably a power of two.  The index of
    # the zero mode is inull=jnull=N/2.

    # Everything is assumed to have been generated by meshgrid, so that
    # f is indexed f(y,x)

    Nx      = np.max(k.shape)
    Ny      = np.max(l.shape)
    x       = x_of_k(k)
    y       = x_of_k(l)
    Periodx = Nx*(x[1]-x[0])
    Periody = Ny*(y[1]-y[0])
    inull   = Nx/2
    jnull   = Ny/2
    f       = (Nx/Periodx)*(Ny/Periody)*np.fft.ifftn(np.roll(np.roll(ft,-int(jnull-1),axis=0),-int(inull-1),axis=1),axes=(0,1))
    if rflag==1:
        f = np.real(f)
    return x,y,f


def mhfft(x,f):
    Nx      = np.max(x.shape)
    k       = k_of_x(x)
    Periodx = Nx*(x[1]-x[0])
    inull   = Nx/2
    ft      = (Periodx/Nx)*np.roll(np.fft.fft(f),int(inull-1))
    return k,ft


def grf1(k,C,n):
    Nx      = np.max(k.shape)
    dk      = k[1]-k[0]
    Periodx = 2*np.pi/dk
    Cmtx    = np.repeat(C[:, np.newaxis], n, axis=1)
    #Cmtx=Cmtx.reshape((n,Nx))
    A       = np.random.randn(Nx,n) 
    B       = np.random.randn(Nx,n)
    phi     = np.sqrt(Periodx*Cmtx/2)*(A+B*1j)
    phi[np.isnan(phi)] = 0
    return phi


def mhifft(k,f):
    Nx      = np.max(k.shape)
    x       = x_of_k(k)
    Periodx = Nx*(x[1]-x[0])

    inull   = Nx/2
    ft      = (Nx/Periodx)*np.fft.ifft(np.roll(f,-int(inull-1)),axis=0)
    return k,ft
