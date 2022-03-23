# Python version for the PEIP book ex and lab lib
# Coded by Yuan Tian at UAF 2021.01
import numpy as np
import scipy
import matplotlib.pyplot as plt


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
    
    bfigure = True

    if iex==1:
        stlab = 'tomography ray tracing (Ex 1.12)'
        print('Aster Example 1.12 (and 3.1) for tomography ray tracing')
        t = np.sqrt(2)
        G = np.array([[1,0,0,1,0,0,1,0,0],
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
            m = argv[0]
            n = argv[1]
        else:
            m = 100
            n = m
        G = np.zeros((m,n))
        if n >= m:
            f = n/m
            for i in range(m):
                ncut=int(np.ceil(f*(i+1)))
                G[i,:ncut]=1
        else:
            f = m/n
            fint = int(np.floor(f))
            for k in range(n):
                for nf in np.arange(1,int(fint+1)):
                    inds = int(k*f+nf)
                    G[inds-1,k] = nf
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
            m = argv[0]
            n = argv[1]
        else:
            m = 210
            n = m
        stlab = 'deconvolution (Ex 3.2)'
        print('Aster Example 3.2 for deconvolution of instrument response (m=n)')
        # Discretizing values for M & N (210 data points)
        t = np.linspace(-5,100,n+1)
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
            m = argv[0]
            n = argv[1]
        else:
            n = 20
            m = n
        G,b,x = shaw(n)

    if bfigure:
        plt.figure(figsize=(7,7))
        plt.imshow(G)
        plt.xlabel('column index k')
        plt.ylabel('row index i')
        #plt.axis('equal')
        plt.colorbar()
        plt.title('G matrix [%i x %i] for %s' % (G.shape[0],G.shape[1],stlab))
    return G


def collocate(xmin,xmax,n):
    #COLLOCATE simple collocation discretization
    # Example: x = collocate(0,1,20)

    # Aster Eqs. 1.34, 1.35
    dx = (xmax-xmin)/n
    x = xmin + dx/2 + (np.arange(1,n+1)-1)*dx
    return x
