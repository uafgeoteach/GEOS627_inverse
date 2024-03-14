import matplotlib.pyplot as plt
import numpy as np

from sympy import Matrix, init_printing
from IPython.display import display

###################################################################

def showmat(A, numdig=None):
    if numdig is not None:
        if numdig == 0:
            A_rounded = np.around(A).astype(int)
        else:
            A_rounded = np.around(A, decimals=numdig)
    else:
        A_rounded = A
    sym_matrix = Matrix(A_rounded)
    display(sym_matrix)
    
###################################################################

def fftvec(t):
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

###################################################################

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

###################################################################

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
    
###################################################################