import copy
import matplotlib.pylab as plt
import numpy as np

# this contains functions called by hw_mogi.ipynb

# bounding box
def extents(vector_component):
    delta = vector_component[1] - vector_component[0]
    return [vector_component[0] - delta/2, vector_component[-1] + delta/2]


# plot 2D model
def plot_model(infile,nline,nsample,posting,output_filename=None,dpi=72,xsol=None,ysol=None,Vsol=None,zsol=None):
    # Calculate the bounding box
    extent_xvec = extents((np.arange(1, nsample*posting, posting)) / 1000)
    extent_yvec = extents((np.arange(1, nline*posting, posting)) / 1000)
    extent_xy = extent_xvec + extent_yvec
    
    plt.rcParams.update({'font.size': 14})
    inwrapped = (infile/10 + np.pi) % (2*np.pi) - np.pi
    cmap = copy.copy(plt.cm.get_cmap("jet"))
    cmap.set_bad('white', 1.)
    
    # Plot displacement
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(infile, interpolation='nearest', cmap=cmap, extent=extent_xy, origin='upper')
    cbar = ax1.figure.colorbar(im, ax=ax1, orientation='horizontal')
    ax1.set_title("Displacement in look direction [mm]")
    ax1.set_xlabel("Easting [km]")
    ax1.set_ylabel("Northing [km]")
    plt.grid()
    
    # Plot interferogram
    im.set_clim(-30, 30)
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(inwrapped, interpolation='nearest', cmap=cmap, extent=extent_xy, origin='upper')
    cbar = ax2.figure.colorbar(im, ax=ax2, orientation='horizontal')
    ax2.set_title("Interferogram phase [rad]")
    ax2.set_xlabel("Easting [km]")
    ax2.set_ylabel("Northing [km]")
    plt.grid()
    
    if xsol and ysol:
        ax1.plot(xsol,ysol,'kp',ms=18,mfc='w')
        ax2.plot(xsol,ysol,'kp',ms=18,mfc='w')
        
    #if Vsol and zsol:
        #ax1.plot(V[indV],zs[indz],'kp',ms=18,mfc='w')
        #ax2.plot(V[indx],zs[indz],'kp',ms=18,mfc='w')
    
    #return extent_xvec, extent_yvec


# Mogi forward model
def rngchg_mogi(n1, e1, depth, delta_volume, northing, easting, plook):
    
    # This geophysical coefficient is needed to describe how pressure relates to volume change
    displacement_coefficient = (1e6*delta_volume*3)/(np.pi*4)
    
    # Calculating the horizontal distance from every point in the displacement map to the x/y source location
    d_mat = np.sqrt(np.square(northing-n1) + np.square(easting-e1))
    
    # denominator of displacement field for mogi source
    tmp_hyp = np.power(np.square(d_mat) + np.square(depth),1.5)
    
    # horizontal displacement
    horizontal_displacement = displacement_coefficient * d_mat / tmp_hyp
    
    # vertical displacement
    vertical_displacement = -displacement_coefficient * depth / tmp_hyp #Amanda: added '-'
    
    # azimuthal angle
    azimuth = np.arctan2((easting-e1), (northing-n1))
    
    # compute north and east displacement from horizontal displacement and azimuth angle
    east_displacement = np.sin(azimuth) * horizontal_displacement
    north_displacement = np.cos(azimuth) * horizontal_displacement
    
    # project displacement field onto look vector
    temp = np.concatenate((east_displacement, north_displacement, vertical_displacement), axis=1)
    delta_range = temp.dot(np.transpose([plook]))
    delta_range = -1.0 * delta_range
    return delta_range
