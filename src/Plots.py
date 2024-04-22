""" Functions to make plots """

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

def grid(model, depthmax=10, ny=101, nlay=2):
    """ Generates a grid from the model to plot a 2D section
    
    I CAN IMPROVE THIS  """
    # Arrays for plotting
    npos = np.shape(model)[0] # number of 1D models
   # ny = 101 # size of the grid in y direction
    y = np.linspace(0, depthmax, ny) # y axis [m]
    grid = np.zeros((npos, ny)) # empty grid
    thk = model[:,:nlay-1].copy() # define electrical conductivities
    sig = model[:,nlay-1:].copy()  # define thicknesses
    
    # Fill the grid with the conductivity values
    
    if nlay == 3:
        for i in range(npos):
            y1 = 0
            # First layer
            while y[y1] < thk[i,0]:
                grid[i, y1] = sig[i, 0]
                y1 += 1
                #y2 = y1
            # Second layer
            while y[y1] < (thk[i,0] + thk[i,1]):
                grid[i, y1] = sig[i, 1]
                y1 += 1
            # Third layer
            grid[i, y1:] = sig[i, 2]
    
    if nlay == 2:   
        for i in range(npos):
            y1 = 0
            # First layer
            while y[y1] < thk[i,0]:
                grid[i, y1] = sig[i, 0]
                y1 += 1
                if y1 > ny-1:
                    break
            while y[y1] >= thk[i,0]:
                grid[i, y1] = sig[i, 1]
                y1 += 1
                if y1 > ny-1:
                    break
        
    return grid



def Plot_Datas(data_true, data_est, ax=None):
    
    if ax == None:
        fig, ax = plt.subplots(3,2, sharex=True)

    ax[0,0].semilogy(data_est[:,0,0,0], '*b', label = '2m est')
    ax[0,0].semilogy(data_est[:,0,0,1], '*r', label = '4m est')
    ax[0,0].semilogy(data_est[:,0,0,2], '*g', label = '8m est')
    ax[0,0].semilogy(data_true[:,0,0,0], ':b', label = '2m true')
    ax[0,0].semilogy(data_true[:,0,0,1], ':r', label = '4m true')
    ax[0,0].semilogy(data_true[:,0,0,2], ':g', label = '8m true')
    ax[0,0].set_title('OP component H coils')

    ax[0,1].semilogy(data_est[:,1,0,0], '*b', label = '2m est')
    ax[0,1].semilogy(data_est[:,1,0,1], '*r', label = '4m est')
    ax[0,1].semilogy(data_est[:,1,0,2], '*g', label = '8m est')
    ax[0,1].semilogy(data_true[:,1,0,0], ':b', label = '2m true')
    ax[0,1].semilogy(data_true[:,1,0,1], ':r', label = '4m true')
    ax[0,1].semilogy(data_true[:,1,0,2], ':g', label = '8m true')
    ax[0,1].set_title('IP component H coils')
    ax[0,1].legend(bbox_to_anchor=(1.1, 1.05))

    ax[1,0].semilogy(data_est[:,0,1,0], '*b', label = '2m est')
    ax[1,0].semilogy(data_est[:,0,1,1], '*r', label = '4m est')
    ax[1,0].semilogy(data_est[:,0,1,2], '*g', label = '8m est')
    ax[1,0].semilogy(data_true[:,0,1,0], ':b', label = '2m true')
    ax[1,0].semilogy(data_true[:,0,1,1], ':r', label = '4m true')
    ax[1,0].semilogy(data_true[:,0,1,2], ':g', label = '8m true')
    ax[1,0].set_title('OP component V coils')

    ax[1,1].semilogy(data_est[:,1,1,0], '*b', label = '2m est')
    ax[1,1].semilogy(data_est[:,1,1,1], '*r', label = '4m est')
    ax[1,1].semilogy(data_est[:,1,1,2], '*g', label = '8m est')
    ax[1,1].semilogy(data_true[:,1,1,0], ':b', label = '2m true')
    ax[1,1].semilogy(data_true[:,1,1,1], ':r', label = '4m true')
    ax[1,1].semilogy(data_true[:,1,1,2], ':g', label = '8m true')
    ax[1,1].set_title('IP component V coils')

    ax[2,0].semilogy(data_est[:,0,2,0], '*b', label = '2m est')
    ax[2,0].semilogy(data_est[:,0,2,1], '*r', label = '4m est')
    ax[2,0].semilogy(data_est[:,0,2,2], '*g', label = '8m est')
    ax[2,0].semilogy(data_true[:,0,2,0], ':b', label = '2m true')
    ax[2,0].semilogy(data_true[:,0,2,1], ':r', label = '4m true')
    ax[2,0].semilogy(data_true[:,0,2,2], ':g', label = '8m true')
    ax[2,0].set_title('OP component P coils')

    ax[2,1].semilogy(data_est[:,1,2,0], '*b', label = '2m est')
    ax[2,1].semilogy(data_est[:,1,2,1], '*r', label = '4m est')
    ax[2,1].semilogy(data_est[:,1,2,2], '*g', label = '8m est')
    ax[2,1].semilogy(data_true[:,1,2,0], ':b', label = '2m true')
    ax[2,1].semilogy(data_true[:,1,2,1], ':r', label = '4m true')
    ax[2,1].semilogy(data_true[:,1,2,2], ':g', label = '8m true')
    ax[2,1].set_title('IP component P coils')

    plt.tight_layout()
    
def Plot2Datas(data_1D, data_3D):

    fig, ax = plt.subplots(3,4, sharex = True, sharey = True)

    ax[0,0].semilogy(data_1D[:,0], '.b', label='H2 Q')
    ax[0,0].semilogy(data_3D[:,0], 'xb' )
    ax[0,0].semilogy(data_1D[:,1], '.k', label = 'H4 Q')
    ax[0,0].semilogy(data_3D[:,1], 'xk')
    ax[0,0].semilogy(data_1D[:,2], '.r', label= 'H8 Q')
    ax[0,0].semilogy(data_3D[:,2], 'xr' )
    ax[0,0].legend(fontsize=7)

    ax[0,1].semilogy(100*np.abs((data_1D[:,0]-data_3D[:,0])/data_3D[:,0]), ':b', label='H2 Q')
    ax[0,1].semilogy(100*np.abs((data_1D[:,1]-data_3D[:,1])/data_3D[:,1]), ':k', label='H4 Q')
    ax[0,1].semilogy(100*np.abs((data_1D[:,2]-data_3D[:,2])/data_3D[:,2]), ':r', label='H8 Q')
    ax[0,1].legend(fontsize=7)

    ax[1,0].semilogy(data_1D[:,3], '.b', label='P2 Q')
    ax[1,0].semilogy(data_3D[:,3], 'xb' )
    ax[1,0].semilogy(data_1D[:,4], '.k', label = 'P4 Q')
    ax[1,0].semilogy(data_3D[:,4], 'xk' )
    ax[1,0].semilogy(data_1D[:,5], '.r', label= 'P8 Q')
    ax[1,0].semilogy(data_3D[:,5], 'xr')
    ax[1,0].legend(fontsize=7)

    ax[1,1].semilogy(100*np.abs((data_1D[:,3]-data_3D[:,3])/data_3D[:,3]), 'b:', label='P2 Q')
    ax[1,1].semilogy(100*np.abs((data_1D[:,4]-data_3D[:,4])/data_3D[:,4]), ':k', label='P4 Q')
    ax[1,1].semilogy(100*np.abs((data_1D[:,5]-data_3D[:,5])/data_3D[:,5]), ':r', label='P8 Q')
    ax[1,1].legend(fontsize=7)

    ax[2,0].semilogy(data_1D[:,6], '.b', label='V2 Q')
    ax[2,0].semilogy(data_3D[:,6], 'xb' )
    ax[2,0].semilogy(data_1D[:,7], '.k', label = 'V4 Q')
    ax[2,0].semilogy(data_3D[:,7], 'xk')
    ax[2,0].semilogy(data_1D[:,8], '.r', label= 'V8 Q')
    ax[2,0].semilogy(data_3D[:,8], 'xr' )
    ax[2,0].legend(fontsize=7)

    ax[2,1].semilogy(100*np.abs((data_1D[:,6]-data_3D[:,6])/data_3D[:,6]), ':b', label='V2 Q')
    ax[2,1].semilogy(100*np.abs((data_1D[:,7]-data_3D[:,7])/data_3D[:,7]), ':k', label='V4 Q')
    ax[2,1].semilogy(100*np.abs((data_1D[:,8]-data_3D[:,8])/data_3D[:,8]), ':r', label='V8 Q')
    ax[2,1].legend(fontsize=7)

    ax[0,2].semilogy(data_1D[:,9], '.b', label='H2 IP')
    ax[0,2].semilogy(data_3D[:,9], 'xb' )
    ax[0,2].semilogy(data_1D[:,10], '.k', label = 'H4 IP')
    ax[0,2].semilogy(data_3D[:,10], 'xk' )
    ax[0,2].semilogy(data_1D[:,11], '.r', label= 'H8 IP')
    ax[0,2].semilogy(data_3D[:,11], 'xr' )
    ax[0,2].legend(fontsize=7)

    ax[0,3].semilogy(100*np.abs((data_1D[:,9]-data_3D[:,9])/data_3D[:,9]), ':b', label = 'H2 IP')
    ax[0,3].semilogy(100*np.abs((data_1D[:,10]-data_3D[:,10])/data_3D[:,10]), ':k', label = 'H4 IP')
    ax[0,3].semilogy(100*np.abs((data_1D[:,11]-data_3D[:,11])/data_3D[:,11]), ':r', label = 'H8 IP')
    ax[0,3].legend(fontsize=7)

    ax[1,2].semilogy(data_1D[:,12], '.b', label='P2 IP')
    ax[1,2].semilogy(data_3D[:,12], 'xb' )
    ax[1,2].semilogy(data_1D[:,13], '.k', label = 'P4 IP')
    ax[1,2].semilogy(data_3D[:,13], 'xk' )
    ax[1,2].semilogy(data_1D[:,14], '.r', label= 'P8 IP')
    ax[1,2].semilogy(data_3D[:,14], 'xr' )
    ax[1,2].legend(fontsize=7)

    ax[1,3].semilogy(100*np.abs((data_1D[:,12]-data_3D[:,12])/data_3D[:,12]), ':b', label = 'P2 IP')
    ax[1,3].semilogy(100*np.abs((data_1D[:,13]-data_3D[:,13])/data_3D[:,13]), ':k', label = 'P4 IP')
    ax[1,3].semilogy(100*np.abs((data_1D[:,14]-data_3D[:,14])/data_3D[:,14]), ':r', label = 'P8 IP')
    ax[1,3].legend(fontsize=7)

    ax[2,2].semilogy(data_1D[:,15], '.b', label='V2 IP')
    ax[2,2].semilogy(data_3D[:,15], 'xb' )
    ax[2,2].semilogy(data_1D[:,16], '.k', label = 'V4 IP')
    ax[2,2].semilogy(data_3D[:,16], 'xk' )
    ax[2,2].semilogy(data_1D[:,17], '.r', label= 'V8 IP')
    ax[2,2].semilogy(data_3D[:,17], 'xr' )
    ax[2,2].legend(fontsize=7)

    ax[2,3].semilogy(100*np.abs((data_1D[:,15]-data_3D[:,15])/data_3D[:,15]), ':b', label='V2 IP')
    ax[2,3].semilogy(100*np.abs((data_1D[:,16]-data_3D[:,16])/data_3D[:,16]), ':k', label='V4 IP')
    ax[2,3].semilogy(100*np.abs((data_1D[:,17]-data_3D[:,17])/data_3D[:,17]), ':r', label='V8 IP')
    ax[2,3].legend(fontsize=7)
    plt.tight_layout()
    
def Plot2Datas_field(data_1D, data_3D):

    fig, ax = plt.subplots(2,4, sharex = True, sharey = True)

    ax[0,0].semilogy(data_1D[:,0], '.b', label='H2 Q')
    ax[0,0].semilogy(data_3D[:,0], 'xb' )
    ax[0,0].semilogy(data_1D[:,1], '.k', label = 'H4 Q')
    ax[0,0].semilogy(data_3D[:,1], 'xk')
    ax[0,0].semilogy(data_1D[:,2], '.r', label= 'H8 Q')
    ax[0,0].semilogy(data_3D[:,2], 'xr' )
    ax[0,0].legend(fontsize=7)

    ax[0,1].semilogy(100*np.abs((data_1D[:,0]-data_3D[:,0])/data_3D[:,0]), ':b', label='H2 Q')
    ax[0,1].semilogy(100*np.abs((data_1D[:,1]-data_3D[:,1])/data_3D[:,1]), ':k', label='H4 Q')
    ax[0,1].semilogy(100*np.abs((data_1D[:,2]-data_3D[:,2])/data_3D[:,2]), ':r', label='H8 Q')
    ax[0,1].legend(fontsize=7)

    ax[1,0].semilogy(data_1D[:,3], '.b', label='P2 Q')
    ax[1,0].semilogy(data_3D[:,3], 'xb' )
    ax[1,0].semilogy(data_1D[:,4], '.k', label = 'P4 Q')
    ax[1,0].semilogy(data_3D[:,4], 'xk' )
    ax[1,0].semilogy(data_1D[:,5], '.r', label= 'P8 Q')
    ax[1,0].semilogy(data_3D[:,5], 'xr')
    ax[1,0].legend(fontsize=7)

    ax[1,1].semilogy(100*np.abs((data_1D[:,3]-data_3D[:,3])/data_3D[:,3]), 'b:', label='P2 Q')
    ax[1,1].semilogy(100*np.abs((data_1D[:,4]-data_3D[:,4])/data_3D[:,4]), ':k', label='P4 Q')
    ax[1,1].semilogy(100*np.abs((data_1D[:,5]-data_3D[:,5])/data_3D[:,5]), ':r', label='P8 Q')
    ax[1,1].legend(fontsize=7)


    ax[0,2].semilogy(data_1D[:,6], '.b', label='H2 IP')
    ax[0,2].semilogy(data_3D[:,6], 'xb' )
    ax[0,2].semilogy(data_1D[:,6], '.k', label = 'H4 IP')
    ax[0,2].semilogy(data_3D[:,6], 'xk' )
    ax[0,2].semilogy(data_1D[:,7], '.r', label= 'H8 IP')
    ax[0,2].semilogy(data_3D[:,7], 'xr' )
    ax[0,2].legend(fontsize=7)

    ax[0,3].semilogy(100*np.abs((data_1D[:,6]-data_3D[:,6])/data_3D[:,6]), ':b', label = 'H2 IP')
    ax[0,3].semilogy(100*np.abs((data_1D[:,7]-data_3D[:,7])/data_3D[:,7]), ':k', label = 'H4 IP')
    ax[0,3].semilogy(100*np.abs((data_1D[:,8]-data_3D[:,8])/data_3D[:,8]), ':r', label = 'H8 IP')
    ax[0,3].legend(fontsize=7)

    ax[1,2].semilogy(data_1D[:,9], '.b', label='P2 IP')
    ax[1,2].semilogy(data_3D[:,9], 'xb' )
    ax[1,2].semilogy(data_1D[:,10], '.k', label = 'P4 IP')
    ax[1,2].semilogy(data_3D[:,10], 'xk' )
    ax[1,2].semilogy(data_1D[:,11], '.r', label= 'P8 IP')
    ax[1,2].semilogy(data_3D[:,11], 'xr' )
    ax[1,2].legend(fontsize=7)

    ax[1,3].semilogy(100*np.abs((data_1D[:,9]-data_3D[:,9])/data_3D[:,9]), ':b', label = 'P2 IP')
    ax[1,3].semilogy(100*np.abs((data_1D[:,10]-data_3D[:,10])/data_3D[:,10]), ':k', label = 'P4 IP')
    ax[1,3].semilogy(100*np.abs((data_1D[:,11]-data_3D[:,11])/data_3D[:,11]), ':r', label = 'P8 IP')
    ax[1,3].legend(fontsize=7)

    plt.tight_layout()
    