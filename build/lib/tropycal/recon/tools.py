import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt
import warnings


def uv_from_wdir(wspd,wdir):
    d2r = np.pi/180.
    theta = (270 - wdir) * d2r
    u = wspd * np.cos(theta)
    v = wspd * np.sin(theta)
    return u,v

#------------------------------------------------------------------------------
# TOOLS FOR RECON INTERPOLATION
#------------------------------------------------------------------------------

def interpRecon(dfRecon,radlim=150):
    
    from scipy.interpolate import griddata
    
    # read in recon data
    data = [k for i,j,k in zip(dfRecon['xdist'],dfRecon['ydist'],dfRecon['wspd']) if np.nan not in [i,j,k]]
    path = [(i,j) for i,j,k in zip(dfRecon['xdist'],dfRecon['ydist'],dfRecon['wspd']) if np.nan not in [i,j,k]]
    
    # polar
    def cart2pol(x, y, offset=0):
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi+offset)
    
    pol_path = [cart2pol(*p) for p in path]
    pol_path_wrap = [cart2pol(*p,offset=-2*np.pi) for p in path]+pol_path+\
                [cart2pol(*p,offset=2*np.pi) for p in path]
    data_wrap = np.concatenate([data]*3)
    
    grid_rho, grid_phi = np.meshgrid(np.linspace(0,radlim,radlim*2+1),np.linspace(-np.pi,np.pi,181))
    grid_z_pol = griddata(pol_path_wrap,data_wrap,(grid_rho,grid_phi),method='linear')
    rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]
    print(rmw)
    filleye = np.where((grid_rho<rmw) & (np.isnan(grid_z_pol)))
    grid_z_pol[filleye]=0
    
    grid_z_pol_wrap = np.concatenate([grid_z_pol]*3)
    
    # smooth
    grid_z_pol_final = np.array([gfilt(grid_z_pol_wrap,(5,2+r/10))[:,i] \
                                 for i,r in enumerate(grid_rho[0,:])]).T[len(grid_phi):2*len(grid_phi)]
    
    # back to cartesian
    def pol2cart(rho, phi):
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return(x, y)
    
    pinterp_grid = [pol2cart(i,j) for i,j in zip(grid_rho.flatten(),grid_phi.flatten())]
    pinterp_z = grid_z_pol_final.flatten()
    
    grid_x, grid_y = np.meshgrid(np.linspace(-radlim,radlim,radlim*2+1),np.linspace(-radlim,radlim,radlim*2+1))
    grid_z = griddata(pinterp_grid,pinterp_z,(grid_x,grid_y),method='linear')

    return grid_x,grid_y,grid_z


#------------------------------------------------------------------------------
# TOOLS FOR PLOTTING
#------------------------------------------------------------------------------

def recon_colors(varname,x,clevs):
    import matplotlib as mlib
    if x=='category':
        if varname in ['sfmr','fl_to_sfc']:
            clevs = [34,64,83,96,113,137,200]
            colors = ['#8FC2F2','#3185D3','#FFFF00','#FF9E00','#DD0000','#FF00FC','#8B0088']
        else:
            warnings.warn('Saffir Simpson category colors allowed only for surface winds')
            x = 'plasma'
    if x!='category':
        if isinstance(x,str):
            cmap = mlib.cm.get_cmap(x)
            norm = mlib.colors.Normalize(vmin=0, vmax=len(clevs)-2)
            colors = cmap(norm(np.arange(len(clevs))))
        elif isinstance(x,list):
            colors = x
        else:
            norm = mlib.colors.Normalize(vmin=0, vmax=len(clevs)-2)
            colors = x(norm(np.arange(len(clevs))))
    return colors,clevs
    