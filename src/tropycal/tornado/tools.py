import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt,maximum_filter
import warnings

def circle_filter(d):
    r=int(d/2)
    if d%2==0:
        y,x = np.ogrid[-r: r, -r: r]
        x=x+.5;y=y+.5
    else:
        y,x = np.ogrid[-r: r+1, -r: r+1]
    disk = x**2+y**2 <= r**2
    disk = disk.astype(float)
    return disk

def getPPH(dfTors,method='daily',res=10):
    
    r"""
    Calculate PPH density from tornado dataframe
    
    Parameters
    ----------
    dfTors : dataframe
    method : 'total' or 'daily'
    """
    
    # set up ~80km grid over CONUS
    latgrid = np.arange(20,55,res/111)
    longrid = np.arange(-130,-65,res/111/np.cos(35*np.pi/180))
    interval = int(80/res)
    disk = circle_filter(interval)
    
    dfTors['SPC_time'] = dfTors['UTC_time'] - timedelta(hours=12)
    dfTors = dfTors.set_index(['SPC_time'])
    groups = dfTors.groupby(pd.Grouper(freq="D"))
    
    aggregate_grid = []
    for group in groups: 
        slon,slat = group[1]['slon'].values,group[1]['slat'].values
        elon,elat = group[1]['elon'].values,group[1]['elat'].values
    
        torlons = [i for x1,x2 in zip(slon,elon) for i in np.linspace(x1,x2, 10)]
        torlats = [i for y1,y2 in zip(slat,elat) for i in np.linspace(y1,y2, 10)]
    
        # get grid count
        grid, _, _ = np.histogram2d(torlats,torlons, bins=[latgrid,longrid])
        grid = (grid>0)*1.0
        grid = maximum_filter(grid,footprint=disk)

        aggregate_grid.append(grid)
        
    if method == 'daily':
        grid = np.mean(aggregate_grid,axis=0)
        PPH = gfilt(grid,sigma=1.5*interval)*100
    if method == 'total':
        grid = np.sum(aggregate_grid,axis=0)
        PPH = gfilt((grid>=1)*1.0,sigma=1.5*interval)*100
        
    return PPH,.5*(longrid[:len(longrid)-1]+longrid[1:]),.5*(latgrid[:len(latgrid)-1]+latgrid[1:])


def ef_colors(x):
    import matplotlib as mlib
    if x == 'default':
        colors = ['lightsalmon','tomato','red','firebrick','darkred','purple']
    elif isinstance(x,str):
        try:
            cmap = mlib.cm.get_cmap(x)
            norm = mlib.colors.Normalize(vmin=0, vmax=5)
            colors = cmap(norm([0,1,2,3,4,5]))
        except:
            colors = [x]*6
    elif isinstance(x,list):
        if len(x) == 6:
            colors = x
    else:
        colors = ['lightsalmon','tomato','red','firebrick','darkred','purple']
    return colors


def PPH_colors(ptype,x,clevs):
    import matplotlib as mlib
    if x=='SPC':
        if ptype=='daily':
            clevs = [2,5,10,15,30,45,60,100]
            colors = ['#008B00',\
                      '#8B4726',\
                      '#FFC800',\
                      '#FF0000',\
                      '#FF00FF',\
                      '#912CEE',\
                      '#104E8B']
        else:
            warnings.warn('SPC colors only allowed for daily PPH.\n'+\
                          'Defaulting to plasma colormap.')
            x = 'plasma'
    if x!='SPC':
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
