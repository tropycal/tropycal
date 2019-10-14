import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt
import warnings

def getPPF(dfTors,method='total'):
    
    r"""
    Calculate PPF density from tornado dataframe
    
    Parameters
    ----------
    dfTors : dataframe
    method : 'total' or 'daily'
    """
    
    # set up ~80km grid over CONUS
    latgrid = np.arange(20,50,80/111)
    longrid = np.arange(-130,-65,80/111/np.cos(35*np.pi/180))
    
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
        aggregate_grid.append((grid>0)*1.0)
        
    if method == 'daily':
        grid = np.mean(aggregate_grid,axis=0)
        PPF = gfilt(grid,sigma=1.5)*100
    if method == 'total':
        grid = np.sum(aggregate_grid,axis=0)
        PPF = gfilt((grid>0)*1.0,sigma=1.5)*100
        
    return PPF,longrid,latgrid


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


def ppf_colors(ptype,x,clevs):
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
            warnings.warn('SPC colors only allowed for daily PPF.\n'+\
                          'Defaulting to plasma colormap.')
            x = 'plasma'
    if x!='SPC':
        if isinstance(x,str):
            cmap = mlib.cm.get_cmap(x)
            norm = mlib.colors.Normalize(vmin=min(clevs), vmax=max(clevs[:-1]))
            colors = cmap(norm(clevs))
        elif isinstance(x,list):
            colors = x
        else:
            norm = mlib.colors.Normalize(vmin=min(clevs), vmax=max(clevs[:-1]))
            colors = x(norm(clevs))
    return colors,clevs
