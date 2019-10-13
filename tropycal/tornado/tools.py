import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
import requests
import urllib
from scipy.ndimage import gaussian_filter as gfilt


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
        aggregate_grid.append(gfilt((grid>0)*1.0,sigma=1.5)*100)
        
    if method == 'daily':
        PPF = np.mean(aggregate_grid,axis=0)
    if method == 'total':
        PPF = np.sum(aggregate_grid,axis=0)
        
    return PPF,longrid,latgrid

def ef_color(mag):
    colors=None
    return colors

def PPF_color(svrtype):
    if svrtype == 'tornado':
        clevs = [2,5,10,15,30,45,60,100]
        colors = ['#008B00',\
                  '#8B4726',\
                  '#FFC800',\
                  '#FF0000',\
                  '#FF00FF',\
                  '#912CEE',\
                  '#104E8B']
    return colors,clevs
