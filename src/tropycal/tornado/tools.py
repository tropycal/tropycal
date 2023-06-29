import numpy as np
import pandas as pd
from datetime import timedelta
from scipy.ndimage import gaussian_filter as gfilt, maximum_filter


def circle_filter(d):
    r = int(d/2)
    if d % 2 == 0:
        y, x = np.ogrid[-r: r, -r: r]
        x = x+.5
        y = y+.5
    else:
        y, x = np.ogrid[-r: r+1, -r: r+1]
    disk = x**2+y**2 <= r**2
    disk = disk.astype(float)
    return disk


def getPPH(dfTors, method='daily', res=10):
    r"""
    Calculate PPH density from tornado dataframe

    Parameters
    ----------
    dfTors : dataframe
    method : 'total' or 'daily'
    """

    # set up ~80km grid over CONUS
    latgrid = np.arange(20, 55, res/111)
    longrid = np.arange(-130, -65, res/111/np.cos(35*np.pi/180))
    interval = int(80/res)
    disk = circle_filter(interval)

    dfTors['SPC_time'] = dfTors['UTC_time'] - timedelta(hours=12)
    dfTors = dfTors.set_index(['SPC_time'])
    groups = dfTors.groupby(pd.Grouper(freq="D"))

    aggregate_grid = []
    for group in groups:
        slon, slat = group[1]['slon'].values, group[1]['slat'].values
        elon, elat = group[1]['elon'].values, group[1]['elat'].values

        torlons = [i for x1, x2 in zip(slon, elon)
                   for i in np.linspace(x1, x2, 10)]
        torlats = [i for y1, y2 in zip(slat, elat)
                   for i in np.linspace(y1, y2, 10)]

        # get grid count
        grid, _, _ = np.histogram2d(torlats, torlons, bins=[latgrid, longrid])
        grid = (grid > 0)*1.0
        grid = maximum_filter(grid, footprint=disk)

        aggregate_grid.append(grid)

    if method == 'daily':
        grid = np.mean(aggregate_grid, axis=0)
        PPH = gfilt(grid, sigma=1.5*interval)*100
    if method == 'total':
        grid = np.sum(aggregate_grid, axis=0)
        PPH = gfilt((grid >= 1)*1.0, sigma=1.5*interval)*100

    return PPH, .5*(longrid[:len(longrid)-1]+longrid[1:]), .5*(latgrid[:len(latgrid)-1]+latgrid[1:])
