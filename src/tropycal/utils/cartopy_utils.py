r"""Adds Tropycal functionality to Cartopy GeoAxes."""

import types
import cartopy.crs as ccrs

def add_tropycal(ax):
    
    r"""
    Adds Tropycal plotting capability to a matplotlib.pyplot axes instance with a Cartopy projection.
    
    This axes instance must have already had a Cartopy projection added (e.g., ``projection=ccrs.PlateCarree()``).
    
    Parameters
    ----------
    ax : cartopy.mpl.geoaxes.GeoAxes
        Instance of a matplotlib axes with a Cartopy projection added.
    
    Returns
    -------
    ax
        The same axes instance is returned, with tropycal plotting functions from `tropycal.utils.cartopy_utils` added to it as methods.
    """
    
    ax.plot_storm = types.MethodType( plot_storm, ax )
    
    return ax
    
def plot_storm(self,storm,*args,**kwargs):
    
    r"""
    Plot a Storm object on the axes instance.
    
    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Instance of a Storm object to be plotted.
    
    Notes
    -----
    Besides the parameters listed above, this function behaves identically to matplotlib's default `plot()` function.
    
    It is not necessary to pass a "transform" keyword argument, as this is already assumed to be ccrs.PlateCarree().
    
    This function is already appended to an axes instance if ``ax = utils.add_tropycal(ax)`` is run beforehand. This allows this method to be called simply via ``ax.plot_storm(...)`` the same way one would call ``ax.plot(...)``.
    """
    
    #Filter to only tropical points if requested
    if 'stormtype' in kwargs.keys():
        stormtype = kwargs.pop('stormtype')
        storm = storm.sel(stormtype=stormtype)
    
    #Pass arguments to ax plot method
    self.plot(storm.lon,storm.lat,*args,**kwargs,transform=ccrs.PlateCarree())
