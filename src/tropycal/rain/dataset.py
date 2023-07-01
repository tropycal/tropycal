r"""Functionality for reading and analyzing SPC tornado dataset."""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime as dt
from scipy.interpolate import griddata
import warnings

from .plot import RainPlot

from ..utils import *


class RainDataset():

    r"""
    Creates an instance of a RainDataset object containing tropical cyclone rainfall data, courtesy of the Weather Prediction Center (WPC).

    Parameters
    ----------
    data_path : str
        Source to read tropical cyclone rainfall data from. Default is "wpc", which reads from the online Weather Prediction Center (WPC) database. Can change this to a local file.

    Returns
    -------
    RainDataset
        An instance of RainDataset.
    """

    def __init__(self, data_path='wpc'):

        # Start timer
        timer_start = dt.now()
        print(f'--> Starting to read in rainfall data')

        # Read in storm rainfall dataset
        if data_path == 'wpc':
            data_path = "https://www.wpc.ncep.noaa.gov/tropical/rain/CONUS_rainfall_obs_1900-2020.csv"
        self.rain_df = pd.read_csv(data_path)

        # Update user on duration
        print(f'--> Completed reading in rainfall data (%.2f seconds)' %
              (dt.now()-timer_start).total_seconds())

    def get_storm_rainfall(self, storm):
        r"""
        Retrieves all rainfall observations in inches associated with a tropical cyclone.

        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Instance of a Storm object.

        Returns
        -------
        pandas.DataFrame
            Pandas DataFrame object containing rainfall data associated with this tropical cyclone, in inches.
        """

        # Filter dataset to this specific storm
        name_1 = f"{storm.name.title()} {storm.year}"
        name_2 = f"{storm.id} {storm.year}"
        df_storm = self.rain_df.loc[(self.rain_df['Storm'] == name_1) | (
            self.rain_df['Storm'] == name_2)]

        # Drop Storm and Year column, no longer necessary
        df_storm = df_storm.drop(columns=['Storm', 'Year'])

        # Remove NaN entries
        df_storm = df_storm.loc[~np.isnan(df_storm['Lat']) & ~np.isnan(
            df_storm['Lon']) & ~np.isnan(df_storm['Total'])]

        # Check if data is empty
        if len(df_storm) == 0:
            raise RuntimeError("No rainfall data is available for this storm.")

        # Return dataframe
        return df_storm

    def interpolate_to_grid(self, storm, grid_res=0.1, method='linear', return_xarray=False):
        r"""
        Interpolates storm rainfall data to a horizontal grid.

        Interpolation is performed using Scipy's `scipy.interpolate.griddata()` interpolation function.

        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Instance of a Storm object.
        grid_res : int or float
            Horizontal resolution of the desired grid in degrees. Default is 0.1 degrees.
        method : str
            Method for interpolation to pass to scipy's interpolation function. Default is "linear".
        return_xarray : bool
            If True, output is returned as an xarray DataArray with coordinates included. Default is false.

        Returns
        -------
        dict or xarray.DataArray
            If return_xarray is True, an xarray DataArray is returned. Otherwise, a dict including the grid lat, lon and grid values is returned.
        """

        # Check if Storm object contains rainfall data
        try:
            storm.rain
        except:
            storm.rain = self.get_storm_rainfall(storm)

        # Create grid to interpolate observations to
        grid_lon = np.arange(-140, -60+grid_res, grid_res)
        grid_lat = np.arange(20, 50+grid_res, grid_res)

        # Retrieve data for interpolation
        rainfall = storm.rain['Total'].values
        lat = storm.rain['Lat'].values
        lon = storm.rain['Lon'].values

        # Perform the interpolation
        grid = griddata((lat, lon), rainfall,
                        (grid_lat[None, :], grid_lon[:, None]), method=method)
        grid = np.transpose(grid)

        # Return data
        if return_xarray:
            return xr.DataArray(grid, coords=[grid_lat, grid_lon], dims=['lat', 'lon'])
        else:
            return {
                'grid': grid,
                'lat': grid_lat,
                'lon': grid_lon
            }

    def plot_rain_grid(self, storm, grid, levels=None, cmap=None, domain="dynamic", plot_all_dots=False, ax=None, cartopy_proj=None, save_path=None, prop={}, map_prop={}):
        r"""
        Creates a plot of a storm track and its associated rainfall (gridded).

        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Storm object to be plotted.
        grid : dict or xarray.DataArray
            Output from `interpolate_to_grid()` to be plotted. Can also be any dict with "lat", "lon" and "grid" entries, or an xarray DataArray with "lat" and "lon" dimensions.
        levels : list or numpy.ndarray
            List of contour fill levels to plot the grid values, in inches. If none, this is automatically generated.
        cmap : colormap
            Colormap to use for contour filling rainfall. If none, this is automatically generated.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        plot_all_dots : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of storm track lines. Please refer to :ref:`options-prop` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Check if Storm object contains rainfall data
        try:
            storm.rain
        except:
            storm.rain = self.get_storm_rainfall(storm)

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = RainPlot()

        # Create cartopy projection
        if cartopy_proj is not None:
            self.plot_obj.proj = cartopy_proj
        elif max(storm['lon']) > 150 or min(storm['lon']) < -150:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Plot storm
        plot_ax = self.plot_obj.plot_storm(storm, grid, levels, cmap,
                                           domain, plot_all_dots, ax=ax,
                                           save_path=save_path, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def plot_rain(self, storm, ms=7.5, mec=None, mew=0.5, minimum_threshold=1.0, levels=None, cmap=None, domain="dynamic", plot_all_dots=False, ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of a storm track and its associated rainfall (individual dots).

        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Storm object to be plotted.
        ms : float or int
            Marker size for individual rainfall observations. Default is 7.5.
        mec : str or rgb tuple
            Marker edge color for dots. If none (default), none will be colored.
        mew : float or int
            Marker edge width for dots. If mec is specified, the default is 0.5.
        minimum_threshold : float or int
            Minimum threshold (in inches) to plot dots. Default is 1.00 inch.
        levels : list or numpy.ndarray
            List of levels, in inches, corresponding to the colormap used to color fill the observation dots. If none, this is automatically generated.
        cmap : colormap
            Colormap to use for color filling the observation dots. If none, this is automatically generated.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        plot_all_dots : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of storm track lines. Please refer to :ref:`options-prop` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Check if Storm object contains rainfall data
        try:
            storm.rain
        except:
            storm.rain = self.get_storm_rainfall(storm)

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = RainPlot()

        # Create cartopy projection
        if cartopy_proj is not None:
            self.plot_obj.proj = cartopy_proj
        elif max(storm['lon']) > 150 or min(storm['lon']) < -150:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Format plot settings
        plot_settings = {
            'ms': ms,
            'minimum_threshold': minimum_threshold,
            'mew': mew,
            'mec': mec
        }

        # Plot storm
        plot_ax = self.plot_obj.plot_storm(storm, plot_settings, levels,
                                           cmap, domain, plot_all_dots,
                                           ax=ax, save_path=save_path,
                                           prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax
