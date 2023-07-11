r"""Functionality for storing and analyzing an individual storm."""

import re
import numpy as np
import xarray as xr
import pandas as pd
import urllib
import warnings
from datetime import datetime as dt, timedelta
import requests
import copy

# Import internal scripts
from .plot import TrackPlot
from ..tornado import *
from ..recon import ReconDataset
from ..ships import Ships

# Import tools
from .tools import *
from ..utils import *

try:
    import zipfile
    import gzip
    from io import BytesIO
    import tarfile
except ImportError:
    warnings.warn(
        "Warning: The libraries necessary for online NHC forecast retrieval aren't available (gzip, io, tarfile).")

try:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


class Storm:

    r"""
    Initializes an instance of Storm, retrieved via ``TrackDataset.get_storm()``.

    Parameters
    ----------
    storm : dict
        Dict entry of the requested storm.

    Other Parameters
    ----------------
    stormTors : dict, optional
        Dict entry containing tornado data assicated with the storm. Populated directly from tropycal.tracks.TrackDataset.

    Returns
    -------
    Storm
        Instance of a Storm object.

    Notes
    -----
    A Storm object is retrieved from TrackDataset's ``get_storm()`` method. For example, if the dataset read in is the default North Atlantic and the desired storm is Hurricane Michael (2018), it would be retrieved as follows:

    .. code-block:: python

        from tropycal import tracks
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))

    Now Hurricane Michael's data is stored in the variable ``storm``, which is an instance of Storm and can access all of the methods and attributes of a Storm object.

    All the variables associated with a Storm object (e.g., lat, lon, time, vmax) can be accessed in two ways. The first is directly from the Storm object:

    >>> storm.lat
    array([17.8, 18.1, 18.4, 18.8, 19.1, 19.7, 20.2, 20.9, 21.7, 22.7, 23.7,
           24.6, 25.6, 26.6, 27.7, 29. , 30. , 30.2, 31.5, 32.8, 34.1, 35.6,
           36.5, 37.3, 39.1, 41.1, 43.1, 44.8, 46.4, 47.6, 48.4, 48.8, 48.6,
           47.5, 45.9, 44.4, 42.8, 41.2])

    The second is via ``storm.vars``, which returns a dictionary of the variables associated with the Storm object. This is also a quick way to access all of the variables associated with a Storm object:

    >>> variable_dict = storm.vars
    >>> lat = variable_dict['lat']
    >>> lon = variable_dict['lon']
    >>> print(variable_dict.keys())
    dict_keys(['time', 'extra_obs', 'special', 'type', 'lat', 'lon', 'vmax', 'mslp', 'wmo_basin'])

    Storm objects also have numerous attributes with information about the storm. ``storm.attrs`` returns a dictionary of the attributes for this Storm object.

    >>> print(storm.attrs)
    {'id': 'AL142018',
     'operational_id': 'AL142018',
     'name': 'MICHAEL',
     'year': 2018,
     'season': 2018,
     'basin': 'north_atlantic',
     'source_info': 'NHC Hurricane Database',
     'source': 'hurdat',
     'ace': 12.5,
     'realtime': False,
     'invest': False}


    """

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):

        # Label object
        summary = ["<tropycal.tracks.Storm>"]

        # Format keys for summary
        type_array = np.array(self.dict['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))[0]
        if self.invest and len(idx) == 0:
            idx = np.array([True for i in type_array])
        if len(idx) == 0:
            start_time = 'N/A'
            end_time = 'N/A'
            max_wind = 'N/A'
            min_mslp = 'N/A'
        else:
            time_tropical = np.array(self.dict['time'])[idx]
            start_time = time_tropical[0].strftime("%H00 UTC %d %B %Y")
            end_time = time_tropical[-1].strftime("%H00 UTC %d %B %Y")
            max_wind = 'N/A' if all_nan(np.array(self.dict['vmax'])[idx]) else int(
                np.nanmax(np.array(self.dict['vmax'])[idx]))
            min_mslp = 'N/A' if all_nan(np.array(self.dict['mslp'])[idx]) else int(
                np.nanmin(np.array(self.dict['mslp'])[idx]))
        summary_keys = {
            'Maximum Wind': f"{max_wind} knots",
            'Minimum Pressure': f"{min_mslp} hPa",
            'Start Time': start_time,
            'End Time': end_time,
        }

        # Format keys for coordinates
        variable_keys = {}
        for key in self.vars.keys():
            dtype = type(self.vars[key][0]).__name__
            dtype = dtype.replace("_", "")
            variable_keys[key] = f"({dtype}) [{self.vars[key][0]} .... {self.vars[key][-1]}]"

        # Add storm summary
        summary.append("Storm Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        # Add coordinates
        summary.append("\nVariables:")
        add_space = np.max([len(key) for key in variable_keys.keys()]) + 3
        for key in variable_keys.keys():
            key_name = key
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{variable_keys[key]}')

        # Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.attrs.keys()]) + 3
        for key in self.attrs.keys():
            key_name = key + ":"
            val = '%0.1f' % (
                self.attrs[key]) if key == 'ace' else self.attrs[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')

        return "\n".join(summary)

    def __init__(self, storm, stormTors=None):

        # Save the dict entry of the storm
        self.dict = storm

        # Add other attributes about the storm
        keys = self.dict.keys()
        self.attrs = {}
        self.vars = {}
        for key in keys:
            if key in ['realtime', 'invest', 'subset']:
                continue
            if not isinstance(self.dict[key], list) and not isinstance(self.dict[key], dict):
                self[key] = self.dict[key]
                self.attrs[key] = self.dict[key]
            if isinstance(self.dict[key], list) and not isinstance(self.dict[key], dict):
                self.vars[key] = np.array(self.dict[key])
                self[key] = np.array(self.dict[key])

        # Assign tornado data
        if stormTors is not None and isinstance(stormTors, dict):
            self.stormTors = stormTors['data']
            self.tornado_dist_thresh = stormTors['dist_thresh']
            self.attrs['Tornado Count'] = len(stormTors['data'])

        # Get Archer track data for this storm, if it exists
        try:
            self.get_archer()
        except:
            pass

        # Initialize recon dataset instance
        self.recon = ReconDataset(storm=self)

        # Determine if storm object was retrieved via realtime object
        if 'realtime' in keys and self.dict['realtime']:
            self.realtime = True
            self.attrs['realtime'] = True
        else:
            self.realtime = False
            self.attrs['realtime'] = False

        # Determine if storm object is an invest
        if 'invest' in keys and self.dict['invest']:
            self.invest = True
            self.attrs['invest'] = True
        else:
            self.invest = False
            self.attrs['invest'] = False
        
        # Determine if storm object is subset
        if 'subset' in keys and self.dict['subset']:
            self.subset = True
            self.attrs['subset'] = True
        else:
            self.subset = False
            self.attrs['subset'] = False

    def sel(self, time=None, lat=None, lon=None, vmax=None, mslp=None,
            dvmax_dt=None, dmslp_dt=None, stormtype=None, method='exact'):
        r"""
        Subset this storm by any of its parameters and return a new storm object.

        Parameters
        ----------
        time : datetime.datetime or list/tuple of datetimes
            Datetime object for single point, or list/tuple of start time and end time.
            Default is None, which returns all points
        lat : float/int or list/tuple of float/int
            Float/int for single point, or list/tuple of latitude bounds (S,N).
            None in either position of a tuple means it is boundless on that side.
        lon : float/int or list/tuple of float/int
            Float/int for single point, or list/tuple of longitude bounds (W,E).
            If either lat or lon is a tuple, the other can be None for no bounds.
            If either is a tuple, the other canNOT be a float/int.
        vmax : list/tuple of float/int
            list/tuple of vmax bounds (min,max).
            None in either position of a tuple means it is boundless on that side.
        mslp : list/tuple of float/int
            list/tuple of mslp bounds (min,max).
            None in either position of a tuple means it is boundless on that side.
        dvmax_dt : list/tuple of float/int
            list/tuple of vmax bounds (min,max). ONLY AVAILABLE AFTER INTERP.
            None in either position of a tuple means it is boundless on that side.
        dmslp_dt : list/tuple of float/int
            list/tuple of mslp bounds (min,max). ONLY AVAILABLE AFTER INTERP.
            None in either position of a tuple means it is boundless on that side.
        stormtype : list/tuple of str
            list/tuple of stormtypes (options: 'LO','EX','TD','SD','TS','SS','HU')
        method : str
            Applies for single point selection in time and lat/lon.
            'exact' requires a point to match exactly with the request. (default)
            'nearest' returns the nearest point to the request
            'floor' ONLY for time, returns the nearest point before the request
            'ceil' ONLY for time, returns the neartest point after the request

        Returns
        -------
        storm object
            A new storm object that satisfies the intersection of all subsetting.
        """

        # create copy of storm object
        new_dict = copy.deepcopy(self.dict)
        new_dict['subset'] = True
        NEW_STORM = Storm(new_dict)
        idx_final = np.arange(len(self.time))

        # apply time filter
        if time is None:
            idx = copy.copy(idx_final)

        elif isinstance(time, dt):
            time_diff = np.array([(time - i).total_seconds()
                                 for i in NEW_STORM.time])
            idx = np.abs(time_diff).argmin()
            if time_diff[idx] != 0:
                if method == 'exact':
                    msg = f'no exact match for {time}. Use different time or method.'
                    raise ValueError(msg)
                elif method == 'floor' and time_diff[idx] < 0:
                    idx += -1
                    if idx < 0:
                        msg = f'no points before {time}. Use different time or method.'
                        raise ValueError(msg)
                elif method == 'ceil' and time_diff[idx] > 0:
                    idx += 1
                    if idx >= len(time_diff):
                        msg = f'no points after {time}. Use different time or method.'
                        raise ValueError(msg)

        elif isinstance(time, (tuple, list)) and len(time) == 2:
            time0, time1 = time
            if time0 is None:
                time0 = min(NEW_STORM.time)
            elif not isinstance(time0, dt):
                msg = 'time bounds must be of type datetime.datetime or None.'
                raise TypeError(msg)
            if time1 is None:
                time1 = max(NEW_STORM.time)
            elif not isinstance(time1, dt):
                msg = 'time bounds must be of type datetime.datetime or None.'
                raise TypeError(msg)
            tmptimes = np.array(NEW_STORM.time)
            idx = np.where((tmptimes >= time0) & (tmptimes <= time1))[0]
            if len(idx) == 0:
                msg = f'no points between {time}. Use different time bounds.'
                raise ValueError(msg)

        else:
            msg = 'time must be of type datetime.datetime, tuple/list, or None.'
            raise TypeError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply lat/lon filter
        if lat is None and lon is None:
            idx = copy.copy(idx_final)

        elif is_number(lat) and is_number(lon):
            dist = np.array([great_circle((lat, lon), (x, y)).kilometers for x, y in zip(
                NEW_STORM.lon, NEW_STORM.lat)])
            idx = np.abs(dist).argmin()
            if dist[idx] != 0:
                if method == 'exact':
                    msg = f'no exact match for {lat}/{lon}. Use different location or method.'
                    raise ValueError(msg)
                elif method in ('floor', 'ceil'):
                    warnings.warn(
                        'floor and ceil do not apply to lat/lon filtering. Using nearest instead.')

        elif (isinstance(lat, (tuple, list)) and len(lat) == 2) or (isinstance(lon, (tuple, list)) and len(lon) == 2):
            if not isinstance(lat, (tuple, list)):
                lat = (None, None)
            if not isinstance(lon, (tuple, list)):
                lon = (None, None)
            lat0, lat1 = lat
            lon0, lon1 = lon
            if lat0 is None:
                lat0 = min(NEW_STORM.lat)
            elif not is_number(lat0):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lat1 is None:
                lat1 = max(NEW_STORM.lat)
            elif not is_number(lat1):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lon0 is None:
                lon0 = min(NEW_STORM.lon)
            elif not is_number(lon0):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lon1 is None:
                lon1 = max(NEW_STORM.lon)
            elif not is_number(lon1):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)

            tmplat, tmplon = np.array(
                NEW_STORM.lat), np.array(NEW_STORM.lon) % 360
            idx = np.where((tmplat >= lat0) & (tmplat <= lat1) &
                           (tmplon >= lon0 % 360) & (tmplon <= lon1 % 360))[0]
            if len(idx) == 0:
                msg = f'no points in {lat}/{lon} box. Use different lat/lon bounds.'
                raise ValueError(msg)

        else:
            msg = 'lat and lon must be of the same type: float/int, tuple/list, or None.'
            raise TypeError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply vmax filter
        if vmax is None:
            idx = copy.copy(idx_final)

        elif isinstance(vmax, (tuple, list)) and len(vmax) == 2:
            vmax0, vmax1 = vmax
            if vmax0 is None:
                vmax0 = np.nanmin(NEW_STORM.vmax)
            elif not is_number(vmax0):
                msg = 'vmax bounds must be of type float/int or None.'
                raise TypeError(msg)
            if vmax1 is None:
                vmax1 = np.nanmax(NEW_STORM.vmax)
            elif not is_number(vmax1):
                msg = 'vmax bounds must be of type float/int or None.'
                raise TypeError(msg)
            tmpvmax = np.array(NEW_STORM.vmax)
            idx = np.where((tmpvmax >= vmax0) & (tmpvmax <= vmax1))[0]
            if len(idx) == 0:
                msg = f'no points with vmax between {vmax}. Use different vmax bounds.'
                raise ValueError(msg)

        else:
            msg = 'vmax must be of type tuple/list, or None.'
            raise TypeError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply mslp filter
        if mslp is None:
            idx = copy.copy(idx_final)

        elif isinstance(mslp, (tuple, list)) and len(mslp) == 2:
            mslp0, mslp1 = mslp
            if mslp0 is None:
                mslp0 = np.nanmin(NEW_STORM.mslp)
            elif not is_number(mslp0):
                msg = 'mslp bounds must be of type float/int or None.'
                raise TypeError(msg)
            if mslp1 is None:
                mslp1 = np.nanmax(NEW_STORM.mslp)
            elif not is_number(mslp1):
                msg = 'mslp bounds must be of type float/int or None.'
                raise TypeError(msg)
            tmpmslp = np.array(NEW_STORM.mslp)
            idx = np.where((tmpmslp >= mslp0) & (tmpmslp <= mslp1))[0]
            if len(idx) == 0:
                msg = f'no points with mslp between {mslp}. Use different dmslp_dt bounds.'
                raise ValueError(msg)

        else:
            msg = 'vmax must be of type tuple/list, or None.'
            raise TypeError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply dvmax_dt filter
        if dvmax_dt is None:
            idx = copy.copy(idx_final)

        elif 'dvmax_dt' not in NEW_STORM.dict.keys():
            msg = 'dvmax_dt not in storm data. Create new object with interp first.'
            raise KeyError(msg)

        elif isinstance(dvmax_dt, (tuple, list)) and len(dvmax_dt) == 2:
            dvmax_dt0, dvmax_dt1 = dvmax_dt
            if dvmax_dt0 is None:
                dvmax_dt0 = np.nanmin(NEW_STORM.dvmax_dt)
            elif not is_number(dvmax_dt0):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)
            if dvmax_dt1 is None:
                dvmax_dt1 = np.nanmax(NEW_STORM.dvmax_dt)
            elif not is_number(dvmax_dt1):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)

            tmpvmax = np.array(NEW_STORM.dvmax_dt)
            idx = np.where((tmpvmax >= dvmax_dt0) & (tmpvmax <= dvmax_dt1))[0]
            if len(idx) == 0:
                msg = f'no points with dvmax_dt between {dvmax_dt}. Use different dvmax_dt bounds.'
                raise ValueError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply dmslp_dt filter
        if dmslp_dt is None:
            idx = copy.copy(idx_final)

        elif 'dmslp_dt' not in NEW_STORM.dict.keys():
            msg = 'dmslp_dt not in storm data. Create new object with interp first.'
            raise KeyError(msg)

        elif isinstance(dmslp_dt, (tuple, list)) and len(dmslp_dt) == 2:
            dmslp_dt0, dmslp_dt1 = dmslp_dt
            if dmslp_dt0 is None:
                dmslp_dt0 = np.nanmin(NEW_STORM.dmslp_dt)
            elif not is_number(dmslp_dt0):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)
            if dmslp_dt1 is None:
                dmslp_dt1 = np.nanmax(NEW_STORM.dmslp_dt)
            elif not is_number(dmslp_dt1):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)
            tmpmslp = np.array(NEW_STORM.dmslp_dt)
            idx = np.where((tmpmslp >= dmslp_dt0) & (tmpmslp <= dmslp_dt1))[0]
            if len(idx) == 0:
                msg = f'no points with dmslp_dt between {dmslp_dt}. Use different dmslp_dt bounds.'
                raise ValueError(msg)

        # update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        # apply stormtype filter
        if stormtype is None:
            idx = copy.copy(idx_final)

        elif isinstance(stormtype, (tuple, list, str)):
            idx = [i for i, j in enumerate(
                NEW_STORM.type) if j in listify(stormtype)]
            if len(idx) == 0:
                msg = f'no points with type {stormtype}. Use different stormtype.'
                raise ValueError(msg)

        else:
            msg = 'stormtype must be of type tuple/list, str, or None.'
            raise TypeError(msg)

        # update idx_final
        idx_final = sorted(list(set(idx_final) & set(listify(idx))))

        # Construct new storm dict with subset elements
        for key in NEW_STORM.dict.keys():
            if isinstance(NEW_STORM.dict[key], list):
                NEW_STORM.dict[key] = [NEW_STORM.dict[key][i]
                                       for i in idx_final]
            else:
                NEW_STORM.dict[key] = NEW_STORM.dict[key]

            # Add other attributes to new storm object
            if key == 'realtime':
                continue
            if not isinstance(NEW_STORM.dict[key], list) and not isinstance(NEW_STORM.dict[key], dict):
                NEW_STORM[key] = NEW_STORM.dict[key]
                NEW_STORM.attrs[key] = NEW_STORM.dict[key]
            if isinstance(NEW_STORM.dict[key], list) and not isinstance(NEW_STORM.dict[key], dict):
                NEW_STORM.vars[key] = np.array(NEW_STORM.dict[key])
                NEW_STORM[key] = np.array(NEW_STORM.dict[key])

        return NEW_STORM

    def interp(self, hours=1, dt_window=24, dt_align='middle', method='linear'):
        r"""
        Interpolate a storm temporally to a specified time resolution.

        Parameters
        ----------
        hours : int or float
            Temporal resolution in hours (or fraction of an hour) to interpolate storm data to. Default is 1 hour.
        dt_window : int
            Time window in hours over which to calculate temporal change data. Default is 24 hours.
        dt_align : str
            Whether to align the temporal change window as "start", "middle" (default) or "end" of the dt_window time period.
        method : str
            Interpolation method for lat/lon coordinates passed to scipy. Options are "linear" (default) or "quadratic".

        Returns
        -------
        tropycal.tracks.Storm
            New Storm object containing the updated dictionary.

        Notes
        -----
        When interpolating data using a non-linear method, all non-standard hour observations (i.e., not within 00, 06, 12 or 18 UTC) are ignored for latitude & longitude interpolation in order to produce a smoother line.
        """

        NEW_STORM = copy.deepcopy(self)
        newdict = interp_storm(NEW_STORM.dict, hours,
                               dt_window, dt_align, method)
        for key in newdict.keys():
            NEW_STORM.dict[key] = newdict[key]

        # Add other attributes to new storm object
        for key in NEW_STORM.dict.keys():
            if key == 'realtime':
                continue
            if not isinstance(NEW_STORM.dict[key], (np.ndarray, list)) and not isinstance(NEW_STORM.dict[key], dict):
                NEW_STORM[key] = NEW_STORM.dict[key]
                NEW_STORM.attrs[key] = NEW_STORM.dict[key]
            if isinstance(NEW_STORM.dict[key], (np.ndarray, list)) and not isinstance(NEW_STORM.dict[key], dict):
                NEW_STORM.dict[key] = list(NEW_STORM.dict[key])
                NEW_STORM.vars[key] = np.array(NEW_STORM.dict[key])
                NEW_STORM[key] = np.array(NEW_STORM.dict[key])

        return NEW_STORM

    def to_dict(self):
        r"""
        Returns the dict entry for the storm.

        Returns
        -------
        dict
            A dictionary containing information about the storm.
        """

        # Return dict
        return self.dict

    def to_xarray(self):
        r"""
        Converts the storm dict into an xarray Dataset object.

        Returns
        -------
        xarray.Dataset
            An xarray Dataset object containing information about the storm.
        """

        # Set up empty dict for dataset
        time = self.dict['time']
        ds = {}
        attrs = {}

        # Add every key containing a list into the dict, otherwise add as an attribute
        keys = [k for k in self.dict.keys() if k != 'time']
        for key in keys:
            if isinstance(self.dict[key], list):
                ds[key] = xr.DataArray(self.dict[key], coords=[
                                       time], dims=['time'])
            else:
                attrs[key] = self.dict[key]

        # Convert entire dict to a Dataset
        ds = xr.Dataset(ds, attrs=attrs)

        # Return dataset
        return ds

    def to_dataframe(self, attrs_as_columns=False):
        r"""
        Converts the storm dict into a pandas DataFrame object.

        Parameters
        ----------
        attrs_as_columns : bool
            If True, adds Storm object attributes as columns in the DataFrame returned. Default is False.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame object containing information about the storm.
        """

        # Set up empty dict for dataframe
        ds = {}

        # Add every key containing a list into the dict
        keys = [k for k in self.dict.keys()]
        for key in keys:
            if isinstance(self.dict[key], list):
                ds[key] = self.dict[key]
            else:
                if attrs_as_columns:
                    ds[key] = self.dict[key]

        # Convert entire dict to a DataFrame
        ds = pd.DataFrame(ds)

        # Return dataset
        return ds

    def plot(self, domain="dynamic", plot_all_dots=False, ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of the observed track of the storm.

        Parameters
        ----------
        domain : str
            Domain for the plot. Default is "dynamic". "dynamic_tropical" is also available. Please refer to :ref:`options-domain` for available domain options.
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

        # Retrieve kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()

        # Create cartopy projection
        if cartopy_proj is not None:
            self.plot_obj.proj = cartopy_proj
        elif max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Plot storm
        plot_ax = self.plot_obj.plot_storms(
            [self.dict], domain, plot_all_dots=plot_all_dots, ax=ax, prop=prop, map_prop=map_prop, save_path=save_path)

        # Return axis
        return plot_ax

    def plot_nhc_forecast(self, forecast, track_labels='fhr', cone_days=5, domain="dynamic_forecast",
                          ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of the operational NHC forecast track along with observed track data.

        Parameters
        ----------
        forecast : int or datetime.datetime
            Integer representing the forecast number, or datetime object for the closest issued forecast to this time.
        track_labels : str
            Label forecast hours with the following methods:

            * **""** = no label
            * **"fhr"** = forecast hour (default)
            * **"valid_utc"** = UTC valid time
            * **"valid_edt"** = EDT valid time
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        domain : str
            Domain for the plot. Default is "dynamic_forecast". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of NHC forecast plot. Please refer to :ref:`options-prop-nhc` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Retrieve kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError(
                "Error: NHC data can only be accessed when HURDAT is used as the data source.")

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            if max(self.dict['lon']) > 140 or min(self.dict['lon']) < -140:
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)

        # Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()

        # Get all NHC forecast entries
        nhc_forecasts = self.forecast_dict['OFCL']
        carq_forecasts = self.forecast_dict['CARQ']

        # Get list of all NHC forecast initializations
        nhc_forecast_init = [k for k in nhc_forecasts.keys()]
        carq_forecast_init = [k for k in carq_forecasts.keys()]

        # Find closest matching time to the provided forecast date, or time
        if isinstance(forecast, int):
            forecast_dict = nhc_forecasts[nhc_forecast_init[forecast - 1]]
            advisory_num = forecast + 0
        elif isinstance(forecast, dt):
            nhc_forecast_init_dt = [dt.strptime(
                k, '%Y%m%d%H') for k in nhc_forecast_init]
            time_diff = np.array(
                [(i - forecast).days + (i - forecast).seconds / 86400 for i in nhc_forecast_init_dt])
            closest_idx = np.abs(time_diff).argmin()
            forecast_dict = nhc_forecasts[nhc_forecast_init[closest_idx]]
            advisory_num = closest_idx + 1
            if np.abs(time_diff[closest_idx]) >= 1.0:
                warnings.warn(
                    "The time provided is outside of the duration of the storm. Returning the closest available NHC forecast.")
        else:
            raise RuntimeError(
                "Error: Input variable 'forecast' must be of type 'int' or 'datetime.datetime'")

        # Get observed track as per NHC analyses
        track_dict = {
            'lat': [],
            'lon': [],
            'vmax': [],
            'type': [],
            'mslp': [],
            'time': [],
            'extra_obs': [],
            'special': [],
            'ace': 0.0,
        }
        use_carq = True
        for k in nhc_forecast_init:
            hrs = nhc_forecasts[k]['fhr']
            hrs_carq = carq_forecasts[k]['fhr'] if k in carq_forecast_init else [
            ]

            # Account for old years when hour 0 wasn't included directly
            # if 0 not in hrs and k in carq_forecast_init and 0 in hrs_carq:
            if self.dict['year'] < 2000 and k in carq_forecast_init and 0 in hrs_carq:

                use_carq = True
                hr_idx = hrs_carq.index(0)
                track_dict['lat'].append(carq_forecasts[k]['lat'][hr_idx])
                track_dict['lon'].append(carq_forecasts[k]['lon'][hr_idx])
                track_dict['vmax'].append(carq_forecasts[k]['vmax'][hr_idx])
                track_dict['mslp'].append(np.nan)
                track_dict['time'].append(carq_forecasts[k]['init'])

                itype = carq_forecasts[k]['type'][hr_idx]
                if itype == "":
                    itype = get_storm_type(carq_forecasts[k]['vmax'][0], False)
                track_dict['type'].append(itype)

                hr = carq_forecasts[k]['init'].strftime("%H%M")
                track_dict['extra_obs'].append(0) if hr in [
                    '0300', '0900', '1500', '2100'] else track_dict['extra_obs'].append(1)
                track_dict['special'].append("")

            else:
                use_carq = False
                if 3 in hrs:
                    hr_idx = hrs.index(3)
                    hr_add = 3
                else:
                    hr_idx = 0
                    hr_add = 0
                track_dict['lat'].append(nhc_forecasts[k]['lat'][hr_idx])
                track_dict['lon'].append(nhc_forecasts[k]['lon'][hr_idx])
                track_dict['vmax'].append(nhc_forecasts[k]['vmax'][hr_idx])
                track_dict['mslp'].append(np.nan)
                track_dict['time'].append(
                    nhc_forecasts[k]['init'] + timedelta(hours=hr_add))

                itype = nhc_forecasts[k]['type'][hr_idx]
                if itype == "":
                    itype = get_storm_type(nhc_forecasts[k]['vmax'][0], False)
                track_dict['type'].append(itype)

                hr = nhc_forecasts[k]['init'].strftime("%H%M")
                track_dict['extra_obs'].append(0) if hr in [
                    '0300', '0900', '1500', '2100'] else track_dict['extra_obs'].append(1)
                track_dict['special'].append("")

        # Add main elements from storm dict
        for key in ['id', 'operational_id', 'name', 'year']:
            track_dict[key] = self.dict[key]

        # Add carq to forecast dict as hour 0, if available
        if use_carq and forecast_dict['init'] in track_dict['time']:
            insert_idx = track_dict['time'].index(forecast_dict['init'])
            if 0 in forecast_dict['fhr']:
                forecast_dict['lat'][0] = track_dict['lat'][insert_idx]
                forecast_dict['lon'][0] = track_dict['lon'][insert_idx]
                forecast_dict['vmax'][0] = track_dict['vmax'][insert_idx]
                forecast_dict['mslp'][0] = track_dict['mslp'][insert_idx]
                forecast_dict['type'][0] = track_dict['type'][insert_idx]
            else:
                forecast_dict['fhr'].insert(0, 0)
                forecast_dict['lat'].insert(0, track_dict['lat'][insert_idx])
                forecast_dict['lon'].insert(0, track_dict['lon'][insert_idx])
                forecast_dict['vmax'].insert(0, track_dict['vmax'][insert_idx])
                forecast_dict['mslp'].insert(0, track_dict['mslp'][insert_idx])
                forecast_dict['type'].insert(0, track_dict['type'][insert_idx])

        # Add other info to forecast dict
        forecast_dict['advisory_num'] = advisory_num
        forecast_dict['basin'] = self.basin

        # Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(
            forecast_dict, track_dict, track_labels, cone_days, domain, ax=ax, save_path=save_path, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def plot_models(self, forecast=None, plot_btk=False, domain="dynamic", ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of operational model forecast tracks.

        Parameters
        ----------
        forecast : datetime.datetime, optional
            Datetime object representing the forecast initialization. If None (default), fetches the latest forecast.
        plot_btk : bool, optional
            If True, Best Track will be plotted alongside operational forecast models. Default is False.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        models : dict
            Dictionary with **key** = model name (case-insensitive) and **value** = model color. Scroll below for available model names.
        prop : dict
            Customization properties of forecast lines. Scroll below for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        .. note::
            1. For years before the HMON model was available, the HMON key instead defaults to the old GFDL model.

            2. For storms in the JTWC area of responsibility, the NHC key defaults to JTWC.

        The following model names are available as keys in the "model" dict. These names are case-insensitive. To avoid plotting any of these models, set the value to None instead of a color (e.g., ``models = {'gfs':None}`` or ``models = {'GFS':None}``).

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Model Acronym
             - Full Model Name
           * - CMC
             - Canadian Meteorological Centre (CMC)
           * - GFS
             - Global Forecast System (GFS)
           * - UKM
             - UK Met Office (UKMET)
           * - ECM
             - European Centre for Medium-range Weather Forecasts (ECMWF)
           * - HMON
             - Hurricanes in a Multi-scale Ocean-coupled Non-hydrostatic Model (HMON)
           * - HWRF
             - Hurricane Weather Research and Forecast (HWRF)
           * - HAFSA
             - Hurricane Analysis and Forecast System A (HAFS-A)
           * - HAFSB
             - Hurricane Analysis and Forecast System B (HAFS-B)
           * - NHC
             - National Hurricane Center (NHC)

        The following properties are available for customizing forecast model tracks, via ``prop``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - linewidth
             - Line width of forecast model track. Default is 2.5.
           * - marker
             - Marker type for forecast hours. Options are 'label' (default), 'dot' or None.
           * - marker_hours
             - List of forecast hours to mark. Default is [24,48,72,96,120,144,168].
        """

        # Dictionary mapping model names to the interpolated model key
        dict_models = {
            'cmc': 'CMC2',
            'gfs': 'AVNI',
            'ukm': 'UKX2',
            'ecm': 'ECO2',
            'hmon': 'HMNI',
            'hwrf': 'HWFI',
            'hafsa': 'HFAI',
            'hafsb': 'HFBI',
            'nhc': 'OFCI',
        }
        backup_models = {
            'gfs': ['AVNO', 'AVNX'],
            'ukm': ['UKM2', 'UKM'],
            'cmc': ['CMC'],
            'hmon': ['GFDI', 'GFDL'],
            'nhc': ['OFCL', 'JTWC'],
            'hwrf': ['HWRF'],
            'hafsa': ['HFSA'],
            'hafsb': ['HFSB'],
        }

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        models = kwargs.pop('models', {})
        map_prop = kwargs.pop('map_prop', {})

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()

        # -------------------------------------------------------------------------

        # Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()

        # Fetch latest forecast if None
        if forecast is None:
            check_keys = ['AVNI', 'OFCI', 'HWFI', 'HFAI']
            if 'HWFI' not in self.forecast_dict.keys():
                check_keys[2] = 'HWRF'
            if 'HWRF' not in self.forecast_dict.keys() and 'HWRF' in check_keys:
                check_keys.pop(check_keys.index('HWRF'))
            if 'OFCI' not in self.forecast_dict.keys():
                check_keys[1] = 'OFCL'
            if 'OFCL' not in self.forecast_dict.keys():
                check_keys[1] = 'JTWC'
            if 'JTWC' not in self.forecast_dict.keys() and 'JTWC' in check_keys:
                check_keys.pop(check_keys.index('JTWC'))
            if 'AVNI' not in self.forecast_dict.keys():
                check_keys[0] = 'AVNO'
            if 'AVNO' not in self.forecast_dict.keys():
                check_keys[0] = 'AVNX'
            if 'AVNX' not in self.forecast_dict.keys() and 'AVNX' in check_keys:
                check_keys.pop(check_keys.index('AVNX'))
            if 'HFAI' not in self.forecast_dict.keys():
                check_keys[3] = 'HFSA'
            if 'HFSA' not in self.forecast_dict.keys() and 'HWRF' in check_keys:
                check_keys.pop(check_keys.index('HFSA'))
            if len(check_keys) == 0:
                raise ValueError("No models are available for this storm.")
            inits = [dt.strptime(
                [k for k in self.forecast_dict[key]][-1], '%Y%m%d%H') for key in check_keys]
            forecast = min(inits)

        # Error check forecast time
        if forecast < self.time[0] or forecast > self.time[-1]:
            raise ValueError(
                "Requested forecast is outside of the storm's duration.")

        # Construct forecast dict
        ds = {}
        proj_lons = []
        forecast_str = forecast.strftime('%Y%m%d%H')
        input_keys = [k for k in models.keys()]
        input_keys_lower = [k.lower() for k in models.keys()]
        for key in dict_models.keys():

            # Only proceed if model isn't not requested
            if key in input_keys_lower:
                idx = input_keys_lower.index(key)
                if models[input_keys[idx]] is None:
                    continue

            # Find official key
            official_key = dict_models[key]
            found = False
            if official_key not in self.forecast_dict.keys():
                if key in backup_models.keys():
                    for backup_key in backup_models[key]:
                        if backup_key in self.forecast_dict.keys():
                            official_key = backup_key
                            found = True
                            break
            else:
                found = True

            # Check for 2 vs. I if needed
            if not found or forecast_str not in self.forecast_dict[official_key].keys():
                if '2' in official_key:
                    official_key = dict_models[key].replace('2', 'I')
                    if official_key not in self.forecast_dict.keys():
                        if key in backup_models.keys():
                            found = False
                            for backup_key_iter in backup_models[key]:
                                backup_key = backup_key_iter.replace('2', 'I')
                                if backup_key in self.forecast_dict.keys():
                                    official_key = backup_key
                                    found = True
                                    break
                            if not found:
                                continue
                        else:
                            continue
                else:
                    continue

            # Append forecast data if it exists for this initialization
            if forecast_str not in self.forecast_dict[official_key].keys():
                continue
            enter_key = key + ''
            if key.lower() == 'hmon' and 'gf' in official_key.lower():
                enter_key = 'gfdl'
            if key.lower() == 'nhc' and 'jt' in official_key.lower():
                enter_key = 'jtwc'
            ds[enter_key] = copy.deepcopy(
                self.forecast_dict[official_key][forecast_str])

            # Filter out to hour 168
            if ds[enter_key]['fhr'][-1] > 168:
                idx = ds[enter_key]['fhr'].index(168)
                for key in ds[enter_key].keys():
                    if isinstance(ds[enter_key][key], list):
                        ds[enter_key][key] = ds[enter_key][key][:idx + 1]

            proj_lons += ds[enter_key]['lon']

        # Proceed if data exists
        if len(ds) == 0:
            raise RuntimeError(
                "No forecasts are available for the given parameters.")

        # Create cartopy projection
        if cartopy_proj is not None:
            self.plot_obj.proj = cartopy_proj
        elif np.nanmax(proj_lons) > 150 or np.nanmin(proj_lons) < -150:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Account for cases crossing dateline
        if np.nanmax(proj_lons) > 150 or np.nanmin(proj_lons) < -150:
            for key in ds.keys():
                new_lons = np.array(ds[key]['lon'])
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                ds[key]['lon'] = new_lons.tolist()

        # Plot storm
        plot_ax = self.plot_obj.plot_models(
            forecast, plot_btk, self.dict, ds, models, domain, ax=ax, prop=prop, map_prop=map_prop, save_path=save_path)

        # Return axis
        return plot_ax

    def plot_ensembles(self, forecast=None, fhr=None, interpolate=True, domain="dynamic", ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of individual GEFS ensemble tracks.

        Parameters
        ----------
        forecast : datetime.datetime, optional
            Datetime object representing the GEFS run initialization. If None (default), fetches the latest run.
        fhr : int, optional
            Forecast hour to plot. If None (default), a cumulative plot of all forecast hours will be produced. If an integer, a single plot will be produced.
        interpolate : bool, optional
            If True, and fhr is None, track density data will be interpolated to hourly. Default is True (1-hourly track density data). False plots density using 6-hourly track data.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        prop_members : dict
            Customization properties of GEFS ensemble member track lines. Scroll down below for available options.
        prop_mean : dict
            Customization properties of GEFS ensemble mean track. Scroll down below for available options.
        prop_gfs : dict
            Customization properties of GFS forecast track. Scroll down below for available options.
        prop_btk : dict
            Customization properties of Best Track line. Scroll down below for available options.
        prop_ellipse : dict
            Customization properties of GEFS ensemble ellipse. Scroll down below for available options.
        prop_density : dict
            Customization properties of GEFS ensemble track density. Scroll down below for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        .. note::
            The total number of GEFS members available for analysis is as follows:

            * **2020 - present** - 31 members
            * **2006 - 2019** - 21 members
            * **2005 & back** - 5 members

            As the density plot and ensemble ellipse require a minimum of 10 ensemble members, they will not be generated for storms from 2005 and earlier.

            Additionally, ellipses are not generated if using the default ``fhr=None``, meaning a cumulative track density plot is generated instead.

        The ensemble ellipse used in this function follows the methodology of `Hamill et al. (2011)`_, denoting the spread in ensemble member cyclone positions. The size of the ellipse is calculated to contain 90% of ensemble members at any given time. This ellipse can be used to determine the primary type of ensemble variability:

        * **Along-track variability** - if the major axis of the ellipse is parallel to the ensemble mean motion vector.
        * **Across-track variability** - if the major axis of the ellipse is normal to the ensemble mean motion vector.

        .. _Hamill et al. (2011): https://doi.org/10.1175/2010MWR3456.1

        The following properties are available for customizing ensemble member tracks, via ``prop_members``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot ensemble member tracks. Default is True.
           * - linewidth
             - Forecast track linewidth. Default is 0.2.
           * - linecolor
             - Forecast track line color. Default is black.
           * - color_var
             - Variable name to color ensemble members by ('vmax' or 'mslp'). Default is None.
           * - cmap
             - If ``color_var`` is specified, matplotlib colormap to color the variable by.
           * - levels
             - If ``color_var`` is specified, list of contour levels to color the variable by.

        The following properties are available for customizing ensemble mean track, via ``prop_mean``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot ensemble mean forecast track. Default is True.
           * - linewidth
             - Forecast track linewidth. Default is 3.0.
           * - linecolor
             - Forecast track line color. Default is black.

        The following properties are available for customizing GFS forecast track, via ``prop_gfs``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot GFS forecast track. Default is True.
           * - linewidth
             - Forecast track linewidth. Default is 3.0.
           * - linecolor
             - Forecast track line color. Default is red.

        The following properties are available for customizing Best Track line, via ``prop_btk``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot Best Track line. Default is True.
           * - linewidth
             - Best Track linewidth. Default is 2.5.
           * - linecolor
             - Best Track line color. Default is blue.

        The following properties are available for customizing the ensemble ellipse plot, via ``prop_ellipse``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot ensemble member ellipse. Default is True.
           * - linewidth
             - Ellipse linewidth. Default is 3.0.
           * - linecolor
             - Ellipse line color. Default is blue.

        The following properties are available for customizing ensemble member track density, via ``prop_density``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - plot
             - Boolean to determine whether to plot ensemble member track density. Default is True.
           * - radius
             - Radius (in km) for which to calculate track density. Default is 200 km.
           * - cmap
             - Matplotlib colormap for track density plot. Default is "plasma_r".
           * - levels
             - List of levels for contour filling track density.

        """

        # Pop kwargs
        prop_members = kwargs.pop('prop_members', {})
        prop_mean = kwargs.pop('prop_mean', {})
        prop_gfs = kwargs.pop('prop_gfs', {})
        prop_btk = kwargs.pop('prop_btk', {})
        prop_ellipse = kwargs.pop('prop_ellipse', {})
        prop_density = kwargs.pop('prop_density', {})
        map_prop = kwargs.pop('map_prop', {})

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()

        # -------------------------------------------------------------------------

        # Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()

        # Fetch latest forecast if None
        if forecast is None:
            inits = []
            for key in ['AC00', 'AP01', 'AP02', 'AP03', 'AP04', 'AP05']:
                if key in self.forecast_dict.keys():
                    inits.append(dt.strptime(
                        [k for k in self.forecast_dict[key]][-1], '%Y%m%d%H'))
            if len(inits) > 0:
                forecast = min(inits)
            else:
                raise RuntimeError(
                    "Error: Could not determine the latest available GEFS forecast.")

        # Determine max members by year
        nens = 21
        if self.year >= 2020 and ('AP21' in self.forecast_dict.keys() or 'AP22' in self.forecast_dict.keys() or 'AP23' in self.forecast_dict.keys()):
            nens = 31

        # Enforce fhr type
        if isinstance(fhr, list):
            fhr = fhr[0]

        # If this forecast init was recently used, don't re-calculate
        init_used = False
        try:
            if self.gefs_init == forecast:
                init_used = True
        except:
            pass

        # Only calculate if needed to
        if not init_used:

            print("--> Starting to calculate ellipse data")

            # Create dict to store all data in
            ds = {'gfs': {'fhr': [], 'lat': [], 'lon': [], 'vmax': [], 'mslp': [], 'time': []},
                  'gefs': {'fhr': [], 'lat': [], 'lon': [], 'vmax': [], 'mslp': [], 'time': [],
                           'members': [], 'ellipse_lat': [], 'ellipse_lon': []}
                  }

            # String formatting for ensembles
            def str2(ens):
                if ens == 0:
                    return "AC00"
                if ens < 10:
                    return f"AP0{ens}"
                return f"AP{ens}"

            # Get GFS forecast entry (AVNX is valid for RAL a-deck source)
            gfs_key = 'AVNO' if 'AVNO' in self.forecast_dict.keys() else 'AVNX'
            try:
                forecast_gfs = self.forecast_dict[gfs_key][forecast.strftime(
                    "%Y%m%d%H")]
            except:
                raise RuntimeError(
                    "The requested GFS initialization isn't available for this storm.")

            # Enter into dict entry
            ds['gfs']['fhr'] = [int(i) for i in forecast_gfs['fhr']]
            ds['gfs']['lat'] = [np.round(i, 1) for i in forecast_gfs['lat']]
            ds['gfs']['lon'] = [np.round(i, 1) for i in forecast_gfs['lon']]
            ds['gfs']['vmax'] = [float(i) for i in forecast_gfs['vmax']]
            ds['gfs']['mslp'] = forecast_gfs['mslp']
            ds['gfs']['time'] = [forecast +
                                 timedelta(hours=i) for i in forecast_gfs['fhr']]

            # Retrieve GEFS ensemble data (30 members 2019-present, 20 members prior)
            for ens in range(0, nens):

                # Create dict entry
                ds[f'gefs_{ens}'] = {
                    'fhr': [],
                    'lat': [],
                    'lon': [],
                    'vmax': [],
                    'mslp': [],
                    'time': [],
                }

                # Retrieve ensemble member data
                ens_str = str2(ens)
                if ens_str not in self.forecast_dict.keys():
                    continue
                if forecast.strftime("%Y%m%d%H") not in self.forecast_dict[ens_str].keys():
                    continue
                forecast_ens = self.forecast_dict[ens_str][forecast.strftime(
                    "%Y%m%d%H")]

                # Enter into dict entry
                ds[f'gefs_{ens}']['fhr'] = [int(i)
                                            for i in forecast_ens['fhr']]
                ds[f'gefs_{ens}']['lat'] = [
                    np.round(i, 1) for i in forecast_ens['lat']]
                ds[f'gefs_{ens}']['lon'] = [
                    np.round(i, 1) for i in forecast_ens['lon']]
                ds[f'gefs_{ens}']['vmax'] = [
                    float(i) for i in forecast_ens['vmax']]
                ds[f'gefs_{ens}']['mslp'] = forecast_ens['mslp']
                ds[f'gefs_{ens}']['time'] = [forecast +
                                             timedelta(hours=i) for i in forecast_ens['fhr']]

            # Construct ensemble mean data
            # Iterate through all forecast hours
            for iter_fhr in range(0, 246, 6):

                # Temporary data arrays
                temp_data = {}
                for key in ds['gfs'].keys():
                    if key not in ['time', 'fhr']:
                        temp_data[key] = []

                # Iterate through ensemble member
                for ens in range(nens):

                    # Determine if member has data valid at this forecast hour
                    if iter_fhr in ds[f'gefs_{ens}']['fhr']:

                        # Retrieve index
                        idx = ds[f'gefs_{ens}']['fhr'].index(iter_fhr)

                        # Append data
                        for key in ds['gfs'].keys():
                            if key not in ['time', 'fhr']:
                                temp_data[key].append(
                                    ds[f'gefs_{ens}'][key][idx])

                # Proceed if 20 or more ensemble members
                if len(temp_data['lat']) >= 10:

                    # Append data
                    for key in ds['gfs'].keys():
                        if key not in ['time', 'fhr']:
                            ds['gefs'][key].append(np.nanmean(temp_data[key]))
                    ds['gefs']['fhr'].append(iter_fhr)
                    ds['gefs']['time'].append(
                        forecast + timedelta(hours=iter_fhr))
                    ds['gefs']['members'].append(len(temp_data['lat']))

                    # Calculate ellipse data
                    if prop_ellipse is not None:
                        try:
                            ellipse_data = calc_ensemble_ellipse(
                                temp_data['lon'], temp_data['lat'])
                            ds['gefs']['ellipse_lon'].append(
                                ellipse_data['ellipse_lon'])
                            ds['gefs']['ellipse_lat'].append(
                                ellipse_data['ellipse_lat'])
                        except:
                            ds['gefs']['ellipse_lon'].append([])
                            ds['gefs']['ellipse_lat'].append([])
                    else:
                        ds['gefs']['ellipse_lon'].append([])
                        ds['gefs']['ellipse_lat'].append([])

            # Save data for future use if needed
            self.gefs_init = forecast
            self.ds = ds

            print("--> Done calculating ellipse data")

        # Determine lon bounds for cartopy projection
        proj_lons = []
        for key in self.ds.keys():
            proj_lons += self.ds[key]['lon']
        if fhr is not None and fhr in self.ds['gefs']['fhr']:
            fhr_idx = self.ds['gefs']['fhr'].index(fhr)
            proj_lons += self.ds['gefs']['ellipse_lon'][fhr_idx]

        # Create cartopy projection
        if cartopy_proj is not None:
            self.plot_obj.proj = cartopy_proj
        elif np.nanmax(proj_lons) > 150 or np.nanmin(proj_lons) < -150:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Account for cases crossing dateline
        ds = copy.deepcopy(self.ds)
        if np.nanmax(proj_lons) > 150 or np.nanmin(proj_lons) < -150:
            for key in ds.keys():
                new_lons = np.array(ds[key]['lon'])
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                ds[key]['lon'] = new_lons.tolist()

            # Re-calculate GEFS mean
            for iter_hr in ds['gefs']['fhr']:
                fhr_idx = ds['gefs']['fhr'].index(iter_hr)
                ds['gefs']['lon'][fhr_idx] = np.nanmean([ds[f'gefs_{ens}']['lon'][ds[f'gefs_{ens}']['fhr'].index(
                    iter_hr)] for ens in range(nens) if iter_hr in ds[f'gefs_{ens}']['fhr']])

        # Plot storm
        plot_ax = self.plot_obj.plot_ensembles(forecast, self.dict, fhr, interpolate, prop_members, prop_mean,
                                               prop_gfs, prop_btk, prop_ellipse, prop_density, nens, domain,
                                               ds, ax=ax, map_prop=map_prop, save_path=save_path)

        # Return axis
        return plot_ax

    def list_nhc_discussions(self):
        r"""
        Retrieves a list of NHC forecast discussions for this storm, archived on https://ftp.nhc.noaa.gov/atcf/archive/.

        Returns
        -------
        dict
            Dictionary containing entries for each forecast discussion for this storm.
        """

        # Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError(
                "Error: NHC data can only be accessed when HURDAT is used as the data source.")

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']

        # Error check
        if storm_id == '':
            raise RuntimeError(
                "Error: This storm was identified post-operationally. No NHC operational data is available.")

        # Get list of available NHC advisories & map discussions
        if storm_year == (dt.now()).year:

            # Get list of all discussions for all storms this year
            try:
                url_disco = 'https://ftp.nhc.noaa.gov/atcf/dis/'
                page = requests.get(url_disco).text
                content = page.split("\n")
                files = []
                for line in content:
                    if ".discus." in line and self.id.lower() in line:
                        filename = line.split('">')[1]
                        filename = filename.split("</a>")[0]
                        files.append(filename)
                del content
            except:
                ftp = FTP('ftp.nhc.noaa.gov')
                ftp.login()
                ftp.cwd('atcf/dis')
                files = ftp.nlst()
                files = [
                    i for i in files if ".discus." in i and self.id.lower() in i]
                ftp.quit()

            # Read in all NHC forecast discussions
            discos = {
                'id': [],
                'utc_time': [],
                'url': [],
                'mode': 0,
            }
            for file in files:

                # Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])

                # Open file to get info about time issued
                f = urllib.request.urlopen(url_disco + file)
                content = f.read()
                content = content.decode("utf-8")
                content = content.split("\n")
                f.close()

                # Figure out time issued
                hr = content[5].split(" ")[0]
                zone = content[5].split(" ")[2]
                disco_time = num_to_str2(int(hr)) + \
                    ' '.join(content[5].split(" ")[1:])

                format_time = content[5].split(" ")[0]
                if len(format_time) == 3:
                    format_time = "0" + format_time
                format_time = format_time + " " + \
                    ' '.join(content[5].split(" ")[1:])
                disco_time = dt.strptime(
                    format_time, f'%I00 %p {zone} %a %b %d %Y')

                time_zones = {
                    'ADT': -3,
                    'AST': -4,
                    'EDT': -4,
                    'EST': -5,
                    'CDT': -5,
                    'CST': -6,
                    'MDT': -6,
                    'MST': -7,
                    'PDT': -7,
                    'PST': -8,
                    'HDT': -9,
                    'HST': -10}
                offset = time_zones.get(zone, 0)
                disco_time = disco_time + timedelta(hours=offset * -1)

                # Add times issued
                discos['id'].append(disco_number)
                discos['utc_time'].append(disco_time)
                discos['url'].append(url_disco + file)

        elif storm_year < 1992:
            raise RuntimeError("NHC discussion data is unavailable.")
        elif storm_year < 2000:
            # Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msg.zip'
            try:
                request = urllib.request.Request(url)
                response = urllib.request.urlopen(request)
                file_like_object = BytesIO(response.read())
                tar = zipfile.ZipFile(file_like_object)
            except:
                try:
                    url_disco = f"ftp://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                    url = url_disco + f'{storm_id.lower()}_msg.zip'
                    request = urllib.request.Request(url)
                    response = urllib.request.urlopen(request)
                    file_like_object = BytesIO(response.read())
                    tar = zipfile.ZipFile(file_like_object)
                except:
                    raise RuntimeError("NHC discussion data is unavailable.")

            # Get file list
            members = '\n'.join([i for i in tar.namelist()])
            nums = "[0123456789]"
            search_pattern = f'n{storm_id[0:4].lower()}{str(storm_year)[2:]}.[01]{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files:
                    files.append(file)  # remove duplicates

            # Read in all NHC forecast discussions
            discos = {
                'id': [],
                'utc_time': [],
                'url': [],
                'mode': 4,
            }
            for file in files:

                # Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[1])

                # Open file to get info about time issued
                members = tar.namelist()
                members_names = [i for i in members]
                idx = members_names.index(file)
                content = (tar.read(members[idx])).decode()
                content = content.split("\n")

                # Figure out time issued
                slice_idx = 5 if storm_year < 1998 else 4
                for temp_idx in [slice_idx, slice_idx - 1, slice_idx + 1, slice_idx - 2, slice_idx + 2]:
                    try:
                        hr = content[temp_idx].split(" ")[0]
                        if 'NOON' in content[temp_idx]:
                            temp_line = content[temp_idx].replace(
                                "NOON", "12 PM")
                            zone = temp_line.split(" ")[1]
                            disco_time = dt.strptime(
                                temp_line.rstrip(), f'%I %p {zone} %a %b %d %Y')
                        else:
                            zone = content[temp_idx].split(" ")[2]
                            disco_time = dt.strptime(
                                content[temp_idx].rstrip(), f'%I %p {zone} %a %b %d %Y')
                    except:
                        pass

                time_zones = {
                    'ADT': -3,
                    'AST': -4,
                    'EDT': -4,
                    'EST': -5,
                    'CDT': -5,
                    'CST': -6,
                    'MDT': -6,
                    'MST': -7,
                    'PDT': -7,
                    'PST': -8,
                    'HDT': -9,
                    'HST': -10}
                offset = time_zones.get(zone, 0)
                disco_time = disco_time + timedelta(hours=offset * -1)

                # Add times issued
                discos['id'].append(disco_number)
                discos['utc_time'].append(disco_time)
                discos['url'].append(file)

            response.close()
            tar.close()

        elif storm_year == 2000:
            # Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
            try:
                request = urllib.request.Request(url)
                response = urllib.request.urlopen(request)
                file_like_object = BytesIO(response.read())
                tar = tarfile.open(fileobj=file_like_object)
            except:
                try:
                    url_disco = f"ftp://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                    url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
                    request = urllib.request.Request(url)
                    response = urllib.request.urlopen(request)
                    file_like_object = BytesIO(response.read())
                    tar = tarfile.open(fileobj=file_like_object)
                except:
                    raise RuntimeError("NHC discussion data is unavailable.")

            # Get file list
            members = '\n'.join([i.name for i in tar.getmembers()])
            nums = "[0123456789]"
            search_pattern = f'N{storm_id[0:4]}{str(storm_year)[2:]}.[01]{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files:
                    files.append(file)  # remove duplicates

            # Read in all NHC forecast discussions
            discos = {
                'id': [],
                'utc_time': [],
                'url': [],
                'mode': 3,
            }
            for file in files:

                # Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[1])

                # Open file to get info about time issued
                members = tar.getmembers()
                members_names = [i.name for i in members]
                idx = members_names.index(file)
                f = tar.extractfile(members[idx])
                content = (f.read()).decode()
                f.close()
                content = content.split("\n")

                # Figure out time issued
                hr = content[4].split(" ")[0]
                zone = content[4].split(" ")[2]
                disco_time = num_to_str2(int(hr)) + \
                    ' '.join(content[4].split(" ")[1:])
                disco_time = dt.strptime(
                    content[4], f'%I %p {zone} %a %b %d %Y')

                time_zones = {
                    'ADT': -3,
                    'AST': -4,
                    'EDT': -4,
                    'EST': -5,
                    'CDT': -5,
                    'CST': -6,
                    'MDT': -6,
                    'MST': -7,
                    'PDT': -7,
                    'PST': -8,
                    'HDT': -9,
                    'HST': -10}
                offset = time_zones.get(zone, 0)
                disco_time = disco_time + timedelta(hours=offset * -1)

                # Add times issued
                discos['id'].append(disco_number)
                discos['utc_time'].append(disco_time)
                discos['url'].append(file)

            response.close()
            tar.close()

        elif storm_year in range(2001, 2006):
            # Get directory path of storm and read it in
            try:
                url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
                if storm_year < 2003:
                    url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
                request = urllib.request.Request(url)
                response = urllib.request.urlopen(request)
                file_like_object = BytesIO(response.read())
                tar = tarfile.open(fileobj=file_like_object)
            except:
                try:
                    url_disco = f"ftp://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                    url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
                    if storm_year < 2003:
                        url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
                    request = urllib.request.Request(url)
                    response = urllib.request.urlopen(request)
                    file_like_object = BytesIO(response.read())
                    tar = tarfile.open(fileobj=file_like_object)
                except:
                    raise RuntimeError("NHC discussion data is unavailable.")

            # Get file list
            members = '\n'.join([i.name for i in tar.getmembers()])
            nums = "[0123456789]"
            search_pattern = f'{storm_id.lower()}.discus.[01]{nums}{nums}.{nums}{nums}{nums}{nums}{nums}{nums}{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files:
                    files.append(file)  # remove duplicates
            response.close()
            tar.close()

            # Read in all NHC forecast discussions
            discos = {
                'id': [],
                'utc_time': [],
                'url': [],
                'mode': 1,
            }
            for file in files:

                # Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])
                disco_time = file_info[3]
                disco_year = storm_year
                if disco_time[0:2] == "01" and int(storm_id[2:4]) > 3:
                    disco_year = storm_year + 1
                disco_time = dt.strptime(
                    str(disco_year) + disco_time, '%Y%m%d%H%M')

                discos['id'].append(disco_number)
                discos['utc_time'].append(disco_time)
                discos['url'].append(file)

            if storm_year < 2003:
                discos['mode'] = 2

        else:
            # Retrieve list of NHC discussions for this storm
            try:
                url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                path_disco = urllib.request.urlopen(url_disco)
                string = path_disco.read().decode('utf-8')
                nums = "[0123456789]"
                search_pattern = f'{storm_id.lower()}.discus.[01]{nums}{nums}.{nums}{nums}{nums}{nums}{nums}{nums}{nums}{nums}'
                pattern = re.compile(search_pattern)
                filelist = pattern.findall(string)
                files = []
                for file in filelist:
                    if file not in files:
                        files.append(file)  # remove duplicates
                path_disco.close()
            except:
                try:
                    url_disco = f"ftp://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
                    path_disco = urllib.request.urlopen(url_disco)
                    string = path_disco.read().decode('utf-8')
                    nums = "[0123456789]"
                    search_pattern = f'{storm_id.lower()}.discus.[01]{nums}{nums}.{nums}{nums}{nums}{nums}{nums}{nums}{nums}{nums}'
                    pattern = re.compile(search_pattern)
                    filelist = pattern.findall(string)
                    files = []
                    for file in filelist:
                        if file not in files:
                            files.append(file)  # remove duplicates
                    path_disco.close()
                except:
                    raise RuntimeError("NHC discussion data is unavailable.")

            # Read in all NHC forecast discussions
            discos = {
                'id': [],
                'utc_time': [],
                'url': [],
                'mode': 0,
            }
            for file in files:

                # Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])
                disco_time = file_info[3]
                disco_year = storm_year
                if disco_time[0:2] == "01" and int(storm_id[2:4]) > 3:
                    disco_year = storm_year + 1
                disco_time = dt.strptime(
                    str(disco_year) + disco_time, '%Y%m%d%H%M')

                discos['id'].append(disco_number)
                discos['utc_time'].append(disco_time)
                discos['url'].append(url_disco + file)

        # Return dict entry
        try:
            discos
        except:
            raise RuntimeError("NHC discussion data is unavailable.")

        if len(discos['id']) == 0:
            raise RuntimeError("NHC discussion data is unavailable.")
        return discos

    def get_nhc_discussion(self, forecast, save_path=None):
        r"""
        Retrieves a single NHC forecast discussion.

        Parameters
        ----------
        forecast : datetime.datetime or int
            Datetime object representing the desired forecast discussion time (in UTC), or integer representing the forecast discussion ID. If -1 is passed, the latest forecast discussion is returned.
        save_path : str, optional
            Directory path to save the forecast discussion text to. If None (default), forecast won't be saved.

        Returns
        -------
        dict
            Dictionary containing the forecast discussion text and accompanying information about this discussion.
        """

        # Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            msg = "Error: NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']

        # Error check
        if storm_id == '':
            msg = "No NHC operational data is available for this storm."
            raise RuntimeError(msg)

        # Error check
        if not isinstance(forecast, int) and not isinstance(forecast, dt):
            msg = "forecast must be of type int or datetime.datetime"
            raise TypeError(msg)

        # Get list of storm discussions
        disco_dict = self.list_nhc_discussions()

        if isinstance(forecast, dt):
            # Find all discussion times
            disco_times = disco_dict['utc_time']
            disco_ids = [int(i) for i in disco_dict['id']]
            disco_diff = np.array(
                [(i - forecast).days + (i - forecast).seconds / 86400 for i in disco_times])
            # Find most recent discussion
            indices = np.argwhere(disco_diff <= 0)
            if len(indices) > 0:
                closest_idx = indices[-1][0]
            else:
                closest_idx = 0
            closest_diff = disco_diff[closest_idx]
            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]

            # Raise warning if difference is >=1 day
            if np.abs(closest_diff) >= 1.0:
                warnings.warn("The time provided is unavailable or outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion.")

        if isinstance(forecast, int):
            # Find closest discussion ID to the one provided
            disco_times = disco_dict['utc_time']
            disco_ids = [int(i) for i in disco_dict['id']]
            if forecast == -1:
                closest_idx = -1
            else:
                disco_diff = np.array([i - forecast for i in disco_ids])
                closest_idx = np.abs(disco_diff).argmin()
                closest_diff = disco_diff[closest_idx]

                # Raise warning if difference is >=1 ids
                if np.abs(closest_diff) >= 2.0:
                    msg = "The ID provided is unavailable or outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion."
                    warnings.warn(msg)

            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]

        # Read content of NHC forecast discussion
        if disco_dict['mode'] == 0:
            url_disco = disco_dict['url'][closest_idx]
            if requests.get(url_disco).status_code != 200:
                raise RuntimeError("NHC discussion data is unavailable.")
            f = urllib.request.urlopen(url_disco)
            content = f.read()
            content = content.decode("utf-8")
            f.close()

        elif disco_dict['mode'] in [1, 2, 3]:
            # Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
            if disco_dict['mode'] in [2, 3]:
                url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
            if requests.get(url).status_code != 200:
                raise RuntimeError("NHC discussion data is unavailable.")
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            file_like_object = BytesIO(response.read())
            tar = tarfile.open(fileobj=file_like_object)

            members = tar.getmembers()
            members_names = [i.name for i in members]
            idx = members_names.index(disco_dict['url'][closest_idx])
            f = tar.extractfile(members[idx])
            content = (f.read()).decode()
            f.close()
            tar.close()
            response.close()

        elif disco_dict['mode'] in [4]:
            # Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msg.zip'
            if requests.get(url).status_code != 200:
                raise RuntimeError("NHC discussion data is unavailable.")
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            file_like_object = BytesIO(response.read())
            tar = zipfile.ZipFile(file_like_object)

            members = tar.namelist()
            members_names = [i for i in members]
            idx = members_names.index(disco_dict['url'][closest_idx])
            content = (tar.read(members[idx])).decode()
            tar.close()
            response.close()

        # Save file, if specified
        if save_path is not None:
            closest_time = disco_times[closest_idx].strftime("%Y%m%d_%H%M")
            fname = f"nhc_disco_{self.name.lower()}_{self.year}_{closest_time}.txt"
            o = open(save_path + fname, "w")
            o.write(content)
            o.close()

        # Return text of NHC forecast discussion
        return {'id': closest_id, 'time_issued': closest_time, 'text': content}

    def query_nhc_discussions(self, query):
        r"""
        Searches for the given word or phrase through all NHC forecast discussions for this storm.

        Parameters
        ----------
        query : str or list
            String or list representing a word(s) or phrase(s) to search for within the NHC forecast discussions (e.g., "rapid intensification"). Query is case insensitive.

        Returns
        -------
        list
            List of dictionaries containing all relevant forecast discussions.
        """

        # Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            msg = "Error: NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']

        # Error check
        if storm_id == '':
            msg = "No NHC operational data is available for this storm."
            raise RuntimeError(msg)
        if not isinstance(query, str) and not isinstance(query, list):
            msg = "'query' must be of type str or list."
            raise TypeError(msg)
        if isinstance(query, list):
            for i in query:
                if not isinstance(i, str):
                    msg = "Entries of list 'query' must be of type str."
                    raise TypeError(msg)

        # Get list of storm discussions
        disco_dict = self.list_nhc_discussions()

        # Iterate over every entry to retrieve its discussion text
        output = []
        for idx, forecast_time in enumerate(disco_dict['utc_time']):

            # Get forecast discussion
            forecast = self.get_nhc_discussion(forecast=forecast_time)

            # Get forecast text and query for word
            text = forecast['text'].lower()

            # If found, add into list
            if isinstance(query, str):
                if text.find(query.lower()) >= 0:
                    output.append(forecast)
            else:
                found = False
                for i_query in query:
                    if text.find(i_query.lower()) >= 0:
                        found = True
                if found:
                    output.append(forecast)

        # Return list
        return output

    def get_operational_forecasts(self):
        r"""
        Retrieves operational model and NHC forecasts throughout the entire life duration of the storm.

        Returns
        -------
        dict
            Dictionary containing all forecast entries.

        Notes
        -----
        This function fetches all available forecasts, whether operational (e.g., NHC or JTWC), deterministic or ensemble model forecasts, from the Best Track a-deck as far back as data allows (1954 for NHC's area of responsibility, 2019 for JTWC).

        For example, this code retrieves all forecasts for Hurricane Michael (2018):

        .. code-block:: python

            #Get Storm object
            from tropycal import tracks
            basin = tracks.TrackDataset()
            storm = basin.get_storm(('michael',2018))

            #Retrieve all forecasts
            forecasts = storm.get_operational_forecasts()

        The resulting dict is structured as follows:

        >>> print(forecasts.keys())
        dict_keys(['CARQ', 'NAM', 'AC00', 'AEMN', 'AP01', 'AP02', 'AP03', 'AP04', 'AP05', 'AP06', 'AP07', 'AP08', 'AP09',
        'AP10', 'AP11', 'AP12', 'AP13', 'AP14', 'AP15', 'AP16', 'AP17', 'AP18', 'AP19', 'AP20', 'AVNO', 'AVNX', 'CLP5',
        'CTCX', 'DSHP', 'GFSO', 'HCCA', 'IVCN', 'IVDR', 'LGEM', 'OCD5', 'PRFV', 'SHF5', 'SHIP', 'TABD', 'TABM', 'TABS',
        'TCLP', 'XTRP', 'CMC', 'NGX', 'UKX', 'AEMI', 'AHNI', 'AVNI', 'CEMN', 'CHCI', 'CTCI', 'DSPE', 'EGRR', 'LGME',
        'NAMI', 'NEMN', 'RVCN', 'RVCX', 'SHPE', 'TBDE', 'TBME', 'TBSE', 'TVCA', 'TVCE', 'TVCN', 'TVCX', 'TVDG', 'UE00',
        'UE01', 'UE02', 'UE03', 'UE04', 'UE05', 'UE06', 'UE07', 'UE08', 'UE09', 'UE10', 'UE11', 'UE12', 'UE13', 'UE14',
        'UE15', 'UE16', 'UE17', 'UE18', 'UE19', 'UE20', 'UE21', 'UE22', 'UE23', 'UE24', 'UE25', 'UE26', 'UE27', 'UE28',
        'UE29', 'UE30', 'UE31', 'UE32', 'UE33', 'UE34', 'UE35', 'UEMN', 'CEMI', 'CMCI', 'COTC', 'EGRI', 'HMON', 'HWRF',
        'NGXI', 'NVGM', 'PRV2', 'PRVI', 'UEMI', 'UKXI', 'CEM2', 'CMC2', 'COTI', 'EGR2', 'HHFI', 'HHNI', 'HMNI', 'HWFI',
        'ICON', 'IVRI', 'NGX2', 'NVGI', 'OFCP', 'TCOA', 'TCOE', 'TCON', 'UEM2', 'UKX2', 'CHC2', 'CTC2', 'NAM2', 'OFCL',
        'OFPI', 'AEM2', 'AHN2', 'AVN2', 'DRCL', 'HHF2', 'HHN2', 'HMN2', 'HWF2', 'NVG2', 'OFCI', 'OFP2', 'FSSE', 'RI25', 'RI30'])

        Each of these keys represents a forecast model/ensemble member/center. If we select the GFS (AVNO), we now get a dictionary containing all forecast initializations for this model:

        >>> print(forecasts['AVNO'].keys())
        dict_keys(['2018100518', '2018100600', '2018100606', '2018100612', '2018100700', '2018100706', '2018100712',
        '2018100718', '2018100800', '2018100806', '2018100812', '2018100818', '2018100900', '2018100906', '2018100912',
        '2018100918', '2018101000', '2018101006', '2018101012', '2018101018', '2018101100', '2018101106', '2018101112',
        '2018101118', '2018101200', '2018101206', '2018101212'])

        Providing a forecast initialization, for example 1200 UTC 8 October 2018 (2018100812), we now get a forecast dict containing the GFS initialized at this time:

        >>> print(forecasts['AVNO']['2018100812'])
        {'fhr': [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144],
         'lat': [20.8, 21.7, 22.7, 24.0, 25.1, 25.9, 26.9, 27.9, 29.0, 30.0, 31.1, 32.1, 33.1, 34.1, 35.4, 37.2, 38.9, 40.2,
                 42.4, 44.9, 46.4, 48.2, 49.3, 50.5, 51.3],
         'lon': [-85.1, -85.1, -85.2, -85.6, -86.4, -86.8, -86.9, -86.9, -86.6, -86.2, -85.3, -84.4, -83.0, -81.3, -78.9,
                 -75.9, -72.2, -67.8, -62.5, -56.5, -50.5, -44.4, -38.8, -32.8, -27.9],
         'vmax': [57, 62, 61, 67, 74, 69, 75, 78, 80, 76, 47, 38, 34, 36, 38, 41, 52, 58, 55, 51, 48, 44, 37, 36, 37],
         'mslp': [984, 982, 979, 974, 972, 970, 966, 964, 957, 959, 969, 982, 988, 993, 994, 990, 984, 978, 978, 975, 979,
                  984, 987, 989, 991],
         'type': ['XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX',
                  'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'],
         'init': datetime.datetime(2018, 10, 8, 12, 0)}
        """

        # Real time ensemble data:
        # https://www.ftp.ncep.noaa.gov/data/nccf/com/ens_tracker/prod/

        # If forecasts dict already exist, simply return the dict
        try:
            self.forecast_dict
            return self.forecast_dict
        except:
            pass

        # Follow HURDAT procedure
        if self.source == "hurdat":

            # Get storm ID & corresponding data URL
            storm_id = self.dict['operational_id']
            storm_year = self.dict['year']
            if storm_year <= 2006:
                storm_id = self.dict['id']
            if storm_year < 1954:
                msg = "Forecast data is unavailable for storms prior to 1954."
                raise RuntimeError(msg)

            # Error check
            if storm_id == '':
                msg = "No NHC operational data is available for this storm."
                raise RuntimeError(msg)

            # Check if archive directory exists for requested year, if not redirect to realtime directory
            url_models = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/a{storm_id.lower()}.dat.gz"
            if requests.get(url_models).status_code != 200:
                url_models = f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{storm_id.lower()}.dat.gz"

            # Retrieve model data text
            if requests.get(url_models).status_code == 200:
                request = urllib.request.Request(url_models)
                response = urllib.request.urlopen(request)
                sio_buffer = BytesIO(response.read())
                gzf = gzip.GzipFile(fileobj=sio_buffer)
                data = gzf.read()
                content = data.splitlines()
                content = [(i.decode()).split(",") for i in content]
                content = [i for i in content if len(i) > 10]
                response.close()
            else:
                raise RuntimeError(
                    "No operational model data is available for this storm.")

        # Follow JTWC procedure
        else:

            url_models_noaa = f"https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/a{self.id.lower()}.dat"
            url_models_ucar = f"http://hurricanes.ral.ucar.edu/repository/data/adecks_open/{self.year}/a{self.id.lower()}.dat"

            # Retrieve model data text
            try:
                content = read_url(url_models_noaa, split=True, subsplit=False)
            except:
                try:
                    content = read_url(
                        url_models_ucar, split=True, subsplit=False)
                except:
                    raise RuntimeError(
                        "No operational model data is available for this storm.")
            content = [i.split(",") for i in content]
            content = [i for i in content if len(i) > 10]

        # Iterate through every line in content:
        forecasts = {}
        for line in content:

            # Get basic components
            lineArray = [i.replace(" ", "") for i in line]
            try:
                basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype, rad, windcode, neq, seq, swq, nwq = lineArray[
                    :17]
                use_wind = True
            except:
                basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype = lineArray[
                    :11]
                use_wind = False

            # Check init time is within storm time range
            run_init_dt = dt.strptime(run_init, '%Y%m%d%H')
            if run_init_dt < self.dict['time'][0] - timedelta(hours=6) or run_init_dt > self.dict['time'][-1] + timedelta(hours=6):
                continue

            # Enter into forecast dict
            if model not in forecasts.keys():
                forecasts[model] = {}
            if run_init not in forecasts[model].keys():
                forecasts[model][run_init] = {
                    'init': run_init_dt, 'fhr': [], 'lat': [], 'lon': [], 'vmax': [], 'mslp': [], 'type': [], 'windrad': []
                }

            # Format lat & lon
            fhr = int(fhr)
            if "N" in lat:
                lat_temp = lat.split("N")[0]
                lat = round(float(lat_temp) * 0.1, 1)
            elif "S" in lat:
                lat_temp = lat.split("S")[0]
                lat = round(float(lat_temp) * -0.1, 1)
            if "W" in lon:
                lon_temp = lon.split("W")[0]
                lon = round(float(lon_temp) * -0.1, 1)
            elif "E" in lon:
                lon_temp = lon.split("E")[0]
                lon = round(float(lon_temp) * 0.1, 1)

            # Format vmax & MSLP
            if vmax == '':
                vmax = np.nan
            else:
                vmax = int(vmax)
                if vmax < 10 or vmax > 300:
                    vmax = np.nan
            if mslp == '':
                mslp = np.nan
            else:
                mslp = int(mslp)
                if mslp < 1:
                    mslp = np.nan

            # Format wind radii
            if use_wind:
                try:
                    rad = int(rad)
                    if rad in [0, 35]:
                        rad = 34
                    neq = np.nan if windcode == '' else int(neq)
                    seq = np.nan if windcode in ['', 'AAA'] else int(seq)
                    swq = np.nan if windcode in ['', 'AAA'] else int(swq)
                    nwq = np.nan if windcode in ['', 'AAA'] else int(nwq)
                except:
                    rad = 34
                    neq = np.nan
                    seq = np.nan
                    swq = np.nan
                    nwq = np.nan
            else:
                rad = 34
                neq = np.nan
                seq = np.nan
                swq = np.nan
                nwq = np.nan

            # Add forecast data to dict if forecast hour isn't already there
            if fhr not in forecasts[model][run_init]['fhr']:
                if model in ['OFCL', 'OFCI'] and fhr > 120:
                    pass
                else:
                    if lat == 0.0 and lon == 0.0:
                        continue
                    forecasts[model][run_init]['fhr'].append(fhr)
                    forecasts[model][run_init]['lat'].append(lat)
                    forecasts[model][run_init]['lon'].append(lon)
                    forecasts[model][run_init]['vmax'].append(vmax)
                    forecasts[model][run_init]['mslp'].append(mslp)
                    forecasts[model][run_init]['windrad'].append(
                        {rad: [neq, seq, swq, nwq]})

                    # Get storm type, if it can be determined
                    if stype in ['', 'DB'] and vmax != 0 and not np.isnan(vmax):
                        stype = get_storm_type(vmax, False)
                    forecasts[model][run_init]['type'].append(stype)
            else:
                ifhr = forecasts[model][run_init]['fhr'].index(fhr)
                forecasts[model][run_init]['windrad'][ifhr][rad] = [
                    neq, seq, swq, nwq]

        # Save dict locally
        self.forecast_dict = forecasts

        # Return dict
        return forecasts

    def get_nhc_forecast_dict(self, time):
        r"""
        Retreive a dictionary of official NHC forecasts for a valid time.

        Parameters
        ----------
        time : datetime.datetime
            Time of requested forecast.

        Returns
        -------
        dict
            Dictionary containing forecast data.

        Notes
        -----
        This dict can be provided to ``utils.generate_nhc_cone()`` to generate the cone of uncertainty. Below is an example forecast dict for Hurricane Michael (2018):

        >>> storm.get_nhc_forecast_dict(dt.datetime(2018,10,8,0))
        {'fhr': [0, 3, 12, 24, 36, 48, 72, 96, 120],
         'lat': [19.8, 20.0, 21.1, 22.7, 24.4, 26.3, 30.4, 34.9, 40.7],
         'lon': [-85.4, -85.4, -85.3, -85.6, -86.0, -86.1, -84.5, -78.4, -64.4],
         'vmax': [50, 50, 60, 65, 75, 85, 75, 55, 55],
         'mslp': [nan, 997, nan, nan, nan, nan, nan, nan, nan],
         'type': ['TS', 'TS', 'TS', 'HU', 'HU', 'HU', 'HU', 'TS', 'TS'],
         'windrad': [{34: [120, 150, 90, 90], 50: [40, 0, 0, 0]},
          {34: [120, 150, 90, 90], 50: [40, 0, 0, 0]},
          {34: [120, 150, 90, 90], 50: [40, 40, 0, 0]},
          {34: [130, 140, 90, 90], 50: [50, 50, 0, 0], 64: [20, 20, 0, 0]},
          {34: [130, 130, 80, 90], 50: [50, 50, 0, 0], 64: [20, 20, 0, 0]},
          {34: [130, 130, 70, 90], 50: [60, 60, 30, 40], 64: [25, 25, 15, 25]},
          {34: [130, 130, 70, 80], 50: [60, 60, 30, 40]},
          {34: [0, 0, 0, 0]},
          {34: [0, 0, 0, 0]}],
         'init': datetime.datetime(2018, 10, 8, 0, 0)}

        As of Tropycal v0.5, ``windrad`` represents the forecast sustained wind radii (34, 50 and 64 knots) organized by [NE quadrant,SE quadrant,SW quadrant,NW quadrant] in nautical miles.
        """

        # Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError(
                "Error: NHC data can only be accessed when HURDAT is used as the data source.")

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()

        # Get all NHC forecast entries
        nhc_forecasts = self.forecast_dict['OFCL']

        # Get list of all NHC forecast initializations
        nhc_forecast_init = [k for k in nhc_forecasts.keys()]

        # Find closest matching time to the provided forecast time
        nhc_forecast_init_dt = [dt.strptime(
            k, '%Y%m%d%H') for k in nhc_forecast_init]
        time_diff = np.array(
            [(i - time).days + (i - time).seconds / 86400 for i in nhc_forecast_init_dt])
        closest_idx = np.abs(time_diff).argmin()
        forecast_dict = nhc_forecasts[nhc_forecast_init[closest_idx]]
        if np.abs(time_diff[closest_idx]) >= 1.0:
            warnings.warn(
                f"The time provided is outside of the duration of the storm. Returning the closest available NHC forecast.")

        return forecast_dict

    def download_tcr(self, save_path=""):
        r"""
        Downloads the NHC offical Tropical Cyclone Report (TCR) for the requested storm to the requested directory. Available only for storms with advisories issued by the National Hurricane Center.

        Parameters
        ----------
        save_path : str
            Path of directory to download the TCR into. Default is current working directory.
        """

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Error check
        if self.source != "hurdat":
            msg = "NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)
        if self.year < 1995:
            msg = "Tropical Cyclone Reports are unavailable prior to 1995."
            raise RuntimeError(msg)
        if not isinstance(save_path, str):
            msg = "'save_path' must be of type str."
            raise TypeError(msg)

        # Format URL
        storm_id = self.dict['id'].upper()
        storm_name = self.dict['name'].title()
        url = f"https://www.nhc.noaa.gov/data/tcr/{storm_id}_{storm_name}.pdf"

        # Check to make sure PDF is available
        request = requests.get(url)
        if request.status_code != 200:
            msg = "This tropical cyclone does not have a Tropical Cyclone Report (TCR) available."
            raise RuntimeError(msg)

        # Retrieve PDF
        response = requests.get(url)
        full_path = os.path.join(save_path, f"TCR_{storm_id}_{storm_name}.pdf")
        with open(full_path, 'wb') as f:
            f.write(response.content)

    def plot_tors(self, dist_thresh=1000, Tors=None, domain="dynamic", plotPPH=False, plot_all=False,
                  ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of the storm and associated tornado tracks.

        Parameters
        ----------
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC. Default is 1000 km.
        Tors : pandas.DataFrame
            DataFrame containing tornado data associated with the storm. If None, data is automatically retrieved from TornadoDatabase. A dataframe of tornadoes associated with the TC will then be saved to this instance of storm for future use.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        plotPPH : bool or str
            Whether to plot practically perfect forecast (PPH). True defaults to "daily". Default is False.

            * **False** - no PPH plot.
            * **True** - defaults to "daily".
            * **"total"** - probability of a tornado within 25mi of a point during the period of time selected.
            * **"daily"** - average probability of a tornado within 25mi of a point during a day starting at 12 UTC.
        plot_all : bool
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
            Customization properties of plot.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Retrieve kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Set default colormap for TC plots to Wistia
        try:
            prop['PPHcolors']
        except:
            prop['PPHcolors'] = 'Wistia'

        if Tors is None:
            try:
                self.stormTors
            except:
                warn_message = "Reading in tornado data for this storm. If you seek to analyze tornado data for multiple storms, run \"TrackDataset.assign_storm_tornadoes()\" to avoid this warning in the future."
                warnings.warn(warn_message)
                Tors = TornadoDataset()
                self.stormTors = Tors.get_storm_tornadoes(self, dist_thresh)

        # Error check if no tornadoes are found
        if len(self.stormTors) == 0:
            raise RuntimeError("No tornadoes were found with this storm.")

        # Warning if few tornadoes were found
        if len(self.stormTors) < 5:
            warn_message = f"{len(self.stormTors)} tornadoes were found with this storm. Default domain to east_conus."
            warnings.warn(warn_message)
            domain = 'east_conus'

        # Create instance of plot object
        self.plot_obj_tc = TrackPlot()
        try:
            self.plot_obj_tor = TornadoPlot()
        except:
            from ..tornado.plot import TornadoPlot
            self.plot_obj_tor = TornadoPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj_tor.create_cartopy(
                    proj='PlateCarree', central_longitude=180.0)
                self.plot_obj_tc.create_cartopy(
                    proj='PlateCarree', central_longitude=180.0)
            else:
                self.plot_obj_tor.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)
                self.plot_obj_tc.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)

        # Plot tornadoes
        plot_ax, leg_tor, domain = self.plot_obj_tor.plot_tornadoes(self.stormTors, domain, ax=ax, return_ax=True, return_domain=True,
                                                                    plotPPH=plotPPH, prop=prop, map_prop=map_prop)
        tor_title = plot_ax.get_title('left')

        # Plot storm
        plot_ax = self.plot_obj_tc.plot_storms(
            [self.dict], domain=domain, ax=plot_ax, prop=prop, map_prop=map_prop)

        plot_ax.add_artist(leg_tor)

        storm_title = plot_ax.get_title('left')
        plot_ax.set_title(f'{storm_title}\n{tor_title}',
                          loc='left', fontsize=17, fontweight='bold')

        # Save plot
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis
        return plot_ax

    def plot_TCtors_rotated(self, dist_thresh=1000, save_path=None):
        r"""
        Plot tracks of tornadoes relative to the storm motion vector of the tropical cyclone.

        Parameters
        ----------
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC. Default is 1000 km. Ignored if tornado data was passed into Storm from TrackDataset.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        The motion vector is oriented upwards (in the +y direction).
        """

        # Checks to see if stormTors exists
        try:
            self.stormTors
            dist_thresh = self.tornado_dist_thresh
        except:
            warn_message = "Reading in tornado data for this storm. If you seek to analyze tornado data for multiple storms, run \"TrackDataset.assign_storm_tornadoes()\" to avoid this warning in the future."
            warnings.warn(warn_message)
            Tors = TornadoDataset()
            stormTors = Tors.get_storm_tornadoes(self, dist_thresh)
            self.stormTors = Tors.rotateToHeading(self, stormTors)

        # Create figure for plotting
        plt.figure(figsize=(9, 9), dpi=150)
        ax = plt.subplot()

        # Default EF color scale
        EFcolors = get_colors_ef('default')

        # Plot all tornado tracks in motion relative coords
        for _, row in self.stormTors.iterrows():
            plt.plot([row['rot_xdist_s'], row['rot_xdist_e'] + .01], [row['rot_ydist_s'], row['rot_ydist_e'] + .01],
                     lw=2, c=EFcolors[row['mag']])

        # Plot dist_thresh radius
        ax.set_facecolor('#F6F6F6')
        circle = plt.Circle((0, 0), dist_thresh, color='w')
        ax.add_artist(circle)
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(dist_thresh * np.cos(an), dist_thresh * np.sin(an), 'k')
        ax.plot([-dist_thresh, dist_thresh], [0, 0], 'k--', lw=.5)
        ax.plot([0, 0], [-dist_thresh, dist_thresh], 'k--', lw=.5)

        # Plot motion vector
        plt.arrow(0, -dist_thresh * .1, 0, dist_thresh * .2, length_includes_head=True,
                  head_width=45, head_length=45, fc='k', lw=2, zorder=100)

        # Labels
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Left/Right of Storm Heading (km)', fontsize=13)
        ax.set_ylabel('Behind/Ahead of Storm Heading (km)', fontsize=13)
        ax.set_title(
            f'{self.name} {self.year} tornadoes relative to heading', fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=11.5)

        # Add legend
        handles = []
        for ef, color in enumerate(EFcolors):
            count = len(self.stormTors[self.stormTors['mag'] == ef])
            handles.append(mlines.Line2D([], [], linestyle='-',
                           color=color, label=f'EF-{ef} ({count})'))
        ax.legend(handles=handles, loc='lower left', fontsize=11.5)

        # Add attribution
        ax.text(0.99, 0.01, plot_credit(), fontsize=8, color='k', alpha=0.7,
                transform=ax.transAxes, ha='right', va='bottom', zorder=10)

        # Save plot
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis or show figure
        return ax

    def get_recon(self, path_vdm=None, path_hdobs=None, path_dropsondes=None):
        r"""
        Retrieves all aircraft reconnaissance data for this storm.

        Parameters
        ----------
        path_vdm : str, optional
            Filepath of pickle file containing VDM data retrieved from ``vdms.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.
        path_hdobs : str, optional
            Filepath of pickle file containing HDOBs data retrieved from ``hdobs.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.
        path_dropsondes : str, optional
            Filepath of pickle file containing dropsonde data retrieved from ``dropsondes.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.

        Returns
        -------
        ReconDataset
            Instance of ReconDataset is returned.

        Notes
        -----
        In addition to returning an instance of ``ReconDataset``, this function additionally stores it as an attribute of this Storm object, such that all attributes and methods associated with the ``vdms``, ``hdobs`` and ``dropsondes`` classes can be directly accessed from this Storm object.

        One method of accessing the ``hdobs.plot_points()`` method is as follows:

        .. code-block:: python

            #Get data for Hurricane Michael (2018)
            from tropycal import tracks
            basin = tracks.TrackDataset()
            storm = basin.get_storm(('michael',2018))

            #Get all recon data for this storm
            storm.get_recon()

            #Plot HDOBs points
            storm.recon.hdobs.plot_points()

        The other method is using the returned ReconDataset instance from this function:

        .. code-block:: python

            #Get data for Hurricane Michael (2018)
            from tropycal import tracks
            basin = tracks.TrackDataset()
            storm = basin.get_storm(('michael',2018))

            #Get all recon data for this storm
            recon = storm.get_recon()

            #Plot HDOBs points
            recon.hdobs.plot_points()
        """

        self.recon.get_vdms(data=path_vdm)
        self.recon.get_hdobs(data=path_hdobs)
        self.recon.get_dropsondes(data=path_dropsondes)
        return self.recon
    
    def search_ships(self):
        r"""
        Searches for available SHIPS files for this storm, if available.

        Returns
        -------
        list
            List of available SHIPS times.

        Notes
        -----
        SHIPS data is available courtesy of the UCAR Research Applications Laboratory (RAL).
        
        These available times can be plugged into ``Storm.get_ships()`` to get an object containing SHIPS data initialized at this time.
        """

        # Error check
        if self.year <= 2010:
            raise ValueError('SHIPS data is unavailable prior to 2011.')

        # Format basin name and ID
        basin_dict = {
            'north_atlantic':'northatlantic',
            'east_pacific':'northeastpacific',
            'west_pacific':'northwestpacific',
            'north_indian':'northindian'
        }
        basin_name = basin_dict.get(self.basin,'southernhemisphere')
        reformatted_id = f'{self.id[:-4]}{self.id[-2:]}'

        # Format URL from RAL and retrieve file list
        url = f'http://hurricanes.ral.ucar.edu/realtime/plots/{basin_name}/{self.year}/{self.id.lower()}/stext/'
        try:
            page = requests.get(url).text
        except:
            raise ValueError('SHIPS data is unavailable for the requested storm.')
        content = page.split("\n")
        files = []
        for line in content:
            if '<a href="' in line and '_ships.txt' in line:
                filename = (line.split('<a href="')[1]).split('">')[0]

                # Remove entries outside of duration of storm
                time = dt.strptime(filename[:8],'%y%m%d%H')
                if time < self.time[0] or time > self.time[-1]: continue

                # Add entries definitely associated with this storm
                if reformatted_id == filename.split('_ships')[0][-6:]:
                    files.append(filename)

        # Organize by date and format for printing
        return sorted([dt.strptime(i[:8],'%y%m%d%H') for i in files])
    
    def get_ships(self, time):
        r"""
        Retrieves a Ships object containing SHIPS data for a requested time.

        Parameters
        ----------
        time : datetime.datetime
            Requested time of SHIPS forecast.

        Returns
        -------
        tropycal.ships.Ships
            Instance of a Ships object containing SHIPS data for the requested time.

        Notes
        -----
        SHIPS data is available courtesy of the UCAR Research Applications Laboratory (RAL).

        1. A list of available times for SHIPS data can be retrieved using ``Storm.search_ships()``.

        2. On rare occasions, SHIPS data files from UCAR have empty data associated with them. In these cases, a value of None is returned.
        """

        # Format URL
        basin_dict = {
            'north_atlantic':'northatlantic',
            'east_pacific':'northeastpacific',
            'west_pacific':'northwestpacific',
            'north_indian':'northindian'
        }
        basin_name = basin_dict.get(self.basin,'southernhemisphere')
        url = f'http://hurricanes.ral.ucar.edu/realtime/plots/{basin_name}/{self.year}/{self.id.lower()}/stext/'
        url += f'{time.strftime("%y%m%d%H")}{self.id[:-4]}{self.id[-2:]}_ships.txt'

        # Fetch SHIPS content
        try:
            content = read_url(url, split=False, subsplit=False)
            if len(content) < 10:
                warnings.warn('Improper SHIPS entry for this time. Returning a value of None.')
                return None
        except:
            raise ValueError('SHIPS data is unavailable for the requested storm or time.')

        return Ships(content)

    def get_archer(self):
        r"""
        Retrieves satellite-derived ARCHER track data for this storm, if available.

        Returns
        -------
        dict
            Dictionary containing ARCHER data for this storm.

        Notes
        -----
        The ARCHER (Automated Rotational Center Hurricane Eye Retrieval) data is provided courtesy of the `University of Wisconsin`_. This data is at a much higher temporal resolution than the Best Track data.

        This function additionally saves the ARCHER data as an attribute of this object (storm.archer).

        .. _University of Wisconsin: http://tropic.ssec.wisc.edu/real-time/archerOnline/web/index.shtml
        """

        # Format URL
        url = f'http://tropic.ssec.wisc.edu/real-time/adt/archive{self.year}/{self.id[2:4]}{self.id[1]}-list.txt'

        # Read in data
        a = requests.get(url).content.decode("utf-8")
        content = [[c.strip() for c in b.split()] for b in a.split('\n')]
        # data = [[dt.strptime(line[0]+'/'+line[1][:4],'%Y%b%d/%H%M'),-1*float(line[-4]),float(line[-5])] for line in content[-100:-3]]
        archer = {}
        for name in ['time', 'lat', 'lon', 'mnCldTmp']:
            archer[name] = []
        for i, line in enumerate(content):
            try:
                ndx = ('MWinit' in line[-1])
                archer['time'].append(dt.strptime(
                    line[0] + '/' + line[1][:4], '%Y%b%d/%H%M'))
                archer['lat'].append(float(line[-5 - ndx]))
                archer['lon'].append(-1 * float(line[-4 - ndx]))
                archer['mnCldTmp'].append(float(line[-9 - ndx]))
            except:
                continue
        self.archer = archer

        return archer
