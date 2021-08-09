r"""Functionality for storing and analyzing an individual storm."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import requests
import copy 

from .plot import TrackPlot
from ..tornado import *
from ..recon import *

#Import tools
from .tools import *
from ..utils import *

try:
    import zipfile
    import gzip
    from io import StringIO, BytesIO
    import tarfile
except:
    warnings.warn("Warning: The libraries necessary for online NHC forecast retrieval aren't available (gzip, io, tarfile).")

try:
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

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
    """
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):
         
        #Label object
        summary = ["<tropycal.tracks.Storm>"]
        
        #Format keys for summary
        type_array = np.array(self.dict['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))[0]
        if len(idx) == 0:
            start_date = 'N/A'
            end_date = 'N/A'
            max_wind = 'N/A'
            min_mslp = 'N/A'
        else:
            time_tropical = np.array(self.dict['date'])[idx]
            start_date = time_tropical[0].strftime("%H00 UTC %d %B %Y")
            end_date = time_tropical[-1].strftime("%H00 UTC %d %B %Y")
            max_wind = 'N/A' if all_nan(np.array(self.dict['vmax'])[idx]) == True else np.nanmax(np.array(self.dict['vmax'])[idx])
            min_mslp = 'N/A' if all_nan(np.array(self.dict['mslp'])[idx]) == True else np.nanmin(np.array(self.dict['mslp'])[idx])
        summary_keys = {'Maximum Wind':f"{max_wind} knots",
                        'Minimum Pressure':f"{min_mslp} hPa",
                        'Start Date':start_date,
                        'End Date':end_date}
        
        #Format keys for coordinates
        variable_keys = {}
        for key in self.vars.keys():
            dtype = type(self.vars[key][0]).__name__
            dtype = dtype.replace("_","")
            variable_keys[key] = f"({dtype}) [{self.vars[key][0]} .... {self.vars[key][-1]}]"

        #Add storm summary
        summary.append("Storm Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')
        
        #Add coordinates
        summary.append("\nVariables:")
        add_space = np.max([len(key) for key in variable_keys.keys()])+3
        for key in variable_keys.keys():
            key_name = key
            summary.append(f'{" "*4}{key_name:<{add_space}}{variable_keys[key]}')
        
        #Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.coords.keys()])+3
        for key in self.coords.keys():
            key_name = key+":"
            val = '%0.1f'%(self.coords[key]) if key == 'ace' else self.coords[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')

        return "\n".join(summary)
    
    def __init__(self,storm,stormTors=None,read_path=""):
        
        if read_path == "" or os.path.isfile(read_path) == False:

            #Save the dict entry of the storm
            self.dict = storm

            #Add other attributes about the storm
            keys = self.dict.keys()
            self.coords = {}
            self.vars = {}
            for key in keys:
                if key == 'realtime': continue
                if isinstance(self.dict[key], list) == False and isinstance(self.dict[key], dict) == False:
                    self[key] = self.dict[key]
                    self.coords[key] = self.dict[key]
                if isinstance(self.dict[key], list) == True and isinstance(self.dict[key], dict) == False:
                    self.vars[key] = np.array(self.dict[key])
                    self[key] = np.array(self.dict[key])

            #Assign tornado data
            if stormTors != None and isinstance(stormTors,dict) == True:
                self.stormTors = stormTors['data']
                self.tornado_dist_thresh = stormTors['dist_thresh']
                self.coords['Tornado Count'] = len(stormTors['data'])

            #Get Archer track data for this storm, if it exists
            try:
                self.get_archer()
            except:
                pass

            #Determine if storm object was retrieved via realtime object
            if 'realtime' in keys and self.dict['realtime'] == True:
                self.realtime = True
                self.coords['realtime'] = True
            else:
                self.realtime = False
                self.coords['realtime'] = False
        
        else:
            
            #This functionality currently does not exist
            raise ExceptionError("This functionality has not been implemented yet.")
    
    def sel(self,time=None,lat=None,lon=None,vmax=None,mslp=None,\
            dvmax_dt=None,dmslp_dt=None,stormtype=None,method='exact'):
        
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
        
        #create copy of storm object
        NEW_STORM = Storm(copy.deepcopy(self.dict))
        idx_final = np.arange(len(self.date))
        
        #apply time filter
        if time is None:
            idx = copy.copy(idx_final)
        
        elif isinstance(time,dt):
            time_diff = np.array([(time-i).total_seconds() for i in NEW_STORM.date])
            idx = np.abs(time_diff).argmin()
            if time_diff[idx]!=0:
                if method=='exact':
                    msg = f'no exact match for {time}. Use different time or method.'
                    raise ValueError(msg)
                elif method=='floor' and time_diff[idx]<0:
                    idx += -1
                    if idx<0:
                        msg = f'no points before {time}. Use different time or method.'
                        raise ValueError(msg)
                elif method=='ceil' and time_diff[idx]>0:
                    idx += 1
                    if idx>=len(time_diff):
                        msg = f'no points after {time}. Use different time or method.'
                        raise ValueError(msg)
        
        elif isinstance(time,(tuple,list)) and len(time)==2:
            time0,time1 = time
            if time0 is None:
                time0 = min(NEW_STORM.date)
            elif not isinstance(time0,dt):
                msg = 'time bounds must be of type datetime.datetime or None.'
                raise TypeError(msg)
            if time1 is None:
                time1 = max(NEW_STORM.date)
            elif not isinstance(time1,dt):
                msg = 'time bounds must be of type datetime.datetime or None.'
                raise TypeError(msg)            
            tmptimes = np.array(NEW_STORM.date)
            idx = np.where((tmptimes>=time0) & (tmptimes<=time1))[0]
            if len(idx)==0:
                msg = f'no points between {time}. Use different time bounds.'
                raise ValueError(msg)
                
        else:
            msg = 'time must be of type datetime.datetime, tuple/list, or None.'
            raise TypeError(msg)
        
        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply lat/lon filter
        if lat is None and lon is None:
            idx = copy.copy(idx_final)
            
        elif isinstance(lat,(int,float)) and isinstance(lon,(int,float)):
            dist = np.array([great_circle((lat,lon),(x,y)).kilometers for x,y in zip(NEW_STORM.lon,NEW_STORM.lat)])
            idx = np.abs(dist).argmin()
            if dist[idx]!=0:
                if method=='exact':
                    msg = f'no exact match for {lat}/{lon}. Use different location or method.'
                    raise ValueError(msg)
                elif method in ('floor','ceil'):
                    print('floor and ceil do not apply to lat/lon filtering. Using nearest instead.')

        elif (isinstance(lat,(tuple,list)) and len(lat)==2) \
            or (isinstance(lon,(tuple,list)) and len(lon)==2):
            if not isinstance(lat,(tuple,list)):
                print('Using no lat bounds')
                lat = (None,None)
            if not isinstance(lon,(tuple,list)):
                print('Using no lon bounds')
                lon = (None,None)
            lat0,lat1 = lat
            lon0,lon1 = lon
            if lat0 is None:
                lat0 = min(NEW_STORM.lat)
            elif not isinstance(lat0,(float,int)):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lat1 is None:
                lat1 = max(NEW_STORM.lat)
            elif not isinstance(lat1,(float,int)):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lon0 is None:
                lon0 = min(NEW_STORM.lon)
            elif not isinstance(lon0,(float,int)):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
            if lon1 is None:
                lon1 = max(NEW_STORM.lon)
            elif not isinstance(lon1,(float,int)):
                msg = 'lat/lon bounds must be of type float/int or None.'
                raise TypeError(msg)
                
            tmplat,tmplon = np.array(NEW_STORM.lat),np.array(NEW_STORM.lon)%360
            idx = np.where((tmplat>=lat0) & (tmplat<=lat1) & \
                           (tmplon>=lon0%360) & (tmplon<=lon1%360))[0]
            if len(idx)==0:
                msg = f'no points in {lat}/{lon} box. Use different lat/lon bounds.'
                raise ValueError(msg)
                
        else:
            msg = 'lat and lon must be of the same type: float/int, tuple/list, or None.'
            raise TypeError(msg)  

        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply vmax filter
        if vmax is None:
            idx = copy.copy(idx_final)
        
        elif isinstance(vmax,(tuple,list)) and len(vmax)==2:
            vmax0,vmax1 = vmax
            if vmax0 is None:
                vmax0 = np.nanmin(NEW_STORM.vmax)
            elif not isinstance(vmax0,(float,int)):
                msg = 'vmax bounds must be of type float/int or None.'
                raise TypeError(msg)
            if vmax1 is None:
                vmax1 = np.nanmax(NEW_STORM.vmax)
            elif not isinstance(vmax1,(float,int)):
                msg = 'vmax bounds must be of type float/int or None.'
                raise TypeError(msg)            
            tmpvmax = np.array(NEW_STORM.vmax)
            idx = np.where((tmpvmax>=vmax0) & (tmpvmax<=vmax1))[0]
            if len(idx)==0:
                msg = f'no points with vmax between {vmax}. Use different vmax bounds.'
                raise ValueError(msg)
                
        else:
            msg = 'vmax must be of type tuple/list, or None.'
            raise TypeError(msg)

        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply mslp filter
        if mslp is None:
            idx = copy.copy(idx_final)
        
        elif isinstance(mslp,(tuple,list)) and len(mslp)==2:
            mslp0,mslp1 = mslp
            if mslp0 is None:
                mslp0 = np.nanmin(NEW_STORM.mslp)
            elif not isinstance(mslp0,(float,int)):
                msg = 'mslp bounds must be of type float/int or None.'
                raise TypeError(msg)
            if mslp1 is None:
                mslp1 = np.nanmax(NEW_STORM.mslp)
            elif not isinstance(mslp1,(float,int)):
                msg = 'mslp bounds must be of type float/int or None.'
                raise TypeError(msg)            
            tmpmslp = np.array(NEW_STORM.mslp)
            idx = np.where((tmpmslp>=mslp0) & (tmpmslp<=mslp1))[0]
            if len(idx)==0:
                msg = f'no points with mslp between {mslp}. Use different dmslp_dt bounds.'
                raise ValueError(msg)
                
        else:
            msg = 'vmax must be of type tuple/list, or None.'
            raise TypeError(msg)

        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply dvmax_dt filter
        if dvmax_dt is None:
            idx = copy.copy(idx_final)
        
        elif 'dvmax_dt' not in NEW_STORM.dict.keys():
            msg = f'dvmax_dt not in storm data. Create new object with interp first.'
            raise KeyError(msg)            
        
        elif isinstance(dvmax_dt,(tuple,list)) and len(dvmax_dt)==2:
            dvmax_dt0,dvmax_dt1 = dvmax_dt
            if dvmax_dt0 is None:
                dvmax_dt0 = np.nanmin(NEW_STORM.dvmax_dt)
            elif not isinstance(dvmax_dt0,(float,int)):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)
            if dvmax_dt1 is None:
                dvmax_dt1 = np.nanmax(NEW_STORM.dvmax_dt)
            elif not isinstance(dvmax_dt1,(float,int)):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)     
                        
            tmpvmax = np.array(NEW_STORM.dvmax_dt)
            idx = np.where((tmpvmax>=dvmax_dt0) & (tmpvmax<=dvmax_dt1))[0]
            if len(idx)==0:
                msg = f'no points with dvmax_dt between {dvmax_dt}. Use different dvmax_dt bounds.'
                raise ValueError(msg)

        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply dmslp_dt filter
        if dmslp_dt is None:
            idx = copy.copy(idx_final)
            
        elif 'dmslp_dt' not in NEW_STORM.dict.keys():
            msg = f'dmslp_dt not in storm data. Create new object with interp first.'
            raise KeyError(msg)   
            
        elif isinstance(dmslp_dt,(tuple,list)) and len(dmslp_dt)==2:
            dmslp_dt0,dmslp_dt1 = dmslp_dt
            if dmslp_dt0 is None:
                dmslp_dt0 = np.nanmin(NEW_STORM.dmslp_dt)
            elif not isinstance(dmslp_dt0,(float,int)):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)
            if dmslp_dt1 is None:
                dmslp_dt1 = np.nanmax(NEW_STORM.dmslp_dt)
            elif not isinstance(dmslp_dt1,(float,int)):
                msg = 'dmslp_dt bounds must be of type float/int or None.'
                raise TypeError(msg)            
            tmpmslp = np.array(NEW_STORM.dmslp_dt)
            idx = np.where((tmpmslp>=dmslp_dt0) & (tmpmslp<=dmslp_dt1))[0]
            if len(idx)==0:
                msg = f'no points with dmslp_dt between {dmslp_dt}. Use different dmslp_dt bounds.'
                raise ValueError(msg)
                
        #update idx_final
        idx_final = list(set(idx_final) & set(listify(idx)))

        #apply stormtype filter
        if stormtype is None:
            idx = copy.copy(idx_final)
        
        elif isinstance(stormtype,(tuple,list,str)):
            idx = [i for i,j in enumerate(NEW_STORM.type) if j in listify(stormtype)]
            if len(idx)==0:
                msg = f'no points with type {stormtype}. Use different stormtype.'
                raise ValueError(msg)
                
        else:
            msg = 'stormtype must be of type tuple/list, str, or None.'
            raise TypeError(msg)
        
        #update idx_final
        idx_final = sorted(list(set(idx_final) & set(listify(idx))))

        #Construct new storm dict with subset elements
        for key in NEW_STORM.dict.keys():
            if isinstance(NEW_STORM.dict[key], list) == True:
                NEW_STORM.dict[key] = [NEW_STORM.dict[key][i] for i in idx_final]
            else:
                NEW_STORM.dict[key] = NEW_STORM.dict[key]
            
            #Add other attributes to new storm object
            if key == 'realtime': continue
            if isinstance(NEW_STORM.dict[key], list) == False and isinstance(NEW_STORM.dict[key], dict) == False:
                NEW_STORM[key] = NEW_STORM.dict[key]
                NEW_STORM.coords[key] = NEW_STORM.dict[key]
            if isinstance(NEW_STORM.dict[key], list) == True and isinstance(NEW_STORM.dict[key], dict) == False:
                NEW_STORM.vars[key] = np.array(NEW_STORM.dict[key])
                NEW_STORM[key] = np.array(NEW_STORM.dict[key])                
                
        return NEW_STORM

    def interp(self,timeres=1,dt_window=24,dt_align='middle'):
        
        r"""
        Interpolate a storm temporally to a specified time resolution.
        
        Parameters
        ----------
        timeres : int
            Temporal resolution in hours to interpolate storm data to. Default is 1 hour.
        dt_window : int
            Time window in hours over which to calculate temporal change data. Default is 24 hours.
        dt_align : str
            Whether to align the temporal change window as "start", "middle" or "end" of the dt_window time period.
        
        Returns
        -------
        storm object
            New storm object containing the updated dictionary.
        """
        
        NEW_STORM = copy.copy(self)
        newdict = interp_storm(self.dict,timeres,dt_window,dt_align)
        for key in newdict.keys(): 
            NEW_STORM.dict[key] = newdict[key]

        #Add other attributes to new storm object
        for key in NEW_STORM.dict.keys():
            if key == 'realtime': continue
            if isinstance(NEW_STORM.dict[key], (np.ndarray,list)) == False and isinstance(NEW_STORM.dict[key], dict) == False:
                NEW_STORM[key] = NEW_STORM.dict[key]
                NEW_STORM.coords[key] = NEW_STORM.dict[key]
            if isinstance(NEW_STORM.dict[key], (np.ndarray,list)) == True and isinstance(NEW_STORM.dict[key], dict) == False:
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
        
        #Return dict
        return self.dict
        
    def to_xarray(self):
        
        r"""
        Converts the storm dict into an xarray Dataset object.
        
        Returns
        -------
        xarray.Dataset
            An xarray Dataset object containing information about the storm.
        """
        
        #Try importing xarray
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("Error: xarray is not available. Install xarray in order to use this function.") from e
            
        #Set up empty dict for dataset
        time = self.dict['date']
        ds = {}
        attrs = {}
        
        #Add every key containing a list into the dict, otherwise add as an attribute
        keys = [k for k in self.dict.keys() if k != 'date']
        for key in keys:
            if isinstance(self.dict[key], list) == True:
                ds[key] = xr.DataArray(self.dict[key],coords=[time],dims=['time'])
            else:
                attrs[key] = self.dict[key]
                    
        #Convert entire dict to a Dataset
        ds = xr.Dataset(ds,attrs=attrs)

        #Return dataset
        return ds

    def to_dataframe(self):
        
        r"""
        Converts the storm dict into a pandas DataFrame object.
        
        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame object containing information about the storm.
        """
        
        #Try importing pandas
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("Error: pandas is not available. Install pandas in order to use this function.") from e
            
        #Set up empty dict for dataframe
        time = self.dict['date']
        ds = {}
        
        #Add every key containing a list into the dict
        keys = [k for k in self.dict.keys()]
        for key in keys:
            if isinstance(self.dict[key], list) == True:
                ds[key] = self.dict[key]
                    
        #Convert entire dict to a DataFrame
        ds = pd.DataFrame(ds)

        #Return dataset
        return ds
    
    #PLOT FUNCTION FOR HURDAT
    def plot(self,domain="dynamic",plot_all=False,ax=None,return_ax=False,cartopy_proj=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of the observed track of the storm.
        
        Parameters
        ----------
        domain : str
            Domain for the plot. Default is "dynamic". "dynamic_tropical" is also available. Please refer to :ref:`options-domain` for available domain options.
        plot_all : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
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
        """
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        else:
            self.plot_obj.proj = cartopy_proj
            
        #Plot storm
        plot_ax = self.plot_obj.plot_storm(self.dict,domain,plot_all,ax=ax,return_ax=return_ax,prop=prop,map_prop=map_prop,save_path=save_path)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
        
    #PLOT FUNCTION FOR HURDAT
    def plot_nhc_forecast(self,forecast,track_labels='fhr',cone_days=5,domain="dynamic_forecast",
                          ax=None,return_ax=False,cartopy_proj=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of the operational NHC forecast track along with observed track data.
        
        Parameters
        ----------
        forecast : int or datetime.datetime
            Integer representing the forecast number, or datetime object for the closest issued forecast to this date.
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
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
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
        """
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError("Error: NHC data can only be accessed when HURDAT is used as the data source.")
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 140 or min(self.dict['lon']) < -140:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            
        #Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()

        #Get all NHC forecast entries
        nhc_forecasts = self.forecast_dict['OFCL']
        carq_forecasts = self.forecast_dict['CARQ']

        #Get list of all NHC forecast initializations
        nhc_forecast_init = [k for k in nhc_forecasts.keys()]
        carq_forecast_init = [k for k in carq_forecasts.keys()]

        #Find closest matching time to the provided forecast date, or time
        if isinstance(forecast,int) == True:
            forecast_dict = nhc_forecasts[nhc_forecast_init[forecast-1]]
            advisory_num = forecast+0
        elif isinstance(forecast,dt) == True:
            nhc_forecast_init_dt = [dt.strptime(k,'%Y%m%d%H') for k in nhc_forecast_init]
            time_diff = np.array([(i-forecast).days + (i-forecast).seconds/86400 for i in nhc_forecast_init_dt])
            closest_idx = np.abs(time_diff).argmin()
            forecast_dict = nhc_forecasts[nhc_forecast_init[closest_idx]]
            advisory_num = closest_idx+1
            if np.abs(time_diff[closest_idx]) >= 1.0:
                warnings.warn(f"The date provided is outside of the duration of the storm. Returning the closest available NHC forecast.")
        else:
            raise RuntimeError("Error: Input variable 'forecast' must be of type 'int' or 'datetime.datetime'")

        #Get observed track as per NHC analyses
        track_dict = {'lat':[],'lon':[],'vmax':[],'type':[],'mslp':[],'date':[],'extra_obs':[],'special':[],'ace':0.0}
        use_carq = True
        for k in nhc_forecast_init:
            hrs = nhc_forecasts[k]['fhr']
            hrs_carq = carq_forecasts[k]['fhr'] if k in carq_forecast_init else []
            
            #Account for old years when hour 0 wasn't included directly
            #if 0 not in hrs and k in carq_forecast_init and 0 in hrs_carq:
            if self.dict['year'] < 2000 and k in carq_forecast_init and 0 in hrs_carq:
                
                use_carq = True
                hr_idx = hrs_carq.index(0)
                track_dict['lat'].append(carq_forecasts[k]['lat'][hr_idx])
                track_dict['lon'].append(carq_forecasts[k]['lon'][hr_idx])
                track_dict['vmax'].append(carq_forecasts[k]['vmax'][hr_idx])
                track_dict['mslp'].append(np.nan)
                track_dict['date'].append(carq_forecasts[k]['init'])

                itype = carq_forecasts[k]['type'][hr_idx]
                if itype == "": itype = get_storm_type(carq_forecasts[k]['vmax'][0],False)
                track_dict['type'].append(itype)

                hr = carq_forecasts[k]['init'].strftime("%H%M")
                track_dict['extra_obs'].append(0) if hr in ['0300','0900','1500','2100'] else track_dict['extra_obs'].append(1)
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
                track_dict['date'].append(nhc_forecasts[k]['init']+timedelta(hours=hr_add))

                itype = nhc_forecasts[k]['type'][hr_idx]
                if itype == "": itype = get_storm_type(nhc_forecasts[k]['vmax'][0],False)
                track_dict['type'].append(itype)

                hr = nhc_forecasts[k]['init'].strftime("%H%M")
                track_dict['extra_obs'].append(0) if hr in ['0300','0900','1500','2100'] else track_dict['extra_obs'].append(1)
                track_dict['special'].append("")
        
        #Add main elements from storm dict
        for key in ['id','operational_id','name','year']:
            track_dict[key] = self.dict[key]

        
        #Add carq to forecast dict as hour 0, if available
        if use_carq == True and forecast_dict['init'] in track_dict['date']:
            insert_idx = track_dict['date'].index(forecast_dict['init'])
            if 0 in forecast_dict['fhr']:
                forecast_dict['lat'][0] = track_dict['lat'][insert_idx]
                forecast_dict['lon'][0] = track_dict['lon'][insert_idx]
                forecast_dict['vmax'][0] = track_dict['vmax'][insert_idx]
                forecast_dict['mslp'][0] = track_dict['mslp'][insert_idx]
                forecast_dict['type'][0] = track_dict['type'][insert_idx]
            else:
                forecast_dict['fhr'].insert(0,0)
                forecast_dict['lat'].insert(0,track_dict['lat'][insert_idx])
                forecast_dict['lon'].insert(0,track_dict['lon'][insert_idx])
                forecast_dict['vmax'].insert(0,track_dict['vmax'][insert_idx])
                forecast_dict['mslp'].insert(0,track_dict['mslp'][insert_idx])
                forecast_dict['type'].insert(0,track_dict['type'][insert_idx])
            
        #Add other info to forecast dict
        forecast_dict['advisory_num'] = advisory_num
        forecast_dict['basin'] = self.basin
        
        #Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(forecast_dict,track_dict,track_labels,cone_days,domain,ax=ax,return_ax=return_ax,save_path=save_path,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
        
    
    #PLOT FUNCTION FOR HURDAT
    def plot_gefs_ensembles(self,forecast,fhr=None,
                            prop_members = {'linewidth':0.5, 'linecolor':'k'},
                            prop_mean = {'linewidth':2.0, 'linecolor':'k'},
                            prop_gfs = {'linewidth':2.0, 'linecolor':'b'},
                            prop_ellipse = {'linewidth':2.0, 'linecolor':'r'},
                            prop_density = {'radius':200, 'cmap':plt.cm.YlOrRd, 'levels':[i for i in range(5,105,5)]},
                            domain="dynamic",ax=None,return_ax=False,cartopy_proj=None,save_path=None,map_prop={}):
        
        r"""
        (Add track history like we do for NHC forecasts)
        (Add verification for archived events)
        
        Creates a plot of individual GEFS ensemble tracks.
        
        Parameters
        ----------
        forecast : datetime.datetime
            Datetime object representing the GEFS run initialization.
        fhr : int or list, optional
            Forecast hour(s) to plot. If None (default), a plot of all forecast hours will be produced. If a list, multiple plots will be produced. If an integer, a single plot will be produced.
        plot_density : bool, optional
            If True, track density will be computed and plotted in addition to individual ensemble tracks.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
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
        """
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        else:
            self.plot_obj.proj = cartopy_proj
        
        #-------------------------------------------------------------------------
        
        #Get forecasts dict saved into storm object, if it hasn't been already
        try:
            self.forecast_dict
        except:
            self.get_operational_forecasts()
        
        #Create dict to store all data in
        ds = {'gfs':{'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'date':[]},
              'gefs':{'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'date':[],
                      'members':[],'ellipse_lat':[],'ellipse_lon':[]}
              }
        
        #String formatting for ensembles
        def str2(ens):
            if ens == 0: return "AC00"
            if ens < 10: return f"AP0{ens}"
            return f"AP{ens}"

        #Get GFS forecast entry (AVNX is valid for RAL a-deck source)
        gfs_key = 'AVNO' if 'AVNO' in self.forecast_dict.keys() else 'AVNX'
        try:
            forecast_gfs = self.forecast_dict[gfs_key][forecast.strftime("%Y%m%d%H")]
        except:
            raise RuntimeError("The requested GFS initialization isn't available for this storm.")
        
        #Enter into dict entry
        ds['gfs']['fhr'] = [int(i) for i in forecast_gfs['fhr']]
        ds['gfs']['lat'] = [np.round(i,1) for i in forecast_gfs['lat']]
        ds['gfs']['lon'] = [np.round(i,1) for i in forecast_gfs['lon']]
        ds['gfs']['vmax'] = [float(i) for i in forecast_gfs['vmax']]
        ds['gfs']['mslp'] = forecast_gfs['mslp']
        ds['gfs']['date'] = [forecast+timedelta(hours=i) for i in forecast_gfs['fhr']]
        
        #Retrieve GEFS ensemble data (30 members 2019-present, 20 members prior)
        nens = 0
        for ens in range(0,31):
            
            #Create dict entry
            ds[f'gefs_{ens}'] = {'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'date':[]}

            #Retrieve ensemble member data
            ens_str = str2(ens)
            if ens_str not in self.forecast_dict.keys(): continue
            forecast_ens = self.forecast_dict[ens_str][forecast.strftime("%Y%m%d%H")]

            #Enter into dict entry
            ds[f'gefs_{ens}']['fhr'] = [int(i) for i in forecast_ens['fhr']]
            ds[f'gefs_{ens}']['lat'] = [np.round(i,1) for i in forecast_ens['lat']]
            ds[f'gefs_{ens}']['lon'] = [np.round(i,1) for i in forecast_ens['lon']]
            ds[f'gefs_{ens}']['vmax'] = [float(i) for i in forecast_ens['vmax']]
            ds[f'gefs_{ens}']['mslp'] = forecast_ens['mslp']
            ds[f'gefs_{ens}']['date'] = [forecast+timedelta(hours=i) for i in forecast_ens['fhr']]
            nens += 1

        #Construct ensemble mean data
        #Iterate through all forecast hours
        for iter_fhr in range(0,246,6):

            #Temporary data arrays
            temp_data = {}
            for key in ds['gfs'].keys():
                if key not in ['date','fhr']: temp_data[key] = []

            #Iterate through ensemble member
            for ens in range(nens):

                #Determine if member has data valid at this forecast hour
                if iter_fhr in ds[f'gefs_{ens}']['fhr']:

                    #Retrieve index
                    idx = ds[f'gefs_{ens}']['fhr'].index(iter_fhr)

                    #Append data
                    for key in ds['gfs'].keys():
                        if key not in ['date','fhr']: temp_data[key].append(ds[f'gefs_{ens}'][key][idx])

            #Proceed if 20 or more ensemble members
            if len(temp_data['lat']) >= 10:

                #Append data
                for key in ds['gfs'].keys():
                    if key not in ['date','fhr']:
                        ds['gefs'][key].append(np.nanmean(temp_data[key]))
                ds['gefs']['fhr'].append(iter_fhr)
                ds['gefs']['date'].append(forecast+timedelta(hours=iter_fhr))
                ds['gefs']['members'].append(len(temp_data['lat']))

                #Calculate ellipse data
                if prop_ellipse != None:
                    ellipse_data = plot_ellipse(temp_data['lat'],temp_data['lon'])
                    ds['gefs']['ellipse_lon'].append(ellipse_data['xell'])
                    ds['gefs']['ellipse_lat'].append(ellipse_data['yell'])
        
        #Convert fhr to list
        if isinstance(fhr,int) == True or isinstance(fhr,float) == True:
            fhr = [fhr]
        
        #Plot storm
        plot_ax = self.plot_obj.plot_ensembles(forecast,self.dict,fhr,prop_members,prop_mean,prop_gfs,prop_ellipse,prop_density,nens,domain,ds,ax=ax,return_ax=return_ax,map_prop=map_prop,save_path=save_path)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
    
    def list_nhc_discussions(self):
        
        r"""
        Retrieves a list of NHC forecast discussions for this storm, archived on https://ftp.nhc.noaa.gov/atcf/archive/.
        
        Returns
        -------
        dict
            Dictionary containing entries for each forecast discussion for this storm.
        """
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError("Error: NHC data can only be accessed when HURDAT is used as the data source.")
        
        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']
        
        #Error check
        if storm_id == '':
            raise RuntimeError("Error: This storm was identified post-operationally. No NHC operational data is available.")
        
        #Get list of available NHC advisories & map discussions
        if storm_year == (dt.now()).year:
            
            #Get list of all discussions for all storms this year
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
            
            #Read in all NHC forecast discussions
            discos = {'id':[],'utc_date':[],'url':[],'mode':0}
            for file in files:
                
                #Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])
                
                #Open file to get info about time issued
                f = urllib.request.urlopen(url_disco + file)
                content = f.read()
                content = content.decode("utf-8")
                content = content.split("\n")
                f.close()
                
                #Figure out time issued
                hr = content[5].split(" ")[0]
                zone = content[5].split(" ")[2]
                disco_time = num_to_str2(int(hr)) + ' '.join(content[5].split(" ")[1:])
                
                format_time = content[5].split(" ")[0]
                if len(format_time) == 3: format_time = "0" + format_time
                format_time = format_time + " " +  ' '.join(content[5].split(" ")[1:])
                disco_date = dt.strptime(format_time,f'%I00 %p {zone} %a %b %d %Y')
                
                time_zones = {
                'ADT':-3,
                'AST':-4,
                'EDT':-4,
                'EST':-5,
                'CDT':-5,
                'CST':-6,
                'MDT':-6,
                'MST':-7,
                'PDT':-7,
                'PST':-8,
                'HDT':-9,
                'HST':-10}
                offset = time_zones.get(zone,0)
                disco_date = disco_date + timedelta(hours=offset*-1)
                
                #Add times issued
                discos['id'].append(disco_number)
                discos['utc_date'].append(disco_date)
                discos['url'].append(url_disco + file)
            
        elif storm_year < 1992:
            raise RuntimeError("NHC discussion data is unavailable.")
        elif storm_year < 2000:
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msg.zip'
            if requests.get(url).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            file_like_object = BytesIO(response.read())
            tar = zipfile.ZipFile(file_like_object)
            
            #Get file list
            members = '\n'.join([i for i in tar.namelist()])
            nums = "[0123456789]"
            search_pattern = f'n{storm_id[0:4].lower()}{str(storm_year)[2:]}.[01]{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files: files.append(file) #remove duplicates
            
            #Read in all NHC forecast discussions
            discos = {'id':[],'utc_date':[],'url':[],'mode':4}
            for file in files:
                
                #Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[1])
                
                #Open file to get info about time issued
                members = tar.namelist()
                members_names = [i for i in members]
                idx = members_names.index(file)
                content = (tar.read(members[idx])).decode()
                content = content.split("\n")
                
                #Figure out time issued
                slice_idx = 5 if storm_year < 1998 else 4
                for temp_idx in [slice_idx,slice_idx-1,slice_idx+1,slice_idx-2,slice_idx+2]:
                    try:
                        hr = content[temp_idx].split(" ")[0]
                        if 'NOON' in content[temp_idx]:
                            temp_line = content[temp_idx].replace("NOON","12 PM")
                            zone = temp_line.split(" ")[1]
                            disco_date = dt.strptime(temp_line.rstrip(),f'%I %p {zone} %a %b %d %Y')
                        else:
                            zone = content[temp_idx].split(" ")[2]
                            disco_date = dt.strptime(content[temp_idx].rstrip(),f'%I %p {zone} %a %b %d %Y')
                    except:
                        pass
                
                time_zones = {
                'ADT':-3,
                'AST':-4,
                'EDT':-4,
                'EST':-5,
                'CDT':-5,
                'CST':-6,
                'MDT':-6,
                'MST':-7,
                'PDT':-7,
                'PST':-8,
                'HDT':-9,
                'HST':-10}
                offset = time_zones.get(zone,0)
                disco_date = disco_date + timedelta(hours=offset*-1)
                
                #Add times issued
                discos['id'].append(disco_number)
                discos['utc_date'].append(disco_date)
                discos['url'].append(file)
                
            response.close()
            tar.close()
            
        elif storm_year == 2000:
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
            if requests.get(url).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            file_like_object = BytesIO(response.read())
            tar = tarfile.open(fileobj=file_like_object)
            
            #Get file list
            members = '\n'.join([i.name for i in tar.getmembers()])
            nums = "[0123456789]"
            search_pattern = f'N{storm_id[0:4]}{str(storm_year)[2:]}.[01]{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files: files.append(file) #remove duplicates
            
            #Read in all NHC forecast discussions
            discos = {'id':[],'utc_date':[],'url':[],'mode':3}
            for file in files:

                #Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[1])
                
                #Open file to get info about time issued
                members = tar.getmembers()
                members_names = [i.name for i in members]
                idx = members_names.index(file)
                f = tar.extractfile(members[idx])
                content = (f.read()).decode()
                f.close()
                content = content.split("\n")
                
                #Figure out time issued
                hr = content[4].split(" ")[0]
                zone = content[4].split(" ")[2]
                disco_time = num_to_str2(int(hr)) + ' '.join(content[4].split(" ")[1:])
                disco_date = dt.strptime(content[4],f'%I %p {zone} %a %b %d %Y')
                
                time_zones = {
                'ADT':-3,
                'AST':-4,
                'EDT':-4,
                'EST':-5,
                'CDT':-5,
                'CST':-6,
                'MDT':-6,
                'MST':-7,
                'PDT':-7,
                'PST':-8,
                'HDT':-9,
                'HST':-10}
                offset = time_zones.get(zone,0)
                disco_date = disco_date + timedelta(hours=offset*-1)
                
                #Add times issued
                discos['id'].append(disco_number)
                discos['utc_date'].append(disco_date)
                discos['url'].append(file)
                
            response.close()
            tar.close()
            
        elif storm_year in range(2001,2006):
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
            if storm_year < 2003: url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
            if requests.get(url).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
            request = urllib.request.Request(url)
            response = urllib.request.urlopen(request)
            file_like_object = BytesIO(response.read())
            tar = tarfile.open(fileobj=file_like_object)

            #Get file list
            members = '\n'.join([i.name for i in tar.getmembers()])
            nums = "[0123456789]"
            search_pattern = f'{storm_id.lower()}.discus.[01]{nums}{nums}.{nums}{nums}{nums}{nums}{nums}{nums}{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(members)
            files = []
            for file in filelist:
                if file not in files: files.append(file) #remove duplicates
            response.close()
            tar.close()
            
            #Read in all NHC forecast discussions
            discos = {'id':[],'utc_date':[],'url':[],'mode':1}
            for file in files:

                #Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])
                disco_time = file_info[3]
                disco_year = storm_year
                if disco_time[0:2] == "01" and int(storm_id[2:4]) > 3:
                    disco_year = storm_year + 1
                disco_date = dt.strptime(str(disco_year)+disco_time,'%Y%m%d%H%M')

                discos['id'].append(disco_number)
                discos['utc_date'].append(disco_date)
                discos['url'].append(file)
                
            if storm_year < 2003: discos['mode'] = 2
            
        else:
            #Retrieve list of NHC discussions for this storm
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            if requests.get(url_disco).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
            path_disco = urllib.request.urlopen(url_disco)
            string = path_disco.read().decode('utf-8')
            nums = "[0123456789]"
            search_pattern = f'{storm_id.lower()}.discus.[01]{nums}{nums}.{nums}{nums}{nums}{nums}{nums}{nums}{nums}{nums}'
            pattern = re.compile(search_pattern)
            filelist = pattern.findall(string)
            files = []
            for file in filelist:
                if file not in files: files.append(file) #remove duplicates
            path_disco.close()

            #Read in all NHC forecast discussions
            discos = {'id':[],'utc_date':[],'url':[],'mode':0}
            for file in files:

                #Get info about forecast
                file_info = file.split(".")
                disco_number = int(file_info[2])
                disco_time = file_info[3]
                disco_year = storm_year
                if disco_time[0:2] == "01" and int(storm_id[2:4]) > 3:
                    disco_year = storm_year + 1
                disco_date = dt.strptime(str(disco_year)+disco_time,'%Y%m%d%H%M')

                discos['id'].append(disco_number)
                discos['utc_date'].append(disco_date)
                discos['url'].append(url_disco + file)
                
        #Return dict entry
        try:
            discos
        except:
            raise RuntimeError("NHC discussion data is unavailable.")
            
        if len(discos['id']) == 0:
            raise RuntimeError("NHC discussion data is unavailable.")
        return discos
        
    def get_nhc_discussion(self,forecast,save_path=None):
        
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
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            msg = "Error: NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)
        
        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']
        
        #Error check
        if storm_id == '':
            msg = "No NHC operational data is available for this storm."
            raise RuntimeError(msg)
        
        #Error check
        if isinstance(forecast,int) == False and isinstance(forecast,dt) == False:
            msg = "forecast must be of type int or datetime.datetime"
            raise TypeError(msg)
        
        #Get list of storm discussions
        disco_dict = self.list_nhc_discussions()
        
        if isinstance(forecast,dt) == True:
            #Find closest discussion to the time provided
            disco_times = disco_dict['utc_date']
            disco_ids = [int(i) for i in disco_dict['id']]
            disco_diff = np.array([(i-forecast).days + (i-forecast).seconds/86400 for i in disco_times])
            closest_idx = np.abs(disco_diff).argmin()
            closest_diff = disco_diff[closest_idx]
            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]
        
            #Raise warning if difference is >=1 day
            if np.abs(closest_diff) >= 1.0:
                warnings.warn(f"The date provided is unavailable or outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion.")
                
        if isinstance(forecast,int) == True:
            #Find closest discussion ID to the one provided
            disco_times = disco_dict['utc_date']
            disco_ids = [int(i) for i in disco_dict['id']]
            if forecast == -1:
                closest_idx = -1
            else:
                disco_diff = np.array([i-forecast for i in disco_ids])
                closest_idx = np.abs(disco_diff).argmin()
                closest_diff = disco_diff[closest_idx]
                
                #Raise warning if difference is >=1 ids
                if np.abs(closest_diff) >= 2.0:
                    msg = f"The ID provided is unavailable or outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion."
                    warnings.warn(msg)

            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]
        
            
        #Read content of NHC forecast discussion
        if disco_dict['mode'] == 0:
            url_disco = disco_dict['url'][closest_idx]
            if requests.get(url_disco).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
            f = urllib.request.urlopen(url_disco)
            content = f.read()
            content = content.decode("utf-8")
            f.close()
            
        elif disco_dict['mode'] in [1,2,3]:
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
            if disco_dict['mode'] in [2,3]: url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
            if requests.get(url).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
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
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msg.zip'
            if requests.get(url).status_code != 200: raise RuntimeError("NHC discussion data is unavailable.")
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
        
        #Save file, if specified
        if save_path != None:
            closest_time = disco_times[closest_idx].strftime("%Y%m%d_%H%M")
            fname = f"nhc_disco_{self.name.lower()}_{self.year}_{closest_time}.txt"
            o = open(save_path+fname,"w")
            o.write(content)
            o.close()
        
        #Return text of NHC forecast discussion
        return {'id':closest_id,'time_issued':closest_time,'text':content}
    
    
    def query_nhc_discussions(self,query):
        
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
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            msg = "Error: NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)
        
        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']
        
        #Error check
        if storm_id == '':
            msg = "No NHC operational data is available for this storm."
            raise RuntimeError(msg)
        if isinstance(query,str) == False and isinstance(query,list) == False:
            msg = "'query' must be of type str or list."
            raise TypeError(msg)
        if isinstance(query,list) == True:
            for i in query:
                if isinstance(i,str) == False:
                    msg = "Entries of list 'query' must be of type str."
                    raise TypeError(msg)
        
        #Get list of storm discussions
        disco_dict = self.list_nhc_discussions()
        
        #Iterate over every entry to retrieve its discussion text
        output = []
        for idx,forecast_date in enumerate(disco_dict['utc_date']):
            
            #Get forecast discussion
            forecast = self.get_nhc_discussion(forecast=forecast_date)
            
            #Get forecast text and query for word
            text = forecast['text'].lower()
            
            #If found, add into list
            if isinstance(query,str) == True:
                if text.find(query.lower()) >= 0: output.append(forecast)
            else:
                found = False
                for i_query in query:
                    if text.find(i_query.lower()) >= 0: found = True
                if found == True: output.append(forecast)
            
        #Return list
        return output
    

    def get_operational_forecasts(self):

        r"""
        Retrieves operational model and NHC forecasts throughout the entire life duration of the storm.

        Returns
        -------
        dict
            Dictionary containing all forecast entries.
        """
        
        #Real time ensemble data:
        #https://www.ftp.ncep.noaa.gov/data/nccf/com/ens_tracker/prod/
        
        #If forecasts dict already exist, simply return the dict
        try:
            self.forecast_dict
            return self.forecast_dict
        except:
            pass
        
        #Follow HURDAT procedure
        if self.source == "hurdat":
            
            #Get storm ID & corresponding data URL
            storm_id = self.dict['operational_id']
            storm_year = self.dict['year']
            if storm_year <= 2006: storm_id = self.dict['id']
            if storm_year < 1954:
                msg = "Forecast data is unavailable for storms prior to 1954."
                raise RuntimeError(msg)

            #Error check
            if storm_id == '':
                msg = "No NHC operational data is available for this storm."
                raise RuntimeError(msg)

            #Check if archive directory exists for requested year, if not redirect to realtime directory
            url_models = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/a{storm_id.lower()}.dat.gz"
            if requests.get(url_models).status_code != 200:
                url_models = f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{storm_id.lower()}.dat.gz"
        
            #Retrieve model data text
            if requests.get(url_models).status_code == 200:
                request = urllib.request.Request(url_models)
                response = urllib.request.urlopen(request)
                sio_buffer = BytesIO(response.read())
                gzf = gzip.GzipFile(fileobj = sio_buffer)
                data = gzf.read()
                content = data.splitlines()
                content = [(i.decode()).split(",") for i in content]
                content = [i for i in content if len(i) > 10]
                response.close()
            else:
                raise RuntimeError("No operational model data is available for this storm.")
        
        #Follow JTWC procedure
        else:
            
            #Get storm ID & corresponding data URL
            storm_year = self.dict['year']
            if storm_year < 2016:
                msg = "Forecast data is unavailable for JTWC storms prior to 2016."
                raise RuntimeError(msg)
            url_models = f"http://hurricanes.ral.ucar.edu/repository/data/adecks_open/a{self.id.lower()}.dat"
            
            #Retrieve model data text
            try:
                f = urllib.request.urlopen(url_models)
                content = f.read()
                content = content.decode("utf-8")
                content = content.split("\n")
                content = [i.split(",") for i in content]
                content = [i for i in content if len(i) > 10]
                f.close()
            except:
                raise RuntimeError("No operational model data is available for this storm.")

        #Iterate through every line in content:
        forecasts = {}
        for line in content:

            #Get basic components
            lineArray = [i.replace(" ","") for i in line]
            basin,number,run_init,n_a,model,fhr,lat,lon,vmax,mslp,stype = lineArray[:11]

            #Enter into forecast dict
            if model not in forecasts.keys(): forecasts[model] = {}
            if run_init not in forecasts[model].keys(): forecasts[model][run_init] = {
                'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'type':[],'init':dt.strptime(run_init,'%Y%m%d%H')
            }

            #Format lat & lon
            fhr = int(fhr)
            if "N" in lat:
                lat_temp = lat.split("N")[0]
                lat = float(lat_temp) * 0.1
            elif "S" in lat:
                lat_temp = lat.split("S")[0]
                lat = float(lat_temp) * -0.1
            if "W" in lon:
                lon_temp = lon.split("W")[0]
                lon = float(lon_temp) * -0.1
            elif "E" in lon:
                lon_temp = lon.split("E")[0]
                lon = float(lon_temp) * 0.1
            
            #Format vmax & MSLP
            if vmax == '':
                vmax = np.nan
            else:
                vmax = int(vmax)
                if vmax < 10 or vmax > 300: vmax = np.nan
            if mslp == '':
                mslp = np.nan
            else:
                mslp = int(mslp)
                if mslp < 1: mslp = np.nan

            #Add forecast data to dict if forecast hour isn't already there
            if fhr not in forecasts[model][run_init]['fhr']:
                if model in ['OFCL','OFCI'] and fhr > 120:
                    pass
                else:
                    if lat == 0.0 and lon == 0.0:
                        continue
                    forecasts[model][run_init]['fhr'].append(fhr)
                    forecasts[model][run_init]['lat'].append(lat)
                    forecasts[model][run_init]['lon'].append(lon)
                    forecasts[model][run_init]['vmax'].append(vmax)
                    forecasts[model][run_init]['mslp'].append(mslp)
                    
                    #Get storm type, if it can be determined
                    if stype in ['','DB'] and vmax != 0 and np.isnan(vmax) == False:
                        stype = get_storm_type(vmax,False)
                    forecasts[model][run_init]['type'].append(stype)

        #Save dict locally
        self.forecast_dict = forecasts
        
        #Return dict
        return forecasts

    
    def download_tcr(self,save_path=""):
        
        r"""
        Downloads the NHC offical Tropical Cyclone Report (TCR) for the requested storm to the requested directory. Available only for storms with advisories issued by the National Hurricane Center.
        
        Parameters
        ----------
        save_path : str
            Path of directory to download the TCR into. Default is current working directory.
        """
        
        #Error check
        if self.source != "hurdat":
            msg = "NHC data can only be accessed when HURDAT is used as the data source."
            raise RuntimeError(msg)
        if self.year < 1995:
            msg = "Tropical Cyclone Reports are unavailable prior to 1995."
            raise RuntimeError(msg)
        if isinstance(save_path,str) == False:
            msg = "'save_path' must be of type str."
            raise TypeError(msg)
        
        #Format URL
        storm_id = self.dict['id'].upper()
        storm_name = self.dict['name'].title()
        url = f"https://www.nhc.noaa.gov/data/tcr/{storm_id}_{storm_name}.pdf"
        
        #Check to make sure PDF is available
        request = requests.get(url)
        if request.status_code != 200:
            msg = "This tropical cyclone does not have a Tropical Cyclone Report (TCR) available."
            raise RuntimeError(msg)
        
        #Retrieve PDF
        response = requests.get(url)
        full_path = os.path.join(save_path,f"TCR_{storm_id}_{storm_name}.pdf")
        with open(full_path, 'wb') as f:
            f.write(response.content)

            
    def plot_tors(self,dist_thresh=1000,Tors=None,domain="dynamic",plotPPH=False,plot_all=False,\
                  ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
                
        r"""
        Creates a plot of the storm and associated tornado tracks.
        
        Parameters
        ----------
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC. Default is 1000 km.
        Tors : pandas.DataFrame
            DataFrame containing tornado data associated with the storm. If None, data is automatically retrieved from TornadoDatabase. A dataframe of tornadoes associated with the TC will then be saved to this instance of storm
                for future use.
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
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        
        Other Parameters
        ----------------
        prop : dict
            Customization properties of plot.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        """
        
        #Set default colormap for TC plots to Wistia
        try:
            prop['PPHcolors']
        except:
            prop['PPHcolors']='Wistia'
        
        if Tors == None:
            try:
                self.stormTors
            except:
                warn_message = "Reading in tornado data for this storm. If you seek to analyze tornado data for multiple storms, run \"TrackDataset.assign_storm_tornadoes()\" to avoid this warning in the future."
                warnings.warn(warn_message)
                Tors = TornadoDataset()
                self.stormTors = Tors.get_storm_tornadoes(self,dist_thresh)
    
        #Error check if no tornadoes are found
        if len(self.stormTors) == 0:
            raise RuntimeError("No tornadoes were found with this storm.")
    
        #Warning if few tornadoes were found
        if len(self.stormTors) < 5:
            warn_message = f"{len(self.stormTors)} tornadoes were found with this storm. Default domain to east_conus."
            warnings.warn(warn_message)
            domain = 'east_conus'
    
        #Create instance of plot object
        self.plot_obj_tc = TrackPlot()
        try:
            self.plot_obj_tor = TornadoPlot()
        except:
            from ..tornado.plot import TornadoPlot
            self.plot_obj_tor = TornadoPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj_tor.create_cartopy(proj='PlateCarree',central_longitude=180.0)
                self.plot_obj_tc.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj_tor.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                self.plot_obj_tc.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                
        #Plot tornadoes
        plot_ax,leg_tor,domain = self.plot_obj_tor.plot_tornadoes(self.stormTors,domain,ax=ax,return_ax=True,return_domain=True,\
                                             plotPPH=plotPPH,prop=prop,map_prop=map_prop)
        tor_title = plot_ax.get_title('left')

        #Plot storm
        plot_ax = self.plot_obj_tc.plot_storm(self.dict,domain=domain,ax=plot_ax,prop=prop,map_prop=map_prop)
        
        plot_ax.add_artist(leg_tor)
        
        storm_title = plot_ax.get_title('left')
        plot_ax.set_title(f'{storm_title}\n{tor_title}',loc='left',fontsize=17,fontweight='bold')
        
        #Return axis
        if ax != None or return_ax == True: 
            return plot_ax
        else:
            plt.show()
            plt.close()


    def plot_TCtors_rotated(self,dist_thresh=1000,return_ax=False):
        
        r"""
        Plot tracks of tornadoes relative to the storm motion vector of the tropical cyclone.
        
        Parameters
        ----------
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC. Default is 1000 km. Ignored if tornado data was passed into Storm from TrackDataset.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        
        Notes
        -----
        The motion vector is oriented upwards (in the +y direction).
        """
        
        #Checks to see if stormTors exists
        try:
            self.stormTors
            dist_thresh = self.tornado_dist_thresh
        except:
            warn_message = "Reading in tornado data for this storm. If you seek to analyze tornado data for multiple storms, run \"TrackDataset.assign_storm_tornadoes()\" to avoid this warning in the future."
            warnings.warn(warn_message)
            Tors = TornadoDataset()
            stormTors = Tors.get_storm_tornadoes(self,dist_thresh)
            self.stormTors = Tors.rotateToHeading(self,stormTors)
        
        #Create figure for plotting
        plt.figure(figsize=(9,9),dpi=150)
        ax = plt.subplot()
        
        #Default EF color scale
        EFcolors = get_colors_ef('default')
        
        #Plot all tornado tracks in motion relative coords
        for _,row in self.stormTors.iterrows():
            plt.plot([row['rot_xdist_s'],row['rot_xdist_e']+.01],[row['rot_ydist_s'],row['rot_ydist_e']+.01],\
                     lw=2,c=EFcolors[row['mag']])
            
        #Plot dist_thresh radius
        ax.set_facecolor('#F6F6F6')
        circle = plt.Circle((0,0), dist_thresh, color='w')
        ax.add_artist(circle)
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(dist_thresh * np.cos(an), dist_thresh * np.sin(an),'k')
        ax.plot([-dist_thresh,dist_thresh],[0,0],'k--',lw=.5)
        ax.plot([0,0],[-dist_thresh,dist_thresh],'k--',lw=.5)
        
        #Plot motion vector
        plt.arrow(0, -dist_thresh*.1, 0, dist_thresh*.2, length_includes_head=True,
          head_width=45, head_length=45,fc='k',lw=2,zorder=100)
        
        #Labels
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Left/Right of Storm Heading (km)',fontsize=13)
        ax.set_ylabel('Behind/Ahead of Storm Heading (km)',fontsize=13)
        ax.set_title(f'{self.name} {self.year} tornadoes relative to heading',fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=11.5)
        
        #Add legend
        handles=[]
        for ef,color in enumerate(EFcolors):
            count = len(self.stormTors[self.stormTors['mag']==ef])
            handles.append(mlines.Line2D([], [], linestyle='-',color=color,label=f'EF-{ef} ({count})'))
        ax.legend(handles=handles,loc='lower left',fontsize=11.5)
        
        #Add attribution
        ax.text(0.99,0.01,plot_credit(),fontsize=8,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='bottom',zorder=10)
        
        #Return axis or show figure
        if return_ax == True:
            return ax
        else:
            plt.show()
            plt.close()
            
    def get_recon(self,deltap_thresh=8,save_path="",read_path="",mission_url_list=None,update=False):
        
        r"""
        Creates an instance of ReconDataset for this storm's data. Saves it as an attribute of this object (storm.recon).
        
        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Requested storm as an instance of a Storm object.
        save_path : str, optional
            Filepath to save recon data in. Recommended in order to avoid having to re-read in the data.
        read_path : str, optional
            Filepath to read saved recon data from. If specified, "save_path" cannot be passed as an argument.
        """
        
        self.recon = ReconDataset(self,deltap_thresh,mission_url_list,save_path,read_path,update)
                
    def get_archer(self):
        
        r"""
        Retrieves satellite-derived Archer track data for this storm, if available. Saves it as an attribute of this object (storm.archer).
        """
        
        #Format URL
        url = f'http://tropic.ssec.wisc.edu/real-time/adt/archive{self.year}/{self.id[2:4]}{self.id[1]}-list.txt'
        
        #Read in data
        a = requests.get(url).content.decode("utf-8") 
        content = [[c.strip() for c in b.split()] for b in a.split('\n')]
        #data = [[dt.strptime(line[0]+'/'+line[1][:4],'%Y%b%d/%H%M'),-1*float(line[-4]),float(line[-5])] for line in content[-100:-3]]
        archer = {}
        for name in ['time','lat','lon','mnCldTmp']:
            archer[name]=[]
        for i,line in enumerate(content):
            try:
                ndx = ('MWinit' in line[-1])
                archer['time'].append(dt.strptime(line[0]+'/'+line[1][:4],'%Y%b%d/%H%M'))
                archer['lat'].append(float(line[-5-ndx]))
                archer['lon'].append(-1*float(line[-4-ndx]))
                archer['mnCldTmp'].append(float(line[-9-ndx]))
            except:
                continue
        self.archer = archer
    