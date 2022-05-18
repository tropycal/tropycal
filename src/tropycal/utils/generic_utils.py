r"""Utility functions that are used across modules.

Ensure these do not reference any function from colors.py. If a function does need to reference another function from colors.py,
add it in that file.

Public utility functions should be added to documentation in the '/docs/_templates/overrides/tropycal.utils.rst' file."""

import os, sys
import math
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
import requests
import urllib
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib as mlib
import warnings
import scipy.interpolate as interp

from .. import constants

#===========================================================================================================
# Public utilities
# These are used internally and have use externally. Add these to documentation.
#===========================================================================================================

def wind_to_category(wind_speed):
    
    r"""
    Convert sustained wind speed in knots to Saffir-Simpson Hurricane Wind Scale category.
    
    Parameters
    ----------
    wind_speed : int
        Sustained wind speed in knots.
    
    Returns
    -------
    int
        Category corresponding to the sustained wind. 0 is tropical storm, -1 is tropical depression.
    """
    
    #Category 5 hurricane
    if wind_speed >= 137:
        return 5
    
    #Category 4 hurricane
    elif wind_speed >= 113:
        return 4
    
    #Category 3 hurricane
    elif wind_speed >= 96:
        return 3
    
    #Category 2 hurricane
    elif wind_speed >= 83:
        return 2
    
    #Category 1 hurricane
    elif wind_speed >= 64:
        return 1
    
    #Tropical storm
    elif wind_speed >= 34:
        return 0
    
    #Tropical depression
    else:
        return -1

def category_to_wind(category):
    
    r"""
    Convert Saffir-Simpson Hurricane Wind Scale category to minimum threshold sustained wind speed in knots.
    
    Parameters
    ----------
    category : int
        Saffir-Simpson Hurricane Wind Scale category. Use 0 for tropical storm, -1 for tropical depression.
    
    Returns
    -------
    int
        Sustained wind speed in knots corresponding to the minimum threshold of the requested category.
    """
    
    #Construct dictionary of thresholds
    conversion = {-1:5,
                  0:34,
                  1:64,
                  2:83,
                  3:96,
                  4:113,
                  5:137}
    
    #Return category
    return conversion.get(category,np.nan)

def classify_subtropical(storm_type):
    
    r"""
    Determine whether a tropical cyclone was purely subtropical.
    
    Parameters
    ----------
    storm_type : list or numpy.ndarray
        List or array containing storm types.
    
    Returns
    -------
    bool
        Boolean identifying whether the tropical cyclone was purely subtropical.
    """
    
    #Ensure storm_type is a numpy array
    storm_type_check = np.array(storm_type)
    
    #Ensure all storm types are uppercase
    storm_track_check = [i.upper() for i in storm_type_check]
    
    #Check for subtropical depression status
    if 'SD' in storm_type_check:
        if 'SD' in storm_type_check and True not in np.isin(storm_type_check,['TD','TS','HU']):
            return True
    
    #Check for subtropical storm status
    if 'SS' in storm_type_check and True not in np.isin(storm_type_check,['TD','TS','HU']):
        return True
    
    #Otherwise, it was a tropical cyclone at some point in its life cycle
    else:
        return False

def get_storm_classification(wind_speed,subtropical_flag,basin):
    
    r"""
    Retrieve the tropical cyclone classification given its subtropical status and current basin.
    
    These strings take the format of "Tropical Storm", "Hurricane", "Typhoon", etc.
    
    Warning: This function currently does not differentiate between 1-minute, 3-minute and 10-minute sustained wind speeds.
    
    Parameters
    ----------
    wind_speed : int
        Integer denoting sustained wind speed in knots.
    subtropical_flag : bool
        Boolean denoting whether the cyclone is subtropical or not.
    basin : str
        String denoting basin in which the tropical cyclone is located.
    
    Returns
    -------
    str
        String denoting the classification of the tropical cyclone.
    """
    
    #North Atlantic and East Pacific basins
    if basin in ['north_atlantic','east_pacific']:
        if wind_speed == 0:
            return "Unknown"
        elif wind_speed < 34:
            if subtropical_flag == True:
                return "Subtropical Depression"
            else:
                return "Tropical Depression"
        elif wind_speed < 63:
            if subtropical_flag == True:
                return "Subtropical Storm"
            else:
                return "Tropical Storm"
        else:
            return "Hurricane"
    
    #West Pacific basin
    elif basin == 'west_pacific':
        if wind_speed == 0:
            return "Unknown"
        elif wind_speed < 34:
            if subtropical_flag == True:
                return "Subtropical Depression"
            else:
                return "Tropical Depression"
        elif wind_speed < 63:
            if subtropical_flag == True:
                return "Subtropical Storm"
            else:
                return "Tropical Storm"
        elif wind_speed < 130:
            return "Typhoon"
        else:
            return "Super Typhoon"
    
    #Australia and South Pacific basins
    elif basin == 'australia' or basin == 'south_pacific':
        if wind_speed == 0:
            return "Unknown"
        elif wind_speed < 63:
            return "Tropical Cyclone"
        else:
            return "Severe Tropical Cyclone"
    
    #North Indian Ocean
    elif basin == 'north_indian':
        if wind_speed == 0:
            return "Unknown"
        elif wind_speed < 28:
            return "Depression"
        elif wind_speed < 34:
            return "Deep Depression"
        elif wind_speed < 48:
            return "Cyclonic Storm"
        elif wind_speed < 64:
            return "Severe Cyclonic Storm"
        elif wind_speed < 90:
            return "Very Severe Cyclonic Storm"
        elif wind_speed < 120:
            return "Extremely Severe Cyclonic Storm"
        else:
            return "Super Cyclonic Storm"
    
    #South Indian Ocean
    elif basin == 'south_indian':
        if wind_speed == 0:
            return "Unknown"
        elif wind_speed < 28:
            return "Tropical Disturbance"
        elif wind_speed < 34:
            return "Tropical Depression"
        elif wind_speed < 48:
            return "Moderate Tropical Storm"
        elif wind_speed < 64:
            return "Severe Tropical Storm"
        elif wind_speed < 90:
            return "Tropical Cyclone"
        elif wind_speed < 115:
            return "Intense Tropical Cyclone"
        else:
            return "Very Intense Tropical Cyclone"
    
    #Otherwise, return a generic "Cyclone" classification
    else:
        return "Cyclone"

def get_storm_type(wind_speed,subtropical_flag):
    
    r"""
    Retrieve the 2-character tropical cyclone type (e.g., "TD", "TS", "HU") given its subtropical status.
    
    Parameters
    ----------
    wind_speed : int
        Integer denoting sustained wind speed in knots.
    subtropical_flag : bool
        Boolean denoting whether the cyclone is subtropical or not.
    
    Returns
    -------
    str
        String denoting the tropical cyclone type.
    """
    
    #Tropical depression
    if wind_speed < 34:
        if subtropical_flag == True:
            return "SD"
        else:
            return "TD"
    
    #Tropical storm
    elif wind_speed < 63:
        if subtropical_flag == True:
            return "SS"
        else:
            return "TS"
    
    #Hurricane
    else:
        return "HU"

def get_basin(lat,lon,storm_id=""):
    
    r"""
    Returns the current basin of the tropical cyclone.
    
    Parameters
    ----------
    lat : int or float
        Latitude of the storm.
    lon : int or float
        Longitude of the storm.
    
    Other Parameters
    ----------------
    storm_id : str
        String representing storm ID. Used to distinguish between Atlantic and Pacific basins.
    
    Returns
    -------
    str
        String representing the current basin (e.g., "north_atlantic", "east_pacific").
    """
    
    #Error check
    if isinstance(lat,float) == False and isinstance(lat,int) == False:
        msg = "\"lat\" must be of type int or float."
        raise TypeError(msg)
    if isinstance(lon,float) == False and isinstance(lon,int) == False:
        msg = "\"lon\" must be of type int or float."
        raise TypeError(msg)
    
    #Fix longitude
    if lon < 0.0: lon = lon + 360.0
    
    #Northern hemisphere check
    if lat >= 0.0:
        
        if lon < 100.0:
            return "north_indian"
        elif lon < 180.0:
            return "west_pacific"
        else:
            if len(storm_id) != 8:
                msg = "Cannot determine whether storm is in North Atlantic or East Pacific basins."
                raise RuntimeError(msg)
            if storm_id[0:2] == "AL":
                return "north_atlantic"
            else:
                return "east_pacific"
    
    #Southern hemisphere check
    else:
        
        if lon < 20.0:
            return "south_atlantic"
        elif lon < 90.0:
            return "south_indian"
        elif lon < 160.0:
            return "australia"
        elif lon < 280.0:
            return "south_pacific"
        else:
            return "south_atlantic"

def knots_to_mph(wind_speed):
    
    r"""
    Convert wind from knots to miles per hour, in increments of 5 as used by NHC.
    
    Parameters
    ----------
    wind_speed : int
        Sustained wind in knots.
    
    Returns
    -------
    int
        Sustained wind in miles per hour.
    """
    
    #Ensure input is rounded down to nearest multiple of 5
    wind_speed = wind_speed - (wind_speed % 5)

    #Define knots and mph conversions
    kts = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185]
    mphs = [0,10,15,20,25,30,35,40,45,50,60,65,70,75,80,85,90,100,105,110,115,120,125,130,140,145,150,155,160,165,175,180,185,190,195,200,205,210]
    
    #If value is in list, return converted value
    if wind_speed in kts:
        return mphs[kts.index(wind_speed)]
    
    #Otherwise, return original value
    return wind_speed

def accumulated_cyclone_energy(wind_speed,hours=6):
    
    r"""
    Calculate Accumulated Cyclone Energy (ACE) based on sustained wind speed in knots.
    
    Parameters
    ----------
    wind_speed : int
        Sustained wind in knots.
    hours : int, optional
        Duration in hours over which the sustained wind was observed. Default is 6 hours.
    
    Returns
    -------
    float
        Accumulated cyclone energy.
    """
    
    #Calculate ACE
    ace = ((10**-4) * (wind_speed**2)) * (hours/6.0)
    
    #Coerce to zero if wind speed less than TS force
    if wind_speed < 34: ace = 0.0
    
    #Return ACE
    return ace

def dropsonde_mslp_estimate(mslp,surface_wind):
    
    r"""
    Apply a NHC rule of thumb for estimating a TC's minimum central MSLP. This is intended for a dropsonde released in the eye, accounting for drifting by factoring in the surface wind in knots.
    
    Parameters
    ----------
    mslp : int or float
        Dropsonde surface MSLP, in hPa.
    surface_wind : int or float
        Surface wind as measured by the dropsonde. This **must** be surface wind; this cannot be substituted by a level just above the surface.
    
    Returns
    -------
    float
        Estimated TC minimum central MSLP using the NHC estimation method.
    
    Notes
    -----
    Source is from NHC presentation:
    https://www.nhc.noaa.gov/outreach/presentations/nhc2013_aircraftData.pdf
    """
    
    return mslp - (surface_wind / 10.0)

def nhc_cone_radii(year,basin,forecast_hour=None):
    
    r"""
    Retrieve the official NHC Cone of Uncertainty radii by basin, year and forecast hour(s). Units are in nautical miles.
    
    Parameters
    ----------
    year : int
        Valid year for cone of uncertainty radii.
    basin : str
        Basin for cone of uncertainty radii. If basin is invalid, return value will be an empty dict.
    forecast_hour : int or list, optional
        Forecast hour(s) to retrieve the cone of uncertainty for. If empty, all available forecast hours will be retrieved.
    
    Returns
    -------
    dict
        Dictionary with forecast hour(s) as the keys, and the cone radius in nautical miles for each respective forecast hour as the values.
    
    Notes
    -----
    1. NHC cone radii are available beginning 2008 onward. Radii for years before 2008 will be defaulted to 2008, and if the current year's radii are not available yet, the radii for the most recent year will be returned.
    
    2. NHC began producing cone radii for forecast hour 60 in 2020. Years before 2020 do not have a forecast hour 60.
    """
    
    #Source: https://www.nhc.noaa.gov/verification/verify3.shtml
    #Source 2: https://www.nhc.noaa.gov/aboutcone.shtml
    #Radii are in nautical miles
    cone_climo_hr = [3,12,24,36,48,72,96,120]
    
    #Basin check
    if basin not in ['north_atlantic','east_pacific']:
        return {}
    
    #Fix for 2020 and later that incorporates 60 hour forecasts
    if year >= 2020:
        cone_climo_hr = [3,12,24,36,48,60,72,96,120]
    
    #Forecast hour check
    if forecast_hour is None:
        forecast_hour = cone_climo_hr
    elif isinstance(forecast_hour,int) == True:
        if forecast_hour not in cone_climo_hr:
            raise ValueError(f"Forecast hour {forecast_hour} is invalid. Available forecast hours for {year} are: {cone_climo_hr}")
        else:
            forecast_hour = [forecast_hour]
    elif isinstance(forecast_hour,list) == True:
        forecast_hour = [i for i in forecast_hour if i in cone_climo_hr]
        if len(forecast_hour) == 0:
            raise ValueError(f"Requested forecast hours are invalid. Available forecast hours for {year} are: {cone_climo_hr}")
    else:
        raise TypeError("forecast_hour must be of type int or list")
    
    #Year check
    if year > np.max([k for k in constants.CONE_SIZE_ATL.keys()]):
        year = [k for k in constants.CONE_SIZE_ATL.keys()][0]
        warnings.warn(f"No cone information is available for the requested year. Defaulting to {year} cone.")
    elif year not in constants.CONE_SIZE_ATL.keys():
        year = 2008
        warnings.warn(f"No cone information is available for the requested year. Defaulting to 2008 cone.")
    
    #Retrieve data
    cone_radii = {}
    for hour in list(np.sort(forecast_hour)):
        hour_index = cone_climo_hr.index(hour)
        if basin == 'north_atlantic':
            cone_radii[hour] = constants.CONE_SIZE_ATL[year][hour_index]
        elif basin == 'east_pacific':
            cone_radii[hour] = constants.CONE_SIZE_PAC[year][hour_index]
    
    return cone_radii

def generate_nhc_cone(forecast,basin,shift_lons=False,cone_days=5,cone_year=None,return_xarray=False):

    r"""
    Generates a gridded cone of uncertainty using forecast data from NHC.

    Parameters
    ----------
    forecast : dict
        Dictionary containing forecast data
    basin : str
        Basin for cone of uncertainty radii.
    shift_lons : bool, optional
        If true, grid will be shifted to +0 to +360 degrees longitude. Default is False (-180 to +180 degrees).
    cone_days : int, optional
        Number of forecast days to generate the cone through. Default is 5 days.
    cone_year : int, optional
        Year valid for cone radii. If None, this fuction will attempt to retrieve the year from the forecast dict.
    return_xarray : bool, optional
        If True, returns output as an xarray Dataset. Default is False, returning output as a dictionary.
    
    Returns
    -------
    dict or xarray.Dataset
        Depending on `return_xarray`, returns either a dictionary or an xarray Dataset containing the gridded cone of uncertainty and its accompanying attributes.
    """
    
    #Check forecast dict has all required keys
    check_dict = [True if i in forecast.keys() else False for i in ['fhr','lat','lon','init']]
    if False in check_dict:
        raise ValueError("forecast dict must contain keys 'fhr', 'lat', 'lon' and 'init'. You may retrieve a forecast dict for a Storm object through 'storm.get_operational_forecasts()'.")
    
    #Check forecast basin
    if basin not in constants.ALL_BASINS:
        raise ValueError("basin cannot be identified.")

    #Retrieve cone of uncertainty year
    if cone_year is None:
        cone_year = forecast['init'].year
    if cone_year > np.max([k for k in constants.CONE_SIZE_ATL.keys()]):
        cone_year = [k for k in constants.CONE_SIZE_ATL.keys()][0]
        warnings.warn(f"No cone information is available for the requested year. Defaulting to {cone_year} cone.")
    elif cone_year not in constants.CONE_SIZE_ATL.keys():
        cone_year = 2008
        warnings.warn(f"No cone information is available for the requested year. Defaulting to 2008 cone.")

    #Retrieve cone size and hours for given year
    if basin in ['north_atlantic','east_pacific']:
        output = nhc_cone_radii(cone_year,basin)
        cone_climo_hr = [k for k in output.keys()]
        cone_size = [output[k] for k in output.keys()]
    else:
        cone_climo_hr = [3,12,24,36,48,72,96,120]
        cone_size = 0

    #Function for interpolating between 2 times
    def temporal_interpolation(value, orig_times, target_times):
        f = interp.interp1d(orig_times,value)
        ynew = f(target_times)
        return ynew

    #Function for plugging small array into larger array
    def plug_array(small,large,small_coords,large_coords):

        small_lat = np.round(small_coords['lat'],2)
        small_lon = np.round(small_coords['lon'],2)
        large_lat = np.round(large_coords['lat'],2)
        large_lon = np.round(large_coords['lon'],2)

        small_minlat = min(small_lat)
        small_maxlat = max(small_lat)
        small_minlon = min(small_lon)
        small_maxlon = max(small_lon)

        if small_minlat in large_lat:
            minlat = np.where(large_lat==small_minlat)[0][0]
        else:
            minlat = min(large_lat)
        if small_maxlat in large_lat:
            maxlat = np.where(large_lat==small_maxlat)[0][0]
        else:
            maxlat = max(large_lat)
        if small_minlon in large_lon:
            minlon = np.where(large_lon==small_minlon)[0][0]
        else:
            minlon = min(large_lon)
        if small_maxlon in large_lon:
            maxlon = np.where(large_lon==small_maxlon)[0][0]
        else:
            maxlon = max(large_lon)

        large[minlat:maxlat+1,minlon:maxlon+1] = small

        return large

    #Function for finding nearest value in an array
    def findNearest(array,val):
        return array[np.abs(array - val).argmin()]

    #Function for adding a radius surrounding a point
    def add_radius(lats,lons,vlat,vlon,rad):

        #construct new array expanding slightly over rad from lat/lon center
        grid_res = 0.05 #1 degree is approx 111 km
        grid_fac = (rad*4)/111.0

        #Make grid surrounding position coordinate & radius of circle
        nlon = np.arange(findNearest(lons,vlon-grid_fac),findNearest(lons,vlon+grid_fac+grid_res),grid_res)
        nlat = np.arange(findNearest(lats,vlat-grid_fac),findNearest(lats,vlat+grid_fac+grid_res),grid_res)
        lons,lats = np.meshgrid(nlon,nlat)
        return_arr = np.zeros((lons.shape))

        #Calculate distance from vlat/vlon at each gridpoint
        r_earth = 6.371 * 10**6
        dlat = np.subtract(np.radians(lats),np.radians(vlat))
        dlon = np.subtract(np.radians(lons),np.radians(vlon))

        a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats)) * np.cos(np.radians(vlat)) * np.sin(dlon/2) * np.sin(dlon/2)
        c = 2 * np.arctan(np.sqrt(a), np.sqrt(1-a));
        dist = (r_earth * c)/1000.0
        dist = dist * 0.621371 #to miles
        dist = dist * 0.868976 #to nautical miles

        #Mask out values less than radius
        return_arr[dist <= rad] = 1

        #Attach small array into larger subset array
        small_coords = {'lat':nlat,'lon':nlon}

        return return_arr, small_coords

    #--------------------------------------------------------------------

    #Check if fhr3 is available, then get forecast data
    flag_12 = 0
    if forecast['fhr'][0] == 12:
        flag_12 = 1
        cone_climo_hr = cone_climo_hr[1:]
        fcst_lon = forecast['lon']
        fcst_lat = forecast['lat']
        fhr = forecast['fhr']
        t = np.array(forecast['fhr'])/6.0
        subtract_by = t[0]
        t = t - t[0]
        interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.1) - t[0]
    elif 3 in forecast['fhr'] and 1 in forecast['fhr'] and 0 in forecast['fhr']:
        fcst_lon = forecast['lon'][2:]
        fcst_lat = forecast['lat'][2:]
        fhr = forecast['fhr'][2:]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
    elif 3 in forecast['fhr'] and 0 in forecast['fhr']:
        idx = np.array([i for i,j in enumerate(forecast['fhr']) if j in cone_climo_hr])
        fcst_lon = np.array(forecast['lon'])[idx]
        fcst_lat = np.array(forecast['lat'])[idx]
        fhr = np.array(forecast['fhr'])[idx]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
    elif forecast['fhr'][1] < 12:
        cone_climo_hr[0] = 0
        fcst_lon = [forecast['lon'][0]]+forecast['lon'][2:]
        fcst_lat = [forecast['lat'][0]]+forecast['lat'][2:]
        fhr = [forecast['fhr'][0]]+forecast['fhr'][2:]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0]/6.0,t[-1]+0.1,0.1)
    else:
        cone_climo_hr[0] = 0
        fcst_lon = forecast['lon']
        fcst_lat = forecast['lat']
        fhr = forecast['fhr']
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.1)

    #Determine index of forecast day cap
    if (cone_days*24) in fhr:
        cone_day_cap = list(fhr).index(cone_days*24)+1
        fcst_lon = fcst_lon[:cone_day_cap]
        fcst_lat = fcst_lat[:cone_day_cap]
        fhr = fhr[:cone_day_cap]
        t = np.array(fhr)/6.0
        interp_fhr_idx = np.arange(interp_fhr_idx[0],t[-1]+0.1,0.1)
    else:
        cone_day_cap = len(fhr)

    #Account for dateline
    if shift_lons == True:
        temp_lon = np.array(fcst_lon)
        temp_lon[temp_lon<0] = temp_lon[temp_lon<0]+360.0
        fcst_lon = temp_lon.tolist()

    #Interpolate forecast data temporally and spatially
    interp_kind = 'quadratic'
    if len(t) == 2: interp_kind = 'linear' #Interpolate linearly if only 2 forecast points
    x1 = interp.interp1d(t,fcst_lon,kind=interp_kind)
    y1 = interp.interp1d(t,fcst_lat,kind=interp_kind)
    interp_fhr = interp_fhr_idx * 6
    interp_lon = x1(interp_fhr_idx)
    interp_lat = y1(interp_fhr_idx)

    #Return if no cone specified
    if cone_size == 0:
        return_dict = {'center_lon':interp_lon,'center_lat':interp_lat}
        if return_xarray:
            import xarray as xr
            return xr.Dataset(return_dict)
        else:
            return return_dict

    #Interpolate cone radius temporally
    cone_climo_hr = cone_climo_hr[:cone_day_cap]
    cone_size = cone_size[:cone_day_cap]

    cone_climo_fhrs = np.array(cone_climo_hr)
    if flag_12 == 1:
        interp_fhr += (subtract_by*6.0)
        cone_climo_fhrs = cone_climo_fhrs[1:]
    idxs = np.nonzero(np.in1d(np.array(fhr),np.array(cone_climo_hr)))
    temp_arr = np.array(cone_size)[idxs]
    interp_rad = np.apply_along_axis(lambda n: temporal_interpolation(n,fhr,interp_fhr),axis=0,arr=temp_arr)

    #Initialize 0.05 degree grid
    gridlats = np.arange(min(interp_lat)-7,max(interp_lat)+7,0.05)
    gridlons = np.arange(min(interp_lon)-7,max(interp_lon)+7,0.05)
    gridlons2d,gridlats2d = np.meshgrid(gridlons,gridlats)

    #Iterate through fhr, calculate cone & add into grid
    large_coords = {'lat':gridlats,'lon':gridlons}
    griddata = np.zeros((gridlats2d.shape))
    for i,(ilat,ilon,irad) in enumerate(zip(interp_lat,interp_lon,interp_rad)):
        temp_grid, small_coords = add_radius(gridlats,gridlons,ilat,ilon,irad)
        plug_grid = np.zeros((griddata.shape))
        plug_grid = plug_array(temp_grid,plug_grid,small_coords,large_coords)
        griddata = np.maximum(griddata,plug_grid)
    
    if return_xarray:
        import xarray as xr
        cone = xr.DataArray(griddata,coords=[gridlats,gridlons],dims=['grid_lat','grid_lon'])
        return_ds = {
            'cone':cone,
            'center_lon':interp_lon,
            'center_lat':interp_lat
        }
        return_ds = xr.Dataset(return_ds)
        return_ds.attrs['year'] = cone_year
        return return_ds
    
    else:
        return_dict = {'lat':gridlats,'lon':gridlons,'lat2d':gridlats2d,'lon2d':gridlons2d,'cone':griddata,
                   'center_lon':interp_lon,'center_lat':interp_lat,'year':cone_year}
        return return_dict

#===========================================================================================================
# Private utilities
# These are primarily intended to be used internally. Do not add these to documentation.
#===========================================================================================================

def all_nan(arr):
    
    r"""
    Determine whether the entire array is filled with NaNs.
    
    Parameters
    ----------
    arr : list or numpy.ndarray
        List or array to be checked.
    
    Returns
    -------
    bool
        Returns whether the array is filled with all NaNs.
    """
    
    #Convert array to numpy array
    arr_copy = np.array(arr)
    
    #Check if there are non-NaN values in the array
    if len(arr_copy[~np.isnan(arr_copy)]) == 0:
        return True
    else:
        return False

def category_label_to_wind(category):
    
    r"""
    Convert Saffir-Simpson Hurricane Wind Scale category label to minimum threshold sustained wind speed in knots. Internal function.
    
    Parameters
    ----------
    category : int
        Saffir-Simpson Hurricane Wind Scale category. Use 0 for tropical storm, -1 for tropical depression.
    
    Returns
    -------
    int
        Sustained wind speed in knots corresponding to the minimum threshold of the requested category.
    """
    
    #Convert category to lowercase
    category_lowercase = category.lower()
    
    #Return thresholds based on category label
    if category_lowercase == 'td' or category_lowercase == 'sd':
        return category_to_wind(0) - 1
    elif category_lowercase == 'ts' or category_lowercase == 'ss':
        return category_to_wind(0)
    elif category_lowercase == 'c1':
        return category_to_wind(1)
    elif category_lowercase == 'c2':
        return category_to_wind(2)
    elif category_lowercase == 'c3':
        return category_to_wind(3)
    elif category_lowercase == 'c4':
        return category_to_wind(4)
    else:
        return category_to_wind(5)

class Distance:
    
    def __init__(self,dist,units='kilometers'):
        
        # Conversion fractions (numerator_denominator)
        mi_km = 0.621371
        nmi_km = 0.539957
        m_km = 1000.
        ft_km = 3280.84
        
        if units in ['kilometers','km']:
            self.kilometers = dist
        elif units in ['miles','mi']:
            self.kilometers = dist / mi_km
        elif units in ['nauticalmiles','nautical','nmi']:
            self.kilometers = dist / nmi_km
        elif units in ['feet','ft']:
            self.kilometers = dist / ft_km
        elif units in ['meters','m']:
            self.kilometers = dist / m_km
        self.miles = self.kilometers * mi_km
        self.nautical = self.kilometers * nmi_km
        self.meters = self.kilometers * m_km
        self.feet = self.kilometers * ft_km


class great_circle(Distance):

    def __init__(self, start_point, end_point, **kwargs):
        r"""
        
        Parameters
        ----------
        start_point : tuple or int
            Starting pair of coordinates, in order of (latitde, longitude)
        end_point : tuple or int
            Starting pair of coordinates, in order of (latitde, longitude)
        radius : float, optional
            Radius of Earth. Default is 6371.009 km.
        
        Returns
        -------
        great_circle
            Instance of a great_circle object. To retrieve the distance, add the requested unit at the end (e.g., "great_circle(start,end).kilometers").
        
        Notes
        -----
        Use spherical geometry to calculate the surface distance between two points. Uses the mean earth radius as defined by the International Union of Geodesy and Geophysics, approx 6371.009 km (for WGS-84), resulting in an error of up to about 0.5%. Otherwise set which radius of the earth to use by specifying a 'radius' keyword argument. It must be in kilometers.
        """
        
        #Set Earth's radius
        self.RADIUS = kwargs.pop('radius', 6371.009)
        
        #Compute 
        dist = self.measure(start_point,end_point)
        Distance.__init__(self,dist,units='kilometers')

    def measure(self, start_point, end_point):
        
        #Retrieve latitude and longitude coordinates from input pairs
        lat1, lon1 = math.radians(start_point[0]), math.radians(start_point[1]%360)
        lat2, lon2 = math.radians(end_point[0]), math.radians(end_point[1]%360)

        #Compute sin and cos of coordinates
        sin_lat1, cos_lat1 = math.sin(lat1), math.cos(lat1)
        sin_lat2, cos_lat2 = math.sin(lat2), math.cos(lat2)

        #Compute sin and cos of delta longitude
        delta_lon = lon2 - lon1
        cos_delta_lon, sin_delta_lon = math.cos(delta_lon), math.sin(delta_lon)

        #Compute great circle distance
        d = math.atan2(math.sqrt((cos_lat2 * sin_delta_lon) ** 2 +
                       (cos_lat1 * sin_lat2 -
                        sin_lat1 * cos_lat2 * cos_delta_lon) ** 2),
                sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_delta_lon)

        #Return great circle distance
        return self.RADIUS * d
