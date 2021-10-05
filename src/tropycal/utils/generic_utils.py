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
                return "north_pacific"
    
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
