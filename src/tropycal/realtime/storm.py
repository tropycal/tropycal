r"""Functionality for storing and analyzing an individual realtime storm."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import requests

from ..tracks.tools import *
from ..tracks import *

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

class RealtimeStorm(Storm):
    
    r"""
    Initializes an instance of RealtimeStorm. This object inherits all the methods and functionality of ``tropycal.tracks.Storm``, but with additional methods unique to this object, all containing the word "realtime" as part of the function name.

    Parameters
    ----------
    storm : dict
        Dict entry of the requested storm.
    
    Returns
    -------
    RealtimeStorm
        Instance of a RealtimeStorm object.
    """
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):
         
        #Label object
        summary = ["<tropycal.realtime.RealtimeStorm>"]
        
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
    
    def __init__(self,storm,stormTors=None):
        
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
        self.get_archer()
        
        #Determine if storm object was retrieved via realtime object
        if 'realtime' in keys and self.dict['realtime'] == True:
            self.realtime = True
            self.coords['realtime'] = True
        else:
            self.realtime = False
            self.coords['realtime'] = False
    
    def get_nhc_discussion_realtime(self):
        
        r"""
        Retrieve the latest available NHC forecast discussion.
        
        Returns
        -------
        dict
            Dict entry containing the latest official NHC forecast discussion.
        """
        
        #Get the latest forecast discussion
        return self.get_nhc_discussion(forecast=-1)
    
    def get_nhc_forecast_realtime(self):
        
        r"""
        Retrieve a dictionary containing the latest official NHC forecast.
        
        Returns
        -------
        dict
            Dict entry containing the latest official NHC forecast.
        """
        
        #Get forecast for this storm
        url = f"https://ftp.nhc.noaa.gov/atcf/fst/{self.id.lower()}.fst"
        if requests.get(url).status_code != 200: raise RuntimeError("NHC forecast data is unavailable for this storm.")

        #Read file content
        f = urllib.request.urlopen(url)
        content = f.read()
        content = content.decode("utf-8")
        content = content.split("\n")
        content = [(i.replace(" ","")).split(",") for i in content]
        f.close()
        
        #Iterate through every line in content:
        nhc_forecasts = {}
        
        for line in content:

            #Get basic components
            lineArray = [i.replace(" ","") for i in line]
            if len(lineArray) < 11: continue
            basin,number,run_init,n_a,model,fhr,lat,lon,vmax,mslp,stype = lineArray[:11]
            if model not in ["OFCL","OFCI"]: continue
            
            if len(nhc_forecasts) == 0:
                nhc_forecasts = {
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
            if fhr not in nhc_forecasts['fhr']:
                if model in ['OFCL','OFCI'] and fhr > 120:
                    pass
                else:
                    if lat == 0.0 and lon == 0.0:
                        continue
                    nhc_forecasts['fhr'].append(fhr)
                    nhc_forecasts['lat'].append(lat)
                    nhc_forecasts['lon'].append(lon)
                    nhc_forecasts['vmax'].append(vmax)
                    nhc_forecasts['mslp'].append(mslp)
                    
                    #Get storm type, if it can be determined
                    if stype in ['','DB'] and vmax != 0 and np.isnan(vmax) == False:
                        stype = get_type(vmax,False)
                    nhc_forecasts['type'].append(stype)
            
        #Save forecast as attribute
        self.latest_forecast = nhc_forecasts
        return self.latest_forecast
    
    def plot_nhc_forecast_realtime(self,track_labels='fhr',cone_days=5,domain="dynamic_forecast",
                                   ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Plots the latest available NHC forecast. This function is available for Realtime retrieved objects only.
        
        Parameters
        ----------
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
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
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
            
        #Get forecast for this storm
        try:
            nhc_forecasts = (self.latest_forecast).copy()
        except:
            nhc_forecasts = self.get_nhc_forecast_realtime()
        
        #Add other info to forecast dict
        nhc_forecasts['advisory_num'] = -1
        nhc_forecasts['basin'] = self.basin
        
        #Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(nhc_forecasts,self.dict,track_labels,cone_days,domain,ax=ax,return_ax=return_ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
        
    def get_realtime_info(self):
        
        r"""
        Returns a dict containing the latest available information about the storm. This function uses NHC Public Advisories, so it will differ from available Best Track data.
        
        Returns
        -------
        dict
            Dict entry containing current storm information.
        """
        
        #Declare empty dict
        current_advisory = {}

        #Get latest available public advisory
        f = urllib.request.urlopen(f"https://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml")
        content = f.read()
        content = content.decode("utf-8")
        content = content.split("\n")
        f.close()

        #Get public advisory number
        results = [i for i in content if 'advisoryNumber' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['advisory_number'] = result

        #Get UTC time of advisory
        results = [i for i in content if 'messageDateTimeUTC' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        result = dt.strptime(result,'%Y%m%d %I:%M:%S %p UTC')
        current_advisory['time_utc'] = result

        #Get storm type
        results = [i for i in content if 'systemType' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['type'] = result.title()

        #Get storm name
        results = [i for i in content if 'systemName' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['name'] = result.title()

        #Get coordinates
        results = [i for i in content if 'centerLocLatitude' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['lat'] = float(result)
        results = [i for i in content if 'centerLocLongitude' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['lon'] = float(result)

        #Get sustained wind speed
        results = [i for i in content if 'systemIntensityMph' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['wind_mph'] = int(result)
        results = [i for i in content if 'systemIntensityKph' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['wind_kph'] = int(result)
        results = [i for i in content if 'systemIntensityKts' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['wind_kt'] = int(result)

        #Get MSLP
        results = [i for i in content if 'systemMslpMb' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['mslp'] = int(result)

        #Get storm category
        current_advisory['category'] = convert_category(current_advisory['wind_kt'])
        
        #Get storm direction
        results = [i for i in content if 'systemDirectionOfMotion' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['motion_direction'] = result.split(" OR ")[0]
        results = [i for i in content if 'systemDirectionOfMotion' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['motion_direction_degrees'] = int((result.split(" OR ")[1]).split(" DEGREES")[0])

        #Get storm speed
        results = [i for i in content if 'systemSpeedMph' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['motion_mph'] = int(result)
        results = [i for i in content if 'systemSpeedKph' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['motion_kph'] = int(result)
        results = [i for i in content if 'systemSpeedKts' in i][0]
        result = (results.split(">")[1]).split("<")[0]
        current_advisory['motion_kt'] = int(result)
        
        #Return dict
        return current_advisory
            
        
    def __get_public_advisory(self):

        #Get list of all public advisories for this storm
        url_disco = 'https://ftp.nhc.noaa.gov/atcf/pub/'
        page = requests.get(url_disco).text
        content = page.split("\n")
        files = []
        for line in content:
            if ".public" in line and self.id.lower() in line:
                filename = line.split('">')[1]
                filename = filename.split("</a>")[0]
                files.append(filename)
        del content

        #Keep only largest number
        numbers = [int(i.split(".")[-1]) for i in files]
        max_number = np.nanmax(numbers)
        if max_number >= 100:
            max_number = str(max_number)
        elif max_number >= 10:
            max_number = f"0{max_number}"
        else:
            max_number = f"00{max_number}"
        files = [i for i in files if f".{max_number}" in i]

        #Determine if there's an intermediate advisory available
        if len(files) > 1:
            advisory_letter = []
            for file in files:
                if 'public_' in file:
                    letter = (file.split("public_")[1]).split(".")[0]
                    advisory_letter.append(letter)
            max_letter = max(advisory_letter)
            files = [i for i in files if f".public_{max_letter}" in i]

        #Read file containing advisory
        f = urllib.request.urlopen(url_disco + files[0])
        content = f.read()
        content = content.decode("utf-8")
        content = content.split("\n")
        f.close()

        #Figure out time issued
        hr = content[6].split(" ")[0]
        zone = content[6].split(" ")[2]
        disco_time = str2(int(hr)) + ' '.join(content[6].split(" ")[1:])

        format_time = content[6].split(" ")[0]
        if len(format_time) == 3: format_time = "0" + format_time
        format_time = format_time + " " +  ' '.join(content[6].split(" ")[1:])
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
