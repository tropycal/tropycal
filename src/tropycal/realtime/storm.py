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

from ..tracks import *
from ..tracks.tools import *
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
    import matplotlib.dates as mdates
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
            
    def download_graphic_realtime(self,save_path=""):
        
        r"""
        Download the latest official forecast track graphic. Available for both NHC and JTWC sources.
        
        Parameters
        ----------
        save_path : str
            Filepath to save the image in. If blank, default is current working directory.
        """
        
        #Determine data source
        if self.source == 'hurdat':
            part1 = f"AT{self.id[2:4]}" if self.id[0:2] == "AL" else self.id[0:4]
            url = f"https://www.nhc.noaa.gov/storm_graphics/{part1}/{self.id}_5day_cone_with_line_and_wind.png"
        else:
            url = f"https://www.nrlmry.navy.mil/atcf_web/docs/current_storms/{self.id.lower()}.gif"
        url_ext = url.split(".")[-1]
        
        #Try to download file
        if requests.get(url).status_code != 200: raise RuntimeError("Official forecast graphic is unavailable for this storm.")
        
        #Download file
        response = requests.get(url)
        full_path = os.path.join(save_path,f"Forecast_{self.id}.{url_ext}")
        with open(full_path, 'wb') as f:
            f.write(response.content)
        
    
    def get_nhc_discussion_realtime(self):
        
        r"""
        Retrieve the latest available forecast discussion. For JTWC storms, the Prognostic Reasoning product is retrieved.
        
        Returns
        -------
        dict
            Dict entry containing the latest official forecast discussion.
        """
        
        #Warn about pending deprecation
        warnings.warn("'get_nhc_discussion_realtime' will be deprecated in future Tropycal versions, use 'get_discussion_realtime' instead",DeprecationWarning)
        
        #Get latest forecast discussion for HURDAT source storm objects
        if self.source == "hurdat":
            return self.get_nhc_discussion(forecast=-1)
        
        #Get latest forecast discussion for JTWC source storm objects
        elif self.source == 'jtwc':
            
            #Read in discussion file
            url = f"https://www.metoc.navy.mil/jtwc/products/{self.id[0:2].lower()}{self.id[2:4]}{self.id[6:8]}prog.txt"
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            f.close()
            return content
        
        #Otherwise, return error message
        else:
            msg = "No realtime forecast discussion is available for this storm."
            raise RuntimeError(msg)
    
    def get_discussion_realtime(self):
        
        r"""
        Retrieve the latest available forecast discussion. For JTWC storms, the Prognostic Reasoning product is retrieved.
        
        Returns
        -------
        dict
            Dict entry containing the latest official forecast discussion.
        """
        
        #Get latest forecast discussion for HURDAT source storm objects
        if self.source == "hurdat":
            return self.get_nhc_discussion(forecast=-1)
        
        #Get latest forecast discussion for JTWC source storm objects
        elif self.source == 'jtwc':
            
            #Read in discussion file
            url = f"https://www.metoc.navy.mil/jtwc/products/{self.id[0:2].lower()}{self.id[2:4]}{self.id[6:8]}prog.txt"
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            f.close()
            return content
        
        #Otherwise, return error message
        else:
            msg = "No realtime forecast discussion is available for this storm."
            raise RuntimeError(msg)
    
    def get_forecast_realtime(self):
        
        r"""
        Retrieve a dictionary containing the latest official forecast. Available for both NHC and JTWC sources.
        
        Returns
        -------
        dict
            Dictionary containing the latest official forecast.
        
        Notes
        -----
        This dictionary includes a calculation for accumulated cyclone energy (ACE), cumulatively for the storm's lifespan through each forecast hour. This is done by linearly interpolating the forecast to 6-hour intervals and calculating 6-hourly ACE at each interval. For storms where forecast tropical cyclone type is available, ACE is not calculated for forecast periods that are neither tropical nor subtropical.
        """
        
        #NHC forecast data
        if self.source == 'hurdat':
        
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
            forecasts = {}

            for line in content:

                #Get basic components
                lineArray = [i.replace(" ","") for i in line]
                if len(lineArray) < 11: continue
                basin,number,run_init,n_a,model,fhr,lat,lon,vmax,mslp,stype = lineArray[:11]
                if model not in ["OFCL","OFCI"]: continue

                if len(forecasts) == 0:
                    forecasts = {
                        'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'cumulative_ace':[],'type':[],'init':dt.strptime(run_init,'%Y%m%d%H')
                    }

                #Format lat & lon
                fhr = int(fhr)
                if "N" in lat:
                    lat_temp = lat.split("N")[0]
                    lat = np.round(float(lat_temp) * 0.1,1)
                elif "S" in lat:
                    lat_temp = lat.split("S")[0]
                    lat = np.round(float(lat_temp) * -0.1,1)
                if "W" in lon:
                    lon_temp = lon.split("W")[0]
                    lon = np.round(float(lon_temp) * -0.1,1)
                elif "E" in lon:
                    lon_temp = lon.split("E")[0]
                    lon = np.round(float(lon_temp) * 0.1,1)

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
                if fhr not in forecasts['fhr']:
                    if model in ['OFCL','OFCI'] and fhr > 120:
                        pass
                    else:
                        if lat == 0.0 and lon == 0.0:
                            continue
                        forecasts['fhr'].append(fhr)
                        forecasts['lat'].append(lat)
                        forecasts['lon'].append(lon)
                        forecasts['vmax'].append(vmax)
                        forecasts['mslp'].append(mslp)

                        #Get storm type, if it can be determined
                        if stype in ['','DB'] and vmax != 0 and np.isnan(vmax) == False:
                            stype = get_storm_type(vmax,False)
                        forecasts['type'].append(stype)
        
        #Retrieve JTWC forecast otherwise
        else:
            
            #Get forecast for this storm
            url = f"https://www.nrlmry.navy.mil/atcf_web/docs/current_storms/{self.id.lower()}.sum"
            if requests.get(url).status_code != 200: raise RuntimeError("JTWC forecast data is unavailable for this storm.")
            
            #Read file content
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            content = content.split("\n")
            f.close()

            #Iterate through every line in content:
            run_init = content[2].split(" ")[0]
            forecasts = {}

            for line in content[3:]:
                
                #Exit once done retrieving forecast
                if line == "AMP": break
                
                #Get basic components
                lineArray = line.split(" ")
                if len(lineArray) < 4: continue
                
                #Exit once done retrieving forecast
                if lineArray[0] == "AMP": break
                    
                if len(forecasts) == 0:
                    forecasts = {
                        'fhr':[],'lat':[],'lon':[],'vmax':[],'mslp':[],'cumulative_ace':[],'type':[],'init':dt.strptime(run_init,'%Y%m%d%H')
                    }

                #Forecast hour
                fhr = int(lineArray[0].split("T")[1])
                
                #Format lat & lon
                lat = lineArray[1]
                lon = lineArray[2]
                if "N" in lat:
                    lat_temp = lat.split("N")[0]
                    lat = np.round(float(lat_temp) * 0.1,1)
                elif "S" in lat:
                    lat_temp = lat.split("S")[0]
                    lat = np.round(float(lat_temp) * -0.1,1)
                if "W" in lon:
                    lon_temp = lon.split("W")[0]
                    lon = np.round(float(lon_temp) * -0.1,1)
                elif "E" in lon:
                    lon_temp = lon.split("E")[0]
                    lon = np.round(float(lon_temp) * 0.1,1)

                #Format vmax & MSLP
                vmax = int(lineArray[3])
                if vmax < 10 or vmax > 300: vmax = np.nan
                mslp = np.nan

                #Add forecast data to dict if forecast hour isn't already there
                if fhr not in forecasts['fhr']:
                    if lat == 0.0 and lon == 0.0: continue
                    forecasts['fhr'].append(fhr)
                    forecasts['lat'].append(lat)
                    forecasts['lon'].append(lon)
                    forecasts['vmax'].append(vmax)
                    forecasts['mslp'].append(mslp)

                    #Get storm type, if it can be determined
                    stype = get_storm_type(vmax,False)
                    forecasts['type'].append(stype)
            
        #Determine ACE thus far (prior to initial forecast hour)
        ace = 0.0
        for i in range(len(self.date)):
            if self.date[i] >= forecasts['init']: continue
            if self.type[i] not in ['TS','SS','HU']: continue
            ace += accumulated_cyclone_energy(self.vmax[i],hours=6)
        
        #Add initial forecast hour ACE
        ace += accumulated_cyclone_energy(forecasts['vmax'][0],hours=6)
        forecasts['cumulative_ace'].append(np.round(ace,1))
        
        #Interpolate forecast to 6-hour increments
        def temporal_interpolation(value, orig_times, target_times, kind='linear'):
            f = interp.interp1d(orig_times,value,kind=kind,fill_value='extrapolate')
            ynew = f(target_times)
            return ynew
        interp_fhr = range(0,forecasts['fhr'][-1]+1,6) #Construct a 6-hour time range
        interp_vmax = temporal_interpolation(forecasts['vmax'],forecasts['fhr'],interp_fhr)
        
        #Interpolate storm type
        interp_type = []
        for dummy_i,(i_hour,i_vmax) in enumerate(zip(interp_fhr,interp_vmax)):
            use_i = 0
            for i in range(len(forecasts['fhr'])):
                if forecasts['fhr'][i] > i_hour:
                    break
                use_i = int(i + 0.0)
            i_type = forecasts['type'][use_i]
            if i_type in ['TD','TS','SD','SS','HU','TY']: i_type = get_storm_type(i_vmax,False)
            interp_type.append(i_type)
        
        #Add forecast ACE
        for i,(i_fhr,i_vmax,i_type) in enumerate(zip(interp_fhr[1:],interp_vmax[1:],interp_type[1:])):
            
            #Add ACE if storm is a TC
            if i_type in ['TS','SS','HU','TY']:
                ace += accumulated_cyclone_energy(i_vmax,hours=6)
            
            #Add ACE to array
            if i_fhr in forecasts['fhr']:
                forecasts['cumulative_ace'].append(np.round(ace,1))
        
        #Save forecast as attribute
        self.latest_forecast = forecasts
        return self.latest_forecast

    def plot_forecast_realtime(self,track_labels='fhr',cone_days=5,domain="dynamic_forecast",
                                   ax=None,return_ax=False,cartopy_proj=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Plots the latest available official forecast. Available for both NHC and JTWC sources.
        
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
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
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
            nhc_forecasts = self.get_forecast_realtime()
        
        #Add other info to forecast dict
        nhc_forecasts['advisory_num'] = -1
        nhc_forecasts['basin'] = self.basin
        if self.source != "hurdat": nhc_forecasts['cone'] = False
        
        #Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(nhc_forecasts,self.dict,track_labels,cone_days,domain,ax=ax,return_ax=return_ax,save_path=save_path,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
        
    def get_realtime_info(self,source='all'):
        
        r"""
        Returns a dict containing the latest available information about the storm. This function uses NHC Public Advisories, so it will differ from available Best Track data.
        
        Parameters
        ----------
        source : str
            Data source to use. Default is "all". Available options are:
            
            * **"all"** = Latest from either public advisory or best track. Both NHC & JTWC.
            * **"public_advisory"** = Latest public advisory. NHC only.
            * **"best_track"** = Latest Best Track file data. Both NHC & JTWC.
        
        Returns
        -------
        dict
            Dictionary containing current storm information.
        """
        
        #Error check
        if isinstance(source,str) == False:
            msg = "\"source\" must be of type str."
            raise TypeError(msg)
        if source not in ['all','public_advisory','best_track']:
            msg = "\"source\" must be 'all', 'public_advisory', or 'best_track'."
            raise ValueError(msg)
        if source == 'public_advisory' and self.source != 'hurdat':
            msg = "A source of 'public_advisory' can only be used for storms in NHC's area of responsibility."
            raise RuntimeError(msg)
        
        #Declare empty dict
        current_advisory = {}
        
        #If source is all, determine which method to use
        if source == 'all':
            if self.source == 'hurdat':
                #Check to see which is the latest advisory
                latest_btk = self.date[-1]
                
                #Get latest available public advisory
                f = urllib.request.urlopen(f"https://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml")
                content = f.read()
                content = content.decode("utf-8")
                content = content.split("\n")
                f.close()
                
                #Get UTC time of advisory
                results = [i for i in content if 'messageDateTimeUTC' in i][0]
                result = (results.split(">")[1]).split("<")[0]
                latest_advisory = dt.strptime(result,'%Y%m%d %I:%M:%S %p UTC')
                
                #Check which one to use
                if latest_btk > latest_advisory:
                    source = 'best_track'
                else:
                    source = 'public_advisory'
            else:
                source = 'best_track'
        
        #If public advisory, retrieve this data
        if source == 'public_advisory':
            
            #Add source
            current_advisory['source'] = 'NHC Public Advisory'

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
            current_advisory['mslp'] = np.int(result)

            #Get storm category
            current_advisory['category'] = wind_to_category(current_advisory['wind_kt'])

            #Get storm direction
            results = [i for i in content if 'systemDirectionOfMotion' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['motion_direction'] = result.split(" OR ")[0]
            results = [i for i in content if 'systemDirectionOfMotion' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            try:
                current_advisory['motion_direction_degrees'] = int((result.split(" OR ")[1]).split(" DEGREES")[0])
            except:
                current_advisory['motion_direction_degrees'] = 0

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
            
        #Best track data
        else:
            
            #Add source
            if self.source == 'hurdat':
                current_advisory['source'] = 'NHC Best Track'
            else:
                current_advisory['source'] = 'JTWC Best Track'

            #Get public advisory number
            current_advisory['advisory_number'] = 'n/a'

            #Get UTC time of advisory
            current_advisory['time_utc'] = self.date[-1]

            #Get storm type
            subtrop_flag = True if self.type[-1] in ['SS','SD'] else False
            current_advisory['type'] = get_storm_classification(self.vmax[-1],subtrop_flag,self.basin)

            #Get storm name
            current_advisory['name'] = self.name.title()

            #Get coordinates
            current_advisory['lat'] = self.lat[-1]
            current_advisory['lon'] = self.lon[-1]

            #Get sustained wind speed
            current_advisory['wind_mph'] = knots_to_mph(self.vmax[-1])
            current_advisory['wind_kph'] = int(self.vmax[-1] * 1.852)
            current_advisory['wind_kt'] = self.vmax[-1]

            #Get MSLP
            current_advisory['mslp'] = np.int(self.mslp[-1])

            #Get storm category
            current_advisory['category'] = wind_to_category(current_advisory['wind_kt'])

            #Determine motion direction and degrees
            try:
                
                #Cannot calculate motion if there's only one data point
                if len(self.lon) == 1:
                    
                    #Get storm direction
                    current_advisory['motion_direction'] = 'n/a'
                    current_advisory['motion_direction_degrees'] = 'n/a'

                    #Get storm speed
                    current_advisory['motion_mph'] = 'n/a'
                    current_advisory['motion_kph'] = 'n/a'
                    current_advisory['motion_kt'] = 'n/a'
                
                #Otherwise, use great_circle to calculate
                else:
                    
                    #Get points
                    start_point = (self.lat[-2],self.lon[-2])
                    end_point = (self.lat[-1],self.lon[-1])
                    
                    #Get time since last update
                    hour_diff = (self.date[-1] - self.date[-2]).total_seconds() / 3600.0
                    
                    #Calculate zonal and meridional position change in km
                    x = great_circle((self.lat[-2],self.lon[-2]), (self.lat[-2],self.lon[-1])).kilometers
                    if self.lon[-1] < self.lon[-2]: x = x * -1

                    y = great_circle((self.lat[-2],self.lon[-2]), (self.lat[-1],self.lon[-2])).kilometers
                    if self.lat[-1] < self.lat[-2]: y = y * -1

                    #Calculate motion direction vector
                    idir = np.degrees(np.arctan2(x,y))
                    if idir < 0: idir += 360.0
                    
                    #Calculate motion direction string
                    def deg_str(d):
                        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                        ix = int((d + 11.25)/22.5)
                        return dirs[ix % 16]
                    dirs = deg_str(idir)

                    #Update storm direction
                    current_advisory['motion_direction'] = dirs
                    current_advisory['motion_direction_degrees'] = int(np.round(idir,0))

                    #Get storm speed
                    current_advisory['motion_mph'] = int(np.round(great_circle(start_point,end_point).miles / float(hour_diff),0))
                    current_advisory['motion_kph'] = int(np.round(great_circle(start_point,end_point).kilometers / float(hour_diff),0))
                    current_advisory['motion_kt'] = int(np.round(current_advisory['motion_mph'] * 0.868976,0))
                
            #Otherwise, can't calculate motion
            except:
                
                #Get storm direction
                current_advisory['motion_direction'] = 'n/a'
                current_advisory['motion_direction_degrees'] = 'n/a'

                #Get storm speed
                current_advisory['motion_mph'] = 'n/a'
                current_advisory['motion_kph'] = 'n/a'
                current_advisory['motion_kt'] = 'n/a'
        
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
        disco_time = num_to_str2(int(hr)) + ' '.join(content[6].split(" ")[1:])

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
        
    def __get_ensembles_eps(self):
        #Here it is:
        #ftp://wmo:essential@dissemination.ecmwf.int/20200518120000/
        #A_JSXX01ECEP181200_C_ECMP_20200518120000_tropical_cyclone_track_AMPHAN_86p3degE_14degN_bufr4.bin
        return
