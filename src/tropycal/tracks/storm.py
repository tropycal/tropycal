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

from .plot import TrackPlot
from .tools import *
from ..tornado import *
from ..recon import *

try:
    import zipfile
    import gzip
    from io import StringIO, BytesIO
    import tarfile
except:
    warnings.warn("Warning: The libraries necessary for online NHC forecast retrieval aren't available (gzip, io, tarfile).")


class Storm:
    
    r"""
    Initializes an instance of Storm, retrieved via ``TrackDataset.get_storm()``.

    Parameters
    ----------
    storm : dict
        Dict entry of the requested storm.

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
        max_wind = 'N/A' if all_nan(self.dict['vmax']) == True else np.nanmax(self.dict['vmax'])
        min_mslp = 'N/A' if all_nan(self.dict['mslp']) == True else np.nanmin(self.dict['mslp'])
        type_array = np.array(self.dict['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
        if len(idx[0]) == 0:
            start_date = 'N/A'
            end_date = 'N/A'
        else:
            time_tropical = np.array(self.dict['date'])[idx]
            start_date = time_tropical[0].strftime("%H00 UTC %d %B %Y")
            end_date = time_tropical[-1].strftime("%H00 UTC %d %B %Y")
        summary_keys = {'Maximum Wind':f"{max_wind} knots",
                        'Minimum Pressure':f"{min_mslp} hPa",
                        'Start Date':start_date,
                        'End Date':end_date}

        #Add storm summary
        summary.append("Storm Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')
        
        #Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.coords.keys()])+3
        for key in self.coords.keys():
            key_name = key+":"
            val = '%0.1f'%(self.coords[key]) if key == 'ace' else self.coords[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')

        return "\n".join(summary)
    
    def __init__(self,storm):
        
        #Save the dict entry of the storm
        self.dict = storm
        
        #Add other attributes about the storm
        keys = self.dict.keys()
        self.coords = {}
        for key in keys:
            if isinstance(self.dict[key], list) == False and isinstance(self.dict[key], dict) == False:
                self[key] = self.dict[key]
                self.coords[key] = self.dict[key]
                
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
    def plot(self,zoom="dynamic",plot_all=False,ax=None,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of the observed track of the storm.
        
        Parameters
        ----------
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the storm track plotted.
            * **(basin_name)** - Any of the acceptable basins (check "TrackDataset" for a list).
            * **lonW/lonE/latS/latN** - Custom plot domain
        plot_all : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Create instance of plot object
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
        return_ax = self.plot_obj.plot_storm(self.dict,zoom,plot_all,ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None: return return_ax
        
    #PLOT FUNCTION FOR HURDAT
    def plot_nhc_forecast(self,forecast,track_labels='fhr',cone_days=5,zoom="dynamic_forecast",
                          ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
        
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
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic_forecast** Default. Dynamically focuses the domain on the forecast track.
            * **dynamic** - Dynamically focuses the domain on the combined observed and forecast track.
            * **lonW/lonE/latS/latN** Custom plot domain.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
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
                if itype == "": itype = get_type(carq_forecasts[k]['vmax'][0],False)
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
                if itype == "": itype = get_type(nhc_forecasts[k]['vmax'][0],False)
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
        plot_ax = self.plot_obj.plot_storm_nhc(forecast_dict,track_dict,track_labels,cone_days,zoom,ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None: return plot_ax
        
    
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
            pass
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
                disco_time = str2(int(hr)) + ' '.join(content[4].split(" ")[1:])
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
            Datetime object representing the desired forecast discussion time, or integer representing the forecast discussion ID.
        save_path : str, optional
            Directory path to save the forecast discussion text to. If None (default), forecast won't be saved.
        
        Returns
        -------
        dict
            Dictionary containing the forecast discussion text and accompanying information about this discussion.
        """
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError("Error: NHC data can only be accessed when HURDAT is used as the data source.")
        
        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']
        
        #Error check
        if storm_id == '':
            raise RuntimeError("No NHC operational data is available for this storm.")
        
        #Error check
        if isinstance(forecast,int) == False and isinstance(forecast,dt) == False:
            raise TypeError("forecast must be of type int or datetime.datetime")
        
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
            disco_diff = np.array([i-forecast for i in disco_ids])
            closest_idx = np.abs(disco_diff).argmin()
            closest_diff = disco_diff[closest_idx]
            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]
        
            #Raise warning if difference is >=1 ids
            if np.abs(closest_diff) >= 2.0:
                warnings.warn(f"The ID provided is unavailable or outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion.")

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
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError("NHC data can only be accessed when HURDAT is used as the data source.")
            
        #If forecasts dict already exist, simply return the dict
        try:
            self.forecast_dict
            return self.forecast_dict
        except:
            pass

        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']
        if storm_year <= 2006: storm_id = self.dict['id']
        if storm_year < 1954:
            raise RuntimeError("Forecast data is unavailable for storms prior to 1954.")
        
        #Error check
        if storm_id == '':
            raise RuntimeError("No NHC operational data is available for this storm.")

        if storm_year == (dt.now()).year:
            url_models = f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{storm_id.lower()}.dat.gz"
        else:
            url_models = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/a{storm_id.lower()}.dat.gz"

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
            raise RuntimeError("No NHC operational data is available for this storm.")

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
                        stype = get_type(vmax,False)
                    forecasts[model][run_init]['type'].append(stype)

        #Save dict locally
        self.forecast_dict = forecasts
        
        #Return dict
        return forecasts
    
    
    def download_tcr(self,dir_path=""):
        
        r"""
        Downloads the NHC offical Tropical Cyclone Report (TCR) for this storm to the requested directory.
        
        Parameters
        ----------
        dir_path : str
            Path of directory to download the TCR into. Default is current working directory.
        """
        
        #Error check
        if self.source != "hurdat":
            raise RuntimeError("NHC data can only be accessed when HURDAT is used as the data source.")
        if self.year < 1995:
            raise RuntimeError("Tropical Cyclone Reports are unavailable prior to 1995.")
        
        #Format URL
        storm_id = self.dict['id'].upper()
        storm_name = self.dict['name'].title()
        url = f"https://www.nhc.noaa.gov/data/tcr/{storm_id}_{storm_name}.pdf"
        
        #Format filepath
        if dir_path != "" and dir_path[-1] != "/": dir_path += "/"
        
        #Retrieve PDF
        response = requests.get(url)
        with open(f"{dir_path}TCR_{storm_id}_{storm_name}.pdf", 'wb') as f:
            f.write(response.content)

            
    def plot_tors(self,dist_thresh=1000,Tors=None,zoom="dynamic",plotPPF=False,plot_all=False,\
                  ax=None,cartopy_proj=None,prop={},map_prop={}):
                
        r"""
        Creates a plot of the storm and associated tornado tracks.
        
        Parameters
        ----------
        Tors : pandas.DataFrame
            DataFrame containing tornado data associated with the storm. If None, data is automatically retrieved from TornadoDatabase. A dataframe of tornadoes associated with the TC will then be saved to this instance of storm
                for future use.
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the storm track(s) plotted and tornadoes it produced.
            * **(basin_name)** - Any of the acceptable basins (check "TrackDataset" for a list).
            * **lonW/lonE/latS/latN** - Custom plot domain
        plotPPF : False / True / "total" / "daily"
            Whether to plot practically perfect forecast (PPF). True defaults to "total". Default is False.
            
            * **total** - probability of a tornado within 25 miles of a point during the period of time selected.
            * **daily** - average probability of a tornado within 25 miles of a point during a day starting at 1200 UTC.
        plot_all : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default colormap for TC plots to Wistia
        try:
            prop['PPFcolors']
        except:
            prop['PPFcolors']='Wistia'
        
        if Tors == None:
            try:
                self.stormTors
            except:
                Tors = TornadoDataset()
                self.stormTors = Tors.getTCtors(self,dist_thresh)
    
        if len(self.stormTors)<5:
            warnings.warn(f"{len(self.stormTors)} tornadoes were found with this storm. Default zoom to east_conus.")
            zoom = 'east_conus'
    
        #Create instance of plot object
        self.plot_obj_tc = TrackPlot()
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
        tor_ax,zoom,leg_tor = self.plot_obj_tor.plot_tornadoes(self.stormTors,zoom,ax=ax,return_ax=True,\
                                             plotPPF=plotPPF,prop=prop,map_prop=map_prop)
        tor_title = tor_ax.get_title('left')
        
        #Plot storm
        return_ax = self.plot_obj_tc.plot_storm(self.dict,zoom,plot_all,ax=tor_ax,prop=prop,map_prop=map_prop)
        
        return_ax.add_artist(leg_tor)
        
        storm_title = return_ax.get_title('left')
        return_ax.set_title(f'{storm_title}\n{tor_title}',loc='left',fontsize=17,fontweight='bold')

        #Return axis
        if ax != None: return return_ax


    def plot_recon(self,stormRecon=None,recon_select=None,zoom="dynamic",barbs=True,scatter=False,plot_all=False,\
                  ax=None,cartopy_proj=None,prop={},map_prop={}):
                
        r"""
        Creates a plot of the storm and associated recon data.
        
        Parameters
        ----------
        StormRecon : tropycal.recon.ReconDataset
            An instance of ReconDataset for this storm. If none, one will be generated.
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the storm track(s) plotted.
            * **(basin_name)** - Any of the acceptable basins (check "TrackDataset" for a list).
            * **lonW/lonE/latS/latN** - Custom plot domain
        plot_all : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """

        #Read in reconaissance data for the storm
        if stormRecon == None and not isinstance(recon_select,pd.core.frame.DataFrame):
            try:
                self.stormRecon
            except:
                self.stormRecon = ReconDataset((self.name,self.year))
        
        if recon_select == None:
            dfRecon = self.stormRecon.recentered
        else:
            if isinstance(recon_select,pd.core.frame.DataFrame):
                dfRecon = recon_select
            elif isinstance(recon_select,dict):
                dfRecon = pd.DataFrame.from_dict(recon_select)
            elif isinstance(recon_select,str):
                dfRecon = self.stormRecon.missiondata[recon_select]
            else:
                dfRecon = self.stormRecon.__getSubTime(recon_select)
        
        
        #Create instance of plot object
        self.plot_obj_tc = TrackPlot()
        self.plot_obj_rec = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj_rec.create_cartopy(proj='PlateCarree',central_longitude=180.0)
                self.plot_obj_tc.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj_rec.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                self.plot_obj_tc.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                
        #Plot recon
        rec_ax,zoom = self.plot_obj_rec.plot_points(dfRecon,barbs=barbs,scatter=scatter,zoom=zoom,\
                                                    ax=ax,return_ax=True,prop=prop,map_prop=map_prop)
        rec_title = rec_ax.get_title('left')
        
        #Plot storm
        return_ax = self.plot_obj_tc.plot_storm(self.dict,zoom,plot_all,ax=rec_ax,prop=prop,map_prop=map_prop)
                
        storm_title = return_ax.get_title('left')
        return_ax.set_title(f'{storm_title}\n{rec_title}',loc='left',fontsize=17,fontweight='bold')

    
        #Return axis
        if ax != None: return return_ax


