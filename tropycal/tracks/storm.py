import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from .plot import Plot
from .tools import *

try:
    import gzip
    from io import StringIO, BytesIO
    import tarfile
except:
    warnings.warn("Warning: The libraries necessary for online NHC forecast retrieval aren't available (gzip, io, tarfile).")


class Storm:
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):
         
        summary = ["<tropycal.Storm>"]

        summary.append("Storm Summary:")
        summary_keys = {'Maximum Wind':np.nanmax(self.dict['vmax']),
                        'Minimum pressure':np.nanmin(self.dict['mslp'])}
        for key in summary_keys.keys():
            summary.append(f"{key:>4}: {summary_keys[key]}")
        
        summary.append("\nMore Information:")
        for key in self.coords.keys():
            summary.append(f"{key:>4}: {self.coords[key]}")

        return "\n".join(summary)
    
    def __init__(self,storm):
        
        r"""
        Initializes an instance of Storm.
        
        Parameters:
        -----------
        storm : dict
            Dict entry of the requested storm.
        
        returns:
        --------
        Storm
            Instance of a Storm object.
        """
        
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
        
        r""" Numpydoc (generate autodoc/autosummary)
        Converts the storm dict into an xarray Dataset object.
        
        Returns
        -------
        xarray Dataset
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
        xarray Dataset
            A pandas DataFrame object containing information about the storm.
        """
        
        #Try importing xarray
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
        Creates a plot of the storm.
        
        Parameters
        ----------
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic" - default. Dynamically focuses the domain using the storm track(s) plotted.
            "north_atlantic" - North Atlantic Ocean basin
            "pacific" - East/Central Pacific Ocean basin
            "latW/latE/lonS/lonN" - Custom plot domain
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
        self.plot_obj = Plot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(self.dict['lon']) > 150 or min(self.dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            
        #Plot storm
        return_ax = self.plot_obj.plot_storm(self.dict,zoom,plot_all,ax,prop,map_prop)
        
        #Return axis
        if ax != None: return return_ax
        
    #PLOT FUNCTION FOR HURDAT
    def plot_nhc_forecast(self,forecast,cone_days=5,zoom="dynamic_forecast",ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of the operational NHC forecast track along with observed track data.
        
        Parameters
        ----------
        forecast : int or datetime.datetime
            Integer representing the forecast number, or datetime object for the closest issued forecast to this date.
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic_forecast" - default. Dynamically focuses the domain on the forecast track.
            "dynamic" - Dynamically focuses the domain on the combined observed and forecast track.
            "latW/latE/lonS/lonN" - Custom plot domain
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
        self.plot_obj = Plot()
        
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
            self.forecast_dict = self.get_operational_forecasts()

        #Get all NHC forecast entries
        nhc_forecasts = self.forecast_dict['OFCL']

        #Get list of all NHC forecast initializations
        nhc_forecast_init = [k for k in nhc_forecasts.keys()]

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
        for k in nhc_forecast_init:
            hrs = nhc_forecasts[k]['fhr']
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
        
        #Cap track, if specified
        #if cap_track == True:
        #    forecast_dict['cap_forecast'] = True
        #else:
        #    forecast_dict['cap_forecast'] = False
            
        #Add other info to forecast dict
        forecast_dict['advisory_num'] = advisory_num
        forecast_dict['basin'] = self.basin
            
        #Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(forecast_dict,track_dict,cone_days,zoom,ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None: return plot_ax
        
    
    def list_nhc_discussions(self):
        
        r"""
        Retrieves a list of NHC forecast discussions for this storm, archived on https://ftp.nhc.noaa.gov/atcf/archive/.
        
        Returns:
        --------
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
        elif storm_year < 2000:
            pass
        elif storm_year == 2000:
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
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
        return discos
        
    def get_nhc_discussion(self,time=None,disco_id=None,save_path=None):
        
        r"""
        Retrieves a single NHC forecast discussion. Provide either time or disco_id for the input argument.
        
        Parameters:
        -----------
        time : datetime.datetime
            Datetime object representing the desired forecast discussion time. If this argument is provided, disco_id must be of type None.
        disco_id : int
            Forecast discussion ID. If this argument is provided, time must be of type None.
        save_path : str
            Directory path to save the forecast discussion text to.
        
        Returns:
        --------
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
        if time != None and disco_id != None:
            raise RuntimeError("Error: Cannot provide both time and disco_id arguments.")
        
        #Get list of storm discussions
        disco_dict = self.list_nhc_discussions()
        
        if time != None:
            #Find closest discussion to the time provided
            disco_times = disco_dict['utc_date']
            disco_ids = [int(i) for i in disco_dict['id']]
            disco_diff = np.array([(i-time).days + (i-time).seconds/86400 for i in disco_times])
            closest_idx = np.abs(disco_diff).argmin()
            closest_diff = disco_diff[closest_idx]
            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]
        
            #Raise warning if difference is >=1 day
            if np.abs(closest_diff) >= 1.0:
                warnings.warn(f"The date provided is outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion.")
                
        if disco_id != None:
            #Find closest discussion ID to the one provided
            disco_times = disco_dict['utc_date']
            disco_ids = [int(i) for i in disco_dict['id']]
            disco_diff = np.array([i-disco_id for i in disco_ids])
            closest_idx = np.abs(disco_diff).argmin()
            closest_diff = disco_diff[closest_idx]
            closest_id = disco_ids[closest_idx]
            closest_time = disco_times[closest_idx]
        
            #Raise warning if difference is >=1 ids
            if np.abs(closest_diff) >= 2.0:
                warnings.warn(f"The ID provided is outside of the duration of the storm. Use the \"list_nhc_discussions()\" function to retrieve a list of available NHC discussions for this storm. Returning the closest available NHC discussion.")

        #Read content of NHC forecast discussion
        if disco_dict['mode'] == 0:
            url_disco = disco_dict['url'][closest_idx]
            f = urllib.request.urlopen(url_disco)
            content = f.read()
            content = content.decode("utf-8")
            f.close()
            
        elif disco_dict['mode'] in [1,2,3]:
            #Get directory path of storm and read it in
            url_disco = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/messages/"
            url = url_disco + f'{storm_id.lower()}_msgs.tar.gz'
            if disco_dict['mode'] in [2,3]: url = url_disco + f'{storm_id.lower()}.msgs.tar.gz'
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
        https://www.ftp.ncep.noaa.gov/data/nccf/com/ens_tracker/prod/

        Returns:
        --------
        dict
            Dictionary containing all forecast entries.
        """
        
        #Check to ensure the data source is HURDAT
        if self.source != "hurdat":
            raise RuntimeError("Error: NHC data can only be accessed when HURDAT is used as the data source.")
            
        #If forecasts dict already exist, simply return the dict
        try:
            self.forecast_dict
            return self.forecast_dict
        except:
            pass

        #Get storm ID & corresponding data URL
        storm_id = self.dict['operational_id']
        storm_year = self.dict['year']

        if storm_year == (dt.now()).year:
            url_models = f"https://ftp.nhc.noaa.gov/atcf/aid_public/a{storm_id.lower()}.dat.gz"
        else:
            url_models = f"https://ftp.nhc.noaa.gov/atcf/archive/{storm_year}/a{storm_id.lower()}.dat.gz"

        #Retrieve model data text
        request = urllib.request.Request(url_models)
        response = urllib.request.urlopen(request)
        sio_buffer = BytesIO(response.read())
        gzf = gzip.GzipFile(fileobj = sio_buffer)
        data = gzf.read()
        content = data.splitlines()
        content = [(i.decode()).split(",") for i in content]
        content = [i for i in content if len(i) > 10]
        response.close()

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
            vmax = int(vmax)
            if vmax < 10 or vmax > 300: vmax = np.nan
            mslp = int(mslp)
            if mslp < 1: mslp = np.nan

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
                    
                    if stype == '' and vmax != 0 and np.isnan(vmax) == False:
                        stype = get_type(vmax,False)
                    forecasts[model][run_init]['type'].append(stype)

        #Save dict locally
        self.forecast_dict = forecasts
        
        #Return dict
        return forecasts

