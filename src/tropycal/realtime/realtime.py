r"""Functionality for managing real-time tropical cyclone data."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warn_message = "Warning: Cartopy is not installed in your python environment. Plotting functions will not work."
    warnings.warn(warn_message)

from .storm import RealtimeStorm

#Import tools
from ..utils import *

class Realtime():
    
    r"""
    Creates an instance of a Realtime object containing currently active tropical cyclones.
    
    Realtime objects are used to retrieve RealtimeStorm objects, which are created for every active tropical cyclone globally upon creating an instance of Realtime.
    
    Data sources are as follows:
    National Hurricane Center: https://ftp.nhc.noaa.gov/atcf/btk/
    Joint Typhoon Warning Center: http://hurricanes.ral.ucar.edu/repository/data/bdecks_open/

    Returns
    -------
    tropycal.realtime.Realtime
        An instance of Realtime.
    """
    
    def __repr__(self):
         
        summary = ["<tropycal.realtime.Realtime>"]

        #Add general summary
        
        #Add dataset summary
        summary.append("Dataset Summary:")
        summary.append(f'{" "*4}Numbers of active storms: {len(self.storms)}')
        
        if len(self.storms) > 0:
            summary.append("\nActive Storms:")
            for key in self.storms:
                summary.append(f'{" "*4}{key}')
        
        return "\n".join(summary)

    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __init__(self):
        
        #Define empty dict to store track data in
        self.data = {}
        
        #Time data reading
        start_time = dt.now()
        print("--> Starting to read in current storm data")
        
        #Read in best track data
        self.__read_btk()
        self.__read_btk_jtwc()
        
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed reading in current storm data ({tsec} seconds)")
        
        #Remove storms that haven't been active in 18 hours
        all_keys = [k for k in self.data.keys()]
        for key in all_keys:
            
            #Filter for storm duration
            if len(self.data[key]['date']) == 0:
                del self.data[key]
                continue
            
            #Get last date
            last_date = self.data[key]['date'][-1]
            current_date = dt.utcnow()
            
            #Get time difference
            hours_diff = (current_date - last_date).total_seconds() / 3600.0
            if hours_diff >= 18.0: del self.data[key]
        
        #For each storm remaining, create a Storm object
        if len(self.data) > 0:
            for key in self.data.keys():
                self[key] = RealtimeStorm(self.data[key])

            #Delete data dict while retaining active storm keys
            self.storms = [k for k in self.data.keys()]
            del self.data
        else:
            
            #Create an empty list signaling no active storms
            self.storms = []
            del self.data
    
    def __read_btk(self):
        
        r"""
        Reads in best track data into the Dataset object.
        """
        
        #Get current year
        current_year = (dt.now()).year

        #Get list of files in online directory
        use_ftp = False
        try:
            urlpath = urllib.request.urlopen('https://ftp.nhc.noaa.gov/atcf/btk/')
            string = urlpath.read().decode('utf-8')
        except:
            use_ftp = True
            urlpath = urllib.request.urlopen('ftp://ftp.nhc.noaa.gov/atcf/btk/')
            string = urlpath.read().decode('utf-8')

        #Get relevant filenames from directory
        files = []
        search_pattern = f'b[aec][lp][01234][0123456789]{current_year}.dat'

        pattern = re.compile(search_pattern)
        filelist = pattern.findall(string)
        for filename in filelist:
            if filename not in files: files.append(filename)

        #For each file, read in file content and add to hurdat dict
        for file in files:

            #Get file ID
            stormid = ((file.split(".dat")[0])[1:]).upper()

            #Determine basin
            add_basin = 'north_atlantic'
            if stormid[0] == 'C':
                add_basin = 'east_pacific'
            elif stormid[0] == 'E':
                add_basin = 'east_pacific'

            #add empty entry into dict
            self.data[stormid] = {'id':stormid,'operational_id':stormid,'name':'','year':int(stormid[4:8]),'season':int(stormid[4:8]),'basin':add_basin,'source_info':'NHC Hurricane Database','realtime':True,'source_method':"NHC's Automated Tropical Cyclone Forecasting System (ATCF)",'source_url':"https://ftp.nhc.noaa.gov/atcf/btk/"}
            self.data[stormid]['source'] = 'hurdat'

            #add empty lists
            for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
                self.data[stormid][val] = []
            self.data[stormid]['ace'] = 0.0

            #Read in file
            if use_ftp == True:
                url = f"ftp://ftp.nhc.noaa.gov/atcf/btk/{file}"
            else:
                url = f"https://ftp.nhc.noaa.gov/atcf/btk/{file}"
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            content = content.split("\n")
            content = [(i.replace(" ","")).split(",") for i in content]
            f.close()

            #iterate through file lines
            for line in content:

                if len(line) < 28: continue

                #Get date of obs
                date = dt.strptime(line[2],'%Y%m%d%H')
                if date.hour not in [0,6,12,18]: continue

                #Ensure obs aren't being repeated
                if date in self.data[stormid]['date']: continue

                #Get latitude into number
                btk_lat_temp = line[6].split("N")[0]
                btk_lat = float(btk_lat_temp) * 0.1

                #Get longitude into number
                if "W" in line[7]:
                    btk_lon_temp = line[7].split("W")[0]
                    btk_lon = float(btk_lon_temp) * -0.1
                elif "E" in line[7]:
                    btk_lon_temp = line[7].split("E")[0]
                    btk_lon = float(btk_lon_temp) * 0.1

                #Get other relevant variables
                btk_wind = int(line[8])
                btk_mslp = int(line[9])
                btk_type = line[10]
                name = line[27]

                #Replace with NaNs
                if btk_wind > 250 or btk_wind < 10: btk_wind = np.nan
                if btk_mslp > 1040 or btk_mslp < 800: btk_mslp = np.nan

                #Add extra obs
                self.data[stormid]['extra_obs'].append(0)

                #Append into dict
                self.data[stormid]['date'].append(date)
                self.data[stormid]['special'].append('')
                self.data[stormid]['type'].append(btk_type)
                self.data[stormid]['lat'].append(btk_lat)
                self.data[stormid]['lon'].append(btk_lon)
                self.data[stormid]['vmax'].append(btk_wind)
                self.data[stormid]['mslp'].append(btk_mslp)
                
                #Add basin
                if add_basin == 'north_atlantic':
                    wmo_agency = 'north_atlantic'
                elif add_basin == 'east_pacific':
                    if btk_lon > 0.0:
                        wmo_agency = 'west_pacific'
                    else:
                        wmo_agency = 'east_pacific'
                else:
                    wmo_agency = 'west_pacific'
                self.data[stormid]['wmo_basin'].append(wmo_agency)

                #Calculate ACE & append to storm total
                if np.isnan(btk_wind) == False:
                    ace = (10**-4) * (btk_wind**2)
                    if btk_type in ['SS','TS','HU']:
                        self.data[stormid]['ace'] += np.round(ace,4)

            #Add storm name
            self.data[stormid]['name'] = name
        
        
    def __read_btk_jtwc(self):
        
        r"""
        Reads in b-deck data from the Tropical Cyclone Guidance Project (TCGP) into the Dataset object.
        """

        #Get current year
        current_year = (dt.now()).year

        #Get list of files in online directory
        urlpath = urllib.request.urlopen(f'https://www.nrlmry.navy.mil/atcf_web/docs/tracks/{current_year}/')
        string = urlpath.read().decode('utf-8')

        #Get relevant filenames from directory
        files = []
        search_pattern = f'b[isw][ohp][01234][0123456789]{current_year}.dat'

        pattern = re.compile(search_pattern)
        filelist = pattern.findall(string)
        for filename in filelist:
            if filename not in files: files.append(filename)
        
        #Search for following year (for SH storms)
        search_pattern = f'b[isw][ohp][01234][0123456789]{current_year+1}.dat'

        pattern = re.compile(search_pattern)
        filelist = pattern.findall(string)
        for filename in filelist:
            if filename not in files: files.append(filename)

        #For each file, read in file content and add to hurdat dict
        for file in files:

            #Get file ID
            stormid = ((file.split(".dat")[0])[1:]).upper()

            #Determine basin based on where storm developed
            add_basin = 'west_pacific'
            if stormid[0] == 'I':
                add_basin = 'north_indian'
            elif stormid[0] == 'S':
                add_basin = ''

            #add empty entry into dict
            self.data[stormid] = {'id':stormid,'operational_id':stormid,'name':'','year':int(stormid[4:8]),'season':int(stormid[4:8]),'basin':add_basin,'source_info':'Joint Typhoon Warning Center','realtime':True,'source_method':"JTWC ATCF",'source_url':f'https://www.nrlmry.navy.mil/atcf_web/docs/tracks/{current_year}/'}
            self.data[stormid]['source'] = 'jtwc'

            #add empty lists
            for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
                self.data[stormid][val] = []
            self.data[stormid]['ace'] = 0.0

            #Read in file
            url = f"https://www.nrlmry.navy.mil/atcf_web/docs/tracks/{current_year}/{file}"
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            content = content.split("\n")
            content = [(i.replace(" ","")).split(",") for i in content]
            f.close()

            #iterate through file lines
            for line in content:

                if len(line) < 28: continue

                #Get date of obs
                date = dt.strptime(line[2],'%Y%m%d%H')
                if date.hour not in [0,6,12,18]: continue

                #Ensure obs aren't being repeated
                if date in self.data[stormid]['date']: continue

                #Get latitude into number
                if "N" in line[6]:
                    btk_lat_temp = line[6].split("N")[0]
                    btk_lat = np.round(float(btk_lat_temp) * 0.1,1)
                elif "S" in line[6]:
                    btk_lat_temp = line[6].split("S")[0]
                    btk_lat = np.round(float(btk_lat_temp) * -0.1,1)
                
                #Get longitude into number
                if "W" in line[7]:
                    btk_lon_temp = line[7].split("W")[0]
                    btk_lon = np.round(float(btk_lon_temp) * -0.1,1)
                elif "E" in line[7]:
                    btk_lon_temp = line[7].split("E")[0]
                    btk_lon = np.round(float(btk_lon_temp) * 0.1,1)

                #Determine basin if unknown
                if add_basin == '':
                    add_basin = get_basin(btk_lat,btk_lon)
                    self.data[stormid]['basin'] = add_basin
                
                #Get other relevant variables
                btk_wind = int(line[8])
                btk_mslp = int(line[9])
                btk_type = line[10]
                if btk_type == "TY": btk_type = "HU"
                name = line[27]

                #Replace with NaNs
                if btk_wind > 250 or btk_wind < 10: btk_wind = np.nan
                if btk_mslp > 1040 or btk_mslp < 800: btk_mslp = np.nan

                #Add extra obs
                self.data[stormid]['extra_obs'].append(0)

                #Append into dict
                self.data[stormid]['date'].append(date)
                self.data[stormid]['special'].append('')
                self.data[stormid]['type'].append(btk_type)
                self.data[stormid]['lat'].append(btk_lat)
                self.data[stormid]['lon'].append(btk_lon)
                self.data[stormid]['vmax'].append(btk_wind)
                self.data[stormid]['mslp'].append(btk_mslp)
                
                #Add basin
                self.data[stormid]['wmo_basin'].append(add_basin)

                #Calculate ACE & append to storm total
                if np.isnan(btk_wind) == False:
                    ace = (10**-4) * (btk_wind**2)
                    if btk_type in ['SS','TS','HU']:
                        self.data[stormid]['ace'] += np.round(ace,4)

            #Add storm name
            self.data[stormid]['name'] = name
            
    def list_active_storms(self):
        
        r"""
        Produces a list of storms currently stored in Realtime.
        
        Returns
        -------
        list
            List containing the storm IDs for currently active storms. Each ID has a Storm object stored as an attribute of Realtime.
        """
        
        return self.storms
    
    def get_storm(self,storm):
        
        r"""
        Returns a RealtimeStorm object for the requested storm ID.
        
        Parameters
        ----------
        storm : str
            Storm ID for the requested storm (e.g., "AL012020").
        
        Returns
        -------
        tropycal.realtime.RealtimeStorm
            An instance of RealtimeStorm.
        """
        
        #Check to see if storm is available
        if isinstance(storm,str) == False:
            msg = "\"storm\" must be of type str."
            raise TypeError(msg)
        if storm not in self.storms:
            msg = "Requested storm ID is not contained in this object."
            raise RuntimeError(msg)
        
        #Return RealtimeStorm object
        return self[storm]

