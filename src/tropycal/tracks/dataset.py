r"""Functionality for storing and analyzing an entire cyclone dataset."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt
#Import other tropycal objects
from ..plot import Plot
from .plot import TrackPlot
from .storm import Storm
from .season import Season
from ..tornado import *

#Import tools
from .tools import *
from ..utils import *

#Import matplotlib
try:
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.dates as mdates
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class TrackDataset:
    
    r"""
    Creates an instance of a TrackDataset object containing various cyclone data.

    Parameters
    ----------
    basin : str
        Ocean basin to load data for. Can be any of the following:
        
        * **north_atlantic** - HURDAT2, ibtracs
        * **east_pacific** - HURDAT2, ibtracs
        * **west_pacific** - ibtracs
        * **north_indian** - ibtracs
        * **south_indian** - ibtracs
        * **australia** - ibtracs
        * **south_pacific** - ibtracs
        * **south_america** - ibtracs
        * **all** - ibtracs
    source : str
        Data source to read in. Default is HURDAT2.
        
        * **hurdat** - HURDAT2 data source for the North Atlantic and East/Central Pacific basins
        * **ibtracs** - ibtracs data source for regional or global data
    include_btk : bool, optional
        If True, the best track data from NHC for the most recent years where it doesn't exist in HURDAT2 will be added into the dataset. Valid for "north_atlantic" and "east_pacific" basins. Default is False.
    
    Other Parameters
    ----------------
    atlantic_url : str, optional
        URL containing the Atlantic HURDAT2 dataset. Can be changed to a local txt reference file. Default is retrieval from online URL.
    pacific_url : str, optional
        URL containing the Pacific HURDAT2 dataset. Can be changed to a local txt reference file. Default is retrieval from online URL.
    ibtracs_url : str, optional
        URL containing the ibtracs dataset. Can be changed to a local txt reference file. Can be a regional or all ibtracs file. If regional, the basin should match the argument basin provided earlier. Default is retrieval from online URL.
    catarina : bool, optional
        Modify the dataset to include cyclone track data for Cyclone Catarina (2004) from McTaggart-Cowan et al. (2006). Default is False.
    ibtracs_hurdat : bool, optional
        Replace ibtracs data for the North Atlantic and East/Central Pacific basins with HURDAT data. Default is False.
    ibtracs_mode : str, optional
        Mode of reading ibtracs in. Default is "jtwc".
        
        * **wmo** = official World Meteorological Organization data. Caveat is sustained wind methodology is inconsistent between basins.
        * **jtwc** = default. Unofficial data from the Joint Typhoon Warning Center. Caveat is some storms are missing and some storm data is inaccurate.
        * **jtwc_neumann** = JTWC data modified with the Neumann reanalysis for the Southern Hemisphere. Improves upon some storms (e.g., Cyclone Tracy 1974) while degrading others.

    Returns
    -------
    Dataset
        An instance of Dataset.
    """
 
    def __repr__(self):
         
        summary = ["<tropycal.tracks.Dataset>"]
        
        #Find maximum wind and minimum pressure
        max_wind = int(np.nanmax([x for stormid in self.keys for x in self.data[stormid]['vmax']]))
        max_wind_name = ""
        min_mslp = int(np.nanmin([x for stormid in self.keys for x in self.data[stormid]['mslp']]))
        min_mslp_name = ""
        
        for key in self.keys[::-1]:
            array_vmax = np.array(self.data[key]['vmax'])
            array_mslp = np.array(self.data[key]['mslp'])
            if len(array_vmax[~np.isnan(array_vmax)]) > 0 and np.nanmax(array_vmax) == max_wind:
                max_wind_name = f"{self.data[key]['name'].title()} {self.data[key]['year']}"
            if len(array_mslp[~np.isnan(array_mslp)]) > 0 and np.nanmin(array_mslp) == min_mslp:
                min_mslp_name = f"{self.data[key]['name'].title()} {self.data[key]['year']}"

        #Add general summary
        emdash = '\u2014'
        summary_keys = {'Basin':self.basin,\
                        'Source':self.source+[', '+self.ibtracs_mode,''][self.source=='hurdat'],\
                        'Number of storms':len(self.keys),\
                        'Maximum wind':f"{max_wind} knots ({max_wind_name})",
                        'Minimum pressure':f"{min_mslp} hPa ({min_mslp_name})",
                        'Year range':f"{self.data[self.keys[0]]['year']} {emdash} {self.data[self.keys[-1]]['year']}"}
        #Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)
    
    
    def __init__(self,basin='north_atlantic',source='hurdat',include_btk=False,doInterp=False,**kwargs):
        
        #kwargs
        atlantic_url = kwargs.pop('atlantic_url', 'https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html')
        pacific_url = kwargs.pop('pacific_url', 'https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-nepac.html')
        ibtracs_url = kwargs.pop('ibtracs_url', 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.(basin).list.v04r00.csv')
        ibtracs_mode = kwargs.pop('ibtracs_mode', 'jtwc')
        catarina = kwargs.pop('catarina', False)
        ibtracs_hurdat = kwargs.pop('ibtracs_hurdat', False)
        
        #Error check
        if ibtracs_mode not in ['wmo','jtwc','jtwc_neumann']:
            raise ValueError("ibtracs_mode must be either 'wmo', 'jtwc', or 'jtwc_neumann'")
        
        #Store input arguments
        self.proj = None #for plotting
        self.basin = basin.lower()
        self.atlantic_url = str(atlantic_url)
        self.pacific_url = str(pacific_url)
        self.ibtracs_url = str(ibtracs_url)
        self.source = source
        
        #Modification flags
        self.catarina = catarina
        self.ibtracs_mode = ibtracs_mode
        if ibtracs_mode == 'jtwc_neumann':
            self.neumann = True
        else:
            self.neumann = False
        
        #initialize empty dict
        self.data = {}
        
        #Read in from specified data source
        if source == 'hurdat':
            self.__read_hurdat()
        elif source == 'ibtracs':
            self.__read_ibtracs()
        else:
            raise RuntimeError("Accepted values for 'source' are 'hurdat' or 'ibtracs'")
            
        #Replace ibtracs with hurdat for atl/pac basins
        if source == 'ibtracs' and ibtracs_hurdat == True:
            if self.basin in ['north_atlantic','east_pacific']:
                self.__read_hurdat()
            elif self.basin == 'all':
                self.basin = 'both'
                self.__read_hurdat(override_basin=True)
                self.basin = 'all'
        
        #Read in best track data
        if include_btk == True and basin in ['north_atlantic','east_pacific']:
            self.__read_btk()
            
        #Add keys of all storms to object
        keys = self.data.keys()
        self.keys = [k for k in keys]
        
        #Placeholder for 2006 Pacific cyclone
        """
        if 'EP182006' in self.keys:
            data = {}
            for key in keys:
                data[key] = self.data[key]
                if key == 'EP182006':
                    data['CP052006'] = pac_2006_cyclone()
            self.data = data
            self.keys = [k for k in self.data.keys()]
         """
        
        #Create array of zero-ones for existence of tornado data for a given storm
        self.keys_tors = [0 for key in self.keys]
        
        #Add dict to store all storm-specific tornado data in
        self.data_tors = {}
        
        # If doInterp, interpolate each storm and save to dictionary.
        if doInterp:
            self.__getInterp()    
    
    def __read_hurdat(self,override_basin=False):
        
        r"""
        Reads in HURDATv2 data into the Dataset object.
        """
        
        #Time duration to read in HURDAT
        start_time = dt.now()
        print("--> Starting to read in HURDAT2 data")
        
        #Quick error check
        atl_online = True
        pac_online = True
        fcheck = "https://www.nhc.noaa.gov/data/hurdat/"
        fcheck2 = "https://www.aoml.noaa.gov/hrd/hurdat/"
        if fcheck not in self.atlantic_url and fcheck2 not in self.atlantic_url:
            if "http" in self.atlantic_url:
                raise RuntimeError("URL provided is not via NHC or HRD")
            else:
                atl_online = False
        if fcheck not in self.pacific_url and fcheck2 not in self.pacific_url:
            if "http" in self.pacific_url:
                raise RuntimeError("URL provided is not via NHC or HRD")
            else:
                pac_online = False
        
        #Check if basin is valid
        if self.basin.lower() not in ['north_atlantic','east_pacific','both']:
            raise RuntimeError("Only valid basins are 'north_atlantic', 'east_pacific' or 'both'")
        
        def read_hurdat(path,flag):
            if flag == True:
                f = urllib.request.urlopen(path)
                content = f.read()
                content = content.decode("utf-8")
                content = content.split("\n")
                content = [(i.replace(" ","")).split(",") for i in content]
                f.close()
            else:
                f = open(path,"r")
                content = f.readlines()
                content = [(i.replace(" ","")).split(",") for i in content]
                f.close()
            return content
        
        #read in HURDAT2 file from URL
        if self.basin == 'north_atlantic':
            content = read_hurdat(self.atlantic_url,atl_online)
        elif self.basin == 'east_pacific':
            content = read_hurdat(self.pacific_url,pac_online)
        elif self.basin == 'both':
            content = read_hurdat(self.atlantic_url,atl_online)
            content += read_hurdat(self.pacific_url,pac_online)
        
        #keep current storm ID for iteration
        current_id = ""
        
        #iterate through every line
        for line in content:
            
            #Skip if line is empty
            if len(line) < 2: continue
            if line[0][0] == "<": continue
            
            #identify if this is a header for a storm or content of storm
            if line[0][0] in ['A','C','E']:
                
                #Determine basin
                add_basin = 'north_atlantic'
                if line[0][0] == 'C':
                    add_basin = 'east_pacific'
                elif line[0][0] == 'E':
                    add_basin = 'east_pacific'
                if override_basin == True:
                    add_basin = 'all'
                
                #add empty entry into dict
                self.data[line[0]] = {'id':line[0],'operational_id':'','name':line[1],'year':int(line[0][4:]),'season':int(line[0][4:]),'basin':add_basin,'source_info':'NHC Hurricane Database'}
                self.data[line[0]]['source'] = self.source
                current_id = line[0]
                
                #add empty lists
                for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
                    self.data[line[0]][val] = []
                self.data[line[0]]['ace'] = 0.0
                
            #if not a header, enter storm info into its dict entry
            else:
                
                #Retrieve important info about storm
                yyyymmdd,hhmm,special,storm_type,lat,lon,vmax,mslp = line[0:8]
                
                #Parse into format to be entered into dict
                date = dt.strptime(yyyymmdd+hhmm,'%Y%m%d%H%M')
                if "N" in lat:
                    lat = float(lat.split("N")[0])
                elif "S" in lat:
                    lat = float(lat.split("N")[0]) * -1.0
                if "W" in lon:
                    lon = float(lon.split("W")[0]) * -1.0
                elif "E" in lon:
                    lon = float(lon.split("E")[0])
                vmax = int(vmax)
                mslp = int(mslp)
                
                #Handle missing data
                if vmax < 0: vmax = np.nan
                if mslp < 800: mslp = np.nan
                    
                #Handle off-hour obs
                if hhmm in ['0000','0600','1200','1800']:
                    self.data[current_id]['extra_obs'].append(0)
                else:
                    self.data[current_id]['extra_obs'].append(1)
                    
                #Fix storm type for cross-dateline storms
                storm_type = storm_type.replace("ST","HU")
                storm_type = storm_type.replace("TY","HU")
                
                #Append into dict
                self.data[current_id]['date'].append(date)
                self.data[current_id]['special'].append(special)
                self.data[current_id]['type'].append(storm_type)
                self.data[current_id]['lat'].append(lat)
                self.data[current_id]['lon'].append(lon)
                self.data[current_id]['vmax'].append(vmax)
                self.data[current_id]['mslp'].append(mslp)
                
                #Add basin
                if add_basin == 'north_atlantic':
                    wmo_agency = 'north_atlantic'
                elif add_basin == 'east_pacific':
                    if lon > 0.0:
                        wmo_agency = 'west_pacific'
                    else:
                        wmo_agency = 'east_pacific'
                else:
                    wmo_agency = 'west_pacific'
                self.data[current_id]['wmo_basin'].append(wmo_agency)
                
                #Calculate ACE & append to storm total
                if np.isnan(vmax) == False:
                    ace = (10**-4) * (vmax**2)
                    if hhmm in ['0000','0600','1200','1800'] and storm_type in ['SS','TS','HU']:
                        self.data[current_id]['ace'] += np.round(ace,4)
        
        #Account for operationally unnamed storms
        current_year = 0
        current_year_id = 1
        for key in self.data.keys():
            
            storm_data = self.data[key]
            storm_name = storm_data['name']
            storm_year = storm_data['year']
            storm_vmax = storm_data['vmax']
            storm_id = storm_data['id']
            
            #Get max wind for storm
            np_wnd = np.array(storm_vmax)
            if len(np_wnd[~np.isnan(np_wnd)]) == 0:
                max_wnd = np.nan
            else:
                max_wnd = int(np.nanmax(storm_vmax))
            
            #Fix current year
            if current_year == 0:
                current_year = storm_year
            else:
                if storm_year != current_year:
                    current_year = storm_year
                    current_year_id = 1
                    
                    #special fix for 1992 in Atlantic
                    if current_year == 1992 and self.data[current_id]['basin'] == 'north_atlantic':
                        current_year_id = 2
                
            #Estimate operational storm ID (which sometimes differs from HURDAT2 ID)
            blocked_list = []
            potential_tcs = ['AL102017']
            increment_but_pass = []
            
            if storm_name == 'UNNAMED' and max_wnd != np.nan and max_wnd >= 34 and storm_id not in blocked_list:
                if storm_id in increment_but_pass: current_year_id += 1
                pass
            elif storm_id[0:2] == 'CP':
                pass
            else:
                #Skip potential TCs
                if f"{storm_id[0:2]}{num_to_str2(current_year_id)}{storm_year}" in potential_tcs:
                    current_year_id += 1
                self.data[key]['operational_id'] = f"{storm_id[0:2]}{num_to_str2(current_year_id)}{storm_year}"
                current_year_id += 1
                
            #Swap operational storm IDs, if necessary
            swap_list = ['EP101994','EP111994']
            swap_pair = ['EP111994','EP101994']
            if self.data[key]['operational_id'] in swap_list:
                swap_idx = swap_list.index(self.data[key]['operational_id'])
                self.data[key]['operational_id'] = swap_pair[swap_idx]

        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed reading in HURDAT2 data ({tsec} seconds)")
    
    
    def __read_btk(self):
        
        r"""
        Reads in best track data into the Dataset object.
        """

        #Time duration to read in best track
        start_time = dt.now()
        print("--> Starting to read in best track data")

        #Get range of years missing
        start_year = self.data[([k for k in self.data.keys()])[-1]]['year'] + 1
        end_year = (dt.now()).year

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
        for iyear in range(start_year,end_year+1):
            if self.basin == 'north_atlantic':
                search_pattern = f'bal[01234][0123456789]{iyear}.dat'
            elif self.basin == 'east_pacific':
                search_pattern = f'b[ec]p[01234][0123456789]{iyear}.dat'
            elif self.basin == 'both':
                search_pattern = f'b[aec][lp][01234][0123456789]{iyear}.dat'

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
            self.data[stormid] = {'id':stormid,'operational_id':stormid,'name':'','year':int(stormid[4:8]),'season':int(stormid[4:8]),'basin':add_basin,'source_info':'NHC Hurricane Database','source_method':"NHC's Automated Tropical Cyclone Forecasting System (ATCF)",'source_url':"https://ftp.nhc.noaa.gov/atcf/btk/"}
            self.data[stormid]['source'] = self.source

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
                if self.basin == 'north_atlantic':
                    wmo_agency = 'north_atlantic'
                elif self.basin == 'east_pacific':
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

        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed reading in best track data ({tsec} seconds)")

        
    def __read_ibtracs(self):
        
        r"""
        Reads in ibtracs data into the Dataset object.
        """

        #Time duration to read in ibtracs
        start_time = dt.now()
        print("--> Starting to read in ibtracs data")
        
        #Quick error check
        ibtracs_online = True
        fcheck = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
        if fcheck not in self.ibtracs_url:
            if "http" in self.ibtracs_url:
                raise RuntimeError("URL provided is not via NCEI")
            else:
                ibtracs_online = False

        #convert to ibtracs basin
        basin_convert = {'all':'ALL',
                         'east_pacific':'EP',
                         'north_atlantic':'NA',
                         'north_indian':'NI',
                         'south_atlantic':'SA',
                         'south_indian':'SI',
                         'south_pacific':'SP',
                         'west_pacific':'WP'}
        ibtracs_basin = basin_convert.get(self.basin,'')
        
        #read in ibtracs file
        if ibtracs_online == True:
            path = self.ibtracs_url.replace("(basin)",ibtracs_basin)
            f = urllib.request.urlopen(path)
            content = f.read()
            content = content.decode("utf-8")
            content = content.split("\n")
            content = [(i.replace(" ","")).split(",") for i in content]
            f.close()
        else:
            f = open(self.ibtracs_url,"r")
            content = f.readlines()
            content = [(i.replace(" ","")).split(",") for i in content]
            f.close()

        #Initialize empty dict for neumann data
        neumann = {}
        
        #ibtracs ID to jtwc ID mapping
        map_all_id = {}
        map_id = {}
        
        for line in content[2:]:
            
            #LANDFALL	IFLAG	USA_AGENCY	USA_ATCF_ID	USA_LAT	USA_LON	USA_RECORD	USA_STATUS	USA_WIND	USA_PRES


            if len(line) < 150: continue
            #sid, year, adv_number, basin, subbasin, name, time, wmo_type, wmo_lat, wmo_lon, wmo_vmax, wmo_mslp, agency, track_type, dist_land = line[:15]
            
            ibtracs_id, year, adv_number, basin, subbasin, name, time, wmo_type, wmo_lat, wmo_lon, wmo_vmax, wmo_mslp, agency, track_type, dist_land, dist_landfall, iflag, usa_agency, sid, lat, lon, special, stype, vmax, mslp = line[:25]
            
            date = dt.strptime(time,'%Y-%m-%d%H:%M:00')
            
            #Fix name to be consistent with HURDAT
            if name == 'NOT_NAMED': name = 'UNNAMED'
            if name[-1] == '-': name = name[:-1]

            #Add storm to list of keys
            if self.ibtracs_mode == 'wmo' and ibtracs_id not in self.data.keys():

                #add empty entry into dict
                self.data[ibtracs_id] = {'id':sid,'operational_id':'','name':name,'year':date.year,'season':int(year),'basin':self.basin}
                self.data[ibtracs_id]['source'] = self.source
                self.data[ibtracs_id]['source_info'] = 'World Meteorological Organization (official)'
                self.data[ibtracs_id]['notes'] = "'vmax' = wind converted to 1-minute using the 0.88 conversion factor. 'vmax_orig' = original vmax as assessed by its respective WMO agency."

                #add empty lists
                for val in ['date','extra_obs','special','type','lat','lon','vmax','vmax_orig','mslp','wmo_basin']:
                    self.data[ibtracs_id][val] = []
                self.data[ibtracs_id]['ace'] = 0.0
                
            elif sid != '' and ibtracs_id not in map_all_id.keys():
                
                #ID entry method to use
                use_id = sid
                
                #Add id to list
                map_all_id[ibtracs_id] = sid

                #add empty entry into dict
                self.data[use_id] = {'id':sid,'operational_id':'','name':name,'year':date.year,'season':int(year),'basin':self.basin}
                self.data[use_id]['source'] = self.source
                self.data[use_id]['source_info'] = 'Joint Typhoon Warning Center (unofficial)'
                if self.neumann == True: self.data[use_id]['source_info'] += '& Charles Neumann reanalysis for South Hemisphere storms'
                current_id = use_id

                #add empty lists
                for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp',
                            'wmo_type','wmo_lat','wmo_lon','wmo_vmax','wmo_mslp','wmo_basin']:
                    self.data[use_id][val] = []
                self.data[use_id]['ace'] = 0.0

            #Get neumann data for storms containing it
            if self.neumann == True:
                neumann_lat, neumann_lon, neumann_type, neumann_vmax, neumann_mslp = line[141:146]
                if neumann_lat != "" and neumann_lon != "":
                    
                    #Add storm to list of keys
                    if ibtracs_id not in neumann.keys():
                        neumann[ibtracs_id] = {'id':sid,'operational_id':'','name':name,'year':date.year,'season':int(year),'basin':self.basin}
                        neumann[ibtracs_id]['source'] = self.source
                        neumann[ibtracs_id]['source_info'] = 'Joint Typhoon Warning Center (unofficial) & Charles Neumann reanalysis for South Hemisphere storms'
                        for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
                            neumann[ibtracs_id][val] = []
                        neumann[ibtracs_id]['ace'] = 0.0
                    
                    #Retrieve data
                    neumann_date = dt.strptime(time,'%Y-%m-%d%H:%M:00')
                    neumann_lat = float(wmo_lat)
                    neumann_lon = float(wmo_lon)
                    neumann_vmax = np.nan if neumann_vmax == "" else int(neumann_vmax)
                    neumann_mslp = np.nan if neumann_mslp == "" else int(neumann_mslp)
                    if neumann_type == 'TC':
                        if neumann_vmax < 34:
                            neumann_type = 'TD'
                        elif neumann_vmax < 64:
                            neumann_type = 'TS'
                        else:
                            neumann_type = 'HU'
                    elif neumann_type == 'MM' or neumann_type == '':
                        neumann_type = 'LO'
                    
                    neumann[ibtracs_id]['date'].append(neumann_date)
                    neumann[ibtracs_id]['special'].append(special)

                    neumann[ibtracs_id]['type'].append(neumann_type)
                    neumann[ibtracs_id]['lat'].append(neumann_lat)
                    neumann[ibtracs_id]['lon'].append(neumann_lon)
                    neumann[ibtracs_id]['vmax'].append(neumann_vmax)
                    neumann[ibtracs_id]['mslp'].append(neumann_mslp)
                    
                    hhmm = neumann_date.strftime('%H%M')
                    if hhmm in ['0000','0600','1200','1800']:
                        neumann[ibtracs_id]['extra_obs'].append(0)
                    else:
                        neumann[ibtracs_id]['extra_obs'].append(1)
                    
                    #Edit basin
                    basin_reverse = {v: k for k, v in basin_convert.items()}
                    wmo_basin = basin_reverse.get(basin,'')
                    if subbasin in ['WA','EA']:
                        wmo_basin = 'australia'
                    neumann[ibtracs_id]['wmo_basin'].append(wmo_basin)
                    
                    #Calculate ACE & append to storm total
                    if np.isnan(neumann_vmax) == False:
                        ace = (10**-4) * (neumann_vmax**2)
                        if hhmm in ['0000','0600','1200','1800'] and neumann_type in ['SS','TS','HU']:
                            neumann[ibtracs_id]['ace'] += np.round(ace,4)
                        
            #Skip missing entries
            if self.ibtracs_mode == 'wmo':
                if wmo_lat == "" or wmo_lon == "":
                    continue
                if agency == "": continue
            else:
                if lat == "" or lon == "":
                    continue
                if usa_agency == "" and track_type != "PROVISIONAL": continue
            
            
            #map JTWC to ibtracs ID (for neumann replacement)
            if self.neumann == True:
                if ibtracs_id not in map_id.keys():
                    map_id[ibtracs_id] = sid
            
            #Handle WMO mode
            if self.ibtracs_mode == 'wmo':
                
                #Retrieve data
                date = dt.strptime(time,'%Y-%m-%d%H:%M:00')
                dist_land = int(dist_land)

                #Properly format WMO variables
                wmo_lat = float(wmo_lat)
                wmo_lon = float(wmo_lon)
                wmo_vmax = np.nan if wmo_vmax == "" else int(wmo_vmax)
                wmo_mslp = np.nan if wmo_mslp == "" else int(wmo_mslp)
                
                #Edit basin
                basin_reverse = {v: k for k, v in basin_convert.items()}
                wmo_basin = basin_reverse.get(basin,'')
                if subbasin in ['WA','EA']:
                    wmo_basin = 'australia'
                self.data[ibtracs_id]['wmo_basin'].append(wmo_basin)
                
                #Account for wind discrepancy
                if wmo_basin not in ['north_atlantic','east_pacific'] and np.isnan(wmo_vmax) == False:
                    jtwc_vmax = int(wmo_vmax / 0.88)
                else:
                    if np.isnan(wmo_vmax) == False:
                        jtwc_vmax = int(wmo_vmax + 0.0)
                    else:
                        jtwc_vmax = np.nan
                
                #Convert storm type from ibtracs to hurdat style
                """
                DS - Disturbance
                TS - Tropical
                ET - Extratropical
                SS - Subtropical
                NR - Not reported
                MX - Mixture (contradicting nature reports from different agencies)
                """
                if wmo_type == "DS":
                    stype = "LO"
                elif wmo_type == "TS":
                    if np.isnan(jtwc_vmax) == True:
                        stype = 'LO'
                    elif jtwc_vmax < 34:
                        stype = 'TD'
                    elif jtwc_vmax < 64:
                        stype = 'TS'
                    else:
                        stype = 'HU'
                elif wmo_type == 'SS':
                    if np.isnan(jtwc_vmax) == True:
                        stype = 'LO'
                    elif jtwc_vmax < 34:
                        stype = 'SD'
                    else:
                        stype = 'SS'
                elif wmo_type in ['ET','MX']:
                    wmo_type = 'EX'
                else:
                    stype = 'LO'

                #Handle missing data
                if wmo_vmax < 0: wmo_vmax = np.nan
                if wmo_mslp < 800: wmo_mslp = np.nan

                self.data[ibtracs_id]['date'].append(date)
                self.data[ibtracs_id]['special'].append(special)

                self.data[ibtracs_id]['type'].append(stype)
                self.data[ibtracs_id]['lat'].append(wmo_lat)
                self.data[ibtracs_id]['lon'].append(wmo_lon)
                self.data[ibtracs_id]['vmax'].append(jtwc_vmax)
                self.data[ibtracs_id]['vmax_orig'].append(wmo_vmax)
                self.data[ibtracs_id]['mslp'].append(wmo_mslp)

                hhmm = date.strftime('%H%M')
                if hhmm in ['0000','0600','1200','1800']:
                    self.data[ibtracs_id]['extra_obs'].append(0)
                else:
                    self.data[ibtracs_id]['extra_obs'].append(1)

                #Calculate ACE & append to storm total
                if np.isnan(jtwc_vmax) == False:
                    ace = (10**-4) * (jtwc_vmax**2)
                    if hhmm in ['0000','0600','1200','1800'] and stype in ['SS','TS','HU']:
                        self.data[ibtracs_id]['ace'] += np.round(ace,4)
                
            #Handle non-WMO mode
            else:
                if sid == '': continue
                sid = map_all_id.get(ibtracs_id)

                #Retrieve data
                date = dt.strptime(time,'%Y-%m-%d%H:%M:00')
                dist_land = int(dist_land)

                #Properly format WMO variables
                wmo_lat = float(wmo_lat)
                wmo_lon = float(wmo_lon)
                wmo_vmax = np.nan if wmo_vmax == "" else int(wmo_vmax)
                wmo_mslp = np.nan if wmo_mslp == "" else int(wmo_mslp)

                #Properly format hurdat-style variables
                lat = float(lat)
                lon = float(lon)
                vmax = np.nan if vmax == "" else int(vmax)
                mslp = np.nan if mslp == "" else int(mslp)

                #Convert storm type from ibtracs to hurdat style
                if stype == "ST" or stype == "TY":
                    stype = "HU"
                elif stype == "":
                    if wmo_type == 'TS':
                        if vmax < 34:
                            stype = 'TD'
                        elif vmax < 64:
                            stype = 'TS'
                        else:
                            stype = 'HU'
                    elif wmo_type == 'SS':
                        if vmax < 34:
                            stype = 'SD'
                        else:
                            stype = 'SS'
                    elif wmo_type in ['ET','MX']:
                        wmo_type = 'EX'
                    elif stype == 'DS':
                        stype = 'LO'
                    else:
                        if np.isnan(vmax) == True:
                            stype = 'LO'
                        elif vmax < 34:
                            stype = 'TD'
                        elif vmax < 64:
                            stype = 'TS'
                        else:
                            stype = 'HU'

                #Handle missing data
                if vmax < 0: vmax = np.nan
                if mslp < 800: mslp = np.nan

                self.data[sid]['date'].append(date)
                self.data[sid]['special'].append(special)

                self.data[sid]['wmo_type'].append(wmo_type)
                self.data[sid]['wmo_lat'].append(wmo_lat)
                self.data[sid]['wmo_lon'].append(wmo_lon)
                self.data[sid]['wmo_vmax'].append(wmo_vmax)
                self.data[sid]['wmo_mslp'].append(wmo_mslp)

                self.data[sid]['type'].append(stype)
                self.data[sid]['lat'].append(lat)
                self.data[sid]['lon'].append(lon)
                self.data[sid]['vmax'].append(vmax)
                self.data[sid]['mslp'].append(mslp)

                #Edit basin
                basin_reverse = {v: k for k, v in basin_convert.items()}
                wmo_basin = basin_reverse.get(basin,'')
                if subbasin in ['WA','EA']:
                    wmo_basin = 'australia'
                self.data[sid]['wmo_basin'].append(wmo_basin)

                hhmm = date.strftime('%H%M')
                if hhmm in ['0000','0600','1200','1800']:
                    self.data[sid]['extra_obs'].append(0)
                else:
                    self.data[sid]['extra_obs'].append(1)

                #Calculate ACE & append to storm total
                if np.isnan(vmax) == False:
                    ace = (10**-4) * (vmax**2)
                    if hhmm in ['0000','0600','1200','1800'] and stype in ['SS','TS','HU']:
                        self.data[sid]['ace'] += np.round(ace,4)
                    
        #Remove empty entries
        all_keys = [k for k in self.data.keys()]
        for key in all_keys:
            if len(self.data[key]['lat']) == 0:
                del(self.data[key])
        
        #Replace neumann entries
        if self.neumann == True:
            
            #iterate through every neumann entry
            for key in neumann.keys():
                
                #get corresponding JTWC ID
                jtwc_id = map_id.get(key,'')
                if jtwc_id == '': continue
                
                #plug dict entry
                old_entry = self.data[jtwc_id]
                self.data[jtwc_id] = neumann[key]
                
                #replace id
                self.data[jtwc_id]['id'] = jtwc_id
                
        #Fix cyclone Catarina, if specified & requested
        all_keys = [k for k in self.data.keys()]
        if '2004086S29318' in all_keys and self.catarina == True:
            self.data['2004086S29318'] = cyclone_catarina()
        elif 'AL502004' in all_keys and self.catarina == True:
            self.data['AL502004'] = cyclone_catarina()
        
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed reading in ibtracs data ({tsec} seconds)")
    
    def __getInterp(self):
        self.data_interp = {}
        for key in self.keys:
            
            #Retrieve storm dict
            istorm = self.data[key].copy()
            
            #Interpolate temporally
            istorm = interp_storm(istorm,timeres=1,dt_window=24,dt_align='middle')
            self.data_interp[key]=istorm

    def get_storm_id(self,storm):
        
        r"""
        Returns the storm ID (e.g., "AL012019") given the storm name and year.
        
        Parameters
        ----------
        storm : tuple
            Tuple containing the storm name and year (e.g., ("Matthew",2016)).
            
        Returns
        -------
        str or list
            If a single storm was found, returns a string containing its ID. Otherwise returns a list of matching IDs.
        """
        
        #Error check
        if isinstance(storm,tuple) == False:
            raise TypeError("storm must be of type tuple.")
        if len(storm) != 2:
            raise ValueError("storm must contain 2 elements, name (str) and year (int)")
        name,year = storm
        
        #Search for corresponding entry in keys
        keys_use = []
        for key in self.keys:
            temp_year = self.data[key]['year']
            if temp_year == year:
                temp_name = self.data[key]['name']
                if temp_name == name.upper():
                    keys_use.append(key)
                
        #return key, or list of keys
        if len(keys_use) == 1: keys_use = keys_use[0]
        if len(keys_use) == 0: raise RuntimeError("Storm not found")
        return keys_use
    
    def get_storm_tuple(self,storm):

        r"""
        Returns the storm tuple (e.g., ("Dorian",2019)) given the storm id.
        
        Parameters
        ----------
        storm : string
            String containing the storm (e.g., "AL052019").
            
        Returns
        -------
        tuple
            Returns a list of matching IDs.
        """

        #Error check
        if isinstance(storm,str) == False:
            raise TypeError("storm must be of type string.")
        try:
            name = self.data[storm]['name']
            year = self.data[storm]['year']
        except:
            raise RuntimeError("Storm not found")
        return (name,year)
    
    def get_storm(self,storm):
        
        r"""
        Retrieves a Storm object for the requested storm.
        
        Parameters
        ----------
        storm : str or tuple
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        
        Returns
        -------
        tropycal.tracks.Storm
            Object containing information about the requested storm, and methods for analyzing and plotting the storm.
        """
        
        #Check if storm is str or tuple
        if isinstance(storm, str) == True:
            key = storm
        elif isinstance(storm, tuple) == True:
            key = self.get_storm_id((storm[0],storm[1]))
        else:
            raise RuntimeError("Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
        
        #Retrieve key of given storm
        if isinstance(key, str) == True:
            
            #Check to see if tornado data exists for this storm
            if np.max(self.keys_tors) == 1:
                if key in self.data_tors.keys():
                    return Storm(self.data[key],{'data':self.data_tors[key],'dist_thresh':self.tornado_dist_thresh})
                else:
                    return Storm(self.data[key])
            else:
                return Storm(self.data[key])
        else:
            error_message = ''.join([f"\n{i}" for i in key])
            error_message = f"Multiple IDs were identified for the requested storm. Choose one of the following storm IDs and provide it as the 'storm' argument instead of a tuple:{error_message}"
            raise RuntimeError(error_message)
    
    
    def plot_storm(self,storm,domain="dynamic",plot_all=False,ax=None,return_ax=False,cartopy_proj=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
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
        
        #Retrieve requested storm
        if isinstance(storm,dict) == False:
            storm_dict = self.get_storm(storm).dict
        else:
            storm_dict = storm
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(storm_dict['lon']) > 150 or min(storm_dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        else:
            self.plot_obj.proj = cartopy_proj
            
        #Plot storm
        plot_ax = self.plot_obj.plot_storm(storm_dict,domain,plot_all,ax=ax,return_ax=return_ax,save_path=save_path,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
    
    
    def plot_storms(self,storms,domain="dynamic",title_text="TC Track Composite",filter_dates=('1/1','12/31'),plot_all_dots=False,ax=None,return_ax=False,cartopy_proj=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of multiple storms.
        
        Parameters
        ----------
        storms : list
            List of requested storms. List can contain either strings of storm ID (e.g., "AL052019"), tuples with storm name and year (e.g., ("Matthew",2016)), or dict entries.
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        plot_all_dots : bool
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
        
        #Identify plot domain for all requested storms
        max_lon = -9999
        min_lon = 9999
        storm_dicts = []
        for storm in storms:
            
            #Retrieve requested storm
            if isinstance(storm,dict) == False:
                storm_dict = self.get_storm(storm).dict
            else:
                storm_dict = storm
            storm_dicts.append(storm_dict)
            
            #Add to array of max/min lat/lons
            if max(storm_dict['lon']) > max_lon: max_lon = max(storm_dict['lon'])
            if min(storm_dict['lon']) < min_lon: min_lon = min(storm_dict['lon'])
            
        #Create cartopy projection
        if cartopy_proj == None:
            if max(storm_dict['lon']) > 150 or min(storm_dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        else:
            self.plot_obj.proj = cartopy_proj
            
        #Plot storm
        plot_ax = self.plot_obj.plot_storms(storm_dicts,domain,title_text,filter_dates,plot_all_dots,ax=ax,return_ax=return_ax,save_path=save_path,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
        
        
    def plot_season(self,year,domain=None,ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single season.
        
        Parameters
        ----------
        year : int
            Year to retrieve season data. If in southern hemisphere, year is the 2nd year of the season (e.g., 1975 for 1974-1975).
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        
        Other Parameters
        ----------------
        prop : dict
            Customization properties of storm track lines. Please refer to :ref:`options-prop` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        """
        
        #Retrieve season object
        season = self.get_season(year)
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if season.basin in ['east_pacific','west_pacific','south_pacific','australia','all']:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        else:
            self.plot_obj.proj = cartopy_proj
            
        #Plot season
        plot_ax = self.plot_obj.plot_season(season,domain,ax=ax,return_ax=return_ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax == True: return plot_ax
    
    
    def search_name(self,name):
        
        r"""
        Searches for hurricane seasons containing a storm of the requested name.
        
        Parameters
        ----------
        name : str
            Name to search through the dataset for.
        
        Returns
        -------
        list
            List containing the hurricane seasons where a storm of the requested name was found.
        """
        
        #get keys for all storms in requested year
        years = [self.data[key]['year'] for key in self.keys if self.data[key]['name'] == name.upper()]
        
        return years
    
    
    def download_tcr(self,storm,save_path=""):
        
        r"""
        Downloads the NHC offical Tropical Cyclone Report (TCR) for the requested storm to the requested directory. Available only for storms with advisories issued by the National Hurricane Center.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        save_path : str
            Path of directory to download the TCR into. Default is current working directory.
        """
        
        #Retrieve requested storm
        if isinstance(storm,dict) == False:
            storm_dict = self.get_storm(storm)
        else:
            storm_dict = self.get_storm(storm.id)
        
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
    
    
    def __retrieve_season(self,year,basin):
        
        #Initialize dict to be populated
        season_dict = {}
        
        #Search for corresponding entry in keys
        basin_list = []
        for key in self.keys:
            temp_year = self.data[key]['season']
            if temp_year == int(year):
                temp_basin = self.data[key]['basin']
                temp_wmo_basin = self.data[key]['wmo_basin']
                if temp_basin == 'all':
                    if basin == 'all':
                        season_dict[key] = self.data[key]
                        basin_list.append('all')
                    elif basin in temp_wmo_basin:
                        season_dict[key] = self.data[key]
                        basin_list.append(self.data[key]['wmo_basin'][0])
                else:
                    season_dict[key] = self.data[key]
                    basin_list.append(self.data[key]['wmo_basin'][0])
                    #basin_list.append(max(set(self.data[key]['wmo_basin']), key=self.data[key]['wmo_basin'].count))
                
        #Error check
        if len(season_dict) == 0:
            raise RuntimeError("No storms were identified for the given year in the given basin.")
                
        #Add attributes
        first_key = [k for k in season_dict.keys()][0]
        season_info = {}
        season_info['year'] = year
        season_info['basin'] = max(set(basin_list), key=basin_list.count)
        season_info['source_basin'] = season_dict[first_key]['basin']
        season_info['source'] = season_dict[first_key]['source']
        season_info['source_info'] = season_dict[first_key]['source_info']
                
        #Return object
        return Season(season_dict,season_info)
                   
    def get_season(self,year,basin='all'):
        
        r"""
        Retrieves a Season object for the requested season or seasons.
        
        Parameters
        ----------
        year : int or list
            Year(s) to retrieve season data. If in southern hemisphere, year is the 2nd year of the season (e.g., 1975 for 1974-1975). Use of multiple years is only permissible for hurdat sources.
        basin : str, optional
            If using a global ibtracs dataset, this specifies which basin to load in. Otherwise this argument is ignored.
        
        Returns
        -------
        tropycal.tracks.Season
            Object containing every storm entry for the given season, and methods for analyzing and plotting the season.
        """
        
        #Error checks
        if isinstance(year,int) == False and isinstance(year,list) == False:
            msg = "'year' must be of type int or list."
            raise TypeError(msg)
        if isinstance(year,list) == True:
            for i in year:
                if isinstance(i,int) == False:
                    msg = "Elements of list 'year' must be of type int."
                    raise TypeError(msg)
        
        #Retrieve season object(s)
        if isinstance(year,int) == True:
            return self.__retrieve_season(year,basin)
        else:
            return_season = self.__retrieve_season(year[0],basin)
            for i_year in year[1:]:
                return_season = return_season + self.__retrieve_season(i_year,basin)
            return return_season
    
    def ace_climo(self,plot_year=None,compare_years=None,climo_year_range=None,date_range=None,rolling_sum=0,return_dict=False,plot=True,return_ax=False,save_path=None):
        
        r"""
        Creates a climatology of accumulated cyclone energy (ACE).
        
        Parameters
        ----------
        plot_year : int
            Year to highlight. If current year, plot will be drawn through today.
        compare_years : int or list
            Seasons to compare against. Can be either a single season (int), or a range or list of seasons (list).
        start_year : int
            Year to begin calculating the climatology over. Default is 1950.
        rolling_sum : int
            Days to calculate a rolling sum over. Default is 0 (annual running sum).
        return_dict : bool
            Determines whether to return data from this function. Default is False.
        plot : bool
            Determines whether to generate a plot or not. If False, function simply returns ace dictionary.
        save_path : str
            Determines the file path to save the image to. If blank or none, image will be directly shown.
        
        Returns
        -------
        None or dict
            If return_dict is True, a dictionary containing data about the ACE climatology is returned.
        """
        
        cur_year = dt.now().year
        
        if climo_year_range is None:
            climo_year_range = (1950,dt.now().year-1)
        #if plot_year!=None and plot_year<start_year:
        #    raise ValueError("One of the years is before the climatology start_year.")            
        #if compare_years!=None and np.any(np.asarray(compare_years)<start_year):
        #    raise ValueError("One of the years is before the climatology start_year.")
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
        #Create empty dict
        ace = {}
        
        #Iterate over every year of HURDAT available
        end_year = self.data[self.keys[-1]]['year']
        years = [yr for yr in range(1851,dt.now().year+1) if (min(climo_year_range)<=yr<=max(climo_year_range)) or yr==plot_year]
        for year in years:
            
            #Get info for this year
            season = self.get_season(year)
            year_info = season.summary()
            
            #Generate list of dates for this year
            year_dates = np.array([dt.strptime(((pd.to_datetime(i)).strftime('%Y%m%d%H')),'%Y%m%d%H') for i in np.arange(dt(year,1,1),dt(year+1,1,1),timedelta(hours=6))])
            
            #Remove 2/29 from dates
            if calendar.isleap(year):
                year_dates = year_dates[year_dates != dt(year,2,29,0)]
                year_dates = year_dates[year_dates != dt(year,2,29,6)]
                year_dates = year_dates[year_dates != dt(year,2,29,12)]
                year_dates = year_dates[year_dates != dt(year,2,29,18)]
            
            #Additional empty arrays
            year_cumace = np.zeros((year_dates.shape))
            year_genesis = []
            
            #Get list of storms for this year
            storm_ids = year_info['id']
            for storm in storm_ids:
                
                #Get HURDAT data for this storm
                storm_data = self.data[storm]
                storm_date_y = np.array([int(i.strftime('%Y')) for i in storm_data['date']])
                storm_date_h = np.array([i.strftime('%H%M') for i in storm_data['date']])
                storm_date_m = [i.strftime('%m%d') for i in storm_data['date']]
                storm_date = np.array(storm_data['date'])
                storm_type = np.array(storm_data['type'])
                storm_vmax = np.array(storm_data['vmax'])
                
                #Subset to remove obs not useful for ace
                idx1 = ((storm_type == 'SS') | (storm_type == 'TS') | (storm_type == 'HU'))
                idx2 = ~np.isnan(storm_vmax)
                idx3 = ((storm_date_h == '0000') | (storm_date_h == '0600') | (storm_date_h == '1200') | (storm_date_h == '1800'))
                idx4 = storm_date_y == year
                storm_date = storm_date[(idx1) & (idx2) & (idx3) & (idx4)]
                storm_type = storm_type[(idx1) & (idx2) & (idx3) & (idx4)]
                storm_vmax = storm_vmax[(idx1) & (idx2) & (idx3) & (idx4)]
                if len(storm_vmax) == 0: continue #Continue if doesn't apply to this storm
                storm_ace = (10**-4) * (storm_vmax**2)
                
                #Account for storms on february 29th by pushing them forward 1 day
                if '0229' in storm_date_m:
                    storm_date_temp = []
                    for idate in storm_date:
                        dt_date = pd.to_datetime(idate)
                        if dt_date.strftime('%m%d') == '0229' or dt_date.strftime('%m') == '03':
                            dt_date += timedelta(hours=24)
                        storm_date_temp.append(dt_date)
                    storm_date = storm_date_temp
                
                #Append ACE to cumulative sum
                idx = np.nonzero(np.in1d(year_dates, storm_date))
                year_cumace[idx] += storm_ace
                year_genesis.append(np.where(year_dates == storm_date[0])[0][0])
                
            #Calculate cumulative sum of year
            if rolling_sum == 0:
                year_cum = np.cumsum(year_cumace)
                year_genesis = np.array(year_genesis)
            
                #Attach to dict
                ace[str(year)] = {}
                ace[str(year)]['date'] = year_dates
                ace[str(year)]['ace'] = year_cum
                ace[str(year)]['genesis_index'] = year_genesis
            else:
                year_cum = np.sum(rolling_window(year_cumace,rolling_sum*4),axis=1)
                year_genesis = np.array(year_genesis) - ((rolling_sum*4)-1)
                
                #Attach to dict
                ace[str(year)] = {}
                ace[str(year)]['date'] = year_dates[((rolling_sum*4)-1):]
                ace[str(year)]['ace'] = year_cum
                ace[str(year)]['genesis_index'] = year_genesis
                 
        #------------------------------------------------------------------------------------------
        
        #Construct non-leap year julian day array
        julian = np.arange(365*4.0) / 4.0
        if rolling_sum != 0:
            julian = julian[((rolling_sum*4)-1):]
          
        #Get julian days for a non-leap year
        months_julian = months_in_julian(2019)
        julian_start = months_julian['start']
        julian_midpoint = months_julian['midpoint']
        julian_name = months_julian['name']
        
        #Construct percentile arrays
        all_ace = np.ones((len(years),len(julian)))*np.nan
        for year in range(min(climo_year_range),max(climo_year_range)+1):
            all_ace[years.index(year)] = ace[str(year)]['ace']
        pmin,p10,p25,p40,p60,p75,p90,pmax = np.nanpercentile(all_ace,[0,10,25,40,60,75,90,100],axis=0)
        
        #Return if not plotting
        if plot == False:
            if return_dict == True:
                return ace
            else:
                return
        
        #------------------------------------------------------------------------------------------
        
        #Create figure
        fig,ax=plt.subplots(figsize=(9,7),dpi=200)
        
        #Set up x-axis
        ax.grid(axis='y',linewidth=0.5,color='k',alpha=0.2,zorder=1,linestyle='--')
        ax.set_xticks(julian_midpoint)
        ax.set_xticklabels(julian_name)
        for i,(istart,iend) in enumerate(zip(julian_start[:-1][::2],julian_start[1:][::2])):
            ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
        
        #Limit plot from May onward
        ax.set_xlim(julian_start[4],julian[-1])

        
        if date_range is None:
            if plot_year == cur_year:
                tmax = dt.now().strftime('%m/%d')
            else:
                tmax = '12/31'
            date_range = ('1/1',tmax)
        date_range = [dt.strptime(t,'%m/%d') for t in date_range]
            
        #Add plot title
        if plot_year == None:
            title_string = f"{self.basin.title().replace('_',' ')} Accumulated Cyclone Energy Climatology"
        else:
            cur_year = (dt.now()).year
            if plot_year == cur_year:
                add_current = f"(through {date_range[-1]:%b %d})"
            else:
                add_current = ""
            title_string = f"{plot_year} {self.basin.title().replace('_',' ')} Accumulated Cyclone Energy {add_current}"
        if rolling_sum != 0:
            title_add = f"\n{rolling_sum}-Day Running Sum"
        else:
            title_add = ""
        ax.set_title(f"{title_string}{title_add}",fontsize=12,fontweight='bold',loc='left')
        
        #Plot requested year
        if plot_year != None:
            
            year_julian = np.copy(julian)
            year_ace = ace[str(plot_year)]['ace']
            year_genesis = ace[str(plot_year)]['genesis_index']
            
            #Check to see if this is current year
            cur_year = (dt.now()).year
            if plot_year == cur_year:
                cur_julian = int(convert_to_julian( (dt.now()).replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
                year_julian = year_julian[:cur_julian+1]
                year_ace = year_ace[:cur_julian+1]
                year_genesis = year_genesis[:cur_julian+1]
        
            END_julian = int(convert_to_julian( date_range[-1].replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
            year_julian = year_julian[:END_julian+1]
            year_ace = year_ace[:END_julian+1]
            year_genesis = year_genesis[:END_julian+1]
            #ax.plot(year_julian[-1],year_ace[-1],'o',color='#FF7CFF',ms=8,mec='#750775',mew=0.8,zorder=8)
            #ax.plot(year_julian,year_ace,'-',color='#750775',linewidth=2.8,zorder=6)
            #ax.plot(year_julian,year_ace,'-',color='#FF7CFF',linewidth=2.0,zorder=6,label=f'{plot_year} ACE ({np.max(year_ace):.1f})')
            #ax.plot(year_julian[year_genesis],year_ace[year_genesis],'D',color='#FF7CFF',ms=5,mec='#750775',mew=0.5,zorder=7,label='TC Genesis')
            
            ax.plot(year_julian[-1],year_ace[-1],'o',color='k',ms=8,mec='w',mew=0.8,zorder=8)
            ax.plot(year_julian,year_ace,'-',color='w',linewidth=2.8,zorder=6)
            ax.plot(year_julian,year_ace,'-',color='k',linewidth=2.0,zorder=6,label=f'{plot_year} ACE ({np.max(year_ace):.1f})')
            ax.plot(year_julian[year_genesis],year_ace[year_genesis],'D',color='k',ms=5,mec='w',mew=0.5,zorder=7,label='TC Genesis')
            
        #Plot comparison years
        if compare_years != None:
            
            if isinstance(compare_years, int) == True: compare_years = [compare_years]
                
            for year in compare_years:
                
                year_julian = np.copy(julian)
                year_ace = ace[str(year)]['ace']
                year_genesis = ace[str(year)]['genesis_index']

                #Check to see if this is current year
                cur_year = (dt.now()).year
                if year == cur_year:
                    cur_julian = int(convert_to_julian( (dt.now()).replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
                    year_julian = year_julian[:cur_julian+1]
                    year_ace = year_ace[:cur_julian+1]
                    year_genesis = year_genesis[:cur_julian+1]
                    ax.plot(year_julian[-1],year_ace[-1],'o',color='#333333',alpha=0.3,ms=6,zorder=5)

                if len(compare_years) <= 5:
                    ax.plot(year_julian,year_ace,'-',color='k',linewidth=1.0,alpha=0.5,zorder=3,label=f'{year} ACE ({np.max(year_ace):.1f})')
                    ax.plot(year_julian[year_genesis],year_ace[year_genesis],'D',color='#333333',ms=3,alpha=0.3,zorder=4)
                    ax.text(year_julian[-2],year_ace[-2]+2,str(year),fontsize=7,fontweight='bold',alpha=0.7,ha='right',va='bottom')
                else:
                    ax.plot(year_julian,year_ace,'-',color='k',linewidth=1.0,alpha=0.15,zorder=3)
            
        
        #Plot all climatological values
        pmin_masked = np.array(pmin)
        pmin_masked = np.ma.masked_where(pmin_masked==0,pmin_masked)
        ax.plot(julian,pmax,'--',color='r',zorder=2,label=f'Max ({np.max(pmax):.1f})')
        ax.plot(julian,pmin_masked,'--',color='b',zorder=2,label=f'Min ({np.max(pmin):.1f})')
        ax.fill_between(julian,p10,p90,color='#60CE56',alpha=0.3,zorder=2,label='Climo 10-90%')
        ax.fill_between(julian,p25,p75,color='#16A147',alpha=0.3,zorder=2,label='Climo 25-75%')
        ax.fill_between(julian,p40,p60,color='#00782A',alpha=0.3,zorder=2,label='Climo 40-60%')

        #Add legend & plot credit
        ax.legend(loc=2)
        endash = u"\u2013"
        
        credit_text = plot_credit()
        add_credit(ax,credit_text)
        ax.text(0.99,0.99,f'Climatology from {climo_year_range[0]}{endash}{climo_year_range[-1]}',fontsize=9,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='top',zorder=10)
        
        #Show/save plot and close
        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path,bbox_inches='tight',facecolor='w')
        plt.close()
        
        if return_dict == True:
            return ace
        else:
            return

    def hurricane_days_climo(self,plot_year=None,compare_years=None,start_year=1950,rolling_sum=0,category=None,return_dict=False,plot=True,save_path=None):
        
        r"""
        Creates a climatology of tropical storm/hurricane/major hurricane days.
        
        Parameters
        ----------
        plot_year : int
            Year to highlight. If current year, plot will be drawn through today.
        compare_years : int or list
            Seasons to compare against. Can be either a single season (int), or a range or list of seasons (list).
        start_year : int
            Year to begin calculating the climatology over. Default is 1950.
        rolling_sum : int
            Days to calculate a rolling sum over. Default is 0 (annual running sum).
        return_dict : bool
            Determines whether to return data from this function. Default is False.
        plot : bool
            Determines whether to generate a plot or not. If False, function simply returns ace dictionary.
        save_path : str
            Determines the file path to save the image to. If blank or none, image will be directly shown.
        
        Returns
        -------
        None or dict
            If return_dict is True, a dictionary containing data about the ACE climatology is returned.
        """
        
        #Create empty dict
        tc_days = {}
        
        #Function for counting TC days above a wind threshold
        def duration_thres(arr,thres):
            arr2 = np.zeros((arr.shape))
            arr2[arr>=thres] = (6.0/24.0)
            return arr2
        
        #Iterate over every year of HURDAT available
        end_year = self.data[self.keys[-1]]['year']
        years = range(start_year,end_year+1)
        for year in years:
            
            #Get info for this year
            season = self.get_season(year)
            year_info = season.summary()
            
            #Generate list of dates for this year
            year_dates = np.array([dt.strptime(((pd.to_datetime(i)).strftime('%Y%m%d%H')),'%Y%m%d%H') for i in np.arange(dt(year,1,1),dt(year+1,1,1),timedelta(hours=6))])
            
            #Remove 2/29 from dates
            if calendar.isleap(year):
                year_dates = year_dates[year_dates != dt(year,2,29,0)]
                year_dates = year_dates[year_dates != dt(year,2,29,6)]
                year_dates = year_dates[year_dates != dt(year,2,29,12)]
                year_dates = year_dates[year_dates != dt(year,2,29,18)]
            
            #Additional empty arrays
            temp_arr = np.zeros((year_dates.shape))
            cumulative = {}
            all_thres = ['ts','c1','c2','c3','c4','c5']
            for thres in all_thres:
                cumulative[thres] = np.copy(temp_arr)
            year_genesis = []
            
            #Get list of storms for this year
            storm_ids = year_info['id']
            for storm in storm_ids:
                
                #Get HURDAT data for this storm
                storm_data = self.data[storm]
                storm_date_y = np.array([int(i.strftime('%Y')) for i in storm_data['date']])
                storm_date_h = np.array([i.strftime('%H%M') for i in storm_data['date']])
                storm_date = np.array(storm_data['date'])
                storm_type = np.array(storm_data['type'])
                storm_vmax = np.array(storm_data['vmax'])
                
                #Subset to remove obs not useful for calculation
                idx1 = ((storm_type == 'SS') | (storm_type == 'TS') | (storm_type == 'HU'))
                idx2 = ~np.isnan(storm_vmax)
                idx3 = ((storm_date_h == '0000') | (storm_date_h == '0600') | (storm_date_h == '1200') | (storm_date_h == '1800'))
                idx4 = storm_date_y == year
                storm_date = storm_date[(idx1) & (idx2) & (idx3) & (idx4)]
                storm_type = storm_type[(idx1) & (idx2) & (idx3) & (idx4)]
                storm_vmax = storm_vmax[(idx1) & (idx2) & (idx3) & (idx4)]
                if len(storm_vmax) == 0: continue #Continue if doesn't apply to this storm
                
                #Append storm days to cumulative sum
                idx = np.nonzero(np.in1d(year_dates, storm_date))
                cumulative['ts'][idx] += duration_thres(storm_vmax,34.0)
                cumulative['c1'][idx] += duration_thres(storm_vmax,64.0)
                cumulative['c2'][idx] += duration_thres(storm_vmax,83.0)
                cumulative['c3'][idx] += duration_thres(storm_vmax,96.0)
                cumulative['c4'][idx] += duration_thres(storm_vmax,113.0)
                cumulative['c5'][idx] += duration_thres(storm_vmax,137.0)
                year_genesis.append(np.where(year_dates == storm_date[0])[0][0])
                
            #Calculate cumulative sum of year
            if rolling_sum == 0:
                year_genesis = np.array(year_genesis)
            
                #Attach to dict
                tc_days[str(year)] = {}
                tc_days[str(year)]['date'] = year_dates
                tc_days[str(year)]['genesis_index'] = year_genesis
                
                #Loop through all thresholds
                for thres in all_thres:
                    tc_days[str(year)][thres] = np.cumsum(cumulative[thres])
            else:
                year_genesis = np.array(year_genesis) - ((rolling_sum*4)-1)
                
                #Attach to dict
                tc_days[str(year)] = {}
                tc_days[str(year)]['date'] = year_dates[((rolling_sum*4)-1):]
                tc_days[str(year)]['genesis_index'] = year_genesis
                
                #Loop through all thresholds
                for thres in all_thres:
                    tc_days[str(year)][thres] = np.sum(rolling_window(cumulative[thres],rolling_sum*4),axis=1)
         
        #------------------------------------------------------------------------------------------
        
        #Construct non-leap year julian day array
        julian = np.arange(365*4.0) / 4.0
        if rolling_sum != 0:
            julian = julian[((rolling_sum*4)-1):]
          
        #Get julian days for a non-leap year
        months_julian = months_in_julian(2019)
        julian_start = months_julian['start']
        julian_midpoint = months_julian['midpoint']
        julian_name = months_julian['name']
        
        #Determine type of plot to make
        category_match = {0:'ts',1:'c1',2:'c2',3:'c3',4:'c4',5:'c5'}
        if category == None:
            cat = 0
        else:
            cat = category_match.get(category,'c1')
        
        #Construct percentile arrays
        if cat == 0:
            p50 = {}
            for thres in all_thres:
                all_tc_days = np.zeros((len(years),len(julian)))
                for year in years:
                    all_tc_days[years.index(year)] = tc_days[str(year)][thres]
                p50[thres] = np.percentile(all_tc_days,50,axis=0)
                p50[thres] = np.average(all_tc_days,axis=0)
        else:
            all_tc_days = np.zeros((len(years),len(julian)))
            for year in years:
                all_tc_days[years.index(year)] = tc_days[str(year)][cat]
            pmin,p10,p25,p40,p60,p75,p90,pmax = np.percentile(all_tc_days,[0,10,25,40,60,75,90,100],axis=0)
        
        #Return if not plotting
        if plot == False:
            if return_dict == True:
                return tc_days
            else:
                return
        
        #------------------------------------------------------------------------------------------
        
        #Create figure
        fig,ax=plt.subplots(figsize=(9,7),dpi=200)
        
        #Set up x-axis
        ax.grid(axis='y',linewidth=0.5,color='k',alpha=0.2,zorder=1,linestyle='--')
        ax.set_xticks(julian_midpoint)
        ax.set_xticklabels(julian_name)
        for i,(istart,iend) in enumerate(zip(julian_start[:-1][::2],julian_start[1:][::2])):
            ax.axvspan(istart,iend,color='#e4e4e4',alpha=0.5,zorder=0)
        
        #Limit plot from May onward
        ax.set_xlim(julian_start[4],julian[-1])
        
        #Format plot title by category
        category_names = {'ts':'Tropical Storm','c1':'Category 1','c2':'Category 2','c3':'Category 3','c4':'Category 4','c5':'Category 5'}
        if cat == 0:
            add_str = "Tropical Cyclone"
        else:
            add_str = category_names.get(cat)
        
        #Add plot title
        if plot_year == None:
            title_string = f"{self.basin.title().replace('_',' ')} Accumulated {add_str} Days"
        else:
            cur_year = (dt.now()).year
            if plot_year == cur_year:
                add_current = f" (through {(dt.now()).strftime('%b %d')})"
            else:
                add_current = ""
            title_string = f"{plot_year} {self.basin.title().replace('_',' ')} Accumulated {add_str} Days{add_current}"
        if rolling_sum != 0:
            title_add = f"\n{rolling_sum}-Day Running Sum"
        else:
            title_add = ""
        ax.set_title(f"{title_string}{title_add}",fontsize=12,fontweight='bold',loc='left')
        
        #Plot requested year
        if plot_year != None:
            
            if cat == 0:
                year_labels = []
                for icat in all_thres[::-1]:
                    year_julian = np.copy(julian)
                    year_tc_days = tc_days[str(plot_year)][icat]

                    #Check to see if this is current year
                    cur_year = (dt.now()).year
                    if plot_year == cur_year:
                        cur_julian = int(convert_to_julian( (dt.now()).replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
                        year_julian = year_julian[:cur_julian+1]
                        year_tc_days = year_tc_days[:cur_julian+1]
                        ax.plot(year_julian[-1],year_tc_days[-1],'o',color=get_colors_sshws(icat),ms=8,mec='k',mew=0.8,zorder=8)

                    year_tc_days_masked = np.array(year_tc_days)
                    year_tc_days_masked = np.ma.masked_where(year_tc_days_masked==0,year_tc_days_masked)
                    ax.plot(year_julian,year_tc_days_masked,'-',color='k',linewidth=2.8,zorder=6)
                    ax.plot(year_julian,year_tc_days_masked,'-',color=get_colors_sshws(icat),linewidth=2.0,zorder=6)
                    year_labels.append(f"{np.max(year_tc_days):.1f}")
                    
            else:
                year_julian = np.copy(julian)
                year_tc_days = tc_days[str(plot_year)][cat]
                year_genesis = tc_days[str(plot_year)]['genesis_index']

                #Check to see if this is current year
                cur_year = (dt.now()).year
                if plot_year == cur_year:
                    cur_julian = int(convert_to_julian( (dt.now()).replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
                    year_julian = year_julian[:cur_julian+1]
                    year_tc_days = year_tc_days[:cur_julian+1]
                    year_genesis = year_genesis[:cur_julian+1]
                    ax.plot(year_julian[-1],year_tc_days[-1],'o',color='#FF7CFF',ms=8,mec='#750775',mew=0.8,zorder=8)

                ax.plot(year_julian,year_tc_days,'-',color='#750775',linewidth=2.8,zorder=6)
                ax.plot(year_julian,year_tc_days,'-',color='#FF7CFF',linewidth=2.0,zorder=6,label=f'{plot_year} ({np.max(year_tc_days):.1f} days)')
                ax.plot(year_julian[year_genesis],year_tc_days[year_genesis],'D',color='#FF7CFF',ms=5,mec='#750775',mew=0.5,zorder=7,label='TC Genesis')
            
        #Plot comparison years
        if compare_years != None and cat != 0:
            
            if isinstance(compare_years, int) == True: compare_years = [compare_years]
                
            for year in compare_years:
                
                year_julian = np.copy(julian)
                year_tc_days = tc_days[str(year)][cat]
                year_genesis = tc_days[str(year)]['genesis_index']

                #Check to see if this is current year
                cur_year = (dt.now()).year
                if year == cur_year:
                    cur_julian = int(convert_to_julian( (dt.now()).replace(year=2019,minute=0,second=0) ))*4 - int(rolling_sum*4)
                    year_julian = year_julian[:cur_julian+1]
                    year_tc_days = year_tc_days[:cur_julian+1]
                    year_genesis = year_genesis[:cur_julian+1]
                    ax.plot(year_julian[-1],year_tc_days[-1],'o',color='#333333',alpha=0.3,ms=6,zorder=5)

                if len(compare_years) <= 5:
                    ax.plot(year_julian,year_tc_days,'-',color='k',linewidth=1.0,alpha=0.5,zorder=3,label=f'{year} ({np.max(year_tc_days):.1f} days)')
                    ax.plot(year_julian[year_genesis],year_tc_days[year_genesis],'D',color='#333333',ms=3,alpha=0.3,zorder=4)
                    ax.text(year_julian[-2],year_tc_days[-2]+2,str(year),fontsize=7,fontweight='bold',alpha=0.7,ha='right',va='bottom')
                else:
                    ax.plot(year_julian,year_tc_days,'-',color='k',linewidth=1.0,alpha=0.15,zorder=3)
            
        
        #Plot all climatological values
        if cat == 0:
            if plot_year == None:
                add_str = ["" for i in all_thres]
            else:
                add_str = [f" | {plot_year}: {i}" for i in year_labels[::-1]]
            xnums = np.zeros((p50['ts'].shape))
            ax.fill_between(julian,p50['c1'],p50['ts'],color=get_colors_sshws(34),alpha=0.3,zorder=2,label=f'TS (Avg: {np.max(p50["ts"]):.1f}{add_str[0]})')
            ax.fill_between(julian,p50['c2'],p50['c1'],color=get_colors_sshws(64),alpha=0.3,zorder=2,label=f'C1 (Avg: {np.max(p50["c1"]):.1f}{add_str[1]})')
            ax.fill_between(julian,p50['c3'],p50['c2'],color=get_colors_sshws(83),alpha=0.3,zorder=2,label=f'C2 (Avg: {np.max(p50["c2"]):.1f}{add_str[2]})')
            ax.fill_between(julian,p50['c4'],p50['c3'],color=get_colors_sshws(96),alpha=0.3,zorder=2,label=f'C3 (Avg: {np.max(p50["c3"]):.1f}{add_str[3]})')
            ax.fill_between(julian,p50['c5'],p50['c4'],color=get_colors_sshws(113),alpha=0.3,zorder=2,label=f'C4 (Avg: {np.max(p50["c4"]):.1f}{add_str[4]})')
            ax.fill_between(julian,xnums,p50['c5'],color=get_colors_sshws(137),alpha=0.3,zorder=2,label=f'C5 (Avg: {np.max(p50["c5"]):.1f}{add_str[5]})')
        else:
            pmin_masked = np.array(pmin)
            pmin_masked = np.ma.masked_where(pmin_masked==0,pmin_masked)
            ax.plot(julian,pmax,'--',color='r',zorder=2,label=f'Max ({np.max(pmax):.1f} days)')
            ax.plot(julian,pmin_masked,'--',color='b',zorder=2,label=f'Min ({np.max(pmin):.1f} days)')
            ax.fill_between(julian,p10,p90,color='#60CE56',alpha=0.3,zorder=2,label='Climo 10-90%')
            ax.fill_between(julian,p25,p75,color='#16A147',alpha=0.3,zorder=2,label='Climo 25-75%')
            ax.fill_between(julian,p40,p60,color='#00782A',alpha=0.3,zorder=2,label='Climo 40-60%')

        #Add legend & plot credit
        ax.legend(loc=2)
        endash = u"\u2013"
        ax.text(0.99,0.01,plot_credit(),fontsize=6,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='bottom',zorder=10)
        ax.text(0.99,0.99,f'Climatology from {start_year}{endash}{end_year}',fontsize=8,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='top',zorder=10)
        
        #Show/save plot and close
        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path,bbox_inches='tight')
        plt.close()
        
        if return_dict == True:
            return tc_days
        else:
            return
    
    def wind_pres_relationship(self,storm=None,year_range=None,return_dict=False,plot=True,save_path=None):
        
        r"""
        Creates a climatology of maximum sustained wind speed vs minimum MSLP relationships.
        
        Parameters
        ----------
        storm : str or tuple
            Storm to plot. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        year_range : list or tuple
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is the start and end of dataset.
        return_dict : bool
            Determines whether to return data from this function. Default is False.
        plot : bool
            Determines whether to generate a plot or not. If False, function simply returns ace dictionary.
        save_path : str
            Determines the file path to save the image to. If blank or none, image will be directly shown.
        
        Returns
        -------
        dict
            If return_dict is True, a dictionary containing data about the wind vs. MSLP relationship climatology is returned.
        """
        
        #Define empty dictionary
        relationship = {}
        
        #Determine year range of dataset
        if year_range == None:
            start_year = self.data[self.keys[0]]['year']
            end_year = self.data[self.keys[-1]]['year']
        elif isinstance(year_range,(list,tuple)) == True:
            if len(year_range) != 2:
                raise ValueError("year_range must be a tuple or list with 2 elements: (start_year, end_year)")
            start_year = int(year_range[0])
            if start_year < self.data[self.keys[0]]['year']: start_year = self.data[self.keys[0]]['year']
            end_year = int(year_range[1])
            if end_year > self.data[self.keys[-1]]['year']: end_year = self.data[self.keys[-1]]['year']
        else:
            raise TypeError("year_range must be of type tuple or list")
        
        #Get velocity & pressure pairs for all storms in dataset
        vp = filter_storms_vp(self,year_min=start_year,year_max=end_year)
        relationship['vp'] = vp

        #Create 2D histogram of v+p relationship
        counts,yedges,xedges = np.histogram2d(*zip(*vp),[np.arange(800,1050,5)-2.5,np.arange(0,250,5)-2.5])
        relationship['counts'] = counts
        relationship['yedges'] = yedges
        relationship['xedges'] = xedges
        
        #Return if plot is not requested
        if plot == False:
            if return_dict == True:
                return relationship
            else:
                return
        
        #Create figure
        fig = plt.figure(figsize=(12,9.5),dpi = 200)

        #Plot climatology
        CS = plt.pcolor(xedges,yedges,counts**0.3,vmin=0,vmax=np.amax(counts)**.3,cmap='gnuplot2_r')
        plt.plot(xedges,[testfit(vp,x,2) for x in xedges],'k--',linewidth=2)
        
        #Plot storm, if specified
        if storm != None:
            
            #Check if storm is str or tuple
            if isinstance(storm, str) == True:
                pass
            elif isinstance(storm, tuple) == True:
                storm = self.get_storm_id((storm[0],storm[1]))
            else:
                raise RuntimeError("Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
                
            #Plot storm
            storm_data = self.data[storm]
            V = np.array(storm_data['vmax'])
            P = np.array(storm_data['mslp'])
            T = np.array(storm_data['type'])

            def get_color(itype):
                if itype in ['SD','SS','TD','TS','HU']:
                    return ['#00EE00','palegreen'] #lime
                else:
                    return ['#00A600','#3BD73B']
                
            def getMarker(itype):
                mtype = '^'
                if itype in ['SD','SS']:
                    mtype = 's'
                elif itype in ['TD','TS','HU']:
                    mtype = 'o'
                return mtype
            
            xt_label = False
            tr_label = False
            for i,(iv,ip,it) in enumerate(zip(V[:-1],P[:-1],T[:-1])):
                check = False
                if it in ['SD','SS','TD','TS','HU'] and tr_label == True: check = True
                if not it in ['SD','SS','TD','TS','HU'] and xt_label == True: check = True
                if check == True:
                    plt.scatter(iv, ip, marker='o',s=80,color=get_color(it)[0],edgecolor='k',zorder=9)
                else:
                    if it in ['SD','SS','TD','TS','HU'] and tr_label == False:
                        tr_label = True
                        label_content = f"{storm_data['name'].title()} {storm_data['year']} (Tropical)"
                    if it not in ['SD','SS','TD','TS','HU'] and xt_label == False:
                        xt_label = True
                        label_content = f"{storm_data['name'].title()} {storm_data['year']} (Non-Tropical)"
                    plt.scatter(iv, ip, marker='o',s=80,color=get_color(it)[0],edgecolor='k',label=label_content,zorder=9)
            
            plt.scatter(V[-1], P[-1], marker='D',s=80,color=get_color(it)[0],edgecolor='k',linewidth=2,zorder=9)
            
            for i,(iv,ip,it,mv,mp,mt) in enumerate(zip(V[1:],P[1:],T[1:],V[:-1],P[:-1],T[:-1])):
                plt.quiver(mv, mp, iv-mv, ip-mp, scale_units='xy', angles='xy',
                           scale=1, width=0.005, color=get_color(it)[1],zorder=8)
            
            #Add legend
            plt.legend(loc='upper right',scatterpoints=1,prop={'weight':'bold','size':14})
            
        
        #Additional plot settings
        plt.xlabel('Maximum sustained winds (kt)',fontsize=14)
        plt.ylabel('Minimum central pressure (hPa)',fontsize=14)
        plt.title(f"TC Pressure vs. Wind \n {self.basin.title().replace('_',' ')} | "+\
                  f"{start_year}-{end_year}",fontsize=18,fontweight='bold')
        plt.xticks(np.arange(20,200,20))
        plt.yticks(np.arange(880,1040,20))
        plt.tick_params(labelsize=14)
        plt.grid()
        plt.axis([0,200,860,1040])
        cbar=fig.colorbar(CS)
        cbar.ax.set_ylabel('Historical Frequency',fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks(np.array([i for i in [0,5,50,200,500,1000,2000] if i<np.amax(counts)])**0.3, update_ticks=True)
        cbar.set_ticklabels([i for i in [0,5,50,200,500,1000,2000] if i<np.amax(counts)], update_ticks=True)

        #add credit
        credit_text = Plot().plot_credit()        
        plt.text(0.99,0.01,credit_text,fontsize=9,color='k',alpha=0.7,backgroundcolor='w',\
                transform=plt.gca().transAxes,ha='right',va='bottom',zorder=10)        
        
        #Show/save plot and close
        if save_path == None:
            plt.show()
        else:
            plt.savefig(save_path,bbox_inches='tight')
        plt.close()
        
        if return_dict == True:
            return relationship
        else:
            return
    
    def rank_storm(self,metric,return_df=True,ascending=False,domain=None,year_range=None,date_range=None,subtropical=True):
        
        r"""
        Ranks storm by a specified metric.
        
        Parameters
        ----------
        metric : str
            Metric to rank storms by. Can be any of the following:
            
            * **ace** = rank storms by ACE
            * **start_lat** = starting latitude of cyclone
            * **start_lon** = starting longitude of cyclone
            * **end_lat** = ending latitude of cyclone
            * **end_lon** = ending longitude of cyclone
            * **start_date** = formation date of cyclone
            * **start_date_indomain** = first time step a cyclone entered the domain
            * **max_wind** = first instance of the maximum sustained wind of cyclone
            * **min_mslp** = first instance of the minimum MSLP of cyclone
            * **wind_ge_XX** = first instance of wind greater than/equal to a certain threshold (knots)
        return_df : bool
            Whether to return a pandas.DataFrame (True) or dict (False). Default is True.
        ascending : bool
            Whether to return rank in ascending order (True) or descending order (False). Default is False.
        domain : str
            String representing either a bounded region 'latW/latE/latS/latN', or a basin name. Default is entire basin.
        year_range : list or tuple
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is start and end years of dataset.
        date_range : list or tuple
            List or tuple representing the start and end dates in 'month/day' format (e.g., (6/1,8/15)). Default is entire year.
        subtropical : bool
            Whether to include subtropical storms in the ranking. Default is True.
        
        Returns
        -------
        pandas.DataFrame
            Returns a pandas DataFrame containing ranked storms. If pandas is not installed, a dict will be returned instead.
        """
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
        #Revise metric if threshold included
        if 'wind_ge' in metric:
            thresh = int(metric.split("_")[2])
            metric = 'wind_ge'
        
        #Error check for metric
        metric = metric.lower()
        metric_bank = {'ace':{'output':['ace'],'subset_type':'domain'},
                       'start_lat':{'output':['lat','lon','type'],'subset_type':'start'},
                       'start_lon':{'output':['lon','lat','type'],'subset_type':'start'},
                       'end_lat':{'output':['lat','lon','type'],'subset_type':'end'},
                       'end_lon':{'output':['lon','lat','type'],'subset_type':'end'},
                       'start_date':{'output':['date','lat','lon','type'],'subset_type':'start'},
                       'start_date_indomain':{'output':['date','lat','lon','type'],'subset_type':'domain'},
                       'max_wind':{'output':['vmax','mslp','lat','lon'],'subset_type':'domain'},
                       'min_mslp':{'output':['mslp','vmax','lat','lon'],'subset_type':'domain'},
                       'wind_ge':{'output':['lat','lon','mslp','vmax','date'],'subset_type':'start'},
                      }
        if metric not in metric_bank.keys():
            raise ValueError("Metric requested for sorting is not available. Please reference the documentation for acceptable entries for 'metric'.")
        
        #Determine year range of dataset
        if year_range == None:
            start_year = self.data[self.keys[0]]['year']
            end_year = self.data[self.keys[-1]]['year']
        elif isinstance(year_range,(list,tuple)) == True:
            if len(year_range) != 2:
                raise ValueError("year_range must be a tuple or list with 2 elements: (start_year, end_year)")
            start_year = int(year_range[0])
            end_year = int(year_range[1])
        else:
            raise TypeError("year_range must be of type tuple or list")
            
        #Initialize empty dict
        analyze_list = metric_bank[metric]['output']
        analyze_list.insert(1,'id'); analyze_list.insert(2,'name'); analyze_list.insert(3,'year');
        analyze_dict = {key:[] for key in analyze_list}
            
        #Iterate over every storm in dataset
        for storm in self.keys:
            
            #Get entry for this storm
            storm_data = self.data[storm]
            
            #Filter by year
            if storm_data['year'] < start_year or storm_data['year'] > end_year: continue
            
            #Filter for purely tropical/subtropical storm locations
            type_array = np.array(storm_data['type'])
            if subtropical == True:
                idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
            else:
                idx = np.where((type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
            
            if len(idx[0]) == 0: continue
            lat_tropical = np.array(storm_data['lat'])[idx]
            lon_tropical = np.array(storm_data['lon'])[idx]
            date_tropical = np.array(storm_data['date'])[idx]
            type_tropical = np.array(storm_data['type'])[idx]
            vmax_tropical = np.array(storm_data['vmax'])[idx]
            mslp_tropical = np.array(storm_data['mslp'])[idx]
            basin_tropical = np.array(storm_data['wmo_basin'])[idx]
            
            #Filter geographically
            if domain != None:
                if isinstance(domain,str) == False:
                    raise TypeError("domain must be of type str.")
                if '/' in domain:
                    bound_w,bound_e,bound_s,bound_n = [float(i) for i in domain.split("/")]
                    idx = np.where((lat_tropical >= bound_s) & (lat_tropical <= bound_n) & (lon_tropical >= bound_w) & (lon_tropical <= bound_e))
                else:
                    idx = np.where(basin_tropical==domain)
                if len(idx[0]) == 0: continue
                
                #Check for subset type
                subset_type = metric_bank[metric]['subset_type']
                if subset_type == 'domain':
                    lat_tropical = lat_tropical[idx]
                    lon_tropical = lon_tropical[idx]
                    date_tropical = date_tropical[idx]
                    type_tropical = type_tropical[idx]
                    vmax_tropical = vmax_tropical[idx]
                    mslp_tropical = mslp_tropical[idx]
                    basin_tropical = basin_tropical[idx]
            
            #Filter by time
            if date_range != None:
                start_time = dt.strptime(f"{storm_data['year']}/{date_range[0]}",'%Y/%m/%d')
                end_time = dt.strptime(f"{storm_data['year']}/{date_range[1]}",'%Y/%m/%d')
                idx = np.array([i for i in range(len(lat_tropical)) if date_tropical[i] >= start_time and date_tropical[i] <= end_time])
                if len(idx) == 0: continue
                
                #Check for subset type
                subset_type = metric_bank[metric]['subset_type']
                if subset_type == 'domain':
                    lat_tropical = lat_tropical[idx]
                    lon_tropical = lon_tropical[idx]
                    date_tropical = date_tropical[idx]
                    type_tropical = type_tropical[idx]
                    vmax_tropical = vmax_tropical[idx]
                    mslp_tropical = mslp_tropical[idx]
                    basin_tropical = basin_tropical[idx]
            
            #Filter by requested metric
            if metric == 'ace':
                
                if storm_data['ace'] == 0: continue
                analyze_dict['ace'].append(np.round(storm_data['ace'],4))
                
            elif metric in ['start_lat','end_lat','start_lon','end_lon']:
                
                use_idx = 0 if 'start' in metric else -1
                analyze_dict['lat'].append(lat_tropical[use_idx])
                analyze_dict['lon'].append(lon_tropical[use_idx])
                analyze_dict['type'].append(type_tropical[use_idx])
                
            elif metric in ['start_date']:
                
                analyze_dict['lat'].append(lat_tropical[0])
                analyze_dict['lon'].append(lon_tropical[0])
                analyze_dict['type'].append(type_tropical[0])
                analyze_dict['date'].append(date_tropical[0].replace(year=2016))
                
            elif metric in ['max_wind','min_mslp']:
                
                #Find max wind or min MSLP
                if metric == 'max_wind' and all_nan(vmax_tropical) == True: continue
                if metric == 'min_mslp' and all_nan(mslp_tropical) == True: continue
                use_idx = np.where(vmax_tropical==np.nanmax(vmax_tropical))[0][0]
                if metric == 'min_mslp': use_idx = np.where(mslp_tropical==np.nanmin(mslp_tropical))[0][0]
                
                analyze_dict['lat'].append(lat_tropical[use_idx])
                analyze_dict['lon'].append(lon_tropical[use_idx])
                analyze_dict['mslp'].append(mslp_tropical[use_idx])
                analyze_dict['vmax'].append(vmax_tropical[use_idx])
            
            elif metric in ['wind_ge']:
                
                #Find max wind or min MSLP
                if metric == 'wind_ge' and all_nan(vmax_tropical) == True: continue
                if metric == 'wind_ge' and np.nanmax(vmax_tropical) < thresh: continue
                use_idx = np.where(vmax_tropical>=thresh)[0][0]
                
                analyze_dict['lat'].append(lat_tropical[use_idx])
                analyze_dict['lon'].append(lon_tropical[use_idx])
                analyze_dict['date'].append(date_tropical[use_idx])
                analyze_dict['mslp'].append(mslp_tropical[use_idx])
                analyze_dict['vmax'].append(vmax_tropical[use_idx])
            
            #Append generic storm attributes
            analyze_dict['id'].append(storm)
            analyze_dict['name'].append(storm_data['name'])
            analyze_dict['year'].append(int(storm_data['year']))
            
        #Error check
        if len(analyze_dict[analyze_list[0]]) == 0:
            raise RuntimeError("No storms were found given the requested criteria.")
        
        #Sort in requested order
        arg_idx = np.argsort(analyze_dict[analyze_list[0]])
        if ascending == False: arg_idx = arg_idx[::-1]
            
        #Sort all variables in requested order
        for key in analyze_dict.keys():
            analyze_dict[key] = (np.array(analyze_dict[key])[arg_idx])
        
        #Enter into new ranked dict
        ranked_dict = {}
        for i in range(len(analyze_dict['id'])):
            ranked_dict[i+1] = {key:analyze_dict[key][i] for key in analyze_list}
            if 'date' in ranked_dict[i+1].keys():
                ranked_dict[i+1]['date'] = ranked_dict[i+1]['date'].replace(year=ranked_dict[i+1]['year'])
            
        #Return ranked dictionary
        try:
            import pandas as pd
            return (pd.DataFrame(ranked_dict).transpose())[analyze_list]
        except:
            return ranked_dict
        
    def storm_ace_vs_season(self,storm,year_range=None):
        
        r"""
        Retrives a list of entire hurricane seasons with lower ACE than the storm provided.
        
        Parameters
        ----------
        storm : str or tuple
            Storm to rank seasons against. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        year_range : list or tuple
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is 1950 through the last year in the dataset.
        
        Returns
        -------
        dict
            Dictionary containing the seasons with less ACE than the requested storm.
        """
        
        #Warning for ibtracs
        if self.source == 'ibtracs':
            warning_str = "This function is not currently configured to optimally work for the ibtracs dataset."
            warnings.warn(warning_str)

        #Determine year range of dataset
        if year_range == None:
            start_year = self.data[self.keys[0]]['year']
            if start_year < 1950: start_year = 1950
            end_year = self.data[self.keys[-1]]['year']
        elif isinstance(year_range,(list,tuple)) == True:
            if len(year_range) != 2:
                raise ValueError("year_range must be a tuple or list with 2 elements: (start_year, end_year)")
            start_year = int(year_range[0])
            if start_year < self.data[self.keys[0]]['year']: start_year = self.data[self.keys[0]]['year']
            end_year = int(year_range[1])
            if end_year > self.data[self.keys[-1]]['year']: end_year = self.data[self.keys[-1]]['year']
        else:
            raise TypeError("year_range must be of type tuple or list")
            
        #Check if storm is str or tuple
        if isinstance(storm, str) == True:
            pass
        elif isinstance(storm, tuple) == True:
            storm = self.get_storm_id((storm[0],storm[1]))
        else:
            raise RuntimeError("Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
            
        #Get ACE for this storm
        storm_data = self.data[storm]
        
        #Retrieve ACE for this event
        storm_name = storm_data['name']
        storm_year = storm_data['year']
        storm_ace = np.round(storm_data['ace'],4)
        
        #Initialize empty dict
        ace_rank = {'year':[],'ace':[]}
            
        #Iterate over every season
        for year in range(start_year,end_year+1):
            season = self.get_season(year)
            year_data = season.summary()
            year_ace = year_data['season_ace']
            
            #Compare year ACE against storm ACE
            if year_ace < storm_ace:
                
                ace_rank['year'].append(year)
                ace_rank['ace'].append(year_ace)
                
        return ace_rank

    def filter_storms(self,year_range=(0,9999),date_range=('1/1','12/31'),thresh={},domain=None,doInterp=False,return_keys=True):
        
        r"""
        Filters all storms by various thresholds.
        
        Parameters
        ----------
        year_range : list or tuple
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is start and end years of dataset.
        date_range : list or tuple
            List or tuple representing the start and end dates as a string in 'month/day' format (e.g., ('6/1','8/15')). Default is ('1/1','12/31') or full year.
        thresh : dict
            Keywords include:
                
            * **sample_min** - minimum number of storms in a grid box for the cmd_request to be applied. For the functions 'percentile' and 'average', 'sample_min' defaults to 5 and will override any value less than 5.
            * **v_min** - minimum wind for a given point to be included in the cmd_request.
            * **p_max** - maximum pressure for a given point to be included in the cmd_request.
            * **dv_min** - minimum change in wind over dt_window for a given point to be included in the cmd_request.
            * **dp_max** - maximum change in pressure over dt_window for a given point to be included in the cmd_request.
            * **dt_window** - time window over which change variables are calculated (hours). Default is 24.
            * **dt_align** - alignment of dt_window for change variables -- 'start','middle','end' -- e.g. 'end' for dt_window=24 associates a TC point with change over the past 24 hours. Default is middle.
            
            Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.
        domain : str
            String or tuple representing a bounded region, 'latW/latE/latS/latN'.
        doInterp : bool
            Whether to interpolate track data to hourly. Default is False.
        return_keys : bool
            If True, returns a list of storm IDs that match the specified criteria. Otherwise returns a pandas.DataFrame object with all matching data points. Default is True.
        
        Returns
        -------
        list or pandas.DataFrame
            Check return_keys for more information.
        """
        
        #Update thresh based on input
        default_thresh={'sample_min':1,'p_max':9999,'v_min':0,'v_max':9999,'dv_min':-9999,'dp_max':9999,'dv_max':9999,'dp_min':-9999,'speed_max':9999,'speed_min':-9999,
                        'dt_window':24,'dt_align':'middle'}
        for key in thresh:
            default_thresh[key] = thresh[key]
        thresh = default_thresh

        #Determine domain over which to filter data
        
        if domain is None:
            domain = (0,360,-90,90)
        else:
            domain = (domain['w'],domain['e'],domain['s'],domain['n'])
        lon_min,lon_max,lat_min,lat_max = domain
        
        #if isinstance(domain,str):
        #    lon_min,lon_max,lat_min,lat_max = [float(i) for i in domain.split("/")]
        #else:
        #    lon_min,lon_max,lat_min,lat_max = domain
        
        #Determine year and date range
        year_min,year_max = year_range
        date_min,date_max = [dt.strptime(i,'%m/%d') for i in date_range]
        date_max += timedelta(days=1,seconds=-1)
        
        #Determine if a date falls within the date range
        def date_range_test(t,t_min,t_max):
            if date_min<date_max:
                test1 = (t>=t_min.replace(year=t.year))
                test2 = (t<=t_max.replace(year=t.year))
                return test1 & test2
            else:
                test1 = (t_min.replace(year=t.year)<=t<dt(t.year+1,1,1))
                test2 = (dt(t.year,1,1)<=t<=t_max.replace(year=t.year))
                return test1 | test2
        
        #Create empty dictionary to store output in
        points = {}
        for name in ['vmax','mslp','type','lat','lon','date','season','stormid','ace']+ \
                    ['dmslp_dt','dvmax_dt','acie','dx_dt','dy_dt','speed']*int(doInterp):
            points[name] = []
        
        #Iterate over every storm in TrackDataset
        for key in self.keys:
            
            #Retrieve storm dict
            istorm = self.data[key].copy()
            
            #Interpolate temporally if requested
            if doInterp:
                try:
                    istorm = self.data_interp[key]
                except:
                    istorm = interp_storm(istorm,timeres=1,dt_window=thresh['dt_window'],dt_align=thresh['dt_align'])
                timeres = 1
            else:
                timeres = 6
            #Iterate over every timestep of the storm
            for i in range(len(istorm['date'])):
                
                #Filter to only tropical cyclones, and filter by dates & coordiates
                if istorm['type'][i] in ['TD','SD','TS','SS','HU','TY'] \
                and lat_min<=istorm['lat'][i]<=lat_max and lon_min<=istorm['lon'][i]%360<=lon_max \
                and year_min<=istorm['date'][i].year<=year_max \
                and date_range_test(istorm['date'][i],date_min,date_max):
                    
                    #Append data points
                    points['vmax'].append(istorm['vmax'][i])
                    points['mslp'].append(istorm['mslp'][i])
                    points['type'].append(istorm['type'][i])
                    points['lat'].append(istorm['lat'][i])
                    points['lon'].append(istorm['lon'][i])
                    points['date'].append(istorm['date'][i])
                    points['season'].append(istorm['season'])
                    points['stormid'].append(key)
                    if istorm['vmax'][i]>34:
                        points['ace'].append(istorm['vmax'][i]**2*1e-4*timeres/6)
                    else:
                        points['ace'].append(0)                        
                        
                    #Append separately for interpolated data
                    if doInterp:
                        points['dvmax_dt'].append(istorm['dvmax_dt'][i])
                        points['acie'].append([0,istorm['dvmax_dt'][i]**2*1e-4*timeres/6][istorm['dvmax_dt'][i]>0])
                        points['dmslp_dt'].append(istorm['dmslp_dt'][i])
                        points['dx_dt'].append(istorm['dx_dt'][i])
                        points['dy_dt'].append(istorm['dy_dt'][i])
                        points['speed'].append(istorm['speed'][i])
        
        #Create a DataFrame from the dictionary
        p = pd.DataFrame.from_dict(points)
        
        #Filter by thresholds
        if thresh['v_min']>0:
            p = p.loc[(p['vmax']>=thresh['v_min'])]
        if thresh['v_max']<9999:
            p = p.loc[(p['vmax']<=thresh['v_max'])]
        if thresh['p_max']<9999:
            p = p.loc[(p['mslp']<=thresh['p_max'])]
        if doInterp:
            if thresh['dv_min']>-9999:
                p = p.loc[(p['dvmax_dt']>=thresh['dv_min'])]
            if thresh['dp_max']<9999:
                p = p.loc[(p['dmslp_dt']<=thresh['dp_max'])]
            if thresh['dv_max']<9999:
                p = p.loc[(p['dvmax_dt']<=thresh['dv_max'])]
            if thresh['dp_min']>-9999:
                p = p.loc[(p['dmslp_dt']>=thresh['dp_min'])]
            if thresh['speed_max']<9999:
                p = p.loc[(p['speed']>=thresh['speed_max'])]
            if thresh['speed_min']>-9999:
                p = p.loc[(p['speed']>=thresh['speed_min'])]
        
        #Determine how to return data
        if return_keys:
            return [g[0] for g in p.groupby("stormid")]
        else:
            return p

    def gridded_stats(self,request,thresh={},year_range=None,year_range_subtract=None,year_average=False,
                      date_range=('1/1','12/31'),binsize=1,domain=None,ax=None,return_ax=False,\
                      return_array=False,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of gridded statistics.
        
        Parameters
        ----------
        request : str
            This string is a descriptor for what you want to plot.
            It will be used to define the variable (e.g. 'wind' --> 'vmax') and the function (e.g. 'maximum' --> np.max()).
            This string is also used as the plot title.
            
            Variable words to use in request:
                
            * **wind** - (kt). Sustained wind.
            * **pressure** - (hPa). Minimum pressure.
            * **wind change** - (kt/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "wind change in 24 hours").
            * **pressure change** - (hPa/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "pressure change in 24 hours").
            * **storm motion** - (km/hour). Can be followed a length of time window. Otherwise defaults to 24 hours.
            
            Units of all wind variables are knots and pressure variables are hPa. These are added into the title.
            
            Function words to use in request:
                
            * **maximum**
            * **minimum**
            * **average** 
            * **percentile** - Percentile must be preceded by an integer [0,100].
            * **number** - Number of storms in grid box satisfying filter thresholds.
            
            Example usage: "maximum wind change in 24 hours", "50th percentile wind", "number of storms"
            
        thresh : dict, optional
            Keywords include:
                
            * **sample_min** - minimum number of storms in a grid box for the request to be applied. For the functions 'percentile' and 'average', 'sample_min' defaults to 5 and will override any value less than 5.
            * **v_min** - minimum wind for a given point to be included in the request.
            * **p_max** - maximum pressure for a given point to be included in the request.
            * **dv_min** - minimum change in wind over dt_window for a given point to be included in the request.
            * **dp_max** - maximum change in pressure over dt_window for a given point to be included in the request.
            * **dt_window** - time window over which change variables are calculated (hours). Default is 24.
            * **dt_align** - alignment of dt_window for change variables -- 'start','middle','end' -- e.g. 'end' for dt_window=24 associates a TC point with change over the past 24 hours. Default is middle.
            
            Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.

        year_range : list or tuple, optional
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is start and end years of dataset.
        year_range_subtract : list or tuple, optional
            A year range to subtract from the previously specified "year_range". If specified, will create a difference plot.
        year_average : bool, optional
            If True, both year ranges will be computed and plotted as an annual average.
        date_range : list or tuple, optional
            List or tuple representing the start and end dates as a string in 'month/day' format (e.g., ('6/1','8/15')). Default is ('1/1','12/31') or full year.
        binsize : float, optional
            Grid resolution in degrees. Default is 1 degree.
        domain : str, optional
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes, optional
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool, optional
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        return_array : bool, optional
            If True, returns the gridded 2D array used to generate the plot. Default is False.
        cartopy_proj : ccrs, optional
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        
        Other Parameters
        ----------------
        prop : dict, optional
            Customization properties of plot. Please refer to :ref:`options-prop-gridded` for available options.
        map_prop : dict, optional
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        """

        default_prop = {'smooth':None}
        for key in prop.keys():
            default_prop[key] = prop[key]
        prop = default_prop
        
        #Update thresh based on input
        default_thresh={'sample_min':np.nan,'p_max':np.nan,'v_min':np.nan,'dv_min':np.nan,'dp_max':np.nan,'dv_max':np.nan,'dp_min':np.nan,'dt_window':24,'dt_align':'middle'}
        for key in thresh:
            default_thresh[key] = thresh[key]
        thresh = default_thresh
        
        #Retrieve the requested function, variable for computing stats, and plot title. These modify thresh if necessary.
        thresh,func = find_func(request,thresh)
        thresh,varname = find_var(request,thresh)
        thresh,plot_subtitle = construct_title(thresh)
        
        #Determine whether request includes a vector (i.e., TC motion vector)
        VEC_FLAG = isinstance(varname,tuple)
        
        #Determine year range of plot
        if year_range == None:
            start_year = self.data[self.keys[0]]['year']
            end_year = self.data[self.keys[-1]]['year']
            year_range = (start_year,end_year)
        
        #Start date in numpy datetime64
        startdate = np.datetime64('2000-'+'-'.join([f'{int(d):02}' for d in date_range[0].split('/')]))
        
        #Determine year range to subtract, if making a difference plot
        if year_range_subtract != None:
            if isinstance(year_range_subtract,(list,tuple)) == False:
                msg = "\"year_range_subtract\" must be of type list or tuple."
                raise TypeError(msg)
            if len(year_range_subtract) != 2:
                msg = "\"year_range_subtract\" must contain 2 elements."
                raise ValueError(msg)
            year_range_subtract = tuple(year_range_subtract)
        
        #---------------------------------------------------------------------------------------------------
        
        #Perform analysis either once or twice depending on year_range_subtract
        if year_range_subtract == None:
            years_analysis = [year_range]
        else:
            years_analysis = [year_range,year_range_subtract]
        grid_x_years = []
        grid_y_years = []
        grid_z_years = []
        
        for year_range_temp in years_analysis:

            #Obtain all data points for the requested threshold and year/date ranges. Interpolate data to hourly.
            print("--> Getting filtered storm tracks")
            points = self.filter_storms(year_range_temp,date_range,thresh=thresh,doInterp=True,return_keys=False)

            #Round lat/lon points down to nearest bin
            to_bin = lambda x: np.floor(x / binsize) * binsize
            points["latbin"] = points.lat.map(to_bin)
            points["lonbin"] = points.lon.map(to_bin)

            #---------------------------------------------------------------------------------------------------

            #Group by latbin,lonbin,stormid
            print("--> Grouping by lat/lon/storm")
            groups = points.groupby(["latbin","lonbin","stormid","season"])

            #Loops through groups, and apply stat func to storms
            #Constructs a new dataframe containing the lat/lon bins, storm ID and the plotting variable
            new_df = {'latbin':[],'lonbin':[],'stormid':[],'season':[],varname:[]}
            for g in groups:
                #Apply function to all time steps in which a storm tracks within a gridbox
                if VEC_FLAG:
                    new_df[varname].append([func(g[1][v].values) for v in varname])
                elif varname == 'date':
                    new_df[varname].append(func([date_diff(dt(2000,t.month,t.day),startdate)\
                          for t in pd.DatetimeIndex(g[1][varname].values)]))
                else:
                    new_df[varname].append(func(g[1][varname].values))                    
                new_df['latbin'].append(g[0][0])
                new_df['lonbin'].append(g[0][1])
                new_df['stormid'].append(g[0][2])
                new_df['season'].append(g[0][3])
            new_df = pd.DataFrame.from_dict(new_df)

            #---------------------------------------------------------------------------------------------------

            #Group again by latbin,lonbin
            #Construct two 1D lists: zi (grid values) and coords, that correspond to the 2D grid
            groups = new_df.groupby(["latbin", "lonbin"])

            #Apply the function to all storms that pass through a gridpoint
            if VEC_FLAG:
                zi = [[func(v) for v in zip(*g[1][varname])] if len(g[1]) >= thresh['sample_min'] else [np.nan]*2 for g in groups]
            elif varname == 'date':
                zi = [func(g[1][varname]) if len(g[1]) >= thresh['sample_min'] else np.nan for g in groups]
                zi = [mdates.date2num(startdate+z) for z in zi]                
            else:
                zi = [func(g[1][varname]) if len(g[1]) >= thresh['sample_min'] else np.nan for g in groups]

            #Construct a 1D array of coordinates
            coords = [g[0] for g in groups]

            #Construct a 2D longitude and latitude grid, using the specified binsize resolution
            if prop['smooth'] is not None:
                all_lats = [(round(l/binsize)*binsize) for key in self.data.keys() for l in self.data[key]['lat']]
                all_lons = [(round(l/binsize)*binsize)%360 for key in self.data.keys() for l in self.data[key]['lon']]
                xi = np.arange(min(all_lons)-binsize,max(all_lons)+2*binsize,binsize)
                yi = np.arange(min(all_lats)-binsize,max(all_lats)+2*binsize,binsize)
                if self.basin == 'all':
                    xi = np.arange(0,360+binsize,binsize)
                    yi = np.arange(-90,90+binsize,binsize)
            else:
                xi = np.arange(np.nanmin(points["lonbin"])-binsize,np.nanmax(points["lonbin"])+2*binsize,binsize)
                yi = np.arange(np.nanmin(points["latbin"])-binsize,np.nanmax(points["latbin"])+2*binsize,binsize)
            grid_x, grid_y = np.meshgrid(xi,yi)
            grid_x_years.append(grid_x)
            grid_y_years.append(grid_y)

            #Construct a 2D grid for the z value, depending on whether vector or scalar quantity
            if VEC_FLAG:
                grid_z_u = np.ones(grid_x.shape) * np.nan
                grid_z_v = np.ones(grid_x.shape) * np.nan
                for c,z in zip(coords,zi):
                    grid_z_u[np.where((grid_y==c[0]) & (grid_x==c[1]))] = z[0]
                    grid_z_v[np.where((grid_y==c[0]) & (grid_x==c[1]))] = z[1]
                grid_z = [grid_z_u,grid_z_v]
            else:
                grid_z = np.ones(grid_x.shape)*np.nan
                for c,z in zip(coords,zi):
                    grid_z[np.where((grid_y==c[0]) & (grid_x==c[1]))] = z

            #Set zero values to nan's if necessary
            if varname == 'type':
                grid_z[np.where(grid_z==0)] = np.nan
            
            #Add to list of grid_z's
            grid_z_years.append(grid_z)
        
        #---------------------------------------------------------------------------------------------------
        
        #Calculate difference between plots, if specified
        if len(grid_z_years) == 2:
            try:
                #Import xarray and construct DataArray
                import xarray as xr
                
                #Determine whether to use averages
                if year_average == True:
                    years_listed = len(range(year_range[0],year_range[1]+1))
                    grid_z_years[0] = grid_z_years[0] / years_listed
                    years_listed = len(range(year_range_subtract[0],year_range_subtract[1]+1))
                    grid_z_years[1] = grid_z_years[1] / years_listed
                   
                #Construct DataArrays
                grid_z_1 = xr.DataArray(np.nan_to_num(grid_z_years[0]),coords=[grid_y_years[0].T[0],grid_x_years[0][0]],dims=['lat','lon'])
                grid_z_2 = xr.DataArray(np.nan_to_num(grid_z_years[1]),coords=[grid_y_years[1].T[0],grid_x_years[1][0]],dims=['lat','lon'])
                
                #Compute difference grid
                grid_z = grid_z_1 - grid_z_2
                
                #Reconstruct lat & lon grids
                xi = grid_z.lon.values
                yi = grid_z.lat.values
                grid_z = grid_z.values
                grid_x, grid_y = np.meshgrid(xi,yi)
                
                #Determine NaNs
                grid_z_years[0][np.isnan(grid_z_years[0])] = -9999
                grid_z_years[1][np.isnan(grid_z_years[1])] = -8999
                grid_z_years[0][grid_z_years[0]!=-9999] = 0
                grid_z_years[1][grid_z_years[1]!=-8999] = 0
                grid_z_1 = xr.DataArray(np.nan_to_num(grid_z_years[0]),coords=[grid_y_years[0].T[0],grid_x_years[0][0]],dims=['lat','lon'])
                grid_z_2 = xr.DataArray(np.nan_to_num(grid_z_years[1]),coords=[grid_y_years[1].T[0],grid_x_years[1][0]],dims=['lat','lon'])
                grid_z_check = (grid_z_1 - grid_z_2).values
                grid_z[grid_z_check==-1000] = np.nan
                
            except ImportError as e:
                raise RuntimeError("Error: xarray is not available. Install xarray in order to use the subtract year functionality.") from e
        else:
            #Determine whether to use averages
            if year_average == True:
                years_listed = len(range(year_range[0],year_range[1]+1))
                grid_z = grid_z / years_listed
        
        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection using basin
        if domain == None:
            domain = self.basin
        if cartopy_proj == None:
            if max(points['lon']) > 150 or min(points['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        
        #Format left title for plot
        endash = u"\u2013"
        dot = u"\u2022"
        title_L = request.lower()
        for name in ['wind','vmax']:
            title_L = title_L.replace(name,'wind (kt)')
        for name in ['pressure','mslp']:
            title_L = title_L.replace(name,'pressure (hPa)')
        for name in ['heading','motion']:
            title_L = title_L.replace(name,f'heading (kt) over {thresh["dt_window"]} hours')
        for name in ['speed','movement']:
            title_L = title_L.replace(name,f'forward speed (kt) over {thresh["dt_window"]} hours')
        if request.find('change') >= 0:
            title_L = title_L+f", {thresh['dt_align']}"
        title_L = title_L[0].upper() + title_L[1:] + plot_subtitle
        
        #Format right title for plot
        date_range = [dt.strptime(d,'%m/%d').strftime('%b/%d') for d in date_range]
        add_avg = ' year-avg' if year_average == True else ''
        if year_range_subtract == None:
            title_R = f'{date_range[0].replace("/"," ")} {endash} {date_range[1].replace("/"," ")} {dot} {year_range[0]} {endash} {year_range[1]}{add_avg}'
        else:
            title_R = f'{date_range[0].replace("/"," ")} {endash} {date_range[1].replace("/"," ")}\n{year_range[0]}{endash}{year_range[1]}{add_avg} minus {year_range_subtract[0]}{endash}{year_range_subtract[1]}{add_avg}'
        prop['title_L'],prop['title_R'] = title_L,title_R
        
        #Change the masking for variables that go out to zero near the edge of the data
        if prop['smooth'] is not None:
            
            #Replace NaNs with zeros to apply Gaussian filter
            grid_z_zeros = grid_z.copy()
            grid_z_zeros[np.isnan(grid_z)] = 0
            initial_mask = grid_z.copy() #Save initial mask
            initial_mask[np.isnan(grid_z)] = -9999
            grid_z_zeros = gfilt(grid_z_zeros,sigma=prop['smooth'])
            
            
            if len(grid_z_years) == 2:
                #grid_z_1_zeros = np.asarray(grid_z_1)
                #grid_z_1_zeros[grid_z_1==-9999]=0
                #grid_z_1_zeros = gfilt(grid_z_1_zeros,sigma=prop['smooth'])
                
                #grid_z_2_zeros = np.asarray(grid_z_2)
                #grid_z_2_zeros[grid_z_2==-8999]=0
                #grid_z_2_zeros = gfilt(grid_z_2_zeros,sigma=prop['smooth'])
                #grid_z_zeros = grid_z_1_zeros - grid_z_2_zeros
                #test_zeros = (grid_z_1_zeros<.02*np.nanmax(grid_z_1_zeros)) & (grid_z_2_zeros<.02*np.nanmax(grid_z_2_zeros))
                pass
            
            elif varname not in [('dx_dt','dy_dt'),'speed','mslp']:
                
                #Apply cutoff at 2% of maximum
                test_zeros = (grid_z_zeros<.02*np.amax(grid_z_zeros))
                grid_z_zeros[test_zeros] = -9999
                initial_mask = grid_z_zeros.copy()
                
            grid_z_zeros[initial_mask==-9999] = np.nan
            grid_z = grid_z_zeros.copy()
        
        #Plot gridded field
        plot_ax = self.plot_obj.plot_gridded(grid_x,grid_y,grid_z,varname,VEC_FLAG,domain,ax=ax,return_ax=True,prop=prop,map_prop=map_prop)
        
        #Format grid into xarray if specified
        if return_array == True:
            try:
                #Import xarray and construct DataArray, replacing NaNs with zeros
                import xarray as xr
                arr = xr.DataArray(np.nan_to_num(grid_z),coords=[grid_y.T[0],grid_x[0]],dims=['lat','lon'])
                return arr
            except ImportError as e:
                raise RuntimeError("Error: xarray is not available. Install xarray in order to use the 'return_array' flag.") from e

        #Return axis
        if return_ax == True and return_array == True:
            return {'ax':plot_ax,'array':arr}
        if return_ax == False and return_array == True:
            return arr
        if ax != None or return_ax == True: return plot_ax
        

    
    def assign_storm_tornadoes(self,dist_thresh=1000,tornado_path='spc'):
        
        r"""
        Assigns tornadoes to all North Atlantic tropical cyclones from TornadoDataset.
        
        Parameters
        ----------
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC. Default is 1000 km.
        tornado_path : str
            Source to read tornado data from. Default is "spc", which reads from the online Storm Prediction Center (SPC) 1950-present tornado database. Can change this to a local file.
        
        Notes
        -----
        If you intend on analyzing tornadoes for multiple tropical cyclones using a Storm object, it is recommended to run this function first to avoid the need to re-read the entire tornado database for each Storm object.
        """
        
        #Check to ensure data source is over North Atlantic
        if self.basin != "north_atlantic":
            raise RuntimeError("Tropical cyclone tornado data is only available for the North Atlantic basin.")
        
        #Check to see if tornado data already exists in this instance
        self.TorDataset = TornadoDataset(tornado_path=tornado_path)
        self.tornado_dist_thresh = dist_thresh
        
        #Iterate through all storms in dataset and assign them tornadoes, if they exist
        timer_start = dt.now()
        print(f'--> Starting to assign tornadoes to storms')
        for i,key in enumerate(self.keys):
            
            #Skip years prior to 1950
            if self.data[key]['year'] < 1950: continue
                
            #Get tornado data for storm
            storm_obj = self.get_storm(key)
            tor_data = self.TorDataset.get_storm_tornadoes(storm_obj,dist_thresh=dist_thresh)
            tor_data = self.TorDataset.rotateToHeading(storm_obj,tor_data)
            self.data_tors[key] = tor_data
            
            #Check if storm contains tornadoes
            if len(tor_data) > 0:
                self.keys_tors[i] = 1
                
        #Update user on status
        print(f'--> Completed assigning tornadoes to storm (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        
    def plot_TCtors_rotated(self,storms,mag_thresh=0,return_ax=False,return_df=False):
        
        r"""
        Plot tracks of tornadoes relative to the storm motion vector of the tropical cyclone.
        
        Parameters
        ----------
        storms : list or str
            Storm(s) for which to plot motion-relative tornado data for. Can be either a list of storm IDs/tuples for which to create a composite of, or a string "all" for all storms containing tornado data.
        mag_thresh : int
            Minimum threshold for tornado rating.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        return_df : bool
            Whether to return the pandas DataFrame containing the composite tornado data. Default is False.
        
        Returns
        -------
        None or dict
            If either "return_ax" or "return_df" are set to True, returns a dict containing their respective data.
        
        Notes
        -----
        The motion vector is oriented upwards (in the +y direction).
        """
        
        #Error check
        try:
            self.TorDataset
        except:
            raise RuntimeError("No tornado data has been attributed to this dataset. Please run \"TrackDataset.assign_storm_tornadoes()\" first.")
        
        #Error check
        if isinstance(mag_thresh,int) == False:
            raise TypeError("mag_thresh must be of type int.")
        elif mag_thresh not in [0,1,2,3,4,5]:
            raise ValueError("mag_thresh must be between 0 and 5.")
        
        #Get IDs of all storms to composite
        if storms == 'all':
            storms = [self.keys[i] for i in range(len(self.keys)) if self.keys_tors[i] == 1]
        else:
            if len(storms)==2 and isinstance(storms[-1],int):
                use_storms = [self.get_storm_id(storms)]
            else:
                use_storms = [i if isinstance(i,str) == True else self.get_storm_id(i) for i in storms]
            storms = [i for i in use_storms if i in self.keys and self.keys_tors[self.keys.index(i)] == 1]
            
        if len(storms) == 0:
            raise RuntimeError("None of the requested storms produced any tornadoes.")
        
        #Get stormTors formatted with requested storm(s)
        stormTors = (self.data_tors[storms[0]]).copy()
        stormTors['storm_id'] = [storms[0]]*len(stormTors)
        if len(storms) > 1:
            for storm in storms[1:]:
                storm_df = self.data_tors[storm]
                storm_df['storm_id'] = [storm]*len(storm_df)
                stormTors = stormTors.append(storm_df)
        
        #Create figure for plotting
        plt.figure(figsize=(9,9),dpi=150)
        ax = plt.subplot()
        
        #Default EF color scale
        EFcolors = get_colors_ef('default')
        
        #Number of storms exceeding mag_thresh
        num_storms = len(np.unique(stormTors.loc[stormTors['mag']>=mag_thresh]['storm_id'].values))
        
        #Filter for mag >= mag_thresh, and sort by mag so highest will be plotted on top
        stormTors = stormTors.loc[stormTors['mag']>=mag_thresh].sort_values('mag')

        #Plot all tornado tracks in motion relative coords
        for _,row in stormTors.iterrows():
            plt.plot([row['rot_xdist_s'],row['rot_xdist_e']+.01],[row['rot_ydist_s'],row['rot_ydist_e']+.01],\
                     lw=2,c=EFcolors[row['mag']])
            
        #Plot dist_thresh radius
        dist_thresh = self.tornado_dist_thresh
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
        ax.set_title(f'Composite motion-relative tornadoes\nMin threshold: EF-{mag_thresh} | n={num_storms} storms',fontsize=14,fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=11.5)
        
        #Add legend
        handles=[]
        for ef,color in enumerate(EFcolors):
            if ef >= mag_thresh:
                count = len(stormTors[stormTors['mag']==ef])
                handles.append(mlines.Line2D([], [], linestyle='-',color=color,label=f'EF-{ef} ({count})'))
        ax.legend(handles=handles,loc='lower left',fontsize=11.5)
        
        #Add attribution
        ax.text(0.99,0.01,plot_credit(),fontsize=8,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='bottom',zorder=10)
        
        #Return axis or show figure
        return_dict = {}
        if return_ax == True:
            return_dict['ax'] = ax
        else:
            plt.show()
            plt.close()

        if return_df == True:
            return_dict['df'] = stormTors
        if len(return_dict) > 0:
            return return_dict
