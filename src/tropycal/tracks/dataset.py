r"""Functionality for storing and analyzing an entire cyclone dataset."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from ..plot import Plot
from .plot import TrackPlot
from .storm import Storm
from .season import Season
from .tools import *

try:
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class Dataset:
    
    r"""
    Creates an instance of a Dataset object containing various cyclone data.

    Parameters
    ----------
    basin : str
        Ocean basin to load data for. Can be any of the following.
        
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
        If True, the best track data from NHC for the current year will be added into the dataset. Valid for "north_atlantic" and "east_pacific" basins. Default is False.
    
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
    
    def __init__(self,basin='north_atlantic',source='hurdat',include_btk=False,**kwargs):
        
        #kwargs
        atlantic_url = kwargs.pop('atlantic_url', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2018-051019.txt')
        pacific_url = kwargs.pop('pacific_url', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2018-071519.txt')
        ibtracs_url = kwargs.pop('ibtracs_url', 'https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.(basin).list.v04r00.csv')
        ibtracs_mode = kwargs.pop('ibtracs_mode', 'jtwc')
        catarina = kwargs.pop('catarina', False)
        ibtracs_hurdat = kwargs.pop('ibtracs_hurdat', False)
        
        #Error check
        if ibtracs_mode not in ['wmo','jtwc','jtwc_neumann']:
            raise ValueError("Error: ibtracs_mode must be either 'wmo', 'jtwc', or 'jtwc_neumann'")
        
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
            raise ExceptionError("Error: Accepted values for 'source' are 'hurdat' or 'ibtracs'")
            
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
        if fcheck not in self.atlantic_url:
            if "http" in self.atlantic_url:
                raise RuntimeError("Error: URL provided is not via NHC")
            else:
                atl_online = False
        if fcheck not in self.pacific_url:
            if "http" in self.pacific_url:
                raise RuntimeError("Error: URL provided is not via NHC")
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
                self.data[line[0]] = {'id':line[0],'operational_id':'','name':line[1],'year':int(line[0][4:]),'season':int(line[0][4:]),'basin':add_basin,}
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
            blocked_list = []#['AL061988']
            potential_tcs = ['AL102017']
            increment_but_pass = []#['AL212005']
            
            if storm_name == 'UNNAMED' and max_wnd != np.nan and max_wnd >= 34 and storm_id not in blocked_list:
                if storm_id in increment_but_pass: current_year_id += 1
                pass
            elif storm_id[0:2] == 'CP':
                pass
            else:
                #Skip potential TCs
                if f"{storm_id[0:2]}{str2(current_year_id)}{storm_year}" in potential_tcs:
                    current_year_id += 1
                self.data[key]['operational_id'] = f"{storm_id[0:2]}{str2(current_year_id)}{storm_year}"
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
            self.data[stormid] = {'id':stormid,'operational_id':stormid,'name':'','year':int(stormid[4:8]),'season':int(stormid[4:8]),'basin':add_basin,}
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
                raise RuntimeError("Error: URL provided is not via NCEI")
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
        if 'AL502004' in all_keys and self.catarina == True:
            self.data['AL502004'] = cyclone_catarina()
        
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed reading in ibtracs data ({tsec} seconds)")
            
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
    
    def get_storm(self,storm,jtwc=False):
        
        r"""
        Retrieves a Storm object for the requested storm.
        
        Parameters
        ----------
        storm : str or tuple
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        
        Returns
        -------
        Storm
            Object containing information about the requested storm, and methods for analyzing and plotting the storm.
        """
        
        #Check if storm is str or tuple
        if isinstance(storm, str) == True:
            key = storm
        elif isinstance(storm, tuple) == True:
            key = self.get_storm_id((storm[0],storm[1]))
        else:
            raise RuntimeError("Error: Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
        
        #Retrieve key of given storm
        if isinstance(key, str) == True:
            return Storm(self.data[key])
        else:
            error_message = ''.join([f"\n{i}" for i in key])
            error_message = f"Error: Multiple IDs were identified for the requested storm. Choose one of the following storm IDs and provide it as the 'storm' argument instead of a tuple:{error_message}"
            raise RuntimeError(error_message)
    
    def plot_storm(self,storm,zoom="dynamic",plot_all=False,ax=None,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        zoom : str
            Zoom for the plot. Default is "dynamic". Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the storm track(s) plotted.
            * **(basin_name)** - Any of the acceptable basins (check "Dataset" for a list).
            * **lonW/lonE/latS/latN** - Custom plot domain.
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
        
        #Retrieve requested storm
        if inistance(storm_dict,dict) == False:
            storm_dict = self.get_storm(storm).dict
        else:
            storm_dict = storm
        
        #Create instance of plot object
        self.plot_obj = TrackPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            if max(storm_dict['lon']) > 150 or min(storm_dict['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            
        #Plot storm
        return_ax = self.plot_obj.plot_storm(storm_dict,zoom,plot_all,ax=ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None: return return_ax
        
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
    
    def get_season(self,year,basin='all'):
        
        r"""
        Retrieves a dictionary with all the storms for a given year.
        
        Parameters
        ----------
        year : int
            Year to retrieve season data. If in southern hemisphere, year is the 2nd year of the season (e.g., 1975 for 1974-1975).
        
        Returns
        -------
        Season
            Object containing every storm entry for the given season, and methods for analyzing and plotting the season.
        """
        
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
            raise RuntimeError("Error: No storms were identified for the given year in the given basin.")
                
        #Add attributes
        first_key = [k for k in season_dict.keys()][0]
        season_info = {}
        season_info['year'] = year
        season_info['basin'] = max(set(basin_list), key=basin_list.count)
        season_info['source_basin'] = season_dict[first_key]['basin']
        season_info['source'] = season_dict[first_key]['source']
                
        #Return object
        return Season(season_dict,season_info)

    def ace_climo(self,plot_year=None,compare_years=None,start_year=1950,rolling_sum=0,return_dict=False,plot=True,save_path=None):
        
        r"""
        Creates a climatology of accumulated cyclone energy (ACE).
        
        Parameters
        ----------
        plot_year : int
            Year to highlight. If current year, plot will be drawn through today.
        compare_years : int or 1D-array
            Seasons to compare against. Can be either a single season (int), or a range or list of seasons (1D-array).
        start_year : int
            Year to begin calculating the climatology over. Default is 1950.
        rolling_sum : int
            Days to calculate a running sum over. Default is 0 (annual running sum).
        return_dict : bool
            Determines whether to return data from this function. Default is False.
        plot : bool
            Determines whether to generate a plot or not. If False, function simply returns ace dictionary.
        save_path : str
            Determines the file path to save the image to. If blank or none, image will be directly shown.
        
        Returns
        -------
        dict
            If return_dict is True, a dictionary containing data about the ACE climatology is returned.
        """
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
        #Create empty dict
        ace = {}
        
        #Iterate over every year of HURDAT available
        end_year = self.data[self.keys[-1]]['year']
        years = range(start_year,end_year+1)
        for year in years:
            
            #Get info for this year
            season = self.get_season(year)
            year_info = season.annual_summary()
            
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
        all_ace = np.zeros((len(years),len(julian)))
        for year in years:
            all_ace[years.index(year)] = ace[str(year)]['ace']
        pmin,p10,p25,p40,p60,p75,p90,pmax = np.percentile(all_ace,[0,10,25,40,60,75,90,100],axis=0)
        
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
        
        #Add plot title
        if plot_year == None:
            title_string = f"{self.basin.title()} Accumulated Cyclone Energy (ACE) Climatology"
        else:
            cur_year = (dt.now()).year
            if plot_year == cur_year:
                add_current = f" (through {(dt.now()).strftime('%b %d')})"
            else:
                add_current = ""
            title_string = f"{plot_year} {self.basin.title()} Accumulated Cyclone Energy (ACE){add_current}"
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
                ax.plot(year_julian[-1],year_ace[-1],'o',color='#FF7CFF',ms=8,mec='#750775',mew=0.8,zorder=8)
            
            ax.plot(year_julian,year_ace,'-',color='#750775',linewidth=2.8,zorder=6)
            ax.plot(year_julian,year_ace,'-',color='#FF7CFF',linewidth=2.0,zorder=6,label=f'{plot_year} ACE ({np.max(year_ace):.1f})')
            ax.plot(year_julian[year_genesis],year_ace[year_genesis],'D',color='#FF7CFF',ms=5,mec='#750775',mew=0.5,zorder=7,label='TC Genesis')
            
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
        ax.text(0.99,0.01,plot_credit(),fontsize=6,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='bottom',zorder=10)
        ax.text(0.99,0.99,f'Climatology from {start_year}{endash}{end_year}',fontsize=8,color='k',alpha=0.7,
                transform=ax.transAxes,ha='right',va='top',zorder=10)
        
        #Show/save plot and close
        if save_path == None:
            plt.show()
        else:
            plt.savefig(savepath,bbox_inches='tight')
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
            Days to calculate a running sum over. Default is 0 (annual running sum).
        return_dict : bool
            Determines whether to return data from this function. Default is False.
        plot : bool
            Determines whether to generate a plot or not. If False, function simply returns ace dictionary.
        save_path : str
            Determines the file path to save the image to. If blank or none, image will be directly shown.
        
        Returns
        -------
        dict
            If return_dict is True, a dictionary containing data about the ACE climatology is returned.
        """
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
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
            year_info = season.annual_summary()
            
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
            title_string = f"{self.basin.title()} Accumulated {add_str} Days"
        else:
            cur_year = (dt.now()).year
            if plot_year == cur_year:
                add_current = f" (through {(dt.now()).strftime('%b %d')})"
            else:
                add_current = ""
            title_string = f"{plot_year} {self.basin.title()} Accumulated {add_str} Days{add_current}"
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
                        ax.plot(year_julian[-1],year_tc_days[-1],'o',color=category_color(icat),ms=8,mec='k',mew=0.8,zorder=8)

                    year_tc_days_masked = np.array(year_tc_days)
                    year_tc_days_masked = np.ma.masked_where(year_tc_days_masked==0,year_tc_days_masked)
                    ax.plot(year_julian,year_tc_days_masked,'-',color='k',linewidth=2.8,zorder=6)
                    ax.plot(year_julian,year_tc_days_masked,'-',color=category_color(icat),linewidth=2.0,zorder=6)
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
            ax.fill_between(julian,p50['c1'],p50['ts'],color=category_color(34),alpha=0.3,zorder=2,label=f'TS (Avg: {np.max(p50["ts"]):.1f}{add_str[0]})')
            ax.fill_between(julian,p50['c2'],p50['c1'],color=category_color(64),alpha=0.3,zorder=2,label=f'C1 (Avg: {np.max(p50["c1"]):.1f}{add_str[1]})')
            ax.fill_between(julian,p50['c3'],p50['c2'],color=category_color(83),alpha=0.3,zorder=2,label=f'C2 (Avg: {np.max(p50["c2"]):.1f}{add_str[2]})')
            ax.fill_between(julian,p50['c4'],p50['c3'],color=category_color(96),alpha=0.3,zorder=2,label=f'C3 (Avg: {np.max(p50["c3"]):.1f}{add_str[3]})')
            ax.fill_between(julian,p50['c5'],p50['c4'],color=category_color(113),alpha=0.3,zorder=2,label=f'C4 (Avg: {np.max(p50["c4"]):.1f}{add_str[4]})')
            ax.fill_between(julian,xnums,p50['c5'],color=category_color(137),alpha=0.3,zorder=2,label=f'C5 (Avg: {np.max(p50["c5"]):.1f}{add_str[5]})')
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
            plt.savefig(savepath,bbox_inches='tight')
        plt.close()
        
        if return_dict == True:
            return tc_days
        else:
            return
    
    def wind_pres_relationship(self,storm=None,start_year=1851,end_year=None,return_dict=False,plot=True,save_path=None):
        
        r"""
        Creates a climatology of maximum sustained wind speed vs. minimum sea level pressure relationships.
        
        Parameters
        ----------
        storm : str or tuple
            Storm to plot. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        start_year : int
            Year to begin calculating the climatology over. Default is 1851.
        end_year : int
            Year to end calculating the climatology over. Default is the latest year available in HURDAT/Best Track.
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
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
        relationship = {}
        
        #Determine end year of hurdat dataset
        if end_year == None: end_year = self.data[self.keys[-1]]['year']
        
        #Get velocity & pressure pairs for all storms in HURDAT
        vp = filter_storms(self.data,self.keys,year_min=start_year,year_max=end_year)
        relationship['vp'] = vp

        #Create 2D histogram of v+p relationship
        counts,yedges,xedges=np.histogram2d(*zip(*vp),[np.arange(800,1050,5)-2.5,np.arange(0,250,5)-2.5])
        relationship['counts'] = counts
        relationship['yedges'] = yedges
        relationship['xedges'] = xedges
        
        if plot == False:
            if return_dict == True:
                return relationship
            else:
                return
        
        #Create figure
        fig=plt.figure(figsize=(12,9.5),dpi=200)
        
        #Plot climatology
        CS=plt.pcolor(xedges,yedges,counts**0.35,vmin=0,vmax=2000**.3,cmap='gnuplot2_r')
        plt.plot(xedges,[testfit(vp,x,2) for x in xedges],'k--',linewidth=2)
        
        #Plot storm, if specified
        if storm != None:
            
            #Check if storm is str or tuple
            if isinstance(storm, str) == True:
                pass
            elif isinstance(storm, tuple) == True:
                storm = self.get_storm_id((storm[0],storm[1]))
            else:
                raise RuntimeError("Error: Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
                
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
        plt.title('TC Pressure vs. Wind \n North Atlantic Basin',fontsize=18,fontweight='bold')
        plt.xticks(np.arange(20,200,20))
        plt.yticks(np.arange(880,1040,20))
        plt.tick_params(labelsize=14)
        plt.grid()
        plt.axis([0,200,860,1040])
        cbar=fig.colorbar(CS)
        cbar.ax.set_ylabel('Historical Frequency',fontsize=14)
        cbar.ax.tick_params(labelsize=14)
        cbar.set_ticks(np.asarray([0,5,50,200,500,2000])**0.3, update_ticks=True)
        cbar.set_ticklabels([0,5,50,200,500,2000], update_ticks=True)
        
        #Show/save plot and close
        if save_path == None:
            plt.show()
        else:
            plt.savefig(savepath,bbox_inches='tight')
        plt.close()
        
        if return_dict == True:
            return relationship
        else:
            return
    
    def rank_storm(self,metric,return_df=True,ascending=False,subset_domain=None,subset_time=None,subtropical=True,
                   start_year=None,end_year=None,return_all=False):
        
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
            * **formation_date** = formation date of cyclone
            * **max_wind** = first instance of the maximum sustained wind of cyclone
            * **min_mslp** = first instance of the minimum MSLP of cyclone
        return_df : bool
            Whether to return a pandas.DataFrame (True) or dict (False). Default is True.
        ascending : bool
            Whether to return rank in ascending order (True) or descending order (False). Default is False.
        subset_domain : str
            String representing either a bounded region 'latW/latE/latS/latN', or a basin name.
        subset_time : list or tuple
            List or tuple representing the start and end dates in 'month/day' format (e.g., [6/1,8/15]).
        subtropical : bool
            Whether to include subtropical storms in the ranking. Default is True.
        start_year : int
            Year to begin calculating the climatology over. Default is the first year available in the dataset.
        end_year : int
            Year to end calculating the climatology over. Default is the latest year available in the dataset.
        
        Returns
        -------
        pandas.DataFrame
            Returns a pandas DataFrame containing ranked storms. If pandas is not installed, a dict will be returned instead.
        """
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
        
        #Error check for metric
        metric = metric.lower()
        metric_bank = {'ace':['ace'],
                       'start_lat':['lat','lon','type'],
                       'start_lon':['lon','lat','type'],
                       'end_lat':['lat','lon','type'],
                       'end_lon':['lon','lat','type'],
                       'formation_date':['date','lat','lon','type'],
                       'max_wind':['vmax','mslp','lat','lon'],
                       'min_mslp':['mslp','vmax','lat','lon'],
                      }
        if metric not in metric_bank.keys():
            raise ValueError("Metric requested for sorting is not available. Please reference the documentation for acceptable entries for 'metric'.")
        
        #Determine year range of dataset
        if start_year == None: start_year = self.data[self.keys[0]]['year']
        if end_year == None: end_year = self.data[self.keys[-1]]['year']
            
        #Initialize empty dict
        analyze_list = metric_bank[metric]
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
            if subset_domain != None:
                if isinstance(subset_domain,str) == False:
                    raise TypeError("subset_domain must be of type str.")
                if '/' in subset_domain:
                    bound_w,bound_e,bound_s,bound_n = [float(i) for i in subset_domain.split("/")]
                    idx = np.where((lat_tropical >= bound_s) & (lat_tropical <= bound_n) & (lon_tropical >= bound_w) & (lon_tropical <= bound_e))
                else:
                    idx = np.where(basin_tropical==subset_domain)
                if len(idx[0]) == 0: continue
                lat_tropical = lat_tropical[idx]
                lon_tropical = lon_tropical[idx]
                date_tropical = date_tropical[idx]
                type_tropical = type_tropical[idx]
                vmax_tropical = vmax_tropical[idx]
                mslp_tropical = mslp_tropical[idx]
                basin_tropical = basin_tropical[idx]
            
            #Filter by time (not working properly yet)
            """
            if subset_time != None:
                start_time = dt.strptime(f"{storm_data['year']}/{subset_time[0]}",'%Y/%m/%d')
                end_time = dt.strptime(f"{storm_data['year']}/{subset_time[1]}",'%Y/%m/%d')
                idx = np.array([i for i in range(len(lat_tropical)) if date_tropical[i] >= start_time and date_tropical[i] <= end_time])
                if len(idx) == 0: continue
                lat_tropical = lat_tropical[idx]
                lon_tropical = lon_tropical[idx]
                date_tropical = date_tropical[idx]
                type_tropical = type_tropical[idx]
                vmax_tropical = vmax_tropical[idx]
                mslp_tropical = mslp_tropical[idx]
                basin_tropical = basin_tropical[idx]
             """
            
            #Filter by requested metric
            if metric == 'ace':
                
                if storm_data['ace'] == 0: continue
                analyze_dict['ace'].append(np.round(storm_data['ace'],4))
                
            elif metric in ['start_lat','end_lat','start_lon','end_lon']:
                
                use_idx = 0 if 'start' in metric else -1
                analyze_dict['lat'].append(lat_tropical[use_idx])
                analyze_dict['lon'].append(lon_tropical[use_idx])
                analyze_dict['type'].append(type_tropical[use_idx])
                
            elif metric in ['formation_date']:
                
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
        
    def storm_ace_vs_season(self,storm,start_year=1950,end_year=None):
        
        r"""
        Retrives a list of entire hurricane seasons with lower ACE than the storm provided.
        
        Parameters
        ----------
        storm : str or tuple
            Storm to plot. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).
        start_year : int
            Year to begin searching from. Default is 1950.
        end_year : int
            Year to end calculating the climatology over. Default is the latest year available in HURDAT/Best Track.
        
        Returns
        -------
        dict
            Dictionary containing the seasons with less ACE than the requested storm.
        """
        
        if self.source == 'ibtracs':
            warnings.warn("This function is not currently configured to work for the ibtracs dataset.")
    
        #Determine end year of hurdat dataset
        if start_year == None: start_year = self.data[self.keys[0]]['year']
        if end_year == None: end_year = self.data[self.keys[-1]]['year']
            
        #Check if storm is str or tuple
        if isinstance(storm, str) == True:
            pass
        elif isinstance(storm, tuple) == True:
            storm = self.get_storm_id((storm[0],storm[1]))
        else:
            raise RuntimeError("Error: Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")
            
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
            year_data = season.annual_summary()
            year_ace = year_data['season_ace']
            
            #Compare year ACE against storm ACE
            if year_ace < storm_ace:
                
                ace_rank['year'].append(year)
                ace_rank['ace'].append(year_ace)
                
        return ace_rank
