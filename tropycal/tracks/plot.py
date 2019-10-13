import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from .tools import *

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class Plot:
    
    def __check_res(self,res):
        
        r"""
        Converts resolution from basemap notation ('l','m','h') to cartopy notation.
        
        Parameters:
        -----------
        res : str
            String representing map resolution ('l','m','h').
        
        Returns:
        --------
        str
            String of the equivalent cartopy map resolution. 
        """
        
        #Check input map resolution and return corresponding map resolution
        compare_dict = {'l':'110m',
                        'm':'50m',
                        'h':'10m'}
        return compare_dict.get(res,'50m')
    
    def create_cartopy(self,proj='PlateCarree',mapobj=None,**kwargs):
        
        r"""
        Initialize a cartopy instance passed projection.
        
        Parameters:
        -----------
        projection
            String representing the cartopy map projection.
        ax
            Axis on which to draw on. Default is None.
        mapobj
            Existing cartopy projection. If passed, will be used instead of generating a new one.
        **kwargs
            Additional arguments that are passed to those associated with projection.
        """
        
        #Initialize an instance of cartopy if not passed
        if mapobj == None:
            self.proj = getattr(ccrs, proj)(**kwargs)
        else:
            self.proj = mapobj
        
    def __create_geography(self,prop):
        
        r"""
        Set up the map geography and colors.
        
        Parameters:
        -----------
        prop : dict
            dict entry containing information about the map geography and colors
        """
        
        #get resolution corresponding to string in prop
        res = self.__check_res(prop['res'])
        
        #fill oceans if specified
        self.ax.set_facecolor(prop['ocean_color'])
        ocean_mask = self.ax.add_feature(cfeature.OCEAN.with_scale(res),facecolor=prop['ocean_color'],edgecolor='face')
        lake_mask = self.ax.add_feature(cfeature.LAKES.with_scale(res),facecolor=prop['ocean_color'],edgecolor='face')
        continent_mask = self.ax.add_feature(cfeature.LAND.with_scale(res),facecolor=prop['land_color'],edgecolor='face')
        
        #draw geography
        states = self.ax.add_feature(cfeature.STATES.with_scale(res),linewidths=prop['linewidth'],linestyle='solid',edgecolor=prop['linecolor'])
        countries = self.ax.add_feature(cfeature.BORDERS.with_scale(res),linewidths=prop['linewidth'],linestyle='solid',edgecolor=prop['linecolor'])
        coastlines = self.ax.add_feature(cfeature.COASTLINE.with_scale(res),linewidths=prop['linewidth'],linestyle='solid',edgecolor=prop['linecolor'])
        
    def dynamic_map_extent(self,min_lon,max_lon,min_lat,max_lat):
        
        r"""
        Sets up a dynamic map extent with an aspect ratio of 3:2 given latitude and longitude bounds.
        
        Parameters:
        -----------
        min_lon : float
            Minimum longitude bound.
        max_lon : float
            Maximum longitude bound.
        min_lat : float
            Minimum latitude bound.
        max_lat : float
            Maximum latitude bound.
        
        Returns:
        --------
        list
            List containing new west, east, north, south map bounds, respectively.
        """

        #Get lat/lon bounds
        bound_w = min_lon+0.0
        bound_e = max_lon+0.0
        bound_s = min_lat+0.0
        bound_n = max_lat+0.0

        #Function for fixing map ratio
        def fix_map_ratio(bound_w,bound_e,bound_n,bound_s,nthres=1.45):
            xrng = abs(bound_w-bound_e)
            yrng = abs(bound_n-bound_s)
            diff = float(xrng) / float(yrng)
            if diff < nthres: #plot too tall, need to make it wider
                goal_diff = nthres * (yrng)
                factor = abs(xrng - goal_diff) / 2.0
                bound_w = bound_w - factor
                bound_e = bound_e + factor
            elif diff > nthres: #plot too wide, need to make it taller
                goal_diff = xrng / nthres
                factor = abs(yrng - goal_diff) / 2.0
                bound_s = bound_s - factor
                bound_n = bound_n + factor
            return bound_w,bound_e,bound_n,bound_s

        #First round of fixing ratio
        bound_w,bound_e,bound_n,bound_s = fix_map_ratio(bound_w,bound_e,bound_n,bound_s,1.45)

        #Adjust map width depending on extent of storm
        xrng = abs(bound_e-bound_w)
        yrng = abs(bound_n-bound_s)
        factor = 0.1
        if min(xrng,yrng) < 15.0:
            factor = 0.2
        if min(xrng,yrng) < 12.0:
            factor = 0.4
        if min(xrng,yrng) < 10.0:
            factor = 0.6
        if min(xrng,yrng) < 8.0:
            factor = 0.75
        if min(xrng,yrng) < 6.0:
            factor = 0.9
        bound_w = bound_w-(xrng*factor)
        bound_e = bound_e+(xrng*factor)
        bound_s = bound_s-(yrng*factor)
        bound_n = bound_n+(yrng*factor)

        #Second round of fixing ratio
        bound_w,bound_e,bound_n,bound_s = fix_map_ratio(bound_w,bound_e,bound_n,bound_s,1.45)
        
        #Return map bounds
        return bound_w,bound_e,bound_s,bound_n
    
    def __plot_lat_lon_lines(self,bounds):
        
        r"""
        Plots parallels and meridians that are constrained by the map bounds.
        
        Parameters:
        -----------
        bounds : list
            List containing map bounds.
        """
        
        #Retrieve bounds from list
        bound_w,bound_e,bound_s,bound_n = bounds
        
        new_xrng = abs(bound_w-bound_e)
        new_yrng = abs(bound_n-bound_s)
        
        #function to round to nearest number
        def rdown(num, divisor):
            return num - (num%divisor)
        def rup(num, divisor):
            return divisor + (num - (num%divisor))
        
        #Calculate parallels and meridians
        rthres = 10
        if new_yrng < 40.0 or new_xrng < 40.0:
            rthres = 5
        if new_yrng < 25.0 or new_xrng < 25.0:
            rthres = 2
        if new_yrng < 9.0 or new_xrng < 9.0:
            rthres = 1
        parallels = np.arange(rdown(bound_s,rthres),rup(bound_n,rthres)+rthres,rthres)
        meridians = np.arange(rdown(bound_w,rthres),rup(bound_e,rthres)+rthres,rthres)
        
        #Fix for dateline crossing
        if self.proj.proj4_params['lon_0'] == 180.0:
            
            #Recalculate parallels and meridians
            parallels = np.arange(rup(bound_s,rthres),rdown(bound_n,rthres)+rthres,rthres)
            meridians = np.arange(rup(bound_w,rthres),rdown(bound_e,rthres)+rthres,rthres)
            meridians2 = np.copy(meridians)
            meridians2[meridians2>180.0] = meridians2[meridians2>180.0]-360.0
            all_meridians = np.arange(0.0,360.0+rthres,rthres)
            all_parallels = np.arange(-90.0,90.0+rthres,rthres)
            
            #First call with no labels but gridlines plotted
            gl1 = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,xlocs=all_meridians,ylocs=all_parallels,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted')
            #Second call with labels but no gridlines
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,xlocs=meridians,ylocs=parallels,linewidth=0.0,color='k',alpha=0.0,linestyle='dotted')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(meridians2)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

        else:
            #Add meridians and parallels
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted')
            gl.xlabels_top = False
            gl.ylabels_right = False
            gl.xlocator = mticker.FixedLocator(meridians)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        #Reset plot bounds
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
        
    def __plot_init(self,ax,map_prop):
        
        r"""
        Initializes the plot by creating a cartopy and axes instance, if one hasn't been created yet, and adds geography.
        
        Parameters:
        -----------
        ax : axes
            Instance of axes
        map_prop : dict
            Dictionary of map properties
        """

        #create cartopy projection, if none existing
        if self.proj == None:
            self.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        
        #create figure
        if ax == None:
            self.fig = plt.figure(figsize=map_prop['figsize'],dpi=map_prop['dpi'])
            self.ax = plt.axes(projection=self.proj)
        else:
            self.ax = ax
        
        #Attach geography to plot, lat/lon lines, etc.
        self.__create_geography(map_prop)
    
    def __add_prop(self,input_prop,default_prop):
        
        r"""
        Overrides default property dictionary elements with those passed as input arguments.
        
        Parameters:
        -----------
        input_prop : dict
            Dictionary to use for overriding default entries.
        default_prop : dict
            Dictionary containing default entries.
        
        Returns:
        --------
        dict
            Default dictionary overriden by entries in input_prop.
        """
        
        #add kwargs to prop and map_prop
        for key in input_prop.keys(): default_prop[key] = input_prop[key]
            
        #Return prop
        return default_prop
    
    def __set_projection(self,zoom):
        
        r"""
        Sets a predefined map projection zoom.
        
        Parameters
        ----------
        zoom : str
            Name of map projection to zoom over.
        """
        
        #North Atlantic plot domain
        if zoom == "north_atlantic":
            bound_w = -105.0
            bound_e = -5.0
            bound_s = 0.0
            bound_n = 65.0
            
        #East Pacific plot domain
        elif zoom == "east_pacific":
            bound_w = -180.0+360.0 
            bound_e = -80+360.0 
            bound_s = 0.0
            bound_n = 65.0
            
        #West Pacific plot domain
        elif zoom == "west_pacific":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = 0.0
            bound_n = 65.0
            
        #North Indian plot domain
        elif zoom == "north_indian":
            bound_w = 30.0
            bound_e = 110.0
            bound_s = -5.0
            bound_n = 40.0
            
        #South Indian plot domain
        elif zoom == "south_indian":
            bound_w = 20.0
            bound_e = 110.0
            bound_s = -50.0
            bound_n = 5.0
            
        #Australia plot domain
        elif zoom == "australia":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = -60.0
            bound_n = 0.0
            
        #South Pacific plot domain
        elif zoom == "south_pacific":
            bound_w = 140.0
            bound_e = -120.0+360.0
            bound_s = -65.0
            bound_n = 0.0
            
        #Global plot domain
        elif zoom == "all":
            bound_w = 0.0
            bound_e = 360.0
            bound_s = -90.0
            bound_n = 90.0
            
        #Set map extent
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        return bound_w, bound_e, bound_s, bound_n
                 
    def plot_storm(self,storm,zoom="dynamic",plot_all=False,ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm track.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
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
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':True,'fillcolor':'category','linecolor':'k','category_colors':'default','linewidth':1.0,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.__add_prop(prop,default_prop)
        map_prop = self.__add_prop(map_prop,default_map_prop)
        self.__plot_init(ax,map_prop)
        
        #set default properties
        input_prop = prop
        input_map_prop = map_prop
        
        #error check
        if isinstance(zoom,str) == False:
            raise TypeError('Error: zoom must be of type str')
        
        #--------------------------------------------------------------------------------------
        
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        #Check for storm type, then get data for storm
        if isinstance(storm, str) == True:
            storm_data = self.data[storm]
        elif isinstance(storm, tuple) == True:
            storm = self.get_storm_id(storm[0],storm[1])
            storm_data = self.data[storm]
        elif isinstance(storm, dict) == True:
            storm_data = storm
        else:
            raise RuntimeError("Error: Storm must be a string (e.g., 'AL052019'), tuple (e.g., ('Matthew',2016)), or dict.")

        #Retrieve storm data
        lats = storm_data['lat']
        lons = storm_data['lon']
        vmax = storm_data['vmax']
        styp = storm_data['type']
        sdate = storm_data['date']
                
        #Account for cases crossing dateline
        if self.proj.proj4_params['lon_0'] == 180.0:
            new_lons = np.array(lons)
            new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
            lons = new_lons.tolist()

        #Add to coordinate extrema
        if max_lat == None:
            max_lat = max(lats)
        else:
            if max(lats) > max_lat: max_lat = max(lats)
        if min_lat == None:
            min_lat = min(lats)
        else:
            if min(lats) < min_lat: min_lat = min(lats)
        if max_lon == None:
            max_lon = max(lons)
        else:
            if max(lons) > max_lon: max_lon = max(lons)
        if min_lon == None:
            min_lon = min(lons)
        else:
            if min(lons) < min_lon: min_lon = min(lons)

        #Plot storm line as specified
        if prop['linecolor'] == 'category':
            type_line = np.array(styp)
            for i in (np.arange(len(lats[1:]))+1):
                ltype = 'solid'
                if type_line[i] not in ['SS','SD','TD','TS','HU']: ltype = 'dotted'
                self.ax.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]],
                              '-',color=category_color(np.nan_to_num(vmax[i])),linewidth=prop['linewidth'],linestyle=ltype,
                              transform=ccrs.PlateCarree(),
                              path_effects=[path_effects.Stroke(linewidth=prop['linewidth']*0.2, foreground='k'), path_effects.Normal()])
        else:
            self.ax.plot(lons,lats,'-',color=prop['linecolor'],linewidth=prop['linewidth'],transform=ccrs.PlateCarree())

        #Plot storm dots as specified
        if prop['dots'] == True:
            #filter dots to only 6 hour intervals
            time_hr = np.array([i.strftime('%H%M') for i in sdate])
            if plot_all == False:
                time_idx = np.where((time_hr == '0000') | (time_hr == '0600') | (time_hr == '1200') | (time_hr == '1800'))
                lat_dots = np.array(lats)[time_idx]
                lon_dots = np.array(lons)[time_idx]
                vmax_dots = np.array(vmax)[time_idx]
                type_dots = np.array(styp)[time_idx]
            else:
                lat_dots = np.array(lats)
                lon_dots = np.array(lons)
                vmax_dots = np.array(vmax)
                type_dots = np.array(styp)
            for i,(ilon,ilat,iwnd,itype) in enumerate(zip(lon_dots,lat_dots,vmax_dots,type_dots)):
                mtype = '^'
                if itype in ['SD','SS']:
                    mtype = 's'
                elif itype in ['TD','TS','HU']:
                    mtype = 'o'
                if prop['fillcolor'] == 'category':
                    ncol = category_color(np.nan_to_num(iwnd))
                else:
                    ncol = 'k'
                self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=0.5,ms=prop['ms'],transform=ccrs.PlateCarree())

        #--------------------------------------------------------------------------------------
        
        
        #Pre-generated zooms
        if zoom in ['north_atlantic','east_pacific','west_pacific','south_pacific','south_indian','north_indian','australia','all']:
            bound_w,bound_e,bound_s,bound_n = self.__set_projection(zoom)
            
        #Storm-centered plot domain
        elif zoom == "dynamic":
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Custom plot domain
        else:
            
            #Check to ensure 3 slashes are provided
            if zoom.count("/") != 3:
                raise ValueError("Error: Custom map projection bounds must be provided as 'west/east/south/north'")
            else:
                try:
                    bound_w,bound_e,bound_s,bound_n = zoom.split("/")
                    bound_w = float(bound_w)
                    bound_e = float(bound_e)
                    bound_s = float(bound_s)
                    bound_n = float(bound_n)
                    self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
                except:
                    raise ValueError("Error: Custom map projection bounds must be provided as 'west/east/south/north'")
        
        #Determine number of lat/lon lines to use for parallels & meridians
        self.__plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        subtrop = classify_subtrop(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(storm_data['vmax']))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_type(np.nanmax(storm_data['vmax']),subtrop,peak_basin)
        self.ax.set_title(f"{storm_type} {storm_data['name']}",loc='left',fontsize=17,fontweight='bold')

        #Add right title
        ace = storm_data['ace']
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))

        #Get storm extrema for display
        mslp_key = 'mslp' if 'wmo_mslp' not in storm_data.keys() else 'wmo_mslp'
        if all_nan(np.array(storm_data[mslp_key])[idx]) == True:
            min_pres = "N/A"
        else:
            min_pres = int(np.nan_to_num(np.nanmin(np.array(storm_data[mslp_key])[idx])))
        if all_nan(np.array(storm_data['vmax'])[idx]) == True:
            max_wind = "N/A"
        else:
            max_wind = int(np.nan_to_num(np.nanmax(np.array(storm_data['vmax'])[idx])))
        start_date = dt.strftime(np.array(storm_data['date'])[idx][0],'%d %b %Y')
        end_date = dt.strftime(np.array(storm_data['date'])[idx][-1],'%d %b %Y')
        endash = u"\u2013"
        dot = u"\u2022"
        self.ax.set_title(f"{start_date} {endash} {end_date}\n{max_wind} kt {dot} {min_pres} hPa {dot} {ace:.1f} ACE",loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':
            
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=category_color(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=category_color(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=category_color(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=category_color(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=category_color(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=category_color(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=category_color(137))
            self.ax.legend(handles=[ex,sb,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()
        
    def plot_storm_nhc(self,forecast,track=None,cone_days=5,zoom="dynamic_forecast",ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of the operational NHC forecast track along with observed track data.
        
        Parameters
        ----------
        forecast : dict
            Dict entry containing forecast data.
        track : dict
            Dict entry containing observed track data. Default is none.
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic_forecast" - default. Dynamically focuses the domain on the forecast track.
            "dynamic" - Dynamically focuses the domain on the combined observed and forecast track.
            "latW/latE/lonS/lonN" - Custom plot domain
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':True,'fillcolor':'category','linecolor':'k','category_colors':'default','linewidth':1.0,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.__add_prop(prop,default_prop)
        map_prop = self.__add_prop(map_prop,default_map_prop)
        self.__plot_init(ax,map_prop)
        
        #set default properties
        input_prop = prop
        input_map_prop = map_prop
        
        #--------------------------------------------------------------------------------------
        
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None
        
        #Add storm or multiple storms
        if track != "":
            
            #Check for storm type, then get data for storm
            if isinstance(track, dict) == True:
                storm_data = track
            else:
                raise RuntimeError("Error: track must be of type dict.")
                
            #Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['date']
            
            #Check if there's enough data points to plot
            matching_times = [i for i in sdate if i <= forecast['init']]
            check_length = len(matching_times)
            if check_length >= 2:

                #Subset until time of forecast
                matching_times = [i for i in sdate if i <= forecast['init']]
                plot_idx = sdate.index(matching_times[-1])+1
                lats = storm_data['lat'][:plot_idx]
                lons = storm_data['lon'][:plot_idx]
                vmax = storm_data['vmax'][:plot_idx]
                styp = storm_data['type'][:plot_idx]
                sdate = storm_data['date'][:plot_idx]

                #Account for cases crossing dateline
                if self.proj.proj4_params['lon_0'] == 180.0:
                    new_lons = np.array(lons)
                    new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                    lons = new_lons.tolist()
                
                #Connect to 1st forecast location
                fcst_hr = np.array(forecast['fhr'])
                start_slice = 0
                if 3 in fcst_hr: start_slice = 3
                iter_hr = np.array(forecast['fhr'])[fcst_hr>=start_slice][0]
                fcst_lon = np.array(forecast['lon'])[fcst_hr>=start_slice][0]
                fcst_lat = np.array(forecast['lat'])[fcst_hr>=start_slice][0]
                fcst_type = np.array(forecast['type'])[fcst_hr>=start_slice][0]
                fcst_vmax = np.array(forecast['vmax'])[fcst_hr>=start_slice][0]
                if fcst_type == "": fcst_type = get_type(fcst_vmax,False)
                if self.proj.proj4_params['lon_0'] == 180.0:
                    if fcst_lon < 0: fcst_lon = fcst_lon + 360.0
                lons.append(fcst_lon)
                lats.append(fcst_lat)
                vmax.append(fcst_vmax)
                styp.append(fcst_type)
                sdate.append(sdate[-1]+timedelta(hours=start_slice))

                #Add to coordinate extrema
                if zoom != "dynamic_forecast":
                    if max_lat == None:
                        max_lat = max(lats)
                    else:
                        if max(lats) > max_lat: max_lat = max(lats)
                    if min_lat == None:
                        min_lat = min(lats)
                    else:
                        if min(lats) < min_lat: min_lat = min(lats)
                    if max_lon == None:
                        max_lon = max(lons)
                    else:
                        if max(lons) > max_lon: max_lon = max(lons)
                    if min_lon == None:
                        min_lon = min(lons)
                    else:
                        if min(lons) < min_lon: min_lon = min(lons)
                else:
                    max_lat = lats[-1]+0.2
                    min_lat = lats[-2]-0.2
                    max_lon = lons[-1]+0.2
                    min_lon = lons[-2]-0.2

                #Plot storm line as specified
                if prop['linecolor'] == 'category':
                    type6 = np.array(styp)
                    for i in (np.arange(len(lats[1:]))+1):
                        ltype = 'solid'
                        if type6[i] not in ['SS','SD','TD','TS','HU']: ltype = 'dotted'
                        self.ax.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]],
                                      '-',color=category_color(np.nan_to_num(vmax[i])),linewidth=prop['linewidth'],linestyle=ltype,
                                      transform=ccrs.PlateCarree(),
                                      path_effects=[path_effects.Stroke(linewidth=prop['linewidth']*0.2, foreground='k'), path_effects.Normal()])
                else:
                    self.ax.plot(lons,lats,'-',color=prop['linecolor'],linewidth=prop['linewidth'],transform=ccrs.PlateCarree())

                #Plot storm dots as specified
                if prop['dots'] == True:
                    #filter dots to only 6 hour intervals
                    time_hr = np.array([i.strftime('%H%M') for i in sdate])
                    #time_idx = np.where((time_hr == '0300') | (time_hr == '0900') | (time_hr == '1500') | (time_hr == '2100'))
                    lat6 = np.array(lats)#[time_idx]
                    lon6 = np.array(lons)#[time_idx]
                    vmax6 = np.array(vmax)#[time_idx]
                    type6 = np.array(styp)#[time_idx]
                    for i,(ilon,ilat,iwnd,itype) in enumerate(zip(lon6,lat6,vmax6,type6)):
                        mtype = '^'
                        if itype in ['SD','SS']:
                            mtype = 's'
                        elif itype in ['TD','TS','HU']:
                            mtype = 'o'
                        if prop['fillcolor'] == 'category':
                            ncol = category_color(np.nan_to_num(iwnd))
                        else:
                            ncol = 'k'
                        self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=0.5,ms=prop['ms'],transform=ccrs.PlateCarree())

        #--------------------------------------------------------------------------------------

        #Error check cone days
        if isinstance(cone_days,int) == False:
            raise TypeError("Error: cone_days must be of type int")
        if cone_days not in [5,4,3,2]:
            raise ValueError("Error: cone_days must be an int between 2 and 5.")
        
        #Error check forecast dict
        if isinstance(forecast, dict) == False:
            raise RuntimeError("Error: Forecast must be of type dict")
            
        #Determine first forecast index
        fcst_hr = np.array(forecast['fhr'])
        start_slice = 0
        if 3 in fcst_hr: start_slice = 3
        check_duration = fcst_hr[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]

        #Check for sufficiently many hours
        if len(check_duration) > 1:

            #Generate forecast cone for forecast data
            dateline = False
            if self.proj.proj4_params['lon_0'] == 180.0: dateline = True
            cone = self.__generate_nhc_cone(forecast,dateline,cone_days)

            #Contour fill cone & account for dateline crossing
            cone_lon = cone['lon']
            cone_lat = cone['lat']
            cone_lon_2d = cone['lon2d']
            cone_lat_2d = cone['lat2d']
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(cone_lon_2d)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                cone_lon_2d = new_lons.tolist()
                new_lons = np.array(cone_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                cone_lon = new_lons.tolist()
            self.ax.contourf(cone_lon_2d,cone_lat_2d,cone['cone'],[0.9,1.1],colors=['#ffffff','#ffffff'],alpha=0.6,zorder=2,transform=ccrs.PlateCarree())
            self.ax.contour(cone_lon_2d,cone_lat_2d,cone['cone'],[0.9],linewidths=1.0,colors=['k'],zorder=3,transform=ccrs.PlateCarree())

            #Plot center line & account for dateline crossing
            center_lon = cone['center_lon']
            center_lat = cone['center_lat']
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(center_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                center_lon = new_lons.tolist()
            self.ax.plot(center_lon,center_lat,color='k',linewidth=2.0,zorder=4,transform=ccrs.PlateCarree())

            #Plot forecast dots
            iter_hr = np.array(forecast['fhr'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_lon = np.array(forecast['lon'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_lat = np.array(forecast['lat'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_type = np.array(forecast['type'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            fcst_vmax = np.array(forecast['vmax'])[(fcst_hr>=start_slice) & (fcst_hr<=cone_days*24)]
            
            #Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(fcst_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                fcst_lon = new_lons.tolist()

            for i,(ilon,ilat,itype,iwnd,ihr) in enumerate(zip(fcst_lon,fcst_lat,fcst_type,fcst_vmax,iter_hr)):
                mtype = '^'
                if itype in ['SD','SS']:
                    mtype = 's'
                elif itype in ['TD','TS','HU','']:
                    mtype = 'o'
                if prop['fillcolor'] == 'category':
                    ncol = category_color(np.nan_to_num(iwnd))
                else:
                    ncol = 'k'
                #Marker width
                mew = 0.5; use_zorder=5
                if i == 0:
                    mew = 2.0; use_zorder=10
                self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=mew,ms=prop['ms']*1.3,transform=ccrs.PlateCarree(),zorder=use_zorder)

            #Add to coordinate extrema
            if zoom == "dynamic_forecast" or max_lat == None:
                max_lat = max(cone_lat)
                min_lat = min(cone_lat)
                max_lon = max(cone_lon)
                min_lon = min(cone_lon)
            else:
                if max(cone_lat) > max_lat: max_lat = max(cone_lat)
                if min(cone_lat) < min_lat: min_lat = min(cone_lat)
                if max(cone_lon) > max_lon: max_lon = max(cone_lon)
                if min(cone_lon) < min_lon: min_lon = min(cone_lon)

        #--------------------------------------------------------------------------------------

        #Storm-centered plot domain
        if zoom == "dynamic" or zoom == 'dynamic_forecast':
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)

            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Custom plot domain
        else:
            
            #Check to ensure 3 slashes are provided
            if zoom.count("/") != 3:
                raise ValueError("Error: Custom map projection bounds must be provided as 'west/east/south/north'")
            else:
                try:
                    bound_w,bound_e,bound_s,bound_n = zoom.split("/")
                    bound_w = float(bound_w)
                    bound_e = float(bound_e)
                    bound_s = float(bound_s)
                    bound_n = float(bound_n)
                    self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
                except:
                    raise ValueError("Error: Custom map projection bounds must be provided as 'west/east/south/north'")
        
        #Determine number of lat/lon lines to use for parallels & meridians
        self.__plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #--------------------------------------------------------------------------------------
        
        #Identify storm type (subtropical, hurricane, etc)
        first_fcst_wind = np.array(forecast['vmax'])[fcst_hr >= start_slice][0]
        first_fcst_mslp = np.array(forecast['mslp'])[fcst_hr >= start_slice][0]
        first_fcst_type = np.array(forecast['type'])[fcst_hr >= start_slice][0]
        if all_nan(first_fcst_wind) == True:
            storm_type = 'Unknown'
        else:
            subtrop = classify_subtrop(np.array(storm_data['type']))
            cur_wind = first_fcst_wind + 0
            storm_type = get_storm_type(np.nan_to_num(cur_wind),subtrop,'north_atlantic')
        
        #Identify storm name (and storm type, if post-tropical or potential TC)
        matching_times = [i for i in storm_data['date'] if i <= forecast['init']]
        if check_length < 2:
            if all_nan(first_fcst_wind) == True:
                storm_name = storm_data['name']
            else:
                storm_name = num_to_str(int(storm_data['operational_id'][2:4])).upper()
                if first_fcst_wind >= 34 and first_fcst_type in ['TD','SD','SS','TS','HU']: storm_name = storm_data['name'];
                if first_fcst_type not in ['TD','SD','SS','TS','HU']: storm_type = 'Potential Tropical Cyclone'
        else:
            storm_name = num_to_str(int(storm_data['operational_id'][2:4])).upper()
            storm_type = 'Potential Tropical Cyclone'
            storm_tropical = False
            if all_nan(vmax) == True:
                storm_type = 'Unknown'
                storm_name = storm_data['name']
            else:
                for i,(iwnd,ityp) in enumerate(zip(vmax,styp)):
                    if ityp in ['SD','SS','TD','TS','HU']:
                        storm_tropical = True
                        subtrop = classify_subtrop(np.array(storm_data['type']))
                        storm_type = get_storm_type(np.nan_to_num(iwnd),subtrop,'north_atlantic')
                        if np.isnan(iwnd) == True: storm_type = 'Unknown'
                    else:
                        if storm_tropical == True: storm_type = 'Post Tropical Cyclone'
                    if ityp in ['SS','TS','HU']:
                        storm_name = storm_data['name']
        
        #Add left title
        self.ax.set_title(f"{storm_type} {storm_name}",loc='left',fontsize=17,fontweight='bold')

        endash = u"\u2013"
        dot = u"\u2022"
        
        #Get current advisory information
        first_fcst_wind = "N/A" if np.isnan(first_fcst_wind) == True else int(first_fcst_wind)
        first_fcst_mslp = "N/A" if np.isnan(first_fcst_mslp) == True else int(first_fcst_mslp)
        
        #Get time of advisory
        fcst_hr = forecast['fhr']
        start_slice = 0
        if 3 in fcst_hr: start_slice = 1
        forecast_date = (forecast['init']+timedelta(hours=fcst_hr[start_slice])).strftime("%H%M UTC %d %b %Y")
        forecast_id = forecast['advisory_num']
        
        title_text = f"{knots_to_mph(first_fcst_wind)} mph {dot} {first_fcst_mslp} hPa {dot} Forecast #{forecast_id}"
        title_text += f"\nForecast Issued: {forecast_date}"
        
        #Add right title
        self.ax.set_title(title_text,loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':
            
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            uk = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Unknown', marker='o', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=category_color(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=category_color(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=category_color(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=category_color(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=category_color(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=category_color(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=category_color(137))
            self.ax.legend(handles=[ex,sb,uk,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        #Add credit
        try:
            credit_text = f"The cone of uncertainty in this product was generated internally using {cone['year']} official\nNHC cone radii. This cone differs slightly from the official NHC cone.\n\n{plot_credit()}"
        except:
            credit_text = plot_credit()
        self.ax.text(0.99,0.01,credit_text,fontsize=9,color='k',alpha=0.7,
                transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()
    
    def plot_season(self,season,ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single season.
        
        Parameters
        ----------
        season : Season
            Instance of Season.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':False,'fillcolor':'category','linecolor':'category','category_colors':'default','linewidth':1.5,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.__add_prop(prop,default_prop)
        map_prop = self.__add_prop(map_prop,default_map_prop)
        self.__plot_init(ax,map_prop)
        
        #set default properties
        input_prop = prop
        input_map_prop = map_prop
        
        #--------------------------------------------------------------------------------------
        
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        sinfo = season.annual_summary()
        storms = sinfo['id']
        for istorm in storms:

            #Get data for this storm
            storm_data = season.dict[istorm]
            
            #Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['date']

            #Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(lons)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                lons = new_lons.tolist()

            #Add to coordinate extrema
            if max_lat == None:
                max_lat = max(lats)
            else:
                if max(lats) > max_lat: max_lat = max(lats)
            if min_lat == None:
                min_lat = min(lats)
            else:
                if min(lats) < min_lat: min_lat = min(lats)
            if max_lon == None:
                max_lon = max(lons)
            else:
                if max(lons) > max_lon: max_lon = max(lons)
            if min_lon == None:
                min_lon = min(lons)
            else:
                if min(lons) < min_lon: min_lon = min(lons)

            #Draw storm lines
            if prop['linecolor'] == 'category':
                type6 = np.array(storm_data['type'])
                for i in (np.arange(len(lats[1:]))+1):
                    ltype = 'solid'
                    if type6[i] not in ['SS','SD','TD','TS','HU']:
                        ltype = 'dotted'
                    peffect = [path_effects.Stroke(linewidth=prop['linewidth']*1.2, foreground='k'), path_effects.Normal()]
                    self.ax.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]],
                                  '-',color=category_color(np.nan_to_num(storm_data['vmax'][i])),linewidth=prop['linewidth'],linestyle=ltype,
                                  transform=ccrs.PlateCarree(),path_effects = peffect)
            else:
                self.ax.plot(lons,lats,'-',color=prop['linecolor'],linewidth=prop['linewidth'],transform=ccrs.PlateCarree())

        #--------------------------------------------------------------------------------------
        
        #Pre-generated zooms
        bound_w,bound_e,bound_s,bound_n = self.__set_projection(season.basin)
            
        #Determine number of lat/lon lines to use for parallels & meridians
        self.__plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #Add storm labels
        for istorm in storms:

            #Get data for this storm
            storm_data = season.dict[istorm]
            
            #Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['date']

            #Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(lons)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                lons = new_lons.tolist()
                
            #Add storm name at start & end (bound_w = -160, bound_e = -120
            display_name = storm_data['name']
            if display_name.lower() == 'unnamed':
                display_name = int(storm_data['id'][2:4]) if len(storm_data['id']) == 8 else 'UNNAMED'
                
            if lons[0]>(bound_w+0.5) and lons[0]<(bound_e-0.5) and lats[0]>(bound_s-0.5) and lats[0]<(bound_n-0.5):
                self.ax.text(lons[0],lats[0]+1.0,display_name,alpha=0.7,
                         fontweight='bold',fontsize=8.5,color='k',ha='center',va='center',transform=ccrs.PlateCarree())
            if lons[-1]>(bound_w+0.5) and lons[-1]<(bound_e-0.5) and lats[-1]>(bound_s-0.5) and lats[-1]<(bound_n-0.5):
                self.ax.text(lons[-1],lats[-1]+1.0,display_name,alpha=0.7,
                         fontweight='bold',fontsize=8.5,color='k',ha='center',va='center',transform=ccrs.PlateCarree())
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        self.ax.set_title(f"{season.year} Atlantic Hurricane Season",loc='left',fontsize=17,fontweight='bold')

        #Add right title
        endash = u"\u2013"
        dot = u"\u2022"
        self.ax.set_title(f"{sinfo['season_named']} named {dot} {sinfo['season_hurricane']} hurricanes {dot} {sinfo['season_major']} major\n{sinfo['season_ace']:.1f} Cumulative ACE",loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':
            
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=category_color(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=category_color(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=category_color(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=category_color(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=category_color(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=category_color(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=category_color(137))
            self.ax.legend(handles=[ex,sb,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()
        
    def __generate_nhc_cone(self,forecast,dateline,cone_days=5):
        
        r"""
        Generates a cone of uncertainty using forecast data from NHC.
        
        Parameters:
        -----------
        forecast : dict
            Dictionary containing forecast data
        dateline : bool
            If true, grid will be shifted to +0 to +360 degrees longitude. Default is False (-180 to +180 degrees).
        cone_days : int
            Number of forecast days to generate the cone through. Default is 5 days.
        
        """

        #Source: https://www.nhc.noaa.gov/verification/verify3.shtml
        cone_climo_hr = [3,12,24,36,48,72,96,120]
        cone_size_atl = {}
        cone_size_atl[2019] = [16,26,41,54,68,102,151,198]
        cone_size_atl[2018] = [16,26,43,56,74,103,151,198]
        cone_size_atl[2017] = [16,29,45,63,78,107,159,211]
        cone_size_atl[2016] = [16,30,49,66,84,115,165,237]
        cone_size_atl[2015] = [16,32,52,71,90,122,170,225]
        cone_size_atl[2014] = [16,33,52,72,92,125,170,226]
        cone_size_atl[2013] = [16,33,52,72,92,128,177,229]
        cone_size_atl[2012] = [16,36,56,75,95,141,180,236]
        cone_size_atl[2011] = [16,36,59,79,98,144,190,239]
        cone_size_atl[2010] = [16,36,62,85,108,161,220,285]
        cone_size_atl[2009] = [16,36,62,89,111,167,230,302]
        cone_size_atl[2008] = [16,39,67,92,118,170,233,305]

        cone_size_pac = {}
        cone_size_pac[2019] = [16,25,38,48,62,88,115,145]
        cone_size_pac[2018] = [16,25,39,50,66,94,125,162]
        cone_size_pac[2017] = [16,25,40,51,66,93,116,151]
        cone_size_pac[2016] = [16,27,42,55,70,100,137,172]
        cone_size_pac[2015] = [16,26,42,54,69,100,143,182]
        cone_size_pac[2014] = [16,30,46,62,79,105,154,190]
        cone_size_pac[2013] = [16,30,49,66,82,111,157,197]
        cone_size_pac[2012] = [16,33,52,72,89,121,170,216]
        cone_size_pac[2011] = [16,33,59,79,98,134,187,230]
        cone_size_pac[2010] = [16,36,59,82,102,138,174,220]
        cone_size_pac[2009] = [16,36,59,85,105,148,187,230]
        cone_size_pac[2008] = [16,36,66,92,115,161,210,256]

        def temporal_interpolation(value, orig_times, target_times):
            f = interp.interp1d(orig_times,value)
            ynew = f(target_times)
            return ynew

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

        def findNearest(array,val):
            return array[np.abs(array - val).argmin()]

        ###### Plot cyclogenesis location density
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

            #Mask out values less than radius
            return_arr[dist <= rad] = 1

            #Attach small array into larger subset array
            small_coords = {'lat':nlat,'lon':nlon}

            return return_arr, small_coords




        #Retrieve cone size for given year
        if forecast['init'].year in cone_size_atl.keys():
            cone_year = forecast['init'].year
            if forecast['basin'] == 'north_atlantic':
                cone_size = cone_size_atl[forecast['init'].year]
            elif forecast['basin'] == 'east_pacific':
                cone_size = cone_size_pac[forecast['init'].year]
            else:
                raise RuntimeError("Error: No cone information is available for the requested basin.")
        else:
            cone_year = 2008
            warnings.warn(f"No cone information is available for the requested year. Defaulting to 2008 cone.")
            if forecast['basin'] == 'north_atlantic':
                cone_size = cone_size_atl[2008]
            elif forecast['basin'] == 'east_pacific':
                cone_size = cone_size_pac[2008]
            else:
                raise RuntimeError("Error: No cone information is available for the requested basin.")
            #raise RuntimeError("Error: No cone information is available for the requested year.")
        
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
        elif 3 in forecast['fhr'] and 0 in forecast['fhr']:
            fcst_lon = forecast['lon'][1:]
            fcst_lat = forecast['lat'][1:]
            fhr = forecast['fhr'][1:]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.1,0.1)
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
            cone_day_cap = fhr.index(cone_days*24)+1
            fcst_lon = fcst_lon[:cone_day_cap]
            fcst_lat = fcst_lat[:cone_day_cap]
            fhr = fhr[:cone_day_cap]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(interp_fhr_idx[0],t[-1]+0.1,0.1)
        else:
            cone_day_cap = len(fhr)
        
        #Account for dateline
        if dateline == True:
            temp_lon = np.array(fcst_lon)
            temp_lon[temp_lon<0] = temp_lon[temp_lon<0]+360.0
            fcst_lon = temp_lon.tolist()

        #Interpolate forecast data temporally
        interp_kind = 'quadratic'
        if len(t) == 2: interp_kind = 'linear'
        x1 = interp.interp1d(t,fcst_lon,kind=interp_kind)
        y1 = interp.interp1d(t,fcst_lat,kind=interp_kind)
        interp_fhr = interp_fhr_idx * 6
        interp_lon = x1(interp_fhr_idx)
        interp_lat = y1(interp_fhr_idx)

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

        return_dict = {'lat':gridlats,'lon':gridlons,'lat2d':gridlats2d,'lon2d':gridlons2d,'cone':griddata,
                       'center_lon':interp_lon,'center_lat':interp_lat,'year':cone_year}
        return return_dict