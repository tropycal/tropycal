import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from .tools import *
from ..tracks.tools import *

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib as mlib
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
            
        #CONUS plot domain
        if zoom == "conus":
            bound_w = -130.0
            bound_e = -65.0
            bound_s = 20.0
            bound_n = 50.0

        #CONUS plot domain
        if zoom == "east_conus":
            bound_w = -105.0
            bound_e = -60.0
            bound_s = 20.0
            bound_n = 48.0
            
        #Set map extent
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        return bound_w, bound_e, bound_s, bound_n
                 
    def plot_tornadoes(self,tornado,zoom="east_conus",plotPPF=False,ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm track.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic" - default. Dynamically focuses the domain using the tornado track(s) plotted.
            "north_atlantic" - North Atlantic Ocean basin
            "conus", "east_conus"
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
        default_prop={'plotType':'tracks','PPFcolors':'SPC','PPFlevels':[2,5,10,15,30,45,60,100],\
                      'EFcolors':'default','linewidth':2.0,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF',\
                          'linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
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
        try:
            tornado_data = tornado
        except:
            raise RuntimeError("Error: Storm must be dataframe")

        #Retrieve storm data
        slat = tornado_data['slat']
        slon = tornado_data['slon']
        elat = tornado_data['elat']
        elon = tornado_data['elon']
        mag = tornado_data['mag']

        mnlat = (slat+elat)*.5
        mnlon = (slon+elon)*.5

        #Add to coordinate extrema
        if max_lat == None:
            max_lat = max(mnlat)
        else:
            if max(mnlat) > max_lat: max_lat = max(mnlat)
        if min_lat == None:
            min_lat = min(mnlat)
        else:
            if min(mnlat) < min_lat: min_lat = min(mnlat)
        if max_lon == None:
            max_lon = max(mnlon)
        else:
            if max(mnlon) > max_lon: max_lon = max(mnlon)
        if min_lon == None:
            min_lon = min(mnlon)
        else:
            if min(mnlon) < min_lon: min_lon = min(mnlon)

        #Plot PPF
        if plotPPF in ['total','daily',True]:
            if plotPPF == True: plotPPF='daily'
            PPF,longrid,latgrid = getPPF(tornado_data,method=plotPPF)
            
            colors,clevs = ppf_colors(plotPPF,prop['PPFcolors'],prop['PPFlevels'])
                    
            cbmap = self.ax.contourf((longrid[:-1]+longrid[1:])*.5,(latgrid[:-1]+latgrid[1:])*.5,PPF,\
                             levels=clevs,colors=colors,alpha=0.5)

        #Plot tornadoes as specified
        EFcolors = ef_colors(prop['EFcolors'])
        
        for _,row in tornado_data.iterrows():
            plt.plot([row['slon'],row['elon']+.01],[row['slat'],row['elat']+.01], \
                lw=prop['linewidth'],color=EFcolors[row['mag']],zorder=row['mag']+100, \
                path_effects=[path_effects.Stroke(linewidth=prop['linewidth']*1.5, foreground='w'), path_effects.Normal()])

        #--------------------------------------------------------------------------------------
        
        #Pre-generated zooms
        if zoom in ['north_atlantic','conus','east_conus']:
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
        
        if return_ax == False:
            #Add left title
            ppf_title = ''
            if plotPPF in ['total',True]:
                ppf_title = ' and total PPF (%)'
            if plotPPF == 'daily':
                ppf_title = ' and daily PPF (%)'
            self.ax.set_title('Tornado tracks'+ppf_title,loc='left',fontsize=17,fontweight='bold')
    
            #Add right title
            #max_ppf = max(PPF)
            start_date = dt.strftime(min(tornado_data['UTC_time']),'%H:%M UTC %d %b %Y')
            end_date = dt.strftime(max(tornado_data['UTC_time']),'%H:%M UTC %d %b %Y')
            endash = u"\u2013"
            dot = u"\u2022"
            self.ax.set_title(f'{start_date}\n {endash} {end_date}',loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        handles=[]
        for ef,color in enumerate(EFcolors):
            count = len(tornado_data[tornado_data['mag']==ef])
            handles.append(mlines.Line2D([], [], linestyle='-',color=color,label=f'EF-{ef} ({count})'))
        leg_tor = self.ax.legend(handles=handles,loc='lower left',fontsize=11.5)
        
        #Add PPF colorbar
        if plotPPF != False:
            cax = self.fig.add_axes([.235, 0.235, 0.015, 0.155])
            self.fig.colorbar(cbmap,cax=cax,orientation='vertical')
        
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax,'/'.join([str(b) for b in [bound_w,bound_e,bound_s,bound_n]]),leg_tor
        else:
            plt.show()
            plt.close()