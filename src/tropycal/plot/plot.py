import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import pkg_resources

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
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class Plot:
    
    def check_res(self,res):
        
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
        if mapobj is None:
            self.proj = getattr(ccrs, proj)(**kwargs)
        else:
            self.proj = mapobj
        
    def create_geography(self,prop):
        
        r"""
        Set up the map geography and colors.
        
        Parameters:
        -----------
        prop : dict
            dict entry containing information about the map geography and colors
        """
        
        #get resolution corresponding to string in prop
        res = self.check_res(prop['res'])
        
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
        
        #If only one coordinate point, artificially induce a spread
        if bound_w == bound_e:
            bound_w = bound_w - 0.6
            bound_e = bound_e + 0.6
        if bound_s == bound_n:
            bound_n = bound_n + 0.6
            bound_s = bound_s - 0.6

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
    
    def plot_lat_lon_lines(self,bounds,zorder=None):
        
        r"""
        Plots parallels and meridians that are constrained by the map bounds.
        
        Parameters:
        -----------
        bounds : list
            List containing map bounds.
        """
        
        #Suppress gridliner warnings
        warnings.filterwarnings("ignore")
        
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
        rthres = 20
        if new_yrng < 160.0 or new_xrng < 160.0:
            rthres = 10
        if new_yrng < 40.0 or new_xrng < 40.0:
            rthres = 5
        if new_yrng < 25.0 or new_xrng < 25.0:
            rthres = 2
        if new_yrng < 9.0 or new_xrng < 9.0:
            rthres = 1
        parallels = np.arange(rdown(bound_s,rthres),rup(bound_n,rthres)+rthres,rthres)
        meridians = np.arange(rdown(bound_w,rthres),rup(bound_e,rthres)+rthres,rthres)
        
        add_kwargs = {}
        if zorder is not None:
            add_kwargs = {'zorder':zorder}
        
        #Fix for dateline crossing
        if self.proj.proj4_params['lon_0'] == 180.0:
            
            #Recalculate parallels and meridians
            parallels = np.arange(rup(bound_s,rthres),rdown(bound_n,rthres)+rthres,rthres)
            meridians = np.arange(rup(bound_w,rthres),rdown(bound_e,rthres)+rthres,rthres)
            meridians2 = np.copy(meridians)
            meridians2[meridians2>180.0] = meridians2[meridians2>180.0]-360.0
            all_meridians = np.arange(0.0,360.0+rthres,rthres)
            all_parallels = np.arange(rdown(-90.0,rthres),90.0+rthres,rthres)
            
            #First call with no labels but gridlines plotted
            gl1 = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=False,xlocs=all_meridians,ylocs=all_parallels,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted',**add_kwargs)
            #Second call with labels but no gridlines
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,xlocs=meridians,ylocs=parallels,linewidth=0.0,color='k',alpha=0.0,linestyle='dotted',**add_kwargs)
            
            #this syntax is deprecated in newer functions of cartopy
            try:
                gl.xlabels_top = False
                gl.ylabels_right = False
            except:
                gl.top_labels = False
                gl.right_labels = False
            
            gl.xlocator = mticker.FixedLocator(meridians2)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER

        else:
            #Add meridians and parallels
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(),draw_labels=True,linewidth=1.0,color='k',alpha=0.5,linestyle='dotted',**add_kwargs)
            
            #this syntax is deprecated in newer functions of cartopy
            try:
                gl.xlabels_top = False
                gl.ylabels_right = False
            except:
                gl.top_labels = False
                gl.right_labels = False
            
            gl.xlocator = mticker.FixedLocator(meridians)
            gl.ylocator = mticker.FixedLocator(parallels)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        #Reset plot bounds
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
        
    def plot_init(self,ax,map_prop,plot_geography=True):
        
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
        if self.proj is None:
            self.create_cartopy(proj='PlateCarree',central_longitude=0.0)
        
        #create figure
        if ax is None:
            self.fig = plt.figure(figsize=map_prop['figsize'],dpi=map_prop['dpi'])
            self.ax = plt.axes(projection=self.proj)
        else:
            fig_numbers = [x.num for x in mlib._pylab_helpers.Gcf.get_all_fig_managers()]
            if len(fig_numbers) > 0:
                self.fig = plt.figure(fig_numbers[-1])
            else:
                self.fig = plt.figure(figsize=map_prop['figsize'],dpi=map_prop['dpi'])
            self.ax = ax
        
        #Attach geography to plot, lat/lon lines, etc.
        if plot_geography:
            self.create_geography(map_prop)
    
    def add_prop(self,input_prop,default_prop):
        
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
    
    def set_projection(self,domain):
        
        r"""
        Sets a predefined map projection domain.
        
        Parameters
        ----------
        domain : str
            Name of map projection to domain over.
        """
        
        #North Atlantic plot domain
        if domain == "north_atlantic":
            bound_w = -105.0
            bound_e = -5.0
            bound_s = 0.0
            bound_n = 65.0
            
        #East Pacific plot domain
        elif domain == "east_pacific":
            bound_w = -180.0+360.0 
            bound_e = -80+360.0 
            bound_s = 0.0
            bound_n = 65.0
            
        #West Pacific plot domain
        elif domain == "west_pacific":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = 0.0
            bound_n = 65.0
            
        #North Indian plot domain
        elif domain == "north_indian":
            bound_w = 30.0
            bound_e = 110.0
            bound_s = -5.0
            bound_n = 40.0
            
        #South Indian plot domain
        elif domain == "south_indian":
            bound_w = 20.0
            bound_e = 110.0
            bound_s = -50.0
            bound_n = 5.0
            
        #Australia plot domain
        elif domain == "australia":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = -60.0
            bound_n = 0.0
            
        #South Pacific plot domain
        elif domain == "south_pacific":
            bound_w = 140.0
            bound_e = -120.0+360.0
            bound_s = -65.0
            bound_n = 0.0
            
        #Global plot domain
        elif domain == "all":
            bound_w = 0.1
            bound_e = 360.0
            bound_s = -90.0
            bound_n = 90.0
            
        #CONUS plot domain
        elif domain == "conus":
            bound_w = -130.0
            bound_e = -65.0
            bound_s = 20.0
            bound_n = 50.0

        #CONUS plot domain
        elif domain == "east_conus":
            bound_w = -105.0
            bound_e = -60.0
            bound_s = 20.0
            bound_n = 48.0
        
        #Custom domain
        else:
            
            #Error check
            if isinstance(domain,dict) == False:
                msg = "Custom domains must be of type dict."
                raise TypeError(msg)
            
            #Retrieve map bounds
            keys = domain.keys()
            check = [False, False, False, False]
            for key in keys:
                if key[0].lower() == 'n': check[0] = True; bound_n = domain[key]
                if key[0].lower() == 's': check[1] = True; bound_s = domain[key]
                if key[0].lower() == 'e': check[2] = True; bound_e = domain[key]
                if key[0].lower() == 'w': check[3] = True; bound_w = domain[key]
            if False in check:
                msg = "Custom domains must be of type dict with arguments for 'n', 's', 'e' and 'w'."
                raise ValueError(msg)
            
        #Set map extent
        self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        return bound_w, bound_e, bound_s, bound_n
    
    def plot_credit(self):
        
        return "Plot generated using troPYcal"
    
    def add_credit(self,text):
        
        if self.use_credit:
            a = self.ax.text(0.99,0.01,text,fontsize=10,color='k',alpha=0.7,fontweight='bold',
                    transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
            a.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                           path_effects.Normal()])
    
    def rgb(self,rgb):
        r,g,b = rgb
        r = int(r)
        g = int(g)
        b = int(b)
        return '#%02x%02x%02x' % (r, g, b)
