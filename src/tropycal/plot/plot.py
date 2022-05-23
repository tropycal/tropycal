import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import pkg_resources

from ..tracks.tools import *
from ..utils import *

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
            all_meridians = np.arange(-180.0,180.0+rthres,rthres)
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
    
    def plot_dot(self,lon,lat,date,vmax,i_type,zorder,storm_data,prop):

        r"""
        Plot a dot on the map per user settings.

        Parameters
        ----------
        lon : int, float
            Longitude of the dot
        lat : int, float
            Latitude of the dot
        date : datetime.datetime
            Datetime object corresponding to the time of the dot in UTC
        vmax : int, float
            Sustained wind in knots
        i_type : str
            Storm type

        Other Parameters
        ----------------
        zorder : int
            Z-order of dots on the map.
        storm_data : dict
            Storm data dictionary.
        prop : dict
            Dictionary containing plot properties.
        
        Returns
        -------
        segmented_colors : bool
            Information for colorbar generation on whether a segmented colormap was used or not.
        """

        #Determine fill color, with SSHWS scale used as default
        if prop['fillcolor'] == 'category':
            segmented_colors = True
            fill_color = get_colors_sshws(np.nan_to_num(vmax))

        #Use user-defined colormap if another storm variable
        elif isinstance(prop['fillcolor'],str) == True and prop['fillcolor'] in ['vmax','mslp','dvmax_dt','speed']:
            segmented_colors = True
            color_variable = storm_data[prop['fillcolor']]
            if prop['levels'] is None: #Auto-determine color levels if needed
                prop['levels'] = [np.nanmin(color_variable),np.nanmax(color_variable)]
            cmap,levels = get_cmap_levels(prop['fillcolor'],prop['cmap'],prop['levels'])
            fill_color = cmap((color_variable-min(levels))/(max(levels)-min(levels)))[i]

        #Otherwise go with user input as is
        else:
            segmented_colors = False
            fill_color = prop['fillcolor']

        #Determine dot type
        marker_type = '^'
        if i_type in constants.SUBTROPICAL_ONLY_STORM_TYPES:
            marker_type = 's'
        elif i_type in constants.TROPICAL_ONLY_STORM_TYPES:
            marker_type = 'o'

        #Plot marker
        self.ax.plot(lon,lat,marker_type,mfc=fill_color,mec='k',mew=0.5,
                     zorder=zorder,ms=prop['ms'],transform=ccrs.PlateCarree())

        return segmented_colors

    
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
    
    def add_legend(self,prop,segmented_colors,levels=None,cmap=None,storm=None):
        
        #Linecolor category with dots
        if prop['fillcolor'] == 'category' and prop['dots'] == True:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex,sb,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5}, loc=1)

        #Linecolor category without dots
        elif prop['linecolor'] == 'category' and prop['dots'] == False:
            ex = mlines.Line2D([], [], linestyle='dotted', label='Non-Tropical', color='k')
            td = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Depression', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Storm', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='solid', label='Category 1', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='solid', label='Category 2', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='solid', label='Category 3', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='solid', label='Category 4', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='solid', label='Category 5', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5}, loc=1)

        #Non-segmented custom colormap with dots
        elif prop['dots'] == True and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
            handles=[ex,sb,td]
            self.ax.legend(handles=handles,fontsize=11.5, prop={'size':11.5}, loc=1)
        
        #Non-segmented custom colormap without dots
        elif prop['dots'] == False and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='dotted',label='Non-Tropical', color=prop['linecolor'])
            td = mlines.Line2D([], [], linestyle='solid',label='Tropical', color=prop['linecolor'])
            handles=[ex,td]
            self.ax.legend(handles=handles,fontsize=11.5, prop={'size':11.5}, loc=1)

        #Custom colormap with dots
        elif prop['dots'] == True and segmented_colors == True:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color='w')
            handles=[ex,sb,td]
            for _ in range(7):
                handles.append(mlines.Line2D([], [], linestyle='-',label='',lw=0))
            l = self.ax.legend(handles=handles,fontsize=11.5)
            plt.draw()
            
            #Get the bbox
            try:
                bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
            except:
                bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())

            #Define colorbar axis
            cax = self.fig.add_axes([bb.x0+0.47*bb.width, bb.y0+.057*bb.height, 0.015, .65*bb.height])
            norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
            cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical',\
                                     ticks=levels)
            
            cax.tick_params(labelsize=11.5)
            cax.yaxis.set_ticks_position('left')
            cbar.set_label(prop['fillcolor'],fontsize=11.5,rotation=90)
        
            rect_offset = 0.0
            if prop['cmap'] == 'category' and prop['fillcolor'] == 'vmax':
                cax.yaxis.set_ticks(np.linspace(min(levels),max(levels),len(levels)))
                cax.yaxis.set_ticklabels(levels)
                cax2 = cax.twinx()
                cax2.yaxis.set_ticks_position('right')
                cax2.yaxis.set_ticks((np.linspace(0,1,len(levels))[:-1]+np.linspace(0,1,len(levels))[1:])*.5)
                cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
                cax2.tick_params('both', length=0, width=0, which='major')
                cax.yaxis.set_ticks_position('left')
                rect_offset = 0.7
            if prop['fillcolor'] == 'date':
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in levels],fontsize=11.5)
        
        #Custom colormap without dots
        else:
            ex = mlines.Line2D([], [], linestyle='dotted',label='Non-Tropical', color='k')
            td = mlines.Line2D([], [], linestyle='solid',label='Tropical', color='k')
            handles=[ex,td]
            for _ in range(7):
                handles.append(mlines.Line2D([], [], linestyle='-',label='',lw=0))
            l=self.ax.legend(handles=handles,fontsize=11.5)
            plt.draw()
            
            #Get the bbox
            try:
                bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
            except:
                bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())
                
            #Define colorbar axis
            cax = self.fig.add_axes([bb.x0+0.47*bb.width, bb.y0+.057*bb.height, 0.015, .65*bb.height])
            norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
            cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical',\
                                     ticks=levels)
            
            cax.tick_params(labelsize=11.5)
            cax.yaxis.set_ticks_position('left')
            cbarlab = make_var_label(prop['linecolor'],storm)
            cbar.set_label(cbarlab,fontsize=11.5,rotation=90)
        
            rect_offset = 0.0
            if prop['cmap'] == 'category' and prop['linecolor'] == 'vmax':
                cax.yaxis.set_ticks(np.linspace(min(levels),max(levels),len(levels)))
                cax.yaxis.set_ticklabels(levels)
                cax2 = cax.twinx()
                cax2.yaxis.set_ticks_position('right')
                cax2.yaxis.set_ticks((np.linspace(0,1,len(levels))[:-1]+np.linspace(0,1,len(levels))[1:])*.5)
                cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
                cax2.tick_params('both', length=0, width=0, which='major')
                cax.yaxis.set_ticks_position('left')
                rect_offset = 0.7
            if prop['linecolor'] == 'date':
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in levels],fontsize=11.5)
