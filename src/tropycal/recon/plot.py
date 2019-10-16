import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from ..plot import Plot
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
    import matplotlib.patches as mpatches

except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class ReconPlot(Plot):
                 
    def plot_points(self,recon_data,zoom="dynamic",varname='wspd',barbs=False,scatter=False,\
                    ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of recon data points
        
        Parameters
        ----------
        recon_data : dataframe
            Recon data, must be dataframe
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic" - default. Dynamically focuses the domain using the tornado track(s) plotted.
            "north_atlantic" - North Atlantic Ocean basin
            "conus", "east_conus"
            "lonW/lonE/latS/latN" - Custom plot domain
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
        default_prop={'obs_colors':'plasma','obs_levels':np.arange(30,151,10),'sortby':varname,'linewidth':1.5,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF',\
                          'linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
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
        if isinstance(recon_data,pd.core.frame.DataFrame):
            pass
        else:
            raise RuntimeError("Error: recon_data must be dataframe")

        #Retrieve storm data
        lats = recon_data['lat']
        lons = recon_data['lon']

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

        #Plot recon data as specified
        
        if barbs:
            
            dataSort = recon_data.sort_values(by='wspd').reset_index(drop=True)
            norm = mlib.colors.Normalize(vmin=min(prop['obs_levels']), vmax=max(prop['obs_levels']))
            cmap = mlib.cm.get_cmap(prop['obs_colors'])
            colors = cmap(norm(dataSort['wspd'].values))
            colors = [tuple(i) for i in colors]
            qv = plt.barbs(dataSort['lon'],dataSort['lat'],\
                       *uv_from_wdir(dataSort['wspd'],dataSort['wdir']),color=colors,length=5,linewidth=0.5)
        
        if scatter:
                        
            dataSort = recon_data.sort_values(by=prop['sortby'],ascending=(prop['sortby']!='p_sfc')).reset_index(drop=True)
            prop['obs_levels']=np.linspace(min(dataSort[varname]),max(dataSort[varname]),256)
            cmap = mlib.cm.get_cmap(prop['obs_colors'])
            
            sc = plt.scatter(dataSort['lon'],dataSort['lat'],c=dataSort[varname],cmap = cmap,\
                             vmin=min(prop['obs_levels']), vmax=max(prop['obs_levels']), s=prop['ms'])

        #--------------------------------------------------------------------------------------
        
        #Pre-generated zooms
        if zoom in ['north_atlantic','conus','east_conus']:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(zoom)
            
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
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        dot = u"\u2022"
        if barbs:
            vartitle = f'{dot} flight level wind'
        if scatter:
            if varname == 'sfmr':
                vartitle = f'{dot} SFMR surface wind'
            if varname == 'wspd':
                vartitle = f'{dot} flight level wind'
            if varname == 'p_sfc':
                vartitle = f'{dot} surface pressure'
        self.ax.set_title('Recon '+vartitle,loc='left',fontsize=17,fontweight='bold')

        #Add right title
        #max_ppf = max(PPF)
        start_date = dt.strftime(min(recon_data['time']),'%H:%M UTC %d %b %Y')
        end_date = dt.strftime(max(recon_data['time']),'%H:%M UTC %d %b %Y')
        self.ax.set_title(f'Start ... {start_date}\nEnd ... {end_date}',loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend

        #Add colorbar
            
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax,'/'.join([str(b) for b in [bound_w,bound_e,bound_s,bound_n]])
        else:
            plt.show()
            plt.close()


    def plot_interp(self,dfRecon,radlim=150,ax=None,return_ax=False,prop={}):

        r"""
        Creates a plot of storm-centered recon data interpolated to a grid
        
        Parameters
        ----------
        recon_data : dataframe
        radlim : int
            Radius (km) from the center of the storm that interpolation is calculated,
            and field plotted ... axis limits will be [-radlim,radlim,-radlim,radlim]
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Properties of plot
        """

        #Set default properties
        default_prop={'colors':'category','levels':[34,64,83,96,113,137,200]}
 
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        
        mlib.rcParams.update({'font.size': 16})
        
        #Interpolate recon segment
        
        grid_x,grid_y,grid_z = interpRecon(dfRecon,radlim=radlim)
        
        fig = plt.figure(figsize=prop['figsize'])
        if ax==None:
            self.ax = plt.subplot()
        else:
            self.ax = ax
        
        colors,clevs = recon_colors(varname,prop['colors'],prop['levels'])
            
        plt.contourf(grid_x,grid_y,grid_z,clevs,colors=colors)
        rightarrow = u"\u2192"
        plt.xlabel(f'W {rightarrow} E Distance (km)')
        plt.ylabel(f'S {rightarrow} N Distance (km)')
        plt.axis([-radlim,radlim,-radlim,radlim])
        plt.axis('equal')
        
        cbar=plt.colorbar()
        cbar.set_label('wind (kt)')
                
        #--------------------------------------------------------------------------------------
        
        #Add left title
        self.ax.set_title('Recon interpolated',loc='left',fontsize=17,fontweight='bold')

        #Add right title
        #max_ppf = max(PPF)
        start_date = dt.strftime(min(dfRecon['time']),'%H:%M UTC %d %b %Y')
        end_date = dt.strftime(max(dfRecon['time']),'%H:%M UTC %d %b %Y')
        self.ax.set_title(f'Start ... {start_date}\nEnd ... {end_date}',loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
           
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()

    
            
    
