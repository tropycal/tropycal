import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt

from ..plot import Plot

#Import tools
from .tools import *
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
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches

except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class ReconPlot(Plot):
    
    def __init__(self):
        
        self.use_credit = True
                 
    def plot_points(self,storm,recon_data,domain="dynamic",varname='wspd',barbs=False,scatter=False,\
                    ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of recon data points
        
        Parameters
        ----------
        recon_data : dataframe
            Recon data, must be dataframe
        domain : str
            Domain for the plot. Can be one of the following:
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
        
        if not barbs and not scatter:
            scatter = True
        
        #Set default properties
        default_prop={'cmap':'category','levels':(np.min(recon_data[varname]),np.max(recon_data[varname])),\
                      'sortby':varname,'ascending':(varname!='p_sfc'),'linewidth':1.5,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF',\
                          'linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
        #set default properties
        input_prop = prop
        input_map_prop = map_prop
        
        #--------------------------------------------------------------------------------------
        
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        #Retrieve storm data
        storm_data = storm.dict
        vmax = storm_data['vmax']
        styp = storm_data['type']
        sdate = storm_data['date']


        #Check recon_data type
        if isinstance(recon_data,pd.core.frame.DataFrame):
            pass
        else:
            raise RuntimeError("Error: recon_data must be dataframe")

        #Retrieve storm data
        lats = recon_data['lat']
        lons = recon_data['lon']

        #Add to coordinate extrema
        if max_lat is None:
            max_lat = max(lats)
        else:
            if max(lats) > max_lat: max_lat = max(lats)
        if min_lat is None:
            min_lat = min(lats)
        else:
            if min(lats) < min_lat: min_lat = min(lats)
        if max_lon is None:
            max_lon = max(lons)
        else:
            if max(lons) > max_lon: max_lon = max(lons)
        if min_lon is None:
            min_lon = min(lons)
        else:
            if min(lons) < min_lon: min_lon = min(lons)

        #Plot recon data as specified
        
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['levels'])
        
        if varname in ['vmax','sfmr','fl_to_sfc'] and prop['cmap'] == 'category':
            vmin = min(clevs); vmax = max(clevs)
        else:
            vmin = min(prop['levels']); vmax = max(prop['levels'])
        
        if barbs:
            
            dataSort = recon_data.sort_values(by='wspd').reset_index(drop=True)
            norm = mlib.colors.Normalize(vmin=min(prop['levels']), vmax=max(prop['levels']))
            cmap = mlib.cm.get_cmap(prop['cmap'])
            colors = cmap(norm(dataSort['wspd'].values))
            colors = [tuple(i) for i in colors]
            qv = plt.barbs(dataSort['lon'],dataSort['lat'],\
                       *uv_from_wdir(dataSort['wspd'],dataSort['wdir']),color=colors,length=5,linewidth=0.5)
            
#            qv.set_path_effects([patheffects.Stroke(linewidth=2, foreground='white'),
#                       patheffects.Normal()])
#    
        if scatter:
                        
            dataSort = recon_data.sort_values(by=prop['sortby'],ascending=prop['ascending']).reset_index(drop=True)
          
            cbmap = plt.scatter(dataSort['lon'],dataSort['lat'],c=dataSort[varname],\
                                cmap=cmap,vmin=vmin,vmax=vmax, s=prop['ms'])

        #--------------------------------------------------------------------------------------
        
        #Storm-centered plot domain
        if domain == "dynamic":
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Determine number of lat/lon lines to use for parallels & meridians
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]
            
        subtrop = classify_subtropical(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
        
        dot = u"\u2022"
        if barbs:
            vartitle = get_recon_title('wspd')
        if scatter:
            vartitle = get_recon_title(varname)
        self.ax.set_title(f"{storm_type} {storm_data['name']}\n" + 'Recon: '+' '.join(vartitle),loc='left',fontsize=17,fontweight='bold')

        #Add right title
        start_date = dt.strftime(min(recon_data['time']),'%H:%M UTC %d %b %Y')
        end_date = dt.strftime(max(recon_data['time']),'%H:%M UTC %d %b %Y')
        self.ax.set_title(f'Start ... {start_date}\nEnd ... {end_date}',loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        
        #Phantom legend
        handles=[]
        for _ in range(10):
            handles.append(mlines.Line2D([], [], linestyle='-',label='',lw=0))
        l = self.ax.legend(handles=handles,loc='upper left',fancybox=True,framealpha=0,fontsize=11.5)
        plt.draw()

        #Get the bbox
        try:
            bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
        except:
            bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())
        bb_ax = self.ax.get_position()

        #Define colorbar axis
        cax = self.fig.add_axes([bb.x0+bb.width, bb.y0-.05*bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical',\
                                 ticks=clevs)
            
        if len(prop['levels'])>2:
            cax.yaxis.set_ticks(np.linspace(min(clevs),max(clevs),len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
        else:
            cax.yaxis.set_ticks(clevs)
        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
    
        rect_offset = 0.0
        if prop['cmap']=='category' and varname=='sfmr':
            cax.yaxis.set_ticks(np.linspace(min(clevs),max(clevs),len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0,1,len(clevs))[:-1]+np.linspace(0,1,len(clevs))[1:])*.5)
            cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            
            rect_offset = 0.7
            
        rectangle = mpatches.Rectangle((bb.x0,bb.y0-0.1*bb.height),(1.8+rect_offset)*bb.width,1.1*bb.height,\
                                       fc = 'w',edgecolor = '0.8',alpha = 0.8,\
                                       transform=self.fig.transFigure, zorder=2)
        self.ax.add_patch(rectangle)

        
        
        
        #Add plot credit
        text = self.plot_credit()
        self.add_credit(text)
        
        #Return axis if specified, otherwise display figure
        if ax is not None or return_ax == True:
            return self.ax,'/'.join([str(b) for b in [bound_w,bound_e,bound_s,bound_n]])
        else:
            plt.show()
            plt.close()
        

    def plot_swath(self,storm,Maps,varname,swathfunc,track_dict,radlim=200,\
                   domain="dynamic",ax=None,return_ax=False,prop={},map_prop={}):

        #Set default properties
        default_prop={'cmap':'category','levels':None,'left_title':'','right_title':'All storms','pcolor':True}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
                          
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
                
        #Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        #Retrieve recon data
        lats = Maps['center_lat']
        lons = Maps['center_lon']
        
        #Retrieve storm data
        storm_data = storm.dict
        vmax = storm_data['vmax']
        styp = storm_data['type']
        sdate = storm_data['date']

        #Add to coordinate extrema
        if max_lat is None:
            max_lat = max(lats)+2.5
        else:
            if max(lats) > max_lat: max_lat = max(lats)
        if min_lat is None:
            min_lat = min(lats)-2.5
        else:
            if min(lats) < min_lat: min_lat = min(lats)
        if max_lon is None:
            max_lon = max(lons)+2.5
        else:
            if max(lons) > max_lon: max_lon = max(lons)
        if min_lon is None:
            min_lon = min(lons)-2.5
        else:
            if min(lons) < min_lon: min_lon = min(lons)      
        
        bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
                
        distproj = ccrs.LambertConformal()
        out = distproj.transform_points(ccrs.PlateCarree(),np.array([bound_w,bound_w,bound_e,bound_e]),\
                                        np.array([bound_s,bound_n,bound_s,bound_n]))
        grid_res = 1*1e3 #m
        xi = np.arange(int(min(out[:,0])/grid_res)*grid_res,int(max(out[:,0])/grid_res)*grid_res+grid_res,grid_res)
        yi = np.arange(int(min(out[:,1])/grid_res)*grid_res,int(max(out[:,1])/grid_res)*grid_res+grid_res,grid_res)
        xmgrid,ymgrid = np.meshgrid(xi,yi)
        
        out = distproj.transform_points(ccrs.PlateCarree(),Maps['center_lon'],Maps['center_lat'])
        
        cx = np.rint(gfilt(out[:,0],1)/grid_res)*grid_res
        cy = np.rint(gfilt(out[:,1],1)/grid_res)*grid_res
        aggregate_grid=np.ones(xmgrid.shape)*np.nan

        def nanfunc(func,a,b):
            c = np.concatenate([a[None],b[None]])
            c = np.ma.array(c, mask=np.isnan(c))
            d = func(c,axis=0)
            e = d.data
            e[d.mask] = np.nan
            return e

        for t,(x_center,y_center,var) in enumerate(zip(cx,cy,Maps['maps'])):
            x_fromc = x_center+Maps['grid_x']*1e3
            y_fromc = y_center+Maps['grid_y']*1e3
            inrecon = np.where((xmgrid>=np.min(x_fromc)) & (xmgrid<=np.max(x_fromc)) & \
                           (ymgrid>=np.min(y_fromc)) & (ymgrid<=np.max(y_fromc)))
            inmap = np.where((x_fromc>=np.min(xmgrid)) & (x_fromc<=np.max(xmgrid)) & \
                           (y_fromc>=np.min(ymgrid)) & (y_fromc<=np.max(ymgrid)))
            aggregate_grid[inrecon] = nanfunc(swathfunc,aggregate_grid[inrecon],var[inmap])
        
    
        if prop['levels'] is None:
            prop['levels'] = (np.nanmin(aggregate_grid),np.nanmax(aggregate_grid))
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['levels'])
                        
        out = self.proj.transform_points(distproj,xmgrid,ymgrid)
        lons = out[:,:,0]
        lats = out[:,:,1]
        
        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(lons,lats,aggregate_grid,cmap=cmap,norm=norm,levels=clevs,transform=ccrs.PlateCarree())
        
        #Storm-centered plot domain
        if domain == "dynamic":
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Determine number of lat/lon lines to use for parallels & meridians
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])

        #--------------------------------------------------------------------------------------
                
        #Add left title
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]
            
        subtrop = classify_subtropical(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
        
        dot = u"\u2022"
        vartitle = get_recon_title(varname)
        self.ax.set_title(f"{storm_type} {storm_data['name']}\n" + 'Recon: '+' '.join(vartitle),loc='left',fontsize=17,fontweight='bold')

        #Add right title
        #max_ppf = max(PPF)
        start_date = dt.strftime(min(Maps['time']),'%H:%M UTC %d %b %Y')
        end_date = dt.strftime(max(Maps['time']),'%H:%M UTC %d %b %Y')
        self.ax.set_title(f'Start ... {start_date}\nEnd ... {end_date}',loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        
        #Phantom legend
        handles=[]
        for _ in range(10):
            handles.append(mlines.Line2D([], [], linestyle='-',label='',lw=0))
        l = self.ax.legend(handles=handles,loc='upper left',fancybox=True,framealpha=0,fontsize=11.5)
        plt.draw()

        #Get the bbox
        try:
            bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
        except:
            bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())
        bb_ax = self.ax.get_position()

        #Define colorbar axis
        cax = self.fig.add_axes([bb.x0+bb.width, bb.y0-.05*bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical',\
                                 ticks=clevs)
                
        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
    
        rect_offset = 0.0
        if prop['cmap']=='category' and varname=='sfmr':
            cax.yaxis.set_ticks(np.linspace(0,1,len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0,1,len(clevs))[:-1]+np.linspace(0,1,len(clevs))[1:])*.5)
            cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            
            rect_offset = 0.7
            
        rectangle = mpatches.Rectangle((bb.x0,bb.y0-0.1*bb.height),(1.8+rect_offset)*bb.width,1.1*bb.height,\
                                       fc = 'w',edgecolor = '0.8',alpha = 0.8,\
                                       transform=self.fig.transFigure, zorder=2)
        self.ax.add_patch(rectangle)
        
 
        #Add plot credit
        text = self.plot_credit()
        self.add_credit(text)


    
    def plot_polar(self,dfRecon,track_dict,time=None,reconInterp=None,radlim=150,ax=None,return_ax=False,prop={}):

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
        
        #mlib.rcParams.update({'font.size': 16})

        fig = plt.figure(figsize=prop['figsize'])
        if ax is None:
            self.ax = plt.subplot()
        else:
            self.ax = ax
        
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['clevs'])
        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(Maps_dict['grid_x'],Maps_dict['grid_y'],Maps_dict['maps'],\
                                 cmap=cmap,norm=norm,levels=clevs,transform=ccrs.PlateCarree())
        
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

        #Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        #--------------------------------------------------------------------------------------
           
        #Return axis if specified, otherwise display figure
        if ax is not None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()
     
    def plot_maps(self,storm,Maps_dict,varname,recon_stats=None,\
                  domain='dynamic',ax=None,return_ax=False,return_domain=False,prop={},map_prop={}):
        
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
        default_prop={'cmap':'category','levels':None,'left_title':'','right_title':'','pcolor':True}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(12.5,8.5),'dpi':120}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)

        MULTIVAR=False
        if isinstance(varname,(tuple,list)):
            varname2 = varname[1]
            varname = varname[0]
            Maps_dict2 = Maps_dict[1]
            Maps_dict = Maps_dict[0]
            MULTIVAR=True
        
        grid_res = 1*1e3 #m
        clon = Maps_dict['center_lon']
        clat = Maps_dict['center_lat']
        distproj = ccrs.LambertConformal()
        out = distproj.transform_points(ccrs.PlateCarree(),np.array([clon]),np.array([clat]))        
        cx = np.rint(out[:,0]/grid_res)*grid_res
        cy = np.rint(out[:,1]/grid_res)*grid_res
        xmgrid = cx+Maps_dict['grid_x']*grid_res
        ymgrid = cy+Maps_dict['grid_y']*grid_res
        out = self.proj.transform_points(distproj,xmgrid,ymgrid)
        lons = out[:,:,0]
        lats = out[:,:,1]
        
        #mlib.rcParams.update({'font.size': 16})
        
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['levels'])

        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(lons,lats,Maps_dict['maps'],\
                                 cmap=cmap,norm=norm,levels=clevs,transform=ccrs.PlateCarree())
    
        if MULTIVAR:
            CS = self.ax.contour(lons,lats,Maps_dict2['maps'],levels = np.arange(0,2000,4),colors='k',linewidths=0.5)
            # Recast levels to new class
            CS.levels = [int(val) for val in CS.levels]
            self.ax.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)
        
        
        #Storm-centered plot domain
        if domain == "dynamic":
            
            bound_w,bound_e,bound_s,bound_n = np.amin(lons)-.1,np.amax(lons)+.1,np.amin(lats)-.1,np.amax(lats)+.1
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)

        #Determine number of lat/lon lines to use for parallels & meridians
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
#        rightarrow = u"\u2192"
#        plt.xlabel(f'W {rightarrow} E Distance (km)')
#        plt.ylabel(f'S {rightarrow} N Distance (km)')
#        plt.axis([-radlim,radlim,-radlim,radlim])
#        plt.axis('equal')
        
        cbar = self.fig.colorbar(cbmap,orientation='vertical',\
                                 ticks=clevs)
#        cbar.set_label('wind (kt)')
                
        #--------------------------------------------------------------------------------------

        storm_data = storm.dict
        #Add left title
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]
            
        subtrop = classify_subtropical(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
        
        vartitle = get_recon_title(varname)
        title_left = f"{storm_type} {storm_data['name']}\n" + 'Recon: '+' '.join(vartitle)
        self.ax.set_title(title_left,loc='left',fontsize=17,fontweight='bold')

        #Add right title
        self.ax.set_title(Maps_dict['time'].strftime('%H:%M UTC %d %b %Y'),loc='right',fontsize=13)

        #Add stats
        if recon_stats is not None:
            a = self.ax.text(0.8,0.97,f"Max FL Wind: {int(recon_stats['pkwnd_max'])} kt\n"+\
                                       f"Max SFMR: {int(recon_stats['sfmr_max'])} kt\n"+\
                                       f"Min SLP: {int(recon_stats['p_min'])} hPa",fontsize=9.5,color='k',\
                                       bbox=dict(facecolor='0.9', edgecolor='black', boxstyle='round,pad=1'),\
                                       transform=self.ax.transAxes,ha='left',va='top',zorder=10)

        #Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        if return_ax:
            if return_domain:
                return self.ax,{'n':bound_n,'e':bound_e,'s':bound_s,'w':bound_w}
            else:
                return self.ax
