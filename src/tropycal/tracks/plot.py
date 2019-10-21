import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta
import scipy.ndimage as ndimage
import networkx as nx

from ..plot import Plot
from .tools import *

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

class TrackPlot(Plot):
    
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
            "lonW/lonE/latS/latN" - Custom plot domain
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
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
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
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]
            
        subtrop = classify_subtrop(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_type(np.nanmax(tropical_vmax),subtrop,peak_basin)
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
        
        #Add plot credit
        if storm_data['source'] == 'ibtracs' and storm_data['source_info'] == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"
        
        self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
        transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
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
        
    def plot_storm_nhc(self,forecast,track=None,track_labels='fhr',cone_days=5,zoom="dynamic_forecast",ax=None,return_ax=False,prop={},map_prop={}):
        
        r"""
        Creates a plot of the operational NHC forecast track along with observed track data.
        
        Parameters
        ----------
        forecast : dict
            Dict entry containing forecast data.
        track : dict
            Dict entry containing observed track data. Default is none.
        track_labels : str
            Label forecast hours with the following methods:
            '' = no label
            'fhr' = forecast hour
            'valid_utc' = UTC valid time
            'valid_edt' = EDT valid time
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        zoom : str
            Zoom for the plot. Can be one of the following:
            "dynamic_forecast" - default. Dynamically focuses the domain on the forecast track.
            "dynamic" - Dynamically focuses the domain on the combined observed and forecast track.
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
        default_prop={'dots':True,'fillcolor':'category','linecolor':'k','category_colors':'default','linewidth':1.0,'ms':7.5}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
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
            cone = self.generate_nhc_cone(forecast,dateline,cone_days)

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
            cone_2d = cone['cone']
            cone_2d = ndimage.gaussian_filter(cone_2d,sigma=0.5,order=0)
            self.ax.contourf(cone_lon_2d,cone_lat_2d,cone_2d,[0.9,1.1],colors=['#ffffff','#ffffff'],alpha=0.6,zorder=2,transform=ccrs.PlateCarree())
            self.ax.contour(cone_lon_2d,cone_lat_2d,cone_2d,[0.9],linewidths=1.0,colors=['k'],zorder=3,transform=ccrs.PlateCarree())

            #Plot center line & account for dateline crossing
            center_lon = cone['center_lon']
            center_lat = cone['center_lat']
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(center_lon)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                center_lon = new_lons.tolist()
            self.ax.plot(center_lon,center_lat,color='k',linewidth=2.0,zorder=4,transform=ccrs.PlateCarree())

            #Retrieve forecast dots
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

            #Plot forecast dots
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

            #Label forecast dots
            if track_labels in ['fhr','valid_utc','valid_edt']:
                valid_dates = [forecast['init']+timedelta(hours=int(i)) for i in iter_hr]
                if track_labels == 'fhr':
                    labels = [str(i) for i in iter_hr]
                if track_labels == 'valid_edt':
                    labels = [str(int(i.strftime('%I'))) + ' ' + i.strftime('%p %a') for i in [j-timedelta(hours=4) for j in valid_dates]]
                    edt_warning = True
                if track_labels == 'valid_utc':
                    labels = [f"{i.strftime('%H UTC')}\n{str(i.month)}/{str(i.day)}" for i in valid_dates]
                self.plot_nhc_labels(self.ax, fcst_lon, fcst_lat, labels, k=1.2)
                
            #Add cone coordinates to coordinate extrema
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
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #--------------------------------------------------------------------------------------
        
        #Identify storm type (subtropical, hurricane, etc)
        first_fcst_wind = np.array(forecast['vmax'])[fcst_hr >= start_slice][0]
        first_fcst_mslp = np.array(forecast['mslp'])[fcst_hr >= start_slice][0]
        first_fcst_type = np.array(forecast['type'])[fcst_hr >= start_slice][0]
        if all_nan(first_fcst_wind) == True:
            storm_type = 'Unknown'
        else:
            #subtrop = classify_subtrop(np.array(storm_data['type']))
            subtrop = True if first_fcst_type in ['SD','SS'] else False
            cur_wind = first_fcst_wind + 0
            storm_type = get_storm_type(np.nan_to_num(cur_wind),subtrop,'north_atlantic')
        
        #Identify storm name (and storm type, if post-tropical or potential TC)
        matching_times = [i for i in storm_data['date'] if i <= forecast['init']]
        if check_length < 2:
            if all_nan(first_fcst_wind) == True:
                storm_name = storm_data['name']
            else:
                storm_name = num_to_str(int(storm_data['id'][2:4])).upper()
                if first_fcst_wind >= 34 and first_fcst_type in ['TD','SD','SS','TS','HU']: storm_name = storm_data['name'];
                if first_fcst_type not in ['TD','SD','SS','TS','HU']: storm_type = 'Potential Tropical Cyclone'
        else:
            storm_name = num_to_str(int(storm_data['id'][2:4])).upper()
            storm_type = 'Potential Tropical Cyclone'
            storm_tropical = False
            if all_nan(vmax) == True:
                storm_type = 'Unknown'
                storm_name = storm_data['name']
            else:
                for i,(iwnd,ityp) in enumerate(zip(vmax,styp)):
                    if ityp in ['SD','SS','TD','TS','HU']:
                        storm_tropical = True
                        #subtrop = classify_subtrop(np.array(storm_data['type']))
                        subtrop = True if ityp in ['SD','SS'] else False
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

        #Add forecast label warning
        try:
            if edt_warning == True:
                warning_text = "All times displayed are in EDT\n\n"
            else:
                warning_text = ""
        except:
            warning_text = ""
        try:
            warning_text += f"The cone of uncertainty in this product was generated internally using {cone['year']} official\nNHC cone radii. This cone differs slightly from the official NHC cone.\n\n"
        except:
            pass
        
        self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
                transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
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
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
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
        bound_w,bound_e,bound_s,bound_n = self.set_projection(season.basin)
            
        #Determine number of lat/lon lines to use for parallels & meridians
        self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        
        #Add storm labels
        if season.basin != 'all':
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
        emdash = u"\u2014"
        basin_name = ((season.basin).replace("_"," ")).title()
        if season.basin == 'all':
            season_title = f"{season.year} Global Tropical Cyclone Season"
        elif season.basin in ['south_indian','south_atlantic','australia','south_pacific']:
            season_title = f"{season.year-1}{emdash}{season.year} {basin_name} Tropical Cyclone Season"
        elif season.basin in ['west_pacific']:
            season_title = f"{season.year} {basin_name.split(' ')[1]} Typhoon Season"
        else:
            season_title = f"{season.year} {basin_name.split(' ')[1]} Hurricane Season"
        self.ax.set_title(season_title,loc='left',fontsize=17,fontweight='bold')

        #Add right title
        endash = u"\u2013"
        dot = u"\u2022"
        self.ax.set_title(f"{sinfo['season_named']} named {dot} {sinfo['season_hurricane']} hurricanes {dot} {sinfo['season_major']} major\n{sinfo['season_ace']:.1f} Cumulative ACE",loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add plot credit
        if season.source == 'ibtracs' and season.source_info == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"
        
        self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
        transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
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
        
    def generate_nhc_cone(self,forecast,dateline,cone_days=5):
        
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
        #Radii are in nautical miles
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
            dist = dist * 0.868976 #to nautical miles

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
    
    def plot_nhc_labels(self, ax, x, y, labels, k=0.01):

        G = nx.DiGraph()
        data_nodes = []
        init_pos = {}
        for xi, yi, label in zip(x, y, labels):
            data_str = 'data_{0}'.format(label)
            G.add_node(data_str)
            G.add_node(label)
            G.add_edge(label, data_str)
            data_nodes.append(data_str)
            init_pos[data_str] = (xi, yi)
            init_pos[label] = (xi, yi)

        pos = nx.spring_layout(G, pos=init_pos, fixed=data_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in data_nodes])
        pos_before = np.vstack([init_pos[d] for d in data_nodes])
        scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
        scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val*scale) + shift

        start = False
        for label, data_str in G.edges():
            if start == False:
                start = True
                continue
            self.ax.annotate(label,
                        xy=pos[data_str], xycoords='data',
                        xytext=pos[label], textcoords='data', fontweight='bold', ha='center', va='center',
                        arrowprops=dict(arrowstyle="-",#->
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3", 
                                        color='k'),
                        transform=ccrs.PlateCarree())

    def plot_gridded(self,xcoord,ycoord,zcoord,VEC_FLAG=False,zoom="north_atlantic",ax=None,return_ax=False,prop={},map_prop={}):
        
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
            "lonW/lonE/latS/latN" - Custom plot domain
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
        default_prop={'cmap':'category','clevs':[np.nanmin(zcoord),np.nanmax(zcoord)],'left_title':'','right_title':'All storms'}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
        #error check
        if isinstance(zoom,str) == False:
            raise TypeError('Error: zoom must be of type str')
        
        #Pre-generated zooms
        if zoom in ['north_atlantic','east_pacific','west_pacific','south_pacific','south_indian','north_indian','australia','all']:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(zoom)
            
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
        
        _,varname = findvar(prop['title_L'],{})
        cmap,clevs = make_cmap(varname,prop['cmap'],prop['clevs'])
        
        if len(xcoord.shape) and len(ycoord.shape)==1:
            xcoord,ycoord = np.meshgrid(xcoord,ycoord)
        
        if VEC_FLAG:
            mag = np.hypot(*zcoord)
            clevs = [np.nanmin(mag),np.nanmax(mag)]
            cbmap = self.ax.pcolor(xcoord,ycoord,mag,cmap=cmap,vmin=min(clevs),vmax=max(clevs),
                               transform=ccrs.PlateCarree())            
            zcoord = zcoord/mag*abs(xcoord[0,0]-xcoord[0,1])
            x_center = (xcoord[:-1,:-1]+xcoord[1:,1:])*.5
            y_center = (ycoord[:-1,:-1]+ycoord[1:,1:])*.5
            u = zcoord[0][:-1,:-1]
            v = zcoord[1][:-1,:-1]
            self.ax.quiver(x_center,y_center,u,v,color='w',alpha=0.6,transform=ccrs.PlateCarree(),\
                           pivot='mid',width=.001,headwidth=3.5,headlength=4.5,headaxislength=4)
            zcoord = mag
        
        else:
            cbmap = self.ax.pcolor(xcoord,ycoord,zcoord,cmap=cmap,vmin=min(clevs),vmax=max(clevs),
                               transform=ccrs.PlateCarree())
        
        #--------------------------------------------------------------------------------------
        
        #Phantom legend
        handles=[]
        for _ in range(10):
            handles.append(mlines.Line2D([], [], linestyle='-',label='',lw=0))
        l = self.ax.legend(handles=handles,loc='upper left',fancybox=True,framealpha=0,fontsize=11.5)
        plt.draw()

        #Get the bbox
        bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
        bb_ax = self.ax.get_position()

        #Define colorbar axis
        cax = self.fig.add_axes([bb.x0+bb.width, bb.y0-.05*bb.height, 0.015, bb.height])
        cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical')
        if len(clevs)>2:
            cbar.ax.yaxis.set_ticks(clevs)
        cbar.ax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
    
        rect_offset = 0.0
        if prop['cmap']=='category' and varname=='vmax':
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks([(i-5)/(201-5) for i in np.mean([clevs[:-1],clevs[1:]],axis=0)])
            cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            
            rect_offset = 0.7
            
        rectangle = mpatches.Rectangle((bb.x0,bb.y0-0.1*bb.height),(1.8+rect_offset)*bb.width,1.1*bb.height,\
                                       fc = 'w',edgecolor = '0.8',alpha = 0.8,\
                                       transform=self.fig.transFigure, zorder=2)
        self.ax.add_patch(rectangle)
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        try:
            self.ax.set_title(prop['title_L'],loc='left',fontsize=17,fontweight='bold')
        except:
            pass
        
        #Add right title
        try:
            self.ax.set_title(prop['title_R'],loc='right',fontsize=15)
        except:
            pass
        
        #--------------------------------------------------------------------------------------
        
        text = self.plot_credit()
        self.add_credit(text)
        
        #Return axis if specified, otherwise display figure
        if ax != None or return_ax == True:
            return self.ax
        else:
            plt.show()
            plt.close()


