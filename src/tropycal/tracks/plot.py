import os, sys
import calendar
import numpy as np
import scipy.interpolate as interp
import warnings
from datetime import datetime as dt,timedelta
import scipy.ndimage as ndimage
import networkx as nx
from scipy.ndimage import gaussian_filter as gfilt

#Import internal scripts
from ..plot import Plot

#Import tools
from .tools import *
from ..utils import *
from .. import constants

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
    
    def __init__(self):
        
        self.use_credit = True
    
    def plot_storms(self,storms,domain="dynamic",title="TC Track Composite",plot_all_dots=False,track_labels=False,ax=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single or multiple storm tracks.
        
        Parameters
        ----------
        storms : list
            List of requested storms. List can contain either strings of storm ID (e.g., "AL052019"), tuples with storm name and year (e.g., ("Matthew",2016)), or dict entries.
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic" - default. Dynamically focuses the domain using the storm track(s) plotted.
            "north_atlantic" - North Atlantic Ocean basin
            "pacific" - East/Central Pacific Ocean basin
            "lonW/lonE/latS/latN" - Custom plot domain
        plot_all_dots : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':True,'fillcolor':'category','cmap':None,'levels':None,'linecolor':'k','linewidth':1.0,'ms':7.5,'plot_names':False}
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
        
        #Iterate through all storms provided
        for storm in storms:

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
            
            #Force dynamic_tropical to tropical if an invest
            invest_bool = False
            if 'invest' in storm_data.keys() and storm_data['invest'] == True:
                invest_bool = True
                if domain == 'dynamic_tropical': domain = 'dynamic'

            #Add to coordinate extrema
            if domain == 'dynamic_tropical':
                type_array = np.array(storm_data['type'])
                idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
                use_lats = (np.array(storm_data['lat'])[idx]).tolist()
                use_lons = (np.array(lons)[idx]).tolist()
            else:
                use_lats = storm_data['lat']
                use_lons = np.copy(lons).tolist()

            #Add to coordinate extrema
            if max_lat is None:
                max_lat = max(use_lats)
            else:
                if max(use_lats) > max_lat: max_lat = max(use_lats)
            if min_lat is None:
                min_lat = min(use_lats)
            else:
                if min(use_lats) < min_lat: min_lat = min(use_lats)
            if max_lon is None:
                max_lon = max(use_lons)
            else:
                if max(use_lons) > max_lon: max_lon = max(use_lons)
            if min_lon is None:
                min_lon = min(use_lons)
            else:
                if min(use_lons) < min_lon: min_lon = min(use_lons)

            #Add storm label at start and end points
            if prop['plot_names'] == True:
                self.ax.text(lons[0]+0.0,storm_data['lat'][0]+1.0,f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center',transform=ccrs.PlateCarree())
                self.ax.text(lons[-1]+0.0,storm_data['lat'][-1]+1.0,f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center',transform=ccrs.PlateCarree())
            
            #Iterate over storm data to plot
            levels = None
            cmap = None
            for i,(i_lat,i_lon,i_vmax,i_mslp,i_date,i_type) in enumerate(zip(storm_data['lat'],lons,storm_data['vmax'],storm_data['mslp'],storm_data['date'],storm_data['type'])):

                #Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))

                #Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'],str) == True and prop['linecolor'] in ['vmax','mslp','dvmax_dt','speed']:
                    segmented_colors = True
                    try:
                        color_variable = storm_data[prop['linecolor']]
                    except:
                        raise ValueError("Storm object must be interpolated to hourly using 'storm.interp().plot(...)' in order to use 'dvmax_dt' or 'speed' for fill color")
                    if prop['levels'] is None: #Auto-determine color levels if needed
                        prop['levels'] = [np.nanmin(color_variable),np.nanmax(color_variable)]
                    cmap,levels = get_cmap_levels(prop['linecolor'],prop['cmap'],prop['levels'])
                    line_color = cmap((color_variable-min(levels))/(max(levels)-min(levels)))[i]

                #Otherwise go with user input as is
                else:
                    segmented_colors = False
                    line_color = prop['linecolor']

                #For tropical/subtropical types, color-code if requested
                if i > 0:
                    if i_type in constants.TROPICAL_STORM_TYPES and storm_data['type'][i-1] in constants.TROPICAL_STORM_TYPES:

                        #Plot underlying black and overlying colored line
                        self.ax.plot([lons[i-1],lons[i]],[storm_data['lat'][i-1],storm_data['lat'][i]],'-',
                                      linewidth=prop['linewidth']*1.33,color='k',zorder=3,
                                      transform=ccrs.PlateCarree())
                        self.ax.plot([lons[i-1],lons[i]],[storm_data['lat'][i-1],storm_data['lat'][i]],'-',
                                      linewidth=prop['linewidth'],color=line_color,zorder=4,
                                      transform=ccrs.PlateCarree())

                    #For non-tropical types, plot dotted lines
                    else:

                        #Restrict line width to 1.5 max
                        line_width = prop['linewidth'] + 0.0
                        if line_width > 1.5: line_width = 1.5

                        #Plot dotted line
                        self.ax.plot([lons[i-1],lons[i]],[storm_data['lat'][i-1],storm_data['lat'][i]],':',
                                      linewidth=line_width,color=line_color,zorder=4,
                                      transform=ccrs.PlateCarree(),
                                      path_effects=[path_effects.Stroke(linewidth=line_width*1.33, foreground='k'),
                                                    path_effects.Normal()])

                #Plot dots if requested
                if prop['dots'] == True:
                    if plot_all_dots == False and i_date.strftime('%H%M') not in constants.STANDARD_HOURS: continue
                    segmented_colors = self.plot_dot(i_lon,i_lat,i_date,i_vmax,i_type,
                                                     zorder=5,storm_data=storm_data,prop=prop)
                
                #Label track dots
                if track_labels in ['valid_utc']:
                    if track_labels == 'valid_utc':
                        strformat = '%H UTC \n%-m/%-d'
                        labels = {t.strftime(strformat):(x,y) for t,x,y in zip(sdate,lons,lats) if t.hour==0}
                        track = {t.strftime(strformat):(x,y) for t,x,y in zip(sdate,lons,lats)}
                    self.plot_track_labels(self.ax, labels, track, k=.9)

        #--------------------------------------------------------------------------------------
        
        #Storm-centered plot domain
        if domain == "dynamic" or domain == "dynamic_tropical":
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
        #--------------------------------------------------------------------------------------
        
        #Add left title
        if len(storms) > 1:
            if title != "": self.ax.set_title(f"{title}",loc='left',fontsize=17,fontweight='bold')
        else:
            #Add left title
            type_array = np.array(storm_data['type'])
            idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
            if invest_bool == False or len(idx[0]) > 0:
                tropical_vmax = np.array(storm_data['vmax'])[idx]

                #Coerce to include non-TC points if storm hasn't been designated yet
                add_ptc_flag = False
                if len(tropical_vmax) == 0:
                    add_ptc_flag = True
                    idx = np.where((type_array == 'LO') | (type_array == 'DB'))
                tropical_vmax = np.array(storm_data['vmax'])[idx]

                subtrop = classify_subtropical(np.array(storm_data['type']))
                peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
                peak_basin = storm_data['wmo_basin'][peak_idx]
                storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
                if add_ptc_flag == True: storm_type = "Potential Tropical Cyclone"
                self.ax.set_title(f"{storm_type} {storm_data['name']}",loc='left',fontsize=17,fontweight='bold')
            else:
                #Use all indices for invests
                idx = np.array([True for i in type_array])
                add_ptc_flag = False
                tropical_vmax = np.array(storm_data['vmax'])

                #Determine letter in front of invest
                add_letter = 'L'
                if storm_data['id'][0] == 'C':
                    add_letter = 'C'
                elif storm_data['id'][0] == 'E':
                    add_letter = 'E'
                elif storm_data['id'][0] == 'W':
                    add_letter = 'W'
                elif storm_data['id'][0] == 'I':
                    add_letter = 'I'
                elif storm_data['id'][0] == 'S':
                    add_letter = 'S'

                #Add title
                self.ax.set_title(f"INVEST {storm_data['id'][2:4]}{add_letter}",loc='left',fontsize=17,fontweight='bold')

            #Add right title
            ace = storm_data['ace']
            if add_ptc_flag == True: ace = 0.0
            type_array = np.array(storm_data['type'])
            
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
        warning_text=""
        if storm_data['source'] == 'ibtracs' and storm_data['source_info'] == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"

            self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
            transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
        #--------------------------------------------------------------------------------------
                
        #Add legend
        self.add_legend(prop,segmented_colors,levels,cmap,storm_data)
                
        #-----------------------------------------------------------------------------------------
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(save_path,bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax
        
    def plot_storm_nhc(self,forecast,track=None,track_labels='fhr',cone_days=5,domain="dynamic_forecast",ax=None,save_path=
None,prop={},map_prop={}):
        
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
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic_forecast" - default. Dynamically focuses the domain on the forecast track.
            "dynamic" - Dynamically focuses the domain on the combined observed and forecast track.
            "lonW/lonE/latS/latN" - Custom plot domain
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':True,'fillcolor':'category','linecolor':'k','linewidth':1.0,'ms':7.5,'cone_lw':1.0,'cone_alpha':0.6}
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
                if fcst_type == "": fcst_type = get_storm_type(fcst_vmax,False)
                if self.proj.proj4_params['lon_0'] == 180.0:
                    if fcst_lon < 0: fcst_lon = fcst_lon + 360.0
                lons.append(fcst_lon)
                lats.append(fcst_lat)
                vmax.append(fcst_vmax)
                styp.append(fcst_type)
                sdate.append(sdate[-1]+timedelta(hours=start_slice))

                #Add to coordinate extrema
                if domain != "dynamic_forecast":
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
                        if type6[i] not in constants.TROPICAL_STORM_TYPES: ltype = 'dotted'
                        self.ax.plot([lons[i-1],lons[i]],[lats[i-1],lats[i]],
                                      '-',color=get_colors_sshws(np.nan_to_num(vmax[i])),linewidth=prop['linewidth'],linestyle=ltype,
                                      transform=ccrs.PlateCarree(),
                                      path_effects=[path_effects.Stroke(linewidth=prop['linewidth']*1.25, foreground='k'), path_effects.Normal()])
                else:
                    self.ax.plot(lons,lats,'-',color=prop['linecolor'],linewidth=prop['linewidth'],transform=ccrs.PlateCarree())

                #Plot storm dots as specified
                if prop['dots'] == True:
                    for i,(ilon,ilat,iwnd,itype) in enumerate(zip(lons,lats,vmax,styp)):
                        mtype = '^'
                        if itype in constants.SUBTROPICAL_ONLY_STORM_TYPES:
                            mtype = 's'
                        elif itype in constants.TROPICAL_ONLY_STORM_TYPES:
                            mtype = 'o'
                        if prop['fillcolor'] == 'category':
                            ncol = get_colors_sshws(np.nan_to_num(iwnd))
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
            cone = generate_nhc_cone(forecast,forecast['basin'],dateline,cone_days)

            #Contour fill cone & account for dateline crossing
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                pass
            else:
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
                self.ax.contourf(cone_lon_2d,cone_lat_2d,cone_2d,[0.9,1.1],colors=['#ffffff','#ffffff'],alpha=prop['cone_alpha'],zorder=2,transform=ccrs.PlateCarree())
                self.ax.contour(cone_lon_2d,cone_lat_2d,cone_2d,[0.9],linewidths=prop['cone_lw'],colors=['k'],zorder=3,transform=ccrs.PlateCarree())

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
                if itype in constants.SUBTROPICAL_ONLY_STORM_TYPES:
                    mtype = 's'
                elif itype in ['TD','TS','HU','']:
                    mtype = 'o'
                if prop['fillcolor'] == 'category':
                    ncol = get_colors_sshws(np.nan_to_num(iwnd))
                else:
                    ncol = 'k'
                #Marker width
                mew = 0.5; use_zorder=5
                if i == 0:
                    mew = 2.0; use_zorder=10
                self.ax.plot(ilon,ilat,mtype,color=ncol,mec='k',mew=mew,ms=prop['ms']*1.3,transform=ccrs.PlateCarree(),zorder=use_zorder)

            #Label forecast dots
            if track_labels in ['fhr','valid_utc','valid_edt','fhr_wind_kt','fhr_wind_mph']:
                valid_dates = [forecast['init']+timedelta(hours=int(i)) for i in iter_hr]
                if track_labels == 'fhr':
                    labels = [str(i) for i in iter_hr]
                if track_labels == 'fhr_wind_kt':
                    labels = [f"Hour {iter_hr[i]}\n{fcst_vmax[i]} kt" for i in range(len(iter_hr))]
                if track_labels == 'fhr_wind_mph':
                    labels = [f"Hour {iter_hr[i]}\n{knots_to_mph(fcst_vmax[i])} mph" for i in range(len(iter_hr))]
                if track_labels == 'valid_edt':
                    labels = [str(int(i.strftime('%I'))) + ' ' + i.strftime('%p %a') for i in [j-timedelta(hours=4) for j in valid_dates]]
                    edt_warning = True
                if track_labels == 'valid_utc':
                    labels = [f"{i.strftime('%H UTC')}\n{str(i.month)}/{str(i.day)}" for i in valid_dates]
                self.plot_nhc_labels(self.ax, fcst_lon, fcst_lat, labels, k=1.2)
                
            #Add cone coordinates to coordinate extrema
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                if domain == "dynamic_forecast" or max_lat is None:
                    max_lat = max(center_lat)
                    min_lat = min(center_lat)
                    max_lon = max(center_lon)
                    min_lon = min(center_lon)
                else:
                    if max(center_lat) > max_lat: max_lat = max(center_lat)
                    if min(center_lat) < min_lat: min_lat = min(center_lat)
                    if max(center_lon) > max_lon: max_lon = max(center_lon)
                    if min(center_lon) < min_lon: min_lon = min(center_lon)
            else:
                if domain == "dynamic_forecast" or max_lat is None:
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
        if domain == "dynamic" or domain == 'dynamic_forecast':
            
            bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
            self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())
            
        #Pre-generated or custom domain
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
        #--------------------------------------------------------------------------------------
        
        #Identify storm type (subtropical, hurricane, etc)
        first_fcst_wind = np.array(forecast['vmax'])[fcst_hr >= start_slice][0]
        first_fcst_mslp = np.array(forecast['mslp'])[fcst_hr >= start_slice][0]
        first_fcst_type = np.array(forecast['type'])[fcst_hr >= start_slice][0]
        if all_nan(first_fcst_wind) == True:
            storm_type = 'Unknown'
        else:
            subtrop = first_fcst_type in constants.SUBTROPICAL_ONLY_STORM_TYPES
            cur_wind = first_fcst_wind + 0
            storm_type = get_storm_classification(np.nan_to_num(cur_wind),subtrop,'north_atlantic')
        
        #Identify storm name (and storm type, if post-tropical or potential TC)
        matching_times = [i for i in storm_data['date'] if i <= forecast['init']]
        if check_length < 2:
            if all_nan(first_fcst_wind) == True:
                storm_name = storm_data['name']
            else:
                storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
                if first_fcst_wind >= 34 and first_fcst_type in constants.TROPICAL_STORM_TYPES: storm_name = storm_data['name'];
                if first_fcst_type not in constants.TROPICAL_STORM_TYPES: storm_type = 'Potential Tropical Cyclone'
        else:
            storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
            storm_type = 'Potential Tropical Cyclone'
            storm_tropical = False
            if all_nan(vmax) == True:
                storm_type = 'Unknown'
                storm_name = storm_data['name']
            else:
                for i,(iwnd,ityp) in enumerate(zip(vmax,styp)):
                    if ityp in constants.TROPICAL_STORM_TYPES:
                        storm_tropical = True
                        subtrop = ityp in constants.SUBTROPICAL_ONLY_STORM_TYPES
                        storm_type = get_storm_classification(np.nan_to_num(iwnd),subtrop,'north_atlantic')
                        if np.isnan(iwnd) == True: storm_type = 'Unknown'
                    else:
                        if storm_tropical == True: storm_type = 'Post Tropical Cyclone'
                    if ityp in constants.NAMED_TROPICAL_STORM_TYPES:
                        storm_name = storm_data['name']
        
        #Fix storm types for non-NHC basins
        if 'cone' in forecast.keys():
            storm_type = get_storm_classification(first_fcst_wind,False,forecast['basin'])
        
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
        
        if forecast_id == -1:
            title_text = f"Current Intensity: {knots_to_mph(first_fcst_wind)} mph {dot} {first_fcst_mslp} hPa"
            if 'cone' in forecast.keys() and forecast['cone'] == False:
                title_text += f"\nJTWC Issued: {forecast_date}"
            else:
                title_text += f"\nNHC Issued: {forecast_date}"
        else:
            if first_fcst_wind != "N/A": first_fcst_wind = knots_to_mph(first_fcst_wind)
            title_text = f"{first_fcst_wind} mph {dot} {first_fcst_mslp} hPa {dot} Forecast #{forecast_id}"
            title_text += f"\nForecast Issued: {forecast_date}"
        
        
        #Add right title
        self.ax.set_title(title_text,loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':
            
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            uk = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Unknown', marker='o', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Depression', marker='o', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical Storm', marker='o', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 1', marker='o', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 2', marker='o', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 3', marker='o', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 4', marker='o', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Category 5', marker='o', color=get_colors_sshws(137))
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
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(os.path.join(save_path,f"{storm_data['name']}_{storm_data['year']}_track.png"),bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax

    def plot_ensembles(self,forecast,storm_dict,fhr,prop_ensemble_members,prop_ensemble_mean,prop_gfs,prop_ellipse,prop_density,nens,
                       domain,ds,ax,map_prop,save_path):
        
        r"""
        
        """
        
        #Set default properties
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        default_prop_ensemble_members = {'linewidth':0.5, 'linecolor':'k'}
        default_prop_ensemble_mean = {'linewidth':2.0, 'linecolor':'k'}
        default_prop_gfs = {'linewidth':2.0, 'linecolor':'b'}
        default_prop_ellipse = {'linewidth':2.0, 'linecolor':'r'}
        default_prop_density = {'radius':200, 'cmap':plt.cm.YlOrRd, 'levels':[i for i in range(5,105,5)]}
        
        #Initialize plot
        map_prop = self.add_prop(map_prop,default_map_prop)
        if prop_ensemble_members is not None: prop_ensemble_members = self.add_prop(prop_ensemble_members,default_prop_ensemble_members)
        if prop_ensemble_mean is not None: prop_ensemble_mean = self.add_prop(prop_ensemble_mean,default_prop_ensemble_mean)
        if prop_gfs is not None: prop_gfs = self.add_prop(prop_gfs,default_prop_gfs)
        if prop_ellipse is not None: prop_ellipse = self.add_prop(prop_ellipse,default_prop_ellipse)
        if prop_density is not None: prop_density = self.add_prop(prop_density,default_prop_density)
        self.plot_init(ax,map_prop)
        
        #================================================================================================
        
        #Iterate over all forecast hours
        for hr in fhr:

            #Keep record of lat/lon coordinate extrema
            max_lat = None
            min_lat = None
            max_lon = None
            min_lon = None

            #================================================================================================

            #Plot density
            if prop_density is not None and hr in ds['gefs']['fhr']:

                #Create 0.5 degree grid for plotting
                gridlats = np.arange(0,90,0.25)
                gridlons = np.arange(180-360.0,180.0,0.25) #gridlons = np.arange(180-360.0,360-360.0,0.25)
                gridlons2d,gridlats2d = np.meshgrid(gridlons,gridlats)
                griddata = np.zeros((gridlons2d.shape))

                #Iterate over all ensemble members
                for ens in range(nens):

                    #Proceed if hour is available
                    if hr in ds[f'gefs_{ens}']['fhr']:
                        idx = ds[f'gefs_{ens}']['fhr'].index(hr)
                        griddata += add_radius(gridlats2d, gridlons2d, ds[f'gefs_{ens}']['lat'][idx],
                                               ds[f'gefs_{ens}']['lon'][idx], prop_density['radius']) #350-km radius

                #Convert density to percent
                idx = ds[f'gefs']['fhr'].index(hr)
                #density_percent = (griddata / ds['eps']['members'][idx]) * 100.0
                density_percent = (griddata / nens) * 100.0

                #Plot density
                clevs = np.arange(5,105,5)
                cs = self.ax.contourf(gridlons, gridlats, density_percent, prop_density['levels'],
                                      cmap=prop_density['cmap'], alpha=0.6, transform=ccrs.PlateCarree())
                cbar = plt.colorbar(cs,ticks=np.arange(5,105,5))
                cbar.ax.tick_params(labelsize=12)

            #Plot ellipse
            if hr in ds['gefs']['fhr'] and prop_ellipse is not None:
                idx = ds['gefs']['fhr'].index(hr)

                try:
                    self.ax.plot(ds['gefs']['ellipse_lon'][idx],ds['gefs']['ellipse_lat'][idx],'-', color='w', linewidth=3.4,
                             transform=ccrs.PlateCarree(), alpha=0.8)
                    self.ax.plot(ds['gefs']['ellipse_lon'][idx],ds['gefs']['ellipse_lat'][idx],'-', color='b', linewidth=2.8,
                             transform=ccrs.PlateCarree(), alpha=0.8)
                except:
                    pass

            #Plot GEFS member tracks
            for i in range(nens):
                
                #Update coordinate bounds
                skip_bounds = False
                if hr in ds[f'gefs_{i}']['fhr']:
                    idx = ds[f'gefs_{i}']['fhr'].index(hr)
                    use_lats = ds[f'gefs_{i}']['lat'][:idx+1]
                    use_lons = ds[f'gefs_{i}']['lon'][:idx+1]
                else:
                    diff = [ihr-hr for ihr in ds[f'gefs_{i}']['fhr']]
                    idx = np.where(np.array(diff)>0)[0]
                    if len(idx) > 0:
                        use_lats = ds[f'gefs_{i}']['lat'][:idx[0]]
                        use_lons = ds[f'gefs_{i}']['lon'][:idx[0]]
                    else:
                        skip_bounds = True
                
                if skip_bounds == False:
                    if max_lat is None:
                        max_lat = np.nanmax(use_lats)
                    else:
                        if np.nanmax(use_lats) > max_lat: max_lat = np.nanmax(use_lats)
                    if min_lat is None:
                        min_lat = np.nanmin(use_lats)
                    else:
                        if np.nanmin(use_lats) < min_lat: min_lat = np.nanmin(use_lats)
                    if max_lon is None:
                        max_lon = np.nanmax(use_lons)
                    else:
                        if np.nanmax(use_lons) > max_lon: max_lon = np.nanmax(use_lons)
                    if min_lon is None:
                        min_lon = np.nanmin(use_lons)
                    else:
                        if np.nanmin(use_lons) < min_lon: min_lon = np.nanmin(use_lons)
                
                if hr in ds[f'gefs_{i}']['fhr']:
                    idx = ds[f'gefs_{i}']['fhr'].index(hr)
                    self.ax.plot(ds[f'gefs_{i}']['lon'][:idx+1], ds[f'gefs_{i}']['lat'][:idx+1], linewidth=0.2,
                             color='k', transform=ccrs.PlateCarree())
                    self.ax.plot(ds[f'gefs_{i}']['lon'][idx], ds[f'gefs_{i}']['lat'][idx], 'o', ms=4, mfc='k',mec='k',
                             alpha=0.6,transform=ccrs.PlateCarree())
                elif len(ds[f'gefs_{i}']['fhr']) > 0:
                    diff = [ihr-hr for ihr in ds[f'gefs_{i}']['fhr']]
                    idx = np.where(np.array(diff)>0)[0]
                    if len(idx) > 0:
                        self.ax.plot(ds[f'gefs_{i}']['lon'][:idx[0]], ds[f'gefs_{i}']['lat'][:idx[0]], linewidth=0.2,
                                 color='k', transform=ccrs.PlateCarree())

            #Plot operational GFS track
            if hr in ds['gfs']['fhr']:
                
                #Update coordinate bounds
                skip_bounds = False
                if hr in ds['gfs']['fhr']:
                    use_lats = ds['gfs']['lat'][:idx+1]
                    use_lons = ds['gfs']['lon'][:idx+1]
                else:
                    diff = [ihr-hr for ihr in ds['gfs']['fhr']]
                    idx = np.where(np.array(diff)>0)[0]
                    if len(idx) > 0:
                        use_lats = ds['gfs']['lat'][:idx[0]]
                        use_lons = ds['gfs']['lon'][:idx[0]]
                    else:
                        skip_bounds = True
                
                if skip_bounds == False:
                    if max_lat is None:
                        max_lat = np.nanmax(use_lats)
                    else:
                        if np.nanmax(use_lats) > max_lat: max_lat = np.nanmax(use_lats)
                    if min_lat is None:
                        min_lat = np.nanmin(use_lats)
                    else:
                        if np.nanmin(use_lats) < min_lat: min_lat = np.nanmin(use_lats)
                    if max_lon is None:
                        max_lon = np.nanmax(use_lons)
                    else:
                        if np.nanmax(use_lons) > max_lon: max_lon = np.nanmax(use_lons)
                    if min_lon is None:
                        min_lon = np.nanmin(use_lons)
                    else:
                        if np.nanmin(use_lons) < min_lon: min_lon = np.nanmin(use_lons)
                
                idx = ds['gfs']['fhr'].index(hr)
                self.ax.plot(ds['gfs']['lon'][:idx+1], ds['gfs']['lat'][:idx+1], linewidth=3.0, color='r', transform=ccrs.PlateCarree())
                self.ax.plot(ds['gfs']['lon'][idx], ds['gfs']['lat'][idx], 'o', ms=12, mfc='r',mec='k', transform=ccrs.PlateCarree())
            elif len(ds['gfs']['fhr']) > 0:
                diff = [ihr-hr for ihr in ds['gfs']['fhr']]
                idx = np.where(np.array(diff)>0)[0]
                if len(idx) > 0:
                    self.ax.plot(ds['gfs']['lon'][:idx[0]], ds['gfs']['lat'][:idx[0]], linewidth=3.0, color='r', transform=ccrs.PlateCarree())

            #Plot ensemble mean track
            if hr in ds['gefs']['fhr']:
                
                #Update coordinate bounds
                skip_bounds = False
                if hr in ds['gefs']['fhr']:
                    use_lats = ds['gefs']['lat'][:idx+1]
                    use_lons = ds['gefs']['lon'][:idx+1]
                else:
                    diff = [ihr-hr for ihr in ds['gefs']['fhr']]
                    idx = np.where(np.array(diff)>0)[0]
                    if len(idx) > 0:
                        use_lats = ds['gefs']['lat'][:idx[0]]
                        use_lons = ds['gefs']['lon'][:idx[0]]
                    else:
                        skip_bounds = True
                
                if skip_bounds == False:
                    if max_lat is None:
                        max_lat = np.nanmax(use_lats)
                    else:
                        if np.nanmax(use_lats) > max_lat: max_lat = np.nanmax(use_lats)
                    if min_lat is None:
                        min_lat = np.nanmin(use_lats)
                    else:
                        if np.nanmin(use_lats) < min_lat: min_lat = np.nanmin(use_lats)
                    if max_lon is None:
                        max_lon = np.nanmax(use_lons)
                    else:
                        if np.nanmax(use_lons) > max_lon: max_lon = np.nanmax(use_lons)
                    if min_lon is None:
                        min_lon = np.nanmin(use_lons)
                    else:
                        if np.nanmin(use_lons) < min_lon: min_lon = np.nanmin(use_lons)
                
                idx = ds['gefs']['fhr'].index(hr)
                self.ax.plot(ds['gefs']['lon'][:idx+1], ds['gefs']['lat'][:idx+1], linewidth=3.0, color='k', transform=ccrs.PlateCarree())
                self.ax.plot(ds['gefs']['lon'][idx], ds['gefs']['lat'][idx], 'o', ms=12, mfc='k',mec='k', transform=ccrs.PlateCarree())
            elif len(ds['gefs']['fhr']) > 0:
                diff = [ihr-hr for ihr in ds['gefs']['fhr']]
                idx = np.where(np.array(diff)>0)[0]
                if len(idx) > 0:
                    self.ax.plot(ds['gefs']['lon'][:idx[0]], ds['gefs']['lat'][:idx[0]], linewidth=3.0, color='k', transform=ccrs.PlateCarree())

            #================================================================================================

            #Add legend
            import matplotlib.patches as mpatches
            import matplotlib.lines as mlines
            p1 = mlines.Line2D([], [], color='r', linewidth=3.0, label='Deterministic GFS')
            p2 = mlines.Line2D([], [], color='k', linewidth=3.0, label='GEFS Mean')
            p3 = mlines.Line2D([], [], color='k', linewidth=0.5, label='GEFS Members')
            p4 = mlines.Line2D([], [], color='w', marker='o', ms=12, mec='b', mew=2.0, label='GEFS Ellipse')
            l = self.ax.legend(handles=[p1,p2,p3,p4],loc=1,prop={'size':14})
            l.set_zorder(200)

            #Plot title
            plot_title = f"GEFS Forecast Tracks for {storm_dict['name'].title()}"
            if prop_density is not None: plot_title += f"\nTrack Density ({np.int(prop_density['radius'])}-km radius)"
            self.ax.set_title(plot_title,fontsize=18,loc='left',fontweight='bold')

            title_str = f"Hour {hr} | Valid {(forecast+timedelta(hours=hr)).strftime('%H%M UTC %d %B %Y')}\n"
            title_str += f"Initialized {forecast.strftime('%H%M UTC %d %B %Y')}"
            self.ax.set_title(title_str,fontsize=14,loc='right')

            #--------------------------------------------------------------------------------------

            #Storm-centered plot domain
            if domain == "dynamic":

                bound_w,bound_e,bound_s,bound_n = self.dynamic_map_extent(min_lon,max_lon,min_lat,max_lat)
                self.ax.set_extent([bound_w,bound_e,bound_s,bound_n], crs=ccrs.PlateCarree())

            #Pre-generated or custom domain
            else:
                bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)

            #Plot parallels and meridians
            #This is currently not supported for all cartopy projections.
            try:
                self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
            except:
                pass

            #--------------------------------------------------------------------------------------

            credit_text = self.plot_credit()
            self.add_credit(credit_text)

            #Save image if specified
            if save_path is not None and isinstance(save_path,str) == True:
                plt.savefig(os.path.join(save_path),bbox_inches='tight')

            #Return axis if specified, otherwise display figure
            return self.ax
    
    def plot_season(self,season,domain=None,ax=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single season.
        
        Parameters
        ----------
        season : Season
            Instance of Season.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'dots':False,'fillcolor':'category','cmap':None,'levels':None,
                      'linecolor':'category','linewidth':1.0,'ms':7.5,'plot_names':True}
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

        #Iterate over all storms in season object
        sinfo = season.summary()
        storms = season.dict.keys()
        for storm_idx,storm_key in enumerate(storms):

            #Get data for this storm
            storm = season.dict[storm_key]
            
            #Retrieve storm data
            lats = storm['lat']
            lons = storm['lon']
            vmax = storm['vmax']
            styp = storm['type']
            sdate = storm['date']

            #Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(lons)
                new_lons[new_lons<0] = new_lons[new_lons<0]+360.0
                lons = new_lons.tolist()

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
            
            #Add storm label at start and end points
            if prop['plot_names'] == True:
                self.ax.text(lons[0]+0.0,storm['lat'][0]+1.0,storm['name'].upper(),
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center',transform=ccrs.PlateCarree())
                self.ax.text(lons[-1]+0.0,storm['lat'][-1]+1.0,storm['name'].upper(),
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center',transform=ccrs.PlateCarree())

            #Iterate over storm data to plot
            levels = None
            cmap = None
            for i,(i_lat,i_lon,i_vmax,i_mslp,i_date,i_type) in enumerate(zip(storm['lat'],lons,storm['vmax'],storm['mslp'],storm['date'],storm['type'])):
                    
                #Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))
                
                #Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'],str) == True and prop['linecolor'] in ['vmax','mslp','dvmax_dt','speed']:
                    segmented_colors = True
                    try:
                        color_variable = storm[prop['linecolor']]
                    except:
                        raise ValueError("Storm object must be interpolated to hourly using 'storm.interp().plot(...)' in order to use 'dvmax_dt' or 'speed' for fill color")
                    if prop['levels'] is None: #Auto-determine color levels if needed
                        prop['levels'] = [np.nanmin(color_variable),np.nanmax(color_variable)]
                    cmap,levels = get_cmap_levels(prop['linecolor'],prop['cmap'],prop['levels'])
                    line_color = cmap((color_variable-min(levels))/(max(levels)-min(levels)))[i]
                
                #Otherwise go with user input as is
                else:
                    segmented_colors = False
                    line_color = prop['linecolor']

                #For tropical/subtropical types, color-code if requested
                if i > 0:
                    if i_type in constants.TROPICAL_STORM_TYPES and storm['type'][i-1] in constants.TROPICAL_STORM_TYPES:

                        #Plot underlying black and overlying colored line
                        self.ax.plot([lons[i-1],lons[i]],[storm['lat'][i-1],storm['lat'][i]],'-',
                                      linewidth=prop['linewidth']*1.33,color='k',zorder=storm_idx*5,
                                      transform=ccrs.PlateCarree())
                        self.ax.plot([lons[i-1],lons[i]],[storm['lat'][i-1],storm['lat'][i]],'-',
                                      linewidth=prop['linewidth'],color=line_color,zorder=i_vmax+(storm_idx*5),
                                      transform=ccrs.PlateCarree())

                    #For non-tropical types, plot dotted lines
                    else:

                        #Restrict line width to 1.5 max
                        line_width = prop['linewidth'] + 0.0
                        if line_width > 1.5: line_width = 1.5

                        #Plot dotted line
                        self.ax.plot([lons[i-1],lons[i]],[storm['lat'][i-1],storm['lat'][i]],':',
                                      linewidth=line_width,color=line_color,zorder=i_vmax+(storm_idx*5),
                                      transform=ccrs.PlateCarree(),
                                      path_effects=[path_effects.Stroke(linewidth=line_width*1.33, foreground='k'),
                                                    path_effects.Normal()])
                
                #Plot dots if requested
                if prop['dots'] == True:
                    segmented_colors = self.plot_dot(i_lon,i_lat,i_date,i_vmax,i_type,
                                                     zorder=900+i_vmax,storm_data=storm,prop=prop)

        #--------------------------------------------------------------------------------------
        
        #Pre-generated domains
        if domain is None:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(season.basin)
        else:
            bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
            
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
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
        count_named = sinfo['season_named']
        count_hurricane = sinfo['season_hurricane']
        count_major = sinfo['season_major']
        count_ace = sinfo['season_ace']
        if isinstance(season.year,list) == True:
            count_named = np.sum(sinfo['season_named'])
            count_hurricane = np.sum(sinfo['season_hurricane'])
            count_major = np.sum(sinfo['season_major'])
            count_ace = np.sum(sinfo['season_ace'])
        self.ax.set_title(f"{count_named} named {dot} {count_hurricane} hurricanes {dot} {count_major} major\n{count_ace:.1f} Cumulative ACE",loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add plot credit
        warning_text=""
        if storm['source'] == 'ibtracs' and storm['source_info'] == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"

            self.ax.text(0.99,0.01,warning_text,fontsize=9,color='k',alpha=0.7,
            transform=self.ax.transAxes,ha='right',va='bottom',zorder=10)
        
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
                
        #--------------------------------------------------------------------------------------
        
        #Add legend
        self.add_legend(prop,segmented_colors,levels,cmap,storm)
                
        #--------------------------------------------------------------------------------------
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(save_path,bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax
        
    def plot_track_labels(self, ax, labels, track, k=0.01):

        label_nodes = list(labels.keys())
        labels['place1'] = (2*labels[label_nodes[0]][0]-labels[label_nodes[1]][0],\
                          2*labels[label_nodes[0]][1]-labels[label_nodes[1]][1])
        labels['place2'] = (2*labels[label_nodes[-1]][0]-labels[label_nodes[-2]][0],\
                          2*labels[label_nodes[-1]][1]-labels[label_nodes[-2]][1])
        track['place1'] = labels['place1']
        track['place2'] = labels['place2']
        
        G = nx.DiGraph()
        track_nodes = []
        init_pos = {}
        
        for lab in track.keys():
            labG = 'track_{0}'.format(lab)
            G.add_node(labG)
            track_nodes.append(labG)
            init_pos[labG] = track[lab]
            
        for lab in labels.keys():
            G.add_node(lab)
            G.add_edge(lab,'track_{0}'.format(lab))
            init_pos[lab] = labels[lab]
            
        pos = nx.spring_layout(G, pos=init_pos, fixed=track_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in track_nodes])
        pos_before = np.vstack([init_pos[d] for d in track_nodes])
        scale, shift_x = np.polyfit(pos_after[:,0], pos_before[:,0], 1)
        scale, shift_y = np.polyfit(pos_after[:,1], pos_before[:,1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val*scale) + shift

        for label, _ in G.edges():
            if 'place' not in label:
                self.ax.annotate(label,
                            xy=init_pos[label], xycoords='data',
                            xytext=pos[label], textcoords='data', fontweight='bold', ha='center', va='center',
                            arrowprops=dict(arrowstyle="-",#->
                                            shrinkA=0, shrinkB=0,
                                            connectionstyle="arc3", 
                                            color='k'),
                            transform=ccrs.PlateCarree())
    
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

        #Apply coordinate transform
        transform = ccrs.PlateCarree()._as_mpl_transform(self.ax)
        
        start = False
        for label, data_str in G.edges():
            if start == False:
                start = True
                continue
            self.ax.annotate(label, #xycoords="data"
                        xy=pos[data_str], xycoords=transform,
                        xytext=pos[label], textcoords=transform, fontweight='bold', ha='center', va='center',
                        arrowprops=dict(arrowstyle="-",#->
                                        shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3", 
                                        color='k'),
                        transform=ccrs.PlateCarree(),clip_on=True)

    def plot_gridded(self,xcoord,ycoord,zcoord,varname='type',VEC_FLAG=False,domain="north_atlantic",ax=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm track.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        domain : str
            Domain for the plot. Default is TrackDataset basin. Can be one of the following:
            "north_atlantic" - North Atlantic Ocean basin
            "pacific" - East/Central Pacific Ocean basin
            "lonW/lonE/latS/latN" - Custom plot domain
        plot_all_dots : bool
            Whether to plot dots for all observations along the track. If false, dots will be plotted every 6 hours. Default is false.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Set default properties
        default_prop={'cmap':'category','levels':None,\
                      'left_title':'','right_title':'All storms',
                      'plot_values':False,'values_size':None}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        
        #Initialize plot
        prop = self.add_prop(prop,default_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
        #Determine if contour levels are automatically generated
        auto_levels = True if prop['levels'] is None or prop['levels'] == [] else False

        #Plot domain
        bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
        #--------------------------------------------------------------------------------------

        if VEC_FLAG:
            vecmag = np.hypot(*zcoord)
            if prop['levels'] is None:
                prop['levels'] = (np.nanmin(vecmag),np.nanmax(vecmag))
        elif prop['levels'] is None:
            prop['levels'] = (np.nanmin(zcoord),np.nanmax(zcoord))
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['levels'])
        
        #Generate contourf levels
        if len(clevs) == 2:
            y0 = min(clevs)
            y1 = max(clevs)
            dy = (y1-y0)/8
            scalemag = int(np.log(dy)/np.log(10))
            dy_scaled = dy*10**-scalemag
            dc = min([1,2,5,10], key=lambda x:abs(x-dy_scaled))
            c0 = np.ceil(y0/dc*10**-scalemag)*dc*10**scalemag
            c1 = np.floor(y1/dc*10**-scalemag)*dc*10**scalemag
            clevs = np.arange(c0,c1+dc,dc)
        
        if varname == 'vmax' and prop['cmap'] == 'category':
            vmin = min(clevs); vmax = max(clevs)
        else:
            vmin = min(prop['levels']); vmax = max(prop['levels'])
        
        #For difference/change plots with automatically generated contour levels, ensure that 0 is in the middle
        if auto_levels == True:
            if varname in ['dvmax_dt','dmslp_dt'] or '\n' in prop['title_R']:
                max_val = np.max([np.abs(vmin),vmax])
                vmin = np.round(max_val * -1.0,2)
                vmax = np.round(max_val * 1.0,2)
                clevs = [vmin,np.round(vmin*0.5,2),0,np.round(vmax*0.5,2),vmax]
        
        if len(xcoord.shape) and len(ycoord.shape)==1:
            xcoord,ycoord = np.meshgrid(xcoord,ycoord)
        
        if VEC_FLAG:
            binsize = abs(xcoord[0,0]-xcoord[0,1])
            cbmap = self.ax.pcolor(xcoord,ycoord,vecmag[:-1,:-1],cmap=cmap,vmin=min(clevs),vmax=max(clevs),
                               transform=ccrs.PlateCarree())            
            zcoord = zcoord/vecmag*binsize
            x_center = (xcoord[:-1,:-1]+xcoord[1:,1:])*.5
            y_center = (ycoord[:-1,:-1]+ycoord[1:,1:])*.5
            u = zcoord[0][:-1,:-1]
            v = zcoord[1][:-1,:-1]
            if not prop['plot_values']:
                self.ax.quiver(x_center,y_center,u,v,color='w',alpha=0.6,transform=ccrs.PlateCarree(),\
                           pivot='mid',width=.001*binsize,headwidth=3.5,headlength=4.5,headaxislength=4)
            zcoord = vecmag
        
        else:
            print('--> Generating plot')
            #if varname=='date' and prop['smooth'] is not None:
            #    zcoord[np.isnan(zcoord)]=0
            #    zcoord=gfilt(zcoord,sigma=prop['smooth'])
            #    zcoord[zcoord<min(clevs)]=np.nan
            
            if prop['cmap']=='category' and varname=='vmax':
                norm = mcolors.BoundaryNorm(clevs,cmap.N)
                cbmap = self.ax.pcolor(xcoord,ycoord,zcoord[:-1,:-1],cmap=cmap,norm=norm,
                                       transform=ccrs.PlateCarree())
            else:
                norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
                cbmap = self.ax.pcolor(xcoord,ycoord,zcoord[:-1,:-1],cmap=cmap,norm=norm,
                                       transform=ccrs.PlateCarree())
        if prop['plot_values']:
            binsize = abs(xcoord[0,0]-xcoord[0,1])
            x_center = (xcoord[:-1,:-1]+xcoord[1:,1:])*.5
            y_center = (ycoord[:-1,:-1]+ycoord[1:,1:])*.5
            xs = x_center.flatten(order='C')
            ys = y_center.flatten(order='C')
            zs = zcoord[:-1,:-1].flatten(order='C')
            if prop['values_size'] is None:
                fs = binsize*4
            else:
                fs = prop['values_size']
            for xtext,ytext,ztext in zip(xs,ys,zs):
                if not np.isnan(ztext) and xtext%360>bound_w%360 and xtext%360<bound_e%360 and\
                    ytext>bound_s and ytext<bound_n:
                    square_color = cmap(norm(ztext))
                    square_brightness = np.mean(square_color[:3])*square_color[3]
                    text_color = 'k' if square_brightness>0.5 else 'w' 
                    self.ax.text(xtext,ytext,ztext.astype(int),ha='center',va='center',fontsize=fs,\
                                     color=text_color,alpha=0.8,transform=ccrs.PlateCarree(), zorder=2)
                

        #--------------------------------------------------------------------------------------

        
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
        cax = self.fig.add_axes([bb.x0+1.2*bb.width, bb.y0-.05*bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap,cax=cax,orientation='vertical',\
                                 ticks=clevs)
            
        """
        if len(prop['levels'])>2:
            cax.yaxis.set_ticks(np.linspace(min(clevs),max(clevs),len(clevs)))
            cax.yaxis.set_ticks(np.linspace(0,1,len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
        else:
            cax.yaxis.set_ticks(clevs)
        """
        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
    
        rect_offset = 0.0
        if prop['cmap']=='category' and varname=='vmax':
            cax.yaxis.set_ticks(np.linspace(min(clevs),max(clevs),len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0,1,len(clevs))[:-1]+np.linspace(0,1,len(clevs))[1:])*.5)
            cax2.set_yticklabels(['TD','TS','Cat-1','Cat-2','Cat-3','Cat-4','Cat-5'],fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            
            rect_offset = 0.7
        if varname == 'date':
            cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
            
        rectangle = mpatches.Rectangle((bb.x0,bb.y0-0.1*bb.height),(2+rect_offset)*bb.width,1.1*bb.height,\
                                       fc = 'w',edgecolor = '0.8',alpha = 0.8,\
                                       transform=self.fig.transFigure, zorder=3)
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
        
        #Add plot credit
        text = self.plot_credit()
        self.add_credit(text)
        
        #Return axis if specified, otherwise display figure
        return self.ax

    def plot_summary(self,storms,forecasts,shapefiles,valid_date,domain,ax=None,save_path=None,two_prop={},invest_prop={},storm_prop={},cone_prop={},map_prop={}):
        
        r"""
        Creates a realtime summary plot.
        """
        
        #Set default properties
        default_two_prop={'plot':True,'fontsize':12,'days':5}
        default_invest_prop={'plot':True,'fontsize':12,'linewidth':0.8,'linecolor':'k','linestyle':'dotted','ms':14}
        default_storm_prop={'plot':True,'fontsize':12,'linewidth':0.8,'linecolor':'k','linestyle':'dotted','fillcolor':'category','label_category':True,'ms':14}
        default_cone_prop={'plot':True,'linewidth':1.5,'linecolor':'k','alpha':0.6,'days':5,'fillcolor':'category','label_category':True,'ms':12}
        default_map_prop={'res':'m','land_color':'#FBF5EA','ocean_color':'#EDFBFF','linewidth':0.5,'linecolor':'k','figsize':(14,9),'dpi':200}
        if domain == 'all': default_map_prop['res'] = 'l'
        
        #Initialize plot
        two_prop = self.add_prop(two_prop,default_two_prop)
        invest_prop = self.add_prop(invest_prop,default_invest_prop)
        storm_prop = self.add_prop(storm_prop,default_storm_prop)
        cone_prop = self.add_prop(cone_prop,default_cone_prop)
        map_prop = self.add_prop(map_prop,default_map_prop)
        self.plot_init(ax,map_prop)
        
        #Plot domain
        bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Format title
        add_title = ""
        if two_prop['plot'] == True:
            if two_prop['days'] == 2:
                add_title = " & NHC 2-Day Formation Outlook"
            else:
                add_title = " & NHC 5-Day Formation Outlook"
        
        #--------------------------------------------------------------------------------------
        
        bbox_prop = {'facecolor':'white','alpha':0.5,'edgecolor':'black','boxstyle':'round,pad=0.3'}
        
        if two_prop['plot'] == True:
            
            #Store color
            color_base = {'Low':'yellow','Medium':'orange','High':'red'}

            #Plot areas
            for record, geom in zip(shapefiles['areas'].records(), shapefiles['areas'].geometries()):

                #Read relevant data
                if two_prop['days'] == 2:
                    color = color_base.get(record.attributes['RISK2DAY'],'yellow')
                else:
                    color = color_base.get(record.attributes['RISK5DAY'],'yellow')

                #Plot area
                self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                    facecolor=color, edgecolor=color, alpha=0.3, linewidth=1.5, zorder=3)

                #Plot hatching
                self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                    facecolor='none', edgecolor='k', linewidth=2.25, zorder=4)
                self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                    facecolor='none', edgecolor=color, linewidth=1.5, zorder=4)

            #Plot points
            for record, point in zip(shapefiles['points'].records(), shapefiles['points'].geometries()):

                #Read relevant data
                lon = (list(point.coords)[0][0])
                lat = (list(point.coords)[0][1])
                prob_2day = record.attributes['PROB2DAY'].replace(" ","")
                prob_5day = record.attributes['PROB5DAY'].replace(" ","")
                risk_2day = record.attributes['RISK2DAY'].replace(" ","")
                risk_5day = record.attributes['RISK5DAY'].replace(" ","")

                #Label area
                if two_prop['days'] == 2:
                    color = color_base.get(risk_2day,'yellow')
                    text = prob_2day
                else:
                    color = color_base.get(risk_5day,'yellow')
                    text = prob_5day
                self.ax.plot(lon,lat,'X',ms=15,color=color,mec='k',mew=1.5,transform=ccrs.PlateCarree(),zorder=20)

                #Transform coordinates for label
                x1, y1 = self.ax.projection.transform_point(lon, lat, ccrs.PlateCarree())
                x2, y2 = self.ax.transData.transform((x1, y1))
                x, y = self.ax.transAxes.inverted().transform((x2, y2))

                # plot same point but using axes coordinates
                a = self.ax.text(x,y-0.03,text,ha='center',va='top',transform=self.ax.transAxes,zorder=30,fontweight='bold',fontsize=two_prop['fontsize'],clip_on=True,bbox=bbox_prop)
                a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'),path_effects.Normal()])
        
        #--------------------------------------------------------------------------------------
        
        if invest_prop['plot'] == True or storm_prop['plot'] == True:
            
            #Iterate over all storms
            for storm_idx,storm in enumerate(storms):
                
                #Skip if it's already associated with a risk area, if TWO is being plotted
                if storm.prob_2day != 'N/A' and two_prop['plot'] == True: continue
                
                #Plot invests
                if storm.invest and invest_prop['plot'] == True:
                    
                    #Test
                    self.ax.plot(storm.lon[-1],storm.lat[-1],'X',ms=invest_prop['ms'],color='k',transform=ccrs.PlateCarree(),zorder=20)
                    
                    #Transform coordinates for label
                    x1, y1 = self.ax.projection.transform_point(storm.lon[-1], storm.lat[-1], ccrs.PlateCarree())
                    x2, y2 = self.ax.transData.transform((x1, y1))
                    x, y = self.ax.transAxes.inverted().transform((x2, y2))

                    # plot same point but using axes coordinates
                    a = self.ax.text(x,y-0.03,f"{storm.name.title()}",ha='center',va='top',transform=self.ax.transAxes,zorder=30,fontweight='bold',fontsize=invest_prop['fontsize'],clip_on=True,bbox=bbox_prop)
                    a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'),path_effects.Normal()])
                    
                    #Plot archive track
                    if invest_prop['linewidth'] > 0:
                        self.ax.plot(storm.lon,storm.lat,color=invest_prop['linecolor'],linestyle=invest_prop['linestyle'],zorder=5,transform=ccrs.PlateCarree())
                
                #Plot TCs
                elif storm.invest == False and storm_prop['plot'] == True:
                    
                    #Label dot
                    #self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=14,color='none',mec='k',mew=3.0,transform=ccrs.PlateCarree(),zorder=5)
                    #self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=14,color='none',mec='r',mew=2.0,transform=ccrs.PlateCarree(),zorder=6)
                    category = str(wind_to_category(storm.vmax[-1]))
                    color = get_colors_sshws(storm.vmax[-1])
                    if category == "0": category = 'S'
                    if category == "-1": category = 'D'
                    
                    if storm_prop['fillcolor'] == 'none':
                        self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=storm_prop['ms'],color='none',mec='k',mew=3.0,transform=ccrs.PlateCarree(),zorder=20)
                        self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=storm_prop['ms'],color='none',mec='r',mew=2.0,transform=ccrs.PlateCarree(),zorder=21)
                    
                    else:
                        if storm_prop['fillcolor'] != 'category': color = storm_prop['fillcolor']
                        self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=storm_prop['ms']*1.14,color='k',transform=ccrs.PlateCarree(),zorder=20)
                        self.ax.plot(storm.lon[-1],storm.lat[-1],'o',ms=storm_prop['ms'],color=color,transform=ccrs.PlateCarree(),zorder=21)
                        
                        if storm_prop['label_category'] == True:
                            color = mcolors.to_rgb(color)
                            red,green,blue = color
                            textcolor = 'w'
                            if (red*0.299 + green*0.587 + blue*0.114) > (160.0/255.0): textcolor = 'k'
                            self.ax.text(storm.lon[-1],storm.lat[-1],category,fontsize=storm_prop['ms']*0.83,ha='center',va='center',color=textcolor,
                                         zorder=30,transform=ccrs.PlateCarree(),clip_on=True)
                    
                    #Transform coordinates for label
                    x1, y1 = self.ax.projection.transform_point(storm.lon[-1], storm.lat[-1], ccrs.PlateCarree())
                    x2, y2 = self.ax.transData.transform((x1, y1))
                    x, y = self.ax.transAxes.inverted().transform((x2, y2))

                    # plot same point but using axes coordinates
                    a = self.ax.text(x,y-0.03,f"{storm.name.title()}",ha='center',va='top',transform=self.ax.transAxes,zorder=30,fontweight='bold',fontsize=storm_prop['fontsize'],clip_on=True,bbox=bbox_prop)
                    a.set_path_effects([path_effects.Stroke(linewidth=0.5,foreground='w'),path_effects.Normal()])
                    
                    #Plot archive track
                    if storm_prop['linewidth'] > 0:
                        self.ax.plot(storm.lon,storm.lat,color=storm_prop['linecolor'],linestyle=storm_prop['linestyle'],zorder=5,transform=ccrs.PlateCarree())
                        
                    #Plot cone
                    if cone_prop['plot'] == True:
                        
                        #Retrieve cone
                        forecast_dict = forecasts[storm_idx]
                        
                        try:
                            cone = generate_nhc_cone(forecast_dict,storm.basin,cone_days=cone_prop['days'])

                            #Plot cone
                            if cone_prop['alpha'] > 0 and storm.basin in constants.NHC_BASINS:
                                cone_2d = cone['cone']
                                cone_2d = ndimage.gaussian_filter(cone_2d,sigma=0.5,order=0)
                                self.ax.contourf(cone['lon2d'],cone['lat2d'],cone_2d,[0.9,1.1],colors=['#ffffff','#ffffff'],alpha=cone_prop['alpha'],zorder=4,transform=ccrs.PlateCarree())
                                self.ax.contour(cone['lon2d'],cone['lat2d'],cone_2d,[0.9],linewidths=1.5,colors=['k'],zorder=4,transform=ccrs.PlateCarree())

                            #Plot center line & account for dateline crossing
                            if cone_prop['linewidth'] > 0:
                                self.ax.plot(cone['center_lon'],cone['center_lat'],color='w',linewidth=2.5,zorder=5,transform=ccrs.PlateCarree())
                                self.ax.plot(cone['center_lon'],cone['center_lat'],color='k',linewidth=2.0,zorder=6,transform=ccrs.PlateCarree()) 

                            #Plot forecast dots
                            for idx in range(len(forecast_dict['lat'])):
                                if cone_prop['ms'] == 0: continue
                                color = get_colors_sshws(forecast_dict['vmax'][idx])
                                if cone_prop['fillcolor'] != 'category': color = cone_prop['fillcolor']

                                self.ax.plot(forecast_dict['lon'][idx],forecast_dict['lat'][idx],'o',ms=cone_prop['ms'],mfc=color,mec='k',zorder=7,transform=ccrs.PlateCarree(),clip_on=True)

                                if cone_prop['label_category'] == True:
                                    category = str(wind_to_category(forecast_dict['vmax'][idx]))
                                    if category == "0": category = 'S'
                                    if category == "-1": category = 'D'

                                    color = mcolors.to_rgb(color)
                                    red,green,blue = color
                                    textcolor = 'w'
                                    if (red*0.299 + green*0.587 + blue*0.114) > (160.0/255.0): textcolor = 'k'

                                    self.ax.text(forecast_dict['lon'][idx],forecast_dict['lat'][idx],category,fontsize=cone_prop['ms']*0.81,ha='center',va='center',color=textcolor,
                                                zorder=19,transform=ccrs.PlateCarree(),clip_on=True)
                        except:
                            pass
        
        #--------------------------------------------------------------------------------------
        
        #Plot domain
        bound_w,bound_e,bound_s,bound_n = self.set_projection(domain)
        
        #Plot parallels and meridians
        #This is currently not supported for all cartopy projections.
        try:
            self.plot_lat_lon_lines([bound_w,bound_e,bound_s,bound_n])
        except:
            pass
        
        #--------------------------------------------------------------------------------------
        
        #Add title
        self.ax.set_title(f"Summary{add_title}",loc='left',fontsize=17,fontweight='bold')
        self.ax.set_title(f"Valid: {valid_date.strftime('%H UTC %d %b %Y')}",loc='right',fontsize=13)

        #--------------------------------------------------------------------------------------
        
        #Add credit
        credit_text = self.plot_credit()
        self.add_credit(credit_text)
        
        #--------------------------------------------------------------------------------------
                
        #Add legend
        #self.add_legend(prop,segmented_colors,levels,cmap,storm_data)
                
        #-----------------------------------------------------------------------------------------
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(save_path,bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax
