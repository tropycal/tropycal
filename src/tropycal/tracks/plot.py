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
    
    def plot_storm(self,storm,domain="dynamic",plot_all_dots=False,ax=None,track_labels=False,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of a single storm track.
        
        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
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
        default_prop={'dots':True,'fillcolor':'category','cmap':None,'levels':None,'linecolor':'k','linewidth':1.0,'ms':7.5}
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

        #Iterate over storm data to plot
        for i,(i_lat,i_lon,i_vmax,i_mslp,i_date,i_type) in enumerate(zip(storm_data['lat'],lons,storm_data['vmax'],storm_data['mslp'],storm_data['date'],storm_data['type'])):

            #Determine line color, with SSHWS scale used as default
            if prop['linecolor'] == 'category':
                segmented_colors = True
                line_color = get_colors_sshws(np.nan_to_num(i_vmax))

            #Use user-defined colormap if another storm variable
            elif isinstance(prop['linecolor'],str) == True and prop['linecolor'] in ['vmax','mslp']:
                segmented_colors = True
                color_variable = storm_data[prop['linecolor']]
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
                
                #Skip if plot_all_dots == False and not in 0,6,12,18z
                if plot_all_dots == False:
                    if i_date.strftime('%H%M') not in constants.STANDARD_HOURS: continue

                #Determine fill color, with SSHWS scale used as default
                if prop['fillcolor'] == 'category':
                    segmented_colors = True
                    fill_color = get_colors_sshws(np.nan_to_num(i_vmax))

                #Use user-defined colormap if another storm variable
                elif isinstance(prop['fillcolor'],str) == True and prop['fillcolor'] in ['vmax','mslp']:
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
                self.ax.plot(i_lon,i_lat,marker_type,mfc=fill_color,mec='k',mew=0.5,
                             zorder=5,ms=prop['ms'],transform=ccrs.PlateCarree())

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
            self.ax.legend(handles=[ex,sb,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        elif prop['linecolor'] == 'category' and prop['dots'] == False:
            ex = mlines.Line2D([], [], linestyle='dotted', label='Non-Tropical', color='k')
            td = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Depression', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Storm', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='solid', label='Category 1', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='solid', label='Category 2', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='solid', label='Category 3', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='solid', label='Category 4', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='solid', label='Category 5', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        elif prop['dots'] and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
            handles=[ex,sb,td]
            self.ax.legend(handles=handles,fontsize=11.5)

        elif prop['dots'] == False and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='dotted',label='Non-Tropical', color=prop['linecolor'])
            td = mlines.Line2D([], [], linestyle='solid',label='Tropical', color=prop['linecolor'])
            handles=[ex,td]
            self.ax.legend(handles=handles,fontsize=11.5)

        elif prop['dots'] == True:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color='w')
            handles=[ex,sb,td]
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
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
            cbarlab = make_var_label(prop['linecolor'],storm_data)            
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
        #-----------------------------------------------------------------------------------------
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(save_path,bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax
    
    def plot_storms(self,storms,domain="dynamic",title="TC Track Composite",plot_all_dots=False,labels=False,ax=None,save_path=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of multiple storm tracks.
        
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
        default_prop={'dots':True,'fillcolor':'category','cmap':None,'levels':None,'linecolor':'k','linewidth':1.0,'ms':7.5}
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
            if labels == True:
                self.ax.text(lons[0]+0.0,storm_data['lat'][0]+1.0,f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center')
                self.ax.text(lons[-1]+0.0,storm_data['lat'][-1]+1.0,f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center')
            
            #Iterate over storm data to plot
            for i,(i_lat,i_lon,i_vmax,i_mslp,i_date,i_type) in enumerate(zip(storm_data['lat'],lons,storm_data['vmax'],storm_data['mslp'],storm_data['date'],storm_data['type'])):

                #Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))

                #Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'],str) == True and prop['linecolor'] in ['vmax','mslp']:
                    segmented_colors = True
                    color_variable = storm_data[prop['linecolor']]
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

                    #Skip if plot_all_dots == False and not in 0,6,12,18z
                    if plot_all_dots == False:
                        if i_date.strftime('%H%M') not in constants.STANDARD_HOURS: continue

                    #Determine fill color, with SSHWS scale used as default
                    if prop['fillcolor'] == 'category':
                        segmented_colors = True
                        fill_color = get_colors_sshws(np.nan_to_num(i_vmax))

                    #Use user-defined colormap if another storm variable
                    elif isinstance(prop['fillcolor'],str) == True and prop['fillcolor'] in ['vmax','mslp']:
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
                    self.ax.plot(i_lon,i_lat,marker_type,mfc=fill_color,mec='k',mew=0.5,
                                 zorder=5,ms=prop['ms'],transform=ccrs.PlateCarree())

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
        
        #Add left title
        if title != "": self.ax.set_title(f"{title}",loc='left',fontsize=17,fontweight='bold')

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
            self.ax.legend(handles=[ex,sb,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        elif prop['linecolor'] == 'category' and prop['dots'] == False:
            ex = mlines.Line2D([], [], linestyle='dotted', label='Non-Tropical', color='k')
            td = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Depression', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='solid', label='Sub/Tropical Storm', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='solid', label='Category 1', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='solid', label='Category 2', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='solid', label='Category 3', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='solid', label='Category 4', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='solid', label='Category 5', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex,td,ts,c1,c2,c3,c4,c5], prop={'size':11.5})

        elif prop['dots'] and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
            handles=[ex,sb,td]
            self.ax.legend(handles=handles,fontsize=11.5)

        elif prop['dots'] == False and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='dotted',label='Non-Tropical', color=prop['linecolor'])
            td = mlines.Line2D([], [], linestyle='solid',label='Tropical', color=prop['linecolor'])
            handles=[ex,td]
            self.ax.legend(handles=handles,fontsize=11.5)

        elif prop['dots'] == True:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color='w')
            handles=[ex,sb,td]
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
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
            cbarlab = make_var_label(prop['linecolor'],storm_data)            
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
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
                    #filter dots to only 6 hour intervals
                    time_hr = np.array([i.strftime('%H%M') for i in sdate])
                    #time_idx = np.where((time_hr == '0300') | (time_hr == '0900') | (time_hr == '1500') | (time_hr == '2100'))
                    lat6 = np.array(lats)#[time_idx]
                    lon6 = np.array(lons)#[time_idx]
                    vmax6 = np.array(vmax)#[time_idx]
                    type6 = np.array(styp)#[time_idx]
                    for i,(ilon,ilat,iwnd,itype) in enumerate(zip(lon6,lat6,vmax6,type6)):
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
            cone = self.generate_nhc_cone(forecast,dateline,cone_days)

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
                      'linecolor':'category','linewidth':1.0,'ms':7.5}
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
            self.ax.text(lons[0]+0.0,storm['lat'][0]+1.0,storm['name'].upper(),
                         fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center')
            self.ax.text(lons[-1]+0.0,storm['lat'][-1]+1.0,storm['name'].upper(),
                         fontsize=9,clip_on=True,zorder=1000,alpha=0.7,ha='center',va='center')

            #Iterate over storm data to plot
            for i,(i_lat,i_lon,i_vmax,i_mslp,i_date,i_type) in enumerate(zip(storm['lat'],lons,storm['vmax'],storm['mslp'],storm['date'],storm['type'])):
                    
                #Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))
                
                #Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'],str) == True and prop['linecolor'] in ['vmax','mslp']:
                    segmented_colors = True
                    color_variable = storm[prop['linecolor']]
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
                    
                    #Determine fill color, with SSHWS scale used as default
                    if prop['fillcolor'] == 'category':
                        segmented_colors = True
                        fill_color = get_colors_sshws(np.nan_to_num(i_vmax))

                    #Use user-defined colormap if another storm variable
                    elif isinstance(prop['fillcolor'],str) == True and prop['fillcolor'] in ['vmax','mslp']:
                        segmented_colors = True
                        color_variable = storm[prop['fillcolor']]
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
                    self.ax.plot(i_lon,i_lat,marker_type,mfc=fill_color,mec='k',mew=0.5,
                                 zorder=900+i_vmax,ms=prop['ms'],transform=ccrs.PlateCarree())

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

        elif prop['dots'] == True and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
            handles=[ex,sb,td]
            self.ax.legend(handles=handles,fontsize=11.5, prop={'size':11.5}, loc=1)

        elif prop['dots'] == False and segmented_colors == False:
            ex = mlines.Line2D([], [], linestyle='dotted',label='Non-Tropical', color=prop['linecolor'])
            td = mlines.Line2D([], [], linestyle='solid',label='Tropical', color=prop['linecolor'])
            handles=[ex,td]
            self.ax.legend(handles=handles,fontsize=11.5, prop={'size':11.5}, loc=1)

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
            if prop['cmap']=='category' and prop['fillcolor']=='vmax':
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
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
            if prop['cmap']=='category' and prop['linecolor']=='vmax':
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
                cax.set_yticklabels([f'{mdates.num2date(i):%b %-d}' for i in clevs],fontsize=11.5)
                
        #--------------------------------------------------------------------------------------
        
        #Save image if specified
        if save_path is not None and isinstance(save_path,str) == True:
            plt.savefig(save_path,bbox_inches='tight')
        
        #Return axis if specified, otherwise display figure
        return self.ax
        
    def generate_nhc_cone(self,forecast,dateline,cone_days=5,cone_year=None):
        
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

        fcst_year = forecast['init'].year
        if cone_year is None:
            cone_year = forecast['init'].year
        
        #Fix cone year
        if cone_year > np.max([k for k in constants.CONE_SIZE_ATL.keys()]):
            cone_year = [k for k in constants.CONE_SIZE_ATL.keys()][0]
            warnings.warn(f"No cone information is available for the requested year. Defaulting to {cone_year} cone.")
        elif cone_year not in constants.CONE_SIZE_ATL.keys():
            cone_year = 2008
            warnings.warn(f"No cone information is available for the requested year. Defaulting to 2008 cone.")

        #Retrieve cone size and hours for given year
        if forecast['basin'] in ['north_atlantic','east_pacific']:
            output = nhc_cone_radii(cone_year,forecast['basin'])
            cone_climo_hr = [k for k in output.keys()]
            cone_size = [output[k] for k in output.keys()]
        else:
            cone_climo_hr = [3,12,24,36,48,72,96,120]
            cone_size = 0

        #Function for interpolating between 2 times
        def temporal_interpolation(value, orig_times, target_times):
            f = interp.interp1d(orig_times,value)
            ynew = f(target_times)
            return ynew

        #Function for plugging small array into larger array
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

        #Function for finding nearest value in an array
        def findNearest(array,val):
            return array[np.abs(array - val).argmin()]

        #Function for adding a radius surrounding a point
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
        
        #--------------------------------------------------------------------

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
        elif 3 in forecast['fhr'] and 1 in forecast['fhr'] and 0 in forecast['fhr']:
            fcst_lon = forecast['lon'][2:]
            fcst_lat = forecast['lat'][2:]
            fhr = forecast['fhr'][2:]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
        elif 3 in forecast['fhr'] and 0 in forecast['fhr']:
            idx = np.array([i for i,j in enumerate(forecast['fhr']) if j in cone_climo_hr])
            fcst_lon = np.array(forecast['lon'])[idx]
            fcst_lat = np.array(forecast['lat'])[idx]
            fhr = np.array(forecast['fhr'])[idx]
            t = np.array(fhr)/6.0
            interp_fhr_idx = np.arange(t[0],t[-1]+0.01,0.1)
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
            cone_day_cap = list(fhr).index(cone_days*24)+1
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

        #Interpolate forecast data temporally and spatially
        interp_kind = 'quadratic'
        if len(t) == 2: interp_kind = 'linear' #Interpolate linearly if only 2 forecast points
        x1 = interp.interp1d(t,fcst_lon,kind=interp_kind)
        y1 = interp.interp1d(t,fcst_lat,kind=interp_kind)
        interp_fhr = interp_fhr_idx * 6
        interp_lon = x1(interp_fhr_idx)
        interp_lat = y1(interp_fhr_idx)
        
        #Return if no cone specified
        if cone_size == 0:
            return_dict = {'center_lon':interp_lon,'center_lat':interp_lat}
            return return_dict

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
            cbmap = self.ax.pcolor(xcoord,ycoord,vecmag,cmap=cmap,vmin=min(clevs),vmax=max(clevs),
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
                cbmap = self.ax.pcolor(xcoord,ycoord,zcoord,cmap=cmap,norm=norm,
                                       transform=ccrs.PlateCarree())
            else:
                norm = mcolors.Normalize(vmin=vmin,vmax=vmax)
                cbmap = self.ax.pcolor(xcoord,ycoord,zcoord,cmap=cmap,norm=norm,
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


