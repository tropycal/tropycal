import os
import numpy as np
import scipy.interpolate as interp
import warnings
from datetime import datetime as dt, timedelta
import scipy.ndimage as ndimage
import networkx as nx

# Import internal scripts
from ..plot import Plot

# Import tools
from .tools import *
from ..utils import *
from .. import constants

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
except ImportError:
    warnings.warn(
        "Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


def plot_dot(ax, lon, lat, time, vmax, i_type, zorder, storm_data, prop, i):
    r"""
    Plot a dot on the map per user settings.

    Parameters
    ----------
    lon : int, float
        Longitude of the dot
    lat : int, float
        Latitude of the dot
    time : datetime.datetime
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

    # Return cmap & levels if needed
    extra = {}

    # Determine fill color, with SSHWS scale used as default
    if prop['fillcolor'] == 'category':
        segmented_colors = True
        fill_color = get_colors_sshws(np.nan_to_num(vmax))

    # Use user-defined colormap if another storm variable
    elif isinstance(prop['fillcolor'], str) and prop['fillcolor'] in ['vmax', 'mslp', 'dvmax_dt', 'speed']:
        segmented_colors = True
        color_variable = storm_data[prop['fillcolor']]
        if prop['levels'] is None:  # Auto-determine color levels if needed
            prop['levels'] = [
                np.nanmin(color_variable), np.nanmax(color_variable)]
        cmap, levels = get_cmap_levels(
            prop['fillcolor'], prop['cmap'], prop['levels'])
        fill_color = cmap((color_variable - min(levels)) /
                          (max(levels) - min(levels)))[i]
        extra['cmap'] = cmap
        extra['levels'] = levels

    # Otherwise go with user input as is
    else:
        segmented_colors = False
        fill_color = prop['fillcolor']

    # Determine dot type
    marker_type = '^'
    if i_type in constants.SUBTROPICAL_ONLY_STORM_TYPES:
        marker_type = 's'
    elif i_type in constants.TROPICAL_ONLY_STORM_TYPES:
        marker_type = 'o'

    # Plot marker
    ax.plot(lon, lat, marker_type, mfc=fill_color, mec='k', mew=0.5,
            zorder=zorder, ms=prop['ms'], transform=ccrs.PlateCarree())

    return ax, segmented_colors, extra


def add_legend(ax, fig, prop, segmented_colors, levels=None, cmap=None, storm=None):

    # Linecolor category with dots
    if prop['fillcolor'] == 'category' and prop['dots']:
        ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                           mec='k', mew=0.5, label='Non-Tropical', marker='^', color='w')
        sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                           mec='k', mew=0.5, label='Subtropical', marker='s', color='w')
        td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k', mew=0.5,
                           label='Tropical Depression', marker='o', color=get_colors_sshws(33))
        ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Tropical Storm', marker='o', color=get_colors_sshws(34))
        c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Category 1', marker='o', color=get_colors_sshws(64))
        c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Category 2', marker='o', color=get_colors_sshws(83))
        c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Category 3', marker='o', color=get_colors_sshws(96))
        c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Category 4', marker='o', color=get_colors_sshws(113))
        c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Category 5', marker='o', color=get_colors_sshws(137))
        l = ax.legend(handles=[ex, sb, td, ts, c1, c2,
                      c3, c4, c5], prop={'size': 11.5}, loc=1)
        l.set_zorder(1001)

    # Linecolor category without dots
    elif prop['linecolor'] == 'category' and not prop['dots']:
        ex = mlines.Line2D([], [], linestyle='dotted',
                           label='Non-Tropical', color='k')
        td = mlines.Line2D([], [], linestyle='solid',
                           label='Sub/Tropical Depression', color=get_colors_sshws(33))
        ts = mlines.Line2D([], [], linestyle='solid',
                           label='Sub/Tropical Storm', color=get_colors_sshws(34))
        c1 = mlines.Line2D([], [], linestyle='solid',
                           label='Category 1', color=get_colors_sshws(64))
        c2 = mlines.Line2D([], [], linestyle='solid',
                           label='Category 2', color=get_colors_sshws(83))
        c3 = mlines.Line2D([], [], linestyle='solid',
                           label='Category 3', color=get_colors_sshws(96))
        c4 = mlines.Line2D([], [], linestyle='solid',
                           label='Category 4', color=get_colors_sshws(113))
        c5 = mlines.Line2D([], [], linestyle='solid',
                           label='Category 5', color=get_colors_sshws(137))
        l = ax.legend(handles=[ex, td, ts, c1, c2, c3,
                      c4, c5], prop={'size': 11.5}, loc=1)
        l.set_zorder(1001)

    # Non-segmented custom colormap with dots
    elif prop['dots'] and not segmented_colors:
        ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
        sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
        td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                           mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
        handles = [ex, sb, td]
        l = ax.legend(handles=handles, fontsize=11.5,
                      prop={'size': 11.5}, loc=1)
        l.set_zorder(1001)

    # Non-segmented custom colormap without dots
    elif not prop['dots'] and not segmented_colors:
        ex = mlines.Line2D([], [], linestyle='dotted',
                           label='Non-Tropical', color=prop['linecolor'])
        td = mlines.Line2D([], [], linestyle='solid',
                           label='Tropical', color=prop['linecolor'])
        handles = [ex, td]
        l = ax.legend(handles=handles, fontsize=11.5,
                      prop={'size': 11.5}, loc=1)
        l.set_zorder(1001)

    # Custom colormap with dots
    elif prop['dots'] and segmented_colors:
        ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                           mec='k', mew=0.5, label='Non-Tropical', marker='^', color='w')
        sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                           mec='k', mew=0.5, label='Subtropical', marker='s', color='w')
        td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                           mec='k', mew=0.5, label='Tropical', marker='o', color='w')
        handles = [ex, sb, td]
        for _ in range(7):
            handles.append(mlines.Line2D(
                [], [], linestyle='-', label='', lw=0))
        l = ax.legend(handles=handles, fontsize=11.5)
        l.set_zorder(1001)
        plt.draw()

        # Get the bbox
        try:
            bb = l.legendPatch.get_bbox().inverse_transformed(fig.transFigure)
        except:
            bb = l.legendPatch.get_bbox().transformed(fig.transFigure.inverted())

        # Define colorbar axis
        cax = fig.add_axes(
            [bb.x0 + 0.47 * bb.width, bb.y0 + .057 * bb.height, 0.015, .65 * bb.height])
        norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(cbmap, cax=cax, orientation='vertical',
                            ticks=levels)

        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
        cbar.set_label(prop['fillcolor'], fontsize=11.5, rotation=90)

        rect_offset = 0.0
        if prop['cmap'] == 'category' and prop['fillcolor'] == 'vmax':
            cax.yaxis.set_ticks(np.linspace(
                min(levels), max(levels), len(levels)))
            cax.yaxis.set_ticklabels(levels)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0, 1, len(levels))[
                                 :-1] + np.linspace(0, 1, len(levels))[1:]) * .5)
            cax2.set_yticklabels(
                ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            rect_offset = 0.7
        if prop['fillcolor'] == 'time':
            cax.set_yticklabels(
                [f'{mdates.num2date(i):%b %-d}' for i in levels], fontsize=11.5)

    # Custom colormap without dots
    else:
        ex = mlines.Line2D([], [], linestyle='dotted',
                           label='Non-Tropical', color='k')
        td = mlines.Line2D([], [], linestyle='solid',
                           label='Tropical', color='k')
        handles = [ex, td]
        for _ in range(7):
            handles.append(mlines.Line2D(
                [], [], linestyle='-', label='', lw=0))
        l = ax.legend(handles=handles, fontsize=11.5)
        l.set_zorder(1001)
        plt.draw()

        # Get the bbox
        try:
            bb = l.legendPatch.get_bbox().inverse_transformed(fig.transFigure)
        except:
            bb = l.legendPatch.get_bbox().transformed(fig.transFigure.inverted())

        # Define colorbar axis
        cax = fig.add_axes(
            [bb.x0 + 0.47 * bb.width, bb.y0 + .057 * bb.height, 0.015, .65 * bb.height])
        norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(cbmap, cax=cax, orientation='vertical',
                            ticks=levels)

        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')
        cbarlab = make_var_label(prop['linecolor'], storm)
        cbar.set_label(cbarlab, fontsize=11.5, rotation=90)

        rect_offset = 0.0
        if prop['cmap'] == 'category' and prop['linecolor'] == 'vmax':
            cax.yaxis.set_ticks(np.linspace(
                min(levels), max(levels), len(levels)))
            cax.yaxis.set_ticklabels(levels)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0, 1, len(levels))[
                                 :-1] + np.linspace(0, 1, len(levels))[1:]) * .5)
            cax2.set_yticklabels(
                ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')
            rect_offset = 0.7
        if prop['linecolor'] == 'time':
            cax.set_yticklabels(
                [f'{mdates.num2date(i):%b %-d}' for i in levels], fontsize=11.5)

    return ax, fig


class TrackPlot(Plot):

    def __init__(self):

        self.use_credit = True

    def plot_storms(self, storms, domain="dynamic", title="TC Track Composite", plot_all_dots=False, track_labels=False, ax=None, save_path=None, prop={}, map_prop={}):
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

        # Set default properties
        default_prop = {'dots': True, 'fillcolor': 'category', 'cmap': None, 'levels': None,
                        'linecolor': 'k', 'linewidth': 1.0, 'ms': 7.5, 'plot_names': False}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # --------------------------------------------------------------------------------------

        # Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        # Iterate through all storms provided
        for storm in storms:

            # Check for storm type, then get data for storm
            if isinstance(storm, str):
                storm_data = self.data[storm]
            elif isinstance(storm, tuple):
                storm = self.get_storm_id(storm[0], storm[1])
                storm_data = self.data[storm]
            elif isinstance(storm, dict):
                storm_data = storm
            else:
                raise RuntimeError(
                    "Error: Storm must be a string (e.g., 'AL052019'), tuple (e.g., ('Matthew',2016)), or dict.")

            # Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['time']

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(lons)
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                lons = new_lons.tolist()

            # Force dynamic_tropical to tropical if an invest
            invest_bool = False
            if 'invest' in storm_data.keys() and storm_data['invest']:
                invest_bool = True
                if domain == 'dynamic_tropical':
                    domain = 'dynamic'

            # Add to coordinate extrema
            if domain == 'dynamic_tropical':
                type_array = np.array(storm_data['type'])
                idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
                    type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
                use_lats = (np.array(storm_data['lat'])[idx]).tolist()
                use_lons = (np.array(lons)[idx]).tolist()
            else:
                use_lats = storm_data['lat']
                use_lons = np.copy(lons).tolist()

            # Add to coordinate extrema
            if max_lat is None:
                max_lat = max(use_lats)
            else:
                if max(use_lats) > max_lat:
                    max_lat = max(use_lats)
            if min_lat is None:
                min_lat = min(use_lats)
            else:
                if min(use_lats) < min_lat:
                    min_lat = min(use_lats)
            if max_lon is None:
                max_lon = max(use_lons)
            else:
                if max(use_lons) > max_lon:
                    max_lon = max(use_lons)
            if min_lon is None:
                min_lon = min(use_lons)
            else:
                if min(use_lons) < min_lon:
                    min_lon = min(use_lons)

            # Add storm label at start and end points
            if prop['plot_names']:
                self.ax.text(lons[0] + 0.0, storm_data['lat'][0] + 1.0, f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9, clip_on=True, zorder=1000, alpha=0.7, ha='center', va='center', transform=ccrs.PlateCarree())
                self.ax.text(lons[-1] + 0.0, storm_data['lat'][-1] + 1.0, f"{storm_data['name'].upper()} {storm_data['year']}",
                             fontsize=9, clip_on=True, zorder=1000, alpha=0.7, ha='center', va='center', transform=ccrs.PlateCarree())

            # Iterate over storm data to plot
            levels = None
            cmap = None
            for i, (i_lat, i_lon, i_vmax, i_mslp, i_time, i_type) in enumerate(zip(storm_data['lat'], lons, storm_data['vmax'], storm_data['mslp'], storm_data['time'], storm_data['type'])):

                # Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))

                # Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'], str) and prop['linecolor'] in ['vmax', 'mslp', 'dvmax_dt', 'speed']:
                    segmented_colors = True
                    try:
                        color_variable = storm_data[prop['linecolor']]
                    except:
                        raise ValueError(
                            "Storm object must be interpolated to hourly using 'storm.interp().plot(...)' in order to use 'dvmax_dt' or 'speed' for fill color")
                    if prop['levels'] is None:  # Auto-determine color levels if needed
                        prop['levels'] = [
                            np.nanmin(color_variable), np.nanmax(color_variable)]
                    cmap, levels = get_cmap_levels(
                        prop['linecolor'], prop['cmap'], prop['levels'])
                    line_color = cmap(
                        (color_variable - min(levels)) / (max(levels) - min(levels)))[i]

                # Otherwise go with user input as is
                else:
                    segmented_colors = False
                    line_color = prop['linecolor']

                # For tropical/subtropical types, color-code if requested
                if i > 0:
                    if i_type in constants.TROPICAL_STORM_TYPES and storm_data['type'][i - 1] in constants.TROPICAL_STORM_TYPES:

                        # Plot underlying black and overlying colored line
                        self.ax.plot([lons[i - 1], lons[i]], [storm_data['lat'][i - 1], storm_data['lat'][i]], '-',
                                     linewidth=prop['linewidth'] * 1.33, color='k', zorder=3,
                                     transform=ccrs.PlateCarree())
                        self.ax.plot([lons[i - 1], lons[i]], [storm_data['lat'][i - 1], storm_data['lat'][i]], '-',
                                     linewidth=prop['linewidth'], color=line_color, zorder=4,
                                     transform=ccrs.PlateCarree())

                    # For non-tropical types, plot dotted lines
                    else:

                        # Restrict line width to 1.5 max
                        line_width = prop['linewidth'] + 0.0
                        if line_width > 1.5:
                            line_width = 1.5

                        # Plot dotted line
                        self.ax.plot([lons[i - 1], lons[i]], [storm_data['lat'][i - 1], storm_data['lat'][i]], ':',
                                     linewidth=line_width, color=line_color, zorder=4,
                                     transform=ccrs.PlateCarree(),
                                     path_effects=[path_effects.Stroke(linewidth=line_width * 1.33, foreground='k'),
                                                   path_effects.Normal()])

                # Plot dots if requested
                if prop['dots']:
                    if not plot_all_dots and i_time.strftime('%H%M') not in constants.STANDARD_HOURS:
                        continue
                    self.ax, segmented_colors, extra = plot_dot(self.ax, i_lon, i_lat, i_time, i_vmax, i_type,
                                                                zorder=5, storm_data=storm_data, prop=prop, i=i)
                    if 'cmap' in extra.keys():
                        cmap = extra['cmap']
                    if 'levels' in extra.keys():
                        levels = extra['levels']

                # Label track dots
                if track_labels in ['valid_utc']:
                    if track_labels == 'valid_utc':
                        strformat = '%H UTC \n%-m/%-d'
                        labels = {t.strftime(strformat): (x, y) for t, x, y in zip(
                            sdate, lons, lats) if t.hour == 0}
                        track = {t.strftime(strformat): (x, y)
                                 for t, x, y in zip(sdate, lons, lats)}
                    self.plot_track_labels(self.ax, labels, track, k=.9)

        # --------------------------------------------------------------------------------------

        # Storm-centered plot domain
        if domain == "dynamic" or domain == "dynamic_tropical":

            bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        # Add left title
        if len(storms) > 1:
            if title != "":
                self.ax.set_title(f"{title}", loc='left',
                                  fontsize=17, fontweight='bold')
        else:
            # Add left title
            type_array = np.array(storm_data['type'])
            idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
                type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))

            # Check if storm is an invest with a leading 9
            add_ptc_flag = False
            check_invest = False
            if len(storm_data['id']) > 4 and str(storm_data['id'][2]) == "9":
                check_invest = True

            if len(storm_data['id']) == 0 and len(idx[0]) == 0:
                idx = np.array([True for i in type_array])
                tropical_vmax = np.array(storm_data['vmax'])
                self.ax.set_title(
                    f"Cyclone {storm_data['name']}", loc='left', fontsize=17, fontweight='bold')
            elif not check_invest and (invest_bool == False or len(idx[0]) > 0):
                tropical_vmax = np.array(storm_data['vmax'])[idx]

                # Coerce to include non-TC points if storm hasn't been designated yet
                if len(tropical_vmax) == 0 and len(storm_data['id']) > 4:
                    add_ptc_flag = True
                    idx = np.where((type_array == 'LO') | (type_array == 'DB'))
                    tropical_vmax = np.array(storm_data['vmax'])[idx]

                if all_nan(tropical_vmax):
                    storm_type = 'Tropical Cyclone'
                else:
                    subtrop = classify_subtropical(
                        np.array(storm_data['type']))
                    peak_idx = storm_data['vmax'].index(
                        np.nanmax(tropical_vmax))
                    peak_basin = storm_data['wmo_basin'][peak_idx]
                    storm_type = get_storm_classification(
                        np.nanmax(tropical_vmax), subtrop, peak_basin)
                    if add_ptc_flag:
                        storm_type = "Potential Tropical Cyclone"
                self.ax.set_title(
                    f"{storm_type} {storm_data['name']}", loc='left', fontsize=17, fontweight='bold')
            else:
                # Use all indices for invests
                idx = np.array([True for i in type_array])
                tropical_vmax = np.array(storm_data['vmax'])

                # Determine letter in front of invest
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

                # Add title
                self.ax.set_title(
                    f"INVEST {storm_data['id'][2:4]}{add_letter}", loc='left', fontsize=17, fontweight='bold')

            # Add right title
            ace = storm_data['ace']
            if add_ptc_flag:
                ace = 0.0
            type_array = np.array(storm_data['type'])

            # Get storm extrema for display
            mslp_key = 'mslp' if 'wmo_mslp' not in storm_data.keys() else 'wmo_mslp'
            if all_nan(np.array(storm_data[mslp_key])[idx]):
                min_pres = "N/A"
            else:
                min_pres = int(np.nan_to_num(
                    np.nanmin(np.array(storm_data[mslp_key])[idx])))
            if all_nan(np.array(storm_data['vmax'])[idx]):
                max_wind = "N/A"
            else:
                max_wind = int(np.nan_to_num(
                    np.nanmax(np.array(storm_data['vmax'])[idx])))
            start_time = dt.strftime(
                np.array(storm_data['time'])[idx][0], '%d %b %Y')
            end_time = dt.strftime(np.array(storm_data['time'])[
                                   idx][-1], '%d %b %Y')
            endash = u"\u2013"
            dot = u"\u2022"
            self.ax.set_title(
                f"{start_time} {endash} {end_time}\n{max_wind} kt {dot} {min_pres} hPa {dot} {ace:.1f} ACE", loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add plot credit
        warning_text = ""
        if storm_data['source'] == 'ibtracs' and storm_data['source_info'] == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"

            self.ax.text(0.99, 0.01, warning_text, fontsize=9, color='k', alpha=0.7,
                         transform=self.ax.transAxes, ha='right', va='bottom', zorder=10)

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # --------------------------------------------------------------------------------------

        # Add legend
        self.ax, self.fig = add_legend(
            self.ax, self.fig, prop, segmented_colors, levels, cmap, storm_data)

        # -----------------------------------------------------------------------------------------

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_storm_nhc(self, forecast, track=None, track_labels='fhr', cone_days=5, domain="dynamic_forecast", ax=None, save_path=None, prop={}, map_prop={}):
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

        # Set default properties
        default_prop = {'dots': True, 'fillcolor': 'category', 'linecolor': 'k',
                        'linewidth': 1.0, 'ms': 7.5, 'cone_lw': 1.0, 'cone_alpha': 0.6}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # --------------------------------------------------------------------------------------

        # Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        # Add storm or multiple storms
        if track != "":

            # Check for storm type, then get data for storm
            if isinstance(track, dict):
                storm_data = track
            else:
                raise RuntimeError("Error: track must be of type dict.")

            # Retrieve storm data
            lats = storm_data['lat']
            lons = storm_data['lon']
            vmax = storm_data['vmax']
            styp = storm_data['type']
            sdate = storm_data['time']

            # Check if there's enough data points to plot
            matching_times = [i for i in sdate if i <= forecast['init']]
            check_length = len(matching_times)
            if check_length >= 2:

                # Subset until time of forecast
                matching_times = [i for i in sdate if i <= forecast['init']]
                plot_idx = sdate.index(matching_times[-1]) + 1
                lats = storm_data['lat'][:plot_idx]
                lons = storm_data['lon'][:plot_idx]
                vmax = storm_data['vmax'][:plot_idx]
                styp = storm_data['type'][:plot_idx]
                sdate = storm_data['time'][:plot_idx]

                # Account for cases crossing dateline
                if self.proj.proj4_params['lon_0'] == 180.0:
                    new_lons = np.array(lons)
                    new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                    lons = new_lons.tolist()

                # Connect to 1st forecast location
                fcst_hr = np.array(forecast['fhr'])
                start_slice = 0
                if 3 in fcst_hr:
                    start_slice = 3
                iter_hr = np.array(forecast['fhr'])[fcst_hr >= start_slice][0]
                fcst_lon = np.array(forecast['lon'])[fcst_hr >= start_slice][0]
                fcst_lat = np.array(forecast['lat'])[fcst_hr >= start_slice][0]
                fcst_type = np.array(forecast['type'])[
                    fcst_hr >= start_slice][0]
                fcst_vmax = np.array(forecast['vmax'])[
                    fcst_hr >= start_slice][0]
                if fcst_type == "":
                    fcst_type = get_storm_type(fcst_vmax, False)
                if self.proj.proj4_params['lon_0'] == 180.0:
                    if fcst_lon < 0:
                        fcst_lon = fcst_lon + 360.0
                lons.append(fcst_lon)
                lats.append(fcst_lat)
                vmax.append(fcst_vmax)
                styp.append(fcst_type)
                sdate.append(sdate[-1] + timedelta(hours=start_slice))

                # Add to coordinate extrema
                if domain != "dynamic_forecast":
                    if max_lat is None:
                        max_lat = max(lats)
                    else:
                        if max(lats) > max_lat:
                            max_lat = max(lats)
                    if min_lat is None:
                        min_lat = min(lats)
                    else:
                        if min(lats) < min_lat:
                            min_lat = min(lats)
                    if max_lon is None:
                        max_lon = max(lons)
                    else:
                        if max(lons) > max_lon:
                            max_lon = max(lons)
                    if min_lon is None:
                        min_lon = min(lons)
                    else:
                        if min(lons) < min_lon:
                            min_lon = min(lons)
                else:
                    max_lat = lats[-1] + 0.2
                    min_lat = lats[-2] - 0.2
                    max_lon = lons[-1] + 0.2
                    min_lon = lons[-2] - 0.2

                # Plot storm line as specified
                if prop['linecolor'] == 'category':
                    type6 = np.array(styp)
                    for i in (np.arange(len(lats[1:])) + 1):
                        ltype = 'solid'
                        if type6[i] not in constants.TROPICAL_STORM_TYPES:
                            ltype = 'dotted'
                        self.ax.plot([lons[i - 1], lons[i]], [lats[i - 1], lats[i]],
                                     '-', color=get_colors_sshws(np.nan_to_num(vmax[i])), linewidth=prop['linewidth'], linestyle=ltype,
                                     transform=ccrs.PlateCarree(),
                                     path_effects=[path_effects.Stroke(linewidth=prop['linewidth'] * 1.25, foreground='k'), path_effects.Normal()])
                else:
                    self.ax.plot(
                        lons, lats, '-', color=prop['linecolor'], linewidth=prop['linewidth'], transform=ccrs.PlateCarree())

                # Plot storm dots as specified
                if prop['dots']:
                    for i, (ilon, ilat, iwnd, itype) in enumerate(zip(lons, lats, vmax, styp)):
                        mtype = '^'
                        if itype in constants.SUBTROPICAL_ONLY_STORM_TYPES:
                            mtype = 's'
                        elif itype in constants.TROPICAL_ONLY_STORM_TYPES:
                            mtype = 'o'
                        if prop['fillcolor'] == 'category':
                            ncol = get_colors_sshws(np.nan_to_num(iwnd))
                        else:
                            ncol = 'k'
                        self.ax.plot(ilon, ilat, mtype, color=ncol, mec='k',
                                     mew=0.5, ms=prop['ms'], transform=ccrs.PlateCarree())

        # --------------------------------------------------------------------------------------

        # Error check cone days
        if not isinstance(cone_days, int):
            raise TypeError("Error: cone_days must be of type int")
        if cone_days not in [5, 4, 3, 2]:
            raise ValueError(
                "Error: cone_days must be an int between 2 and 5.")

        # Error check forecast dict
        if not isinstance(forecast, dict):
            raise RuntimeError("Error: Forecast must be of type dict")

        # Determine first forecast index
        fcst_hr = np.array(forecast['fhr'])
        start_slice = 0
        if 3 in fcst_hr:
            start_slice = 3
        check_duration = fcst_hr[(fcst_hr >= start_slice) & (
            fcst_hr <= cone_days * 24)]

        # Check for sufficiently many hours
        if len(check_duration) > 1:

            # Generate forecast cone for forecast data
            dateline = False
            if self.proj.proj4_params['lon_0'] == 180.0:
                dateline = True
            cone = generate_nhc_cone(forecast, forecast['basin'], dateline, cone_days)

            # Contour fill cone & account for dateline crossing
            cone_lon = cone['lon']
            cone_lat = cone['lat']
            if 'cone' in forecast.keys() and not forecast['cone']:
                pass
            else:
                if self.proj.proj4_params['lon_0'] == 180.0:
                    new_lons = np.array(cone['lon2d'])
                    new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                    cone['lon2d'] = new_lons
                    center_lon = np.array(cone['center_lon'])
                    center_lon[center_lon < 0] = center_lon[center_lon < 0] + 360.0
                    cone['center_lon'] = center_lon
                plot_cone(self.ax, cone, plot_center_line=True, center_linewidth=2.0, zorder=3)

            # Retrieve forecast dots
            iter_hr = np.array(forecast['fhr'])[
                (fcst_hr >= start_slice) & (fcst_hr <= cone_days * 24)]
            fcst_lon = np.array(forecast['lon'])[
                (fcst_hr >= start_slice) & (fcst_hr <= cone_days * 24)]
            fcst_lat = np.array(forecast['lat'])[
                (fcst_hr >= start_slice) & (fcst_hr <= cone_days * 24)]
            fcst_type = np.array(forecast['type'])[
                (fcst_hr >= start_slice) & (fcst_hr <= cone_days * 24)]
            fcst_vmax = np.array(forecast['vmax'])[
                (fcst_hr >= start_slice) & (fcst_hr <= cone_days * 24)]

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(fcst_lon)
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                fcst_lon = new_lons.tolist()

            # Plot forecast dots
            for i, (ilon, ilat, itype, iwnd, ihr) in enumerate(zip(fcst_lon, fcst_lat, fcst_type, fcst_vmax, iter_hr)):
                mtype = '^'
                if itype in constants.SUBTROPICAL_ONLY_STORM_TYPES:
                    mtype = 's'
                elif itype in list(constants.TROPICAL_ONLY_STORM_TYPES) + ['']:
                    mtype = 'o'
                if prop['fillcolor'] == 'category':
                    ncol = get_colors_sshws(np.nan_to_num(iwnd))
                else:
                    ncol = 'k'
                # Marker width
                mew = 0.5
                use_zorder = 5
                if i == 0:
                    mew = 2.0
                    use_zorder = 10
                self.ax.plot(ilon, ilat, mtype, color=ncol, mec='k', mew=mew,
                             ms=prop['ms'] * 1.3, transform=ccrs.PlateCarree(), zorder=use_zorder)

            # Label forecast dots
            if track_labels in ['fhr', 'valid_utc', 'valid_edt', 'fhr_wind_kt', 'fhr_wind_mph']:
                valid_dates = [forecast['init'] +
                               timedelta(hours=int(i)) for i in iter_hr]
                if track_labels == 'fhr':
                    labels = [str(i) for i in iter_hr]
                if track_labels == 'fhr_wind_kt':
                    labels = [
                        f"Hour {iter_hr[i]}\n{fcst_vmax[i]} kt" for i in range(len(iter_hr))]
                if track_labels == 'fhr_wind_mph':
                    labels = [f"Hour {iter_hr[i]}\n{knots_to_mph(fcst_vmax[i])} mph" for i in range(
                        len(iter_hr))]
                if track_labels == 'valid_edt':
                    labels = [str(int(i.strftime('%I'))) + ' ' + i.strftime('%p %a')
                              for i in [j - timedelta(hours=4) for j in valid_dates]]
                    edt_warning = True
                if track_labels == 'valid_utc':
                    labels = [
                        f"{i.strftime('%H UTC')}\n{str(i.month)}/{str(i.day)}" for i in valid_dates]
                self.plot_nhc_labels(
                    self.ax, fcst_lon, fcst_lat, labels, k=1.2)

            # Add cone coordinates to coordinate extrema
            if 'cone' in forecast.keys() and not forecast['cone']:
                if domain == "dynamic_forecast" or max_lat is None:
                    max_lat = max(center_lat)
                    min_lat = min(center_lat)
                    max_lon = max(center_lon)
                    min_lon = min(center_lon)
                else:
                    if max(center_lat) > max_lat:
                        max_lat = max(center_lat)
                    if min(center_lat) < min_lat:
                        min_lat = min(center_lat)
                    if max(center_lon) > max_lon:
                        max_lon = max(center_lon)
                    if min(center_lon) < min_lon:
                        min_lon = min(center_lon)
            else:
                if domain == "dynamic_forecast" or max_lat is None:
                    max_lat = max(cone_lat)
                    min_lat = min(cone_lat)
                    max_lon = max(cone_lon)
                    min_lon = min(cone_lon)
                else:
                    if max(cone_lat) > max_lat:
                        max_lat = max(cone_lat)
                    if min(cone_lat) < min_lat:
                        min_lat = min(cone_lat)
                    if max(cone_lon) > max_lon:
                        max_lon = max(cone_lon)
                    if min(cone_lon) < min_lon:
                        min_lon = min(cone_lon)

        # --------------------------------------------------------------------------------------

        # Storm-centered plot domain
        if domain == "dynamic" or domain == 'dynamic_forecast':

            bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        # Identify storm type (subtropical, hurricane, etc)
        first_fcst_wind = np.array(forecast['vmax'])[fcst_hr >= start_slice][0]
        first_fcst_mslp = np.array(forecast['mslp'])[fcst_hr >= start_slice][0]
        first_fcst_type = np.array(forecast['type'])[fcst_hr >= start_slice][0]
        if all_nan(first_fcst_wind):
            storm_type = 'Unknown'
        else:
            subtrop = first_fcst_type in constants.SUBTROPICAL_ONLY_STORM_TYPES
            cur_wind = first_fcst_wind + 0
            storm_type = get_storm_classification(
                np.nan_to_num(cur_wind), subtrop, 'north_atlantic')

        # Identify storm name (and storm type, if post-tropical or potential TC)
        matching_times = [i for i in storm_data['time']
                          if i <= forecast['init']]
        if check_length < 2:
            if all_nan(first_fcst_wind):
                storm_name = storm_data['name']
            else:
                storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
                if first_fcst_wind >= 34 and first_fcst_type in constants.TROPICAL_STORM_TYPES:
                    storm_name = storm_data['name']
                if first_fcst_type not in constants.TROPICAL_STORM_TYPES:
                    storm_type = 'Potential Tropical Cyclone'
        else:
            storm_name = num_to_text(int(storm_data['id'][2:4])).upper()
            storm_type = 'Potential Tropical Cyclone'
            storm_tropical = False
            if all_nan(vmax):
                storm_type = 'Unknown'
                storm_name = storm_data['name']
            else:
                for i, (iwnd, ityp) in enumerate(zip(vmax, styp)):
                    if ityp in constants.TROPICAL_STORM_TYPES:
                        storm_tropical = True
                        subtrop = ityp in constants.SUBTROPICAL_ONLY_STORM_TYPES
                        storm_type = get_storm_classification(
                            np.nan_to_num(iwnd), subtrop, 'north_atlantic')
                        if np.isnan(iwnd):
                            storm_type = 'Unknown'
                    else:
                        if storm_tropical:
                            storm_type = 'Post Tropical Cyclone'
                    if ityp in constants.NAMED_TROPICAL_STORM_TYPES:
                        storm_name = storm_data['name']

        # Fix storm types for non-NHC basins
        if 'cone' in forecast.keys():
            storm_type = get_storm_classification(
                first_fcst_wind, False, forecast['basin'])

        # Add left title
        self.ax.set_title(f"{storm_type} {storm_name}",
                          loc='left', fontsize=17, fontweight='bold')

        endash = u"\u2013"
        dot = u"\u2022"

        # Get current advisory information
        first_fcst_wind = "N/A" if np.isnan(first_fcst_wind) else int(first_fcst_wind)
        first_fcst_mslp = "N/A" if np.isnan(first_fcst_mslp) else int(first_fcst_mslp)

        # Get time of advisory
        fcst_hr = forecast['fhr']
        start_slice = 0
        if 3 in fcst_hr:
            start_slice = 1
        forecast_date = (
            forecast['init'] + timedelta(hours=fcst_hr[start_slice])).strftime("%H%M UTC %d %b %Y")
        forecast_id = forecast['advisory_num']

        if forecast_id == -1:
            title_text = f"Current Intensity: {knots_to_mph(first_fcst_wind)} mph {dot} {first_fcst_mslp} hPa"
            if 'cone' in forecast.keys() and not forecast['cone']:
                title_text += f"\nJTWC Issued: {forecast_date}"
            else:
                title_text += f"\nNHC Issued: {forecast_date}"
        else:
            if first_fcst_wind != "N/A":
                first_fcst_wind = knots_to_mph(first_fcst_wind)
            title_text = f"{first_fcst_wind} mph {dot} {first_fcst_mslp} hPa {dot} Forecast #{forecast_id}"
            title_text += f"\nForecast Issued: {forecast_date}"

        # Add right title
        self.ax.set_title(title_text, loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add legend
        if prop['fillcolor'] == 'category' or prop['linecolor'] == 'category':

            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                               mec='k', mew=0.5, label='Non-Tropical', marker='^', color='w')
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                               mec='k', mew=0.5, label='Subtropical', marker='s', color='w')
            uk = mlines.Line2D([], [], linestyle='None', ms=prop['ms'],
                               mec='k', mew=0.5, label='Unknown', marker='o', color='w')
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k', mew=0.5,
                               label='Tropical Depression', marker='o', color=get_colors_sshws(33))
            ts = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Tropical Storm', marker='o', color=get_colors_sshws(34))
            c1 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Category 1', marker='o', color=get_colors_sshws(64))
            c2 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Category 2', marker='o', color=get_colors_sshws(83))
            c3 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Category 3', marker='o', color=get_colors_sshws(96))
            c4 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Category 4', marker='o', color=get_colors_sshws(113))
            c5 = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Category 5', marker='o', color=get_colors_sshws(137))
            self.ax.legend(handles=[ex, sb, uk, td, ts,
                           c1, c2, c3, c4, c5], prop={'size': 11.5})

        # Add forecast label warning
        try:
            if edt_warning:
                warning_text = "All times displayed are in EDT\n\n"
            else:
                warning_text = ""
        except:
            warning_text = ""
        try:
            warning_text += f"The cone of uncertainty in this product was generated internally using {cone['year']} official\nNHC cone radii. This cone differs slightly from the official NHC cone.\n\n"
        except:
            pass

        self.ax.text(0.99, 0.01, warning_text, fontsize=9, color='k', alpha=0.7,
                     transform=self.ax.transAxes, ha='right', va='bottom', zorder=10)

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(os.path.join(
                save_path, f"{storm_data['name']}_{storm_data['year']}_track.png"), bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_models(self, forecast, plot_btk, storm_dict, forecast_dict, models, domain, ax, prop, map_prop, save_path):
        r"""
        Plot multi-model forecast tracks.
        """

        # Set default properties
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}
        default_model = {'nhc': 'k',
                         'gfs': '#0000ff',
                         'ecm': '#ff1493',
                         'cmc': '#1e90ff',
                         'ukm': '#00ff00',
                         'hmon': '#ff8c00',
                         'hwrf': '#66cdaa',
                         'hafsa': '#C659F9',
                         'hafsb': '#8915BB'}
        default_prop = {'linewidth': 2.5, 'marker': 'label',
                        'marker_hours': [24, 48, 72, 96, 120, 144, 168]}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        model_prop = self.add_prop(models, default_model)
        self.plot_init(ax, map_prop)

        # Fix GFDL
        if 'gfdl' in forecast_dict.keys():
            model_prop['gfdl'] = model_prop['hmon']
        if 'jtwc' in forecast_dict.keys():
            model_prop['jtwc'] = model_prop['nhc']

        # ================================================================================================

        # Keep record of lat/lon coordinate extrema
        lat_max_extrema = []
        lat_min_extrema = []
        lon_max_extrema = []
        lon_min_extrema = []

        # ================================================================================================

        # Plot models
        for model in forecast_dict.keys():

            # Fix label for HAFS
            if 'hafs' in model:
                idx = model.index('hafs')
                model_label = f"HAFS-{model[idx + len('hafs'):].upper()}"
            else:
                model_label = model.upper()

            # Plot forecast track
            lons = forecast_dict[model]['lon']
            lats = forecast_dict[model]['lat']
            self.ax.plot(
                lons, lats, color=model_prop[model], linewidth=prop['linewidth'], label=model_label, transform=ccrs.PlateCarree())

            # Add labels if requested
            if prop['marker'] is not None and len(prop['marker_hours']) >= 1:
                for hour in prop['marker_hours']:
                    if hour not in forecast_dict[model]['fhr']:
                        continue
                    idx = forecast_dict[model]['fhr'].index(hour)
                    if prop['marker'] == 'label':
                        self.ax.text(lons[idx], lats[idx], str(
                            hour), ha='center', va='center', zorder=100, transform=ccrs.PlateCarree())
                    elif prop['marker'] == 'dot':
                        self.ax.plot(lons[idx], lats[idx], 'o', ms=prop['linewidth'] * 3, zorder=100,
                                     mfc=model_prop[model], mec='k', transform=ccrs.PlateCarree())
                    else:
                        raise ValueError(
                            "Acceptable values for 'marker' prop are 'label' or 'dot'.")

            # Add to lat/lon extrema
            lat_max_extrema.append(np.nanmax(lats))
            lat_min_extrema.append(np.nanmin(lats))
            lon_max_extrema.append(np.nanmax(lons))
            lon_min_extrema.append(np.nanmin(lons))

        # Plot best track if requested
        if plot_btk:

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(storm_dict['lon'])
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                use_lons = new_lons.tolist()
            else:
                use_lons = storm_dict['lon']

            # Determine maximum forecast hour
            max_fhr = max([max(forecast_dict[model]['fhr'])
                          for model in forecast_dict.keys()])

            # Determine range of forecast data in best track
            idx_start = storm_dict['time'].index(forecast)
            end_date = forecast + timedelta(hours=max_fhr)
            if end_date in storm_dict['time']:
                idx_end = storm_dict['time'].index(end_date)
            else:
                idx_end = len(storm_dict['time'])

            # Plot best track
            lons = use_lons[idx_start:idx_end + 1]
            lats = storm_dict['lat'][idx_start:idx_end + 1]
            storm_times = storm_dict['time'][idx_start:idx_end + 1]
            self.ax.plot(lons, lats, ':', color='k',
                         linewidth=prop['linewidth'] * 0.8, label='Best Track', transform=ccrs.PlateCarree())

            # Add to lat/lon extrema
            lat_max_extrema.append(np.nanmax(lats))
            lat_min_extrema.append(np.nanmin(lats))
            lon_max_extrema.append(np.nanmax(lons))
            lon_min_extrema.append(np.nanmin(lons))

            # Add labels if requested
            if prop['marker'] is not None and len(prop['marker_hours']) >= 1:
                for hour in prop['marker_hours']:
                    valid_date = forecast + timedelta(hours=hour)
                    if valid_date not in storm_times:
                        continue
                    idx = storm_dict['time'].index(valid_date)
                    if prop['marker'] == 'label':
                        self.ax.text(use_lons[idx], storm_dict['lat'][idx],
                                     str(hour), ha='center', va='center', zorder=100, clip_on=True, transform=ccrs.PlateCarree())
                    elif prop['marker'] == 'dot':
                        self.ax.plot(use_lons[idx], storm_dict['lat'][idx], 'o', ms=prop['linewidth'] * 3, zorder=100,
                                     mfc='k', mec='k', transform=ccrs.PlateCarree())
                    else:
                        raise ValueError(
                            "Acceptable values for 'marker' prop are 'label' or 'dot'.")

        # ================================================================================================

        # Calcuate lat/lon extrema
        lat_max_extrema = np.sort(lat_max_extrema)
        lat_min_extrema = np.sort(lat_min_extrema)
        lon_max_extrema = np.sort(lon_max_extrema)
        lon_min_extrema = np.sort(lon_min_extrema)

        max_lat = np.nanpercentile(lat_max_extrema, 95)
        min_lat = np.nanpercentile(lat_min_extrema, 5)
        max_lon = np.nanpercentile(lon_max_extrema, 95)
        min_lon = np.nanpercentile(lon_min_extrema, 5)

        # ================================================================================================

        # Add legend
        l = self.ax.legend(loc=1, prop={'size': 12})
        l.set_zorder(1001)

        # Plot title
        plot_title = f"Model Forecast Tracks for {storm_dict['name'].title()}"
        self.ax.set_title(plot_title, fontsize=16,
                          loc='left', fontweight='bold')

        title_str = f"Initialized {forecast.strftime('%H%M UTC %d %B %Y')}"
        self.ax.set_title(title_str, fontsize=12, loc='right')

        # --------------------------------------------------------------------------------------

        # Storm-centered plot domain
        if domain == "dynamic":

            bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(os.path.join(save_path), bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_ensembles(self, forecast, storm_dict, fhr, interpolate, prop_ensemble_members, prop_ensemble_mean, prop_gfs, prop_btk, prop_ellipse, prop_density, nens,
                       domain, ds, ax, map_prop, save_path):
        r"""
        Plot GEFS ensemble forecast tracks.
        """

        # Set default properties
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}
        default_prop_ensemble_members = {'plot': True, 'linewidth': 0.2,
                                         'linecolor': 'k', 'color_var': None, 'cmap': None, 'levels': None}
        default_prop_ensemble_mean = {
            'plot': True, 'linewidth': 3.0, 'linecolor': 'k'}
        default_prop_gfs = {'plot': True, 'linewidth': 3.0, 'linecolor': 'r'}
        default_prop_btk = {'plot': True, 'linewidth': 2.5, 'linecolor': 'b'}
        default_prop_ellipse = {'plot': True,
                                'linewidth': 3.0, 'linecolor': 'b'}
        default_prop_density = {'plot': True, 'radius': 200, 'cmap': plt.cm.plasma_r, 'levels': [
            1] + [i for i in range(10, 101, 10)]}

        # Initialize plot
        map_prop = self.add_prop(map_prop, default_map_prop)
        prop_ensemble_members = self.add_prop(
            prop_ensemble_members, default_prop_ensemble_members)
        prop_ensemble_mean = self.add_prop(
            prop_ensemble_mean, default_prop_ensemble_mean)
        prop_gfs = self.add_prop(prop_gfs, default_prop_gfs)
        prop_btk = self.add_prop(prop_btk, default_prop_btk)
        prop_ellipse = self.add_prop(prop_ellipse, default_prop_ellipse)
        prop_density = self.add_prop(prop_density, default_prop_density)
        self.plot_init(ax, map_prop)

        # ================================================================================================

        # Get valid time
        hr = fhr

        # Keep record of lat/lon coordinate extrema
        lat_max_extrema = []
        lat_min_extrema = []
        lon_max_extrema = []
        lon_min_extrema = []

        # ================================================================================================

        # Function for temporal interpolation
        def temporal_interpolation(value, orig_times, target_times):
            f = interp.interp1d(orig_times, value)
            ynew = f(target_times)
            return ynew

        # Plot density
        density_colorbar = False
        if prop_density['plot']:

            # Error check radius
            if prop_density['radius'] > 500 or prop_density['radius'] < 50:
                raise ValueError("Radius must be between 50 and 500 km.")

            if hr is None or (hr is not None and hr in ds['gefs']['fhr']):

                # Create 0.25 degree grid for plotting
                gridlats = np.arange(0, 90, 0.25)
                if np.nanmax(storm_dict['lat']) < 0:
                    gridlats = np.arange(-90, 0, 0.25)
                gridlons = np.arange(-180.0, 180.1, 0.25)
                if self.proj.proj4_params['lon_0'] == 180.0:
                    gridlons = np.arange(0.0, 360.1, 0.25)
                gridlons2d, gridlats2d = np.meshgrid(gridlons, gridlats)
                griddata = np.zeros((gridlons2d.shape))

                # Iterate over all ensemble members
                if hr is None:
                    start_time = dt.now()
                    print("--> Starting to calculate track density")
                for ens in range(nens):

                    # Calculate for one hour
                    if hr is not None:

                        # Proceed if hour is available
                        if hr in ds[f'gefs_{ens}']['fhr']:
                            idx = ds[f'gefs_{ens}']['fhr'].index(hr)
                            griddata += add_radius_quick(gridlats, gridlons, ds[f'gefs_{ens}']['lat'][idx],
                                                         ds[f'gefs_{ens}']['lon'][idx], prop_density['radius'])

                    # Calculate for cumulative
                    else:

                        # Ensemble temporary gridded field
                        temp_grid = np.zeros((gridlons2d.shape))

                        # Interpolate temporally to hourly if requested
                        if interpolate:
                            if len(ds[f'gefs_{ens}']['lat']) == 0:
                                continue
                            elif len(ds[f'gefs_{ens}']['lat']) == 1:
                                new_lats = ds[f'gefs_{ens}']['lat']
                                new_lons = ds[f'gefs_{ens}']['lon']
                            else:
                                new_hours = np.arange(min(ds[f'gefs_{ens}']['fhr']), max(
                                    ds[f'gefs_{ens}']['fhr']), 1)
                                new_lats = temporal_interpolation(
                                    ds[f'gefs_{ens}']['lat'], ds[f'gefs_{ens}']['fhr'], new_hours)
                                new_lons = temporal_interpolation(
                                    ds[f'gefs_{ens}']['lon'], ds[f'gefs_{ens}']['fhr'], new_hours)

                            # Iterate over all forecast hours
                            for i, (i_lon, i_lat) in enumerate(zip(new_lons, new_lats)):
                                radius_grid = add_radius_quick(
                                    gridlats, gridlons, i_lat, i_lon, prop_density['radius'])
                                temp_grid = np.maximum(temp_grid, radius_grid)

                        # Otherwise don't interpolate
                        else:

                            # Iterate over all forecast hours
                            for idx, iter_hr in enumerate(ds[f'gefs_{ens}']['fhr']):
                                radius_grid = add_radius(gridlats2d, gridlons2d, ds[f'gefs_{ens}']['lat'][idx],
                                                         ds[f'gefs_{ens}']['lon'][idx], prop_density['radius'])
                                temp_grid = np.maximum(temp_grid, radius_grid)

                        # Add temporary grid to full grid
                        griddata += temp_grid

                # Convert density to percent
                if hr is None:
                    time_elapsed = dt.now() - start_time
                    tsec = str(round(time_elapsed.total_seconds(), 2))
                    print(
                        f"--> Completed calculating track density ({tsec} seconds)")
                density_percent = (griddata / nens) * 100.0

                # Plot density
                norm = mcolors.BoundaryNorm(
                    prop_density['levels'], prop_density['cmap'].N)
                cs = self.ax.contourf(gridlons, gridlats, density_percent, prop_density['levels'],
                                      cmap=prop_density['cmap'], norm=norm, alpha=0.6, transform=ccrs.PlateCarree())
                cbar = add_colorbar(
                    cs, ticks=prop_density['levels'], ax=self.ax)
                cbar.ax.tick_params(labelsize=12)
                density_colorbar = True

        # -------------------------------------------------------------------
        # Plot ellipse
        if hr is not None and hr in ds['gefs']['fhr'] and prop_ellipse['plot']:
            idx = ds['gefs']['fhr'].index(hr)

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(ds['gefs']['ellipse_lon'][idx])
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                new_lons = new_lons.tolist()
            else:
                new_lons = ds['gefs']['ellipse_lon'][idx]

            try:
                self.ax.plot(new_lons, ds['gefs']['ellipse_lat'][idx], '-',
                             color='w', linewidth=prop_ellipse['linewidth'] * 1.2,
                             transform=ccrs.PlateCarree(), alpha=0.8)
                self.ax.plot(new_lons, ds['gefs']['ellipse_lat'][idx], '-',
                             color=prop_ellipse['linecolor'], linewidth=prop_ellipse['linewidth'],
                             transform=ccrs.PlateCarree(), alpha=0.8)
            except:
                pass

        # -------------------------------------------------------------------
        # Plot GEFS member tracks
        for i in range(nens):

            # Update coordinate bounds
            skip_bounds = False
            if hr in ds[f'gefs_{i}']['fhr']:
                idx = ds[f'gefs_{i}']['fhr'].index(hr)
                use_lats = ds[f'gefs_{i}']['lat'][:idx + 1]
                use_lons = ds[f'gefs_{i}']['lon'][:idx + 1]
            elif len(ds[f'gefs_{i}']['fhr']) > 0:
                idx = 0
                if hr is None:
                    idx = len(ds[f'gefs_{i}']['lon'])
                else:
                    for idx_hr in ds[f'gefs_{i}']['fhr']:
                        if idx_hr <= hr:
                            idx = ds[f'gefs_{i}']['fhr'].index(idx_hr)
                use_lons = ds[f'gefs_{i}']['lon'][:idx + 1]
                use_lats = ds[f'gefs_{i}']['lat'][:idx + 1]
            else:
                skip_bounds = True

            if not skip_bounds:
                lat_max_extrema.append(np.nanmax(use_lats))
                lat_min_extrema.append(np.nanmin(use_lats))
                lon_max_extrema.append(np.nanmax(use_lons))
                lon_min_extrema.append(np.nanmin(use_lons))

                # Plot cumulative track
                if len(ds[f'gefs_{i}']['fhr']) > 0:
                    if prop_ensemble_members['color_var'] not in ['vmax', 'mslp']:
                        self.ax.plot(ds[f'gefs_{i}']['lon'][:idx + 1], ds[f'gefs_{i}']['lat'][:idx + 1],
                                     linewidth=prop_ensemble_members['linewidth'],
                                     color=prop_ensemble_members['linecolor'], transform=ccrs.PlateCarree())
                    else:
                        # Color by variable
                        cmap = prop_ensemble_members['cmap']
                        levels = prop_ensemble_members['levels']
                        norm = mcolors.BoundaryNorm(levels, cmap.N)
                        for j in range(1, idx + 1):
                            if j >= len(ds[f'gefs_{i}'][prop_ensemble_members['color_var']]):
                                continue
                            i_val = ds[f'gefs_{i}'][prop_ensemble_members['color_var']][j]
                            color = 'w' if np.isnan(
                                i_val) else cmap(norm(i_val))
                            self.ax.plot([ds[f'gefs_{i}']['lon'][j - 1], ds[f'gefs_{i}']['lon'][j]],
                                         [ds[f'gefs_{i}']['lat'][j - 1],
                                             ds[f'gefs_{i}']['lat'][j]],
                                         linewidth=prop_ensemble_members['linewidth'],
                                         color=color, transform=ccrs.PlateCarree())

                        # Add colorbar
                        if not density_colorbar:
                            cs = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                            cs.set_array([])
                            cbar = add_colorbar(
                                cs, ticks=prop_ensemble_members['levels'], ax=self.ax)
                            cbar.ax.tick_params(labelsize=12)
                            density_colorbar = True

                # Plot latest dot if applicable
                if hr in ds[f'gefs_{i}']['fhr']:
                    idx = ds[f'gefs_{i}']['fhr'].index(hr)
                    self.ax.plot(ds[f'gefs_{i}']['lon'][idx], ds[f'gefs_{i}']['lat'][idx], 'o', ms=4,
                                 mfc=prop_ensemble_members['linecolor'], mec='k',
                                 alpha=0.6, transform=ccrs.PlateCarree())

        # -------------------------------------------------------------------
        # Plot best track
        if prop_btk['plot']:

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(storm_dict['lon'])
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                new_lons = new_lons.tolist()
            else:
                new_lons = storm_dict['lon']

            # Get valid time
            valid_time = np.nan if hr is None else forecast + timedelta(hours=hr)
            end_time = forecast + timedelta(hours=240) if storm_dict['year'] >= 2015 else forecast + timedelta(hours=240)

            # Update coordinate bounds
            skip_bounds = False
            idx_start = storm_dict['time'].index(
                forecast) if forecast in storm_dict['time'] else 0
            if valid_time in storm_dict['time']:
                idx = storm_dict['time'].index(valid_time)

                use_lats = storm_dict['lat'][idx_start:idx + 1]
                use_lons = new_lons[idx_start:idx + 1]
            else:
                idx = 0
                if hr is None:
                    if end_time in storm_dict['time']:
                        idx = storm_dict['time'].index(end_time)
                    else:
                        idx = len(storm_dict['lat'])
                else:
                    for idx_date in storm_dict['time']:
                        if idx_date <= valid_time:
                            idx = storm_dict['time'].index(idx_date)
                use_lats = storm_dict['lat'][idx_start:idx + 1]
                use_lons = new_lons[idx_start:idx + 1]

            if not skip_bounds:
                lat_max_extrema.append(np.nanmax(use_lats))
                lat_min_extrema.append(np.nanmin(use_lats))
                lon_max_extrema.append(np.nanmax(use_lons))
                lon_min_extrema.append(np.nanmin(use_lons))

            # Plot observed track before now
            self.ax.plot(new_lons[:idx_start + 1], storm_dict['lat'][:idx_start + 1],
                         linewidth=1.5, color='k', linestyle=":", transform=ccrs.PlateCarree())

            if valid_time in storm_dict['time']:
                idx = storm_dict['time'].index(valid_time)
                self.ax.plot(new_lons[idx_start:idx + 1], storm_dict['lat'][idx_start:idx + 1],
                             linewidth=prop_btk['linewidth'], color=prop_btk['linecolor'], transform=ccrs.PlateCarree())
                self.ax.plot(new_lons[idx], storm_dict['lat'][idx], 'o', ms=12,
                             mfc=prop_btk['linecolor'], mec='k', transform=ccrs.PlateCarree())
            elif len(storm_dict['time']) > 0:
                idx = 0
                if hr is None:
                    if end_time in storm_dict['time']:
                        idx = storm_dict['time'].index(end_time)
                    else:
                        idx = len(storm_dict['lat'])
                else:
                    for idx_date in storm_dict['time']:
                        if idx_date <= valid_time:
                            idx = storm_dict['time'].index(idx_date)
                self.ax.plot(new_lons[idx_start:idx + 1], storm_dict['lat'][idx_start:idx + 1],
                             linewidth=prop_btk['linewidth'], color=prop_btk['linecolor'], transform=ccrs.PlateCarree())

        # -------------------------------------------------------------------
        # Plot operational GFS track
        if prop_gfs['plot']:

            # Update coordinate bounds
            skip_bounds = False
            if hr in ds['gfs']['fhr']:
                idx = ds['gfs']['fhr'].index(hr)
                use_lats = ds['gfs']['lat'][:idx + 1]
                use_lons = ds['gfs']['lon'][:idx + 1]
            elif len(ds['gfs']['fhr']) > 0:
                idx = 0
                if hr is None:
                    idx = len(ds['gfs']['lon'])
                else:
                    for idx_hr in ds['gfs']['fhr']:
                        if idx_hr <= hr:
                            idx = ds['gfs']['fhr'].index(idx_hr)
                use_lats = ds['gfs']['lat'][:idx + 1]
                use_lons = ds['gfs']['lon'][:idx + 1]
            else:
                skip_bounds = True

            if not skip_bounds:
                lat_max_extrema.append(np.nanmax(use_lats))
                lat_min_extrema.append(np.nanmin(use_lats))
                lon_max_extrema.append(np.nanmax(use_lons))
                lon_min_extrema.append(np.nanmin(use_lons))

            # Plot GFS forecast line and latest dot
            if hr in ds['gfs']['fhr']:
                idx = ds['gfs']['fhr'].index(hr)
                self.ax.plot(ds['gfs']['lon'][:idx + 1], ds['gfs']['lat'][:idx + 1],
                             linewidth=prop_gfs['linewidth'], color=prop_gfs['linecolor'], transform=ccrs.PlateCarree())
                self.ax.plot(ds['gfs']['lon'][idx], ds['gfs']['lat'][idx], 'o', ms=12,
                             mfc=prop_gfs['linecolor'], mec='k', transform=ccrs.PlateCarree())

            elif len(ds['gfs']['fhr']) > 0:
                idx = 0
                if hr is None:
                    idx = len(ds['gfs']['lon'])
                else:
                    for idx_hr in ds['gfs']['fhr']:
                        if idx_hr <= hr:
                            idx = ds['gfs']['fhr'].index(idx_hr)
                self.ax.plot(ds['gfs']['lon'][:idx + 1], ds['gfs']['lat'][:idx + 1],
                             linewidth=prop_gfs['linewidth'], color=prop_gfs['linecolor'], transform=ccrs.PlateCarree())

        # -------------------------------------------------------------------
        # Plot ensemble mean track
        if prop_ensemble_mean['plot']:

            # Update coordinate bounds
            skip_bounds = False
            if hr in ds['gefs']['fhr']:
                idx = ds['gefs']['fhr'].index(hr)
                use_lats = ds['gefs']['lat'][:idx + 1]
                use_lons = ds['gefs']['lon'][:idx + 1]
            elif len(ds['gefs']['fhr']) > 0:
                idx = 0
                if hr is None:
                    idx = len(ds['gefs']['lon'])
                else:
                    for idx_hr in ds['gefs']['fhr']:
                        if idx_hr <= hr:
                            idx = ds['gefs']['fhr'].index(idx_hr)
                use_lats = ds['gefs']['lat'][:idx + 1]
                use_lons = ds['gefs']['lon'][:idx + 1]
            else:
                skip_bounds = True

            if not skip_bounds:
                lat_max_extrema.append(np.nanmax(use_lats))
                lat_min_extrema.append(np.nanmin(use_lats))
                lon_max_extrema.append(np.nanmax(use_lons))
                lon_min_extrema.append(np.nanmin(use_lons))

            if hr in ds['gefs']['fhr']:
                idx = ds['gefs']['fhr'].index(hr)
                self.ax.plot(ds['gefs']['lon'][:idx + 1], ds['gefs']['lat'][:idx + 1],
                             linewidth=prop_ensemble_mean['linewidth'],
                             color=prop_ensemble_mean['linecolor'], transform=ccrs.PlateCarree())
                self.ax.plot(ds['gefs']['lon'][idx], ds['gefs']['lat'][idx], 'o', ms=12,
                             mfc=prop_ensemble_mean['linecolor'], mec='k', transform=ccrs.PlateCarree())
            elif len(ds['gefs']['fhr']) > 0:
                idx = 0
                if hr is None:
                    idx = len(ds['gefs']['lon'])
                else:
                    for idx_hr in ds['gefs']['fhr']:
                        if idx_hr <= hr:
                            idx = ds['gefs']['fhr'].index(idx_hr)
                self.ax.plot(ds['gefs']['lon'][:idx + 1], ds['gefs']['lat'][:idx + 1],
                             linewidth=prop_ensemble_mean['linewidth'],
                             color=prop_ensemble_mean['linecolor'], transform=ccrs.PlateCarree())

        # ================================================================================================

        # Calcuate lat/lon extrema
        lat_max_extrema = np.sort(lat_max_extrema)
        lat_min_extrema = np.sort(lat_min_extrema)
        lon_max_extrema = np.sort(lon_max_extrema)
        lon_min_extrema = np.sort(lon_min_extrema)

        if hr is None:
            max_lat = np.nanpercentile(lat_max_extrema, 95)
            min_lat = np.nanpercentile(lat_min_extrema, 5)
            max_lon = np.nanpercentile(lon_max_extrema, 95)
            min_lon = np.nanpercentile(lon_min_extrema, 5)
        else:
            max_lat = np.nanmax(lat_max_extrema)
            min_lat = np.nanmin(lat_min_extrema)
            max_lon = np.nanmax(lon_max_extrema)
            min_lon = np.nanmin(lon_min_extrema)

        # ================================================================================================

        # Add legend
        p1 = mlines.Line2D([], [], color=prop_btk['linecolor'],
                           linewidth=prop_btk['linewidth'], label='Best Track')
        p2 = mlines.Line2D([], [], color=prop_gfs['linecolor'],
                           linewidth=prop_gfs['linewidth'], label='Deterministic GFS')
        p3 = mlines.Line2D([], [], color=prop_ensemble_mean['linecolor'],
                           linewidth=prop_ensemble_mean['linewidth'], label='GEFS Mean')
        p4 = mlines.Line2D([], [], color=prop_ensemble_members['linecolor'],
                           linewidth=prop_ensemble_members['linewidth'], label='GEFS Members')
        p5 = mlines.Line2D([], [], color='w', marker='o', ms=12, mec=prop_ellipse['linecolor'],
                           mew=prop_ellipse['linewidth'], label='GEFS Ellipse')
        handles_list = []
        if prop_btk['plot']:
            handles_list.append(p1)
        if prop_gfs['plot']:
            handles_list.append(p2)
        if prop_ensemble_mean['plot']:
            handles_list.append(p3)
        if prop_ensemble_members['plot']:
            handles_list.append(p4)
        if hr is not None and prop_ellipse['plot']:
            handles_list.append(p5)
        if len(handles_list) > 0:
            l = self.ax.legend(handles=handles_list, loc=1, prop={'size': 12})
            l.set_zorder(1001)

        # Plot title
        format_title = {
            'vmax': 'Ensemble member sustained wind (knots)', 'mslp': 'Ensemble member minimum MSLP (hPa)'}
        plot_title = f"GEFS Forecast Tracks for {storm_dict['name'].title()}"
        if prop_density['plot']:
            plot_title += f"\nTrack Density ({int(prop_density['radius'])}-km radius)"
        if prop_ensemble_members['color_var'] in ['vmax', 'mslp']:
            plot_title += f"\n{format_title.get(prop_ensemble_members['color_var'])}"
        self.ax.set_title(plot_title, fontsize=16,
                          loc='left', fontweight='bold')

        if hr is None:
            title_str = f"Initialized {forecast.strftime('%H%M UTC %d %B %Y')}"
        else:
            title_str = f"Hour {hr} | Valid {(forecast+timedelta(hours=hr)).strftime('%H%M UTC %d %B %Y')}\n"
            title_str += f"Initialized {forecast.strftime('%H%M UTC %d %B %Y')}"
        self.ax.set_title(title_str, fontsize=12, loc='right')

        # --------------------------------------------------------------------------------------

        # Storm-centered plot domain
        if domain == "dynamic":

            bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(os.path.join(save_path), bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_season(self, season, domain=None, ax=None, save_path=None, prop={}, map_prop={}):
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

        # Set default properties
        default_prop = {'dots': False, 'fillcolor': 'category', 'cmap': None, 'levels': None,
                        'linecolor': 'category', 'linewidth': 1.0, 'ms': 7.5, 'plot_names': True}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # --------------------------------------------------------------------------------------

        # Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        # Iterate over all storms in season object
        sinfo = season.summary()
        storms = season.dict.keys()
        for storm_idx, storm_key in enumerate(storms):

            # Get data for this storm
            storm = season.dict[storm_key]

            # Retrieve storm data
            lats = storm['lat']
            lons = storm['lon']
            vmax = storm['vmax']
            styp = storm['type']
            sdate = storm['time']

            # Account for cases crossing dateline
            if self.proj.proj4_params['lon_0'] == 180.0:
                new_lons = np.array(lons)
                new_lons[new_lons < 0] = new_lons[new_lons < 0] + 360.0
                lons = new_lons.tolist()

            # Add to coordinate extrema
            if max_lat is None:
                max_lat = max(lats)
            else:
                if max(lats) > max_lat:
                    max_lat = max(lats)
            if min_lat is None:
                min_lat = min(lats)
            else:
                if min(lats) < min_lat:
                    min_lat = min(lats)
            if max_lon is None:
                max_lon = max(lons)
            else:
                if max(lons) > max_lon:
                    max_lon = max(lons)
            if min_lon is None:
                min_lon = min(lons)
            else:
                if min(lons) < min_lon:
                    min_lon = min(lons)

            # Add storm label at start and end points
            if prop['plot_names']:
                self.ax.text(lons[0] + 0.0, storm['lat'][0] + 1.0, storm['name'].upper(),
                             fontsize=9, clip_on=True, zorder=1000, alpha=0.7, ha='center', va='center', transform=ccrs.PlateCarree())
                self.ax.text(lons[-1] + 0.0, storm['lat'][-1] + 1.0, storm['name'].upper(),
                             fontsize=9, clip_on=True, zorder=1000, alpha=0.7, ha='center', va='center', transform=ccrs.PlateCarree())

            # Iterate over storm data to plot
            levels = None
            cmap = None
            for i, (i_lat, i_lon, i_vmax, i_mslp, i_time, i_type) in enumerate(zip(storm['lat'], lons, storm['vmax'], storm['mslp'], storm['time'], storm['type'])):

                # Determine line color, with SSHWS scale used as default
                if prop['linecolor'] == 'category':
                    segmented_colors = True
                    line_color = get_colors_sshws(np.nan_to_num(i_vmax))

                # Use user-defined colormap if another storm variable
                elif isinstance(prop['linecolor'], str) and prop['linecolor'] in ['vmax', 'mslp', 'dvmax_dt', 'speed']:
                    segmented_colors = True
                    try:
                        color_variable = storm[prop['linecolor']]
                    except:
                        raise ValueError(
                            "Storm object must be interpolated to hourly using 'storm.interp().plot(...)' in order to use 'dvmax_dt' or 'speed' for fill color")
                    if prop['levels'] is None:  # Auto-determine color levels if needed
                        prop['levels'] = [
                            np.nanmin(color_variable), np.nanmax(color_variable)]
                    cmap, levels = get_cmap_levels(
                        prop['linecolor'], prop['cmap'], prop['levels'])
                    line_color = cmap(
                        (color_variable - min(levels)) / (max(levels) - min(levels)))[i]

                # Otherwise go with user input as is
                else:
                    segmented_colors = False
                    line_color = prop['linecolor']

                # For tropical/subtropical types, color-code if requested
                if i > 0:
                    if i_type in constants.TROPICAL_STORM_TYPES and storm['type'][i - 1] in constants.TROPICAL_STORM_TYPES:

                        # Plot underlying black and overlying colored line
                        self.ax.plot([lons[i - 1], lons[i]], [storm['lat'][i - 1], storm['lat'][i]], '-',
                                     linewidth=prop['linewidth'] * 1.33, color='k', zorder=storm_idx * 5,
                                     transform=ccrs.PlateCarree())
                        self.ax.plot([lons[i - 1], lons[i]], [storm['lat'][i - 1], storm['lat'][i]], '-',
                                     linewidth=prop['linewidth'], color=line_color, zorder=i_vmax + (
                                         storm_idx * 5),
                                     transform=ccrs.PlateCarree())

                    # For non-tropical types, plot dotted lines
                    else:

                        # Restrict line width to 1.5 max
                        line_width = prop['linewidth'] + 0.0
                        if line_width > 1.5:
                            line_width = 1.5

                        # Plot dotted line
                        self.ax.plot([lons[i - 1], lons[i]], [storm['lat'][i - 1], storm['lat'][i]], ':',
                                     linewidth=line_width, color=line_color, zorder=i_vmax +
                                     (storm_idx * 5),
                                     transform=ccrs.PlateCarree(),
                                     path_effects=[path_effects.Stroke(linewidth=line_width * 1.33, foreground='k'),
                                                   path_effects.Normal()])

                # Plot dots if requested
                if prop['dots']:
                    self.ax, segmented_colors, extra = plot_dot(self.ax, i_lon, i_lat, i_time, i_vmax, i_type,
                                                                zorder=900 + i_vmax, storm_data=storm, prop=prop, i=i)
                    if 'cmap' in extra.keys():
                        cmap = extra['cmap']
                    if 'levels' in extra.keys():
                        levels = extra['levels']

        # --------------------------------------------------------------------------------------

        # Pre-generated domains
        if domain is None:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(
                season.basin)
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        # Add left title
        emdash = u"\u2014"
        basin_name = ((season.basin).replace("_", " ")).title()
        if season.basin == 'all':
            season_title = f"{season.year} Global Tropical Cyclone Season"
        elif season.basin == 'both':
            season_title = f"{season.year} Atlantic-Pacific Hurricane Season"
        elif season.basin in ['south_indian', 'south_atlantic', 'australia', 'south_pacific']:
            season_title = f"{season.year-1}{emdash}{season.year} {basin_name} Tropical Cyclone Season"
        elif season.basin in ['west_pacific']:
            season_title = f"{season.year} {basin_name.split(' ')[1]} Typhoon Season"
        else:
            season_title = f"{season.year} {basin_name.split(' ')[1]} Hurricane Season"
        self.ax.set_title(season_title, loc='left',
                          fontsize=17, fontweight='bold')

        # Add right title
        endash = u"\u2013"
        dot = u"\u2022"
        count_named = sinfo['season_named']
        count_hurricane = sinfo['season_hurricane']
        count_major = sinfo['season_major']
        count_ace = sinfo['season_ace']
        if isinstance(season.year, list):
            count_named = np.sum(sinfo['season_named'])
            count_hurricane = np.sum(sinfo['season_hurricane'])
            count_major = np.sum(sinfo['season_major'])
            count_ace = np.sum(sinfo['season_ace'])
        self.ax.set_title(
            f"{count_named} named {dot} {count_hurricane} hurricanes {dot} {count_major} major\n{count_ace:.1f} Cumulative ACE", loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add plot credit
        warning_text = ""
        if storm['source'] == 'ibtracs' and storm['source_info'] == 'World Meteorological Organization (official)':
            warning_text = f"This plot uses 10-minute averaged WMO official wind data converted\nto 1-minute average (factor of 0.88). Use this wind data with caution.\n\n"

            self.ax.text(0.99, 0.01, warning_text, fontsize=9, color='k', alpha=0.7,
                         transform=self.ax.transAxes, ha='right', va='bottom', zorder=10)

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # --------------------------------------------------------------------------------------

        # Add legend
        self.ax, self.fig = add_legend(
            self.ax, self.fig, prop, segmented_colors, levels, cmap, storm)

        # --------------------------------------------------------------------------------------

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_track_labels(self, ax, labels, track, k=0.01):

        label_nodes = list(labels.keys())
        labels['place1'] = (2 * labels[label_nodes[0]][0] - labels[label_nodes[1]][0],
                            2 * labels[label_nodes[0]][1] - labels[label_nodes[1]][1])
        labels['place2'] = (2 * labels[label_nodes[-1]][0] - labels[label_nodes[-2]][0],
                            2 * labels[label_nodes[-1]][1] - labels[label_nodes[-2]][1])
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
            G.add_edge(lab, 'track_{0}'.format(lab))
            init_pos[lab] = labels[lab]

        pos = nx.spring_layout(G, pos=init_pos, fixed=track_nodes, k=k)

        # undo spring_layout's rescaling
        pos_after = np.vstack([pos[d] for d in track_nodes])
        pos_before = np.vstack([init_pos[d] for d in track_nodes])
        scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
        scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val * scale) + shift

        for label, _ in G.edges():
            if 'place' not in label:
                self.ax.annotate(label,
                                 xy=init_pos[label], xycoords='data',
                                 xytext=pos[label], textcoords='data', fontweight='bold', ha='center', va='center',
                                 arrowprops=dict(arrowstyle="-",  # ->
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
        scale, shift_x = np.polyfit(pos_after[:, 0], pos_before[:, 0], 1)
        scale, shift_y = np.polyfit(pos_after[:, 1], pos_before[:, 1], 1)
        shift = np.array([shift_x, shift_y])
        for key, val in pos.items():
            pos[key] = (val * scale) + shift

        # Apply coordinate transform
        transform = ccrs.PlateCarree()._as_mpl_transform(self.ax)

        start = False
        for label, data_str in G.edges():
            if not start:
                start = True
                continue
            self.ax.annotate(label,  # xycoords="data"
                             xy=pos[data_str], xycoords=transform,
                             xytext=pos[label], textcoords=transform, fontweight='bold', ha='center', va='center',
                             arrowprops=dict(arrowstyle="-",  # ->
                                             shrinkA=0, shrinkB=0,
                                             connectionstyle="arc3",
                                             color='k'),
                             transform=ccrs.PlateCarree(), clip_on=True)

    def plot_gridded(self, xcoord, ycoord, zcoord, varname='type', VEC_FLAG=False, domain="north_atlantic", ax=None, prop={}, map_prop={}):
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

        # Set default properties
        default_prop = {'cmap': 'category', 'levels': None,
                        'left_title': '', 'right_title': 'All storms',
                        'plot_values': False, 'values_size': None}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # Determine if contour levels are automatically generated
        auto_levels = True if prop['levels'] is None or prop['levels'] == [
        ] else False

        # Plot domain
        bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        if VEC_FLAG:
            vecmag = np.hypot(*zcoord)
            if prop['levels'] is None:
                prop['levels'] = (np.nanmin(vecmag), np.nanmax(vecmag))
        elif prop['levels'] is None:
            prop['levels'] = (np.nanmin(zcoord), np.nanmax(zcoord))
        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['levels'])

        # Generate contourf levels
        if len(clevs) == 2:
            y0 = min(clevs)
            y1 = max(clevs)
            dy = (y1 - y0) / 8
            scalemag = int(np.log(dy) / np.log(10))
            dy_scaled = dy * 10**-scalemag
            dc = min([1, 2, 5, 10], key=lambda x: abs(x - dy_scaled))
            c0 = np.ceil(y0 / dc * 10**-scalemag) * dc * 10**scalemag
            c1 = np.floor(y1 / dc * 10**-scalemag) * dc * 10**scalemag
            clevs = np.arange(c0, c1 + dc, dc)

        if varname == 'vmax' and prop['cmap'] == 'category':
            vmin = min(clevs)
            vmax = max(clevs)
        else:
            vmin = min(prop['levels'])
            vmax = max(prop['levels'])

        # For difference/change plots with automatically generated contour levels, ensure that 0 is in the middle
        if auto_levels:
            if varname in ['dvmax_dt', 'dmslp_dt'] or '\n' in prop['title_R']:
                max_val = np.max([np.abs(vmin), vmax])
                vmin = np.round(max_val * -1.0, 2)
                vmax = np.round(max_val * 1.0, 2)
                clevs = [vmin, np.round(vmin * 0.5, 2),
                         0, np.round(vmax * 0.5, 2), vmax]

        if len(xcoord.shape) and len(ycoord.shape) == 1:
            xcoord, ycoord = np.meshgrid(xcoord, ycoord)

        if VEC_FLAG:
            binsize = abs(xcoord[0, 0] - xcoord[0, 1])
            cbmap = self.ax.pcolor(xcoord, ycoord, vecmag[:-1, :-1], cmap=cmap, vmin=min(clevs), vmax=max(clevs),
                                   transform=ccrs.PlateCarree())
            zcoord = zcoord / vecmag * binsize
            x_center = (xcoord[:-1, :-1] + xcoord[1:, 1:]) * .5
            y_center = (ycoord[:-1, :-1] + ycoord[1:, 1:]) * .5
            u = zcoord[0][:-1, :-1]
            v = zcoord[1][:-1, :-1]
            if not prop['plot_values']:
                self.ax.quiver(x_center, y_center, u, v, color='w', alpha=0.6, transform=ccrs.PlateCarree(),
                               pivot='mid', width=.001 * binsize, headwidth=3.5, headlength=4.5, headaxislength=4)
            zcoord = vecmag

        else:
            print('--> Generating plot')
            # if varname=='date' and prop['smooth'] is not None:
            #    zcoord[np.isnan(zcoord)]=0
            #    zcoord=gfilt(zcoord,sigma=prop['smooth'])
            #    zcoord[zcoord<min(clevs)]=np.nan

            if prop['cmap'] == 'category' and varname == 'vmax':
                norm = mcolors.BoundaryNorm(clevs, cmap.N)
                cbmap = self.ax.pcolor(xcoord, ycoord, zcoord[:-1, :-1], cmap=cmap, norm=norm,
                                       transform=ccrs.PlateCarree())
            else:
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                cbmap = self.ax.pcolor(xcoord, ycoord, zcoord[:-1, :-1], cmap=cmap, norm=norm,
                                       transform=ccrs.PlateCarree())
        if prop['plot_values']:
            binsize = abs(xcoord[0, 0] - xcoord[0, 1])
            x_center = (xcoord[:-1, :-1] + xcoord[1:, 1:]) * .5
            y_center = (ycoord[:-1, :-1] + ycoord[1:, 1:]) * .5
            xs = x_center.flatten(order='C')
            ys = y_center.flatten(order='C')
            zs = zcoord[:-1, :-1].flatten(order='C')
            if prop['values_size'] is None:
                fs = binsize * 4
            else:
                fs = prop['values_size']
            for xtext, ytext, ztext in zip(xs, ys, zs):
                if not np.isnan(ztext) and xtext % 360 > bound_w % 360 and xtext % 360 < bound_e % 360 and\
                        ytext > bound_s and ytext < bound_n:
                    square_color = cmap(norm(ztext))
                    square_brightness = np.mean(
                        square_color[:3]) * square_color[3]
                    text_color = 'k' if square_brightness > 0.5 else 'w'
                    self.ax.text(xtext, ytext, ztext.astype(int), ha='center', va='center', fontsize=fs,
                                 color=text_color, alpha=0.8, transform=ccrs.PlateCarree(), zorder=2)

        # --------------------------------------------------------------------------------------

        # Phantom legend
        handles = []
        for _ in range(10):
            handles.append(mlines.Line2D(
                [], [], linestyle='-', label='', lw=0))
        l = self.ax.legend(handles=handles, loc='upper left',
                           fancybox=True, framealpha=0, fontsize=11.5)
        plt.draw()

        # Get the bbox
        try:
            bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
        except:
            bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())
        bb_ax = self.ax.get_position()

        # Define colorbar axis
        cax = self.fig.add_axes(
            [bb.x0 + 1.2 * bb.width, bb.y0 - .05 * bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical',
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
        if prop['cmap'] == 'category' and varname == 'vmax':
            cax.yaxis.set_ticks(np.linspace(
                min(clevs), max(clevs), len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0, 1, len(clevs))[
                                 :-1] + np.linspace(0, 1, len(clevs))[1:]) * .5)
            cax2.set_yticklabels(
                ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')

            rect_offset = 0.7
        if varname == 'date':
            cax.set_yticklabels(
                [f'{mdates.num2date(i):%b %-d}' for i in clevs], fontsize=11.5)

        rectangle = mpatches.Rectangle((bb.x0, bb.y0 - 0.1 * bb.height), (2 + rect_offset) * bb.width, 1.1 * bb.height,
                                       fc='w', edgecolor='0.8', alpha=0.8,
                                       transform=self.fig.transFigure, zorder=3)
        self.ax.add_patch(rectangle)

        # --------------------------------------------------------------------------------------

        # Add left title
        try:
            self.ax.set_title(prop['title_L'], loc='left',
                              fontsize=17, fontweight='bold')
        except:
            pass

        # Add right title
        try:
            self.ax.set_title(prop['title_R'], loc='right', fontsize=15)
        except:
            pass

        # --------------------------------------------------------------------------------------

        # Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        # Return axis if specified, otherwise display figure
        return self.ax

    def plot_summary(self, storms, forecasts, shapefiles, valid_date, domain, ax=None, save_path=None, two_prop={}, invest_prop={}, storm_prop={}, cone_prop={}, map_prop={}):
        r"""
        Creates a realtime summary plot.
        """

        # Set default properties
        default_two_prop = {'plot': True, 'fontsize': 12, 'days': 7, 'ms': 15}
        default_invest_prop = {'plot': True, 'fontsize': 12,
                               'linewidth': 0.8, 'linecolor': 'k', 'linestyle': 'dotted', 'ms': 14}
        default_storm_prop = {'plot': True, 'fontsize': 12, 'linewidth': 0.8, 'linecolor': 'k',
                              'linestyle': 'dotted', 'fillcolor': 'category', 'label_category': True, 'ms': 14}
        default_cone_prop = {'plot': True, 'linewidth': 1.5, 'linecolor': 'k', 'alpha': 0.6,
                             'days': 5, 'fillcolor': 'category', 'label_category': True, 'ms': 12}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}
        if domain == 'all':
            default_map_prop['res'] = 'l'

        # Initialize plot
        two_prop = self.add_prop(two_prop, default_two_prop)
        invest_prop = self.add_prop(invest_prop, default_invest_prop)
        storm_prop = self.add_prop(storm_prop, default_storm_prop)
        cone_prop = self.add_prop(cone_prop, default_cone_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # Plot domain
        bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Format title
        add_title = ""
        if two_prop['plot']:
            if two_prop['days'] == 2 or valid_date <= dt(2014, 7, 1):
                add_title = " & NHC 2-Day Formation Outlook"
            elif valid_date <= dt(2023, 5, 1):
                add_title = " & NHC 5-Day Formation Outlook"
            else:
                add_title = " & NHC 7-Day Formation Outlook"

        # --------------------------------------------------------------------------------------

        bbox_prop = {'facecolor': 'white', 'alpha': 0.5,
                     'edgecolor': 'black', 'boxstyle': 'round,pad=0.3'}

        if two_prop['plot']:

            # Store color
            color_base = {'Low': 'yellow', 'Medium': 'orange', 'High': 'red'}

            # Plot areas
            if shapefiles['areas'] is not None:
                for record, geom in zip(shapefiles['areas'].records(), shapefiles['areas'].geometries()):

                    # Read relevant data
                    if 'RISK2DAY' in record.attributes.keys() or 'RISK5DAY' in record.attributes.keys() or 'RISK7DAY' in record.attributes.keys():
                        if two_prop['days'] == 2:
                            color = color_base.get(
                                record.attributes['RISK2DAY'], 'yellow')
                        elif 'RISK5DAY' in record.attributes.keys():
                            color = color_base.get(
                                record.attributes['RISK5DAY'], 'yellow')
                        else:
                            color = color_base.get(
                                record.attributes['RISK7DAY'], 'yellow')
                    else:
                        color = color_base.get(
                            record.attributes['GENCAT'], 'yellow')

                    # Plot area
                    self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                        facecolor=color, edgecolor=color, alpha=0.3, linewidth=1.5, zorder=3)

                    # Plot hatching
                    self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                        facecolor='none', edgecolor='k', linewidth=2.25, zorder=4)
                    self.ax.add_feature(cfeature.ShapelyFeature([geom], ccrs.PlateCarree()),
                                        facecolor='none', edgecolor=color, linewidth=1.5, zorder=4)

                    # Add label if needed
                    plot_coords = []
                    if 'GENCAT' in record.attributes.keys() or shapefiles['points'] is None:
                        bounds = record.geometry.bounds
                        plot_coords.append(
                            (bounds[0] + bounds[2]) * 0.5)  # lon
                        plot_coords.append(bounds[1])  # lat
                        plot_coords.append(record.attributes['GENPROB'])
                    else:
                        found_areas = []
                        for i_record, i_point in zip(shapefiles['points'].records(), shapefiles['points'].geometries()):
                            found_areas.append([i_record.attributes['BASIN'],i_record.attributes['AREA']])
                        check_area = [record.attributes['BASIN'],record.attributes['AREA']]
                        if 'AREA' in record.attributes.keys() and check_area not in found_areas:
                            bounds = record.geometry.bounds
                            plot_coords.append(
                                (bounds[0] + bounds[2]) * 0.5)  # lon
                            plot_coords.append(bounds[1])  # lat
                            if two_prop['days'] == 2:
                                plot_coords.append(
                                    record.attributes['PROB2DAY'])
                            elif 'PROB5DAY' in record.attributes.keys():
                                plot_coords.append(
                                    record.attributes['PROB5DAY'])
                            else:
                                plot_coords.append(
                                    record.attributes['PROB7DAY'])

                    if len(plot_coords) > 0:
                        # Transform coordinates for label
                        x1, y1 = self.ax.projection.transform_point(
                            plot_coords[0], plot_coords[1], ccrs.PlateCarree())
                        x2, y2 = self.ax.transData.transform((x1, y1))
                        x, y = self.ax.transAxes.inverted().transform((x2, y2))

                        # plot same point but using axes coordinates
                        text = plot_coords[2]
                        a = self.ax.text(x, y - 0.02, text, ha='center', va='top', transform=self.ax.transAxes,
                                         zorder=30, fontweight='bold', fontsize=two_prop['fontsize'], clip_on=True, bbox=bbox_prop)
                        a.set_path_effects([path_effects.Stroke(
                            linewidth=0.5, foreground='w'), path_effects.Normal()])

            # Plot points
            if shapefiles['points'] is not None:
                for record, point in zip(shapefiles['points'].records(), shapefiles['points'].geometries()):

                    lon = (list(point.coords)[0][0])
                    lat = (list(point.coords)[0][1])

                    # Determine if 5 or 7 day outlook exists
                    prob_2day = record.attributes['PROB2DAY'].replace(" ", "")
                    risk_2day = record.attributes['RISK2DAY'].replace(" ", "")
                    if 'PROB5DAY' in record.attributes.keys():
                        prob_5day = record.attributes['PROB5DAY'].replace(
                            " ", "")
                        risk_5day = record.attributes['RISK5DAY'].replace(
                            " ", "")
                    else:
                        prob_5day = record.attributes['PROB7DAY'].replace(
                            " ", "")
                        risk_5day = record.attributes['RISK7DAY'].replace(
                            " ", "")

                    # Label area
                    if two_prop['days'] == 2:
                        color = color_base.get(risk_2day, 'yellow')
                        text = prob_2day
                    else:
                        color = color_base.get(risk_5day, 'yellow')
                        text = prob_5day
                    self.ax.plot(lon, lat, 'X', ms=two_prop['ms'], color=color, mec='k', mew=1.5 * (
                        two_prop['ms'] / 15.0), transform=ccrs.PlateCarree(), zorder=20)

                    # Transform coordinates for label
                    x1, y1 = self.ax.projection.transform_point(
                        lon, lat, ccrs.PlateCarree())
                    x2, y2 = self.ax.transData.transform((x1, y1))
                    x, y = self.ax.transAxes.inverted().transform((x2, y2))

                    # plot same point but using axes coordinates
                    a = self.ax.text(x, y - 0.03, text, ha='center', va='top', transform=self.ax.transAxes,
                                     zorder=30, fontweight='bold', fontsize=two_prop['fontsize'], clip_on=True, bbox=bbox_prop)
                    a.set_path_effects([path_effects.Stroke(
                        linewidth=0.5, foreground='w'), path_effects.Normal()])

        # --------------------------------------------------------------------------------------

        if invest_prop['plot'] or storm_prop['plot']:

            # Iterate over all storms
            for storm_idx, storm in enumerate(storms):

                # Skip if it's already associated with a risk area, if TWO is being plotted
                if storm.realtime and storm.prob_2day != 'N/A' and two_prop['plot']:
                    continue

                # Plot invests
                if storm.invest and invest_prop['plot']:

                    # Test
                    self.ax.plot(storm.lon[-1], storm.lat[-1], 'X', ms=invest_prop['ms'],
                                 color='k', transform=ccrs.PlateCarree(), zorder=20)

                    # Transform coordinates for label
                    x1, y1 = self.ax.projection.transform_point(
                        storm.lon[-1], storm.lat[-1], ccrs.PlateCarree())
                    x2, y2 = self.ax.transData.transform((x1, y1))
                    x, y = self.ax.transAxes.inverted().transform((x2, y2))

                    # plot same point but using axes coordinates
                    a = self.ax.text(x, y - 0.03, f"{storm.name.title()}", ha='center', va='top', transform=self.ax.transAxes,
                                     zorder=30, fontweight='bold', fontsize=invest_prop['fontsize'], clip_on=True, bbox=bbox_prop)
                    a.set_path_effects([path_effects.Stroke(
                        linewidth=0.5, foreground='w'), path_effects.Normal()])

                    # Plot archive track
                    if invest_prop['linewidth'] > 0:
                        self.ax.plot(
                            storm.lon, storm.lat, color=invest_prop['linecolor'], linestyle=invest_prop['linestyle'], zorder=5, transform=ccrs.PlateCarree())

                # Plot TCs
                elif not storm.invest and storm_prop['plot']:

                    # Label dot
                    category = str(wind_to_category(storm.vmax[-1]))
                    color = get_colors_sshws(storm.vmax[-1])
                    if category == "0":
                        category = 'S'
                    if category == "-1":
                        category = 'D'
                    if np.isnan(storm.vmax[-1]):
                        category = 'U'
                        color = 'w'

                    if storm_prop['fillcolor'] == 'none':
                        self.ax.plot(storm.lon[-1], storm.lat[-1], 'o', ms=storm_prop['ms'],
                                     color='none', mec='k', mew=3.0, transform=ccrs.PlateCarree(), zorder=20)
                        self.ax.plot(storm.lon[-1], storm.lat[-1], 'o', ms=storm_prop['ms'],
                                     color='none', mec='r', mew=2.0, transform=ccrs.PlateCarree(), zorder=21)

                    else:
                        if storm_prop['fillcolor'] != 'category':
                            color = storm_prop['fillcolor']
                        self.ax.plot(storm.lon[-1], storm.lat[-1], 'o', ms=storm_prop['ms']
                                     * 1.14, color='k', transform=ccrs.PlateCarree(), zorder=20)
                        self.ax.plot(storm.lon[-1], storm.lat[-1], 'o', ms=storm_prop['ms'],
                                     color=color, transform=ccrs.PlateCarree(), zorder=21)

                        if storm_prop['label_category']:
                            color = mcolors.to_rgb(color)
                            red, green, blue = color
                            textcolor = 'w'
                            if (red * 0.299 + green * 0.587 + blue * 0.114) > (160.0 / 255.0):
                                textcolor = 'k'
                            self.ax.text(storm.lon[-1], storm.lat[-1], category, fontsize=storm_prop['ms'] * 0.83, ha='center', va='center', color=textcolor,
                                         zorder=30, transform=ccrs.PlateCarree(), clip_on=True)

                    # Transform coordinates for label
                    x1, y1 = self.ax.projection.transform_point(
                        storm.lon[-1], storm.lat[-1], ccrs.PlateCarree())
                    x2, y2 = self.ax.transData.transform((x1, y1))
                    x, y = self.ax.transAxes.inverted().transform((x2, y2))

                    # plot same point but using axes coordinates
                    a = self.ax.text(x, y - 0.03, f"{storm.name.title()}", ha='center', va='top', transform=self.ax.transAxes,
                                     zorder=30, fontweight='bold', fontsize=storm_prop['fontsize'], clip_on=True, bbox=bbox_prop)
                    a.set_path_effects([path_effects.Stroke(
                        linewidth=0.5, foreground='w'), path_effects.Normal()])

                    # Plot archive track
                    if storm_prop['linewidth'] > 0:

                        # Fix longitudes for track if crossing dateline
                        plot_lon = list(storm.lon)
                        if np.nanmax(plot_lon) > 165 or np.nanmin(plot_lon) < -165:
                            plot_lon = [i if i > 0 else i +
                                        360.0 for i in plot_lon]
                        self.ax.plot(plot_lon, storm.lat, color=storm_prop['linecolor'], linestyle=storm_prop['linestyle'],
                                     zorder=5, transform=ccrs.PlateCarree())

                    # Plot cone
                    if cone_prop['plot']:

                        try:

                            # Retrieve cone
                            forecast_dict = forecasts[storm_idx]

                            # Fix longitudes for cone if crossing dateline
                            if np.nanmax(forecast_dict['lon']) > 165 or np.nanmin(forecast_dict['lon']) < -165:
                                forecast_dict['lon'] = [
                                    i if i > 0 else i + 360.0 for i in forecast_dict['lon']]
                            cone = generate_nhc_cone(
                                forecast_dict, storm.basin, cone_days=cone_prop['days'])

                            # Plot cone
                            if cone_prop['alpha'] > 0 and storm.basin in constants.NHC_BASINS:
                                cone_2d = cone['cone']
                                cone_2d = ndimage.gaussian_filter(
                                    cone_2d, sigma=0.5, order=0)
                                self.ax.contourf(cone['lon2d'], cone['lat2d'], cone_2d, [0.9, 1.1], colors=[
                                                 '#ffffff', '#ffffff'], alpha=cone_prop['alpha'], zorder=4, transform=ccrs.PlateCarree())
                                self.ax.contour(cone['lon2d'], cone['lat2d'], cone_2d, [
                                                0.9], linewidths=1.5, colors=['k'], zorder=4, transform=ccrs.PlateCarree())

                            # Plot center line & account for dateline crossing
                            if cone_prop['linewidth'] > 0:
                                self.ax.plot(cone['center_lon'], cone['center_lat'], color='w',
                                             linewidth=2.5, zorder=5, transform=ccrs.PlateCarree())
                                self.ax.plot(cone['center_lon'], cone['center_lat'], color='k',
                                             linewidth=2.0, zorder=6, transform=ccrs.PlateCarree())

                            # Plot forecast dots
                            for idx in range(len(forecast_dict['lat'])):
                                if forecast_dict['fhr'][idx] / 24.0 > cone_prop['days']:
                                    continue
                                if cone_prop['ms'] == 0:
                                    continue
                                color = get_colors_sshws(
                                    forecast_dict['vmax'][idx])
                                if np.isnan(forecast_dict['vmax'][idx]):
                                    color = 'w'
                                if cone_prop['fillcolor'] != 'category':
                                    color = cone_prop['fillcolor']

                                marker = 'o'
                                if forecast_dict['type'][idx] not in constants.TROPICAL_STORM_TYPES:
                                    marker = '^'
                                if np.isnan(forecast_dict['vmax'][idx]):
                                    marker = 'o'
                                self.ax.plot(forecast_dict['lon'][idx], forecast_dict['lat'][idx], marker, ms=cone_prop['ms'],
                                             mfc=color, mec='k', zorder=7, transform=ccrs.PlateCarree(), clip_on=True)

                                if cone_prop['label_category'] and marker == 'o':
                                    category = str(wind_to_category(
                                        forecast_dict['vmax'][idx]))
                                    if category == "0":
                                        category = 'S'
                                    if category == "-1":
                                        category = 'D'
                                    if np.isnan(forecast_dict['vmax'][idx]):
                                        category = 'U'

                                    color = mcolors.to_rgb(color)
                                    red, green, blue = color
                                    textcolor = 'w'
                                    if (red * 0.299 + green * 0.587 + blue * 0.114) > (160.0 / 255.0):
                                        textcolor = 'k'

                                    self.ax.text(forecast_dict['lon'][idx], forecast_dict['lat'][idx], category, fontsize=cone_prop['ms'] * 0.81, ha='center', va='center', color=textcolor,
                                                 zorder=19, transform=ccrs.PlateCarree(), clip_on=True)
                        except:
                            pass

        # --------------------------------------------------------------------------------------

        # Plot domain
        bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Plot parallels and meridians
        # This is currently not supported for all cartopy projections.
        if map_prop['plot_gridlines']:
            try:
                self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])
            except:
                pass

        # --------------------------------------------------------------------------------------

        # Add title
        self.ax.set_title(f"Summary{add_title}",
                          loc='left', fontsize=17, fontweight='bold')
        self.ax.set_title(
            f"Valid: {valid_date.strftime('%H UTC %d %b %Y')}", loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add credit
        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # --------------------------------------------------------------------------------------

        # Add legend
        # self.add_legend(prop,segmented_colors,levels,cmap,storm_data)

        # -----------------------------------------------------------------------------------------

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax
