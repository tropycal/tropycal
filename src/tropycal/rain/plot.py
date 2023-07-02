import numpy as np
import warnings
from datetime import datetime as dt

from ..plot import Plot
from ..utils import *
from .. import constants

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
except ImportError:
    warnings.warn(
        "Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib as mlib
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.colors as col

except ImportError:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


class RainPlot(Plot):

    def __init__(self):

        self.use_credit = True

    def plot_storm(self, storm, grid, levels, cmap, domain="dynamic", plot_all_dots=False, ax=None, track_labels=False, save_path=None, prop={}, map_prop={}):
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

        # Set default properties
        default_prop = {'dots': True, 'fillcolor': 'category', 'cmap': None,
                        'levels': None, 'linecolor': 'k', 'linewidth': 1.0, 'ms': 7.5}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop, plot_geography=False)

        # --------------------------------------------------------------------------------------

        # get resolution corresponding to string in prop
        res = self.check_res(map_prop['res'])

        # fill oceans if specified
        self.ax.set_facecolor(map_prop['ocean_color'])
        ocean_mask = self.ax.add_feature(cfeature.OCEAN.with_scale(
            res), facecolor=map_prop['ocean_color'], edgecolor='face', zorder=3)
        lake_mask = self.ax.add_feature(cfeature.LAKES.with_scale(
            res), facecolor=map_prop['ocean_color'], edgecolor='face', zorder=1)
        continent_mask = self.ax.add_feature(cfeature.LAND.with_scale(
            res), facecolor=map_prop['land_color'], edgecolor='face', zorder=0)

        # draw geography
        states = self.ax.add_feature(cfeature.STATES.with_scale(res), linewidths=map_prop['linewidth'],
                                     linestyle='solid', edgecolor=map_prop['linecolor'], zorder=4)
        countries = self.ax.add_feature(cfeature.BORDERS.with_scale(res), linewidths=map_prop['linewidth'],
                                        linestyle='solid', edgecolor=map_prop['linecolor'], zorder=4)
        coastlines = self.ax.add_feature(cfeature.COASTLINE.with_scale(res), linewidths=map_prop['linewidth'],
                                         linestyle='solid', edgecolor=map_prop['linecolor'], zorder=4)

        # --------------------------------------------------------------------------------------

        # Retrieve grid coordinates and values
        plot_grid = True
        if isinstance(grid, dict):
            if 'lat' in grid.keys():
                grid_lat = grid['lat']
                grid_lon = grid['lon']
                grid_val = grid['grid']
            else:
                ms = grid['ms']
                mec = grid['mec']
                mew = grid['mew']
                minimum_threshold = grid['minimum_threshold']
                plot_grid = False
        else:
            grid_lat = grid.lat.values
            grid_lon = grid.lon.values
            grid_val = grid.values

        # Determine levels and colormap
        if levels is None:
            levels = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20]
        if cmap is None:
            cmap = plt.cm.YlGn
        norm = col.BoundaryNorm(levels, cmap.N)

        # Contour fill grid if requested
        if plot_grid:
            self.ax.contourf(grid_lon, grid_lat, grid, levels,
                             cmap=cmap, norm=norm, zorder=2)

        # Plot dots if requested
        else:

            # Iterate over sorted data (so highest totals show up on top)
            iter_df = storm.rain.sort_values('Total')
            for _, row in iter_df.iterrows():

                # Retrieve rain total and determine color
                rain_value = row['Total']
                if rain_value < minimum_threshold:
                    continue
                color = rgb_tuple_to_str(
                    cmap(norm(rain_value), bytes=True)[:-1])

                # Specify additional kwargs
                ms_kwargs = {}
                if mec is not None:
                    ms_kwargs = {'mec': mec, 'mew': mew}
                self.ax.plot(row['Lon'], row['Lat'], 'o', ms=ms,
                             color=color, **ms_kwargs, transform=ccrs.PlateCarree())

        # Produce colorbar
        cs = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cs.set_array([])
        self.fig.colorbar(cs, ax=self.ax)

        # --------------------------------------------------------------------------------------

        # Keep record of lat/lon coordinate extrema
        max_lat = np.nanmax(storm.rain['Lat'].values)
        min_lat = np.nanmin(storm.rain['Lat'].values)
        max_lon = np.nanmax(storm.rain['Lon'].values)
        min_lon = np.nanmin(storm.rain['Lon'].values)

        # Get storm data
        storm_data = storm.dict

        # Retrieve storm data
        lats = storm_data['lat']
        lons = storm_data['lon']
        vmax = storm_data['vmax']
        styp = storm_data['type']
        sdate = storm_data['time']

        # Account for cases crossing dateline
        if self.proj.proj4_params['lon_0'] == 180.0:
            new_lons = np.array(lons)
            new_lons[new_lons < 0] = new_lons[new_lons < 0]+360.0
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

        # Iterate over storm data to plot
        for i, (i_lat, i_lon, i_vmax, i_mslp, i_time, i_type) in enumerate(zip(storm_data['lat'], lons, storm_data['vmax'], storm_data['mslp'], storm_data['time'], storm_data['type'])):

            # Determine line color, with SSHWS scale used as default
            if prop['linecolor'] == 'category':
                segmented_colors = True
                line_color = get_colors_sshws(np.nan_to_num(i_vmax))

            # Use user-defined colormap if another storm variable
            elif isinstance(prop['linecolor'], str) and prop['linecolor'] in ['vmax', 'mslp']:
                segmented_colors = True
                color_variable = storm_data[prop['linecolor']]
                if prop['levels'] is None:  # Auto-determine color levels if needed
                    prop['levels'] = [
                        np.nanmin(color_variable), np.nanmax(color_variable)]
                cmap, levels = get_cmap_levels(
                    prop['linecolor'], prop['cmap'], prop['levels'])
                line_color = cmap((color_variable-min(levels)) /
                                  (max(levels)-min(levels)))[i]

            # Otherwise go with user input as is
            else:
                segmented_colors = False
                line_color = prop['linecolor']

            # For tropical/subtropical types, color-code if requested
            if i > 0:
                if i_type in constants.TROPICAL_STORM_TYPES:

                    # Plot underlying black and overlying colored line
                    self.ax.plot([lons[i-1], lons[i]], [storm_data['lat'][i-1], storm_data['lat'][i]], '-',
                                 linewidth=prop['linewidth']*1.33, color='k', zorder=5,
                                 transform=ccrs.PlateCarree())
                    self.ax.plot([lons[i-1], lons[i]], [storm_data['lat'][i-1], storm_data['lat'][i]], '-',
                                 linewidth=prop['linewidth'], color=line_color, zorder=6,
                                 transform=ccrs.PlateCarree())

                # For non-tropical types, plot dotted lines
                else:

                    # Restrict line width to 1.5 max
                    line_width = prop['linewidth'] + 0.0
                    if line_width > 1.5:
                        line_width = 1.5

                    # Plot dotted line
                    self.ax.plot([lons[i-1], lons[i]], [storm_data['lat'][i-1], storm_data['lat'][i]], ':',
                                 linewidth=line_width, color=line_color, zorder=6,
                                 transform=ccrs.PlateCarree(),
                                 path_effects=[path_effects.Stroke(linewidth=line_width*1.33, foreground='k'),
                                               path_effects.Normal()])

            # Plot dots if requested
            if prop['dots']:

                # Skip if plot_all_dots is False and not in 0,6,12,18z
                if not plot_all_dots:
                    if i_time.strftime('%H%M') not in ['0000', '0600', '1200', '1800']:
                        continue

                # Determine fill color, with SSHWS scale used as default
                if prop['fillcolor'] == 'category':
                    segmented_colors = True
                    fill_color = get_colors_sshws(np.nan_to_num(i_vmax))

                # Use user-defined colormap if another storm variable
                elif isinstance(prop['fillcolor'], str) and prop['fillcolor'] in ['vmax', 'mslp']:
                    segmented_colors = True
                    color_variable = storm_data[prop['fillcolor']]
                    if prop['levels'] is None:  # Auto-determine color levels if needed
                        prop['levels'] = [
                            np.nanmin(color_variable), np.nanmax(color_variable)]
                    cmap, levels = get_cmap_levels(
                        prop['fillcolor'], prop['cmap'], prop['levels'])
                    fill_color = cmap(
                        (color_variable-min(levels))/(max(levels)-min(levels)))[i]

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
                self.ax.plot(i_lon, i_lat, marker_type, mfc=fill_color, mec='k', mew=0.5,
                             zorder=7, ms=prop['ms'], transform=ccrs.PlateCarree())

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
        try:
            self.plot_lat_lon_lines(
                [bound_w, bound_e, bound_s, bound_n], zorder=9)
        except:
            pass

        # --------------------------------------------------------------------------------------

        # Add left title
        type_array = np.array(storm_data['type'])
        if not invest_bool:
            idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
                type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
            tropical_vmax = np.array(storm_data['vmax'])[idx]

            # Coerce to include non-TC points if storm hasn't been designated yet
            add_ptc_flag = False
            if len(tropical_vmax) == 0:
                add_ptc_flag = True
                idx = np.where((type_array == 'LO') | (type_array == 'DB'))
            tropical_vmax = np.array(storm_data['vmax'])[idx]

            subtrop = classify_subtropical(np.array(storm_data['type']))
            peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
            peak_basin = storm_data['wmo_basin'][peak_idx]
            storm_type = get_storm_classification(
                np.nanmax(tropical_vmax), subtrop, peak_basin)
            if add_ptc_flag:
                storm_type = "Potential Tropical Cyclone"
            left_title_string = f"{storm_type} {storm_data['name']}"
        else:
            # Use all indices for invests
            idx = np.array([True for i in type_array])
            add_ptc_flag = False
            tropical_vmax = np.array(storm_data['vmax'])

            # Determine letter in front of invest
            add_letter = 'L'
            if storm_data['id'][0] == 'C':
                add_letter = 'C'
            elif storm_data['id'][0] == 'E':
                add_letter = 'E'

            # Add title
            left_title_string = f"INVEST {storm_data['id'][2:4]}{add_letter}"

        # Add left title
        if plot_grid:
            left_title_string += "\nInterpolated WPC Storm Rainfall (in)"
        else:
            if minimum_threshold > 1:
                left_title_string += f"\nWPC Storm Rainfall (>{np.round(minimum_threshold,2)} inch)"
            else:
                left_title_string += f"\nWPC Storm Rainfall (inch)"
        self.ax.set_title(left_title_string, loc='left',
                          fontsize=16, fontweight='bold')

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

        credit_text = "Rainfall data (inch) from\nWeather Prediction Center (WPC)\n\n"
        self.ax.text(0.99, 0.01, credit_text, fontsize=9, color='k', alpha=0.7,
                     transform=self.ax.transAxes, ha='right', va='bottom', zorder=10)

        credit_text = self.plot_credit()
        self.add_credit(credit_text)

        # --------------------------------------------------------------------------------------

        # Add legend
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
            self.ax.legend(handles=[ex, sb, td, ts, c1,
                           c2, c3, c4, c5], prop={'size': 11.5})

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
            self.ax.legend(handles=[ex, td, ts, c1, c2, c3, c4, c5], prop={
                           'size': 11.5}).set_zorder(10)

        elif prop['dots'] and not segmented_colors:
            ex = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Non-Tropical', marker='^', color=prop['fillcolor'])
            sb = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Subtropical', marker='s', color=prop['fillcolor'])
            td = mlines.Line2D([], [], linestyle='None', ms=prop['ms'], mec='k',
                               mew=0.5, label='Tropical', marker='o', color=prop['fillcolor'])
            handles = [ex, sb, td]
            self.ax.legend(handles=handles, fontsize=11.5).set_zorder(10)

        elif not prop['dots'] and not segmented_colors:
            ex = mlines.Line2D([], [], linestyle='dotted',
                               label='Non-Tropical', color=prop['linecolor'])
            td = mlines.Line2D([], [], linestyle='solid',
                               label='Tropical', color=prop['linecolor'])
            handles = [ex, td]
            self.ax.legend(handles=handles, fontsize=11.5).set_zorder(10)

        elif prop['dots']:
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
            l = self.ax.legend(handles=handles, fontsize=11.5).set_zorder(10)
            plt.draw()

            # Get the bbox
            try:
                bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
            except:
                bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())

            # Define colorbar axis
            cax = self.fig.add_axes(
                [bb.x0+0.47*bb.width, bb.y0+.057*bb.height, 0.015, .65*bb.height])
            norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
            cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical',
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
                                     :-1]+np.linspace(0, 1, len(levels))[1:])*.5)
                cax2.set_yticklabels(
                    ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
                cax2.tick_params('both', length=0, width=0, which='major')
                cax.yaxis.set_ticks_position('left')
                rect_offset = 0.7
            if prop['fillcolor'] == 'time':
                cax.set_yticklabels(
                    [f'{mdates.num2date(i):%b %-d}' for i in clevs], fontsize=11.5)

        else:
            ex = mlines.Line2D([], [], linestyle='dotted',
                               label='Non-Tropical', color='k')
            td = mlines.Line2D([], [], linestyle='solid',
                               label='Tropical', color='k')
            handles = [ex, td]
            for _ in range(7):
                handles.append(mlines.Line2D(
                    [], [], linestyle='-', label='', lw=0))
            l = self.ax.legend(handles=handles, fontsize=11.5).set_zorder(10)
            plt.draw()

            # Get the bbox
            try:
                bb = l.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
            except:
                bb = l.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())

            # Define colorbar axis
            cax = self.fig.add_axes(
                [bb.x0+0.47*bb.width, bb.y0+.057*bb.height, 0.015, .65*bb.height])
            norm = mlib.colors.Normalize(vmin=min(levels), vmax=max(levels))
            cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical',
                                     ticks=levels)

            cax.tick_params(labelsize=11.5)
            cax.yaxis.set_ticks_position('left')
            cbarlab = make_var_label(prop['linecolor'], storm_data)
            cbar.set_label(cbarlab, fontsize=11.5, rotation=90)

            rect_offset = 0.0
            if prop['cmap'] == 'category' and prop['linecolor'] == 'vmax':
                cax.yaxis.set_ticks(np.linspace(
                    min(levels), max(levels), len(levels)))
                cax.yaxis.set_ticklabels(levels)
                cax2 = cax.twinx()
                cax2.yaxis.set_ticks_position('right')
                cax2.yaxis.set_ticks((np.linspace(0, 1, len(levels))[
                                     :-1]+np.linspace(0, 1, len(levels))[1:])*.5)
                cax2.set_yticklabels(
                    ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
                cax2.tick_params('both', length=0, width=0, which='major')
                cax.yaxis.set_ticks_position('left')
                rect_offset = 0.7
            if prop['linecolor'] == 'time':
                cax.set_yticklabels(
                    [f'{mdates.num2date(i):%b %-d}' for i in clevs], fontsize=11.5)

        # -----------------------------------------------------------------------------------------

        # Save image if specified
        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        # Return axis if specified, otherwise display figure
        return self.ax
