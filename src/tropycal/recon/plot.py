import numpy as np
import pandas as pd
import warnings
from datetime import datetime as dt
from scipy.ndimage import gaussian_filter as gfilt
import copy

from ..plot import Plot

# Import tools
from .tools import *
from ..utils import *

try:
    from cartopy import crs as ccrs
except ImportError:
    warnings.warn(
        "Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib as mlib
    import matplotlib.lines as mlines
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
except ImportError:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


class ReconPlot(Plot):

    def __init__(self):

        self.use_credit = True

    def plot_points(self, storm, recon_data, domain="dynamic", varname='wspd', radlim=None, barbs=False, scatter=False,
                    ax=None, return_domain=False, prop={}, map_prop={}, mission=False, vdms=[], mission_id=''):
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
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """

        if not barbs and not scatter:
            scatter = True

        if isinstance(varname, tuple):
            titleinput = copy.copy(varname)
            varname = varname[0]
        else:
            titleinput = (varname, None)

        # Set default properties
        default_prop = {'cmap': 'category', 'levels': (np.nanmin(recon_data[varname]), np.nanmax(recon_data[varname])),
                        'sortby': varname, 'ascending': (varname != 'p_sfc'), 'linewidth': 1.5, 'ms': 7.5, 'marker': 'o', 'zorder': None}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF', 'linewidth': 0.5, 'linecolor': 'k',
                            'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # set default properties
        input_prop = prop
        input_map_prop = map_prop

        # --------------------------------------------------------------------------------------

        # Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        # Retrieve storm data
        storm_data = storm.dict

        # Check recon_data type
        if isinstance(recon_data, pd.core.frame.DataFrame):
            pass
        else:
            raise RuntimeError("Error: recon_data must be dataframe")

        # Retrieve storm data
        if radlim is None:
            lats = recon_data['lat']
            lons = recon_data['lon']
        else:
            temp_df = recon_data.loc[recon_data['distance'] <= radlim]
            lats = temp_df['lat']
            lons = temp_df['lon']

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

        # Get colormap and level extrema
        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['levels'])
        if varname in ['vmax', 'sfmr', 'wspd', 'fl_to_sfc'] and prop['cmap'] in ['category', 'category_recon']:
            vmin = min(clevs)
            vmax = max(clevs)
        else:
            vmin = min(prop['levels'])
            vmax = max(prop['levels'])

        # Plot recon data as specified
        if barbs:
            dataSort = recon_data.sort_values(
                by=prop['sortby']).reset_index(drop=True)
            if radlim is not None:
                dataSort = dataSort.loc[dataSort['distance'] <= radlim]
            norm = mlib.colors.Normalize(vmin=vmin, vmax=vmax)
            colors = cmap(norm(dataSort[prop['sortby']].values))
            colors = [tuple(i) for i in colors]
            qv = plt.barbs(dataSort['lon'], dataSort['lat'],
                           *uv_from_wdir(dataSort[prop['sortby']], dataSort['wdir']), color=colors, length=5, linewidth=0.5)
            cbmap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbmap.set_array([])

        if scatter:
            dataSort = recon_data.sort_values(
                by=prop['sortby'], ascending=prop['ascending']).reset_index(drop=True)
            if radlim is not None:
                dataSort = dataSort.loc[dataSort['distance'] <= radlim]
            cbmap = plt.scatter(dataSort['lon'], dataSort['lat'], c=dataSort[varname],
                                cmap=cmap, vmin=vmin, vmax=vmax, s=prop['ms'], marker=prop['marker'], zorder=prop['zorder'])

        # Plot latest point if from a Mission object
        if mission:
            plt.plot(recon_data['lon'].values[-1], recon_data['lat'].values[-1],
                     'o', mfc='none', mec='k', mew=1.5, ms=10)

        # Plot VDMs
        if len(vdms) > 0:
            for vdm_idx in range(len(vdms)):
                vdm = vdms[vdm_idx]

                # Transform coordinates for label
                try:
                    a = self.ax.text(vdm['lon'], vdm['lat'], str(int(np.round(vdm['Minimum Sea Level Pressure (hPa)']))),
                                     zorder=30, clip_on=True, fontsize=12, fontweight='bold', ha='center', va='center')
                    a.set_path_effects(
                        [patheffects.Stroke(linewidth=0.5, foreground='w'), patheffects.Normal()])
                except:
                    pass

        # --------------------------------------------------------------------------------------

        # Storm-centered plot domain
        if domain == "dynamic":

            bound_w, bound_e, bound_s, bound_n = dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat, recon=True)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Determine number of lat/lon lines to use for parallels & meridians
        if map_prop['plot_gridlines']:
            self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])

        # --------------------------------------------------------------------------------------

        # Add left title
        type_array = np.array(storm_data['type'])
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
        if add_ptc_flag == True:
            storm_type = "Potential Tropical Cyclone"

        dot = u"\u2022"
        try:
            vartitle = get_recon_title(*titleinput)
        except:
            vartitle = [varname]
        self.ax.set_title(f"{storm_type} {storm_data['name']}\n" + 'Recon: ' + ' '.join(
            vartitle), loc='left', fontsize=17, fontweight='bold')
        if mission_id != '':
            self.ax.set_title(f"Mission ID: {mission_id}\nRecon: " + ' '.join(
                vartitle), loc='left', fontsize=17, fontweight='bold')

        # Add right title
        start_time = dt.strftime(min(recon_data['time']), '%H:%M UTC %d %b %Y')
        end_time = dt.strftime(max(recon_data['time']), '%H:%M UTC %d %b %Y')
        self.ax.set_title(
            f'Start ... {start_time}\nEnd ... {end_time}', loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add legend

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
            [bb.x0 + bb.width, bb.y0 - .05 * bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical',
                                 ticks=clevs)

        if len(prop['levels']) > 2:
            cax.yaxis.set_ticks(np.linspace(
                min(clevs), max(clevs), len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
        else:
            cax.yaxis.set_ticks(clevs)
        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')

        rect_offset = 0.0
        if prop['cmap'] in ['category', 'category_recon'] and varname in ['sfmr', 'wspd']:
            cax.yaxis.set_ticks(np.linspace(
                min(clevs), max(clevs), len(clevs)))
            cax.yaxis.set_ticklabels(clevs)
            cax2 = cax.twinx()
            cax2.yaxis.set_ticks_position('right')
            cax2.yaxis.set_ticks((np.linspace(0, 1, len(clevs))[
                                 :-1] + np.linspace(0, 1, len(clevs))[1:]) * .5)
            if prop['cmap'] == 'category':
                cax2.set_yticklabels(
                    ['TD', 'TS', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
            else:
                cax2.set_yticklabels(
                    ['TD', 'TS', '>50 kt', 'Cat-1', 'Cat-2', 'Cat-3', 'Cat-4', 'Cat-5'], fontsize=11.5)
            cax2.tick_params('both', length=0, width=0, which='major')
            cax.yaxis.set_ticks_position('left')

            rect_offset = 0.7

        rectangle = mpatches.Rectangle((bb.x0, bb.y0 - 0.1 * bb.height), (1.8 + rect_offset) * bb.width, 1.1 * bb.height,
                                       fc='w', edgecolor='0.8', alpha=0.8,
                                       transform=self.fig.transFigure, zorder=2)
        self.ax.add_patch(rectangle)

        # Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        # Return axis and domain
        if return_domain:
            return self.ax, '/'.join([str(b) for b in [bound_w, bound_e, bound_s, bound_n]])
        else:
            return self.ax

    def plot_swath(self, storm, Maps, varname, swathfunc, track_dict,
                   domain="dynamic", ax=None, prop={}, map_prop={}):

        # Set default properties
        default_prop = {'cmap': 'category', 'levels': None,
                        'left_title': '', 'right_title': 'All storms', 'pcolor': True}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        # Keep record of lat/lon coordinate extrema
        max_lat = None
        min_lat = None
        max_lon = None
        min_lon = None

        # Retrieve recon data
        lats = Maps['center_lat']
        lons = Maps['center_lon']

        # Retrieve storm data
        storm_data = storm.dict

        # Add to coordinate extrema
        if max_lat is None:
            max_lat = max(lats) + 2.5
        else:
            if max(lats) > max_lat:
                max_lat = max(lats)
        if min_lat is None:
            min_lat = min(lats) - 2.5
        else:
            if min(lats) < min_lat:
                min_lat = min(lats)
        if max_lon is None:
            max_lon = max(lons) + 2.5
        else:
            if max(lons) > max_lon:
                max_lon = max(lons)
        if min_lon is None:
            min_lon = min(lons) - 2.5
        else:
            if min(lons) < min_lon:
                min_lon = min(lons)

        bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
            min_lon, max_lon, min_lat, max_lat)

        distproj = ccrs.LambertConformal()
        out = distproj.transform_points(ccrs.PlateCarree(), np.array([bound_w, bound_w, bound_e, bound_e]),
                                        np.array([bound_s, bound_n, bound_s, bound_n]))
        grid_res = 1 * 1e3  # m
        xi = np.arange(int(min(out[:, 0]) / grid_res) * grid_res,
                       int(max(out[:, 0]) / grid_res) * grid_res + grid_res, grid_res)
        yi = np.arange(int(min(out[:, 1]) / grid_res) * grid_res,
                       int(max(out[:, 1]) / grid_res) * grid_res + grid_res, grid_res)
        xmgrid, ymgrid = np.meshgrid(xi, yi)

        out = distproj.transform_points(
            ccrs.PlateCarree(), Maps['center_lon'], Maps['center_lat'])

        cx = np.rint(gfilt(out[:, 0], 1) / grid_res) * grid_res
        cy = np.rint(gfilt(out[:, 1], 1) / grid_res) * grid_res
        aggregate_grid = np.ones(xmgrid.shape) * np.nan

        def nanfunc(func, a, b):
            c = np.concatenate([a[None], b[None]])
            c = np.ma.array(c, mask=np.isnan(c))
            d = func(c, axis=0)
            e = d.data
            e[d.mask] = np.nan
            return e

        for t, (x_center, y_center, var) in enumerate(zip(cx, cy, Maps['maps'])):
            x_fromc = x_center + Maps['grid_x'] * 1e3
            y_fromc = y_center + Maps['grid_y'] * 1e3
            inrecon = np.where((xmgrid >= np.min(x_fromc)) & (xmgrid <= np.max(x_fromc)) &
                               (ymgrid >= np.min(y_fromc)) & (ymgrid <= np.max(y_fromc)))
            inmap = np.where((x_fromc >= np.min(xmgrid)) & (x_fromc <= np.max(xmgrid)) &
                             (y_fromc >= np.min(ymgrid)) & (y_fromc <= np.max(ymgrid)))
            aggregate_grid[inrecon] = nanfunc(
                swathfunc, aggregate_grid[inrecon], var[inmap])

        if prop['levels'] is None:
            prop['levels'] = (np.nanmin(aggregate_grid),
                              np.nanmax(aggregate_grid))
        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['levels'])

        out = self.proj.transform_points(distproj, xmgrid, ymgrid)
        lons = out[:, :, 0]
        lats = out[:, :, 1]

        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(lons, lats, aggregate_grid, cmap=cmap,
                                 norm=norm, levels=clevs, transform=ccrs.PlateCarree())

        # Storm-centered plot domain
        if domain == "dynamic":

            bound_w, bound_e, bound_s, bound_n = self.dynamic_map_extent(
                min_lon, max_lon, min_lat, max_lat)
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Determine number of lat/lon lines to use for parallels & meridians
        if map_prop['plot_gridlines']:
            self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])

        # --------------------------------------------------------------------------------------

        # Add left title
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]

        subtrop = classify_subtropical(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_classification(
            np.nanmax(tropical_vmax), subtrop, peak_basin)

        dot = u"\u2022"
        vartitle = get_recon_title(varname)
        self.ax.set_title(f"{storm_type} {storm_data['name']}\n" + 'Recon: ' + ' '.join(
            vartitle), loc='left', fontsize=17, fontweight='bold')

        # Add right title
        # max_ppf = max(PPF)
        start_time = dt.strftime(min(Maps['time']), '%H:%M UTC %d %b %Y')
        end_time = dt.strftime(max(Maps['time']), '%H:%M UTC %d %b %Y')
        self.ax.set_title(
            f'Start ... {start_time}\nEnd ... {end_time}', loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add legend

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
            [bb.x0 + bb.width, bb.y0 - .05 * bb.height, 0.015, bb.height])
#        cbmap = mlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical',
                                 ticks=clevs)

        cax.tick_params(labelsize=11.5)
        cax.yaxis.set_ticks_position('left')

        rect_offset = 0.0
        if prop['cmap'] == 'category' and varname in ['sfmr', 'wspd']:
            cax.yaxis.set_ticks(np.linspace(0, 1, len(clevs)))
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

        rectangle = mpatches.Rectangle((bb.x0, bb.y0 - 0.1 * bb.height), (1.8 + rect_offset) * bb.width, 1.1 * bb.height,
                                       fc='w', edgecolor='0.8', alpha=0.8,
                                       transform=self.fig.transFigure, zorder=2)
        self.ax.add_patch(rectangle)

        # Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

    def plot_polar(self, dfRecon, track_dict, time=None, reconInterp=None, radlim=150, ax=None, prop={}):
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
        prop : dict
            Properties of plot
        """

        # Set default properties
        default_prop = {'colors': 'category',
                        'levels': [34, 64, 83, 96, 113, 137, 200]}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)

        # mlib.rcParams.update({'font.size': 16})

        fig = plt.figure(figsize=prop['figsize'])
        if ax is None:
            self.ax = plt.subplot()
        else:
            self.ax = ax

        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['clevs'])
        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(Maps_dict['grid_x'], Maps_dict['grid_y'], Maps_dict['maps'],
                                 cmap=cmap, norm=norm, levels=clevs, transform=ccrs.PlateCarree())

        rightarrow = u"\u2192"
        plt.xlabel(f'W {rightarrow} E Distance (km)')
        plt.ylabel(f'S {rightarrow} N Distance (km)')
        plt.axis([-radlim, radlim, -radlim, radlim])
        plt.axis('equal')

        cbar = plt.colorbar()
        cbar.set_label('wind (kt)')

        # --------------------------------------------------------------------------------------

        # Add left title
        self.ax.set_title('Recon interpolated', loc='left',
                          fontsize=17, fontweight='bold')

        # Add right title
        # max_ppf = max(PPF)
        start_time = dt.strftime(min(dfRecon['time']), '%H:%M UTC %d %b %Y')
        end_time = dt.strftime(max(dfRecon['time']), '%H:%M UTC %d %b %Y')
        self.ax.set_title(
            f'Start ... {start_time}\nEnd ... {end_time}', loc='right', fontsize=13)

        # Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        # --------------------------------------------------------------------------------------

        # Display figure, return axis
        plt.show()
        return self.ax
        plt.close()

    def plot_maps(self, storm, Maps_dict, varname, recon_stats=None,
                  domain='dynamic', ax=None, return_domain=False, prop={}, map_prop={}):
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
        prop : dict
            Properties of plot
        """

        # Set default properties
        default_prop = {'cmap': 'category', 'levels': None,
                        'left_title': '', 'right_title': '', 'pcolor': True}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (12.5, 8.5), 'dpi': 120, 'plot_gridlines': True}

        # Initialize plot
        prop = self.add_prop(prop, default_prop)
        map_prop = self.add_prop(map_prop, default_map_prop)
        self.plot_init(ax, map_prop)

        MULTIVAR = False
        if isinstance(varname, (tuple, list)):
            varname2 = varname[1]
            varname = varname[0]
            Maps_dict2 = Maps_dict[1]
            Maps_dict = Maps_dict[0]
            MULTIVAR = True

        grid_res = 1 * 1e3  # m
        clon = Maps_dict['center_lon']
        clat = Maps_dict['center_lat']
        distproj = ccrs.LambertConformal()
        out = distproj.transform_points(
            ccrs.PlateCarree(), np.array([clon]), np.array([clat]))
        cx = np.rint(out[:, 0] / grid_res) * grid_res
        cy = np.rint(out[:, 1] / grid_res) * grid_res
        xmgrid = cx + Maps_dict['grid_x'] * grid_res
        ymgrid = cy + Maps_dict['grid_y'] * grid_res
        out = self.proj.transform_points(distproj, xmgrid, ymgrid)
        lons = out[:, :, 0]
        lats = out[:, :, 1]

        # mlib.rcParams.update({'font.size': 16})

        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['levels'])

        norm = mlib.colors.BoundaryNorm(clevs, cmap.N)
        cbmap = self.ax.contourf(lons, lats, Maps_dict['maps'],
                                 cmap=cmap, norm=norm, levels=clevs, transform=ccrs.PlateCarree())

        if MULTIVAR:
            CS = self.ax.contour(lons, lats, Maps_dict2['maps'], levels=np.arange(
                0, 2000, 4), colors='k', linewidths=0.5)
            # Recast levels to new class
            CS.levels = [int(val) for val in CS.levels]
            self.ax.clabel(CS, CS.levels, fmt='%i', inline=True, fontsize=10)

        # Storm-centered plot domain
        if domain == "dynamic":

            bound_w, bound_e, bound_s, bound_n = np.amin(
                lons) - .1, np.amax(lons) + .1, np.amin(lats) - .1, np.amax(lats) + .1
            self.ax.set_extent(
                [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        # Pre-generated or custom domain
        else:
            bound_w, bound_e, bound_s, bound_n = self.set_projection(domain)

        # Determine number of lat/lon lines to use for parallels & meridians
        if map_prop['plot_gridlines']:
            self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])

#        rightarrow = u"\u2192"
#        plt.xlabel(f'W {rightarrow} E Distance (km)')
#        plt.ylabel(f'S {rightarrow} N Distance (km)')
#        plt.axis([-radlim,radlim,-radlim,radlim])
#        plt.axis('equal')

        cbar = self.fig.colorbar(cbmap, orientation='vertical',
                                 ticks=clevs)
#        cbar.set_label('wind (kt)')

        # --------------------------------------------------------------------------------------

        storm_data = storm.dict
        # Add left title
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
        tropical_vmax = np.array(storm_data['vmax'])[idx]

        subtrop = classify_subtropical(np.array(storm_data['type']))
        peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
        peak_basin = storm_data['wmo_basin'][peak_idx]
        storm_type = get_storm_classification(
            np.nanmax(tropical_vmax), subtrop, peak_basin)

        vartitle = get_recon_title(varname)
        title_left = f"{storm_type} {storm_data['name']}\n" + \
            'Recon: ' + ' '.join(vartitle)
        self.ax.set_title(title_left, loc='left',
                          fontsize=17, fontweight='bold')

        # Add right title
        self.ax.set_title(Maps_dict['time'].strftime(
            '%H:%M UTC %d %b %Y'), loc='right', fontsize=13)

        # Add stats
        if recon_stats is not None:
            a = self.ax.text(0.8, 0.97, f"Max FL Wind: {int(recon_stats['pkwnd_max'])} kt\n" +
                             f"Max SFMR: {int(recon_stats['sfmr_max'])} kt\n" +
                             f"Min SLP: {int(recon_stats['p_min'])} hPa", fontsize=9.5, color='k',
                             bbox=dict(
                                 facecolor='0.9', edgecolor='black', boxstyle='round,pad=1'),
                             transform=self.ax.transAxes, ha='left', va='top', zorder=10)

        # Add plot credit
        text = self.plot_credit()
        self.add_credit(text)

        if return_domain:
            return self.ax, {'n': bound_n, 'e': bound_e, 's': bound_s, 'w': bound_w}
        else:
            return self.ax


def plot_skewt(dict_list, storm_name_title):

    def time2text(time):
        try:
            return f'{time:%H:%M UTC %d %b %Y}'
        except:
            return 'N/A'

    def location_text(indict):
        try:
            loc = indict['location'].lower()
        except:
            return ''
        if loc == 'eyewall':
            return r"$\bf{" + loc.capitalize() + '}$, '
        else:
            return r"$\bf{" + loc.capitalize() + '}$, '

    degsym = u"\u00B0"

    def latlon2text(lat, lon):
        NA = False
        if lat < 0:
            lattx = f'{abs(lat)}{degsym}S'
        elif lat >= 0:
            lattx = f'{lat}{degsym}N'
        else:
            NA = True
        if lon < 0:
            lontx = f'{abs(lon)}{degsym}W'
        elif lon >= 0:
            lontx = f'{lon}{degsym}E'
        else:
            NA = True
        if NA:
            return 'N/A'
        else:
            return lattx + ' ' + lontx

    def mission2text(x):
        try:
            return int(x[:2])
        except:
            return x[:2]

    def wind_components(speed, direction):
        u = -speed * np.sin(direction * np.pi / 180)
        v = -speed * np.cos(direction * np.pi / 180)
        return u, v

    def deg2dir(x):
        dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        try:
            idx = int(round(x * 16 / 360, 0) % 16)
            return dirs[idx]
        except:
            return 'N/A'

    def rh_from_dp(t, td):
        rh = np.exp(17.67 * (td) / (td + 273.15 - 29.65)) / \
            np.exp(17.67 * (t) / (t + 273.15 - 29.65))
        return rh * 100

    def cellcolor(color, value):
        if np.isnan(value):
            return 'w'
        else:
            return list(color[:3]) + [.5]

    def skew_t(t, p):
        t0 = np.log(p / 1050) * 80 / np.log(100 / 1050)
        return t0 + t, p

    figs = []
    for data in dict_list:

        # Retrieve dropsondes data
        df = data['levels'].sort_values('pres', ascending=True)
        Pres = df['pres']
        Temp = df['temp']
        Dwpt = df['dwpt']
        wind_speed = df['wspd']
        wind_dir = df['wdir']
        U, V = wind_components(wind_speed, wind_dir)

        ytop = int(np.nanmin(Pres) - 50)
        yticks = np.arange(1000, ytop, -100)
        xticks = np.arange(-30, 51, 10)

        # Get mandatory and significant wind sub-dataframes
        dfmand = df.loc[df['pres'].isin(
            (1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100))]
        sfc = df.loc[df['hgt'] == 0]
        if len(sfc) > 0:
            SLP = sfc['pres'].values[0]
            dfmand = pd.concat([dfmand, sfc])
            dfmand = dfmand.loc[dfmand['pres'] <= SLP]
        else:
            SLP = None
        dfwind = df.loc[df['pres'] >= 700]

        # Start figure
        fig = plt.figure(figsize=(17, 11), facecolor='w')
        gs = gridspec.GridSpec(2, 3, width_ratios=(2, .2, 1.1), height_ratios=(
            len(dfmand) + 3, len(dfwind) + 3), wspace=0.0)

        ax1 = fig.add_subplot(gs[:, 0])

        # Determine dropsonde time
        try:
            if np.isnan(data["TOPtime"]):
                use_time = data["BOTTOMtime"]
            else:
                use_time = data["TOPtime"]
        except:
            use_time = data["TOPtime"]

        # Add title for main axis
        storm_name_title = storm_name_title.replace("DDD", str(data["obsnum"]))
        storm_name_title = storm_name_title.replace(
            "MMM", str(mission2text(data["mission"])))
        ax1.set_title(storm_name_title, loc='left',
                      fontsize=17, fontweight='bold')
        ax1.set_title(f'Drop time: {time2text(use_time)}' +
                      f'\nDrop location: {location_text(data)}{latlon2text(data["lat"],data["lon"])}', loc='right', fontsize=13)
        plt.yscale('log')
        plt.yticks(yticks, [f'{i:d}' for i in yticks], fontsize=12)
        plt.xticks(xticks, [f'{i:d}' for i in xticks], fontsize=12)
        for y in range(1000, ytop, -50):
            plt.plot([-30, 50], [y] * 2, color='0.5', lw=0.5)
        for x in range(-30 - 80, 50, 10):
            plt.plot([x, x + 80], [1050, 100],
                     color='0.5', linestyle='--', lw=0.5)

        plt.plot(*skew_t(Temp.loc[~np.isnan(Temp)],
                 Pres.loc[~np.isnan(Temp)]), 'o-', color='r')
        plt.plot(*skew_t(Dwpt.loc[~np.isnan(Dwpt)],
                 Pres.loc[~np.isnan(Dwpt)]), 'o-', color='g')
        plt.xlabel(f'Temperature ({degsym}C)', fontsize=13)
        plt.ylabel('Pressure (hPa)', fontsize=13)
        plt.axis([-30, 50, 1050, ytop])

        # Try plotting dropsonde location with respect to storm center (unavailable for realtime)
        try:
            lim = max([i for stage in ('TOP', 'BOTTOM') for i in [
                      1.5 * abs(data[f'{stage}xdist']) + .1, 1.5 * abs(data[f'{stage}ydist']) + .1]])
            iscoords = np.isnan(lim)
            if iscoords:
                lim = 1
            for stage, ycoord in zip(('TOP', 'BOTTOM'), (.8, .05)):
                ax1in1 = ax1.inset_axes([0.05, ycoord, 0.15, 0.15])
                if iscoords:
                    ax1in1.set_title('distance N/A')
                else:
                    ax1in1.scatter(0, 0, c='k')
                    ax1in1.scatter(
                        data[f'{stage}xdist'], data[f'{stage}ydist'], c='w', marker='v', edgecolor='k')
                    ax1in1.set_title(
                        f'{data[f"{stage}distance"]:0.0f} km {deg2dir(90-math.atan2(data[f"{stage}ydist"],data[f"{stage}xdist"])*180/np.pi)}')
                ax1in1.axis([-lim, lim, -lim, lim])
                ax1in1.xaxis.set_major_locator(plt.NullLocator())
                ax1in1.yaxis.set_major_locator(plt.NullLocator())
        except:
            pass

        ax4 = fig.add_subplot(gs[:, 1], sharey=ax1)
        barbs = {k: [v.values[-1]]
                 for k, v in zip(('p', 'u', 'v'), (Pres, U, V))}
        for p, u, v in zip(Pres.values[::-1], U.values[::-1], V.values[::-1]):
            if abs(p - barbs['p'][-1]) > 10 and not np.isnan(u):
                for k, v in zip(('p', 'u', 'v'), (p, u, v)):
                    barbs[k].append(v)
        plt.barbs([.4] * len(barbs['p']), barbs['p'],
                  barbs['u'], barbs['v'], pivot='middle')
        ax4.set_xlim(0, 1)
        ax4.axis('off')

        RH = [rh_from_dp(i, j) for i, j in zip(dfmand['temp'], dfmand['dwpt'])]
        cellText = np.array([['' if np.isnan(i) else f'{int(i)} hPa' for i in dfmand['pres']],
                             ['' if np.isnan(
                                 i) else f'{int(i)} m' for i in dfmand['hgt']],
                             ['' if np.isnan(
                                 i) else f'{i:.1f} {degsym}C' for i in dfmand['temp']],
                             ['' if np.isnan(
                                 i) else f'{int(i)} %' for i in RH],
                             ['' if np.isnan(i) else f'{deg2dir(j)} at {int(i)} kt' for i, j in zip(dfmand['wspd'], dfmand['wdir'])]]).T
        colLabels = ['Pressure', 'Height', 'Temp', 'RH', 'Wind']

        cmap_rh = mlib.cm.get_cmap('BrBG')
        cmap_temp = mlib.cm.get_cmap('RdBu_r')
        cmap_wind = mlib.cm.get_cmap('Purples')

        colors = [['w', 'w', cellcolor(cmap_temp(t / 120 + .5), t),
                   cellcolor(cmap_rh(r / 100), r),
                   cellcolor(cmap_wind(w / 200), w)] for t, r, w in zip(dfmand['temp'], RH, dfmand['wspd'])]

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.xaxis.set_visible(False)  # hide the x axis
        ax2.yaxis.set_visible(False)  # hide the y axis
        TB = ax2.table(cellText=cellText, colLabels=colLabels,
                       cellColours=colors, cellLoc='center', bbox=[0, .05, 1, .95])
        if SLP is not None:
            TB[(len(cellText), 0)].get_text().set_weight('bold')
        ax2.axis('off')
        TB.auto_set_font_size(False)
        TB.set_fontsize(9)
        # TB.scale(3,1.2)
        try:
            ax2.text(
                0, .05, f'\nDeep Layer Mean Wind: {deg2dir(data["DLMdir"])} at {int(data["DLMspd"])} kt', va='top', fontsize=12)
        except:
            ax2.text(0, .05, f'\nDeep Layer Mean Wind: N/A',
                     va='top', fontsize=12)

        ax2.set_title('Generated using Tropycal\nInspired by Tropical Tidbits', fontsize=12,
                      fontweight='bold', color='0.7', loc='right')

        cellText = np.array([[f'{int(i)} hPa' for i, j in zip(dfwind['pres'], dfwind['wspd']) if not np.isnan(j)],
                             [f'{deg2dir(j)} at {int(i)} kt' for i, j in zip(dfwind['wspd'], dfwind['wdir']) if not np.isnan(i)]]).T
        colLabels = ['Pressure', 'Wind']
        colors = [['w', cellcolor(cmap_wind(i / 200), i)]
                  for i in dfwind['wspd'] if not np.isnan(i)]

        ax3 = fig.add_subplot(gs[1, 2])

        try:
            TB = ax3.table(cellText=cellText, colLabels=colLabels,
                           cellColours=colors, cellLoc='center', bbox=[0, .1, 1, .9])
            TB.auto_set_font_size(False)
            TB.set_fontsize(9)
            meanwindoffset = 0
        except:
            meanwindoffset = 0.9
        # TB.scale(2,1.2)
        ax3.xaxis.set_visible(False)  # hide the x axis
        ax3.yaxis.set_visible(False)  # hide the y axis
        ax3.axis('off')

        try:
            ax3.text(0, .1 + meanwindoffset,
                     f'\nMean Wind in Lowest 500 m: {deg2dir(data["MBLdir"])} at {int(data["MBLspd"])} kt', va='top', fontsize=12)
        except:
            ax3.text(0, .1 + meanwindoffset,
                     f'\nMean Wind in Lowest 500 m: N/A', va='top', fontsize=12)
        try:
            ax3.text(0, .1 + meanwindoffset,
                     f'\n\nMean Wind in Lowest 150 m: {deg2dir(data["WL150dir"])} at {int(data["WL150spd"])} kt', va='top', fontsize=12)
        except:
            ax3.text(0, .1 + meanwindoffset,
                     f'\n\nMean Wind in Lowest 150 m: N/A', va='top', fontsize=12)

        figs.append(fig)
        plt.close()

    if len(figs) > 1:
        return figs
    elif len(figs) == 1:
        return fig
    else:
        print("No dropsondes in selection")
