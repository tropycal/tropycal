import warnings
from datetime import datetime as dt

from ..plot import Plot

# Import tools
from .tools import *
from ..utils import *

try:
    from cartopy import crs as ccrs
except:
    warnings.warn(
        "Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

except:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


class TornadoPlot(Plot):

    def __init__(self):

        self.use_credit = True

    def plot_tornadoes(self, tornado, domain="east_conus", plotPPH=False, ax=None, return_ax=False, return_domain=False, prop={}, map_prop={}):
        r"""
        Creates a plot of a single storm track.

        Parameters
        ----------
        storm : str, tuple or dict
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), tuple with storm name and year (e.g., ("Matthew",2016)), or a dict entry.
        domain : str
            Domain for the plot. Can be one of the following:

            * **dynamic** - default. Dynamically focuses the domain using the tornado track(s) plotted.
            * **north_atlantic** - North Atlantic Ocean basin.
            * **conus** - Contiguous United States.
            * **east_conus** - Eastern Contiguous United States and western Atlantic.
            * **lonW/lonE/latS/latN** - Custom plot domain.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            Whether to return axis at the end of the function. If false, plot will be displayed on the screen. Default is false.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """

        # Set default properties
        default_prop = {'plotType': 'tracks', 'PPHcolors': 'spc', 'PPHlevels': [2, 5, 10, 15, 30, 45, 60, 100],
                        'EFcolors': 'default', 'linewidth': 1.5, 'ms': 7.5}
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9), 'dpi': 200}

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

        # Check for storm type, then get data for storm
        try:
            tornado_data = tornado
        except:
            raise RuntimeError("Error: tornado must be dataframe")

        # Retrieve storm data
        slat = tornado_data['slat']
        slon = tornado_data['slon']
        elat = tornado_data['elat']
        elon = tornado_data['elon']
        mag = tornado_data['mag']

        mnlat = (slat+elat)*.5
        mnlon = (slon+elon)*.5

        # Add to coordinate extrema
        if max_lat is None:
            max_lat = max(mnlat)
        else:
            if max(mnlat) > max_lat:
                max_lat = max(mnlat)
        if min_lat is None:
            min_lat = min(mnlat)
        else:
            if min(mnlat) < min_lat:
                min_lat = min(mnlat)
        if max_lon is None:
            max_lon = max(mnlon)
        else:
            if max(mnlon) > max_lon:
                max_lon = max(mnlon)
        if min_lon is None:
            min_lon = min(mnlon)
        else:
            if min(mnlon) < min_lon:
                min_lon = min(mnlon)

        # Plot PPH
        if plotPPH in ['total', 'daily', True]:
            if plotPPH == True:
                plotPPH = 'daily'
            PPH, longrid, latgrid = getPPH(tornado_data, method=plotPPH)

            colors, clevs = get_colors_pph(
                plotPPH, prop['PPHcolors'], prop['PPHlevels'])

            cbmap = self.ax.contourf(longrid, latgrid, PPH,
                                     levels=clevs, colors=colors, alpha=0.5)

        # Plot tornadoes as specified
        EFcolors = get_colors_ef(prop['EFcolors'])

        tornado_data = tornado_data.sort_values('mag')
        for _, row in tornado_data.iterrows():
            plt.plot([row['slon'], row['elon']+.01], [row['slat'], row['elat']+.01],
                     lw=prop['linewidth'], color=EFcolors[row['mag']],
                     path_effects=[path_effects.Stroke(
                         linewidth=prop['linewidth']*1.5, foreground='w'), path_effects.Normal()],
                     transform=ccrs.PlateCarree())

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

        # Determine number of lat/lon lines to use for parallels & meridians
        self.plot_lat_lon_lines([bound_w, bound_e, bound_s, bound_n])

        # --------------------------------------------------------------------------------------

        # Add left title
        PPH_title = ''
        if plotPPH in ['total', True]:
            PPH_title = ' and total PPH (%)'
        if plotPPH == 'daily':
            PPH_title = ' and daily PPH (%)'
        self.ax.set_title('Tornado tracks'+PPH_title,
                          loc='left', fontsize=17, fontweight='bold')

        # Add right title
        # max_PPH = max(PPH)
        start_time = dt.strftime(
            min(tornado_data['UTC_time']), '%H:%M UTC %d %b %Y')
        end_time = dt.strftime(
            max(tornado_data['UTC_time']), '%H:%M UTC %d %b %Y')
        self.ax.set_title(
            f'Start ... {start_time}\nEnd ... {end_time}', loc='right', fontsize=13)

        # --------------------------------------------------------------------------------------

        # Add legend
        handles = []
        for ef, color in enumerate(EFcolors):
            count = len(tornado_data[tornado_data['mag'] == ef])
            handles.append(mlines.Line2D([], [], linestyle='-',
                           color=color, label=f'EF-{ef} ({count})'))
        leg_tor = self.ax.legend(
            handles=handles, loc='lower left', fancybox=True, framealpha=0, fontsize=11.5)
        leg_tor.set_zorder(101)
        plt.draw()

        # Get the bbox
        try:
            bb = leg_tor.legendPatch.get_bbox().inverse_transformed(self.fig.transFigure)
        except:
            bb = leg_tor.legendPatch.get_bbox().transformed(self.fig.transFigure.inverted())
        bb_ax = self.ax.get_position()

        rectangle = mpatches.Rectangle((bb_ax.x0, bb_ax.y0), bb.width+bb.x0-bb_ax.x0, bb.height+2*bb.y0-2*bb_ax.y0,
                                       fc='w', edgecolor='0.8', alpha=0.8,
                                       transform=self.fig.transFigure, zorder=100)

        # Add PPH colorbar
        if plotPPH != False:

            # Define colorbar axis
            cax = self.fig.add_axes(
                [bb.width + 3*bb.x0 - 2*bb_ax.x0, bb.y0, 0.015, bb.height])
            cbar = self.fig.colorbar(cbmap, cax=cax, orientation='vertical')
            iticks = round(len(clevs)/len(cbar.ax.get_yticks()))
            cbar.ax.set_yticklabels([round(clevs[i], 1) for i in range(
                0, len(clevs), iticks)], fontsize=11.5, color='k')

            rectangle = mpatches.Rectangle((bb_ax.x0, bb_ax.y0), bb.width+bb.x0-bb_ax.x0+.06, bb.height+2*bb.y0-2*bb_ax.y0,
                                           fc='w', edgecolor='0.8', alpha=0.8,
                                           transform=self.fig.transFigure, zorder=100)

        self.ax.add_patch(rectangle)

        # add credit
        text = self.plot_credit()
        self.add_credit(text)

        # Return axis if specified, otherwise display figure
        if ax is not None or return_ax:
            if return_domain:
                return self.ax, leg_tor, {'n': bound_n, 'e': bound_e, 's': bound_s, 'w': bound_w}
            else:
                return self.ax, leg_tor
        else:
            plt.show()
            plt.close()
