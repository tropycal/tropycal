import numpy as np
import warnings
import pkg_resources

from ..utils import *
from .. import constants

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn(
        "Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

try:
    import matplotlib as mlib
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")


class Plot:

    def check_res(self, res):
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

        # Check input map resolution and return corresponding map resolution
        compare_dict = {'l': '110m',
                        'm': '50m',
                        'h': '10m'}
        return compare_dict.get(res, '50m')

    def create_cartopy(self, proj='PlateCarree', mapobj=None, **kwargs):
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

        # Initialize an instance of cartopy if not passed
        if mapobj is None:
            self.proj = getattr(ccrs, proj)(**kwargs)
        else:
            self.proj = mapobj

    def create_geography(self, prop):
        r"""
        Set up the map geography and colors.

        Parameters:
        -----------
        prop : dict
            dict entry containing information about the map geography and colors
        """

        # get resolution corresponding to string in prop
        res = self.check_res(prop['res'])

        # Add "hidden" state alpha prop
        if 'state_alpha' not in prop.keys():
            prop['state_alpha'] = 1.0
        
        # Get zorder kwargs
        zorder = {}
        for key in ['ocean', 'lake', 'continent', 'states', 'countries', 'coastlines']:
            if f'zorder_{key}' in prop.keys():
                zorder[key] = {'zorder': prop[f'zorder_{key}']}
            else:
                zorder[key] = {}

        # fill oceans if specified
        self.ax.set_facecolor(prop['ocean_color'])
        ocean_mask = self.ax.add_feature(cfeature.OCEAN.with_scale(
            res), facecolor=prop['ocean_color'], edgecolor='face', **zorder['ocean'])
        lake_mask = self.ax.add_feature(cfeature.LAKES.with_scale(
            res), facecolor=prop['ocean_color'], edgecolor='face', **zorder['lake'])
        continent_mask = self.ax.add_feature(cfeature.LAND.with_scale(
            res), facecolor=prop['land_color'], edgecolor='face', **zorder['continent'])

        # draw geography
        states = self.ax.add_feature(cfeature.STATES.with_scale(
            res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],
            alpha=prop['state_alpha'], **zorder['states'])
        countries = self.ax.add_feature(cfeature.BORDERS.with_scale(
            res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],
             **zorder['countries'])
        coastlines = self.ax.add_feature(cfeature.COASTLINE.with_scale(
            res), linewidths=prop['linewidth'], linestyle='solid', edgecolor=prop['linecolor'],
             **zorder['coastlines'])

    def dynamic_map_extent(self, min_lon, max_lon, min_lat, max_lat):
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

        return dynamic_map_extent(min_lon, max_lon, min_lat, max_lat)

    def plot_lat_lon_lines(self, bounds, zorder=None, check_prop=False):
        r"""
        Plots parallels and meridians that are constrained by the map bounds.

        Parameters:
        -----------
        bounds : list
            List containing map bounds.
        """
        
        # Skip if map_prop set to not plot this
        if check_prop and not self.map_prop['plot_gridlines']:
            return

        # Suppress gridliner warnings
        warnings.filterwarnings("ignore")

        # Retrieve bounds from list
        bound_w, bound_e, bound_s, bound_n = bounds

        new_xrng = abs(bound_w-bound_e)
        new_yrng = abs(bound_n-bound_s)

        # function to round to nearest number
        def rdown(num, divisor):
            return num - (num % divisor)

        def rup(num, divisor):
            return divisor + (num - (num % divisor))

        # Calculate parallels and meridians
        rthres = 20
        if new_yrng < 160.0 or new_xrng < 160.0:
            rthres = 10
        if new_yrng < 40.0 or new_xrng < 40.0:
            rthres = 5
        if new_yrng < 25.0 or new_xrng < 25.0:
            rthres = 2
        if new_yrng < 9.0 or new_xrng < 9.0:
            rthres = 1
        parallels = np.arange(rdown(bound_s, rthres),
                              rup(bound_n, rthres)+rthres, rthres)
        meridians = np.arange(rdown(bound_w, rthres),
                              rup(bound_e, rthres)+rthres, rthres)

        add_kwargs = {}
        if zorder is not None:
            add_kwargs = {'zorder': zorder}

        # Fix for dateline crossing
        if self.proj.proj4_params['lon_0'] == 180.0:

            # Recalculate parallels and meridians
            parallels = np.arange(rup(bound_s, rthres), rdown(
                bound_n, rthres)+rthres, rthres)
            meridians = np.arange(rup(bound_w, rthres), rdown(
                bound_e, rthres)+rthres, rthres)
            meridians2 = np.copy(meridians)
            meridians2[meridians2 >
                       180.0] = meridians2[meridians2 > 180.0]-360.0
            all_meridians = np.arange(-180.0, 180.0+rthres, rthres)
            all_parallels = np.arange(
                rdown(-90.0, rthres), 90.0+rthres, rthres)

            # First call with no labels but gridlines plotted
            gl1 = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, xlocs=all_meridians,
                                    ylocs=all_parallels, linewidth=1.0, color='k', alpha=0.5, linestyle='dotted', **add_kwargs)
            # Second call with labels but no gridlines
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, xlocs=meridians,
                                   ylocs=parallels, linewidth=0.0, color='k', alpha=0.0, linestyle='dotted', **add_kwargs)

            # this syntax is deprecated in newer functions of cartopy
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
            # Add meridians and parallels
            gl = self.ax.gridlines(crs=ccrs.PlateCarree(
            ), draw_labels=True, linewidth=1.0, color='k', alpha=0.5, linestyle='dotted', **add_kwargs)

            # this syntax is deprecated in newer functions of cartopy
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

        # Reset plot bounds
        self.ax.set_extent(
            [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

    def plot_init(self, ax, map_prop, plot_geography=True):
        r"""
        Initializes the plot by creating a cartopy and axes instance, if one hasn't been created yet, and adds geography.

        Parameters:
        -----------
        ax : axes
            Instance of axes
        map_prop : dict
            Dictionary of map properties
        """
        
        #Set default map properties
        default_map_prop = {'res': 'm', 'land_color': '#FBF5EA', 'ocean_color': '#EDFBFF',
                            'linewidth': 0.5, 'linecolor': 'k', 'figsize': (14, 9),
                            'dpi': 200, 'plot_gridlines': True}
        self.map_prop = self.add_prop(map_prop, default_map_prop)

        # create cartopy projection, if none existing
        if self.proj is None:
            self.create_cartopy(proj='PlateCarree', central_longitude=0.0)

        # create figure
        if ax is None:
            self.fig = plt.figure(
                figsize=self.map_prop['figsize'], dpi=self.map_prop['dpi'])
            self.ax = plt.axes(projection=self.proj)
        else:
            fig_numbers = [
                x.num for x in mlib._pylab_helpers.Gcf.get_all_fig_managers()]
            if len(fig_numbers) > 0:
                self.fig = plt.figure(fig_numbers[-1])
            else:
                self.fig = plt.figure(
                    figsize=self.map_prop['figsize'], dpi=self.map_prop['dpi'])
            self.ax = ax

        # Attach geography to plot, lat/lon lines, etc.
        if plot_geography:
            self.create_geography(self.map_prop)

    def add_prop(self, input_prop, default_prop):
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

        # add kwargs to prop and map_prop
        for key in input_prop.keys():
            default_prop[key] = input_prop[key]

        # Return prop
        return default_prop

    def set_projection(self, domain):
        r"""
        Sets a predefined map projection domain.

        Parameters
        ----------
        domain : str
            Name of map projection to domain over.
        """

        # North Atlantic plot domain
        if domain == "both":
            bound_w = -179.0+360.0
            bound_e = -15.0+360.0
            bound_s = 0.0
            bound_n = 70.0

        # North Atlantic plot domain
        elif domain == "north_atlantic":
            bound_w = -105.0
            bound_e = -5.0
            bound_s = 0.0
            bound_n = 65.0

        # South Atlantic plot domain
        elif domain == "south_atlantic":
            bound_w = -105.0
            bound_e = -5.0
            bound_s = -65.0
            bound_n = 0.0

        # East Pacific plot domain
        elif domain == "east_pacific":
            bound_w = -180.0+360.0
            bound_e = -80+360.0
            bound_s = 0.0
            bound_n = 65.0

        # West Pacific plot domain
        elif domain == "west_pacific":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = 0.0
            bound_n = 58.0

        # North Indian plot domain
        elif domain == "north_indian":
            bound_w = 35.0
            bound_e = 110.0
            bound_s = -5.0
            bound_n = 42.0

        # South Indian plot domain
        elif domain == "south_indian":
            bound_w = 20.0
            bound_e = 110.0
            bound_s = -50.0
            bound_n = 5.0

        # Australia plot domain
        elif domain == "australia":
            bound_w = 90.0
            bound_e = 180.0
            bound_s = -60.0
            bound_n = 0.0

        # South Pacific plot domain
        elif domain == "south_pacific":
            bound_w = 140.0
            bound_e = -120.0+360.0
            bound_s = -65.0
            bound_n = 0.0

        # Global plot domain
        elif domain == "all":
            bound_w = 0.1
            bound_e = 360.0
            bound_s = -90.0
            bound_n = 90.0

        # CONUS plot domain
        elif domain == "conus":
            bound_w = -130.0
            bound_e = -65.0
            bound_s = 20.0
            bound_n = 50.0

        # CONUS plot domain
        elif domain == "east_conus":
            bound_w = -105.0
            bound_e = -60.0
            bound_s = 20.0
            bound_n = 48.0

        # Custom domain
        else:

            # Error check
            if not isinstance(domain, dict):
                msg = "Custom domains must be of type dict."
                raise TypeError(msg)

            # Retrieve map bounds
            keys = domain.keys()
            check = [False, False, False, False]
            for key in keys:
                if key[0].lower() == 'n':
                    check[0] = True
                    bound_n = domain[key]
                if key[0].lower() == 's':
                    check[1] = True
                    bound_s = domain[key]
                if key[0].lower() == 'e':
                    check[2] = True
                    bound_e = domain[key]
                if key[0].lower() == 'w':
                    check[3] = True
                    bound_w = domain[key]
            if False in check:
                msg = "Custom domains must be of type dict with arguments for 'n', 's', 'e' and 'w'."
                raise ValueError(msg)

        # Set map extent
        self.ax.set_extent(
            [bound_w, bound_e, bound_s, bound_n], crs=ccrs.PlateCarree())

        return bound_w, bound_e, bound_s, bound_n

    def plot_credit(self):

        return "Plot generated using troPYcal"

    def add_credit(self, text):

        if self.use_credit:
            a = self.ax.text(0.99, 0.01, text, fontsize=10, color='k', alpha=0.7, fontweight='bold',
                             transform=self.ax.transAxes, ha='right', va='bottom', zorder=10)
            a.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                                path_effects.Normal()])

    def format_storm_title(self, storm_data, calculate_extrema=True):
        r"""
        Formats the title for a storm (e.g., Tropical Storm, Invest).

        Parameters
        ----------
        storm_data : dict
            Storm entry dictionary, must contain keys 'id' for storm ID and 'name' for storm name
        calculate_extrema : bool, optional
            If True, also calculates storm extrema. Default is False.

        Returns
        -------
        dict
            Dictionary containing the formatted storm title, and extrema if requested.
        """

        # Check where storm is a tropical or subtropical cyclone
        type_array = np.array(storm_data['type'])
        idx = np.where(np.isin(type_array, list(constants.TROPICAL_STORM_TYPES)))

        # Check if storm is an invest with a leading 9
        flag_invest = False
        flag_ptc = False
        if len(storm_data['id']) > 4 and str(storm_data['id'][2]) == "9":
            flag_invest = True

        # Check if storm is classified as an invest
        invest_tag = False
        if 'invest' in storm_data.keys() and storm_data['invest']:
            invest_tag = True

        # Case 1 - No ID is available and type is unknown
        if len(storm_data['id']) == 0 and len(idx[0]) == 0:
            idx = np.array([True for i in type_array])
            storm_name = f"Cyclone {storm_data['name']}"

        # Case 2 - Not an invest, but might be a potential tropical cyclone
        elif not flag_invest and (invest_tag == False or len(idx[0]) > 0):
            tropical_vmax = np.array(storm_data['vmax'])[idx]

            # Coerce to include non-TC points if storm hasn't been designated yet
            if len(tropical_vmax) == 0 and len(storm_data['id']) > 4:
                flag_ptc = True
                idx = np.where((type_array == 'LO') | (type_array == 'DB'))
                tropical_vmax = np.array(storm_data['vmax'])[idx]

            # Case 2a: No wind data available
            if all_nan(tropical_vmax):
                storm_name = f"Tropical Cyclone {storm_data['name']}"

            # Case 2b: Potential tropical cyclone
            elif flag_ptc:
                storm_name = f"Potential Tropical Cyclone {storm_data['name']}"

            # Case 2c: tropical cyclone
            else:
                subtrop = classify_subtropical(
                    np.array(storm_data['type']))
                peak_idx = storm_data['vmax'].index(
                    np.nanmax(tropical_vmax))
                peak_basin = storm_data['wmo_basin'][peak_idx]
                storm_type = get_storm_classification(
                    np.nanmax(tropical_vmax), subtrop, peak_basin)
                storm_name = f"{storm_type} {storm_data['name']}"

        # Case 3 - storm is an invest
        else:
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
            storm_name = f"Invest {storm_data['id'][2:4]}{add_letter}"

        data = {
            'name': storm_name,
        }

        #-----------------------------------------------------------------------

        # Get storm extrema
        if calculate_extrema:
            ace = storm_data['ace'] if not flag_ptc else 0.0

            # Get MSLP extrema
            mslp_key = 'mslp' if 'wmo_mslp' not in storm_data.keys() else 'wmo_mslp'
            if all_nan(np.array(storm_data[mslp_key])[idx]):
                min_pres = "N/A"
            else:
                min_pres = int(np.nan_to_num(np.nanmin(np.array(storm_data[mslp_key])[idx])))

            # Get wind extrema
            if all_nan(np.array(storm_data['vmax'])[idx]):
                max_wind = "N/A"
            else:
                max_wind = int(np.nan_to_num(np.nanmax(np.array(storm_data['vmax'])[idx])))

            # Get start and end times
            start_time = dt.strftime(np.array(storm_data['time'])[idx][0], '%d %b %Y')
            end_time = dt.strftime(np.array(storm_data['time'])[idx][-1], '%d %b %Y')

            data['ace'] = ace
            data['mslp'] = str(min_pres)
            data['vmax'] = str(max_wind)
            data['start_time'] = start_time
            data['end_time'] = end_time

        # Return all data
        return data

