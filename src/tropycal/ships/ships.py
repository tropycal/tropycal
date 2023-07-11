r"""Functionality for storing and analyzing SHIPS data."""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as col
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from ..utils import ships_parser, get_colors_sshws, wind_to_category, dynamic_map_extent

class Ships():
    r"""
    Initializes an instance of Ships.
    
    Parameters
    ----------
    content : str
        SHIPS file content. If initialized via a ``tropycal.tracks.Storm`` object, this does not need to be provided.
    
    Other Parameters
    ----------------
    storm_name : str, optional
        If provided, overrides storm name from SHIPS text data with the provided storm name.
    
    Returns
    -------
    tropycal.ships.Ships
        Instance of a Ships object.
    
    Notes
    -----
    A Ships object is best retrieved from a Storm object's ``get_ships()`` method. For example, if the dataset read in is the default North Atlantic and the desired storm is Hurricane Michael (2018), SHIPS data would be retrieved as follows:
    
    .. code-block:: python

        from tropycal import tracks
        import datetime as dt
        
        # Retrieve storm object for Hurricane Michael
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))
        
        # Retrieve instance of a Ships object
        ships = storm.get_ships(dt.datetime(2018, 10, 8, 0))

    """
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):

        # Label object
        summary = ["<tropycal.ships.Ships>"]

        # Format keys for coordinates
        variable_keys = {}
        for key in self.dict.keys():
            dtype = type(self.dict[key][0]).__name__
            dtype = dtype.replace("_", "")
            variable_keys[key] = f"({dtype}) [{self.dict[key][0]} .... {self.dict[key][-1]}]"

        # Add data
        summary.append('\nVariables:')
        add_space = np.max([len(key) for key in variable_keys.keys()]) + 3
        for key in variable_keys.keys():
            key_name = key
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{variable_keys[key]}')
        
        # Add additional information
        summary.append('\nRI Probabilities:')
        add_space = np.max([len(key) for key in self.dict_ri.keys()]) + 3
        for key in self.dict_ri.keys():
            key_name = key + ":"
            entry = f'{self.dict_ri[key]["probability"]}% ({self.dict_ri[key]["prob / climo"]}x climo mean)'
            summary.append(f'{" "*4}{key_name:<{add_space}}{entry}')
            
        # Add additional information
        summary.append('\nAttributes:')
        add_space = np.max([len(key) for key in self.attrs.keys()]) + 3
        for key in self.attrs.keys():
            key_name = key + ":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{self.attrs[key]}')

        return "\n".join(summary)
    
    def __init__(self, content, storm_name=None, forecast_init=None):
        
        ships = ships_parser(content)
        self.dict = ships['data']
        self.dict_ri = ships['data_ri']
        self.attrs = ships['data_attrs']
        
        # Override storm name if requested
        if storm_name is not None: self.attrs['storm_name'] = storm_name

        # Set data variables as attributes of this object
        for key in self.dict:
            self[key] = np.array(self.dict[key])
    
    def get_variables(self):
        r"""
        Return list of available forecast variables.
        
        Returns
        -------
        list
            List containing available forecast variables.
        
        Notes
        -----
        These variables can be accessed from two methods:
        
        >>> # 1. Access directly from object
        >>> ships.fhr
        array([  0,   6,  12,  18,  24,  36,  48,  60,  72,  84,  96, 108, 120])
        
        >>> # 2. Access from internal data dictionary
        >>> ships.dict['fhr']
        array([  0,   6,  12,  18,  24,  36,  48,  60,  72,  84,  96, 108, 120])
        
        """
        
        return list(self.dict.keys())
    
    def get_snapshot(self, hour):
        r"""
        Return all variables valid at a single forecast hour.
        
        Parameters
        ----------
        hour : int
            Requested forecast hour.
        
        Returns
        -------
        dict
            Dictionary containing all variables valid at this forecast hour.
        """
        
        # Error check
        if hour not in self.dict['fhr']:
            raise ValueError('Requested forecast hour is not available.')
        idx = self.dict['fhr'].index(hour)
        
        # Format dict
        data = {}
        for key in self.dict:
            data[key] = self.dict[key][idx]
        
        return data
    
    def get_ri_prob(self):
        r"""
        Return rapid intensification probabilities.
        
        Returns
        -------
        dict
            Dictionary containing rapid intensification probabilities.
        """
        
        return self.dict_ri
    
    def to_xarray(self):
        r"""
        Convert data to an xarray Dataset.
        
        Returns
        -------
        xarray.Dataset
            SHIPS data converted to an xarray Dataset.
        """
        
        # Set up empty dict for dataset
        time = self.dict['fhr']
        ds = {}

        # Add every key containing a list into the dict, otherwise add as an attribute
        keys = [k for k in self.dict.keys() if k != 'fhr']
        for key in keys:
            ds[key] = xr.DataArray(self.dict[key],
                                   coords = [time],
                                   dims = ['fhr'])

        # Convert entire dict to a Dataset
        ds = xr.Dataset(ds, attrs=self.attrs)

        # Return dataset
        return ds
    
    def to_dataframe(self, attrs_as_columns=False):
        r"""
        Convert data to a pandas DataFrame object.

        Parameters
        ----------
        attrs_as_columns : bool
            If True, adds Ships object attributes as columns in the DataFrame returned. Default is False.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame object containing SHIPS data.
        """

        # Set up empty dict for dataframe
        df = {}

        # Add every key containing a list into the dict
        for key in self.dict:
            df[key] = self.dict[key]
        if attrs_as_columns:
            for key in self.attrs:
                df[key] = self.attrs[key]

        # Convert entire dict to a DataFrame
        df = pd.DataFrame(df)

        # Return dataset
        return df
    
    def plot_summary(self):
        r"""
        Generates a plot summarizing the SHIPS forecast.
        
        Returns
        -------
        matplotlib.figure
            Matplotlib figure containing multiple axes.
        """

        # Create the figure and gridspec
        fig = plt.figure(figsize=(9,6),dpi=200)
        gs = gridspec.GridSpec(6, 32)

        # Left column & format Cartopy projection
        plot_lon = np.copy(self.lon)
        if np.nanmax(self.lon) > 150 or np.nanmin(self.lon) < -150:
            proj = ccrs.PlateCarree(central_longitude = 180.0)
            plot_lon[plot_lon < 0] += 360.0
        else:
            proj = ccrs.PlateCarree()
        ax1 = plt.subplot(gs[:3, :14])
        ax2 = plt.subplot(gs[3:, :14], projection=proj)

        # Right column
        ax3 = plt.subplot(gs[:1, 16:])
        ax4 = plt.subplot(gs[1:2, 16:])
        ax5 = plt.subplot(gs[2:3, 16:])
        ax6 = plt.subplot(gs[3:4, 16:])
        ax7 = plt.subplot(gs[4:5, 16:])
        ax8 = plt.subplot(gs[5:6, 16:])

        # ================================================================================================

        # Hide axes boundaries
        for spine in ['top', 'bottom', 'left', 'right']:
            ax1.spines[spine].set_visible(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.tick_params(axis='both', which='both', length=0)

        # Plot title
        ax1.text(0.5, 0.95, f"SHIPS Forecast for {self.attrs['storm_name']}", fontweight='bold',
                 ha='center', va='center', fontsize=14, transform=ax1.transAxes)
        ax1.text(0.5, 0.86, f"Initialized: {self.attrs['forecast_init'].strftime('%H%M UTC %d %b %Y')}",
                 ha='center', va='center', fontsize=9, transform=ax1.transAxes)

        # Current storm information
        ax1.text(0.5, 0.70, 'Current Storm Information:', color='#222286', fontweight='bold',
                 ha='center', va='center', fontsize=9, transform=ax1.transAxes)
        dot = u"\u2022"
        deg = u'\N{DEGREE SIGN}'
        current_info = f'Lat: {self.lat[0]}{deg} {dot} Lon: {self.lon[0]}{deg} {dot} '
        current_info += f'Wind: {int(self.vmax_land_kt[0])} kt'
        ax1.text(0.5, 0.62, current_info,
                 color='#222286', alpha=0.6, ha='center', va='center', fontsize=9, transform=ax1.transAxes)

        # Colormap for RI probabilities
        cmap = plt.cm.Reds
        norm = col.BoundaryNorm(np.arange(0,100,1),cmap.N)

        # Prepare table for RI probabilities
        def format_label(value, entry_label):
            if np.isnan(value):
                return 'N/A'
            return f'{value}{entry_label}'
        data = []
        colors = []
        data.append([i.replace('/','/\n') for i in self.dict_ri.keys()])
        colors.append(['#8CBCE1' for i in self.dict_ri.keys()])
        entries = ['probability', 'prob / climo']
        entries_label = ['%', 'x\nclimo']
        for entry, entry_label in zip(entries, entries_label):
            data.append([format_label(self.dict_ri[i][entry], entry_label) for i in self.dict_ri.keys()])
            if entry == 'prob / climo':
                colors.append(['#fff' for i in self.dict_ri.keys()])
            else:
                colors.append([cmap(norm(self.dict_ri[i][entry]))[:-1] + (0.5,) for i in self.dict_ri.keys()])

        # Set table to occupy bottom half of axes
        table_bbox = [0, 0, 1, 0.40]

        # Insert table into axes
        ax1.text(0.5, 0.46, 'Rapid Intensification Probabilities:', fontweight='bold',
                 ha='center', va='center', fontsize=9, transform=ax1.transAxes)
        table = ax1.table(cellText=data, cellLoc='center', loc='center',
                          cellColours=colors, bbox=table_bbox, edges='closed')

        # Customize table display
        table.set_fontsize(8)
        table.scale(1, 1.5)
        table.auto_set_column_width(range(len(data[0])))

        # ================================================================================================

        # Plot forecast
        ax2.plot(plot_lon, self.lat, color='k', linewidth=1.5, transform=ccrs.PlateCarree())
        for i_lon, i_lat, i_vmax, i_type in zip(plot_lon, self.lat, self.vmax_land_kt, self.storm_type):
            if True in [np.isnan(i) for i in [i_lon, i_lat, i_vmax]]: continue
            marker_type = 'o' if i_type == 'TROP' else '^'
            marker_size = 9 if i_type == 'TROP' else 8
            ax2.plot(i_lon, i_lat, marker_type, mfc=get_colors_sshws(i_vmax),
                     mec='k', mew=0.5, ms=marker_size, transform=ccrs.PlateCarree())
            if i_type != 'TROP': continue
            category = str(wind_to_category(i_vmax))
            if category == "0":
                category = 'S'
            if category == "-1":
                category = 'D'
            ax2.text(i_lon, i_lat, category, ha='center', va='center', fontsize=7, transform=ccrs.PlateCarree())

        # Plot coastlines and political boundaries
        ax2.add_feature(cfeature.STATES.with_scale('50m'), linewidths=0.25, linestyle='solid', edgecolor='#444')
        ax2.add_feature(cfeature.BORDERS.with_scale('50m'), linewidths=0.5, linestyle='solid', edgecolor='#444')
        ax2.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidths=0.5, linestyle='solid', edgecolor='#444')
        ax2.add_feature(cfeature.LAND.with_scale('50m'), facecolor='#EEEEEE', edgecolor='face')

        plot_domain = dynamic_map_extent(np.nanmin(plot_lon),
                                         np.nanmax(plot_lon),
                                         np.nanmin(self.lat),
                                         np.nanmax(self.lat))
        ax2.set_extent(plot_domain)

        # Add lat/lon labels
        gl = ax2.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='dashed')
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
        
        # Add plot credit
        ax2.text(0.02, 0.98, "Plot generated by Tropycal\nInspired by Deelan Jariwala",
                 fontsize=8, alpha=0.6, ha='left', va='top', transform=ax2.transAxes)

        # ================================================================================================

        # Set grid
        ax3.grid(linestyle='dotted')
        ax3.set_xticks(range(0,max(self.fhr)+1,24))
        ax3.set_xticklabels([])
        ax3.tick_params(axis='y', labelsize=8)
        ax3.set_xlim(0,max(self.fhr))

        # Plot vmax
        ax3.plot(self.fhr, self.vmax_land_kt, color='k', label='Max Wind (kt)')
        ax3.plot(self.fhr, self.vmax_noland_kt, color='k', alpha=0.3, label='Max Wind (kt, no land)')
        ax3.legend(fontsize=7)

        # ================================================================================================

        # Set grid
        ax4.grid(linestyle='dotted')
        ax4.set_xticks(range(0,max(self.fhr)+1,24))
        ax4.set_xticklabels([])
        ax4.tick_params(axis='y', labelsize=8)
        ax4.set_xlim(0,max(self.fhr))

        # Plot max potential intensity
        ax4.plot(self.fhr, self.vmax_pot_kt, color='r', label='Max Potential Intensity (kt)')
        ax4.legend(fontsize=7)

        # ================================================================================================

        # Set grid
        ax5.grid(linestyle='dotted')
        ax5.set_xticks(range(0,max(self.fhr)+1,24))
        ax5.set_xticklabels([])
        ax5.tick_params(axis='y', labelsize=8)
        ax5.set_xlim(0,max(self.fhr))

        # Plot shear
        ax5.plot(self.fhr, self.shear_kt, color='b', label='Wind Shear (kt)')
        ax5.legend(fontsize=7)

        # ================================================================================================

        # Set grid
        ax6.grid(linestyle='dotted')
        ax6.set_xticks(range(0,max(self.fhr)+1,24))
        ax6.set_xticklabels([])
        ax6.tick_params(axis='y', labelsize=8)
        ax6.set_xlim(0,max(self.fhr))

        # Plot SSTs
        ax6.plot(self.fhr, self.sst_c, color='orange', label=f'SST ({deg}C)')
        ax6.legend(fontsize=7)

        # ================================================================================================

        # Set grid
        ax7.grid(linestyle='dotted')
        ax7.set_xticks(range(0,max(self.fhr)+1,24))
        ax7.set_xticklabels([])
        ax7.tick_params(axis='y', labelsize=8)
        ax7.set_xlim(0,max(self.fhr))

        # Plot 700-500mb RH
        ax7.plot(self.fhr, self.dict['700_500_rh'], color='g', label='700-500mb RH (%)')
        ax7.legend(fontsize=7)

        # ================================================================================================

        # Set grid
        ax8.grid(linestyle='dotted')
        ax8.set_xticks(range(0,max(self.fhr)+1,24))
        ax8.tick_params(axis='both', labelsize=8)
        ax8.set_xlim(0,max(self.fhr))

        # Plot heat potential
        ax8.plot(self.fhr, self.heat_content, color='purple', label=r'Heat Content (KJ/cm$^2$)')
        ax8.legend(fontsize=7)

        return fig
