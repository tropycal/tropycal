r"""Functionality for storing and analyzing a year/season of cyclones."""

import numpy as np

# Import internal scripts
from .plot import TrackPlot
from .storm import Storm

# Import tools
from .tools import *
from ..utils import *
from .. import constants


class Season:

    r"""
    Initializes an instance of Season, retrieved via ``TrackDataset.get_season()``.

    Parameters
    ----------
    season : dict
        Dict entry containing all storms within the requested season.
    info : dict
        Dict entry containing general information about the season.

    Returns
    -------
    Season
        Instance of a Season object.
    """

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __add__(self, new):
        # Add seasons

        # Ensure data sources and basins are the same
        if self.source_basin != new.source_basin:
            msg = 'Seasons can only be added for the same basin.'
            raise ValueError(msg)
        if self.source != new.source:
            msg = 'Seasons can only be added from the same source.'
            raise ValueError(msg)

        # Retrieve old & new dict entries
        dict_original = self.dict.copy()
        dict_new = new.dict.copy()

        # Retrieve copy of coordinates
        new_attrs = self.attrs.copy()

        # Add year to list of years
        if isinstance(self.attrs['year'], int):
            new_attrs['year'] = [self.year, new.year]
        else:
            new_attrs['year'].append(new.year)

        # Sort list of years
        new_attrs['year'] = (np.sort(new_attrs['year'])).tolist()

        # Update dict
        dict_original.update(dict_new)

        # Iterate over every year to create a new dict
        new_dict = {}
        for year in new_attrs['year']:
            for key in dict_original.keys():
                if dict_original[key]['season'] == year:
                    new_dict[key] = dict_original[key]

        # Return new Season object
        return Season(new_dict, new_attrs)

    def __repr__(self):

        # Label object
        summary = ["<tropycal.tracks.Season>"]

        # Format keys for summary
        season_summary = self.summary()
        summary_keys = {
            'Total Storms': season_summary['season_storms'],
            'Named Storms': season_summary['season_named'],
            'Hurricanes': season_summary['season_hurricane'],
            'Major Hurricanes': season_summary['season_major'],
            'Season ACE': season_summary['season_ace'],
        }

        # Add season summary
        summary.append("Season Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        # Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.attrs.keys()]) + 3
        for key in self.attrs.keys():
            key_name = key + ":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{self.attrs[key]}')

        return "\n".join(summary)

    def __init__(self, season, info):

        # Save the dict entry of the season
        self.dict = season

        # Add other attributes about the storm
        keys = info.keys()
        self.attrs = {}
        for key in keys:
            if not isinstance(info[key], list) and not isinstance(info[key], dict):
                self[key] = info[key]
                self.attrs[key] = info[key]
            if isinstance(info[key], list) and key == 'year':
                self[key] = info[key]
                self.attrs[key] = info[key]

    def to_dataframe(self):
        r"""
        Converts the season dict into a pandas DataFrame object.

        Returns
        -------
        `pandas.DataFrame`
            A pandas DataFrame object containing information about the season.
        """

        # Try importing pandas
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError(
                "Error: pandas is not available. Install pandas in order to use this function.") from e

        # Get season info
        season_info = self.summary()
        season_info_keys = season_info['id']

        # Set up empty dict for dataframe
        ds = {
            'id': [],
            'name': [],
            'vmax': [],
            'mslp': [],
            'category': [],
            'ace': [],
            'start_time': [],
            'end_time': [],
            'start_lat': [],
            'start_lon': [],
        }

        # Add every key containing a list into the dict
        keys = [k for k in self.dict.keys()]
        for key in keys:
            # Get tropical duration
            temp_type = np.array(self.dict[key]['type'])
            tropical_idx = np.where((temp_type == 'SS') | (temp_type == 'SD') | (temp_type == 'TD') | (
                temp_type == 'TS') | (temp_type == 'HU') | (temp_type == 'TY') | (temp_type == 'ST'))
            if key in season_info_keys:
                sidx = season_info_keys.index(key)
                ds['id'].append(key)
                ds['name'].append(self.dict[key]['name'])
                ds['vmax'].append(season_info['max_wspd'][sidx])
                ds['mslp'].append(season_info['min_mslp'][sidx])
                ds['category'].append(season_info['category'][sidx])
                ds['start_time'].append(
                    np.array(self.dict[key]['time'])[tropical_idx][0])
                ds['end_time'].append(
                    np.array(self.dict[key]['time'])[tropical_idx][-1])
                ds['start_lat'].append(
                    np.array(self.dict[key]['lat'])[tropical_idx][0])
                ds['start_lon'].append(
                    np.array(self.dict[key]['lon'])[tropical_idx][0])
                ds['ace'].append(np.round(season_info['ace'][sidx], 1))

        # Convert entire dict to a DataFrame
        ds = pd.DataFrame(ds)

        # Return dataset
        return ds

    def get_storm_id(self, storm):
        r"""
        Returns the storm ID (e.g., "AL012019") given the storm name and year.

        Parameters
        ----------
        storm : tuple
            Tuple containing the storm name and year (e.g., ("Matthew",2016)).

        Returns
        -------
        str or list
            If a single storm was found, returns a string containing its ID. Otherwise returns a list of matching IDs.
        """

        # Error check
        if not isinstance(storm, tuple):
            raise TypeError("storm must be of type tuple.")
        if len(storm) != 2:
            raise ValueError(
                "storm must contain 2 elements, name (str) and year (int)")
        name, year = storm

        # Search for corresponding entry in keys
        keys_use = []
        for key in self.dict.keys():
            temp_year = self.dict[key]['year']
            if temp_year == year:
                temp_name = self.dict[key]['name']
                if temp_name == name.upper():
                    keys_use.append(key)

        # return key, or list of keys
        if len(keys_use) == 1:
            keys_use = keys_use[0]
        if len(keys_use) == 0:
            raise RuntimeError("Storm not found")
        return keys_use

    def get_storm(self, storm):
        r"""
        Retrieves a Storm object for the requested storm.

        Parameters
        ----------
        storm : str or tuple
            Requested storm. Can be either string of storm ID (e.g., "AL052019"), or tuple with storm name and year (e.g., ("Matthew",2016)).

        Returns
        -------
        tropycal.tracks.Storm
            Object containing information about the requested storm, and methods for analyzing and plotting the storm.
        """

        # Check if storm is str or tuple
        if isinstance(storm, str):
            key = storm
        elif isinstance(storm, tuple):
            key = self.get_storm_id((storm[0], storm[1]))
        else:
            raise RuntimeError(
                "Storm must be a string (e.g., 'AL052019') or tuple (e.g., ('Matthew',2016)).")

        # Retrieve key of given storm
        if isinstance(key, str):
            return Storm(self.dict[key])
        else:
            error_message = ''.join([f"\n{i}" for i in key])
            error_message = f"Multiple IDs were identified for the requested storm. Choose one of the following storm IDs and provide it as the 'storm' argument instead of a tuple:{error_message}"
            raise RuntimeError(error_message)

    def plot(self, domain=None, ax=None, cartopy_proj=None, save_path=None, **kwargs):
        r"""
        Creates a plot of this season.

        Parameters
        ----------
        domain : str
            Domain for the plot. Default is basin-wide. Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of storm track lines. Please refer to :ref:`options-prop` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Retrieve kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Create instance of plot object
        self.plot_obj = TrackPlot()

        if self.basin in ['east_pacific', 'west_pacific', 'south_pacific', 'australia', 'all']:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)

        # Plot storm
        plot_ax = self.plot_obj.plot_season(
            self, domain, ax=ax, save_path=save_path, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def summary(self):
        r"""
        Generates a summary for this season with various cumulative statistics.

        Returns
        -------
        dict
            Dictionary containing various statistics about this season.
        """

        # Determine if season object has a single or multiple seasons
        multi_season = isinstance(self.year, list)

        # Initialize dict with info about all of year's storms
        if not multi_season:
            summary_dict = {
                'id': [],
                'operational_id': [],
                'name': [],
                'max_wspd': [],
                'min_mslp': [],
                'category': [],
                'ace': [],
            }
        else:
            summary_dict = {
                'id': [[] for i in range(len(self.year))],
                'operational_id': [[] for i in range(len(self.year))],
                'name': [[] for i in range(len(self.year))],
                'max_wspd': [[] for i in range(len(self.year))],
                'min_mslp': [[] for i in range(len(self.year))],
                'category': [[] for i in range(len(self.year))],
                'ace': [[] for i in range(len(self.year))],
                'seasons': self.year + [],
                'season_start': [0 for i in range(len(self.year))],
                'season_end': [0 for i in range(len(self.year))],
                'season_storms': [0 for i in range(len(self.year))],
                'season_named': [0 for i in range(len(self.year))],
                'season_hurricane': [0 for i in range(len(self.year))],
                'season_major': [0 for i in range(len(self.year))],
                'season_ace': [0 for i in range(len(self.year))],
                'season_subtrop_pure': [0 for i in range(len(self.year))],
                'season_subtrop_partial': [0 for i in range(len(self.year))],
            }

        # Iterate over season(s)
        list_seasons = [self.year] if not multi_season else self.year + []
        for season_idx, iter_season in enumerate(list_seasons):

            # Search for corresponding entry in keys
            count_ss_pure = 0
            count_ss_partial = 0
            iterate_id = 1
            for key in self.dict.keys():

                # Skip if using multi-season object and storm is outside of this season
                if multi_season and self.dict[key]['season'] != iter_season:
                    continue

                # Retrieve info about storm, only in this basin
                temp_name = self.dict[key]['name']
                temp_vmax = np.array(self.dict[key]['vmax'])
                temp_mslp = np.array(self.dict[key]['mslp'])
                temp_type = np.array(self.dict[key]['type'])
                temp_time = np.array(self.dict[key]['time'])
                temp_basin = np.array(self.dict[key]['wmo_basin'])
                temp_year = np.array([i.year for i in self.dict[key]['time']])

                # Calculate ACE within basin
                temp_ace = 0.0
                for ace_i, (i_time, i_vmax, i_basin, i_type) in enumerate(zip(temp_time, temp_vmax, temp_basin, temp_type)):
                    if self.basin not in ['all', 'both'] and i_basin != self.basin:
                        continue
                    if i_time.strftime('%H%M') not in constants.STANDARD_HOURS:
                        continue
                    if i_type not in constants.NAMED_TROPICAL_STORM_TYPES:
                        continue
                    if self.basin == 'all' and i_time.year != self.year:
                        continue
                    if np.isnan(i_vmax):
                        continue
                    temp_ace += accumulated_cyclone_energy(i_vmax)
                temp_ace = np.round(temp_ace, 1)

                # Get indices of all tropical/subtropical time steps
                if self.basin == 'all':
                    idx = np.where(((temp_type == 'SS') | (temp_type == 'SD') | (temp_type == 'TD') | (temp_type == 'TS') | (
                        temp_type == 'HU') | (temp_type == 'TY') | (temp_type == 'ST')) & (temp_year == self.year))
                elif self.basin == 'both':
                    idx = np.where(((temp_type == 'SS') | (temp_type == 'SD') | (temp_type == 'TD') | (
                        temp_type == 'TS') | (temp_type == 'HU') | (temp_type == 'TY') | (temp_type == 'ST')))
                else:
                    idx = np.where(((temp_type == 'SS') | (temp_type == 'SD') | (temp_type == 'TD') | (temp_type == 'TS') | (
                        temp_type == 'HU') | (temp_type == 'TY') | (temp_type == 'ST')) & (temp_basin == self.basin))

                # Get times during existence of trop/subtrop storms
                if len(idx[0]) == 0:
                    continue
                trop_time = temp_time[idx]

                if not multi_season:
                    if 'season_start' not in summary_dict.keys():
                        summary_dict['season_start'] = trop_time[0]
                    else:
                        if trop_time[0] < summary_dict['season_start']:
                            summary_dict['season_start'] = trop_time[0]
                    if 'season_end' not in summary_dict.keys():
                        summary_dict['season_end'] = trop_time[-1]
                    else:
                        if trop_time[-1] > summary_dict['season_end']:
                            summary_dict['season_end'] = trop_time[-1]
                else:
                    if summary_dict['season_start'][season_idx] == 0:
                        summary_dict['season_start'][season_idx] = trop_time[0]
                    else:
                        if trop_time[0] < summary_dict['season_start'][season_idx]:
                            summary_dict['season_start'][season_idx] = trop_time[0]
                    if summary_dict['season_end'][season_idx] == 0:
                        summary_dict['season_end'][season_idx] = trop_time[-1]
                    else:
                        if trop_time[-1] > summary_dict['season_end'][season_idx]:
                            summary_dict['season_end'][season_idx] = trop_time[-1]

                # Get max/min values and check for nan's
                np_wnd = np.array(temp_vmax[idx])
                np_slp = np.array(temp_mslp[idx])
                if len(np_wnd[~np.isnan(np_wnd)]) == 0:
                    max_wnd = np.nan
                    max_cat = -1
                else:
                    max_wnd = int(np.nanmax(temp_vmax[idx]))
                    max_cat = wind_to_category(np.nanmax(temp_vmax[idx]))
                if len(np_slp[~np.isnan(np_slp)]) == 0:
                    min_slp = np.nan
                else:
                    min_slp = int(np.nanmin(temp_mslp[idx]))

                # Append to dict
                if not multi_season:
                    summary_dict['id'].append(key)
                    summary_dict['name'].append(temp_name)
                    summary_dict['max_wspd'].append(max_wnd)
                    summary_dict['min_mslp'].append(min_slp)
                    summary_dict['category'].append(max_cat)
                    summary_dict['ace'].append(temp_ace)
                    summary_dict['operational_id'].append(
                        self.dict[key]['operational_id'])
                else:
                    summary_dict['id'][season_idx].append(key)
                    summary_dict['name'][season_idx].append(temp_name)
                    summary_dict['max_wspd'][season_idx].append(max_wnd)
                    summary_dict['min_mslp'][season_idx].append(min_slp)
                    summary_dict['category'][season_idx].append(max_cat)
                    summary_dict['ace'][season_idx].append(temp_ace)
                    summary_dict['operational_id'][season_idx].append(
                        self.dict[key]['operational_id'])

                # Handle operational vs. non-operational storms

                # Check for purely subtropical storms
                if 'SS' in temp_type and True not in np.isin(temp_type, list(constants.TROPICAL_ONLY_STORM_TYPES)):
                    count_ss_pure += 1

                # Check for partially subtropical storms
                if 'SS' in temp_type:
                    count_ss_partial += 1

            # Add generic season info
            if not multi_season:
                narray = np.array(summary_dict['max_wspd'])
                narray = narray[~np.isnan(narray)]
                summary_dict['season_storms'] = len(narray[narray >= 0])
                summary_dict['season_named'] = len(narray[narray >= 34])
                summary_dict['season_hurricane'] = len(narray[narray >= 65])
                summary_dict['season_major'] = len(narray[narray >= 100])
                summary_dict['season_ace'] = np.round(
                    np.sum(summary_dict['ace']), 1)
                summary_dict['season_subtrop_pure'] = count_ss_pure
                summary_dict['season_subtrop_partial'] = count_ss_partial
            else:
                narray = np.array(summary_dict['max_wspd'][season_idx])
                narray = narray[~np.isnan(narray)]
                summary_dict['season_storms'][season_idx] = len(
                    narray[narray >= 0])
                summary_dict['season_named'][season_idx] = len(
                    narray[narray >= 34])
                summary_dict['season_hurricane'][season_idx] = len(
                    narray[narray >= 65])
                summary_dict['season_major'][season_idx] = len(
                    narray[narray >= 100])
                summary_dict['season_ace'][season_idx] = np.round(
                    np.sum(summary_dict['ace'][season_idx]), 1)
                summary_dict['season_subtrop_pure'][season_idx] = count_ss_pure
                summary_dict['season_subtrop_partial'][season_idx] = count_ss_partial

        # Return object
        return summary_dict
