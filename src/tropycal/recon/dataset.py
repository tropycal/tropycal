import os
import numpy as np
from datetime import datetime as dt, timedelta
import pandas as pd
import requests
import pickle
import copy
import urllib3

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d as gfilt1d
from scipy.ndimage import minimum_filter
import matplotlib.dates as mdates

try:
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
except ImportError:
    warnings.warn(
        "Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

from .plot import *
from ..tracks.plot import TrackPlot
from .realtime import Mission

# Import tools
from .tools import *
from ..utils import *


class ReconDataset:

    r"""
    Creates an instance of a ReconDataset object containing all recon data for a single storm.

    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Requested Storm object.

    Returns
    -------
    ReconDataset
        An instance of ReconDataset.

    Notes
    -----
    .. warning::

        Recon data is currently only available from 1989 onwards.

    ReconDataset and its subclasses (hdobs, dropsondes and vdms) consist the **storm-centric** part of the recon module, meaning that recon data is retrieved specifically for tropical cyclones, and all recon missions for the requested storm are additionally transformed to storm-centric coordinates. This differs from realtime recon functionality, which is **mission-centric**.

    This storm-centric functionality allows for additional recon analysis and visualization functions, such as derived hovmollers and spatial maps for example. As of Tropycal v0.4, Recon data can only be retrieved for tropical cyclones, not for invests.

    ReconDataset will contain nothing the first time it's initialized, but contains methods to retrieve the three sub-classes of recon:

    .. list-table:: 
       :widths: 25 75
       :header-rows: 1

       * - Class
         - Description
       * - hdobs
         - Class containing all High Density Observations (HDOBs) for this Storm.
       * - dropsondes
         - Class containing all dropsondes for this Storm.
       * - vdms
         - Class containing all Vortex Data Messages (VDMs) for this Storm.

    Each of these sub-classes can be initialized as a sub-class of ReconDataset as follows. Note that this may take some time, especially for storms with many recon missions.

    .. code-block:: python

        #Retrieve Hurricane Michael (2018) from TrackDataset
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))

        #Retrieve all HDOBs for this storm
        storm.recon.get_hdobs()

        #Retrieve all dropsondes for this storm
        storm.recon.get_dropsondes()

        #Retrieve all VDMs for this storm
        storm.recon.get_vdms()

    Once this data has been read in, these subclasses and their associated methods and attributes can be accessed from within the `recon` object as follows, using HDOBs for example:

    .. code-block:: python

        #Retrieve Pandas DataFrame of all HDOB observations
        storm.recon.hdobs.data

        #Plot all HDOB points
        storm.recon.hdobs.plot_points()

        #Plot derived hovmoller from HDOB points
        storm.recon.hdobs.plot_hovmoller()

    Individual missions can also be retrieved as ``Mission`` objects. For example, this code retrieves a ``Mission`` object for the second mission of this storm:

    .. code-block:: python

        #This line prints all available mission IDs from this storm
        print(storm.recon.find_mission())

        #This line retrieves the 2nd mission from this storm
        mission = storm.recon.get_mission(2)

    """

    def __repr__(self):

        info = []
        for name in ['hdobs', 'dropsondes', 'vdms']:
            try:
                info.append(self.__dict__[name].__repr__())
            except:
                info.append('')
        return '\n'.join(info)

    def __init__(self, storm):

        self.source = 'https://www.nhc.noaa.gov/archive/recon/'
        self.storm = storm

    def get_hdobs(self, data=None):
        r"""
        Retrieve High Density Observations (HDOBs) for this storm.

        Parameters
        ----------
        data : str, optional
            String representing the path of a pickle file containing HDOBs data, saved via ``hdobs.to_pickle()``. If none, data is read from NHC.

        Notes
        -----
        This function has no return value, but stores the resulting HDOBs object within this ReconDataset instance. All of its methods can then be accessed as follows, for the following example storm:

        .. code-block:: python

            from tropycal import tracks

            #Read basin dataset
            basin = tracks.TrackDataset()

            #Read storm object
            storm = basin.get_storm(('michael',2018))

            #Read hdobs data
            storm.recon.get_hdobs()

            #Plot all HDOB points
            storm.recon.hdobs.plot_points()
        """

        self.hdobs = hdobs(self.storm, data)

    def get_dropsondes(self, data=None):
        r"""
        Retrieve dropsondes for this storm.

        Parameters
        ----------
        data : str, optional
            String representing the path of a pickle file containing dropsonde data, saved via ``dropsondes.to_pickle()``. If none, data is read from NHC.

        Notes
        -----
        This function has no return value, but stores the resulting dropsondes object within this ReconDataset instance. All of its methods can then be accessed as follows, for the following example storm:

        .. code-block:: python

            from tropycal import tracks

            #Read basin dataset
            basin = tracks.TrackDataset()

            #Read storm object
            storm = basin.get_storm(('michael',2018))

            #Read dropsondes data
            storm.recon.get_dropsondes()

            #Plot all dropsonde points
            storm.recon.dropsondes.plot_points()
        """

        self.dropsondes = dropsondes(self.storm, data)

    def get_vdms(self, data=None):
        r"""
        Retrieve Vortex Data Messages (VDMs) for this storm.

        Parameters
        ----------
        data : str, optional
            String representing the path of a pickle file containing VDM data, saved via ``vdms.to_pickle()``. If none, data is read from NHC.

        Notes
        -----
        This function has no return value, but stores the resulting VDMs object within this ReconDataset instance. All of its methods can then be accessed as follows, for the following example storm:

        .. code-block:: python

            from tropycal import tracks

            #Read basin dataset
            basin = tracks.TrackDataset()

            #Read storm object
            storm = basin.get_storm(('michael',2018))

            #Read VDM data
            storm.recon.get_vdms()

            #Plot all VDM points
            storm.recon.vdms.plot_points()
        """

        self.vdms = vdms(self.storm, data)

    def get_mission(self, number):
        r"""
        Retrieve a Mission object for a given mission number for this storm.

        Parameters
        ----------
        number : int or str
            Requested mission number. Can be an integer (1) or a string with two characters ("01").

        Returns
        -------
        Mission
            Instance of a Mission object for the requested mission.
        """

        def str2(number):
            if isinstance(number, str):
                return number
            if number < 10:
                return f"0{number}"
            return str(number)

        # Automatically retrieve data if not already available
        try:
            self.vdms
        except:
            self.get_vdms()
        try:
            self.hdobs
        except:
            self.get_hdobs()
        try:
            self.dropsondes
        except:
            self.get_dropsondes()

        # Search through all missions to find the full mission ID
        missions = []
        for mission in np.unique(self.hdobs.data['mission']):
            try:
                missions.append(int(mission))
            except:
                pass
        missions = list(np.sort(missions))
        if isinstance(number, str):
            missions = [str2(i) for i in missions]
        if number not in missions:
            raise ValueError("Requested mission ID is not available.")

        # Retrieve data for mission
        hdobs_mission = self.hdobs.data.loc[self.hdobs.data['mission'] == str2(
            number)]
        mission_id = hdobs_mission['mission_id'].values[0]
        vdms_mission = [
            i for i in self.vdms.data if i['mission_id'] == mission_id]
        dropsondes_mission = [
            i for i in self.dropsondes.data if i['mission_id'] == mission_id]

        mission_dict = {
            'hdobs': hdobs_mission,
            'vdms': vdms_mission,
            'dropsondes': dropsondes_mission,
            'aircraft': mission_id.split("-")[0],
            'storm_name': mission_id.split("-")[2]
        }

        # Get sources
        try:
            sources = list(
                np.unique([self.vdms.source, self.hdobs.source, self.dropsondes.source]))
            if len(sources) == 1:
                sources = sources[0]
        except:
            sources = 'National Hurricane Center (NHC)'
        mission_dict['source'] = sources

        return Mission(mission_dict, mission_id)

    def update(self):
        r"""
        Update with the latest data for an ongoing storm.

        Notes
        -----
        This function has no return value, but simply updates all existing sub-classes of ReconDataset.
        """

        for name in ['hdobs', 'dropsondes', 'vdms']:
            try:
                self.__dict__[name].update()
            except:
                print(f'No {name} object to update')

    def get_track(self, time=None):
        r"""
        Retrieve coordinates of recon track for one or more times.

        Parameters
        ----------
        time : datetime.datetime or list, optional
            Datetime object or list of datetime objects representing the requested time.

        Returns
        -------
        tuple
            (lon,lat) coordinates.

        Notes
        -----
        The track from which coordinate(s) are returned is generated by an optimal combination of Best Track and Recon (VDMs and/or HDOBs) tracks.
        """

        if time is None or 'trackfunc' not in self.__dict__.keys():
            btk = self.storm.to_dataframe()[['time', 'lon', 'lat']]

            try:
                if 'vdms' not in self.__dict__.keys():
                    print('Getting VDMs for track')
                    self.get_vdms()
                rec = pd.DataFrame(
                    [{k: d[k] for k in ('time', 'lon', 'lat')} for d in self.vdms.data])
            except:
                try:
                    rec = storm.recon.hdobs.sel(iscenter=1).data
                except:
                    rec = None

            if rec is None:
                track = copy.copy(btk)
            else:
                track = copy.copy(rec)
                for i, row in btk.iterrows():
                    if min(abs(row['time'] - rec['time'])) > timedelta(hours=3):
                        track.loc[len(track.index)] = row
            track = track.sort_values(by='time').reset_index(drop=True)

            # Interpolate center position to time of each ob
            datenum = [mdates.date2num(t) for t in track['time']]
            f1 = interp1d(
                datenum, track['lon'], fill_value='extrapolate', kind='quadratic')
            f2 = interp1d(
                datenum, track['lat'], fill_value='extrapolate', kind='quadratic')
            self.trackfunc = (f1, f2)

        if time is not None:
            datenum = []
            for t in time:
                try:
                    datenum.append(mdates.date2num(t))
                except:
                    datenum.append(np.nan)
            track = tuple([f(datenum) for f in self.trackfunc])
            return track

    def find_mission(self, time=None, distance=None):
        r"""
        Returns the name of a mission or list of recon missions for this storm.

        Parameters
        ----------
        time : datetime.datetime or list, optional
            Datetime object or list of datetime objects representing the time of the requested mission. If none, all missions will be returned.
        distance : int, optional
            Distance from storm center, in kilometers.

        Returns
        -------
        list
            The IDs of any/all missions that had in-storm observations during the specified time.

        Notes
        -----
        Especially in earlier years, missions are not always numbered sequentially (e.g., the first mission might not have an ID of "01").

        To get a ``Mission`` object for one or more mission IDs, use the mission ID as an argument in ``ReconDataset.get_mission()``. For example, to retrieve a Mission object for every mission valid at a requested time, assuming that ``ReconDataset`` is linked to a Storm object:

        .. code-block:: python

            #Enter a requested time here
            import datetime as dt
            requested_time = dt.datetime(2020,8,12,12) #enter your requested time here

            #Get all active mission ID(s), if any, for this time
            mission_ids = storm.recon.find_mission(requested_time)

            #Get Mission object for each mission
            for mission_id in mission_ids:
                mission = storm.recon.get_mission(mission_id)
                print(mission)
        """

        # Return all missions if time is None
        if time is None:
            if distance is None:
                data = self.hdobs.data
            else:
                data = self.hdobs.sel(distance=distance).data
            missions = np.unique(data['mission'].values)
            return list(missions)

        # Filter temporally
        if isinstance(time, list):
            t1 = min(time)
            t2 = max(time)
        else:
            t1 = t2 = time

        # Filter spatially
        if distance is None:
            data = self.hdobs.data
        else:
            data = self.hdobs.sel(distance=distance).data

        # Find and return missions
        selected = []
        mission_groups = data.groupby('mission')
        for g in mission_groups:
            t_start, t_end = (min(g[1]['time']), max(g[1]['time']))
            if t_start <= t1 <= t_end or t_start <= t2 <= t_end or t1 < t_start < t2:
                selected.append(g[0])
        return selected

    def plot_summary(self, mission=None, save_path=None):
        r"""
        Plot summary map of all recon data.

        Parameters
        ----------
        mission : str, optional
            String with mission name. Will plot summary for the specified mission, otherwise plots for all missions (default).
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Returns
        -------
        ax
            Instance of axes containing the plot.

        Notes
        -----
        HDOB data needs to be read into the recon object to use this function. To do so, use the ``ReconDataset.get_hdobs()`` function.
        """

        # Error check
        if 'hdobs' not in self.__dict__.keys():
            raise RuntimeError(
                "hdobs needs to be read into the 'recon' object first. Use the 'ReconDataset.get_hdobs()' method to read in HDOBs data.")

        prop = {
            'hdobs': {'ms': 5, 'marker': 'o'},
            'dropsondes': {'ms': 25, 'marker': 'v'},
            'vdms': {'ms': 100, 'marker': 's'}
        }

        hdobs = self.hdobs.sel(mission=mission)
        ax = hdobs.plot_points('pkwnd', prop={'cmap': {
                               1: 'firebrick', 2: 'tomato', 4: 'gold', 6: 'lemonchiffon'}, 'levels': (0, 200), 'ms': 2})

        if 'dropsondes' in self.__dict__.keys():
            dropsondes = self.dropsondes.sel(mission=mission)
            drop_lons = []
            drop_lats = []
            for drop in dropsondes.data:
                if np.isnan(drop['TOPlon']):
                    drop_lons.append(drop['lon'])
                else:
                    drop_lons.append(drop['TOPlon'])
                if np.isnan(drop['TOPlat']):
                    drop_lats.append(drop['lat'])
                else:
                    drop_lats.append(drop['TOPlat'])
            ax.scatter(drop_lons, drop_lats, s=50, marker='v', edgecolor='w',
                       linewidth=0.5, color='darkblue', transform=ccrs.PlateCarree())

        if 'vdms' in self.__dict__.keys():
            vdms = self.vdms.sel(mission=mission)
            ax.scatter(*zip(*[(d['lon'], d['lat']) for d in vdms.data]), s=80, marker='H',
                       edgecolor='w', linewidth=1, color='k', transform=ccrs.PlateCarree())

        title_left = ax.get_title(loc='left').split('\n')
        newtitle = title_left[0] + '\nRecon summary' + \
            ['', f' for mission {mission}'][mission is not None]
        ax.set_title(newtitle, fontsize=17, fontweight='bold', loc='left')

        if save_path is not None and isinstance(save_path, str):
            plt.savefig(save_path, bbox_inches='tight')

        return ax


class hdobs:

    r"""
    Creates an instance of an HDOBs object containing all recon High Density Observations (HDOBs) for a single storm.

    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Requested storm.
    data : str, optional
        Filepath of pickle file containing HDOBs data retrieved from ``hdobs.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.
    update : bool
        If True, search for new data, following existing data in the dropsonde object, and concatenate. Default is False.

    Returns
    -------
    Dataset
        An instance of HDOBs, initialized with a dataframe of HDOB.

    Notes
    -----
    .. warning::

        Recon data is currently only available from 1989 onwards.

    There are two recommended ways of retrieving an hdob object. Since the ``ReconDataset``, ``hdobs``, ``dropsondes`` and ``vdms`` classes are **storm-centric**, a Storm object is required for both methods.

    .. code-block:: python

        #Retrieve Hurricane Michael (2018) from TrackDataset
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))

    The first method is to use the empty instance of ReconDataset already initialized in the Storm object, which has a ``get_hdobs()`` method thus allowing all of the hdobs attributes and methods to be accessed from the Storm object. As a result, a Storm object does not need to be provided as an argument.

    .. code-block:: python

        #Retrieve all HDOBs for this storm
        storm.recon.get_hdobs()

        #Retrieve the raw HDOBs data
        storm.recon.hdobs.data

        #Use the plot_points() method of hdobs
        storm.recon.hdobs.plot_points()

    The second method is to use the hdobs class independently of the other recon classes:

    .. code-block:: python

        from tropycal.recon import hdobs

        #Retrieve all HDOBs for this storm, passing the Storm object as an argument
        hdobs_obj = hdobs(storm)

        #Retrieve the raw HDOBs data
        hdobs_obj.data

        #Use the plot_points() method of hdobs
        hdobs_obj.plot_points()
    """

    def __repr__(self):

        summary = ["<tropycal.recon.hdobs>"]

        # Find maximum wind and minimum pressure
        max_wspd = np.nanmax(self.data['wspd'])
        max_pkwnd = np.nanmax(self.data['pkwnd'])
        max_sfmr = np.nanmax(self.data['sfmr'])
        min_psfc = np.nanmin(self.data['p_sfc'])
        time_range = [pd.to_datetime(t) for t in (
            np.nanmin(self.data['time']), np.nanmax(self.data['time']))]

        # Add general summary
        emdash = '\u2014'
        summary_keys = {
            'Storm': f'{self.storm.name} {self.storm.year}',
            'Missions': len(set(self.data['mission'])),
            'Time range': f"{time_range[0]:%b-%d %H:%M} {emdash} {time_range[1]:%b-%d %H:%M}",
            'Max 30sec flight level wind': f"{max_wspd} knots",
            'Max 10sec flight level wind': f"{max_pkwnd} knots",
            'Max SFMR wind': f"{max_sfmr} knots",
            'Min surface pressure': f"{min_psfc} hPa",
            'Source': self.source
        }

        # Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)

    def __init__(self, storm, data=None, update=False):

        self.storm = storm
        self.data = None
        self.format = 1
        self.source = 'National Hurricane Center (NHC)'

        # Get URL based on storm year
        if storm.year >= 2012:
            self.format = 1
            if storm.basin == 'north_atlantic':
                archive_url = [
                    f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/AHONT1/']
            else:
                archive_url = [
                    f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/AHOPN1/']
        elif storm.year <= 2011 and storm.year >= 2008:
            self.format = 2
            if storm.basin == 'north_atlantic':
                archive_url = [f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/NOAA/URNT15/',
                               f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/USAF/URNT15/']
            else:
                archive_url = [f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/NOAA/URPN15/',
                               f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/USAF/URPN15/']
        elif storm.year == 2007:
            self.format = 3
            archive_url = [f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/NOAA/',
                           f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/USAF/']
        elif storm.year == 2006:
            self.format = 4
            archive_url = [
                f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/HDOB/']
        elif storm.year >= 2002:
            self.format = 5
            archive_url = [
                f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/{self.storm.name.upper()}/']
        elif storm.year <= 2001 and storm.year >= 1989:
            self.format = 6
            archive_url = [
                f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/{self.storm.name.lower()}/']
        else:
            raise RuntimeError("Recon data is not available prior to 1989.")

        if isinstance(data, str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        elif data is not None:
            self.data = data

        if data is None or update:
            try:
                start_time = max(self.data['time'])
            except:
                start_time = min(self.storm.dict['time']) - timedelta(hours=12)
            end_time = max(self.storm.dict['time']) + timedelta(hours=12)

            timestr = [f'{start_time:%Y%m%d}'] +\
                      [f'{t:%Y%m%d}' for t in self.storm.dict['time'] if t > start_time] +\
                      [f'{end_time:%Y%m%d}']

            # Retrieve list of files in URL(s) and filter by storm dates
            if self.format in [1, 3, 4]:
                page = requests.get(archive_url[0]).text
                content = page.split("\n")
                files = []
                for line in content:
                    if ".txt" in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]).split("."))
                del content
                files = sorted([i for i in files if i[1][:8]
                               in timestr], key=lambda x: x[1])
                linksub = [archive_url[0] + '.'.join(l) for l in files]
            elif self.format == 2:
                linksub = []
                for url in archive_url:
                    files = []
                    page = requests.get(url).text
                    content = page.split("\n")
                    for line in content:
                        if ".txt" in line:
                            files.append(
                                ((line.split('txt">')[1]).split("</a>")[0]).split("."))
                    del content
                    linksub += sorted([url + '.'.join(i)
                                      for i in files if i[1][:8] in timestr], key=lambda x: x[1])
                linksub = sorted(linksub)
            elif self.format == 5:
                page = requests.get(archive_url[0]).text
                content = page.split("\n")
                files = []
                for line in content:
                    if ".txt" in line and 'HDOBS' in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]))
                del content
                linksub = [archive_url[0] + l for l in files]
            elif self.format == 6:
                page = requests.get(archive_url[0]).text
                content = page.split("\n")
                files = []
                for line in content:
                    if ".txt" in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]))
                del content
                linksub = [archive_url[0] + l for l in files if l[0]
                           in ['H', 'h', 'M', 'm']]

            # Initiate urllib3
            urllib3.disable_warnings()
            http = urllib3.PoolManager()

            # Read through all files
            timer_start = dt.now()
            print(
                f'Searching through recon HDOB files between {timestr[0]} and {timestr[-1]} ...')
            filecount, unreadable = 0, 0
            found = False
            for link in linksub:

                # Read URL
                response = http.request('GET', link)
                content = response.data.decode('utf-8')

                # Find mission name line
                row = 3 if self.format <= 4 else 0
                while len(content.split("\n")[row]) < 3 or content.split("\n")[row][:3] in ["SXX", "URN", "URP", "YYX"]:
                    row += 1
                    if row >= 100:
                        break
                if row >= 100:
                    continue
                missionname = [i.split() for i in content.split('\n')][row][1]

                # Check for mission name to storm match by format
                if self.format != 6:
                    check = missionname[2:5] == self.storm.operational_id[2:4] + \
                        self.storm.operational_id[0]
                else:
                    check = True

                # Read HDOBs if this file matches the requested storm
                if check:
                    filecount += 1

                    try:

                        # Decode HDOBs by format
                        if self.format <= 3:
                            iter_hdob = decode_hdob(content, mission_row=row)
                        elif self.format == 4:
                            strdate = (link.split('.')[-2])[:8]
                            iter_hdob = decode_hdob_2006(
                                content, strdate, mission_row=row)
                        elif self.format == 6:
                            # Check for date
                            day = int(content.split("\n")[
                                      row - 1].split()[2][:2])
                            for iter_date in storm.dict['time']:
                                found_date = False
                                if iter_date.day == day:
                                    date = dt(iter_date.year,
                                              iter_date.month, iter_date.day)
                                    strdate = date.strftime('%Y%m%d')
                                    found_date = True
                                    break
                            if not found_date:
                                continue
                            iter_hdob = decode_hdob_2006(
                                content, strdate, mission_row=row)
                        elif self.format == 5:
                            # Split content by 10/20 minute blocks
                            strdate = (link.split('.')[-3]).split("_")[-1]
                            content_split = content.split("NNNN")
                            iter_hdob = None
                            for iter_content in content_split:
                                iter_split = iter_content.split("\n")
                                if len(iter_split) < 10:
                                    continue

                                # Search for starting line of data within sub-block
                                found = False
                                for line in iter_split:
                                    if missionname in line:
                                        found = True
                                temp_row = 0
                                while len(iter_split[temp_row]) < 3 or iter_split[temp_row][:3] in ["SXX", "URN", "URP"]:
                                    temp_row += 1
                                    if temp_row >= 100:
                                        break
                                if temp_row >= 100:
                                    break

                                # Parse data by format
                                if 'NOAA' in link:
                                    iter_hdob_loop = decode_hdob_2005_noaa(
                                        iter_content, strdate, temp_row)
                                else:
                                    iter_hdob_loop = decode_hdob_2006(
                                        iter_content, strdate, temp_row)

                                # Append HDOBs to full data
                                if iter_hdob is None:
                                    iter_hdob = copy.copy(iter_hdob_loop)
                                elif max(iter_hdob_loop['time']) > start_time:
                                    iter_hdob = pd.concat(
                                        [iter_hdob, iter_hdob_loop])
                                else:
                                    pass

                        # Append HDOBs to full data
                        if self.data is None:
                            self.data = copy.copy(iter_hdob)
                        elif max(iter_hdob['time']) > start_time:
                            self.data = pd.concat([self.data, iter_hdob])
                        else:
                            pass

                    except:
                        unreadable += 1

            print(f'--> Completed reading in recon HDOB files ({(dt.now()-timer_start).total_seconds():.1f} seconds)' +
                  f'\nRead {filecount} files' +
                  f'\nUnable to decode {unreadable} files')

        # This code will crash if no HDOBs are available
        try:

            # Sort data by time
            self.data.sort_values(['time'], inplace=True)

            # Recenter
            self._recenter()
            self.keys = list(self.data.keys())

        except:
            self.keys = []

    def update(self):
        r"""
        Update with the latest data for an ongoing storm.

        Notes
        -----
        This function has no return value, but simply updates the internal HDOB data with new observations since the object was created.
        """

        self = self.__init__(storm=self.storm, data=self.data, update=True)

    def _find_centers(self, data=None):

        if data is None:
            data = self.data
        data = data.sort_values(['mission', 'time'])

        def fill_nan(A):
            # Interpolate to fill nan values
            A = np.array(A)
            inds = np.arange(len(A))
            good = np.where(np.isfinite(A))
            good_grad = np.interp(inds, good[0], np.gradient(good[0]))
            if len(good[0]) >= 3:
                f = interp1d(inds[good], A[good],
                             bounds_error=False, kind='quadratic')
                B = np.where((np.isfinite(A)[good[0][0]:good[0][-1] + 1]) | (good_grad[good[0][0]:good[0][-1] + 1] > 3),
                             A[good[0][0]:good[0][-1] + 1],
                             f(inds[good[0][0]:good[0][-1] + 1]))
                return [np.nan] * good[0][0] + list(B) + [np.nan] * (inds[-1] - good[0][-1])
            else:
                return [np.nan] * len(A)

        missiondata = data.groupby('mission')
        dfs = []
        for group in missiondata:
            mdata = group[1]
            # Check that sfc pressure spread is big enough to identify real minima
            if np.nanpercentile(mdata['p_sfc'], 95) - np.nanpercentile(mdata['p_sfc'], 5) > 8:
                # Interp p_sfc across missing data
                p_sfc_interp = fill_nan(mdata['p_sfc'])
                # Interp wspd across missing data
                wspd_interp = fill_nan(mdata['wspd'])
                # Smooth p_sfc and wspd
                p_sfc_smooth = [
                    np.nan] * 1 + list(np.convolve(p_sfc_interp, [1 / 3] * 3, mode='valid')) + [np.nan] * 1
                wspd_smooth = [
                    np.nan] * 1 + list(np.convolve(wspd_interp, [1 / 3] * 3, mode='valid')) + [np.nan] * 1
                # Add wspd to p_sfc to encourage finding p mins with wspd mins
                # and prevent finding p mins in intense thunderstorms
                pw_test = np.array(p_sfc_smooth) + np.array(wspd_smooth) * .1
                # Find mins in 20-minute windows
                imin = np.nonzero(pw_test == minimum_filter(pw_test, 40))[0]
                # Only use mins if below 10th %ile of mission p_sfc data and when plane p is 550-950mb
                # and not in takeoff and landing time windows
                plane_p = fill_nan(mdata['plane_p'])
                imin = [i for i in imin if 800 < p_sfc_interp[i] < np.nanpercentile(mdata['p_sfc'], 10) and
                        550 < plane_p[i] < 950 and i > 60 and i < len(mdata) - 60]
            else:
                imin = []
            mdata['iscenter'] = np.array(
                [1 if i in imin else 0 for i in range(len(mdata))])
            dfs.append(mdata)

        data = pd.concat(dfs)
        numcenters = sum(data['iscenter'])
        print(f'Found {numcenters} center passes')
        return data

    def _recenter(self):
        data = copy.copy(self.data)
        # Interpolate center position to time of each ob
        interp_clon, interp_clat = self.storm.recon.get_track(data['time'])

        # Get x,y distance of each ob from coinciding interped center position
        data['xdist'] = [great_circle((interp_clat[i], interp_clon[i]),
                                      (interp_clat[i], data['lon'].values[i])).kilometers *
                         [1, -1][int(data['lon'].values[i] < interp_clon[i])] for i in range(len(data))]
        data['ydist'] = [great_circle((interp_clat[i], interp_clon[i]),
                                      (data['lat'].values[i], interp_clon[i])).kilometers *
                         [1, -1][int(data['lat'].values[i] < interp_clat[i])] for i in range(len(data))]
        data['distance'] = [(i**2 + j**2)**.5 for i,
                            j in zip(data['xdist'], data['ydist'])]

        imin = np.nonzero(data['distance'].values ==
                          minimum_filter(data['distance'].values, 40))[0]
        data['iscenter'] = np.array(
            [1 if i in imin and data['distance'].values[i] < 10 else 0 for i in range(len(data))])

        # print('Completed hdob center-relative coordinates')
        self.data = data

    def sel(self, mission=None, time=None, domain=None, plane_p=None, plane_z=None, p_sfc=None,
            temp=None, dwpt=None, wdir=None, wspd=None, pkwnd=None, sfmr=None, noflag=None,
            iscenter=None, distance=None):
        r"""
        Select a subset of HDOBs by any of its parameters and return a new hdobs object.

        Parameters
        ----------
        mission : str
            Mission name (number + storm id), e.g. mission 7 for AL05 is '0705L'
        time : list/tuple of datetimes
            list/tuple of start time and end time datetime objects.
            Default is None, which returns all points
        domain : dict
            dictionary with keys 'n', 's', 'e', 'w' corresponding to boundaries of domain
        plane_p : list/tuple of float/int
            list/tuple of plane_p bounds (min,max).
            None in either position of a tuple means it is boundless on that side. 
        plane_z : list/tuple of float/int
            list/tuple of plane_z bounds (min,max).
            None in either position of a tuple means it is boundless on that side.

        Returns
        -------
        hdobs object
            A new hdobs object that satisfies the intersection of all subsetting.
        """

        NEW_DATA = copy.copy(self.data)

        # Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission'] == mission]

        # Apply time filter
        if time is not None:
            bounds = get_bounds(NEW_DATA['time'], time)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['time'] > bounds[0]) & (
                NEW_DATA['time'] < bounds[1])]

        # Apply domain filter
        if domain is not None:
            tmp = {k[0].lower(): v for k, v in domain.items()}
            domain = {'n': 90, 's': -90, 'e': 359.99, 'w': 0}
            domain.update(tmp)
            bounds = get_bounds(
                NEW_DATA['lon'] % 360, (domain['w'] % 360, domain['e'] % 360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon'] % 360 >= bounds[0]) & (
                NEW_DATA['lon'] % 360 <= bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'], (domain['s'], domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat'] >= bounds[0]) & (
                NEW_DATA['lat'] <= bounds[1])]

        # Apply flight pressure filter
        if plane_p is not None:
            bounds = get_bounds(NEW_DATA['plane_p'], plane_p)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['plane_p'] > bounds[0]) & (
                NEW_DATA['plane_p'] < bounds[1])]

        # Apply flight height filter
        if plane_z is not None:
            bounds = get_bounds(NEW_DATA['plane_z'], plane_z)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['plane_z'] > bounds[0]) & (
                NEW_DATA['plane_z'] < bounds[1])]

        # Apply surface pressure filter
        if p_sfc is not None:
            bounds = get_bounds(NEW_DATA['p_sfc'], p_sfc)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['p_sfc'] > bounds[0]) & (
                NEW_DATA['p_sfc'] < bounds[1])]

        # Apply temperature filter
        if temp is not None:
            bounds = get_bounds(NEW_DATA['temp'], temp)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['temp'] > bounds[0]) & (
                NEW_DATA['temp'] < bounds[1])]

        # Apply dew point filter
        if dwpt is not None:
            bounds = get_bounds(NEW_DATA['dwpt'], dwpt)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['dwpt'] > bounds[0]) & (
                NEW_DATA['dwpt'] < bounds[1])]

        # Apply wind direction filter
        if wdir is not None:
            bounds = get_bounds(NEW_DATA['wdir'], wdir)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['wdir'] > bounds[0]) & (
                NEW_DATA['wdir'] < bounds[1])]

        # Apply wind speed filter
        if wspd is not None:
            bounds = get_bounds(NEW_DATA['wspd'], wspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['wspd'] > bounds[0]) & (
                NEW_DATA['wspd'] < bounds[1])]

        # Apply peak wind filter
        if pkwnd is not None:
            bounds = get_bounds(NEW_DATA['pkwnd'], pkwnd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['pkwnd'] > bounds[0]) & (
                NEW_DATA['pkwnd'] < bounds[1])]

        # Apply sfmr filter
        if sfmr is not None:
            bounds = get_bounds(NEW_DATA['sfmr'], sfmr)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['sfmr'] > bounds[0]) & (
                NEW_DATA['sfmr'] < bounds[1])]

        # Apply iscenter filter
        if iscenter is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['iscenter'] == iscenter]

        # Apply distance filter
        if distance is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['distance'] < distance]

        NEW_OBJ = hdobs(storm=self.storm, data=NEW_DATA)

        return NEW_OBJ

    def to_pickle(self, filename):
        r"""
        Save HDOB data (Pandas dataframe) to a pickle file.

        Parameters
        ----------
        filename : str
            name of file to save pickle file to.

        Notes
        -----
        This method saves the HDOBs data as a pickle within the current working directory, given a filename as an argument.

        For example, assume ``hdobs`` was retrieved from a Storm object (using the first method described in the ``hdobs`` class documentation). The HDOBs data would be saved to a pickle file as follows:

        >>> storm.recon.hdobs.to_pickle("mystorm_hdobs.pickle")

        Now the HDOBs data is saved locally, and next time recon data for this storm needs to be analyzed, this allows to bypass re-reading the HDOBs data from the NHC server by providing the pickle file as an argument:

        >>> storm.recon.get_hdobs("mystorm_hdobs.pickle")

        """

        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def plot_time_series(self, varname=('p_sfc', 'wspd'), mission=None, time=None, realtime=False, **kwargs):
        r"""
        Plots a time series of one or two variables on an axis.

        Parameters
        ----------
        varname : str or tuple
            If one variable to plot, varname is a string of the variable name. If two variables to plot, varname is a tuple of the left and right variable names, respectively. Available varnames are:

            * **p_sfc** - Mean Sea Level Pressure (hPa)
            * **temp** - Flight Level Temperature (C)
            * **dwpt** - Flight Level Dewpoint (C)
            * **wspd** - Flight Level Wind (kt)
            * **sfmr** - Surface Wind (kt)
            * **pkwnd** - Peak Wind Gust (kt)
            * **rain** - Rain Rate (mm/hr)
            * **plane_z** - Geopotential Height (m)
            * **plane_p** - Pressure (hPa)
        mission : int
            Mission number to plot. If None, all missions for this storm are plotted.
        time : tuple
            Tuple of start and end times (datetime.datetime) to plot. If None, all times available are plotted.
        realtime : bool
            If True, the most recent 2 hours of the mission will plot, overriding the time argument. Default is False.

        Other Parameters
        ----------------
        left_prop : dict
            Dictionary of properties for the left line. Scroll down for more information.
        right_prop : dict
            Dictionary of properties for the right line. Scroll down for more information.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        The following properties are available for customizing the plot, via ``left_prop`` and ``right_prop``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - ms
             - Marker size. If zero, none will be plotted. Default is zero.
           * - color
             - Color of lines (and markers if used). Default varies per varname.
           * - linewidth
             - Line width. Default is 1.0.
        """

        # Pop kwargs
        left_prop = kwargs.pop('left_prop', {})
        right_prop = kwargs.pop('right_prop', {})

        # Retrieve variables
        twin_ax = False
        if isinstance(varname, tuple):
            varname_right = varname[1]
            varname = varname[0]
            twin_ax = True
            varname_right_info = time_series_plot(varname_right)
        varname_info = time_series_plot(varname)

        # Filter by mission
        def str2(number):
            if number < 10:
                return f'0{number}'
            return str(number)
        if mission is not None:
            df = self.data.loc[self.data['mission'] == str2(mission)]
            if len(df) == 0:
                raise ValueError("Mission number provided is invalid.")
        else:
            df = self.data

        # Filter by time or realtime flag
        if realtime:
            end_time = pd.to_datetime(df['time'].values[-1])
            df = df.loc[(df['time'] >= end_time - timedelta(hours=2))
                        & (df['time'] <= end_time)]
        elif time is not None:
            df = df.loc[(df['time'] >= time[0]) & (df['time'] <= time[1])]
        if len(df) == 0:
            raise ValueError("Time range provided is invalid.")

        # Filter by default kwargs
        left_prop_default = {
            'ms': 0,
            'color': varname_info['color'],
            'linewidth': 1
        }
        for key in left_prop.keys():
            left_prop_default[key] = left_prop[key]
        left_prop = left_prop_default
        if twin_ax:
            right_prop_default = {
                'ms': 0,
                'color': varname_right_info['color'],
                'linewidth': 1
            }
            for key in right_prop.keys():
                right_prop_default[key] = right_prop[key]
            right_prop = right_prop_default

        # ----------------------------------------------------------------------------------

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
        if twin_ax:
            ax.grid(axis='x')
        else:
            ax.grid()

        # Plot line
        line1 = ax.plot(df['time'], df[varname], color=left_prop['color'],
                        linewidth=left_prop['linewidth'], label=varname_info['name'])
        ax.set_ylabel(varname_info['full_name'])

        # Plot dots
        if left_prop['ms'] >= 1:
            plot_times = df['time'].values
            plot_var = df[varname].values
            plot_times = [plot_times[i] for i in range(
                len(plot_times)) if varname not in df['flag'].values[i]]
            plot_var = [plot_var[i] for i in range(
                len(plot_var)) if varname not in df['flag'].values[i]]
            ax.plot(plot_times, plot_var, 'o',
                    color=left_prop['color'], ms=left_prop['ms'])

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%Mz\n%m/%d'))

        # Add twin axis
        if twin_ax:
            ax2 = ax.twinx()

            # Plot line
            line2 = ax2.plot(df['time'], df[varname_right], color=right_prop['color'],
                             linewidth=right_prop['linewidth'], label=varname_right_info['name'])
            ax2.set_ylabel(varname_right_info['full_name'])

            # Plot dots
            if right_prop['ms'] >= 1:
                plot_times = df['time'].values
                plot_var = df[varname_right].values
                plot_times = [plot_times[i] for i in range(
                    len(plot_times)) if varname_right not in df['flag'].values[i]]
                plot_var = [plot_var[i] for i in range(
                    len(plot_var)) if varname_right not in df['flag'].values[i]]
                ax2.plot(plot_times, plot_var, 'o',
                         color=right_prop['color'], ms=right_prop['ms'])

            # Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels)

            # Special handling if both are in units of Celsius
            same_unit = False
            if varname in ['temp', 'dwpt'] and varname_right in ['temp', 'dwpt']:
                same_unit = True
            if varname in ['sfmr', 'wspd', 'pkwnd'] and varname_right in ['sfmr', 'wspd', 'pkwnd']:
                same_unit = True
            if same_unit:
                min_val = np.nanmin(
                    [np.nanmin(df[varname]), np.nanmin(df[varname_right])])
                max_val = np.nanmax(
                    [np.nanmax(df[varname]), np.nanmax(df[varname_right])]) * 1.05
                min_val = min_val * 1.05 if min_val < 0 else min_val * 0.95
                if np.isnan(min_val):
                    min_val = 0
                if np.isnan(max_val):
                    max_val = 0
                if min_val == max_val:
                    min_val = 0
                    max_val = 10
                ax.set_ylim(min_val, max_val)
                ax2.set_ylim(min_val, max_val)

        # Add titles
        storm_data = self.storm.dict
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
        if ('invest' in storm_data.keys() and not storm_data['invest']) or len(idx[0]) > 0:
            tropical_vmax = np.array(storm_data['vmax'])[idx]

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

        # Plot title
        title_string = f"{storm_type} {storm_data['name']}"
        if mission is None:
            title_string += f"\nRecon Aircraft HDOBs | All Missions"
        else:
            title_string += f"\nRecon Aircraft HDOBs | Mission #{mission}"
        ax.set_title(title_string, loc='left', fontweight='bold')
        ax.set_title("Plot generated using Tropycal", fontsize=8, loc='right')

        # Return plot
        return ax

    def plot_points(self, varname='wspd', domain="dynamic", radlim=None, barbs=False, ax=None, cartopy_proj=None, **kwargs):
        r"""
        Creates a plot of recon data points.

        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the following keys in dataframe:

            * **"sfmr"** = SFMR surface wind
            * **"wspd"** = 30-second flight level wind (default)
            * **"pkwnd"** = 10-second flight level wind
            * **"p_sfc"** = extrapolated surface pressure
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        radlim : int
            Radius (in km) away from storm center to include points. If none (default), all points are plotted.
        barbs : bool
            If True, plots wind barbs. If False (default), plots dots.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-plot` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        1. Plotting wind barbs only works for wind related variables. ``barbs`` will be automatically set to False for non-wind variables.

        2. The special colormap **category_recon** can be used in the prop dict (``prop={'cmap':'category_recon'}``). This uses the standard SSHWS colormap, but with a new color for wind between 50 and 64 knots.
        """

        # Change barbs
        if varname == 'p_sfc':
            barbs = False

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Get plot data
        dfRecon = self.data

        # Create instance of plot object
        self.plot_obj = ReconPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj

        # Plot recon
        plot_ax = self.plot_obj.plot_points(
            self.storm, dfRecon, domain, varname=varname, radlim=radlim, barbs=barbs, ax=ax, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def plot_hovmoller(self, varname='wspd', radlim=None, window=6, align='center', ax=None, **kwargs):
        r"""
        Creates a hovmoller plot of azimuthally-averaged recon data.

        Parameters
        ----------
        varname : str
            Variable to average and plot. Available variable names are:

            * **"sfmr"** = SFMR surface wind
            * **"wspd"** = 30-second flight level wind (default)
            * **"pkwnd"** = 10-second flight level wind
            * **"p_sfc"** = extrapolated surface pressure
        radlim : int, optional
            Radius from storm center, in kilometers, to plot in hovmoller. Default is 200 km.
        window : int, optional
            Window of hours to interpolate between observations. Default is 6 hours.
        align : str, optional
            Alignment of window. Default is 'center'.
        ax : axes, optional
            Instance of axes to plot on. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict
            Customization properties for recon plot. Please refer to :ref:`options-prop-recon-hovmoller` for available options.
        track_dict : dict, optional
            Storm track dictionary. If None (default), internal storm center track is used.

        Returns
        -------
        ax
            Axes instance containing the plot.

        Notes
        -----
        The special colormap **category_recon** can be used in the prop dict (``prop={'cmap':'category_recon'}``). This uses the standard SSHWS colormap, but with a new color for wind between 50 and 64 knots.
        """

        # Pop kwargs
        track_dict = kwargs.pop('track_dict', None)
        prop = kwargs.pop('prop', {})
        default_prop = {
            'cmap': 'category',
            'levels': None,
            'smooth_contourf': False
        }
        for key in default_prop.keys():
            if key not in prop.keys():
                prop[key] = default_prop[key]

        # Get recon data
        dfRecon = self.data

        # Retrieve track dictionary if none is specified
        if track_dict is None:
            track_dict = self.storm.dict

        # Interpolate recon data to a hovmoller
        iRecon = interpRecon(dfRecon, varname, radlim,
                             window=window, align=align)
        Hov_dict = iRecon.interpHovmoller(track_dict)

        # title = get_recon_title(varname) #may not be necessary
        # If no contour levels specified, generate levels based on data min and max
        if prop['levels'] is None:
            prop['levels'] = (np.nanmin(Hov_dict['hovmoller']),
                              np.nanmax(Hov_dict['hovmoller']))

        # Retrieve updated contour levels and colormap based on input arguments and variable type
        cmap, clevs = get_cmap_levels(varname, prop['cmap'], prop['levels'])

        # Retrieve hovmoller times, radii and data
        time = Hov_dict['time']
        radius = Hov_dict['radius']
        vardata = Hov_dict['hovmoller']

        # Error check time
        time = [dt.strptime((i.strftime('%Y%m%d%H%M')), '%Y%m%d%H%M')
                for i in time]

        # ------------------------------------------------------------------------------

        # Create plot
        plt.figure(figsize=(9, 9), dpi=150)
        ax = plt.subplot()

        # Plot surface category colors individually, necessitating normalizing colormap
        if varname in ['vmax', 'sfmr', 'wspd', 'fl_to_sfc'] and prop['cmap'] in ['category', 'category_recon']:
            norm = mcolors.BoundaryNorm(clevs, cmap.N)
            cf = ax.contourf(radius, time, gfilt1d(vardata, sigma=3, axis=1),
                             levels=clevs, cmap=cmap, norm=norm)

        # Multiple clevels or without smooth contouring
        elif len(prop['levels']) > 2 or not prop['smooth_contourf']:
            cf = ax.contourf(radius, time, gfilt1d(vardata, sigma=3, axis=1),
                             levels=clevs, cmap=cmap)

        # Automatically generated levels with smooth contouring
        else:
            cf = ax.contourf(radius, time, gfilt1d(vardata, sigma=3, axis=1),
                             cmap=cmap, levels=np.linspace(min(prop['levels']), max(prop['levels']), 256))
        ax.axis([0, max(radius), min(time), max(time)])

        # Plot colorbar
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.1)

        # Format y-label ticks and labels as dates
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)

        # Set axes labels
        ax.set_ylabel('UTC Time (MM-DD HH)', fontsize=15)
        ax.set_xlabel('Radius (km)', fontsize=15)

        # --------------------------------------------------------------------------------------

        # Generate left and right title strings
        title_left, title_right = hovmoller_plot_title(
            self.storm, Hov_dict, varname)
        ax.set_title(title_left, loc='left', fontsize=16, fontweight='bold')
        ax.set_title(title_right, loc='right', fontsize=12)
        
        # Add plot credit
        ax.text(0.02, 0.02, 'Plot generated by Tropycal',
                ha='left', va='bottom', transform=ax.transAxes)

        # Return axis
        return ax

    def plot_maps(self, time=None, varname='wspd', recon_stats=None, domain="dynamic",
                  window=6, align='center', radlim=None, ax=None, cartopy_proj=None, save_dir=None, **kwargs):
        r"""
        Creates maps of interpolated recon data. 

        Parameters
        ----------
        time : datetime.datetime or list
            Single datetime object, or list/tuple of datetime objects containing the start and end times to plot between. If None (default), all times will be plotted.
        varname : str or tuple
            Variable to plot. Can be one of the following keys in dataframe:

            * **"sfmr"** = SFMR surface wind
            * **"wspd"** = 30-second flight level wind (default)
            * **"pkwnd"** = 10-second flight level wind
            * **"p_sfc"** = extrapolated surface pressure
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        radlim : int, optional
            Radius from storm center, in kilometers, to plot. Default is 200 km.
        window : int, optional
            Window of hours to interpolate between observations. Default is 6 hours.
        align : str, optional
            Alignment of window. Default is 'center'.
        ax : axes, optional
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs, optional
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_dir : str, optional
            Directory to save output images in. If None, images will not be saved. Default is None.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-swath` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        track_dict : dict, optional
            Storm track dictionary. If None (default), internal storm center track is used.
        """

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})
        track_dict = kwargs.pop('track_dict', None)

        # Get plot data
        ONE_MAP = False
        if time is None:
            dfRecon = self.data
        elif isinstance(time, (tuple, list)):
            dfRecon = self.sel(time=time).data
        elif isinstance(time, dt):
            dfRecon = self.sel(
                time=(time - timedelta(hours=6), time + timedelta(hours=6))).data
            ONE_MAP = True

        MULTIVAR = False
        if isinstance(varname, (tuple, list)):
            MULTIVAR = True

        if track_dict is None:
            track_dict = self.storm.dict

            # Error check for time dimension name
            if 'time' not in track_dict.keys():
                track_dict['time'] = track_dict['time']

        if ONE_MAP:
            f = interp1d(mdates.date2num(
                track_dict['time']), track_dict['lon'], fill_value='extrapolate')
            clon = f(mdates.date2num(time))
            f = interp1d(mdates.date2num(
                track_dict['time']), track_dict['lat'], fill_value='extrapolate')
            clat = f(mdates.date2num(time))

            # clon = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lon'])
            # clat = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lat'])
            track_dict = {
                'time': time,
                'lon': clon,
                'lat': clat
            }

        if MULTIVAR:
            Maps = []
            for v in varname:
                iRecon = interpRecon(dfRecon, v, radlim,
                                     window=window, align=align)
                tmpMaps = iRecon.interpMaps(track_dict)
                Maps.append(tmpMaps)
        else:
            iRecon = interpRecon(dfRecon, varname, radlim,
                                 window=window, align=align)
            Maps = iRecon.interpMaps(track_dict)

        # titlename,units = get_recon_title(varname)

        if 'levels' not in prop.keys() or 'levels' in prop.keys() and prop['levels'] is None:
            prop['levels'] = np.arange(np.floor(np.nanmin(Maps['maps']) / 10) * 10,
                                       np.ceil(np.nanmax(Maps['maps']) / 10) * 10 + 1, 10)

        if not ONE_MAP:

            if save_dir is True:
                save_dir = f'{self.storm}{self.year}_maps'
            try:
                os.system(f'mkdir {save_dir}')
            except:
                pass

            if MULTIVAR:
                Maps2 = Maps[1]
                Maps = Maps[0]

                print(np.nanmax(Maps['maps']), np.nanmin(Maps2['maps']))

            figs = []
            for i, t in enumerate(Maps['time']):
                Maps_sub = {
                    'time': t,
                    'grid_x': Maps['grid_x'],
                    'grid_y': Maps['grid_y'],
                    'maps': Maps['maps'][i],
                    'center_lon': Maps['center_lon'][i],
                    'center_lat': Maps['center_lat'][i],
                    'stats': Maps['stats']
                }

                # Create instance of plot object
                self.plot_obj = ReconPlot()

                # Create cartopy projection
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)
                cartopy_proj = self.plot_obj.proj

                # Maintain the same lat / lon dimensions for all dynamic maps
                # Determined by the dynamic domain from the first map
                if i > 0 and domain == 'dynamic':
                    d1 = {
                        'n': Maps_sub['center_lat'] + dlat,
                        's': Maps_sub['center_lat'] - dlat,
                        'e': Maps_sub['center_lon'] + dlon,
                        'w': Maps_sub['center_lon'] - dlon
                    }
                else:
                    d1 = domain

                # Plot recon

                if MULTIVAR:
                    Maps_sub1 = dict(Maps_sub)
                    Maps_sub2 = dict(Maps_sub)
                    Maps_sub = [Maps_sub1, Maps_sub2]
                    Maps_sub[1]['maps'] = Maps2['maps'][i]

                    print(np.nanmax(Maps_sub[0]['maps']),
                          np.nanmin(Maps_sub[1]['maps']))

                plot_ax, d0 = self.plot_obj.plot_maps(self.storm, Maps_sub, varname, recon_stats,
                                                      domain=d1, ax=ax, return_domain=True, prop=prop, map_prop=map_prop)

                # Get domain dimensions from the first map
                if i == 0:
                    dlat = .5 * (d0['n'] - d0['s'])
                    dlon = .5 * (d0['e'] - d0['w'])

                figs.append(plot_ax)

                if save_dir is not None:
                    plt.savefig(
                        f'{save_dir}/{t.strftime("%Y%m%d%H%M")}.png', bbox_inches='tight')
                plt.close()

            if save_dir is None:
                return figs

        else:
            # Create instance of plot object
            self.plot_obj = ReconPlot()

            # Create cartopy projection
            if cartopy_proj is None:
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)
                cartopy_proj = self.plot_obj.proj

            # Plot recon
            plot_ax = self.plot_obj.plot_maps(
                self.storm, Maps, varname, recon_stats, domain, ax, prop=prop, map_prop=map_prop)

            # Return axis
            return plot_ax

    def plot_swath(self, varname='wspd', domain="dynamic", ax=None, cartopy_proj=None, **kwargs):
        r"""
        Creates a map plot of a swath of interpolated recon data.

        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the following keys in dataframe:

            * **"sfmr"** = SFMR surface wind
            * **"wspd"** = 30-second flight level wind (default)
            * **"pkwnd"** = 10-second flight level wind
            * **"p_sfc"** = extrapolated surface pressure

        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-swath` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        track_dict : dict, optional
            Storm track dictionary. If None (default), internal storm center track is used.
        swathfunc : function
            Function to operate on interpolated recon data (e.g., np.max, np.min, or percentile function). Default is np.min for pressure, otherwise np.max.
        """

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})
        track_dict = kwargs.pop('track_dict', None)
        swathfunc = kwargs.pop('swathfunc', None)

        # Get plot data
        dfRecon = self.data

        if track_dict is None:
            track_dict = self.storm.dict

        if swathfunc is None:
            if varname == 'p_sfc':
                swathfunc = np.min
            else:
                swathfunc = np.max

        iRecon = interpRecon(dfRecon, varname)
        Maps = iRecon.interpMaps(track_dict, interval=.2)

        # Create instance of plot object
        self.plot_obj = ReconPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj

        # Plot recon
        plot_ax = self.plot_obj.plot_swath(
            self.storm, Maps, varname, swathfunc, track_dict, domain, ax, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def gridded_stats(self, request, thresh={}, binsize=1, domain="dynamic", ax=None,
                      return_array=False, cartopy_proj=None, prop={}, map_prop={}):
        r"""
        Creates a plot of gridded statistics.

        Parameters
        ----------
        request : str
            This string is a descriptor for what you want to plot.
            It will be used to define the variable (e.g. 'wind' --> 'vmax') and the function (e.g. 'maximum' --> np.max()).
            This string is also used as the plot title.

            Variable words to use in request:

            * **wind** - (kt). Sustained wind.
            * **pressure** - (hPa). Minimum pressure.
            * **wind change** - (kt/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "wind change in 24 hours").
            * **pressure change** - (hPa/time). Must be followed by an integer value denoting the length of the time window '__ hours' (e.g., "pressure change in 24 hours").
            * **storm motion** - (km/hour). Can be followed a length of time window. Otherwise defaults to 24 hours.

            Units of all wind variables are knots and pressure variables are hPa. These are added into the title.

            Function words to use in request:

            * **maximum**
            * **minimum**
            * **average** 
            * **percentile** - Percentile must be preceded by an integer [0,100].
            * **number** - Number of storms in grid box satisfying filter thresholds.

            Example usage: "maximum wind change in 24 hours", "50th percentile wind", "number of storms"

        thresh : dict, optional
            Keywords in self.keys

            Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.

        binsize : float, optional
            Grid resolution in degrees. Default is 1 degree.
        domain : str, optional
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes, optional
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_array : bool, optional
            If True, returns the gridded 2D array used to generate the plot. Default is False.
        cartopy_proj : ccrs, optional
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict, optional
            Customization properties of plot. Please refer to :ref:`options-prop-gridded` for available options.
        map_prop : dict, optional
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        By default, the plot axes is returned. If "return_array" are set to True, a dictionary is returned containing both the axes and data array.
        """

        default_prop = {
            'smooth': None
        }
        for key in prop.keys():
            default_prop[key] = prop[key]
        prop = default_prop

        # Update thresh based on input
        default_thresh = {
            'sample_min': 1,
            'p_max': np.nan,
            'v_min': np.nan,
            'dv_min': np.nan,
            'dp_max': np.nan,
            'dv_max': np.nan,
            'dp_min': np.nan,
            'dt_window': 24,
            'dt_align': 'middle'
        }
        for key in thresh:
            default_thresh[key] = thresh[key]
        thresh = default_thresh

        # Retrieve the requested function, variable for computing stats, and plot title. These modify thresh if necessary.
        thresh, func = find_func(request, thresh)
        thresh, varname = find_var(request, thresh)

        # ---------------------------------------------------------------------------------------------------

        points = self.data
        # Round lat/lon points down to nearest bin
        def to_bin(x): return np.floor(x / binsize) * binsize
        points["latbin"] = points.lat.map(to_bin)
        points["lonbin"] = points.lon.map(to_bin)

        # ---------------------------------------------------------------------------------------------------

        # Group by latbin,lonbin,stormid
        print("--> Grouping by lat/lon")
        groups = points.groupby(["latbin", "lonbin"])

        # Loops through groups, and apply stat func to obs
        # Constructs a new dataframe containing the lat/lon bins and plotting variable
        new_df = {
            'latbin': [],
            'lonbin': [],
            'varname': []
        }
        for g in groups:
            new_df[varname].append(func(g[1][varname].values))
            new_df['latbin'].append(g[0][0])
            new_df['lonbin'].append(g[0][1])
        new_df = pd.DataFrame.from_dict(new_df)

        # ---------------------------------------------------------------------------------------------------

        # Group again by latbin,lonbin
        # Construct two 1D lists: zi (grid values) and coords, that correspond to the 2D grid
        groups = new_df.groupby(["latbin", "lonbin"])

        zi = [func(g[1][varname]) if len(g[1]) >=
              thresh['sample_min'] else np.nan for g in groups]

        # Construct a 1D array of coordinates
        coords = [g[0] for g in groups]

        # Construct a 2D longitude and latitude grid, using the specified binsize resolution
        if prop['smooth'] is not None:
            all_lats = [(round(l / binsize) * binsize)
                        for l in self.data['lat']]
            all_lons = [(round(l / binsize) * binsize) %
                        360 for l in self.data['lon']]
            xi = np.arange(min(all_lons) - binsize,
                           max(all_lons) + 2 * binsize, binsize)
            yi = np.arange(min(all_lats) - binsize,
                           max(all_lats) + 2 * binsize, binsize)
        else:
            xi = np.arange(np.nanmin(
                points["lonbin"]) - binsize, np.nanmax(points["lonbin"]) + 2 * binsize, binsize)
            yi = np.arange(np.nanmin(
                points["latbin"]) - binsize, np.nanmax(points["latbin"]) + 2 * binsize, binsize)
        grid_x, grid_y = np.meshgrid(xi, yi)

        # Construct a 2D grid for the z value, depending on whether vector or scalar quantity
        grid_z = np.ones(grid_x.shape) * np.nan
        for c, z in zip(coords, zi):
            grid_z[np.where((grid_y == c[0]) & (grid_x == c[1]))] = z

        # ---------------------------------------------------------------------------------------------------

        # Create instance of plot object
        plot_obj = TrackPlot()

        # Create cartopy projection using basin
        if cartopy_proj is None:
            if max(points['lon']) > 150 or min(points['lon']) < -150:
                plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=180.0)
            else:
                plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)

        prop['title_L'], prop['title_R'] = self.storm.name, 'things'

        if domain == "dynamic":
            domain = {
                'W': min(self.data['lon']),
                'E': max(self.data['lon']),
                'S': min(self.data['lat']),
                'N': max(self.data['lat'])
            }

        # Plot gridded field
        plot_ax = plot_obj.plot_gridded(
            grid_x, grid_y, grid_z, varname, domain=domain, ax=ax, prop=prop, map_prop=map_prop)

        # Format grid into xarray if specified
        if return_array:
            try:
                # Import xarray and construct DataArray, replacing NaNs with zeros
                import xarray as xr
                arr = xr.DataArray(np.nan_to_num(grid_z), coords=[
                                   grid_y.T[0], grid_x[0]], dims=['lat', 'lon'])
                return arr
            except ImportError as e:
                raise RuntimeError(
                    "Error: xarray is not available. Install xarray in order to use the 'return_array' flag.") from e

        # Return axis
        if return_array:
            return {'ax': plot_ax, 'array': arr}
        else:
            return plot_ax


class dropsondes:

    r"""
    Creates an instance of a Dropsondes object containing all dropsonde data for a single storm.

    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Requested storm.
    data : str, optional
        Filepath of pickle file containing dropsondes data retrieved from ``dropsondes.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.
    update : bool
        True = search for new data, following existing data in the dropsonde object, and concatenate.

    Returns
    -------
    Dataset
        An instance of dropsondes.

    Notes
    -----
    .. warning::

        Recon data is currently only available from 2006 onwards.

    There are two recommended ways of retrieving a dropsondes object. Since the ``ReconDataset``, ``hdobs``, ``dropsondes`` and ``vdms`` classes are **storm-centric**, a Storm object is required for both methods.

    .. code-block:: python

        #Retrieve Hurricane Michael (2018) from TrackDataset
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))

    The first method is to use the empty instance of ReconDataset already initialized in the Storm object, which has a ``get_dropsondes()`` method thus allowing all of the dropsondes attributes and methods to be accessed from the Storm object. As a result, a Storm object does not need to be provided as an argument.

    .. code-block:: python

        #Retrieve all dropsondes for this storm
        storm.recon.get_dropsondes()

        #Retrieve the raw dropsondes data
        storm.recon.dropsondes.data

        #Use the plot_points() method of dropsondes
        storm.recon.dropsondes.plot_points()

    The second method is to use the dropsondes class independently of the other recon classes:

    .. code-block:: python

        from tropycal.recon import dropsondes

        #Retrieve all dropsondes for this storm, passing the Storm object as an argument
        dropsondes_obj = dropsondes(storm)

        #Retrieve the raw dropsondes data
        dropsondes_obj.data

        #Use the plot_points() method of dropsondes
        dropsondes_obj.plot_points()
    """

    def __repr__(self):

        summary = ["<tropycal.recon.dropsondes>"]

        def isNA(x, units):
            if np.isnan(x):
                return 'N/A'
            else:
                return f'{x} {units}'
        # Find maximum wind and minimum pressure
        max_MBLspd = isNA(np.nanmax([i['MBLspd'] for i in self.data]), 'knots')
        max_DLMspd = isNA(np.nanmax([i['DLMspd'] for i in self.data]), 'knots')
        max_WL150spd = isNA(np.nanmax([i['WL150spd']
                            for i in self.data]), 'knots')
        min_slp = isNA(np.nanmin([i['slp'] for i in self.data]), 'hPa')
        missions = set([i['mission'] for i in self.data])

        # Add general summary
        emdash = '\u2014'
        summary_keys = {
            'Storm': f'{self.storm.name} {self.storm.year}',
            'Missions': len(missions),
            'Dropsondes': len(self.data),
            'Max 500m-avg wind': max_MBLspd,
            'Max 150m-avg wind': max_WL150spd,
            'Min sea level pressure': min_slp,
            'Source': self.source
        }

        # Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)

    def __init__(self, storm, data=None, update=False):

        self.storm = storm
        self.source = 'National Hurricane Center (NHC)'

        if storm.year >= 2006:
            self.format = 1
            if storm.basin == 'north_atlantic':
                archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPNT3/'
            else:
                archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPPN3/'
        elif storm.year >= 2002 and storm.year <= 2005:
            self.format = 2
            archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/{self.storm.name.upper()}/'
        elif storm.year >= 1989 and storm.year <= 2001:
            self.format = 3
            archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/{self.storm.name.lower()}/'
        else:
            raise RuntimeError("Recon data is not available prior to 1989.")
        self.data = None

        if isinstance(data, str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        elif data is not None:
            self.data = data

        if data is None or update:
            try:
                start_time = max(self.data['time'])
            except:
                start_time = min(self.storm.dict['time']) - timedelta(days=1)
            end_time = max(self.storm.dict['time']) + timedelta(days=1)

            timeboundstrs = [f'{t:%Y%m%d%H%M}' for t in (start_time, end_time)]

            # Retrieve list of files in URL and filter by storm dates
            page = requests.get(archive_url).text
            content = page.split("\n")
            files = []
            if self.format == 1:
                for line in content:
                    if ".txt" in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]).split("."))
                del content
                files = sorted([i for i in files if i[1] >= min(
                    timeboundstrs) and i[1] <= max(timeboundstrs)], key=lambda x: x[1])
                linksub = [archive_url + '.'.join(l) for l in files]
            elif self.format == 2:
                for line in content:
                    if ".txt" in line and 'DROPS' in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]))
                del content
                linksub = [archive_url + l for l in files]
            elif self.format == 3:
                for line in content:
                    if ".txt" in line:
                        files.append(
                            ((line.split('txt">')[1]).split("</a>")[0]))
                del content
                linksub = [archive_url +
                           l for l in files if l[0] in ['D', 'd']]

            urllib3.disable_warnings()
            http = urllib3.PoolManager()

            timer_start = dt.now()
            print(
                f'Searching through recon dropsonde files between {timeboundstrs[0]} and {timeboundstrs[-1]} ...')
            filecount = 0
            for link in linksub:
                response = http.request('GET', link)
                content = response.data.decode('utf-8')

                # Post-2006 format
                if self.format == 1:
                    datestamp = dt.strptime(link.split('.')[-2], '%Y%m%d%H%M')
                    try:
                        missionname, tmp = decode_dropsonde(
                            content, date=datestamp)
                    except:
                        continue

                    testkeys = ('TOPtime', 'lat', 'lon')
                    if missionname[2:5] == self.storm.operational_id[2:4] + self.storm.operational_id[0]:
                        filecount += 1
                        if self.data is None:
                            self.data = [copy.copy(tmp)]
                        elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                            self.data.append(tmp)
                        else:
                            pass

                # Pre-2002 format
                elif self.format == 3:

                    # Check for date
                    try:
                        day = int(content.split("\n")[0].split()[2][:2])
                        for iter_date in storm.dict['time']:
                            found_date = False
                            if iter_date.day == day:
                                date = dt(iter_date.year,
                                          iter_date.month, iter_date.day)
                                found_date = True
                                break
                        if not found_date:
                            continue
                        missionname, tmp = decode_dropsonde(
                            content.replace(";", ""), date=date)

                        # Add date to mission
                        hh = int(content.split("\n")[0].split()[2][2:4])
                        mm = int(content.split("\n")[0].split()[2][4:6])
                        tmp['TOPtime'] = dt(
                            iter_date.year, iter_date.month, iter_date.day, hh, mm)
                        if np.isnan(tmp['TOPlat']):
                            tmp['TOPlat'] = tmp['lat']
                        if np.isnan(tmp['TOPlon']):
                            tmp['TOPlon'] = tmp['lon']

                        testkeys = ('TOPtime', 'lat', 'lon')
                        filecount += 1
                        if self.data is None:
                            self.data = [copy.copy(tmp)]
                        elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                            self.data.append(tmp)
                        else:
                            pass
                    except:
                        pass

                # Pre-2006 format
                elif self.format == 2:
                    strdate = (link.split('.')[-3]).split("_")[-1]
                    content_split = content.split("NNNN")

                    for iter_content in content_split:

                        iter_split = iter_content.split("\n")
                        if len(iter_split) < 6:
                            continue

                        # Format date
                        found_date = False
                        for line in iter_split:
                            if 'UZNT13' in line:
                                date_string = line.split()[2]
                                found_date = True
                                datestamp = dt.strptime(strdate, '%Y%m%d')
                                datestamp = datestamp.replace(day=int(date_string[:2]), hour=int(
                                    date_string[2:4]), minute=int(date_string[4:6]))

                        # Decode dropsondes
                        if not found_date:
                            continue
                        try:
                            missionname, tmp = decode_dropsonde(
                                iter_content, date=datestamp)
                        except:
                            continue

                        testkeys = ('lat', 'lon')
                        filecount += 1
                        if self.data is None:
                            self.data = [copy.copy(tmp)]
                        elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                            self.data.append(tmp)
                        else:
                            pass

            print(f'--> Completed reading in recon dropsonde files ({(dt.now()-timer_start).total_seconds():.1f} seconds)' +
                  f'\nRead {filecount} files')

        try:
            self._recenter()
            self.keys = sorted(
                list(set([k for d in self.data for k in d.keys()])))
        except:
            self.keys = []

    def update(self):
        r"""
        Update with the latest data for an ongoing storm.

        Notes
        -----
        This function has no return value, but simply updates the internal dropsonde data with new observations since the object was created.
        """

        newobj = dropsondes(storm=self.storm, data=self.data, update=True)
        return newobj

    def _recenter(self):
        data = copy.copy(self.data)

        # Get x,y distance of each ob from coinciding interped center position
        for stage in ('TOP', 'BOTTOM'):
            # Interpolate center position to time of each ob
            interp_clon, interp_clat = self.storm.recon.get_track(
                [d[f'{stage}time'] for d in data])

            # Get x,y distance of each ob from coinciding interped center position
            for i, d in enumerate(data):
                d.update({f'{stage}xdist': great_circle((interp_clat[i], interp_clon[i]),
                                                        (interp_clat[i], d[f'{stage}lon'])).kilometers *
                          [1, -1][int(d[f'{stage}lon'] < interp_clon[i])]})
                d.update({f'{stage}ydist': great_circle((interp_clat[i], interp_clon[i]),
                                                        (d[f'{stage}lat'], interp_clon[i])).kilometers *
                          [1, -1][int(d[f'{stage}lat'] < interp_clat[i])]})
                d.update({f'{stage}distance': (
                    d[f'{stage}xdist']**2 + d[f'{stage}ydist']**2)**.5})

        # print('Completed dropsonde center-relative coordinates')
        self.data = data

    def isel(self, index):
        r"""
        Select a single dropsonde by index of the list.

        Parameters
        ----------
        index : int
            Integer containing the index of the dropsonde.

        Returns
        -------
        dropsondes
            Instance of Dropsondes for the single requested dropsonde.
        """

        NEW_DATA = copy.copy(self.data)
        NEW_DATA = [NEW_DATA[index]]
        NEW_OBJ = dropsondes(storm=self.storm, data=NEW_DATA)

        return NEW_OBJ

    def sel(self, mission=None, time=None, domain=None, location=None, top=None,
            slp=None, MBLspd=None, WL150spd=None, DLMspd=None):
        r"""
        Select a subset of dropsondes by any of its parameters and return a new dropsondes object.

        Parameters
        ----------
        mission : str
            Mission name (number + storm id), e.g. mission 7 for AL05 is '0705L'
        time : list/tuple of datetimes
            list/tuple of start time and end time datetime objects.
            Default is None, which returns all points
        domain : dict
            dictionary with keys 'n', 's', 'e', 'w' corresponding to boundaries of domain.
        location : str
            Location of dropsonde. Can be "eyewall" or "center".
        top : tuple
            Tuple containing range of pressures (in hPa) of the top of the dropsonde level.
        slp : tuple
            Tuple containing range of pressures (in hPa) of the bottom of the dropsonde near surface.

        Returns
        -------
        dropsondes
            A new dropsondes object that satisfies the intersection of all subsetting.
        """

        NEW_DATA = copy.copy(pd.DataFrame(self.data))

        # Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission'] == mission]

        # Apply time filter
        if time is not None:
            try:
                if isinstance(time, (tuple, list)):
                    bounds = get_bounds(NEW_DATA['TOPtime'], time)
                    NEW_DATA = NEW_DATA.loc[(NEW_DATA['TOPtime'] >= bounds[0]) & (
                        NEW_DATA['TOPtime'] <= bounds[1])]
                else:
                    i = np.argmin(abs(time - NEW_DATA['TOPtime']))
                    return self.isel(i)
            except:
                if isinstance(time, (tuple, list)):
                    bounds = get_bounds(NEW_DATA['BOTTOMtime'], time)
                    NEW_DATA = NEW_DATA.loc[(NEW_DATA['BOTTOMtime'] >= bounds[0]) & (
                        NEW_DATA['BOTTOMtime'] <= bounds[1])]
                else:
                    i = np.argmin(abs(time - NEW_DATA['BOTTOMtime']))
                    return self.isel(i)

        # Apply domain filter
        if domain is not None:
            tmp = {k[0].lower(): v for k, v in domain.items()}
            domain = {'n': 90, 's': -90, 'e': 359.99, 'w': 0}
            domain.update(tmp)
            bounds = get_bounds(
                NEW_DATA['lon'] % 360, (domain['w'] % 360, domain['e'] % 360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon'] % 360 >= bounds[0]) & (
                NEW_DATA['lon'] % 360 <= bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'], (domain['s'], domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat'] >= bounds[0]) & (
                NEW_DATA['lat'] <= bounds[1])]

        # Apply location filter
        if location is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['location'] == location.upper()]

        # Apply top standard level filter
        if top is not None:
            bounds = get_bounds(NEW_DATA['top'], top)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['top'] >= bounds[0]) & (
                NEW_DATA['top'] <= bounds[1])]

        # Apply surface pressure filter
        if slp is not None:
            bounds = get_bounds(NEW_DATA['slp'], slp)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['slp'] >= bounds[0]) & (
                NEW_DATA['slp'] <= bounds[1])]

        # Apply MBL wind speed filter
        if MBLspd is not None:
            bounds = get_bounds(NEW_DATA['MBLspd'], MBLspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['MBLspd'] >= bounds[0]) & (
                NEW_DATA['MBLspd'] <= bounds[1])]

        # Apply DLM wind speed filter
        if DLMspd is not None:
            bounds = get_bounds(NEW_DATA['DLMspd'], DLMspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['DLMspd'] >= bounds[0]) & (
                NEW_DATA['DLMspd'] <= bounds[1])]

        # Apply WL150 wind speed filter
        if WL150spd is not None:
            bounds = get_bounds(NEW_DATA['WL150spd'], WL150spd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['WL150spd'] >= bounds[0]) & (
                NEW_DATA['WL150spd'] <= bounds[1])]

        NEW_OBJ = dropsondes(storm=self.storm, data=list(
            NEW_DATA.T.to_dict().values()))

        return NEW_OBJ

    def to_pickle(self, filename):
        r"""
        Save dropsonde data (list of dictionaries) to a pickle file

        Parameters
        ----------
        filename : str
            name of file to save pickle file to.

        Notes
        -----
        This method saves the dropsondes data as a pickle within the current working directory, given a filename as an argument.

        For example, assume ``dropsondes`` was retrieved from a Storm object (using the first method described in the ``dropsondes`` class documentation). The dropsondes data would be saved to a pickle file as follows:

        >>> storm.recon.dropsondes.to_pickle(data="mystorm_dropsondes.pickle")

        Now the dropsondes data is saved locally, and next time recon data for this storm needs to be analyzed, this allows to bypass re-reading the dropsondes data from the NHC server by providing the pickle file as an argument:

        >>> storm.recon.get_dropsondes(data="mystorm_dropsondes.pickle")

        """

        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def plot_points(self, varname='slp', level=None, domain="dynamic", ax=None, cartopy_proj=None, **kwargs):
        r"""
        Creates a plot of dropsonde data points.

        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the keys in the dropsonde dictionary (retrieved using ``recon.dropsondes.data``).
        level : int, optional
            Pressure level (in hPa) to plot varname for. Only valid if varname is in "pres", "hgt", "temp", "dwpt", "wdir", "wspd".
        domain : str/dict
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-plot` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.

        Notes
        -----
        To retrieve all possible varnames, check the ``data`` attribute of this Dropsonde object. For example, if ReconDataset was retrieved through a Storm object as in the example below, the possible varnames would be retrieved as follows:

        .. code-block:: python

            from tropycal import tracks

            #Get dataset object
            basin = tracks.TrackDataset()

            #Get storm object
            storm = basin.get_storm(('michael',2018))

            #Get dropsondes for this storm
            storm.recon.get_dropsondes()

            #Retrieve list of all possible varnames
            print(storm.recon.dropsondes.data)
        """

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Get plot data
        if level is not None:
            plotdata = [m['levels'].loc[m['levels']['pres'] == level][varname].to_numpy()[0]
                        if 'levels' in m.keys() and level in m['levels']['pres'].to_numpy() else np.nan
                        for m in self.data]
        else:
            plotdata = [m[varname] if varname in m.keys()
                        else np.nan for m in self.data]

        # Make sure data doesn't have NaNs
        check_data = [m['BOTTOMlat']
                      for m in self.data if not np.isnan(m['BOTTOMlat'])]
        if len(check_data) == 0:
            dfRecon = pd.DataFrame.from_dict({'time': [m['TOPtime'] for m in self.data],
                                              'lat': [m['TOPlat'] for m in self.data],
                                              'lon': [m['TOPlon'] for m in self.data],
                                              varname: plotdata})
        else:
            dfRecon = pd.DataFrame.from_dict({'time': [m['BOTTOMtime'] for m in self.data],
                                              'lat': [m['BOTTOMlat'] for m in self.data],
                                              'lon': [m['BOTTOMlon'] for m in self.data],
                                              varname: plotdata})

        # Create instance of plot object
        self.plot_obj = ReconPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj

        # Plot recon
        plot_ax = self.plot_obj.plot_points(self.storm, dfRecon, domain, varname=(
            varname, level), ax=ax, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def plot_skewt(self, time=None):
        r"""
        Plot a Skew-T chart for selected dropsondes.

        Parameters
        ----------
        time : datetime.datetime, optional
            Time closest to requested Skew-T. If none, all dropsondes will plot.

        Returns
        -------
        list
            Returns a list of figures, or a single figure for a single plot.
        """
        storm_data = self.storm.dict

        if time is None:
            dict_list = self.data
        else:
            dict_list = self.sel(time=time).data

        # Format storm name
        storm_data = self.storm.dict
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
        if ('invest' in storm_data.keys() and not storm_data['invest']) or len(idx[0]) > 0:
            tropical_vmax = np.array(storm_data['vmax'])[idx]

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
        title_string = f'{storm_type} {storm_data["name"]}\nDropsonde DDD, Mission MMM'

        # Plot Skew-T
        return plot_skewt(dict_list, title_string)


class vdms:

    r"""
    Creates an instance of a VDMs object containing all Vortex Data Message (VDM) data for a single storm.

    Parameters
    ----------
    storm : tropycal.tracks.Storm
        Requested storm.
    data : str, optional
        Filepath of pickle file containing VDM data retrieved from ``vdms.to_pickle()``. If provided, data will be retrieved from the local pickle file instead of the NHC server.
    update : bool
        True = search for new data, following existing data in the dropsonde object, and concatenate.

    Returns
    -------
    Dataset
        An instance of VDMs.

    Notes
    -----
    .. warning::

        Recon data is currently only available from 1989 onwards.

    VDM data is currently retrieved from the National Hurricane Center from 2006 onwards, and UCAR from 1989 through 2005.

    There are two recommended ways of retrieving a vdms object. Since the ``ReconDataset``, ``hdobs``, ``dropsondes`` and ``vdms`` classes are **storm-centric**, a Storm object is required for both methods.

    .. code-block:: python

        #Retrieve Hurricane Michael (2018) from TrackDataset
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))

    The first method is to use the empty instance of ReconDataset already initialized in the Storm object, which has a ``get_vdms()`` method thus allowing all of the vdms attributes and methods to be accessed from the Storm object. As a result, a Storm object does not need to be provided as an argument.

    .. code-block:: python

        #Retrieve all VDMs for this storm
        storm.recon.get_vdms()

        #Retrieve the raw VDM data
        storm.recon.vdms.data

        #Use the plot_points() method of hdobs
        storm.recon.vdms.plot_points()

    The second method is to use the vdms class independently of the other recon classes:

    .. code-block:: python

        from tropycal.recon import vdms

        #Retrieve all VDMs for this storm, passing the Storm object as an argument
        vdms_obj = vdms(storm)

        #Retrieve the raw VDM data
        vdms_obj.data

        #Use the plot_points() method of vdms
        vdms_obj.plot_points()
    """

    def __repr__(self):
        summary = ["<tropycal.recon.vdms>"]

        # Find maximum wind and minimum pressure
        time_range = (np.nanmin([i['time'] for i in self.data]), np.nanmax(
            [i['time'] for i in self.data]))
        time_range = list(set(time_range))
        min_slp = np.nanmin([i['Minimum Sea Level Pressure (hPa)']
                            for i in self.data])
        min_slp = 'N/A' if np.isnan(min_slp) else min_slp
        missions = set([i['mission'] for i in self.data])

        # Add general summary
        emdash = '\u2014'
        summary_keys = {
            'Storm': f'{self.storm.name} {self.storm.year}',
            'Missions': len(missions),
            'VDMs': len(self.data),
            'Min sea level pressure': f"{min_slp} hPa",
            'Source': self.source
        }

        # Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()]) + 3
        for key in summary_keys.keys():
            key_name = key + ":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)

    def __init__(self, storm, data=None, update=False):

        self.storm = storm
        self.source = 'National Hurricane Center (NHC)'
        if storm.year >= 2006:
            self.format = 1
            if storm.basin == 'north_atlantic':
                archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPNT2/'
            else:
                archive_url = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPPN2/'
        elif storm.year >= 1989:
            self.format = 2
            self.source = "UCAR's Tropical Cyclone Guidance Project (TCGP)"
            archive_url = f'http://hurricanes.ral.ucar.edu/structure/vortex/vdm_data/{self.storm.year}/'
        else:
            raise RuntimeError("Recon data is not available prior to 1989.")

        timestr = [f'{t:%Y%m%d}' for t in self.storm.dict['time']]

        # Retrieve list of files in URL and filter by storm dates
        page = requests.get(archive_url).text
        content = page.split("\n")
        files = []
        for line in content:
            if ".txt" in line:
                files.append(
                    ((line.split('txt">')[1]).split("</a>")[0]).split("."))
        del content
        if self.format == 1:
            files = sorted([i for i in files if i[1][:8]
                           in timestr], key=lambda x: x[1])
            linksub = [archive_url + '.'.join(l) for l in files]
        elif self.format == 2:
            files = [f[0] for f in files]
            linksub = [archive_url +
                       l for l in files if storm.name.upper() in l]

        self.data = []

        if data is None:

            urllib3.disable_warnings()
            http = urllib3.PoolManager()

            filecount = 0
            timer_start = dt.now()
            print(
                f'Searching through recon VDM files between {timestr[0]} and {timestr[-1]} ...')
            for link in linksub:
                response = http.request('GET', link)
                content = response.data.decode('utf-8')

                # Parse with NHC format
                if self.format == 1:
                    try:
                        date = link.split('.')[-2]
                        date = dt(int(date[:4]), int(
                            date[4:6]), int(date[6:8]))
                        missionname, tmp = decode_vdm(content, date)
                    except:
                        continue

                    testkeys = ('time', 'lat', 'lon')
                    if missionname[2:5] == self.storm.operational_id[2:4] + self.storm.operational_id[0]:
                        if self.data is None:
                            self.data = [copy.copy(tmp)]
                            filecount += 1
                        elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                            self.data.append(tmp)
                            filecount += 1
                        else:
                            pass

                # Parse with UCAR format
                elif self.format == 2:
                    content_split = content.split("URNT12")
                    content_split = ['URNT12' + i for i in content_split]
                    for iter_content in content_split:

                        try:
                            # Check for line length
                            iter_split = iter_content.split("\n")
                            if len(iter_split) < 10:
                                continue

                            # Check for date
                            for line in iter_split:
                                if line[:2] == 'A.':
                                    day = int((line[3:].split('/'))[0])
                            for iter_date in storm.dict['time']:
                                found_date = False
                                if iter_date.day == day:
                                    date = dt(iter_date.year,
                                              iter_date.month, iter_date.day)
                                    found_date = True
                                    break
                            if not found_date:
                                continue

                            # Decode VDMs
                            missionname, tmp = decode_vdm(iter_content, date)

                            testkeys = ('time', 'lat', 'lon')
                            if self.data is None:
                                self.data = [copy.copy(tmp)]
                                filecount += 1
                            elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                                self.data.append(tmp)
                                filecount += 1
                            else:
                                pass

                        except:
                            continue

            print(f'--> Completed reading in recon VDM files ({(dt.now()-timer_start).total_seconds():.1f} seconds)' +
                  f'\nRead {filecount} files')

        elif isinstance(data, str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data
        self.keys = sorted(list(set([k for d in self.data for k in d.keys()])))

    def update(self):
        r"""
        Update with the latest data for an ongoing storm.

        Notes
        -----
        This function has no return value, but simply updates the internal VDM data with new observations since the object was created.
        """

        newobj = vdms(storm=self.storm, data=self.data, update=True)
        return newobj

    def isel(self, index):
        r"""
        Select a single VDM by index of the list.

        Parameters
        ----------
        index : int
            Integer containing the index of the dropsonde.

        Returns
        -------
        vdms
            Instance of VDMs for the single requested VDM.
        """

        NEW_DATA = copy.copy(self.data)
        NEW_DATA = [NEW_DATA[index]]
        NEW_OBJ = vdms(storm=self.storm, data=NEW_DATA)

        return NEW_OBJ

    def sel(self, mission=None, time=None, domain=None):
        r"""
        Select a subset of VDMs by any of its parameters and return a new vdms object.

        Parameters
        ----------
        mission : str
            Mission name (number + storm id), e.g. mission 7 for AL05 is '0705L'
        time : list/tuple of datetimes
            list/tuple of start time and end time datetime objects.
            Default is None, which returns all points
        domain : dict
            dictionary with keys 'n', 's', 'e', 'w' corresponding to boundaries of domain

        Returns
        -------
        vdms object
            A new vdms object that satisfies the intersection of all subsetting.
        """

        NEW_DATA = copy.copy(pd.DataFrame(self.data))

        # Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission'] == mission]

        # Apply time filter
        if time is not None:
            bounds = get_bounds(NEW_DATA['time'], time)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['time'] >= bounds[0]) & (
                NEW_DATA['time'] <= bounds[1])]

        # Apply domain filter
        if domain is not None:
            tmp = {k[0].lower(): v for k, v in domain.items()}
            domain = {'n': 90, 's': -90, 'e': 359.99, 'w': 0}
            domain.update(tmp)
            bounds = get_bounds(
                NEW_DATA['lon'] % 360, (domain['w'] % 360, domain['e'] % 360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon'] % 360 >= bounds[0]) & (
                NEW_DATA['lon'] % 360 <= bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'], (domain['s'], domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat'] >= bounds[0]) & (
                NEW_DATA['lat'] <= bounds[1])]

        NEW_OBJ = vdms(storm=self.storm, data=list(
            NEW_DATA.T.to_dict().values()))

        return NEW_OBJ

    def to_pickle(self, filename):
        r"""
        Save VDM data (list of dictionaries) to a pickle file

        Parameters
        ----------
        filename : str
            name of file to save pickle file to.

        Notes
        -----
        This method saves the VDMs data as a pickle within the current working directory, given a filename as an argument.

        For example, assume ``vdms`` was retrieved from a Storm object (using the first method described in the ``vdms`` class documentation). The VDMs data would be saved to a pickle file as follows:

        >>> storm.recon.vdms.to_pickle(data="mystorm_vdms.pickle")

        Now the VDMs data is saved locally, and next time recon data for this storm needs to be analyzed, this allows to bypass re-reading the VDMs data from the NHC server by providing the pickle file as an argument:

        >>> storm.recon.get_vdms("mystorm_vdms.pickle")

        """

        with open(filename, 'wb') as f:
            pickle.dump(self.data, f)

    def plot_time_series(self, time=None, best_track=False, dots=True):
        r"""
        Creates a time series of MSLP VDM data.

        Parameters
        ----------
        time : tuple, optional
            Tuple of start and end datetime.datetime objects for plot. If None, all times will be plotted.
        best_track : bool, optional
            If True, Best Track MSLP will be plotted alongside VDM MSLP. Default is False.
        dots : bool, optional
            If True, dots will be plotted for each VDM point. Default is True.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Retrieve data
        storm_data = self.storm.dict
        data = self.data

        # Retrive data and subset by time
        if time is not None:
            times = [i['time'] for i in data if i['time']
                     >= time[0] and i['time'] <= time[1]]
            mslp = [i['Minimum Sea Level Pressure (hPa)']
                    for i in data if i['time'] >= time[0] and i['time'] <= time[1]]
        else:
            times = [i['time'] for i in data]
            mslp = [i['Minimum Sea Level Pressure (hPa)'] for i in data]

        # Create figure
        fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
        ax.grid()

        # Plot VDM MSLP
        ax.plot(times, mslp, color='b', alpha=0.5, label='VDM MSLP (hPa)')
        if dots:
            ax.plot(times, mslp, 'o', color='b')

        # Retrieve & plot Best Track data
        if best_track:
            if time is not None:
                times_btk = [i for i in storm_data['time']
                             if i >= time[0] and i <= time[1]]
                mslp_btk = [storm_data['mslp'][i] for i in range(len(
                    storm_data['mslp'])) if storm_data['time'][i] >= time[0] and storm_data['time'][i] <= time[1]]
            else:
                times_btk = [i for i in storm_data['time']]
                mslp_btk = [i for i in storm_data['mslp']]
            ax.plot(times_btk, mslp_btk, color='r',
                    alpha=0.25, label='Best Track MSLP (hPa)')
            if dots:
                ax.plot(times_btk, mslp_btk, 'o', color='r', alpha=0.5)

        # Add labels
        ax.set_ylabel("MSLP (hPa)")
        ax.set_xlabel("Vortex Data Message time (UTC)")

        # Add time labels
        times_use = []
        start_time = times[0].replace(hour=0)
        total_days = (times[-1] - start_time).total_seconds() / 86400
        increment_hour = 6
        if total_days > 3:
            increment_hour = 12
        if total_days > 6:
            increment_hour = 24
        while start_time <= (times[-1] + timedelta(hours=increment_hour)):
            times_use.append(start_time)
            start_time += timedelta(hours=increment_hour)
        ax.set_xticks(times_use)
        ax.set_xlim(times[0] - timedelta(hours=6),
                    times[-1] + timedelta(hours=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H UTC\n%b %d'))

        # Add titles
        type_array = np.array(storm_data['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))
        if ('invest' in storm_data.keys() and not storm_data['invest']) or len(idx[0]) > 0:
            tropical_vmax = np.array(storm_data['vmax'])[idx]

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

        if best_track:
            ax.legend()
        ax.set_title(
            f'{storm_type} {storm_data["name"]}\nVDM Minimum Sea Level Pressure (hPa)', loc='left', fontweight='bold')

        return ax

    def plot_points(self, varname='Minimum Sea Level Pressure (hPa)', domain="dynamic", ax=None, cartopy_proj=None, **kwargs):
        r"""
        Creates a plot of recon data points.

        Parameters
        ----------
        varname : str
            Variable to plot. Currently the best option is "Minimum Sea Level Pressure (hPa)".
        domain : str
            Domain for the plot. Default is "dynamic". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.

        Other Parameters
        ----------------
        prop : dict
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-plot` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Pop kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Get plot data
        plotdata = [m[varname] if varname in m.keys()
                    else np.nan for m in self.data]

        dfRecon = pd.DataFrame.from_dict({'time': [m['time'] for m in self.data],
                                          'lat': [m['lat'] for m in self.data],
                                          'lon': [m['lon'] for m in self.data],
                                          varname: plotdata})

        # Create instance of plot object
        self.plot_obj = ReconPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(
                proj='PlateCarree', central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj

        # Plot recon
        plot_ax = self.plot_obj.plot_points(
            self.storm, dfRecon, domain, varname=varname, ax=ax, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax
