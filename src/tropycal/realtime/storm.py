r"""Functionality for storing and analyzing an individual realtime storm."""

import numpy as np
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt, timedelta
import requests
from ftplib import FTP

from ..tracks import *
from ..tracks.tools import *
from ..tracks.plot import TrackPlot
from ..utils import *
from .. import constants
from ..recon import ReconDataset


class RealtimeStorm(Storm):

    r"""
    Initializes an instance of RealtimeStorm. This object inherits all the methods and functionality of ``tropycal.tracks.Storm``, but with additional methods unique to this object, all containing the word "realtime" as part of the function name.

    Parameters
    ----------
    storm : dict
        Dict entry of the requested storm.

    Returns
    -------
    RealtimeStorm
        Instance of a RealtimeStorm object.

    Notes
    -----
    A RealtimeStorm object is retrieved from a Realtime object's ``get_storm()`` method, or directly as an attribute of the Realtime object. For example, if an active storm has an ID of 'EP012022', it can be retrieved as such:

    .. code-block:: python

        from tropycal import realtime
        realtime_obj = realtime.Realtime()
        storm = realtime_obj.get_storm('EP012022')

    Now this storm's data is stored in the variable ``storm``, which is an instance of RealtimeStorm and can access all of the methods and attributes of a RealtimeStorm object.

    All the variables associated with a RealtimeStorm object (e.g., lat, lon, time, vmax) can be accessed in two ways. The first is directly from the RealtimeStorm object:

    >>> storm.lat
    array([ 9.8, 10.3, 10.8, 11.4, 11.9, 12.1, 12.2, 12.4, 12.6, 12.8, 13. ,
           12.9, 12.8, 12.9, 13.2, 13.6, 13.8, 13.9, 14. , 14. , 14.3, 14.6,
           15.1, 15.4])

    The second is via ``storm.vars``, which returns a dictionary of the variables associated with the RealtimeStorm object. This is also a quick way to access all of the variables associated with a RealtimeStorm object:

    >>> variable_dict = storm.vars
    >>> lat = variable_dict['lat']
    >>> lon = variable_dict['lon']
    >>> print(variable_dict.keys())
    dict_keys(['time', 'extra_obs', 'special', 'type', 'lat', 'lon', 'vmax', 'mslp', 'wmo_basin'])

    RealtimeStorm objects also have numerous attributes with information about the storm. ``storm.attrs`` returns a dictionary of the attributes for this RealtimeStorm object.

    It should be noted that RealtimeStorm objects have additional attributes that Storm objects do not, specifically for 2 and 5 day NHC formation probability. These only display values for invests within NHC's area of responsibility; tropical cyclones or invests in JTWC's area of responsibility display "N/A".

    >>> print(storm.attrs)
    {'id': 'EP012022',
     'operational_id': 'EP012022',
     'name': 'AGATHA',
     'year': 2022,
     'season': 2022,
     'basin': 'east_pacific',
     'source_info': 'NHC Hurricane Database',
     'invest': False,
     'source_method': "NHC's Automated Tropical Cyclone Forecasting System (ATCF)",
     'source_url': 'https://ftp.nhc.noaa.gov/atcf/btk/',
     'source': 'hurdat',
     'ace': 6.055,
     'prob_2day': 'N/A',
     'prob_5day': 'N/A',
     'risk_2day': 'N/A',
     'risk_5day': 'N/A',
     'realtime': True}

    """

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):

        # Label object
        summary = ["<tropycal.realtime.RealtimeStorm>"]

        # Format keys for summary
        type_array = np.array(self.dict['type'])
        idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (
            type_array == 'TS') | (type_array == 'HU') | (type_array == 'TY') | (type_array == 'ST'))[0]
        if self.invest and len(idx) == 0:
            idx = np.array([True for i in type_array])
        if len(idx) == 0:
            start_time = 'N/A'
            end_time = 'N/A'
            max_wind = 'N/A'
            min_mslp = 'N/A'
        else:
            time_tropical = np.array(self.dict['time'])[idx]
            start_time = time_tropical[0].strftime("%H00 UTC %d %B %Y")
            end_time = time_tropical[-1].strftime("%H00 UTC %d %B %Y")
            max_wind = 'N/A' if all_nan(np.array(self.dict['vmax'])[idx]) else int(np.nanmax(np.array(self.dict['vmax'])[idx]))
            min_mslp = 'N/A' if all_nan(np.array(self.dict['mslp'])[idx]) else int(np.nanmin(np.array(self.dict['mslp'])[idx]))
        summary_keys = {
            'Maximum Wind': f"{max_wind} knots",
            'Minimum Pressure': f"{min_mslp} hPa",
            'Start Time': start_time,
            'End Time': end_time,
        }

        # Format keys for coordinates
        variable_keys = {}
        for key in self.vars.keys():
            dtype = type(self.vars[key][0]).__name__
            dtype = dtype.replace("_", "")
            variable_keys[key] = f"({dtype}) [{self.vars[key][0]} .... {self.vars[key][-1]}]"

        # Add storm summary
        summary.append("Storm Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        # Add coordinates
        summary.append("\nVariables:")
        add_space = np.max([len(key) for key in variable_keys.keys()])+3
        for key in variable_keys.keys():
            key_name = key
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{variable_keys[key]}')

        # Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.attrs.keys()])+3
        for key in self.attrs.keys():
            key_name = key+":"
            val = '%0.1f' % (
                self.attrs[key]) if key == 'ace' else self.attrs[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')

        return "\n".join(summary)

    def __init__(self, storm, stormTors=None):

        # Save the dict entry of the storm
        self.dict = storm

        # Add other attributes about the storm
        keys = self.dict.keys()
        self.attrs = {}
        self.vars = {}
        for key in keys:
            if key == 'realtime':
                continue
            if not isinstance(self.dict[key], list) and not isinstance(self.dict[key], dict):
                self[key] = self.dict[key]
                self.attrs[key] = self.dict[key]
            if isinstance(self.dict[key], list) and not isinstance(self.dict[key], dict):
                self.vars[key] = np.array(self.dict[key])
                self[key] = np.array(self.dict[key])

        # Assign tornado data
        if stormTors is not None and isinstance(stormTors, dict):
            self.stormTors = stormTors['data']
            self.tornado_dist_thresh = stormTors['dist_thresh']
            self.attrs['Tornado Count'] = len(stormTors['data'])

        # Get Archer track data for this storm, if it exists
        try:
            self.get_archer()
        except:
            pass

        # Initialize recon dataset instance
        self.recon = ReconDataset(storm=self)

        # Determine if storm object was retrieved via realtime object
        if 'realtime' in keys and self.dict['realtime']:
            self.realtime = True
            self.attrs['realtime'] = True
        else:
            self.realtime = False
            self.attrs['realtime'] = False

    def get_realtime_formation_prob(self):
        r"""
        Retrieve the latest NHC formation probability. Only valid for invests within NHC's area of responsibility.

        Returns
        -------
        dict
            Dictionary containing latest NHC forecast formation probability, if available. If none, defaults to zero or N/A.
        """

        return {
            'prob_2day': self.dict['prob_2day'],
            'risk_2day': self.dict['risk_2day'],
            'prob_7day': self.dict['prob_7day'],
            'risk_7day': self.dict['risk_7day']
        }

    def download_graphic_realtime(self, save_path=""):
        r"""
        Download the latest official forecast track graphic. Available for both NHC and JTWC sources.

        Parameters
        ----------
        save_path : str
            Filepath to save the image in. If blank, default is current working directory.
        """

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Determine data source
        if self.source == 'hurdat':
            part1 = f"AT{self.id[2:4]}" if self.id[0:2] == "AL" else self.id[0:4]
            url = f"https://www.nhc.noaa.gov/storm_graphics/{part1}/{self.id}_5day_cone_with_line_and_wind.png"
        else:
            url = f"https://www.nrlmry.navy.mil/atcf_web/docs/current_storms/{self.id.lower()}.gif"
        url_ext = url.split(".")[-1]

        # Try to download file
        if requests.get(url).status_code != 200:
            raise RuntimeError(
                "Official forecast graphic is unavailable for this storm.")

        # Download file
        response = requests.get(url)
        full_path = os.path.join(save_path, f"Forecast_{self.id}.{url_ext}")
        with open(full_path, 'wb') as f:
            f.write(response.content)

    def get_discussion_realtime(self):
        r"""
        Retrieve the latest available forecast discussion. For JTWC storms, the Prognostic Reasoning product is retrieved.

        Returns
        -------
        dict
            Dict entry containing the latest official forecast discussion.
        """

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Get latest forecast discussion for HURDAT source storm objects
        if self.source == "hurdat":
            return self.get_nhc_discussion(forecast=-1)

        # Get latest forecast discussion for JTWC source storm objects
        elif self.source == 'jtwc':

            # Read in discussion file
            url = f"https://www.metoc.navy.mil/jtwc/products/{self.id[0:2].lower()}{self.id[2:4]}{self.id[6:8]}prog.txt"
            f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            f.close()
            return content

        # Otherwise, return error message
        else:
            msg = "No realtime forecast discussion is available for this storm."
            raise RuntimeError(msg)

    def get_forecast_realtime(self, ssl_certificate=True):
        r"""
        Retrieve a dictionary containing the latest official forecast. Available for both NHC and JTWC sources.

        Parameters
        ----------
        ssl_certificate : boolean, optional
            If a JTWC forecast, this determines whether to disable SSL certificate when retrieving data from JTWC. Default is True. Use False *ONLY* if True causes an SSL certification error.

        Returns
        -------
        dict
            Dictionary containing the latest official forecast.

        Notes
        -----
        This dictionary includes a calculation for accumulated cyclone energy (ACE), cumulatively for the storm's lifespan through each forecast hour. This is done by linearly interpolating the forecast to 6-hour intervals and calculating 6-hourly ACE at each interval. For storms where forecast tropical cyclone type is available, ACE is not calculated for forecast periods that are neither tropical nor subtropical.
        """

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # NHC forecast data
        if self.source == 'hurdat':

            # Get forecast for this storm
            try:
                content = read_url(
                    f"https://ftp.nhc.noaa.gov/atcf/fst/{self.id.lower()}.fst")
            except:
                try:
                    content = read_url(
                        f"ftp://ftp.nhc.noaa.gov/atcf/fst/{self.id.lower()}.fst")
                except:
                    raise RuntimeError(
                        "NHC forecast data is unavailable for this storm.")

            # Iterate through every line in content:
            forecasts = {}

            for line in content:

                # Get basic components
                lineArray = [i.replace(" ", "") for i in line]
                if len(lineArray) < 11:
                    continue
                try:
                    basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype, rad, windcode, neq, seq, swq, nwq = lineArray[:17]
                    use_wind = True
                except:
                    basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype = lineArray[:11]
                    use_wind = False
                if model not in ["OFCL", "OFCI"]:
                    continue

                if len(forecasts) == 0:
                    forecasts = {
                        'init': dt.strptime(run_init, '%Y%m%d%H'),
                        'fhr': [],
                        'lat': [],
                        'lon': [],
                        'vmax': [],
                        'mslp': [],
                        'type': [],
                        'windrad': [],
                        'cumulative_ace': [],
                        'cumulative_ace_fhr': [],
                    }

                # Format lat & lon
                fhr = int(fhr)
                if "N" in lat:
                    lat_temp = lat.split("N")[0]
                    lat = np.round(float(lat_temp) * 0.1, 1)
                elif "S" in lat:
                    lat_temp = lat.split("S")[0]
                    lat = np.round(float(lat_temp) * -0.1, 1)
                if "W" in lon:
                    lon_temp = lon.split("W")[0]
                    lon = np.round(float(lon_temp) * -0.1, 1)
                elif "E" in lon:
                    lon_temp = lon.split("E")[0]
                    lon = np.round(float(lon_temp) * 0.1, 1)

                # Format vmax & MSLP
                if vmax == '':
                    vmax = np.nan
                else:
                    vmax = int(vmax)
                    if vmax < 10 or vmax > 300:
                        vmax = np.nan
                if mslp == '':
                    mslp = np.nan
                else:
                    mslp = int(mslp)
                    if mslp < 1:
                        mslp = np.nan

                # Format wind radii
                if use_wind:
                    try:
                        rad = int(rad)
                        if rad in [0, 35]:
                            rad = 34
                        neq = np.nan if windcode == '' else int(neq)
                        seq = np.nan if windcode in ['', 'AAA'] else int(seq)
                        swq = np.nan if windcode in ['', 'AAA'] else int(swq)
                        nwq = np.nan if windcode in ['', 'AAA'] else int(nwq)
                    except:
                        rad = 34
                        neq = np.nan
                        seq = np.nan
                        swq = np.nan
                        nwq = np.nan
                else:
                    rad = 34
                    neq = np.nan
                    seq = np.nan
                    swq = np.nan
                    nwq = np.nan

                # Add forecast data to dict if forecast hour isn't already there
                if fhr not in forecasts['fhr']:
                    if model in ['OFCL', 'OFCI'] and fhr > 120:
                        pass
                    else:
                        if lat == 0.0 and lon == 0.0:
                            continue
                        forecasts['fhr'].append(fhr)
                        forecasts['lat'].append(lat)
                        forecasts['lon'].append(lon)
                        forecasts['vmax'].append(vmax)
                        forecasts['mslp'].append(mslp)
                        forecasts['windrad'].append(
                            {rad: [neq, seq, swq, nwq]})

                        # Get storm type, if it can be determined
                        if stype in ['', 'DB'] and vmax != 0 and not np.isnan(vmax):
                            stype = get_storm_type(vmax, False)
                        forecasts['type'].append(stype)
                else:
                    ifhr = forecasts['fhr'].index(fhr)
                    forecasts['windrad'][ifhr][rad] = [neq, seq, swq, nwq]

        # Retrieve JTWC forecast otherwise
        else:

            # Get forecast for this storm
            if self.jtwc_source in ['jtwc', 'ucar']:
                url = f"https://www.nrlmry.navy.mil/atcf_web/docs/current_storms/{self.id.lower()}.sum"
            else:
                url = f"https://www.ssd.noaa.gov/PS/TROP/DATA/ATCF/JTWC/{self.id.lower()}.fst"
            if not ssl_certificate and self.jtwc_source in ['jtwc', 'ucar']:
                import ssl
                if requests.get(url, verify=False).status_code != 200:
                    raise RuntimeError(
                        "JTWC forecast data is unavailable for this storm.")
            else:
                if requests.get(url).status_code != 200:
                    raise RuntimeError(
                        "JTWC forecast data is unavailable for this storm.")

            # Read file content
            if not ssl_certificate and self.jtwc_source in ['jtwc', 'ucar']:
                import ssl
                f = urllib.request.urlopen(
                    url, context=ssl._create_unverified_context())
            else:
                f = urllib.request.urlopen(url)
            content = f.read()
            content = content.decode("utf-8")
            content = content.split("\n")
            f.close()

            # Find starting index
            start_idx = 0
            for idx, line in enumerate(content):
                lineArray = line.split(" ")
                if 'WARNING' in lineArray[0] and len(lineArray) > 2:
                    start_idx = idx
                    break

            if self.jtwc_source in ['jtwc', 'ucar']:
                # Iterate through every line in content:
                run_init = content[start_idx+2].split(" ")[0]
                forecasts = {}

                for line in content[start_idx+3:]:

                    # Exit once done retrieving forecast
                    if line == "AMP":
                        break

                    # Get basic components
                    lineArray = line.split(" ")
                    if len(lineArray) < 4:
                        continue

                    # Exit once done retrieving forecast
                    if lineArray[0] == "AMP":
                        break

                    if len(forecasts) == 0:
                        forecasts = {
                            'init': dt.strptime(run_init, '%Y%m%d%H'),
                            'fhr': [],
                            'lat': [],
                            'lon': [],
                            'vmax': [],
                            'mslp': [],
                            'windrad': [],
                            'cumulative_ace': [],
                            'cumulative_ace_fhr': [],
                            'type': []
                        }

                    # Forecast hour
                    fhr = int(lineArray[0].split("T")[1])

                    # Format lat & lon
                    lat = lineArray[1]
                    lon = lineArray[2]
                    if "N" in lat:
                        lat_temp = lat.split("N")[0]
                        lat = np.round(float(lat_temp) * 0.1, 1)
                    elif "S" in lat:
                        lat_temp = lat.split("S")[0]
                        lat = np.round(float(lat_temp) * -0.1, 1)
                    if "W" in lon:
                        lon_temp = lon.split("W")[0]
                        lon = np.round(float(lon_temp) * -0.1, 1)
                    elif "E" in lon:
                        lon_temp = lon.split("E")[0]
                        lon = np.round(float(lon_temp) * 0.1, 1)

                    # Format vmax & MSLP
                    vmax = int(lineArray[3])
                    if vmax < 10 or vmax > 300:
                        vmax = np.nan
                    mslp = np.nan

                    # Format wind radii
                    windrad = {}
                    for rad in (34, 50, 64):
                        try:
                            irad = list(lineArray).index(f'R{rad:03}')
                            windrad[rad] = [int(lineArray[irad+j])
                                            for j in (1, 4, 7, 10)]
                        except:
                            continue

                    # Add forecast data to dict if forecast hour isn't already there
                    if fhr not in forecasts['fhr']:
                        if lat == 0.0 and lon == 0.0:
                            continue
                        forecasts['fhr'].append(fhr)
                        forecasts['lat'].append(lat)
                        forecasts['lon'].append(lon)
                        forecasts['vmax'].append(vmax)
                        forecasts['mslp'].append(mslp)
                        forecasts['windrad'].append(windrad)

                        # Get storm type, if it can be determined
                        stype = get_storm_type(vmax, False)
                        forecasts['type'].append(stype)

            else:

                content = [(i.replace(" ", "")).split(",") for i in content]

                # Iterate through every line in content:
                forecasts = {}

                for line in content:

                    # Get basic components
                    lineArray = [i.replace(" ", "") for i in line]
                    if len(lineArray) < 11:
                        continue
                    try:
                        basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype, rad, windcode, neq, seq, swq, nwq = lineArray[ :17]
                        use_wind = True
                    except:
                        basin, number, run_init, n_a, model, fhr, lat, lon, vmax, mslp, stype = lineArray[:11]
                        use_wind = False
                    if model not in ["JTWC"]:
                        continue

                    if len(forecasts) == 0:
                        forecasts = {
                            'init': dt.strptime(run_init, '%Y%m%d%H'),
                            'fhr': [],
                            'lat': [],
                            'lon': [],
                            'vmax': [],
                            'mslp': [],
                            'type': [],
                            'windrad': [],
                            'cumulative_ace': [],
                            'cumulative_ace_fhr': []
                        }

                    # Format lat & lon
                    fhr = int(fhr)
                    if "N" in lat:
                        lat_temp = lat.split("N")[0]
                        lat = np.round(float(lat_temp) * 0.1, 1)
                    elif "S" in lat:
                        lat_temp = lat.split("S")[0]
                        lat = np.round(float(lat_temp) * -0.1, 1)
                    if "W" in lon:
                        lon_temp = lon.split("W")[0]
                        lon = np.round(float(lon_temp) * -0.1, 1)
                    elif "E" in lon:
                        lon_temp = lon.split("E")[0]
                        lon = np.round(float(lon_temp) * 0.1, 1)

                    # Format vmax & MSLP
                    if vmax == '':
                        vmax = np.nan
                    else:
                        vmax = int(vmax)
                        if vmax < 10 or vmax > 300:
                            vmax = np.nan
                    if mslp == '':
                        mslp = np.nan
                    else:
                        mslp = int(mslp)
                        if mslp < 1:
                            mslp = np.nan

                    # Format wind radii
                    if use_wind:
                        try:
                            rad = int(rad)
                            if rad in [0, 35]:
                                rad = 34
                            neq = np.nan if windcode == '' else int(neq)
                            seq = np.nan if windcode in [
                                '', 'AAA'] else int(seq)
                            swq = np.nan if windcode in [
                                '', 'AAA'] else int(swq)
                            nwq = np.nan if windcode in [
                                '', 'AAA'] else int(nwq)
                        except:
                            rad = 34
                            neq = np.nan
                            seq = np.nan
                            swq = np.nan
                            nwq = np.nan
                    else:
                        rad = 34
                        neq = np.nan
                        seq = np.nan
                        swq = np.nan
                        nwq = np.nan

                    # Add forecast data to dict if forecast hour isn't already there
                    if fhr not in forecasts['fhr']:
                        if model in ['OFCL', 'OFCI'] and fhr > 120:
                            pass
                        else:
                            if lat == 0.0 and lon == 0.0:
                                continue
                            forecasts['fhr'].append(fhr)
                            forecasts['lat'].append(lat)
                            forecasts['lon'].append(lon)
                            forecasts['vmax'].append(vmax)
                            forecasts['mslp'].append(mslp)
                            forecasts['windrad'].append(
                                {rad: [neq, seq, swq, nwq]})

                            # Get storm type, if it can be determined
                            if stype in ['', 'DB'] and vmax != 0 and not np.isnan(vmax):
                                stype = get_storm_type(vmax, False)
                            forecasts['type'].append(stype)
                    else:
                        ifhr = forecasts['fhr'].index(fhr)
                        forecasts['windrad'][ifhr][rad] = [neq, seq, swq, nwq]

        # Determine ACE thus far (prior to initial forecast hour)
        ace = 0.0
        for i in range(len(self.time)):
            if self.time[i] >= forecasts['init']:
                continue
            if self.type[i] not in constants.NAMED_TROPICAL_STORM_TYPES:
                continue
            ace += accumulated_cyclone_energy(self.vmax[i], hours=6)

        # Add initial forecast hour ACE
        ace += accumulated_cyclone_energy(forecasts['vmax'][0], hours=6)
        forecasts['cumulative_ace_fhr'].append(0)
        forecasts['cumulative_ace'].append(np.round(ace, 1))

        # Interpolate forecast to 6-hour increments
        def temporal_interpolation(value, orig_times, target_times, kind='linear'):
            f = interp.interp1d(orig_times, value, kind=kind,
                                fill_value='extrapolate')
            ynew = f(target_times)
            return ynew
        # Construct a 6-hour time range
        interp_fhr = range(0, forecasts['fhr'][-1]+1, 6)
        interp_vmax = temporal_interpolation(
            forecasts['vmax'], forecasts['fhr'], interp_fhr)

        # Interpolate storm type
        interp_type = []
        for dummy_i, (i_hour, i_vmax) in enumerate(zip(interp_fhr, interp_vmax)):
            use_i = 0
            for i in range(len(forecasts['fhr'])):
                if forecasts['fhr'][i] > i_hour:
                    break
                use_i = int(i + 0.0)
            i_type = forecasts['type'][use_i]
            if i_type in constants.TROPICAL_STORM_TYPES:
                i_type = get_storm_type(i_vmax, False)
            interp_type.append(i_type)

        # Add forecast ACE
        for i, (i_fhr, i_vmax, i_type) in enumerate(zip(interp_fhr[1:], interp_vmax[1:], interp_type[1:])):

            # Add ACE if storm is a TC
            if i_type in constants.NAMED_TROPICAL_STORM_TYPES:
                ace += accumulated_cyclone_energy(i_vmax, hours=6)

            # Add ACE to array
            if i_fhr in forecasts['fhr']:
                forecasts['cumulative_ace'].append(np.round(ace, 1))
                forecasts['cumulative_ace_fhr'].append(i_fhr)

        # Save forecast as attribute
        self.latest_forecast = forecasts
        return self.latest_forecast

    def plot_forecast_realtime(self, track_labels='fhr', cone_days=5, domain="dynamic_forecast",
                               ax=None, cartopy_proj=None, save_path=None, ssl_certificate=True, **kwargs):
        r"""
        Plots the latest available official forecast. Available for both NHC and JTWC sources.

        Parameters
        ----------
        track_labels : str
            Label forecast hours with the following methods:

            * **""** = no label
            * **"fhr"** = forecast hour (default)
            * **"valid_utc"** = UTC valid time
            * **"valid_edt"** = EDT valid time
        cone_days : int
            Number of days to plot the forecast cone. Default is 5 days. Can select 2, 3, 4 or 5 days.
        domain : str
            Domain for the plot. Default is "dynamic_forecast". Please refer to :ref:`options-domain` for available domain options.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        save_path : str
            Relative or full path of directory to save the image in. If none, image will not be saved.

        Other Parameters
        ----------------
        ssl_certificate : boolean, optional
            If a JTWC forecast, this determines whether to disable SSL certificate when retrieving data from JTWC. Default is True. Use False *ONLY* if True causes an SSL certification error.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.

        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """

        # Retrieve kwargs
        prop = kwargs.pop('prop', {})
        map_prop = kwargs.pop('map_prop', {})

        # Check to ensure storm is not an invest
        if self.invest:
            raise RuntimeError(
                "Error: NHC does not issue advisories for invests that have not been designated as Potential Tropical Cyclones.")

        # Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()

        # Create cartopy projection
        if cartopy_proj is None:
            if max(self.dict['lon']) > 140 or min(self.dict['lon']) < -140:
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(
                    proj='PlateCarree', central_longitude=0.0)

        # Get forecast for this storm
        try:
            nhc_forecasts = (self.latest_forecast).copy()
        except:
            nhc_forecasts = self.get_forecast_realtime(
                ssl_certificate=ssl_certificate)

        # Add other info to forecast dict
        nhc_forecasts['advisory_num'] = -1
        nhc_forecasts['basin'] = self.basin
        if self.source != "hurdat":
            nhc_forecasts['cone'] = False

        # Plot storm
        plot_ax = self.plot_obj.plot_storm_nhc(
            nhc_forecasts, self.dict, track_labels, cone_days, domain, ax=ax, save_path=save_path, prop=prop, map_prop=map_prop)

        # Return axis
        return plot_ax

    def get_realtime_info(self, source='all'):
        r"""
        Returns a dict containing the latest available information about the storm. This function uses NHC Public Advisories, so it will differ from available Best Track data.

        Parameters
        ----------
        source : str
            Data source to use. Default is "all". Available options are:

            * **"all"** = Latest from either public advisory or best track. Both NHC & JTWC.
            * **"public_advisory"** = Latest public advisory. NHC only.
            * **"best_track"** = Latest Best Track file data. Both NHC & JTWC.

        Returns
        -------
        dict
            Dictionary containing current storm information.
        """

        # Error check
        if not isinstance(source, str):
            msg = "\"source\" must be of type str."
            raise TypeError(msg)
        if source not in ['all', 'public_advisory', 'best_track']:
            msg = "\"source\" must be 'all', 'public_advisory', or 'best_track'."
            raise ValueError(msg)
        if source == 'public_advisory' and self.source != 'hurdat':
            msg = "A source of 'public_advisory' can only be used for storms in NHC's area of responsibility."
            raise RuntimeError(msg)

        # Check to ensure storm is not an invest
        if self.invest:
            if self.source == 'hurdat':
                msg = "NHC does not issue public advisories on invests. Defaulting to best track method."
                warnings.warn(msg)
            source = 'best_track'

        # Declare empty dict
        current_advisory = {}

        # If source is all, determine which method to use
        if source == 'all':
            if self.source == 'hurdat':
                # Check to see which is the latest advisory
                latest_btk = self.time[-1]

                # Get latest available public advisory
                try:
                    content = read_url(
                        f"https://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml", subsplit=False)
                except:
                    content = read_url(
                        f"ftp://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml", subsplit=False)

                # Get UTC time of advisory
                results = [i for i in content if 'messageDateTimeUTC' in i][0]
                result = (results.split(">")[1]).split("<")[0]
                latest_advisory = dt.strptime(result, '%Y%m%d %I:%M:%S %p UTC')

                # Check which one to use
                if latest_btk > latest_advisory:
                    source = 'best_track'
                else:
                    source = 'public_advisory'
            else:
                source = 'best_track'

        # If public advisory, retrieve this data
        if source == 'public_advisory':

            # Add source
            current_advisory['source'] = 'NHC Public Advisory'

            # Get latest available public advisory
            try:
                content = read_url(
                    f"https://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml", subsplit=False)
            except:
                content = read_url(
                    f"ftp://ftp.nhc.noaa.gov/atcf/adv/{self.id.lower()}_info.xml", subsplit=False)

            # Get public advisory number
            results = [i for i in content if 'advisoryNumber' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['advisory_number'] = result

            # Get UTC time of advisory
            results = [i for i in content if 'messageDateTimeUTC' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            result = dt.strptime(result, '%Y%m%d %I:%M:%S %p UTC')
            current_advisory['time_utc'] = result

            # Get storm type
            results = [i for i in content if 'systemType' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['type'] = result.title()

            # Get storm name
            results = [i for i in content if 'systemName' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['name'] = result.title()

            # Get coordinates
            results = [i for i in content if 'centerLocLatitude' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['lat'] = float(result)
            results = [i for i in content if 'centerLocLongitude' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['lon'] = float(result)

            # Get sustained wind speed
            results = [i for i in content if 'systemIntensityMph' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['wind_mph'] = int(result)
            results = [i for i in content if 'systemIntensityKph' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['wind_kph'] = int(result)
            results = [i for i in content if 'systemIntensityKts' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['wind_kt'] = int(result)

            # Get MSLP
            results = [i for i in content if 'systemMslpMb' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['mslp'] = int(result)

            # Get storm category
            current_advisory['category'] = wind_to_category(
                current_advisory['wind_kt'])

            # Get storm direction
            results = [i for i in content if 'systemDirectionOfMotion' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['motion_direction'] = result.split(" OR ")[0]
            results = [i for i in content if 'systemDirectionOfMotion' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            try:
                current_advisory['motion_direction_degrees'] = int(
                    (result.split(" OR ")[1]).split(" DEGREES")[0])
            except:
                current_advisory['motion_direction_degrees'] = 0

            # Get storm speed
            results = [i for i in content if 'systemSpeedMph' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['motion_mph'] = int(result)
            results = [i for i in content if 'systemSpeedKph' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['motion_kph'] = int(result)
            results = [i for i in content if 'systemSpeedKts' in i][0]
            result = (results.split(">")[1]).split("<")[0]
            current_advisory['motion_kt'] = int(result)

        # Best track data
        else:

            # Add source
            if self.source == 'hurdat':
                current_advisory['source'] = 'NHC Best Track'
            else:
                current_advisory['source'] = 'JTWC Best Track'

            # Get public advisory number
            current_advisory['advisory_number'] = 'n/a'

            # Get UTC time of advisory
            current_advisory['time_utc'] = self.time[-1]

            # Get storm type
            subtrop_flag = self.type[-1] in constants.SUBTROPICAL_ONLY_STORM_TYPES
            current_advisory['type'] = get_storm_classification(
                self.vmax[-1], subtrop_flag, self.wmo_basin[-1])

            # Check for non-tropical storm types
            if self.type[-1] not in constants.TROPICAL_STORM_TYPES:
                if all(type not in self.type for type in constants.TROPICAL_STORM_TYPES):
                    if self.invest:
                        current_advisory['type'] = 'Invest'
                    else:
                        current_advisory['type'] = 'Potential Tropical Cyclone'
                else:
                    current_advisory['type'] = 'Post-Tropical Cyclone'
            elif self.source in ['jtwc', 'ucar']:
                if self.invest:
                    current_advisory['type'] = 'Invest'

            # Get storm name
            current_advisory['name'] = self.name.title()

            # Get coordinates
            current_advisory['lat'] = self.lat[-1]
            current_advisory['lon'] = self.lon[-1]

            # Get sustained wind speed
            current_advisory['wind_mph'] = knots_to_mph(self.vmax[-1])
            current_advisory['wind_kph'] = int(self.vmax[-1] * 1.852)
            current_advisory['wind_kt'] = self.vmax[-1]

            # Get MSLP
            current_advisory['mslp'] = int(self.mslp[-1])

            # Get storm category
            current_advisory['category'] = wind_to_category(
                current_advisory['wind_kt'])

            # Determine motion direction and degrees
            try:

                # Cannot calculate motion if there's only one data point
                if len(self.lon) == 1:

                    # Get storm direction
                    current_advisory['motion_direction'] = 'n/a'
                    current_advisory['motion_direction_degrees'] = 'n/a'

                    # Get storm speed
                    current_advisory['motion_mph'] = 'n/a'
                    current_advisory['motion_kph'] = 'n/a'
                    current_advisory['motion_kt'] = 'n/a'

                # Otherwise, use great_circle to calculate
                else:

                    # Get points
                    start_point = (self.lat[-2], self.lon[-2])
                    end_point = (self.lat[-1], self.lon[-1])

                    # Get time since last update
                    hour_diff = (
                        self.time[-1] - self.time[-2]).total_seconds() / 3600.0

                    # Calculate zonal and meridional position change in km
                    x = great_circle(
                        (self.lat[-2], self.lon[-2]), (self.lat[-2], self.lon[-1])).kilometers
                    if self.lon[-1] < self.lon[-2]:
                        x = x * -1

                    y = great_circle(
                        (self.lat[-2], self.lon[-2]), (self.lat[-1], self.lon[-2])).kilometers
                    if self.lat[-1] < self.lat[-2]:
                        y = y * -1

                    # Calculate motion direction vector
                    idir = np.degrees(np.arctan2(x, y))
                    if idir < 0:
                        idir += 360.0

                    # Calculate motion direction string
                    def deg_str(d):
                        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
                        ix = int((d + 11.25)/22.5)
                        return dirs[ix % 16]
                    dirs = deg_str(idir)

                    # Update storm direction
                    current_advisory['motion_direction'] = dirs
                    current_advisory['motion_direction_degrees'] = int(
                        np.round(idir, 0))

                    # Get storm speed
                    current_advisory['motion_mph'] = int(
                        np.round(great_circle(start_point, end_point).miles / float(hour_diff), 0))
                    current_advisory['motion_kph'] = int(np.round(great_circle(
                        start_point, end_point).kilometers / float(hour_diff), 0))
                    current_advisory['motion_kt'] = int(
                        np.round(current_advisory['motion_mph'] * 0.868976, 0))

            # Otherwise, can't calculate motion
            except:

                # Get storm direction
                current_advisory['motion_direction'] = 'n/a'
                current_advisory['motion_direction_degrees'] = 'n/a'

                # Get storm speed
                current_advisory['motion_mph'] = 'n/a'
                current_advisory['motion_kph'] = 'n/a'
                current_advisory['motion_kt'] = 'n/a'

        # Return dict
        return current_advisory

    def __get_public_advisory(self):

        # Get list of all public advisories for this storm
        url_disco = 'https://ftp.nhc.noaa.gov/atcf/pub/'
        try:
            page = requests.get(url_disco).text
            content = page.split("\n")
            files = []
            for line in content:
                if ".public" in line and self.id.lower() in line:
                    filename = line.split('">')[1]
                    filename = filename.split("</a>")[0]
                    files.append(filename)
            del content
        except:
            ftp = FTP('ftp.nhc.noaa.gov')
            ftp.login()
            ftp.cwd('atcf/pub')
            files = ftp.nlst()
            out = ftp.quit()

        # Keep only largest number
        numbers = [int(i.split(".")[-1]) for i in files]
        max_number = np.nanmax(numbers)
        if max_number >= 100:
            max_number = str(max_number)
        elif max_number >= 10:
            max_number = f"0{max_number}"
        else:
            max_number = f"00{max_number}"
        files = [i for i in files if f".{max_number}" in i]

        # Determine if there's an intermediate advisory available
        if len(files) > 1:
            advisory_letter = []
            for file in files:
                if 'public_' in file:
                    letter = (file.split("public_")[1]).split(".")[0]
                    advisory_letter.append(letter)
            max_letter = max(advisory_letter)
            files = [i for i in files if f".public_{max_letter}" in i]

        # Read file containing advisory
        content = read_url(url_disco + files[0], subsplit=False)

        # Figure out time issued
        hr = content[6].split(" ")[0]
        zone = content[6].split(" ")[2]
        disco_time = num_to_str2(int(hr)) + ' '.join(content[6].split(" ")[1:])

        format_time = content[6].split(" ")[0]
        if len(format_time) == 3:
            format_time = "0" + format_time
        format_time = format_time + " " + ' '.join(content[6].split(" ")[1:])
        disco_time = dt.strptime(format_time, f'%I00 %p {zone} %a %b %d %Y')

        time_zones = {
            'ADT': -3,
            'AST': -4,
            'EDT': -4,
            'EST': -5,
            'CDT': -5,
            'CST': -6,
            'MDT': -6,
            'MST': -7,
            'PDT': -7,
            'PST': -8,
            'HDT': -9,
            'HST': -10
        }
        offset = time_zones.get(zone, 0)
        disco_time = disco_time + timedelta(hours=offset*-1)

    def __get_ensembles_eps(self):
        # This function is currently not functioning. The path to retrieve EPS ensemble data is:
        # ftp://wmo:essential@dissemination.ecmwf.int/20200518120000/
        # A_JSXX01ECEP181200_C_ECMP_20200518120000_tropical_cyclone_track_AMPHAN_86p3degE_14degN_bufr4.bin
        return
