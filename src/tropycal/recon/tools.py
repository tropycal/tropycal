import numpy as np
import pandas as pd
from datetime import datetime as dt, timedelta
from scipy.ndimage import gaussian_filter as gfilt
from scipy.interpolate import griddata, interp1d
import matplotlib.dates as mdates
import copy

from ..utils import classify_subtropical, get_storm_classification


def uv_from_wdir(wspd, wdir):
    d2r = np.pi / 180.
    theta = (270 - wdir) * d2r
    u = wspd * np.cos(theta)
    v = wspd * np.sin(theta)
    return u, v

# ------------------------------------------------------------------------------
# TOOLS FOR RECON INTERPOLATION
# ------------------------------------------------------------------------------


class interpRecon:

    """
    Interpolates storm-centered data by time and space.
    """

    def __init__(self, dfRecon, varname, radlim=None, window=6, align='center'):

        # Retrieve dataframe containing recon data, and variable to be interpolated
        self.dfRecon = dfRecon
        self.varname = varname
        self.window = window
        self.align = align

        # Specify outer radius cutoff in kilometer
        if radlim is None:
            self.radlim = 200  # km
        else:
            self.radlim = radlim

    def interpPol(self):
        r"""
        Interpolates storm-centered recon data into a polar grid, and outputs the radius grid, azimuth grid and interpolated variable.
        """

        # Read in storm-centered data and storm-relative coordinates for all times
        data = [k for i, j, k in zip(self.dfRecon['xdist'], self.dfRecon['ydist'],
                                     self.dfRecon[self.varname]) if not np.isnan([i, j, k]).any()]
        path = [(i, j) for i, j, k in zip(self.dfRecon['xdist'], self.dfRecon['ydist'],
                                          self.dfRecon[self.varname]) if not np.isnan([i, j, k]).any()]

        # Function for interpolating cartesian to polar coordinates
        def cart2pol(x, y, offset=0):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return (rho, phi + offset)

        # Interpolate every storm-centered coordinate pair into polar coordinates
        pol_path = [cart2pol(*p) for p in path]

        # Wraps around the data to ensure no cutoff around 0 degrees
        pol_path_wrap = [cart2pol(*p, offset=-2 * np.pi) for p in path] + pol_path +\
            [cart2pol(*p, offset=2 * np.pi) for p in path]
        data_wrap = np.concatenate([data] * 3)

        # Creates a grid of rho (radius) and phi (azimuth)
        grid_rho, grid_phi = np.meshgrid(
            np.arange(0, self.radlim + .1, .5), np.linspace(-np.pi, np.pi, 181))

        # Interpolates storm-centered point data in polar coordinates onto a gridded polar coordinate field
        grid_z_pol = griddata(pol_path_wrap, data_wrap,
                              (grid_rho, grid_phi), method='linear')

        try:
            # Calculate radius of maximum wind (RMW)
            rmw = grid_rho[0, np.nanargmax(np.mean(grid_z_pol, axis=0))]

            # Within the RMW, replace NaNs with minimum value within the RMW
            filleye = np.where((grid_rho < rmw) & (np.isnan(grid_z_pol)))

            grid_z_pol[filleye] = np.nanmin(
                grid_z_pol[np.where(grid_rho < rmw)])
        except:
            pass

        # Return fields
        return grid_rho, grid_phi, grid_z_pol

    def interpCart(self):
        r"""
        Interpolates polar storm-centered gridded fields into cartesian coordinates
        """

        # Interpolate storm-centered recon data into gridded polar grid (rho, phi and gridded data)
        grid_rho, grid_phi, grid_z_pol = self.interpPol()

        # Calculate RMW
        rmw = grid_rho[0, np.nanargmax(np.mean(grid_z_pol, axis=0))]

        # Wraps around the data to ensure no cutoff around 0 degrees
        grid_z_pol_wrap = np.concatenate([grid_z_pol] * 3)

        # Radially smooth based on RMW - more smoothing farther out from RMW
        grid_z_pol_final = np.array([gfilt(grid_z_pol_wrap, (6, 3 + abs(r - rmw) / 10))[:, i]
                                     for i, r in enumerate(grid_rho[0, :])]).T[len(grid_phi):2 * len(grid_phi)]

        # Function for interpolating polar cartesian to coordinates
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        # Interpolate the rho and phi gridded fields to a 1D cartesian list
        pinterp_grid = [pol2cart(i, j) for i, j in zip(
            grid_rho.flatten(), grid_phi.flatten())]

        # Flatten the radially smoothed variable grid to match with the shape of pinterp_grid
        pinterp_z = grid_z_pol_final.flatten()

        # Setting up the grid in cartesian coordinate space, based on previously specified radial limit
        # Grid resolution = 1 kilometer
        grid_x, grid_y = np.meshgrid(np.linspace(-self.radlim, self.radlim, self.radlim * 2 + 1),
                                     np.linspace(-self.radlim, self.radlim, self.radlim * 2 + 1))
        grid_z = griddata(pinterp_grid, pinterp_z,
                          (grid_x, grid_y), method='linear')

        # Return output grid
        return grid_x, grid_y, grid_z

    def interpHovmoller(self, target_track):
        r"""
        Creates storm-centered interpolated data in polar coordinates for each timestep, and averages azimuthally to create a hovmoller.

        target_track = dict
            dict of either archer or hurdat data (contains lat, lon, time/date)
        window = hours
            sets window in hours relative to the time of center pass for interpolation use.
        """

        window = self.window
        align = self.align

        # Store the dataframe containing recon data
        tmpRecon = self.dfRecon.copy()
        # Sets window as a timedelta object
        window = timedelta(seconds=int(window * 3600))

        # Error check for time dimension name
        if 'time' not in target_track.keys():
            target_track['time'] = target_track['date']

        # Find times of all center passes
        centerTimes = tmpRecon[tmpRecon['iscenter'] == 1]['time']

        # Data is already centered on center time, so shift centerTimes to the end of the window
        spaceInterpTimes = [t + window / 2 for t in centerTimes]

        # Takes all times within track dictionary that fall between spaceInterpTimes
        trackTimes = [t for t in target_track['time'] if min(
            spaceInterpTimes) < t < max(spaceInterpTimes)]

        # Iterate through all data surrounding a center pass given the window previously specified, and create a polar
        # grid for each
        start_time = dt.now()
        print("--> Starting interpolation")

        spaceInterpData = {}
        for time in spaceInterpTimes:
            # Temporarily set dfRecon to this centered subset window
            self.dfRecon = tmpRecon[(
                tmpRecon['time'] > time - window) & (tmpRecon['time'] <= time)]
            # print(time) #temporarily disabling this
            grid_rho, grid_phi, grid_z_pol = self.interpPol()  # Create polar centered grid
            grid_azim_mean = np.mean(grid_z_pol, axis=0)  # Average azimuthally
            # Append data for this time step to dictionary
            spaceInterpData[time] = grid_azim_mean

        # Sets dfRecon back to original full data
        self.dfRecon = tmpRecon
        reconArray = np.array([i for i in spaceInterpData.values()])

        # Interpolate over every half hour
        newTimes = np.arange(mdates.date2num(
            trackTimes[0]), mdates.date2num(trackTimes[-1]) + 1e-3, 1 / 48)
        oldTimes = mdates.date2num(np.array(list(spaceInterpData.keys())))
        # print(len(oldTimes),reconArray.shape)
        reconTimeInterp = np.apply_along_axis(lambda x: np.interp(newTimes, oldTimes, x),
                                              axis=0, arr=reconArray)
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(), 2))
        print(f"--> Completed interpolation ({tsec} seconds)")

        # Output RMW and hovmoller data and store as an attribute in the object
        self.rmw = grid_rho[0, np.nanargmax(reconTimeInterp, axis=1)]
        self.Hovmoller = {'time': mdates.num2date(
            newTimes), 'radius': grid_rho[0, :], 'hovmoller': reconTimeInterp}
        return self.Hovmoller

    def interpMaps(self, target_track, interval=0.5, stat_vars=None):
        r"""
        1. Can just output a single map (interpolated to lat/lon grid and projected onto the Cartopy map
        2. If target_track is longer than 1, outputs multiple maps to a directory
        """

        window = self.window
        align = self.align

        # Store the dataframe containing recon data
        tmpRecon = self.dfRecon.copy()
        # Sets window as a timedelta object
        window = timedelta(seconds=int(window * 3600))

        # Error check for time dimension name
        if 'time' not in target_track.keys():
            target_track['time'] = target_track['date']

        # If target_track > 1 (tuple or list of times), then retrieve multiple center pass times and center around the window
        if isinstance(target_track['time'], (tuple, list, np.ndarray)):
            centerTimes = tmpRecon[tmpRecon['iscenter'] == 1]['time']
            spaceInterpTimes = [t for t in centerTimes]
            trackTimes = [t for t in target_track['time'] if min(
                spaceInterpTimes) - window / 2 < t < max(spaceInterpTimes) + window / 2]
        # Otherwise, just use a single time
        else:
            spaceInterpTimes = list([target_track['time']])
            trackTimes = spaceInterpTimes.copy()

        # Experimental - add recon statistics (e.g., wind, MSLP) to plot
        # **** CHECK BACK ON THIS ****
        spaceInterpData = {}
        recon_stats = None
        if stat_vars is not None:
            recon_stats = {name: [] for name in stat_vars.keys()}
        # Iterate through all data surrounding a center pass given the window previously specified, and create a polar
        # grid for each
        start_time = dt.now()
        print("--> Starting interpolation")

        for time in spaceInterpTimes:
            print(time)
            self.dfRecon = tmpRecon[(
                tmpRecon['time'] > time - window / 2) & (tmpRecon['time'] <= time + window / 2)]
            grid_x, grid_y, grid_z = self.interpCart()
            spaceInterpData[time] = grid_z
            if stat_vars is not None:
                for name in stat_vars.keys():
                    recon_stats[name].append(
                        stat_vars[name](self.dfRecon[name]))

        # Sets dfRecon back to original full data
        self.dfRecon = tmpRecon
        reconArray = np.array([i for i in spaceInterpData.values()])

        # If multiple times, create a lat & lon grid for half hour intervals
        if len(trackTimes) > 1:
            newTimes = np.arange(mdates.date2num(trackTimes[0]), mdates.date2num(
                trackTimes[-1]) + interval / 24, interval / 24)
            oldTimes = mdates.date2num(np.array(list(spaceInterpData.keys())))
            reconTimeInterp = np.apply_along_axis(lambda x: np.interp(newTimes, oldTimes, x),
                                                  axis=0, arr=reconArray)
            # Get centered lat and lon by interpolating from target_track dictionary (whether archer or HURDAT)
            clon = np.interp(newTimes, mdates.date2num(
                target_track['time']), target_track['lon'])
            clat = np.interp(newTimes, mdates.date2num(
                target_track['time']), target_track['lat'])
        else:
            newTimes = mdates.date2num(trackTimes)[0]
            reconTimeInterp = reconArray[0]
            clon = target_track['lon']
            clat = target_track['lat']

        # Interpolate storm stats to corresponding times
        if stat_vars is not None:
            for varname in recon_stats.keys():
                recon_stats[varname] = np.interp(
                    newTimes, oldTimes, recon_stats[varname])

        # Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(), 2))
        print(f"--> Completed interpolation ({tsec} seconds)")

        # Create dict of map data (interpolation time, x & y grids, 'maps' (3D grid of field, time/x/y), and return
        self.Maps = {'time': mdates.num2date(newTimes), 'grid_x': grid_x, 'grid_y': grid_y, 'maps': reconTimeInterp,
                     'center_lon': clon, 'center_lat': clat, 'stats': recon_stats}
        return self.Maps

    @staticmethod
    def _interpFunc(data1, times1, times2):
        # Interpolate data
        f = interp1d(mdates.date2num(times1), data1)
        data2 = f(mdates.date2num(times2))
        return data2

# ------------------------------------------------------------------------------
# TOOLS FOR PLOTTING
# ------------------------------------------------------------------------------


def find_var(request, thresh):
    r"""
    Given a variable and threshold, returns the variable for plotting. Referenced from ``TrackDataset.gridded_stats()`` and ``TrackPlot.plot_gridded()``. Internal function.

    Parameters
    ----------
    request : str
        Descriptor of the requested plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the request.
    varname : str
        String denoting the variable for plotting.
    """

    # Convert command to lowercase
    request = request.lower()

    # Count of number of storms
    if request.find('count') >= 0 or request.find('num') >= 0:
        return thresh, 'time'

    if request.find('time') >= 0 or request.find('day') >= 0:
        return thresh, 'time'

    # Sustained wind, or change in wind speed
    if request.find('wind') >= 0 or request.find('vmax') >= 0:
        return thresh, 'wspd'
    if request.find('30s wind') >= 0 or request.find('vmax') >= 0:
        return thresh, 'wspd'
    if request.find('10s wind') >= 0 or request.find('vmax') >= 0:
        return thresh, 'pkwnd'

    # Minimum MSLP
    elif request.find('pressure') >= 0 or request.find('slp') >= 0:
        return thresh, 'p_sfc'

    # Otherwise, error
    else:
        msg = "Error: Could not decipher variable. Please refer to documentation for examples on how to phrase the \"request\" string."
        raise RuntimeError(msg)


def find_func(request, thresh):
    r"""
    Given a request and threshold, returns the requested function. Referenced from ``TrackDataset.gridded_stats()``. Internal function.

    Parameters
    ----------
    request : str
        Descriptor of the requested plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the request.
    func : lambda
        Returns a function to apply to the data.
    """

    print(request)

    # Convert command to lowercase
    request = request.lower()

    # Numpy maximum function
    if request.find('max') == 0 or request.find('latest') == 0:
        return thresh, lambda x: np.nanmax(x)

    # Numpy minimum function
    if request.find('min') == 0 or request.find('earliest') == 0:
        return thresh, lambda x: np.nanmin(x)

    # Numpy average function
    elif request.find('mean') >= 0 or request.find('average') >= 0 or request.find('avg') >= 0:
        # Ensure sample minimum is at least 5 per gridpoint
        thresh['sample_min'] = max([5, thresh['sample_min']])
        return thresh, lambda x: np.nanmean(x)

    # Numpy percentile function
    elif request.find('percentile') >= 0:
        ptile = int(''.join([c for i, c in enumerate(
            request) if c.isdigit() and i < request.find('percentile')]))
        # Ensure sample minimum is at least 5 per gridpoint
        thresh['sample_min'] = max([5, thresh['sample_min']])
        return thresh, lambda x: np.nanpercentile(x, ptile)

    # Count function
    elif request.find('count') >= 0 or request.find('num') >= 0:
        return thresh, lambda x: len(x)

    # ACE - cumulative function
    elif request.find('ace') >= 0:
        return thresh, lambda x: np.nansum(x)
    elif request.find('acie') >= 0:
        return thresh, lambda x: np.nansum(x)

    # Otherwise, function cannot be identified
    else:
        msg = "Cannot decipher the function. Please refer to documentation for examples on how to phrase the \"request\" string."
        raise RuntimeError(msg)


def get_recon_title(varname, level=None):
    r"""
    Generate title descriptor for plots.
    """

    if level is None:
        if varname.lower() == 'top':
            titlename = 'Top mandatory level'
            unitname = r'(hPa)'
        if varname.lower() == 'slp':
            titlename = 'Sea level pressure'
            unitname = r'(hPa)'
        if varname.lower() == 'mbldir':
            titlename = '500m mean wind direction'
            unitname = r'(deg)'
        if varname.lower() == 'wl150dir':
            titlename = '150m mean wind direction'
            unitname = r'(dir)'
        if varname.lower() == 'mblspd':
            titlename = '500m mean wind speed'
            unitname = r'(kt)'
        if varname.lower() == 'wl150spd':
            titlename = '150m mean wind speed'
            unitname = r'(kt)'
        if varname.lower() == 'wspd':
            titlename = '30s FL wind speed'
            unitname = r'(kt)'
        if varname.lower() == 'pkwnd':
            titlename = '10s FL wind speed'
            unitname = r'(kt)'
        if varname.lower() == 'sfmr':
            titlename = 'SFMR 10s sfc wind speed'
            unitname = r'(kt)'
        if varname.lower() == 'p_sfc':
            titlename = 'Sfc pressure'
            unitname = r'(hPa)'
        if varname.lower() == 'plane_p':
            titlename = 'Flight level pressure'
            unitname = r'(hPa)'
        if varname.lower() == 'plane_z':
            titlename = 'Flight level height'
            unitname = r'(m)'
    else:
        if varname.lower() == 'pres':
            titlename = f'{level}hPa pressure'
            unitname = r'(hPa)'
        if varname.lower() == 'hgt':
            titlename = f'{level}hPa height'
            unitname = r'(m)'
        if varname.lower() == 'temp':
            titlename = f'{level}hPa temperature'
            unitname = r'(deg C)'
        if varname.lower() == 'dwpt':
            titlename = f'{level}hPa dew point'
            unitname = r'(deg C)'
        if varname.lower() == 'wdir':
            titlename = f'{level}hPa wind direction'
            unitname = r'(deg)'
        if varname.lower() == 'wspd':
            titlename = f'{level}hPa wind speed'
            unitname = r'(kt)'

    return titlename, unitname


def hovmoller_plot_title(storm_obj, Hov, varname):
    r"""
    Generate plot title for hovmoller.
    """

    # Retrieve storm dictionary from Storm object
    storm_data = storm_obj.dict

    # ------- construct left title ---------

    # Subset sustained wind array to when the storm was tropical
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

    # Determine storm classification based on subtropical status & basin
    subtrop = classify_subtropical(np.array(storm_data['type']))
    peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
    peak_basin = storm_data['wmo_basin'][peak_idx]
    storm_type = get_storm_classification(
        np.nanmax(tropical_vmax), subtrop, peak_basin)
    if add_ptc_flag:
        storm_type = "Potential Tropical Cyclone"

    # Get title descriptor based on variable
    vartitle = get_recon_title(varname)

    # Add left title
    dot = u"\u2022"
    title_left = f"{storm_type} {storm_data['name']}\n" + \
        'Recon: ' + ' '.join(vartitle)

    # ------- construct right title ---------

    # Determine start and end dates of hovmoller
    start_time = dt.strftime(min(Hov['time']), '%H:%M UTC %d %b %Y')
    end_time = dt.strftime(max(Hov['time']), '%H:%M UTC %d %b %Y')
    title_right = f'Start ... {start_time}\nEnd ... {end_time}'

    # Return both titles
    return title_left, title_right


def get_bounds(data, bounds):
    try:
        datalims = (np.nanmin(data), np.nanmax(data))
    except:
        datalims = bounds
    bounds = [l if b is None else b for b, l in zip(bounds, datalims)]
    bounds = [b for b in bounds if b is not None]
    return (min(bounds), max(bounds))


def time_series_plot(varname):
    r"""
    Returns a default color and name associated with each varname for hdob time series.

    Parameters
    ----------
    varname : str
        Requested variable name.

    Returns
    -------
    dict
        Dictionary with color, name and full name for variable to plot.
    """

    colors = {
        'p_sfc': 'red',
        'temp': 'red',
        'dwpt': 'green',
        'wspd': 'blue',
        'sfmr': '#282893',
        'pkwnd': '#5A9AF0',
        'rain': '#C551DC',
        'plane_z': '#909090',
        'plane_p': '#4D4D4D',
    }

    names = {
        'p_sfc': 'MSLP',
        'temp': 'Temperature',
        'dwpt': 'Dewpoint',
        'wspd': 'Flight Level Wind',
        'sfmr': 'Surface Wind',
        'pkwnd': 'Peak Wind Gust',
        'rain': 'Rain Rate',
        'plane_z': 'Altitude',
        'plane_p': 'Pressure',
    }

    full_names = {
        'p_sfc': 'Mean Sea Level Pressure (hPa)',
        'temp': 'Temperature (C)',
        'dwpt': 'Dewpoint (C)',
        'wspd': 'Flight Level Wind (kt)',
        'sfmr': 'Surface Wind (kt)',
        'pkwnd': 'Peak Wind Gust (kt)',
        'rain': 'Rain Rate (mm/hr)',
        'plane_z': 'Geopotential Height (m)',
        'plane_p': 'Pressure (hPa)',
    }

    color = colors.get(varname, '')
    name = names.get(varname, '')
    full_name = full_names.get(varname, '')

    return {'color': color, 'name': name, 'full_name': full_name}

# =======================================================================================================
# Decoding HDOBs
# =======================================================================================================


def decode_hdob_2005_noaa(content, strdate, mission_row=0):
    r"""
    Function for decoding HDOBs in the format between 2002 and 2005, for NOAA aircraft.
    """

    def check_error(string):
        if '/' in string:
            return True
        if ';' in string:
            return True
        if 'off' in string:
            return True
        return False

    # Split items by lines
    temp = [i.split() for i in content.split('\n')]
    temp = [i for j, i in enumerate(temp) if len(i) > 0]
    items = []
    found_sxxx = False
    for j, i in enumerate(temp):
        if j > mission_row and len(i[0]) >= 2 and i[0][:2] == "$$":
            break
        if j > mission_row + 12 and len(i[0]) >= 3 and i[0][:3] == "000":
            break  # avoid reading in extra set of HDOBs
        if j <= mission_row:
            if len(i[0]) >= 4 and i[0][:3] in ["URN", "URP"]:
                if not found_sxxx:
                    found_sxxx = True
                    items.append(i)
            else:
                items.append(i)
        if j > mission_row and i[0][0].isdigit() and len(i) > 8:
            items.append(i)

    # Parse dates
    data = {}
    data['time'] = [dt.strptime(strdate + i[0], '%Y%m%d%H%M%S')
                    for i in items[3:]]
    if data['time'][0].hour > 12 and data['time'][-1].hour < 12:
        data['time'] = [t + timedelta(days=[0, 1][t.hour < 12])
                        for t in data['time']]

    # Derive plane altitude using D-value
    altitude = []
    for i in items[3:]:
        if '/' in i[3] or '/' in i[4]:
            altitude.append(np.nan)
        else:
            pres_altitude = int(i[3])
            d_value = int(i[4][1:]) * -1 if i[4][0] == '-' else int(i[4][1:])
            altitude.append(pres_altitude + d_value)
    data['plane_z'] = altitude

    # Parse full data
    data['lat'] = [np.nan if check_error(i[1]) else round((float(i[1][:-2]) + float(i[1][-2:]) / 60), 2)
                   for i in items[3:]]
    data['lon'] = [np.nan if check_error(i[2]) else round((float(i[2][:-2]) + float(i[2][-2:]) / 60), 2) * -1
                   for i in items[3:]]
    data['temp'] = [np.nan if check_error(i[6]) else round(
        float(i[7]) * 0.1, 1) for i in items[3:]]
    data['dwpt'] = [np.nan if check_error(i[7]) else round(
        float(i[8]) * 0.1, 1) for i in items[3:]]
    data['wdir'] = [np.nan if check_error(i[5][:3]) else round(
        float(i[5][:3]), 0) for i in items[3:]]
    data['wspd'] = [np.nan if check_error(i[5][3:]) else round(
        float(i[5][3:]), 0) for i in items[3:]]
    data['pkwnd'] = [np.nan if check_error(i[8]) else round(
        float(i[8][3:]), 0) for i in items[3:]]

    # Fix erroneous data
    data['wspd'] = [np.nan if i > 300 else i for i in data['wspd']]
    data['pkwnd'] = [np.nan if i > 300 else i for i in data['pkwnd']]

    # Data not available prior to 2007
    data['plane_p'] = [np.nan for i in items[3:]]
    data['p_sfc'] = [np.nan for i in items[3:]]
    data['sfmr'] = [np.nan for i in items[3:]]
    data['rain'] = [np.nan for i in items[3:]]
    data['flag'] = [[] for i in items[3:]]

    # Ignore entries with lat/lon of 0
    orig_lat = np.copy(data['lat'])
    orig_lon = np.copy(data['lon'])
    for key in data.keys():
        data[key] = [data[key][i] for i in range(len(orig_lat)) if orig_lat[i] != 0 and orig_lon[i] != 0 and not np.isnan(
            orig_lat[i]) and not np.isnan(orig_lat[i])]

    # Identify mission number and ID
    content_split = content.split("\n")
    mission_id = '-'.join(
        (content_split[mission_row].replace("  ", " ")).split(" ")[:3])
    missionname = (mission_id.split("-")[1])[:2]
    data['mission'] = [missionname[:2]] * len(data['time'])
    data['mission_id'] = [mission_id] * len(data['time'])

    # remove nan's for lat/lon coordinates
    return_data = pd.DataFrame.from_dict(data).reset_index()
    return_data = return_data.dropna(subset=['lat', 'lon'])

    return return_data


def decode_hdob_2006(content, strdate, mission_row=3):
    r"""
    Function for decoding HDOBs in the format between 2006 and early 2007. This also serves as the USAF decoder between 2002 and 2005.
    """

    def check_error(string):
        if '/' in string:
            return True
        if ';' in string:
            return True
        if 'off' in string:
            return True
        return False

    # Split items by lines
    temp = [i.split() for i in content.split('\n')]
    temp = [i for j, i in enumerate(temp) if len(i) > 0]
    items = []
    found_sxxx = False
    for j, i in enumerate(temp):
        if j > mission_row and len(i[0]) >= 2 and i[0][:2] == "$$":
            break
        if j > mission_row + 12 and len(i[0]) >= 3 and i[0][:3] == "000":
            break  # avoid reading in extra set of HDOBs
        if j <= mission_row:
            if len(i[0]) >= 4 and i[0][:4] == "SXXX":
                if not found_sxxx:
                    found_sxxx = True
                    items.append(i)
            else:
                items.append(i)
        if j > mission_row and i[0][0].isdigit() and len(i) > 8:
            if int(strdate[0:4]) >= 2006:
                items.append(i)
            else:
                if '.' in i[0] and i[0][-1] != '.' and i[0][-2] != '.':
                    new_i = [i[0].split(".")[0] + '.',
                             i[0].split(".")[1]] + i[1:]
                else:
                    new_i = i
                items.append(new_i)

    # Parse dates
    missionname = items[2][1]
    data = {}
    data['time'] = [dt.strptime(strdate + (i[0].split('.')[0]) + '30', '%Y%m%d%H%M%S') if '.' in i[0]
                    else dt.strptime(strdate + i[0] + '00', '%Y%m%d%H%M%S') for i in items[3:]]
    if data['time'][0].hour > 12 and data['time'][-1].hour < 12:
        data['time'] = [t + timedelta(days=[0, 1][t.hour < 12])
                        for t in data['time']]

    # Derive plane altitude using D-value
    altitude = []
    for i in items[3:]:
        if '/' in i[3] or '/' in i[4]:
            altitude.append(np.nan)
        else:
            pres_altitude = int(i[3])
            d_value = int(i[4][1:]) * -1 if i[4][0] == '5' else int(i[4][1:])
            altitude.append(pres_altitude + d_value)
    data['plane_z'] = altitude

    # Parse full data
    data['lat'] = [np.nan if check_error(i[1]) else round((float(i[1][:-3]) + float(i[1][-3:-1]) / 60) * [-1, 1][i[1][-1] == 'N'], 2)
                   for i in items[3:]]
    data['lon'] = [np.nan if check_error(i[2]) else round((float(i[2][:-3]) + float(i[2][-3:-1]) / 60) * [-1, 1][i[2][-1] == 'E'], 2)
                   for i in items[3:]]
    data['temp'] = [np.nan if check_error(i[6]) else round(
        float(i[7]) * 0.1, 1) for i in items[3:]]
    data['dwpt'] = [np.nan if check_error(i[7]) else round(
        float(i[8]) * 0.1, 1) for i in items[3:]]
    data['wdir'] = [np.nan if check_error(i[5]) else round(
        float(i[5]), 0) for i in items[3:]]
    data['wspd'] = [np.nan if check_error(i[6]) else round(
        float(i[6]), 0) for i in items[3:]]
    data['pkwnd'] = [np.nan if check_error(
        i[9]) else round(float(i[9]), 0) for i in items[3:]]

    # Data not available prior to 2007
    data['plane_p'] = [np.nan for i in items[3:]]
    data['p_sfc'] = [np.nan for i in items[3:]]
    data['sfmr'] = [np.nan for i in items[3:]]
    data['rain'] = [np.nan for i in items[3:]]
    data['flag'] = [[] for i in items[3:]]

    # Fix erroneous data
    data['wspd'] = [np.nan if i > 300 else i for i in data['wspd']]
    data['pkwnd'] = [np.nan if i > 300 else i for i in data['pkwnd']]

    # Ignore entries with lat/lon of 0
    orig_lat = np.copy(data['lat'])
    orig_lon = np.copy(data['lon'])
    for key in data.keys():
        data[key] = [data[key][i] for i in range(len(orig_lat)) if orig_lat[i] != 0 and orig_lon[i] != 0 and not np.isnan(
            orig_lat[i]) and not np.isnan(orig_lat[i])]

    # Identify mission number and ID
    content_split = content.split("\n")
    mission_id = '-'.join(
        (content_split[mission_row].replace("  ", " ")).split(" ")[:3])
    if int(strdate[0:4]) < 2006:
        missionname = (mission_id.split("-")[1])[:2]
    data['mission'] = [missionname[:2]] * len(data['time'])
    data['mission_id'] = [mission_id] * len(data['time'])

    # remove nan's for lat/lon coordinates
    return_data = pd.DataFrame.from_dict(data).reset_index()
    return_data = return_data.dropna(subset=['lat', 'lon'])
    if np.nanmax(data['lat']) < 0:
        return_data = {}

    return return_data


def decode_hdob(content, mission_row=3):
    r"""
    Function for decoding HDOBs in the present format (2007-present).
    """

    # Split items by lines
    tmp = [i.split() for i in content.split('\n')]
    tmp = [i for j, i in enumerate(tmp) if len(i) > 0]
    items = []
    for j, i in enumerate(tmp):
        if j > mission_row and len(i[0]) >= 2 and i[0][:2] == "$$":
            break
        if j > mission_row + 12 and len(i[0]) >= 3 and i[0][:3] == "000":
            break  # avoid reading in extra set of HDOBs
        if j <= mission_row:
            items.append(i)
        if j > mission_row and i[0][0].isdigit() and len(i) > 3:
            items.append(i)

    # Parse dates
    missionname = items[2][1]
    data = {}
    data['time'] = [dt.strptime(
        items[2][-1] + i[0], '%Y%m%d%H%M%S') for i in items[3:]]
    if data['time'][0].hour > 12 and data['time'][-1].hour < 12:
        data['time'] = [t + timedelta(days=[0, 1][t.hour < 12])
                        for t in data['time']]

    # Parse full data
    data['lat'] = [np.nan if '/' in i[1] else round((float(i[1][:-3]) + float(i[1][-3:-1]) / 60) * [-1, 1][i[1][-1] == 'N'], 2)
                   for i in items[3:]]
    data['lon'] = [np.nan if '/' in i[2] else round((float(i[2][:-3]) + float(i[2][-3:-1]) / 60) * [-1, 1][i[2][-1] == 'E'], 2)
                   for i in items[3:]]
    data['plane_p'] = [np.nan if '/' in i[3]
                       else round(float(i[3]) * 0.1 + [0, 1000][float(i[3]) < 1000], 1) for i in items[3:]]
    data['plane_z'] = [np.nan if '/' in i[4]
                       else round(float(i[4]), 0) for i in items[3:]]
    data['p_sfc'] = [np.nan if (('/' in i[5]) | (p < 550))
                     else round(float(i[5]) * 0.1 + [0, 1000][float(i[5]) < 1000], 1) for i, p in zip(items[3:], data['plane_p'])]
    data['temp'] = [np.nan if '/' in i[6]
                    else round(float(i[6]) * 0.1, 1) for i in items[3:]]
    data['dwpt'] = [np.nan if '/' in i[7]
                    else round(float(i[7]) * 0.1, 1) for i in items[3:]]
    data['wdir'] = [np.nan if '/' in i[8][:3]
                    else round(float(i[8][:3]), 0) for i in items[3:]]
    data['wspd'] = [np.nan if '/' in i[8][3:]
                    else round(float(i[8][3:]), 0) for i in items[3:]]
    data['pkwnd'] = [np.nan if '/' in i[9]
                     else round(float(i[9]), 0) for i in items[3:]]
    data['sfmr'] = [np.nan if '/' in i[10]
                    else round(float(i[10]), 0) for i in items[3:]]
    data['rain'] = [np.nan if '/' in i[11]
                    else round(float(i[11]), 0) for i in items[3:]]

    # Fix erroneous SFMR
    data['sfmr'] = [np.nan if i > 300 else i for i in data['sfmr']]
    data['wspd'] = [np.nan if i > 300 else i for i in data['wspd']]
    data['pkwnd'] = [np.nan if i > 300 else i for i in data['pkwnd']]

    # Ignore entries with lat/lon of 0
    orig_lat = np.copy(data['lat'])
    orig_lon = np.copy(data['lon'])
    for key in data.keys():
        data[key] = [data[key][i] for i in range(
            len(orig_lat)) if orig_lat[i] != 0 and orig_lon[i] != 0]

    # Add flag
    data['flag'] = []
    for i in items[3:]:
        flag = []
        if int(i[12][0]) in [1, 3]:
            flag.extend(['lat', 'lon'])
        if int(i[12][0]) in [2, 3]:
            flag.extend(['plane_p', 'plane_z'])
        if int(i[12][1]) in [1, 4, 5, 9]:
            flag.extend(['temp', 'dwpt'])
        if int(i[12][1]) in [2, 4, 6, 9]:
            flag.extend(['wdir', 'wspd', 'pkwnd'])
        if int(i[12][1]) in [3, 5, 6, 9]:
            flag.extend(['sfmr', 'rain'])
        data['flag'].append(flag)

    # QC p_sfc
    if any(abs(np.gradient(data['p_sfc'], np.array(data['time']).astype('datetime64[s]').astype(float))) > 1):
        data['p_sfc'] = [np.nan] * len(data['p_sfc'])
        data['flag'] = [d.append('p_sfc') for d in data['flag']]

    # Identify mission number and ID
    content_split = content.split("\n")
    mission_id = '-'.join(
        (content_split[mission_row].replace("  ", " ")).split(" ")[:3])
    data['mission'] = [missionname[:2]] * len(data['time'])
    data['mission_id'] = [mission_id] * len(data['time'])

    # remove nan's for lat/lon coordinates
    return_data = pd.DataFrame.from_dict(data).reset_index()
    return_data = return_data.dropna(subset=['lat', 'lon'])

    return return_data

# =======================================================================================================
# Decoding VDMs
# =======================================================================================================


def decode_vdm(content, date):
    data = {}
    lines = content.split('\n')
    RemarksNext = False
    LonNext = False
    missionname = ''

    if date.year >= 2018:
        FORMAT = 1
    elif date.year >= 1999:
        FORMAT = 2
    elif date.year == 1998:
        FORMAT = 3
    else:
        FORMAT = 4

    def isNA(x):
        if x == 'NA':
            return np.nan
        else:
            try:
                return float(x)
            except:
                return x.lower()

    for line in lines:

        if RemarksNext:
            data['Remarks'] += (' ' + line)
        if LonNext:
            info = line.split()
            data['lon'] = np.round(
                (float(info[0]) + float(info[2]) / 60) * [-1, 1][info[4] == 'E'], 2)
            LonNext = False

        if 'VORTEX DATA MESSAGE' in line:
            stormid = line.split()[-1]
        if line[:2] == 'A.':
            if ':' in line[3:]:
                info = line[3:].split('/')
                day = int(info[0])
                month = (date.month - int(day - date.day > 15) - 1) % 12 + 1
                year = date.year - \
                    int(date.month - int(day - date.day > 15) == 0)
                hour, minute, second = [int(i) for i in (
                    info[1].split("Z")[0]).split(':')]
                data['time'] = dt(year, month, day, hour, minute, second)
            else:
                info = line[3:].split('/')
                day = int(info[0])
                month = (date.month - int(day - date.day > 15) - 1) % 12 + 1
                year = date.year - \
                    int(date.month - int(day - date.day > 15) == 0)
                hour = int(info[1].split("Z")[0][:2])
                minute = int(info[1].split("Z")[0][2:4])
                data['time'] = dt(year, month, day, hour, minute)

        if line[:2] == 'B.':
            info = line[3:].split()
            if FORMAT >= 2:
                data['lat'] = np.round(
                    (float(info[0]) + float(info[2]) / 60) * [-1, 1][info[4] == 'N'], 2)
                LonNext = True
            if FORMAT == 1:
                data['lat'] = float(info[0]) * [-1, 1][info[2] == 'N']
                data['lon'] = float(info[3]) * [-1, 1][info[5] == 'E']

        if line[:2] == 'C.':
            info = line[3:].split() * 5
            data[f'Standard Level (hPa)'] = isNA(info[0])
            data[f'Minimum Height at Standard Level (m)'] = isNA(info[2])

        if line[:2] == 'D.':
            info = line[3:].split() * 5
            if FORMAT >= 2:
                data['Estimated Maximum Surface Wind Inbound (kt)'] = isNA(
                    info[0])
            if FORMAT == 1:
                data['Minimum Sea Level Pressure (hPa)'] = isNA(info[-2])

        if line[:2] == 'E.':
            info = line[3:].split() * 5
            if FORMAT >= 2:
                data['Dropsonde Surface Wind Speed at Center (kt)'] = isNA(
                    info[2])
                data['Dropsonde Surface Wind Direction at Center (deg)'] = isNA(
                    info[0])
            if FORMAT == 1:
                data['Location of Estimated Maximum Surface Wind Inbound'] = isNA(
                    line[3:])

        if line[:2] == 'F.':
            info = line[3:]
            if FORMAT >= 2:
                data['Maximum Flight Level Wind Inbound'] = isNA(info)
            if FORMAT == 1:
                data['Eye character'] = isNA(info)

        if line[:2] == 'G.':
            info = line[3:]
            if FORMAT >= 2:
                data['Location of the Maximum Flight Level Wind Inbound'] = isNA(
                    info)
            if FORMAT == 1:
                if isNA(info) == np.nan:
                    data.update(
                        {'Eye Shape': np.nan, 'Eye Diameter (nmi)': np.nan})
                else:
                    shape = ''.join([i for i in info[:2] if not i.isdigit()])
                    size = info[len(shape):]
                    if shape == 'C':
                        if '-' in size:
                            data.update(
                                {'Eye Shape': 'circular', 'Eye Diameter (nmi)': float(size.split('-')[0])})
                        else:
                            data.update({'Eye Shape': 'circular',
                                        'Eye Diameter (nmi)': float(size)})
                    elif shape == 'CO':
                        data['Eye Shape'] = 'concentric'
                        if '-' in size:
                            data.update({f'Eye Diameter {i+1} (nmi)': float(s)
                                        for i, s in enumerate(size.split('-'))})
                        else:
                            data.update({f'Eye Diameter {i+1} (nmi)': float(s)
                                        for i, s in enumerate(size.split(' ')[1:])})
                    elif shape == 'E':
                        einfo = size.split('/')
                        data.update({'Eye Shape': 'elliptical', 'Orientation': float(einfo[0]) * 10,
                                     'Eye Major Axis (nmi)': float(einfo[1]), 'Eye Minor Axis (nmi)': float(einfo[1])})
                    else:
                        data.update(
                            {'Eye Shape': np.nan, 'Eye Diameter (nmi)': np.nan})

        if line[:2] == 'H.':
            info = line[3:].split() * 5
            if FORMAT >= 3:
                info = (line[3:].split("MB")[0]).split()
                if info[0] == '/':
                    data['Minimum Sea Level Pressure (hPa)'] = np.nan
                else:
                    parsed_mslp = info[-1]
                    if '/' in parsed_mslp:
                        parsed_mslp = parsed_mslp.split("/")[1]
                    data['Minimum Sea Level Pressure (hPa)'] = isNA(
                        parsed_mslp)
            elif FORMAT == 2:
                data['Minimum Sea Level Pressure (hPa)'] = isNA(info[-2])
            elif FORMAT == 1:
                data['Estimated Maximum Surface Wind Inbound (kt)'] = isNA(
                    info[0])

        if line[:2] == 'I.':
            info = line[3:]
            if FORMAT >= 2:
                data['Maximum Flight Level Temp Outside Eye (C)'] = isNA(
                    info.split()[0])
            if FORMAT == 1:
                data['Location & Time of the Estimated Maximum Surface Wind Inbound'] = isNA(
                    info)

        if line[:2] == 'J.':
            info = line[3:]
            if FORMAT >= 2:
                data['Maximum Flight Level Temp Inside Eye (C)'] = isNA(
                    info.split()[0])
            if FORMAT == 1:
                data['Maximum Flight Level Wind Inbound (kt)'] = isNA(info)

        if line[:2] == 'K.':
            info = line[3:]
            if FORMAT >= 2:
                data['Dew Point Inside Eye (C)'] = isNA(info.split()[0])
            if FORMAT == 1:
                data['Location & Time of the Maximum Flight Level Wind Inbound'] = isNA(
                    info)

        if line[:2] == 'L.':
            info = line[3:]
            if FORMAT >= 2:
                data['Eye character'] = isNA(info)
            if FORMAT == 1:
                data['Estimated Maximum Surface Wind Outbound (kt)'] = isNA(
                    info)

        if line[:2] == 'M.':
            info = line[3:]
            if FORMAT >= 2:
                if isNA(info) == np.nan:
                    data.update(
                        {'Eye Shape': np.nan, 'Eye Diameter (nmi)': np.nan})
                else:
                    shape = ''.join([i for i in info[:2] if not i.isdigit()])
                    size = info[len(shape):]
                    if shape == 'C':
                        if '-' in size:
                            data.update(
                                {'Eye Shape': 'circular', 'Eye Diameter (nmi)': float(size.split('-')[0])})
                        else:
                            data.update({'Eye Shape': 'circular',
                                        'Eye Diameter (nmi)': float(size)})
                    elif shape == 'CO':
                        data['Eye Shape'] = 'concentric'
                        if '-' in size:
                            data.update({f'Eye Diameter {i+1} (nmi)': float(s)
                                        for i, s in enumerate(size.split('-'))})
                        else:
                            data.update({f'Eye Diameter {i+1} (nmi)': float(s)
                                        for i, s in enumerate(size.split(' ')[1:])})
                    elif shape == 'E':
                        einfo = size.split('/')
                        try:
                            data.update({'Eye Shape': 'elliptical', 'Orientation': float(einfo[0]) * 10,
                                         'Eye Major Axis (nmi)': float(einfo[1]), 'Eye Minor Axis (nmi)': float(einfo[1])})
                        except:
                            data.update({'Eye Shape': 'elliptical', 'Orientation': np.nan,
                                         'Eye Major Axis (nmi)': np.nan, 'Eye Minor Axis (nmi)': np.nan})
                    else:
                        data.update(
                            {'Eye Shape': np.nan, 'Eye Diameter (nmi)': np.nan})
            if FORMAT == 1:
                data['Location & Time of the Estimated Maximum Surface Wind Outbound'] = isNA(
                    info)

        if line[:2] == 'N.':
            info = line[3:]
            if FORMAT == 1:
                data['Maximum Flight Level Wind Outbound (kt)'] = isNA(info)

        if line[:2] == 'O.':
            info = line[3:]
            if FORMAT == 1:
                data['Location & Time of the Maximum Flight Level Wind Outbound'] = isNA(
                    info)

        if line[:2] == 'P.':
            info = line[3:]
            if FORMAT >= 2:
                data['Aircraft'] = info.split()[0]
                missionname = info.split()[1]
                data['mission'] = missionname[:2]
                data['Remarks'] = ''
                RemarksNext = True
            if FORMAT == 1:
                data['Maximum Flight Level Temp & Pressure Altitude Outside Eye'] = isNA(
                    info)

        if line[:2] == 'Q.':
            info = line[3:]
            if FORMAT == 1:
                data['Maximum Flight Level Temp & Pressure Altitude Inside Eye'] = isNA(
                    info)
            if FORMAT == 3:
                data['Aircraft'] = info.split()[0]
                missionname = info.split()[1]
                data['mission'] = missionname[:2]
                data['Remarks'] = ''

        if line[:2] == 'R.':
            info = line[3:]
            if FORMAT == 1:
                data['Dewpoint Temp (collected at same location as temp inside eye)'] = isNA(
                    info)

        if line[:2] == 'S.':
            info = line[3:]
            if FORMAT == 1:
                data['Fix'] = isNA(info)

        if line[:2] == 'T.':
            info = line[3:]
            if FORMAT == 1:
                data['Accuracy'] = isNA(info)

        if line[:2] == 'U.':
            info = line[3:]
            if FORMAT == 1:
                data['Aircraft'] = info.split()[0]
                missionname = info.split()[1]
                data['mission'] = missionname[:2]
                data['Remarks'] = ''
                RemarksNext = True

    content_split = content.split("\n")
    if FORMAT == 4:
        mission_id = '-'.join(content_split[1].replace("  ",
                              " ").split(" ")[:3])
        data['Aircraft'] = mission_id.split("-")[0]
        missionname = mission_id.split("-")[1]
        data['mission'] = missionname[:2]
        data['Remarks'] = ''
    elif FORMAT == 3:
        mission_id = ['-'.join(i.split("Q. ")[1].replace("  ", " ").split(" ")[:3])
                      for i in content_split if i[:2] == "Q."][0]
    elif FORMAT == 2:
        mission_id = ['-'.join(i.split("P. ")[1].replace("  ", " ").split(" ")[:3])
                      for i in content_split if i[:2] == "P."][0]
    elif FORMAT == 1:
        mission_id = ['-'.join(i.split("U. ")[1].replace("  ", " ").split(" ")[:3])
                      for i in content_split if i[:2] == "U."][0]
    data['mission_id'] = mission_id

    return missionname, data

# =======================================================================================================
# Decoding dropsondes
# =======================================================================================================


def decode_dropsonde(content, date):

    NOLOCFLAG = False
    missionname = '_____'

    delimiters = ['XXAA', '31313', '51515',
                  '61616', '62626', 'XXBB', '21212', '_____']
    sections = {}
    for i, d in enumerate(delimiters[:-1]):
        a = content.split('\n' + d)
        if len(a) > 1:
            a = ('\n' + d).join(a[1:]) if len(a) > 2 else a[1]
            b = a.split('\n' + delimiters[i + 1])[0]
            sections[d] = b

    for k, v in sections.items():
        tmp = copy.copy(v)
        for d in delimiters:
            tmp = tmp.split('\n' + d)[0]
        tmp = [i for i in tmp.split(' ') if len(i) > 0]
        tmp = [j.replace('\n', '') if '\n' in j and (len(j) < (
            7 + j.count('\n')) or len(j) == (11 + j.count('\n'))) else j for j in tmp]
        tmp = [i for j in tmp for i in j.split('\n') if len(i) > 0]
        sections[k] = tmp

    def _time(timestr):
        try:
            if timestr < f'{date:%H%M}':
                return date.replace(hour=int(timestr[:2]), minute=int(timestr[2:4]))
            else:
                return date.replace(hour=int(timestr[:2]), minute=int(timestr[2:4])) - timedelta(days=1)
        except:
            return None

    def _tempdwpt(item):
        if '/' in item[:3]:
            temp = np.nan
            dwpt = np.nan
        elif '/' in item[4:]:
            z = round(float(item[:3]), 0)
            temp = round(z * 0.1, 1) if z % 2 == 0 else round(z * -0.1, 1)
            dwpt = np.nan
        else:
            z = round(float(item[:3]), 0)
            temp = round(z * 0.1, 1) if z % 2 == 0 else round(z * -0.1, 1)
            z = round(float(item[3:]), 0)
            dwpt = temp - (round(z * 0.1, 1) if z <= 50 else z - 50)
        return temp, dwpt

    def _wdirwspd(item):
        try:
            wdir = round(
                np.floor(float(item[:3]) / 5) * 5, 0) if '/' not in item else np.nan
            wspd = round(float(item[3:]) + 100 * (float(item[2]) %
                         5), 0) if '/' not in item else np.nan
        except:
            wdir = np.nan
            wspd = np.nan
        return wdir, wspd

    def _standard(I3):
        levkey = I3[0][:2]
        levdict = {'99': -1, '00': 1000, '92': 925, '85': 850, '70': 700, '50': 500,
                   '40': 400, '30': 300, '25': 250, '20': 200, '15': 150, '10': 100, '__': None}
        pres = float(levdict[levkey])
        output = {}
        output['pres'] = pres
        if pres == -1:
            output['pres'] = np.nan if '/' in I3[0][2:] else round(
                float(I3[0][2:]) + [0, 1000][float(I3[0][2:]) < 100], 1)
            output['hgt'] = 0.0
        elif pres == 1000:
            z = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:]), 0)
            output['hgt'] = round(500 - z, 0) if z >= 500 else z
        elif pres == 925:
            output['hgt'] = np.nan if '/' in I3[0][2:
                                                   ] else round(float(I3[0][2:]), 0)
        elif pres == 850:
            output['hgt'] = np.nan if '/' in I3[0][2:
                                                   ] else round(float(I3[0][2:]) + 1000, 0)
        elif pres == 700:
            z = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:]), 0)
            output['hgt'] = round(
                z + 3000, 0) if z < 500 else round(z + 2000, 0)
        elif pres in (500, 400, 300):
            output['hgt'] = np.nan if '/' in I3[0][2:
                                                   ] else round(float(I3[0][2:]) * 10, 0)
        else:
            output['hgt'] = np.nan if '/' in I3[0][2:] else round(
                1e4 + float(I3[0][2:]) * 10, 0)
        output['temp'], output['dwpt'] = _tempdwpt(I3[1])
        if I3[2][:2] == '88':
            output['wdir'], output['wspd'] = np.nan, np.nan
            skipflag = 0
        elif I3[2][:2] == list(levdict.keys())[list(levdict.keys()).index(levkey) + 1] and \
                I3[3][:2] != list(levdict.keys())[list(levdict.keys()).index(levkey) + 1]:
            output['wdir'], output['wspd'] = np.nan, np.nan
            skipflag = 1
        else:
            output['wdir'], output['wspd'] = _wdirwspd(I3[2])
            skipflag = 0
        endflag = True if '88' in [i[:2] for i in I3] else False
        return output, skipflag, endflag

    data = {k: np.nan for k in ('lat', 'lon', 'slp',
                                'TOPlat', 'TOPlon', 'TOPtime',
                                'BOTTOMlat', 'BOTTOMlon', 'BOTTOMtime',
                                'MBLdir', 'MBLspd', 'DLMdir', 'DLMspd',
                                'WL150dir', 'WL150spd', 'top', 'LSThgt', 'software', 'levels')}
    data['TOPtime'] = None
    data['BOTTOMtime'] = None

    for sec, items in sections.items():

        if sec == '61616' and len(items) > 0:
            missionname = items[1]
            data['mission'] = items[1][:2]
            data['stormname'] = items[2]
            try:
                data['obsnum'] = int(items[-1])
            except:
                data['obsnum'] = items[-1]

        if sec == 'XXAA' and len(items) > 0 and not NOLOCFLAG:
            if '/' in items[1] + items[2]:
                NOLOCFLAG = True
            else:
                octant = int(items[2][0])
                data['lat'] = round(float(items[1][2:]) *
                                    0.1 * [-1, 1][octant in (2, 3, 7, 8)], 1)
                data['lon'] = round(float(items[2][1:]) *
                                    0.1 * [-1, 1][octant in (0, 1, 2, 3)], 1)
                data['slp'] = np.nan if '/' in items[4][2:] else round(
                    float(items[4][2:]) + [0, 1000][float(items[4][2:]) < 100], 1)

                standard = {k: []
                            for k in ['pres', 'hgt', 'temp', 'dwpt', 'wdir', 'wspd']}
                skips = 0
                for jj, item in enumerate(items[4::3]):
                    if items[4 + jj * 3 - skips][:2] == '88':
                        break
                    output, skipflag, endflag = _standard(
                        items[4 + jj * 3 - skips:8 + jj * 3 - skips])
                    skips += skipflag
                    for k in standard.keys():
                        standard[k].append(output[k])
                    if endflag:
                        break
                standard = pd.DataFrame.from_dict(
                    standard).sort_values('pres', ascending=False)

        if sec == '62626' and len(items) > 0 and not NOLOCFLAG:
            if items[0] in ['CENTER', 'MXWNDBND', 'RAINBAND', 'EYEWALL']:
                data['location'] = items[0]
                if items[0] == 'EYEWALL':
                    data['octant'] = {'000': 'N', '045': 'NE', '090': 'E', '135': 'SE',
                                      '180': 'S', '225': 'SW', '270': 'W', '315': 'NW'}[items[1]]
            if 'REL' in items:
                tmp = items[items.index('REL') + 1]
                data['TOPlat'] = round(
                    float(tmp[:4]) * .01 * [-1, 1][tmp[4] == 'N'], 2)
                data['TOPlon'] = round(
                    float(tmp[5:10]) * .01 * [-1, 1][tmp[10] == 'E'], 2)
                tmp = items[items.index('REL') + 2]
                if data['TOPtime'] is None:
                    data['TOPtime'] = _time(tmp)
            if 'SPG' in items:
                tmp = items[items.index('SPG') + 1]
                data['BOTTOMlat'] = round(
                    float(tmp[:4]) * .01 * [-1, 1][tmp[4] == 'N'], 2)
                data['BOTTOMlon'] = round(
                    float(tmp[5:10]) * .01 * [-1, 1][tmp[10] == 'E'], 2)
                tmp = items[items.index('SPG') + 2]
                if data['BOTTOMtime'] is None:
                    data['BOTTOMtime'] = _time(tmp)
            elif 'SPL' in items:
                tmp = items[items.index('SPL') + 1]
                data['BOTTOMlat'] = round(
                    float(tmp[:4]) * .01 * [-1, 1][tmp[4] == 'N'], 2)
                data['BOTTOMlon'] = round(
                    float(tmp[5:10]) * .01 * [-1, 1][tmp[10] == 'E'], 2)
                tmp = items[items.index('SPL') + 2]
                if data['BOTTOMtime'] is None:
                    data['BOTTOMtime'] = _time(tmp)
            if 'MBL' in items:
                tmp = items[items.index('MBL') + 2]
                wdir, wspd = _wdirwspd(tmp)
                data['MBLdir'] = wdir
                data['MBLspd'] = wspd
            if 'DLM' in items:
                tmp = items[items.index('DLM') + 2]
                wdir, wspd = _wdirwspd(tmp)
                data['DLMdir'] = wdir
                data['DLMspd'] = wspd
            if 'WL150' in items:
                tmp = items[items.index('WL150') + 1]
                wdir, wspd = _wdirwspd(tmp)
                data['WL150dir'] = wdir
                data['WL150spd'] = wspd
            if 'LST' in items:
                tmp = items[items.index('LST') + 2]
                tmp = tmp.replace('=', '')
                data['LSThgt'] = round(float(tmp), 0)
            if 'AEV' in items:
                tmp = items[items.index('AEV') + 1]
                data['software'] = 'AEV ' + tmp

        if sec == 'XXBB' and len(items) > 0 and not NOLOCFLAG:
            sigtemp = {k: [] for k in ['pres', 'temp', 'dwpt']}
            for jj, item in enumerate(items[6::2]):
                z = np.nan if '/' in items[6 + jj *
                                           2][2:] else round(float(items[6 + jj * 2][2:]), 0)
                sigtemp['pres'].append(round(z + 1000, 0) if z < 100 else z)
                temp, dwpt = _tempdwpt(items[7 + jj * 2])
                sigtemp['temp'].append(temp)
                sigtemp['dwpt'].append(dwpt)
            sigtemp = pd.DataFrame.from_dict(
                sigtemp).sort_values('pres', ascending=False)

        if sec == '21212' and len(items) > 0 and not NOLOCFLAG:
            sigwind = {k: [] for k in ['pres', 'wdir', 'wspd']}
            for jj, item in enumerate(items[2::2]):
                z = np.nan if '/' in items[2 + jj *
                                           2][2:] else round(float(items[2 + jj * 2][2:]), 0)
                sigwind['pres'].append(round(z + 1000, 0) if z < 100 else z)
                wdir, wspd = _wdirwspd(items[3 + jj * 2])
                sigwind['wdir'].append(wdir)
                sigwind['wspd'].append(wspd)
            sigwind = pd.DataFrame.from_dict(
                sigwind).sort_values('pres', ascending=False)

        if sec == '31313' and len(items) > 0 and not NOLOCFLAG:
            tmp = [i for i in items if i[0] == '8'][0]
            data['TOPtime'] = _time(tmp[1:])

    if not NOLOCFLAG:
        def _justify(a, axis=0):
            mask = pd.notnull(a)
            arg_justified = np.argsort(mask, axis=0)[-1]
            anew = [col[i] for i, col in zip(arg_justified, a.T)]
            return anew
        df = pd.concat([standard, sigtemp, sigwind], ignore_index=True,
                       sort=False).sort_values('pres', ascending=False)
        data['levels'] = pd.DataFrame(np.vstack(df.groupby('pres', sort=False)
                                                .apply(lambda gp: _justify(gp.to_numpy()))), columns=df.columns)

        data['top'] = np.nanmin(data['levels']['pres'])

    content_split = content.split("\n")
    try:
        mission_id = ['-'.join(i.split("61616 ")[1].replace("  ", " ").split(" ")[:3])
                      for i in content_split if i[:5] == "61616"][0]
    except:
        mission_id = '-'.join(content.split("\n")
                              [1].replace("  ", " ").split(" ")[:3])
    data['mission_id'] = mission_id

    # Fix NaNs
    if data['BOTTOMtime'] is None:
        data['BOTTOMtime'] = np.nan
    if data['TOPtime'] is None:
        data['TOPtime'] = np.nan

    return missionname, data


def get_status(plane_p, use_z=False):

    status = []
    in_storm = False
    finished = False
    for idx, pres in enumerate(plane_p):
        if idx < 8:
            status.append('En Route')
            continue
        if np.isnan(pres):
            status.append(status[-1])
            continue

        # Use default pressure method
        if not use_z:
            if np.nanmin(plane_p[:idx + 1]) >= 850:
                status.append('En Route')
                continue
            if np.abs(plane_p[idx] - plane_p[idx - 8]) < 10:
                if finished:
                    if idx > 40 and pres < 800 and pres > 650 and np.nanmax(np.abs(plane_p[idx - 40:idx] - plane_p[idx - 39:idx + 1])) < 20:
                        if idx > 100 and 'In Storm' in status[-100:]:
                            finished = False
                        else:
                            status.append('Finished')
                    else:
                        status.append('Finished')
                if not finished:
                    if pres < 650:
                        status.append('En Route')
                    else:
                        in_storm = True
                        status.append('In Storm')
            else:
                if not in_storm:
                    status.append('En Route')
                else:
                    if pres < 650:
                        finished = True
                    if finished:
                        status.append('Finished')
                    else:
                        status.append('In Storm')

        # Use height method
        else:
            if np.nanmax(plane_p[:idx + 1]) <= 1515:
                status.append('En Route')
                continue
            if np.abs(plane_p[idx] - plane_p[idx - 8]) < 60:
                if finished:
                    if idx > 40 and pres > 2050 and pres < 3820 and np.nanmin(np.abs(plane_p[idx - 40:idx] - plane_p[idx - 39:idx + 1])) < 120:
                        if idx > 100 and 'In Storm' in status[-100:]:
                            finished = False
                        else:
                            status.append('Finished')
                    else:
                        status.append('Finished')
                if not finished:
                    if pres > 3820:
                        status.append('En Route')
                    else:
                        in_storm = True
                        status.append('In Storm')
            else:
                if not in_storm:
                    status.append('En Route')
                else:
                    if pres > 3820:
                        finished = True
                    if finished:
                        status.append('Finished')
                    else:
                        status.append('In Storm')

    return status
