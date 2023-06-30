import os
import numpy as np
from datetime import datetime as dt, timedelta
import requests
import re
import urllib
import matplotlib.dates as mdates
import scipy.interpolate as interp
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .. import constants
from ..utils import create_storm_dict

def find_latest_hurdat_files():
    r"""
    Identifies latest available HURDATv2 files from the NHC web server for the Atlantic and Pacific basins.

    Returns
    -------
    list
        List containing URL strings for the latest available Best Track file for the Atlantic & Pacific basins.
    """

    # Check if NHC website is up. If not, return HRD HURDATv2 URL
    url_found = True
    try:
        check = urllib.request.urlopen(
            "https://www.nhc.noaa.gov/data/hurdat/").getcode()
        if check != 200:
            url_found = False
    except:
        url_found = False
    if not url_found:
        atlantic_url = 'https://www.aoml.noaa.gov/hrd/hurdat/hurdat2.html'
        pacific_url = 'https://www.aoml.noaa.gov/hrd/hurdat/hurdat2-nepac.html'
        return atlantic_url, pacific_url

    # Store data for iteration
    atlantic_url = ''
    pacific_url = ''
    url_data = {
        'url': {
            'atl': [],
            'pac': [],
        },
        'date': {
            'atl': [],
            'pac': [],
        }
    }

    # Iterate over all HURDATv2 files available on NHC's website
    nhc_directory = 'https://www.nhc.noaa.gov/data/hurdat/'
    page = requests.get(nhc_directory).text
    content = page.split("\n")
    for line in content:
        if ".txt" in line:
            fname = (line.split('href="')[1]).split('">')[0]

            # Identify Atlantic vs. Pacific
            if 'nepac' in fname:
                file_basin = 'pac'
                fname_split = '-'.join(fname.split('-')[4:])
            else:
                file_basin = 'atl'
                fname_split = '-'.join(fname.split('-')[3:])

            # Try to identify date file was added
            try:
                strdate = fname_split.split(".txt")[0]
                strdate = re.sub("[^0-9]", "", strdate)
                if len(strdate) == 6:
                    strdate = f'{strdate[0:4]}20{strdate[4:]}'
                date_obj = dt.strptime(strdate, '%m%d%Y')

                if file_basin == 'pac':
                    url_data['url']['pac'].append(f'{nhc_directory}{fname}')
                    url_data['date']['pac'].append(date_obj)
                else:
                    url_data['url']['atl'].append(f'{nhc_directory}{fname}')
                    url_data['date']['atl'].append(date_obj)
            except:
                continue

    min_date_atl = max(url_data['date']['atl'])
    atlantic_url = url_data['url']['atl'][url_data['date']
                                          ['atl'].index(min_date_atl)]
    min_date_pac = max(url_data['date']['pac'])
    pacific_url = url_data['url']['pac'][url_data['date']
                                         ['pac'].index(min_date_pac)]

    return atlantic_url, pacific_url


def find_var(request, thresh):
    r"""
    Given a request and threshold, returns the variable for plotting. Referenced from ``TrackDataset.gridded_stats()`` and ``TrackPlot.plot_gridded()``. Internal function.

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
        return thresh, 'type'

    if request.find('date') >= 0 or request.find('day') >= 0:
        return thresh, 'date'

    # Sustained wind, or change in wind speed
    if request.find('wind') >= 0 or request.find('vmax') >= 0:
        # If change in wind, determine time interval
        if request.find('change') >= 0:
            try:
                thresh['dt_window'] = int(''.join([c for i, c in enumerate(request)
                                                   if c.isdigit() and i > request.find('hour') - 4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh, 'dvmax_dt'
        # Otherwise, sustained wind
        else:
            return thresh, 'vmax'

    elif request.find('ace') >= 0:
        return thresh, 'ace'
    elif request.find('acie') >= 0:
        return thresh, 'acie'

    # Minimum MSLP, or change in MSLP
    elif request.find('pressure') >= 0 or request.find('slp') >= 0:
        # If change in MSLP, determine time interval
        if request.find('change') >= 0:
            try:
                thresh['dt_window'] = int(''.join([c for i, c in enumerate(request)
                                                   if c.isdigit() and i > request.find('hour') - 4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh, 'dmslp_dt'
        # Otherwise, minimum MSLP
        else:
            return thresh, 'mslp'

    # Storm motion or heading (vector)
    elif request.find('heading') >= 0 or request.find('motion') >= 0:
        return thresh, ('dx_dt', 'dy_dt')

    elif request.find('movement') >= 0 or request.find('speed') >= 0:
        return thresh, 'speed'

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


def construct_title(thresh):
    r"""
    Construct a plot title for ``TrackDataset.gridded_stats()``. Internal function.

    Parameters
    ----------
    thresh : dict
        Dictionary containing thresholds for the plot. Detailed more in the ``TrackDataset.gridded_dataset()`` function.

    Returns
    -------
    thresh : dict
        Returns the thresh dictionary, modified depending on the threshold(s) specified.
    plot_subtitle : str
        String denoting the title for the plot.
    """

    # List containing entry for plot title, later merged into a string
    plot_subtitle = []

    # Symbols for greater/less than or equal to signs
    gteq = u"\u2265"
    lteq = u"\u2264"

    # Add sample minimum
    if not np.isnan(thresh['sample_min']):
        plot_subtitle.append(f"{gteq} {thresh['sample_min']} storms/bin")
    else:
        thresh['sample_min'] = 0

    # Add minimum wind speed
    if not np.isnan(thresh['v_min']):
        plot_subtitle.append(f"{gteq} {thresh['v_min']}kt")
    else:
        thresh['v_min'] = 0

    # Add maximum MSLP
    if not np.isnan(thresh['p_max']):
        plot_subtitle.append(f"{lteq} {thresh['p_max']}hPa")
    else:
        thresh['p_max'] = 9999

    # Add minimum change in wind speed
    if not np.isnan(thresh['dv_min']):
        plot_subtitle.append(
            f"{gteq} {thresh['dv_min']}kt / {thresh['dt_window']}hr")
    else:
        thresh['dv_min'] = -9999

    # Add maximum change in MSLP
    if not np.isnan(thresh['dp_max']):
        plot_subtitle.append(
            f"{lteq} {thresh['dp_max']}hPa / {thresh['dt_window']}hr")
    else:
        thresh['dp_max'] = 9999

    # Add maximum change in wind speed
    if not np.isnan(thresh['dv_max']):
        plot_subtitle.append(
            f"{lteq} {thresh['dv_max']}kt / {thresh['dt_window']}hr")
    else:
        thresh['dv_max'] = 9999

    # Add minimum change in MSLP
    if not np.isnan(thresh['dp_min']):
        plot_subtitle.append(
            f"{gteq} {thresh['dp_min']}hPa / {thresh['dt_window']}hr")
    else:
        thresh['dp_min'] = -9999

    # Combine plot_subtitle into string
    if len(plot_subtitle) > 0:
        plot_subtitle = '\n' + ', '.join(plot_subtitle)
    else:
        plot_subtitle = ''

    # Return modified thresh and plot title
    return thresh, plot_subtitle


def interp_storm(storm_dict, hours=1, dt_window=24, dt_align='middle', method='linear'):
    r"""
    Interpolate a storm dictionary temporally to a specified time resolution. Referenced from ``TrackDataset.filter_storms()``. Internal function.

    Parameters
    ----------
    storm_dict : dict
        Dictionary containing a storm entry.
    hours : int
        Temporal resolution in hours to interpolate storm data to. Default is 1 hour.
    dt_window : int
        Time window in hours over which to calculate temporal change data. Default is 24 hours.
    dt_align : str
        Whether to align the temporal change window as "start", "middle" or "end" of the dt_window time period.
    method : str
        Method by which to interpolate lat & lon coordinates. Options are "linear" (default) or "quadratic".

    Returns
    -------
    dict
        Dictionary containing the updated storm entry.
    """

    # Create an empty dict for the new storm entry
    new_storm = {}

    # Copy over non-list attributes
    for key in storm_dict.keys():
        if not isinstance(storm_dict[key], list):
            new_storm[key] = storm_dict[key]

    # Create an empty list for entries
    for name in ['time', 'vmax', 'mslp', 'lat', 'lon', 'type']:
        new_storm[name] = []

    # Convert times to numbers for ease of calculation
    times = mdates.date2num(storm_dict['time'])

    # Convert lat & lons to arrays, and ensure lons are out of 360 degrees
    storm_dict['type'] = np.asarray(storm_dict['type'])
    storm_dict['lon'] = np.array(storm_dict['lon']) % 360

    def round_datetime(tm, nearest_minute=10):
        discard = timedelta(minutes=tm.minute % nearest_minute,
                            seconds=tm.second,
                            microseconds=tm.microsecond)
        tm -= discard
        if discard >= timedelta(minutes=int(nearest_minute / 2)):
            tm += timedelta(minutes=nearest_minute)
        return tm

    # Attempt temporal interpolation
    try:

        # Create a list of target times given the requested temporal resolution
        targettimes = np.arange(
            times[0], times[-1] + hours / 24.0, hours / 24.0)
        targettimes = targettimes[targettimes <= times[-1] + 0.001]

        # Update times
        use_minutes = 10 if hours > (1.0 / 6.0) else hours * 60.0
        new_storm['time'] = [round_datetime(
            t.replace(tzinfo=None), use_minutes) for t in mdates.num2date(targettimes)]
        targettimes = mdates.date2num(np.array(new_storm['time']))

        # Create same-length lists for other things
        new_storm['special'] = [''] * len(new_storm['time'])
        new_storm['extra_obs'] = [0] * len(new_storm['time'])

        # WMO basin. Simple linear interpolation.
        basinnum = np.cumsum([0] + [1 if storm_dict['wmo_basin'][i + 1] != j else 0
                                    for i, j in enumerate(storm_dict['wmo_basin'][:-1])])
        basindict = {k: v for k, v in zip(basinnum, storm_dict['wmo_basin'])}
        basininterp = np.round(
            np.interp(targettimes, times, basinnum)).astype(int)
        new_storm['wmo_basin'] = [basindict[k] for k in basininterp]

        # Interpolate and fill in storm type
        stormtype = [1 if i in constants.TROPICAL_STORM_TYPES else -
                     1 for i in storm_dict['type']]
        isTROP = np.interp(targettimes, times, stormtype)
        stormtype = [
            1 if i in constants.SUBTROPICAL_ONLY_STORM_TYPES else -1 for i in storm_dict['type']]
        isSUB = np.interp(targettimes, times, stormtype)
        stormtype = [1 if i == 'LO' else -1 for i in storm_dict['type']]
        isLO = np.interp(targettimes, times, stormtype)
        stormtype = [1 if i == 'DB' else -1 for i in storm_dict['type']]
        isDB = np.interp(targettimes, times, stormtype)
        newtype = np.where(isTROP > 0, 'TROP', 'EX')
        newtype[newtype == 'TROP'] = np.where(
            isSUB[newtype == 'TROP'] > 0, 'SUB', 'TROP')
        newtype[newtype == 'EX'] = np.where(
            isLO[newtype == 'EX'] > 0, 'LO', 'EX')
        newtype[newtype == 'EX'] = np.where(
            isDB[newtype == 'EX'] > 0, 'DB', 'EX')

        # Interpolate and fill in other variables
        for name in ['vmax', 'mslp']:
            new_storm[name] = np.interp(targettimes, times, storm_dict[name])
            new_storm[name] = np.array([int(round(i)) if not np.isnan(i) else np.nan for i in new_storm[name]])
        for name in ['lat', 'lon']:
            filtered_array = np.array(storm_dict[name])
            new_times = np.array(storm_dict['time'])
            if 'linear' not in method:
                converted_hours = np.array([1 if i.strftime(
                    '%H%M') in constants.STANDARD_HOURS else 0 for i in storm_dict['time']])
                filtered_array = filtered_array[converted_hours == 1]
                new_times = new_times[converted_hours == 1]
            new_times = mdates.date2num(new_times)
            if len(filtered_array) >= 3:
                func = interp.interp1d(new_times, filtered_array, kind=method)
                new_storm[name] = func(targettimes)
                new_storm[name] = np.array([round(i, 2) if not np.isnan(i) else np.nan for i in new_storm[name]])
            else:
                new_storm[name] = np.interp(
                    targettimes, times, storm_dict[name])
                new_storm[name] = np.array([int(round(i)) if not np.isnan(i) else np.nan for i in new_storm[name]])

        # Correct storm type by intensity
        newtype[newtype == 'TROP'] = [['TD', 'TS', 'HU', 'TY', 'ST'][int(
            i > 34) + int(i > 63)] for i in new_storm['vmax'][newtype == 'TROP']]
        newtype[newtype == 'SUB'] = [['SD', 'SS']
                                     [int(i > 34)] for i in new_storm['vmax'][newtype == 'SUB']]
        new_storm['type'] = newtype

        # Calculate change in wind & MSLP over temporal resolution
        new_storm['dvmax_dt'] = [np.nan] + \
            list((new_storm['vmax'][1:] - new_storm['vmax'][:-1]) / hours)
        new_storm['dmslp_dt'] = [np.nan] + \
            list((new_storm['mslp'][1:] - new_storm['mslp'][:-1]) / hours)

        # Calculate x and y position change over temporal window
        rE = 6.371e3 * 0.539957  # nautical miles
        d2r = np.pi / 180.
        new_storm['dx_dt'] = [np.nan] + list(d2r * (new_storm['lon'][1:] - new_storm['lon'][:-1]) *
                                             rE * np.cos(d2r * np.mean([new_storm['lat'][1:], new_storm['lat'][:-1]], axis=0)) / hours)
        new_storm['dy_dt'] = [np.nan] + list(d2r * (new_storm['lat'][1:] - new_storm['lat'][:-1]) *
                                             rE / hours)
        new_storm['speed'] = [(x**2 + y**2)**0.5 for x,
                              y in zip(new_storm['dx_dt'], new_storm['dy_dt'])]

        # Convert change in wind & MSLP to change over specified window
        for name in ['dvmax_dt', 'dmslp_dt']:
            tmp = np.round(np.convolve(new_storm[name], [
                           1] * int(dt_window / hours), mode='valid'), 1)
            if dt_align == 'end':
                new_storm[name] = [np.nan] * \
                    (len(new_storm[name]) - len(tmp)) + list(tmp)
            if dt_align == 'middle':
                tmp2 = [np.nan] * \
                    int((len(new_storm[name]) - len(tmp)) // 2) + list(tmp)
                new_storm[name] = tmp2 + [np.nan] * \
                    (len(new_storm[name]) - len(tmp2))
            if dt_align == 'start':
                new_storm[name] = list(tmp) + [np.nan] * \
                    (len(new_storm[name]) - len(tmp))
            new_storm[name] = list(np.array(new_storm[name]) * (hours))

        # Convert change in position to change over specified window
        for name in ['dx_dt', 'dy_dt', 'speed']:
            tmp = np.convolve(new_storm[name], [
                              hours / dt_window] * int(dt_window / hours), mode='valid')
            if dt_align == 'end':
                new_storm[name] = [np.nan] * \
                    (len(new_storm[name]) - len(tmp)) + list(tmp)
            if dt_align == 'middle':
                tmp2 = [np.nan] * \
                    int((len(new_storm[name]) - len(tmp)) // 2) + list(tmp)
                new_storm[name] = tmp2 + [np.nan] * \
                    (len(new_storm[name]) - len(tmp2))
            if dt_align == 'start':
                new_storm[name] = list(tmp) + [np.nan] * \
                    (len(new_storm[name]) - len(tmp))

        new_storm['dt_window'] = dt_window
        new_storm['dt_align'] = dt_align

        # Return new dict
        return new_storm

    # Otherwise, simply return NaNs
    except:
        for name in new_storm.keys():
            try:
                storm_dict[name]
            except:
                storm_dict[name] = np.ones(len(new_storm[name])) * np.nan
        return storm_dict


def filter_storms_vp(trackdata, year_min=0, year_max=9999, subset_domain=None):
    r"""
    Calculate a wind-pressure relationship. Referenced from ``TrackDataset.wind_pres_relationship()``. Internal function.

    Parameters
    ----------
    trackdata : tropycal.tracks.TrackDataset
        TrackDataset object.
    year_min : int
        Starting year of analysis.
    year_max : int
        Ending year of analysis.
    subset_domain : str
        String representing either a bounded region 'latW/latE/latS/latN', or a basin name.

    Returns
    -------
    list
        List representing pressure-wind relationship.
    """

    # If no subset domain is passed, use global data
    if subset_domain is None:
        lon_min, lon_max, lat_min, lat_max = [0, 360, -90, 90]
    else:
        lon_min, lon_max, lat_min, lat_max = [
            float(i) for i in subset_domain.split("/")]

    # Empty list for v-p relationship data
    vp = []

    # Iterate over every storm in dataset
    for key in trackdata.keys:

        # Retrieve storm dictionary
        istorm = trackdata.data[key]

        # Iterate over every storm time step
        for i, (iwind, imslp, itype, ilat, ilon, itime) in \
                enumerate(zip(istorm['vmax'], istorm['mslp'], istorm['type'], istorm['lat'], istorm['lon'], istorm['time'])):

            # Ensure both have data and are while the cyclone is tropical
            if np.nan not in [iwind, imslp] and itype in constants.TROPICAL_STORM_TYPES \
                and lat_min <= ilat <= lat_max and lon_min <= ilon % 360 <= lon_max \
                    and year_min <= itime.year <= year_max:
                vp.append([imslp, iwind])

    # Return v-p relationship list
    return vp


def testfit(data, x, order):
    r"""
    Calculate a line of best fit for wind-pressure relationship. Referenced from ``TrackDataset.wind_pres_relationship()``. Internal function.

    Parameters
    ----------
    data : list
        List of tuples representing wind-pressure relationship. Obtained from ``filter_storms_vp()``.
    x : float
        x value corresponding to the maximum sustained wind.
    order : int
        Function order to pass to ``np.polyfit()``.

    Returns
    -------
    float
        y value corresponding to the polyfit function for the given x value.
    """

    # Make sure there are enough samples
    if len(data) > 50:
        f = np.polyfit([i[1] for i in data], [i[0] for i in data], order)
        y = sum([f[i] * x**(order - i) for i in range(order + 1)])
        return y
    else:
        return np.nan


def rolling_window(a, window):
    r"""
    Calculate a rolling window of an array given a window. Referenced from ``TrackDataset.ace_climo()`` and ``TrackDataset.hurricane_days_climo()``. Internal function.

    Parameters
    ----------
    a : numpy.ndarray
        1D array containing data.
    window : int
        Window over which to compute a rolling window.

    Returns
    -------
    numpy.ndarray
        Array containing the rolling window.
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def convert_to_julian(time):
    r"""
    Convert a datetime object to Julian days. Referenced from ``TrackDataset.ace_climo()`` and ``TrackDataset.hurricane_days_climo()``. Internal function.

    Parameters
    ----------
    time : datetime.datetime
        Datetime object of the time to be converted to Julian days.

    Returns
    -------
    int
        Integer representing the Julian day of the requested time.
    """

    year = time.year
    return ((time - dt(year, 1, 1, 0)).days + (time - dt(year, 1, 1, 0)).seconds / 86400.0) + 1


def months_in_julian(year):
    r"""
    Format months in Julian days for plotting time series. Referenced from ``TrackDataset.ace_climo()`` and ``TrackDataset.hurricane_days_climo()``. Internal function.

    Parameters
    ----------
    year : int
        Year for which to determine Julian month days.

    Returns
    -------
    dict
        Dictionary containing data for constructing time series.
    """

    # Get number of days in year
    length_of_year = convert_to_julian(dt(year, 12, 31, 0)) + 1.0

    # Construct a list of months and names
    months = range(1, 13, 1)
    months_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                    'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months_dates = [dt(year, i, 1, 0) for i in months]

    # Get midpoint x-axis location of month
    months_julian = [int(convert_to_julian(i)) for i in months_dates]
    midpoint_julian = (np.array(months_julian) +
                       np.array(months_julian[1:] + [length_of_year])) / 2.0
    return {'start': months_julian, 'midpoint': midpoint_julian.tolist(), 'name': months_names}


def num_to_str2(number):
    r"""
    Convert an integer to a 2-character string. Internal function.

    Parameters
    ----------
    number : int
        Integer to be converted to a string.

    Returns
    -------
    str
        Two character string.
    """

    # If number is less than 10, add a leading zero out front
    if number < 10:
        return f'0{number}'

    # Otherwise, simply convert to a string
    return str(number)


def plot_credit():
    return "Plot generated using troPYcal"


def add_credit(ax, text):
    import matplotlib.patheffects as path_effects
    a = ax.text(0.99, 0.01, text, fontsize=9, color='k', alpha=0.7, fontweight='bold',
                transform=ax.transAxes, ha='right', va='bottom', zorder=10)
    a.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                        path_effects.Normal()])


def pac_2006_cyclone():
    """
    Data for 2006 Central Pacific cyclone obtained from a simple MSLP minimum based tracker applied to the ERA-5 reanalysis dataset. Sustained wind values from the duration of the storm's subtropical and tropical stages were obtained from an estimate from Dr. Karl Hoarau of the Cergy-Pontoise University in Paris:

    https://australiasevereweather.com/cyclones/2007/trak0611.htm
    """
    
    #Find storm path
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, 'pacific_2006.csv')

    #Create storm dict
    storm_dict = create_storm_dict(
        filepath,
        storm_name = 'UNNAMED',
        storm_id = 'CP052006',
    )
    storm_dict['operational_id'] = ''
    storm_dict['source'] = 'hurdat'
    storm_dict['source_info'] = 'ERA5 Reanalysis & Dr. Karl Hoarau reanalysis'

    # Replace original entry with this
    return storm_dict


def cyclone_catarina():
    """
    https://journals.ametsoc.org/doi/pdf/10.1175/MWR3330.1
    """
    
    #Find storm path
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, 'catarina.csv')
    
    #Create storm dict
    storm_dict = create_storm_dict(
        filepath,
        storm_name = 'CATARINA',
        storm_id = 'AL502004',
    )
    storm_dict['operational_id'] = ''
    storm_dict['source'] = 'ibtracs'
    storm_dict['source_info'] = 'McTaggart-Cowan et al. (2006): https://doi.org/10.1175/MWR3330.1'

    # Replace original entry with this
    return storm_dict


def num_to_text(number):
    r"""
    Retrieve a text representation of a number less than 100. Internal function.

    Parameters
    ----------
    number : int
        Integer to be converted to a string.

    Returns
    -------
    str
        Text representing the number.
    """

    # Dictionary mapping numbers to string representations
    d = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
         6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
         11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
         15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
         19: 'nineteen', 20: 'twenty',
         30: 'thirty', 40: 'forty', 50: 'fifty', 60: 'sixty',
         70: 'seventy', 80: 'eighty', 90: 'ninety'}

    # If number is less than 20, return string
    if number < 20:
        return d[number]

    # Otherwise, form number from combination of strings
    elif number < 100:
        if number % 10 == 0:
            return d[number]
        else:
            return d[number // 10 * 10] + '-' + d[number % 10]

    # If larger than 100, raise error
    else:
        msg = "Please choose a number less than 100."
        raise ValueError(msg)


def listify(x):
    if isinstance(x, (tuple, list, np.ndarray)):
        return [i for i in x]
    else:
        return [x]


def make_var_label(x, storm_dict):
    delta = u"\u0394"
    x = list(x)
    if x[0] == 'd' and x[-3:-1] == ['_', 'd']:
        x[0] = delta
        del x[-3:]
        x.append(f' / {storm_dict["dt_window"]}hr')
    x = ''.join(x)
    if 'mslp' in x:
        x = x.replace('mslp', 'mslp (hPa)')
    if 'vmax' in x:
        x = x.replace('vmax', 'vmax (kt)')
    if 'speed' in x:
        x = x.replace('speed', 'speed (kt)')
    return ''.join(x)


def date_diff(a, b):
    if isinstance(a, np.datetime64):
        a = dt.utcfromtimestamp(
            (a - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    if isinstance(b, np.datetime64):
        b = dt.utcfromtimestamp(
            (b - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    c = a.replace(year=2000) - b.replace(year=2000)
    if c < timedelta(0):
        try:
            c = a.replace(year=2001) - b.replace(year=2000)
        except:
            c = a.replace(year=2000) - b.replace(year=1999)
    return c


def add_colorbar(mappable=None, location='right', size="2.5%", pad='1%', fig=None, ax=None, **kwargs):
    """
    Uses the axes_grid toolkit to add a colorbar to the parent axis and rescale its size to match
    that of the parent axis. This is adapted from Basemap's original ``colorbar()`` method.

    Parameters
    ----------
    mappable
        The image mappable to which the colorbar applies. If none specified, matplotlib.pyplot.gci() is
        used to retrieve the latest mappable.
    location
        Location in which to place the colorbar ('right','left','top','bottom'). Default is right.
    size
        Size of the colorbar. Default is 3%.
    pad
        Pad of colorbar from axis. Default is 1%.
    ax
        Axes instance to associated the colorbar with. If none provided, or if no
        axis is associated with the instance of Map, then plt.gca() is used.
    """

    # Get current mappable if none is specified
    if fig is None or mappable is None:
        import matplotlib.pyplot as plt
    if fig is None:
        fig = plt.gcf()

    if mappable is None:
        mappable = plt.gci()

    # Create axis to insert colorbar in
    divider = make_axes_locatable(ax)

    if location == "left":
        orientation = 'vertical'
        ax_cb = divider.new_horizontal(
            size, pad, pack_start=True, axes_class=plt.Axes)
    elif location == "right":
        orientation = 'vertical'
        ax_cb = divider.new_horizontal(
            size, pad, pack_start=False, axes_class=plt.Axes)
    elif location == "bottom":
        orientation = 'horizontal'
        ax_cb = divider.new_vertical(
            size, pad, pack_start=True, axes_class=plt.Axes)
    elif location == "top":
        orientation = 'horizontal'
        ax_cb = divider.new_vertical(
            size, pad, pack_start=False, axes_class=plt.Axes)
    else:
        raise ValueError('Improper location entered')

    # Create colorbar
    fig.add_axes(ax_cb)
    cb = plt.colorbar(mappable, orientation=orientation, cax=ax_cb, **kwargs)

    # Reset parent axis as the current axis
    fig.sca(ax)
    return cb
