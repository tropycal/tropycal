import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
import requests
import urllib
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib as mlib
import warnings

def find_var(request,thresh):
    
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
    
    #Convert command to lowercase
    request = request.lower()
    
    #Count of number of storms
    if request.find('count') >= 0 or request.find('num') >= 0:
        return thresh, 'date' #not sure what date stands for
    
    #Sustained wind, or change in wind speed
    if request.find('wind') >= 0 or request.find('vmax') >= 0:
        #If change in wind, determine time interval
        if request.find('change') >= 0:
            try:
                thresh['dt_window'] = int(''.join([c for i,c in enumerate(request) \
                      if c.isdigit() and i > request.find('hour')-4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh,'dvmax_dt'
        #Otherwise, sustained wind
        else:
            return thresh,'vmax'
    
    #Minimum MSLP, or change in MSLP
    elif request.find('pressure') >= 0 or request.find('slp') >= 0:
        #If change in MSLP, determine time interval
        if request.find('change') >= 0:
            try:
                thresh['dt_window'] = int(''.join([c for i,c in enumerate(request) \
                      if c.isdigit() and i > request.find('hour')-4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh,'dmslp_dt'
        #Otherwise, minimum MSLP
        else:
            return thresh,'mslp'
    
    #Storm motion or heading (vector)
    elif request.find('heading') >= 0 or request.find('movement') >= 0 or request.find('motion') >= 0:
        return thresh,('dx_dt','dy_dt')
    
    #Otherwise, error
    else:
        msg = "Error: Could not decipher variable. Please refer to documentation for examples on how to phrase the \"request\" string."
        raise RuntimeError(msg)
        
def find_func(request,thresh):
    
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
    
    #Convert command to lowercase
    request = request.lower()
    
    #Numpy maximum function
    if request.find('max') == 0:
        return thresh, lambda x: np.nanmax(x)
    
    #Numpy minimum function
    if request.find('min') == 0:
        return thresh, lambda x: np.nanmin(x)
    
    #Numpy average function
    elif request.find('mean') >= 0 or request.find('average') >= 0 or request.find('avg') >= 0:
        thresh['sample_min'] = max([5,thresh['sample_min']]) #Ensure sample minimum is at least 5 per gridpoint
        return thresh, lambda x: np.nanmean(x)
    
    #Numpy percentile function
    elif request.find('percentile') >= 0:
        ptile = int(''.join([c for i,c in enumerate(request) if c.isdigit() and i < request.find('percentile')]))
        thresh['sample_min'] = max([5,thresh['sample_min']]) #Ensure sample minimum is at least 5 per gridpoint
        return thresh, lambda x: np.nanpercentile(x,ptile)
    
    #Count function
    elif request.find('count') >= 0 or request.find('num') >= 0:
        return thresh, lambda x: len(x)
    
    #Otherwise, function cannot be identified
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
    
    #List containing entry for plot title, later merged into a string
    plot_subtitle = []
    
    #Symbols for greater/less than or equal to signs
    gteq = u"\u2265"
    lteq = u"\u2264"
    
    #Add sample minimum
    if not np.isnan(thresh['sample_min']):
        plot_subtitle.append(f"{gteq} {thresh['sample_min']} storms/bin")
    else:
        thresh['sample_min'] = 0
    
    #Add minimum wind speed
    if not np.isnan(thresh['v_min']):
        plot_subtitle.append(f"{gteq} {thresh['v_min']}kt")
    else:
        thresh['v_min'] = 0
    
    #Add maximum MSLP
    if not np.isnan(thresh['p_max']):
        plot_subtitle.append(f"{lteq} {thresh['p_max']}hPa")            
    else:
        thresh['p_max'] = 9999
    
    #Add minimum change in wind speed
    if not np.isnan(thresh['dv_min']):
        plot_subtitle.append(f"{gteq} {thresh['dv_min']}kt / {thresh['dt_window']}hr")            
    else:
        thresh['dv_min'] = -9999
    
    #Add maximum change in MSLP
    if not np.isnan(thresh['dp_max']):
        plot_subtitle.append(f"{lteq} {thresh['dp_max']}hPa / {thresh['dt_window']}hr")            
    else:
        thresh['dp_max'] = 9999
    
    #Add maximum change in wind speed
    if not np.isnan(thresh['dv_max']):
        plot_subtitle.append(f"{lteq} {thresh['dv_max']}kt / {thresh['dt_window']}hr")            
    else:
        thresh['dv_max'] = 9999
    
    #Add minimum change in MSLP
    if not np.isnan(thresh['dp_min']):
        plot_subtitle.append(f"{gteq} {thresh['dp_min']}hPa / {thresh['dt_window']}hr")            
    else:
        thresh['dp_min'] = -9999
    
    #Combine plot_subtitle into string
    if len(plot_subtitle)>0:
        plot_subtitle = '\n'+', '.join(plot_subtitle)
    else:
        plot_subtitle = ''
    
    #Return modified thresh and plot title
    return thresh, plot_subtitle


def interp_storm(storm_dict,timeres=1,dt_window=24,dt_align='middle'):
    
    r"""
    Interpolate a storm dictionary temporally to a specified time resolution. Referenced from ``TrackDataset.filter_storms()``. Internal function.
    
    Parameters
    ----------
    storm_dict : dict
        Dictionary containing a storm entry.
    timeres : int
        Temporal resolution in hours to interpolate storm data to. Default is 1 hour.
    dt_window : int
        Time window in hours over which to calculate temporal change data. Default is 24 hours.
    dt_align : str
        Whether to align the temporal change window as "start", "middle" or "end" of the dt_window time period.
    
    Returns
    -------
    dict
        Dictionary containing the updated storm entry.
    """
    
    #Create an empty dict for the new storm entry
    new_storm = {}
    
    #Copy over non-list attributes
    for key in storm_dict.keys():
        if isinstance(storm_dict[key],list) == False:
            new_storm[key] = storm_dict[key]
    
    #Create an empty list for entries
    for name in ['date','vmax','mslp','lat','lon','type']:
        new_storm[name] = []
    
    #Convert dates to numbers for ease of calculation
    times = mdates.date2num(storm_dict['date'])
    
    #Convert lat & lons to arrays, and ensure lons are out of 360 degrees
    storm_dict['type'] = np.asarray(storm_dict['type'])
    storm_dict['lon'] = np.array(storm_dict['lon']) % 360
    
    #Attempt temporal interpolation
    try:
        
        #Create a list of target times given the requested temporal resolution
        targettimes = np.arange(times[0],times[-1]+timeres/24,timeres/24)
        
        #Update dates
        new_storm['date'] = [t.replace(tzinfo=None) for t in mdates.num2date(targettimes)]
        
        #Interpolate and fill in storm type
        stormtype = np.ones(len(storm_dict['type']))*-99
        stormtype[np.where((storm_dict['type']=='TD') | (storm_dict['type']=='SD') | (storm_dict['type']=='TS') | \
                           (storm_dict['type']=='SS') | (storm_dict['type']=='HU'))] = 0
        new_storm['type'] = np.interp(targettimes,times,stormtype)
        new_storm['type'] = np.where(new_storm['type']<0,'NT','TD')
        
        #Interpolate and fill in other variables
        for name in ['vmax','mslp','lat','lon']:
            new_storm[name] = np.interp(targettimes,times,storm_dict[name])
        
        #Calculate change in wind & MSLP over temporal resolution
        new_storm['dvmax_dt'] = [np.nan] + list((new_storm['vmax'][1:]-new_storm['vmax'][:-1]) / timeres)
        new_storm['dmslp_dt'] = [np.nan] + list((new_storm['mslp'][1:]-new_storm['mslp'][:-1]) / timeres)
        
        #Calculate x and y position change over temporal window
        rE = 6.371e3 #km
        d2r = np.pi/180.
        new_storm['dx_dt'] = [np.nan]+list(d2r*(new_storm['lon'][1:]-new_storm['lon'][:-1])* \
                 rE*np.cos(d2r*np.mean([new_storm['lat'][1:],new_storm['lat'][:-1]],axis=0))/timeres)
        new_storm['dy_dt'] = [np.nan]+list(d2r*(new_storm['lat'][1:]-new_storm['lat'][:-1])* \
                 rE/timeres)
        
        #Convert change in wind & MSLP to change over specified window
        for name in ['dvmax_dt','dmslp_dt']:
            tmp = np.round(np.convolve(new_storm[name],[1]*int(dt_window/timeres),mode='valid'),1)         
            if dt_align=='end':
                new_storm[name] = [np.nan]*(len(new_storm[name])-len(tmp))+list(tmp)
            if dt_align=='middle':
                tmp2 = [np.nan]*int((len(new_storm[name])-len(tmp))//2)+list(tmp)
                new_storm[name] = tmp2+[np.nan]*(len(new_storm[name])-len(tmp2))
            if dt_align=='start':
                new_storm[name] = list(tmp)+[np.nan]*(len(new_storm[name])-len(tmp))
        
        #Convert change in position to change over specified window
        for name in ['dx_dt','dy_dt']:
            tmp = np.convolve(new_storm[name],[timeres/dt_window]*int(dt_window/timeres),mode='valid')
            if dt_align=='end':
                new_storm[name] = [np.nan]*(len(new_storm[name])-len(tmp))+list(tmp)
            if dt_align=='middle':
                tmp2 = [np.nan]*int((len(new_storm[name])-len(tmp))//2)+list(tmp)
                new_storm[name] = tmp2+[np.nan]*(len(new_storm[name])-len(tmp2))
            if dt_align=='start':
                new_storm[name] = list(tmp)+[np.nan]*(len(new_storm[name])-len(tmp))
        
        #Return new dict
        return new_storm
    
    #Otherwise, simply return NaNs
    except:
        for name in new_storm.keys():
            try:
                storm_dict[name]
            except:
                storm_dict[name]=np.ones(len(new_storm[name]))*np.nan
        return storm_dict


def filter_storms_vp(trackdata,year_min=0,year_max=9999,subset_domain=None):
    
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
    
    #If no subset domain is passed, use global data
    if subset_domain == None:
        lon_min,lon_max,lat_min,lat_max = [0,360,-90,90]
    else:
        lon_min,lon_max,lat_min,lat_max = [float(i) for i in subset_domain.split("/")]
    
    #Empty list for v-p relationship data
    vp = []
    
    #Iterate over every storm in dataset
    for key in trackdata.keys:
        
        #Retrieve storm dictionary
        istorm = trackdata.data[key]
        
        #Iterate over every storm time step
        for i,(iwind,imslp,itype,ilat,ilon,itime) in \
        enumerate(zip(istorm['vmax'],istorm['mslp'],istorm['type'],istorm['lat'],istorm['lon'],istorm['date'])):
            
            #Ensure both have data and are while the cyclone is tropical
            if np.nan not in [iwind,imslp] and itype in ['TD','TS','SS','HU','TY'] \
                   and lat_min<=ilat<=lat_max and lon_min<=ilon%360<=lon_max \
                   and year_min<=itime.year<=year_max:
                vp.append([imslp,iwind])
    
    #Return v-p relationship list
    return vp

def testfit(data,x,order):
    
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
    
    #Make sure there are enough samples
    if len(data) > 50:
        f = np.polyfit([i[1] for i in data],[i[0] for i in data],order)
        y = sum([f[i]*x**(order-i) for i in range(order+1)])
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

def convert_to_julian(date):
    
    r"""
    Convert a date to Julian days. Referenced from ``TrackDataset.ace_climo()`` and ``TrackDataset.hurricane_days_climo()``. Internal function.
    
    Parameters
    ----------
    date : datetime.datetime
        Datetime object of the date to be converted to Julian days.
    
    Returns
    -------
    int
        Integer representing the Julian day of the requested date.
    """
    
    year = date.year
    return ((date - dt(year,1,1,0)).days + (date - dt(year,1,1,0)).seconds/86400.0) + 1

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
    
    #Get number of days in year
    length_of_year = convert_to_julian(dt(year,12,31,0))+1.0
    
    #Construct a list of months and names
    months = range(1,13,1)
    months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_dates = [dt(year,i,1,0) for i in months]
    
    #Get midpoint x-axis location of month
    months_julian = [int(convert_to_julian(i)) for i in months_dates]
    midpoint_julian = (np.array(months_julian) + np.array(months_julian[1:]+[length_of_year]))/2.0
    return {'start':months_julian,'midpoint':midpoint_julian.tolist(),'name':months_names}

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
    
    #If number is less than 10, add a leading zero out front
    if number < 10:
        return f'0{number}'
    
    #Otherwise, simply convert to a string
    return str(number)

def plot_credit():
    return "Plot generated using troPYcal"

def add_credit(ax,text):
    import matplotlib.patheffects as path_effects    
    a = ax.text(0.99,0.01,text,fontsize=9,color='k',alpha=0.7,fontweight='bold',
            transform=ax.transAxes,ha='right',va='bottom',zorder=10)
    a.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                   path_effects.Normal()])

def pac_2006_cyclone():
    
    """
    Data for 2006 Central Pacific cyclone obtained from a simple MSLP minimum based tracker applied to the ERA-5 reanalysis dataset. Sustained wind values from the duration of the storm's subtropical and tropical stages were obtained from an estimate from Dr. Karl Hoarau of the Cergy-Pontoise University in Paris:
    
    https://australiasevereweather.com/cyclones/2007/trak0611.htm
    """
    
    #add empty entry into dict
    storm_id = 'CP052006'
    storm_dict = {}
    
    storm_dict = {'id':'CP052006','operational_id':'','name':'UNNAMED','season':2006,'year':2006,'basin':'east_pacific'}
    storm_dict['source'] = 'hurdat'
    storm_dict['source_info'] = 'ERA5 Reanalysis'

    #add empty lists
    for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
        storm_dict[val] = []
    storm_dict['ace'] = 0.0
    
    #Add obs from reference
    storm_dict['date'] = ['2006102812', '2006102815', '2006102818', '2006102821', '2006102900', '2006102903', '2006102906', '2006102909', '2006102912', '2006102915', '2006102918', '2006102921', '2006103000', '2006103003', '2006103006', '2006103009', '2006103012', '2006103015', '2006103018', '2006103021', '2006103100', '2006103103', '2006103106', '2006103109', '2006103112', '2006103115', '2006103118', '2006103121', '2006110100', '2006110103', '2006110106', '2006110109', '2006110112', '2006110115', '2006110118', '2006110121', '2006110200', '2006110203', '2006110206', '2006110209', '2006110212', '2006110215', '2006110218', '2006110221', '2006110300', '2006110303', '2006110306', '2006110309', '2006110312', '2006110315', '2006110318']
    storm_dict['lat'] = [36.0, 37.75, 38.25, 38.5, 39.5, 39.75, 40.0, 40.0, 39.25, 38.5, 37.5, 37.0, 36.75, 36.75, 36.25, 36.0, 36.0, 36.25, 36.75, 37.25, 37.75, 38.5, 38.75, 39.25, 39.75, 40.25, 40.75, 41.25, 42.0, 42.5, 42.75, 42.75, 42.75, 42.75, 42.5, 42.25, 42.25, 42.0, 42.0, 42.25, 42.5, 42.75, 43.0, 43.5, 44.0, 44.5, 45.5, 46.25, 46.75, 47.75, 48.5]
    storm_dict['lon'] = [-148.25, -147.75, -148.25, -148.25, -148.5, -148.75, -149.75, -150.5, -151.5, -151.75, -151.75, -151.0, -150.25, -150.0, -149.5, -148.5, -147.5, -146.5, -145.5, -144.75, -144.0, -143.5, -143.25, -143.0, -142.75, -142.5, -142.5, -143.0, -143.5, -144.0, -144.75, -145.5, -146.0, -146.25, -146.0, -145.75, -145.25, -144.25, -143.25, -142.25, -140.75, -139.5, -138.0, -136.5, -135.0, -133.5, -132.0, -130.5, -128.5, -126.75, -126.0]
    storm_dict['mslp'] = [1007, 1003, 999, 995, 992, 989, 990, 990, 991, 991, 992, 993, 993, 992, 994, 994, 994, 994, 995, 995, 993, 993, 994, 993, 993, 993, 993, 993, 990, 989, 989, 989, 988, 988, 989, 989, 988, 989, 990, 991, 991, 991, 993, 994, 993, 994, 995, 996, 996, 996, 997]
    storm_dict['vmax'] = [30, 40, 50, 45, 45, 45, 40, 40, 40, 40, 35, 35, 35, 35, 35, 35, 30, 30, 35, 35, 35, 40, 45, 45, 40, 40, 45, 45, 45, 45, 50, 50, 55, 55, 50, 50, 50, 50, 50, 50, 45, 40, 35, 35, 30, 30, 25, 25, 30, 30, 25]
    storm_dict['vmax_era5'] = [31, 38, 47, 46, 47, 45, 38, 42, 43, 40, 37, 36, 35, 33, 35, 35, 33, 31, 33, 32, 31, 29, 28, 30, 28, 28, 29, 29, 30, 32, 31, 29, 28, 26, 25, 26, 28, 27, 28, 29, 28, 27, 27, 27, 28, 26, 27, 27, 31, 30, 26]
    storm_dict['type'] = ['EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'SS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX', 'EX']
    storm_dict['date'] = [dt.strptime(i,'%Y%m%d%H') for i in storm_dict['date']]
    
    #Add other variables
    storm_dict['extra_obs'] = [0 if i.hour in [0,6,12,18] else 1 for i in storm_dict['date']]
    storm_dict['special'] = ['' for i in storm_dict['date']]
    storm_dict['wmo_basin'] = ['east_pacific' for i in storm_dict['date']]
    
    #Calculate ACE
    for i,(vmax,storm_type,idate) in enumerate(zip(storm_dict['vmax'],storm_dict['type'],storm_dict['date'])):
        ace = (10**-4) * (vmax**2)
        hhmm = idate.strftime('%H%M')
        if hhmm in ['0000','0600','1200','1800'] and storm_type in ['SS','TS','HU']:
            storm_dict['ace'] += ace
    
    #Replace original entry with this
    return storm_dict

def cyclone_catarina():
    
    """
    https://journals.ametsoc.org/doi/pdf/10.1175/MWR3330.1
    """
    
    #add empty entry into dict
    storm_id = 'AL502004'
    storm_dict = {}
    
    storm_dict = {'id':'AL502004','operational_id':'','name':'CATARINA','season':2004,'year':2004,'basin':'south_atlantic'}
    storm_dict['source'] = 'McTaggart-Cowan et al. (2006): https://doi.org/10.1175/MWR3330.1'

    #add empty lists
    for val in ['date','extra_obs','special','type','lat','lon','vmax','mslp','wmo_basin']:
        storm_dict[val] = []
    storm_dict['ace'] = 0.0
    
    #Add obs from reference
    storm_dict['date'] = ['200403191800','200403200000','200403200600','200403201200','200403201800','200403210000','200403210600','200403211200','200403211800','200403220000','200403220600','200403221200','200403221800','200403230000','200403230600','200403231200','200403231800','200403240000','200403240600','200403241200','200403241800','200403250000','200403250600','200403251200','200403251800','200403260000','200403260600','200403261200','200403261800','200403270000','200403270600','200403271200','200403271800','200403280000','200403280600','200403281200','200403281800']
    storm_dict['lat'] = [-27.0,-26.5,-25.3,-25.5,-26.5,-26.8,-27.5,-28.7,-29.5,-30.9,-31.9,-32.3,-31.5,-30.7,-29.8,-29.5,-29.4,-29.3,-29.2,-29.1,-29.1,-29.0,-28.9,-28.7,-28.7,-28.7,-28.7,-28.8,-28.9,-29.1,-29.2,-29.5,-29.5,-29.3,-29.0,-28.5,-28.5]
    storm_dict['lon'] = [-49.0,-48.5,-48.0,-46.0,-44.5,-43.0,-42.0,-40.5,-39.5,-38.5,-37.0,-36.7,-36.5,-36.7,-37.0,-37.5,-38.1,-38.5,-38.8,-39.0,-39.4,-39.9,-40.4,-41.2,-41.9,-42.6,-43.1,-43.7,-44.2,-44.9,-45.6,-46.4,-47.5,-48.3,-49.7,-50.1,-51.0]
    storm_dict['mslp'] = [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,1002,990,991,993,992,990,990,993,993,994,994,989,989,982,975,974,974,972,972,972,np.nan,np.nan,np.nan]
    storm_dict['vmax'] = [25.0,25.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,30.0,35.0,35.0,35.0,35.0,40.0,40.0,45.0,50.0,55.0,60.0,65.0,70.0,70.0,70.0,75.0,75.0,80.0,80.0,85.0,60.0,45.0]
    storm_dict['type'] = ['EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','EX','SS','SS','SS','SS','SS','TS','TS','TS','TS','HU','HU','HU','HU','HU','HU','HU','HU','HU','TS','TS']
    storm_dict['date'] = [dt.strptime(i,'%Y%m%d%H%M') for i in storm_dict['date']]
    
    #Add other variables
    storm_dict['extra_obs'] = [0 for i in storm_dict['date']]
    storm_dict['special'] = ['' for i in storm_dict['date']]
    storm_dict['wmo_basin'] = ['south_atlantic' for i in storm_dict['date']]
    
    #Calculate ACE
    for i,(vmax,storm_type,idate) in enumerate(zip(storm_dict['vmax'],storm_dict['type'],storm_dict['date'])):
        ace = (10**-4) * (vmax**2)
        hhmm = idate.strftime('%H%M')
        if hhmm in ['0000','0600','1200','1800'] and storm_type in ['SS','TS','HU']:
            storm_dict['ace'] += ace
    
    #Replace original entry with this
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
    
    #Dictionary mapping numbers to string representations
    d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
          6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
          11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
          15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
          19 : 'nineteen', 20 : 'twenty',
          30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
          70 : 'seventy', 80 : 'eighty', 90 : 'ninety' }

    #If number is less than 20, return string
    if number < 20:
        return d[number]
    
    #Otherwise, form number from combination of strings
    elif number < 100:
        if number % 10 == 0:
            return d[number]
        else:
            return d[number // 10 * 10] + '-' + d[number % 10]
    
    #If larger than 100, raise error
    else:
        msg = "Please choose a number less than 100."
        raise ValueError(msg)
