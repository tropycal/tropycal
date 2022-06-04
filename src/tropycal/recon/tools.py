import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
from scipy.ndimage import gaussian_filter as gfilt,gaussian_filter1d as gfilt1d
from scipy.interpolate import griddata,interp2d,interp1d,SmoothBivariateSpline
import warnings
import matplotlib as mlib
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

from ..utils import classify_subtropical, get_storm_classification

def uv_from_wdir(wspd,wdir):
    d2r = np.pi/180.
    theta = (270 - wdir) * d2r
    u = wspd * np.cos(theta)
    v = wspd * np.sin(theta)
    return u,v

#------------------------------------------------------------------------------
# TOOLS FOR RECON INTERPOLATION
#------------------------------------------------------------------------------

class interpRecon:
    
    """
    Interpolates storm-centered data by time and space.
    """
    
    def __init__(self,dfRecon,varname,radlim=None,window=6,align='center'):
        
        #Retrieve dataframe containing recon data, and variable to be interpolated
        self.dfRecon = dfRecon
        self.varname = varname
        self.window = window
        self.align = align
        
        #Specify outer radius cutoff in kilometer
        if radlim is None:
            self.radlim = 200 #km
        else:
            self.radlim = radlim
    
    
    def interpPol(self):
        r"""
        Interpolates storm-centered recon data into a polar grid, and outputs the radius grid, azimuth grid and interpolated variable.
        """
        
        #Read in storm-centered data and storm-relative coordinates for all times
        data = [k for i,j,k in zip(self.dfRecon['xdist'],self.dfRecon['ydist'],self.dfRecon[self.varname]) if not np.isnan([i,j,k]).any()]
        path = [(i,j) for i,j,k in zip(self.dfRecon['xdist'],self.dfRecon['ydist'],self.dfRecon[self.varname]) if not np.isnan([i,j,k]).any()]

        #Function for interpolating cartesian to polar coordinates
        def cart2pol(x, y, offset=0):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return(rho, phi+offset)
        
        #Interpolate every storm-centered coordinate pair into polar coordinates
        pol_path = [cart2pol(*p) for p in path]

        #Wraps around the data to ensure no cutoff around 0 degrees
        pol_path_wrap = [cart2pol(*p,offset=-2*np.pi) for p in path]+pol_path+\
                    [cart2pol(*p,offset=2*np.pi) for p in path]
        data_wrap = np.concatenate([data]*3)
        
        #Creates a grid of rho (radius) and phi (azimuth)
        grid_rho, grid_phi = np.meshgrid(np.arange(0,self.radlim+.1,.5),np.linspace(-np.pi,np.pi,181))
    
        #Interpolates storm-centered point data in polar coordinates onto a gridded polar coordinate field
        grid_z_pol = griddata(pol_path_wrap,data_wrap,(grid_rho,grid_phi),method='linear')
        
        try:
            #Calculate radius of maximum wind (RMW)
            rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]

            #Within the RMW, replace NaNs with minimum value within the RMW
            filleye = np.where((grid_rho<rmw) & (np.isnan(grid_z_pol)))

            grid_z_pol[filleye]=np.nanmin(grid_z_pol[np.where(grid_rho<rmw)])
        except:
            pass
    
        #Return fields
        return grid_rho, grid_phi, grid_z_pol      
        
    
    def interpCart(self):
        r"""
        Interpolates polar storm-centered gridded fields into cartesian coordinates
        """
        
        #Interpolate storm-centered recon data into gridded polar grid (rho, phi and gridded data)
        grid_rho, grid_phi, grid_z_pol = self.interpPol()
        
        #Calculate RMW
        rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]
        
        #Wraps around the data to ensure no cutoff around 0 degrees
        grid_z_pol_wrap = np.concatenate([grid_z_pol]*3)
        
        #Radially smooth based on RMW - more smoothing farther out from RMW
        grid_z_pol_final = np.array([gfilt(grid_z_pol_wrap,(6,3+abs(r-rmw)/10))[:,i] \
                                     for i,r in enumerate(grid_rho[0,:])]).T[len(grid_phi):2*len(grid_phi)]
        
        #Function for interpolating polar cartesian to coordinates
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return(x, y)
        
        #Interpolate the rho and phi gridded fields to a 1D cartesian list
        pinterp_grid = [pol2cart(i,j) for i,j in zip(grid_rho.flatten(),grid_phi.flatten())]
        
        #Flatten the radially smoothed variable grid to match with the shape of pinterp_grid
        pinterp_z = grid_z_pol_final.flatten()
        
        #Setting up the grid in cartesian coordinate space, based on previously specified radial limit
        #Grid resolution = 1 kilometer
        grid_x, grid_y = np.meshgrid(np.linspace(-self.radlim,self.radlim,self.radlim*2+1),\
                                     np.linspace(-self.radlim,self.radlim,self.radlim*2+1))
        grid_z = griddata(pinterp_grid,pinterp_z,(grid_x,grid_y),method='linear')
    
        #Return output grid
        return grid_x, grid_y, grid_z
    

    def interpHovmoller(self,target_track):
        r"""
        Creates storm-centered interpolated data in polar coordinates for each timestep, and averages azimuthally to create a hovmoller.
        
        target_track = dict
            dict of either archer or hurdat data (contains lat, lon, time/date)
        window = hours
            sets window in hours relative to the time of center pass for interpolation use.
        """
    
        window = self.window
        align = self.align
    
        #Store the dataframe containing recon data
        tmpRecon = self.dfRecon.copy()
        #Sets window as a timedelta object
        window = timedelta(seconds=int(window*3600))
        
        #Error check for time dimension name
        if 'time' not in target_track.keys():
            target_track['time']=target_track['date']
        
        #Find times of all center passes
        centerTimes = tmpRecon[tmpRecon['iscenter']==1]['time']
        
        #Data is already centered on center time, so shift centerTimes to the end of the window
        spaceInterpTimes = [t+window/2 for t in centerTimes]
        
        #Takes all times within track dictionary that fall between spaceInterpTimes
        trackTimes = [t for t in target_track['time'] if min(spaceInterpTimes)<t<max(spaceInterpTimes)]
        
        #Iterate through all data surrounding a center pass given the window previously specified, and create a polar
        #grid for each
        start_time = dt.now()
        print("--> Starting interpolation")
        
        spaceInterpData={}
        for time in spaceInterpTimes:
            #Temporarily set dfRecon to this centered subset window
            self.dfRecon = tmpRecon[(tmpRecon['time']>time-window) & (tmpRecon['time']<=time)]
            #print(time) #temporarily disabling this
            grid_rho, grid_phi, grid_z_pol = self.interpPol() #Create polar centered grid
            grid_azim_mean = np.mean(grid_z_pol,axis=0) #Average azimuthally
            spaceInterpData[time] = grid_azim_mean #Append data for this time step to dictionary
        
        #Sets dfRecon back to original full data
        self.dfRecon = tmpRecon
        reconArray = np.array([i for i in spaceInterpData.values()])

        #Interpolate over every half hour
        newTimes = np.arange(mdates.date2num(trackTimes[0]),mdates.date2num(trackTimes[-1])+1e-3,1/48)    
        oldTimes = mdates.date2num(np.array(list(spaceInterpData.keys())))
        #print(len(oldTimes),reconArray.shape)
        reconTimeInterp=np.apply_along_axis(lambda x: np.interp(newTimes,oldTimes,x),
                                 axis=0,arr=reconArray)
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed interpolation ({tsec} seconds)")
        
        #Output RMW and hovmoller data and store as an attribute in the object
        self.rmw = grid_rho[0,np.nanargmax(reconTimeInterp,axis=1)]
        self.Hovmoller = {'time':mdates.num2date(newTimes),'radius':grid_rho[0,:],'hovmoller':reconTimeInterp}
        return self.Hovmoller

    def interpMaps(self,target_track,interval=0.5,stat_vars=None):
        r"""
        1. Can just output a single map (interpolated to lat/lon grid and projected onto the Cartopy map
        2. If target_track is longer than 1, outputs multiple maps to a directory
        """
        
        window = self.window
        align = self.align
        
        #Store the dataframe containing recon data
        tmpRecon = self.dfRecon.copy()
        #Sets window as a timedelta object
        window = timedelta(seconds=int(window*3600))
 
        #Error check for time dimension name
        if 'time' not in target_track.keys():
            target_track['time']=target_track['date']
       
        #If target_track > 1 (tuple or list of times), then retrieve multiple center pass times and center around the window
        if isinstance(target_track['time'],(tuple,list,np.ndarray)):
            centerTimes=tmpRecon[tmpRecon['iscenter']==1]['time']
            spaceInterpTimes=[t for t in centerTimes]
            trackTimes=[t for t in target_track['time'] if min(spaceInterpTimes)-window/2<t<max(spaceInterpTimes)+window/2]
        #Otherwise, just use a single time
        else:
            spaceInterpTimes=list([target_track['time']])
            trackTimes=spaceInterpTimes.copy()
        
        #Experimental - add recon statistics (e.g., wind, MSLP) to plot
        # **** CHECK BACK ON THIS ****
        spaceInterpData={}
        recon_stats=None
        if stat_vars is not None:
            recon_stats={name:[] for name in stat_vars.keys()}
        #Iterate through all data surrounding a center pass given the window previously specified, and create a polar
        #grid for each
        start_time = dt.now()
        print("--> Starting interpolation")
        
        for time in spaceInterpTimes:
            print(time)
            self.dfRecon = tmpRecon[(tmpRecon['time']>time-window/2) & (tmpRecon['time']<=time+window/2)]
            grid_x,grid_y,grid_z = self.interpCart()
            spaceInterpData[time] = grid_z
            if stat_vars is not None:
                for name in stat_vars.keys():
                    recon_stats[name].append(stat_vars[name](self.dfRecon[name]))
        
        #Sets dfRecon back to original full data
        self.dfRecon = tmpRecon        
        reconArray = np.array([i for i in spaceInterpData.values()])

        #If multiple times, create a lat & lon grid for half hour intervals
        if len(trackTimes)>1:
            newTimes = np.arange(mdates.date2num(trackTimes[0]),mdates.date2num(trackTimes[-1])+interval/24,interval/24)    
            oldTimes = mdates.date2num(np.array(list(spaceInterpData.keys())))
            reconTimeInterp=np.apply_along_axis(lambda x: np.interp(newTimes,oldTimes,x),
                                 axis=0,arr=reconArray)
            #Get centered lat and lon by interpolating from target_track dictionary (whether archer or HURDAT)
            clon = np.interp(newTimes,mdates.date2num(target_track['time']),target_track['lon'])
            clat = np.interp(newTimes,mdates.date2num(target_track['time']),target_track['lat'])
        else:
            newTimes = mdates.date2num(trackTimes)[0]
            reconTimeInterp = reconArray[0]
            clon = target_track['lon']
            clat = target_track['lat']

        #Interpolate storm stats to corresponding times
        if stat_vars is not None:
            for varname in recon_stats.keys():
                recon_stats[varname] = np.interp(newTimes,oldTimes,recon_stats[varname])
            
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed interpolation ({tsec} seconds)")            
            
        #Create dict of map data (interpolation time, x & y grids, 'maps' (3D grid of field, time/x/y), and return
        self.Maps = {'time':mdates.num2date(newTimes),'grid_x':grid_x,'grid_y':grid_y,'maps':reconTimeInterp,
                           'center_lon':clon,'center_lat':clat,'stats':recon_stats}
        return self.Maps

    @staticmethod
    def _interpFunc(data1, times1, times2):
        #Interpolate data
        f = interp1d(mdates.date2num(times1),data1)
        data2 = f(mdates.date2num(times2))
        return data2

#------------------------------------------------------------------------------
# TOOLS FOR PLOTTING
#------------------------------------------------------------------------------

def find_var(request,thresh):
    
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
    
    #Convert command to lowercase
    request = request.lower()
    
    #Count of number of storms
    if request.find('count') >= 0 or request.find('num') >= 0:
        return thresh, 'time'

    if request.find('time') >= 0 or request.find('day') >= 0:
        return thresh, 'time'
    
    #Sustained wind, or change in wind speed
    if request.find('wind') >= 0 or request.find('vmax') >= 0:
        return thresh,'wspd'
    if request.find('30s wind') >= 0 or request.find('vmax') >= 0:
        return thresh,'wspd'
    if request.find('10s wind') >= 0 or request.find('vmax') >= 0:
        return thresh,'pkwnd'
    
    #Minimum MSLP
    elif request.find('pressure') >= 0 or request.find('slp') >= 0:
        return thresh,'p_sfc'

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
    
    print(request)
    
    #Convert command to lowercase
    request = request.lower()
    
    #Numpy maximum function
    if request.find('max') == 0 or request.find('latest') == 0:
        return thresh, lambda x: np.nanmax(x)
    
    #Numpy minimum function
    if request.find('min') == 0 or request.find('earliest') == 0:
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
    
    #ACE - cumulative function
    elif request.find('ace') >=0:
        return thresh, lambda x: np.nansum(x)
    elif request.find('acie') >=0:
        return thresh, lambda x: np.nansum(x)
    
    #Otherwise, function cannot be identified
    else:
        msg = "Cannot decipher the function. Please refer to documentation for examples on how to phrase the \"request\" string."
        raise RuntimeError(msg)


def get_recon_title(varname,level=None):
    
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

    return titlename,unitname

def hovmoller_plot_title(storm_obj,Hov,varname):
    
    r"""
    Generate plot title for hovmoller.
    """
    
    #Retrieve storm dictionary from Storm object
    storm_data = storm_obj.dict
    
    #------- construct left title ---------
    
    #Subset sustained wind array to when the storm was tropical
    type_array = np.array(storm_data['type'])
    idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
    tropical_vmax = np.array(storm_data['vmax'])[idx]
    
    #Coerce to include non-TC points if storm hasn't been designated yet
    add_ptc_flag = False
    if len(tropical_vmax) == 0:
        add_ptc_flag = True
        idx = np.where((type_array == 'LO') | (type_array == 'DB'))
    tropical_vmax = np.array(storm_data['vmax'])[idx]
    
    #Determine storm classification based on subtropical status & basin
    subtrop = classify_subtropical(np.array(storm_data['type']))
    peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
    peak_basin = storm_data['wmo_basin'][peak_idx]
    storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
    if add_ptc_flag == True: storm_type = "Potential Tropical Cyclone"
    
    #Get title descriptor based on variable
    vartitle = get_recon_title(varname)
    
    #Add left title
    dot = u"\u2022"
    title_left = f"{storm_type} {storm_data['name']}\n" + 'Recon: '+' '.join(vartitle)

    #------- construct right title ---------
    
    #Determine start and end dates of hovmoller
    start_date = dt.strftime(min(Hov['time']),'%H:%M UTC %d %b %Y')
    end_date = dt.strftime(max(Hov['time']),'%H:%M UTC %d %b %Y')
    title_right = f'Start ... {start_date}\nEnd ... {end_date}'
    
    #Return both titles
    return title_left,title_right


def get_bounds(data,bounds):
    try:
        datalims = (np.nanmin(data),np.nanmax(data))
    except:
        datalims = bounds
    bounds = [l if b is None else b for b,l in zip(bounds,datalims)]
    bounds = [b for b in bounds if b is not None]
    return (min(bounds),max(bounds))

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
        'p_sfc':'red',
        'temp':'red',
        'dwpt':'green',
        'wspd':'blue',
        'sfmr':'#282893',
        'pkwnd':'#5A9AF0',
        'rain':'#C551DC',
        'plane_z':'#909090',
        'plane_p':'#4D4D4D',
    }
    
    names = {
        'p_sfc':'MSLP',
        'temp':'Temperature',
        'dwpt':'Dewpoint',
        'wspd':'Flight Level Wind',
        'sfmr':'Surface Wind',
        'pkwnd':'Peak Wind Gust',
        'rain':'Rain Rate',
        'plane_z':'Altitude',
        'plane_p':'Pressure',
    }
    
    full_names = {
        'p_sfc':'Mean Sea Level Pressure (hPa)',
        'temp':'Temperature (C)',
        'dwpt':'Dewpoint (C)',
        'wspd':'Flight Level Wind (kt)',
        'sfmr':'Surface Wind (kt)',
        'pkwnd':'Peak Wind Gust (kt)',
        'rain':'Rain Rate (mm/hr)',
        'plane_z':'Geopotential Height (m)',
        'plane_p':'Pressure (hPa)',
    }
    
    color = colors.get(varname,'')
    name = names.get(varname,'')
    full_name = full_names.get(varname,'')
    
    return {'color':color,'name':name,'full_name':full_name}
