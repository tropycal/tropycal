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
        
        #Calculate radius of maximum wind (RMW)
        rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]
        
        #Within the RMW, replace NaNs with minimum value within the RMW
        filleye = np.where((grid_rho<rmw) & (np.isnan(grid_z_pol)))
        try:
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

def get_recon_title(varname):
    
    r"""
    Generate plot title descriptor for plots.
    """
    
    if varname.lower() == 'wspd':
        titlename = '30s FL wind'
        unitname = r'(kt)'
    if varname.lower() == 'pkwnd':
        titlename = '10s FL wind'
        unitname = r'(kt)'
    if varname.lower() == 'sfmr':
        titlename = 'SFMR 10s sfc wind'
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
    
    #Determine storm classification based on subtropical status & basin
    subtrop = classify_subtropical(np.array(storm_data['type']))
    peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
    peak_basin = storm_data['wmo_basin'][peak_idx]
    storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
    
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
