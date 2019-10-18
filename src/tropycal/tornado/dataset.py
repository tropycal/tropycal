r"""Functionality for reading and analyzing SPC tornado dataset."""

import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
from geopy.distance import great_circle
from scipy.interpolate import interp1d
import matplotlib.dates as mdates
import warnings

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
import matplotlib.patheffects as patheffects

try:
    import cartopy.feature as cfeature
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
except:
    warnings.warn("Warning: Cartopy is not installed in your python environment. Plotting functions will not work.")

from .plot import TornadoPlot
#from ...tropycal import tracks

class TornadoDataset():
    
    r"""
    Creates an instance of a Dataset object containing tornado data.

    Parameters
    ----------
    mag_thresh : int
        Minimum threshold for tornado rating.

    Returns
    -------
    Dataset
        An instance of Dataset.
    """

    def __init__(self, mag_thresh=0):
        
        #Error check
        if isinstance(mag_thresh,int) == False:
            raise TypeError("mag_thresh must be of type int.")
        elif mag_thresh not in [0,1,2,3,4,5]:
            raise ValueError("mag_thresh must be between 0 and 5.")
        
        #Read in tornado dataset
        timer_start = dt.now()
        yrnow = timer_start.year
        print(f'--> Starting to read in tornado track data')
        try:
            yrlast = yrnow-1
            Tors = pd.read_csv(f'https://www.spc.noaa.gov/wcm/data/1950-{yrlast}_actual_tornadoes.csv',\
                               error_bad_lines=False,parse_dates=[['mo','dy','yr','time']])
        except:
            yrlast = yrnow-2
            Tors = pd.read_csv(f'https://www.spc.noaa.gov/wcm/data/1950-{yrlast}_actual_tornadoes.csv',\
                               error_bad_lines=False,parse_dates=[['mo','dy','yr','time']])
        print(f'--> Completed reading in tornado data for 1950-{yrlast} (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        
        #Get UTC from timezone (most are 3 = CST, but some 0 and 9 = GMT)
        tz = np.array([timedelta(hours=9-int(i)) for i in Tors['tz']])
        tors_dt = [pd.to_datetime(i) for i in Tors['mo_dy_yr_time']]
        Tors = Tors.assign(UTC_time = tors_dt+tz)
        
        #Filter for only those tors at least F/EF scale mag_thresh.
        Tors = Tors[Tors['mag']>=mag_thresh]

        #Clean up lat/lons       
        Tors = Tors[(Tors['slat']!=0)|(Tors['slon']!=0)]
        Tors = Tors[(Tors['slat']>=20) & (Tors['slat']<=50)]
        Tors = Tors[(Tors['slon']>=-130) & (Tors['slon']<=-65)]
        Tors = Tors[(Tors['elat']>=20)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elat']<=50)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elon']>=-130)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elon']<=-65)|(Tors['elat']==0)]
        Tors = Tors.assign(elat = [Tors['slat'].values[u] if i==0 else i for u, i in enumerate(Tors['elat'].values)])
        self.Tors = Tors.assign(elon = [Tors['slon'].values[u] if i==0 else i for u, i in enumerate(Tors['elon'].values)])

    def getTCtors(self,storm,dist_thresh=1000):
        
        r"""
        Retrieves all tornado tracks that occur within a distance threshold (dist_thresh) 
        of the position of a tropical cyclone along its track.
        
        Parameters
        ----------
        storm : Storm object containing info on the TC.
        dist_thresh : threshold distance within which tornadoes are attributed to the TC.
        
        Returns
        -------
        Dataframe of tornadoes,
        """
        
        self.dist_thresh = dist_thresh
        stormdict = storm.to_dict()
        self.stormdict = stormdict
    
        stormTors = self.Tors[(self.Tors['UTC_time']>=min(stormdict['date'])) & \
                         (self.Tors['UTC_time']<=max(stormdict['date']))]
        
        f = interp1d(mdates.date2num(stormdict['date']),stormdict['lon'])
        interp_clon = f(mdates.date2num(stormTors['UTC_time']))
        f = interp1d(mdates.date2num(stormdict['date']),stormdict['lat'])
        interp_clat = f(mdates.date2num(stormTors['UTC_time']))
        
        stormTors = stormTors.assign(xdist_s = [great_circle((.5*(lat1+lat2),lon1),(.5*(lat1+lat2),lon2)).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['slat'],stormTors['slon'])])
        stormTors = stormTors.assign(ydist_s = [great_circle((lat1,.5*(lon1+lon2)),(lat2,.5*(lon1+lon2))).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['slat'],stormTors['slon'])])

        stormTors = stormTors.assign(xdist_e = [great_circle((.5*(lat1+lat2),lon1),(.5*(lat1+lat2),lon2)).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['elat'],stormTors['elon'])])
        stormTors = stormTors.assign(ydist_e = [great_circle((lat1,.5*(lon1+lon2)),(lat2,.5*(lon1+lon2))).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['elat'],stormTors['elon'])])
        
        self.stormTors = stormTors[stormTors['xdist_s']**2 + stormTors['ydist_s']**2 < dist_thresh**2]
        return self.stormTors

    def rotateToHeading(self,storm):
        
        r"""
        Rotate tornado tracks to their position relative to the heading of the TC at the time.
        
        To be called after getTCtors.
        """
        
        self.stormTors = self.getTCtors(storm)
        
        dx = np.gradient(self.stormdict['lon'])
        dy = np.gradient(self.stormdict['lat'])
        
        f = interp1d(mdates.date2num(self.stormdict['date']),dx)
        interp_dx = f(mdates.date2num(self.stormTors['UTC_time']))
        f = interp1d(mdates.date2num(self.stormdict['date']),dy)
        interp_dy = f(mdates.date2num(self.stormTors['UTC_time']))
        
        ds = np.hypot(interp_dx,interp_dy)
        
        # Rotation matrix for +x pointing 90deg right of storm heading
        rot = np.array([[interp_dy,-interp_dx],[interp_dx,interp_dy]])/ds
        
        oldvec_s = np.array([self.stormTors['xdist_s'],self.stormTors['ydist_s']])
        newvec_s = [np.dot(rot[:,:,i],v) for i,v in enumerate(oldvec_s.T)]
        
        oldvec_e = np.array([self.stormTors['xdist_e'],self.stormTors['ydist_e']])
        newvec_e = [np.dot(rot[:,:,i],v) for i,v in enumerate(oldvec_e.T)]
        
        self.stormTors['rot_xdist_s'] = [v[0] for v in newvec_s]
        self.stormTors['rot_xdist_e'] = [v[0] for v in newvec_e]
        self.stormTors['rot_ydist_s'] = [v[1] for v in newvec_s]
        self.stormTors['rot_ydist_e'] = [v[1] for v in newvec_e]
        

    def makePolarPlot(self):
        
        r"""
        Plot tracks of tornadoes relative to the heading of the TC at the time, in the +y direction.
        """
        
        plt.figure(figsize=(7,7))
        ax = plt.subplot()
        for _,row in self.stormTors.iterrows():
            plt.plot([row['rot_xdist_s'],row['rot_xdist_e']+.01],[row['rot_ydist_s'],row['rot_ydist_e']+.01],lw=2,c='r')
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(self.dist_thresh * np.cos(an), self.dist_thresh * np.sin(an),'k')
        ax.plot([-self.dist_thresh,self.dist_thresh],[0,0],'k--',lw=.5)
        ax.plot([0,0],[-self.dist_thresh,self.dist_thresh],'k--',lw=.5)
        plt.arrow(0, -self.dist_thresh*.1, 0, self.dist_thresh*.2, length_includes_head=True,
          head_width=45, head_length=45,fc='k')
        ax.set_aspect('equal', 'box')
        plt.savefig(''.join([str(i) for i in stormtuple])+'polartors.png')


    #PLOT FUNCTION FOR TORNADOES
    def plot_tors(self,tor_info,zoom="conus",plotPPF=False,ax=None,return_ax=False,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of tornado tracks and PPF.
        
        Parameters
        ----------
        tor_info : pandas.DataFrame or dict, or datetime or list of start/end datetimes
            Requested tornadoes
        zoom : str
            Zoom for the plot. Can be one of the following:
            * **dynamic** - default. Dynamically focuses the domain using the tornado track(s) plotted.
            * **conus** - Contiguous United States
            * **east_conus** - Eastern CONUS
            * **lonW/lonE/latS/latN** - Custom plot domain
        plotPPF : bool or str
            * **False** - no PPF plot
            * **True** - defaults to "total"
            * **total** - probability of a tornado within 25mi of a point during the period of time selected.
            * **daily** - average probability of a tornado within 25mi of a point during a day starting at 12 UTC.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of tornado tracks.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Get plot data
        
        if isinstance(tor_info,pd.core.frame.DataFrame):
            dfTors = tor_info
        elif isinstance(tor_info,dict):
            dfTors = pd.DataFrame.from_dict(tor_info)
        else:
            dfTors = self.__getTimeTors(tor_info)
        
        #Create instance of plot object
        self.plot_obj = TornadoPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot tornadoes
        plot_info = self.plot_obj.plot_tornadoes(dfTors,zoom,plotPPF,ax,return_ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax==True:
            return plot_info[0],plot_info[1],plot_info[2]

    def __getTimeTors(self,time):
        
        if isinstance(time,list):
            t1=min(time)
            t2=max(time)
        else:
            t1 = time.replace(hour=12)
            t2 = t1+timedelta(hours=24)
        subTors = self.Tors.loc[(self.Tors['UTC_time']>=t1) & \
                               (self.Tors['UTC_time']<t2)]
        return subTors


