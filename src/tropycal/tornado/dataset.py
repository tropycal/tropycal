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
    warn_message = "Warning: Cartopy is not installed in your python environment. Plotting functions will not work."
    warnings.warn(warn_message)

from .plot import TornadoPlot
from .tools import *
#from ...tropycal import tracks

class TornadoDataset():
    
    r"""
    Creates an instance of a TornadoDataset object containing tornado data.

    Parameters
    ----------
    mag_thresh : int
        Minimum threshold for tornado rating.
    tornado_path : str
        Source to read tornado data from. Default is "spc", which reads from the online Storm Prediction Center (SPC) 1950-present tornado database. Can change this to a local file.

    Returns
    -------
    TornadoDataset
        An instance of TornadoDataset.
    """

    def __init__(self, mag_thresh=0, len_thresh=0, tornado_path='spc'):
        
        #Error check
        if isinstance(mag_thresh,int) == False:
            raise TypeError("mag_thresh must be of type int.")
        elif mag_thresh not in [0,1,2,3,4,5]:
            raise ValueError("mag_thresh must be between 0 and 5.")
        
        #Read in tornado dataset
        timer_start = dt.now()
        yrnow = timer_start.year
        print(f'--> Starting to read in tornado track data')
        if tornado_path == 'spc':
            try:
                yrlast = yrnow-1
                Tors = pd.read_csv(f'https://www.spc.noaa.gov/wcm/data/1950-{yrlast}_actual_tornadoes.csv',\
                                   error_bad_lines=False,parse_dates=[['mo','dy','yr','time']])
            except:
                yrlast = yrnow-2
                Tors = pd.read_csv(f'https://www.spc.noaa.gov/wcm/data/1950-{yrlast}_actual_tornadoes.csv',\
                                   error_bad_lines=False,parse_dates=[['mo','dy','yr','time']])
            print(f'--> Completed reading in tornado data for 1950-{yrlast} (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        else:
            Tors = pd.read_csv(tornado_path,\
                               error_bad_lines=False,parse_dates=[['mo','dy','yr','time']])
            print(f'--> Completed reading in tornado data from local file (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        
        #Get UTC from timezone (most are 3 = CST, but some 0 and 9 = GMT)
        tz = np.array([timedelta(hours=9-int(i)) for i in Tors['tz']])
        tors_dt = [pd.to_datetime(i) for i in Tors['mo_dy_yr_time']]
        Tors = Tors.assign(UTC_time = tors_dt+tz)
        Tors = Tors.assign(UTC_year = [t.year for t in tors_dt+tz])
        Tors = Tors.assign(SPC_time = Tors['UTC_time'] - timedelta(hours=12))
        
        #Filter for only those tors at least F/EF scale mag_thresh.
        Tors = Tors[Tors['mag']>=mag_thresh]

        #Filter for only those tors at least F/EF scale mag_thresh.
        Tors = Tors[Tors['len']>=len_thresh]
        
        #Clean up lat/lons       
        Tors = Tors[(Tors['slat']!=0)|(Tors['slon']!=0)]
        Tors = Tors[(Tors['slat']>=20) & (Tors['slat']<=50)]
        Tors = Tors[(Tors['slon']>=-130) & (Tors['slon']<=-65)]
        Tors = Tors[(Tors['elat']>=20)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elat']<=50)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elon']>=-130)|(Tors['elat']==0)]
        Tors = Tors[(Tors['elon']<=-65)|(Tors['elat']==0)]
        Tors = Tors.assign(elat = [Tors['slat'].values[u] if i==0 else i for u, i in enumerate(Tors['elat'].values)])
        Tors = Tors.assign(elon = [Tors['slon'].values[u] if i==0 else i for u, i in enumerate(Tors['elon'].values)])
                
        self.Tors = Tors
        
    def get_storm_tornadoes(self,storm,dist_thresh):
        
        r"""
        Retrieves all tornado tracks that occur along the track of a tropical cyclone.
        
        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Instance of a Storm object.
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC.
        
        Returns
        -------
        pandas.DataFrame
            Pandas DataFrame object containing data about the tornadoes associated with this tropical cyclone.
        """
        
        #Get storm dict from object
        stormdict = storm.to_dict()
    
        stormTors = self.Tors[(self.Tors['UTC_time']>=min(stormdict['date'])) & \
                         (self.Tors['UTC_time']<=max(stormdict['date']))]
        
        #Interpolate storm track time to the time of each tornado
        f = interp1d(mdates.date2num(stormdict['date']),stormdict['lon'])
        interp_clon = f(mdates.date2num(stormTors['UTC_time']))
        f = interp1d(mdates.date2num(stormdict['date']),stormdict['lat'])
        interp_clat = f(mdates.date2num(stormTors['UTC_time']))
        
        #Retrieve x&y distance of each tornado from TC center
        stormTors = stormTors.assign(xdist_s = [great_circle((.5*(lat1+lat2),lon1),(.5*(lat1+lat2),lon2)).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['slat'],stormTors['slon'])])
        stormTors = stormTors.assign(ydist_s = [great_circle((lat1,.5*(lon1+lon2)),(lat2,.5*(lon1+lon2))).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['slat'],stormTors['slon'])])

        stormTors = stormTors.assign(xdist_e = [great_circle((.5*(lat1+lat2),lon1),(.5*(lat1+lat2),lon2)).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['elat'],stormTors['elon'])])
        stormTors = stormTors.assign(ydist_e = [great_circle((lat1,.5*(lon1+lon2)),(lat2,.5*(lon1+lon2))).kilometers \
                 for lat1,lon1,lat2,lon2 in zip(interp_clat,interp_clon,stormTors['elat'],stormTors['elon'])])
        
        #Assign tornado within specified distance threshold to this storm
        stormTors = stormTors[stormTors['xdist_s']**2 + stormTors['ydist_s']**2 < dist_thresh**2]
        
        #Return DataFrame
        return stormTors

    def rotateToHeading(self,storm,stormTors):
        
        r"""
        Rotate tornado tracks to their position relative to the heading of the TC at the time.
        
        Parameters
        ----------
        stormTors : pandas.DataFrame
            Pandas DataFrame containing tornado tracks.
        
        Returns
        -------
        pandas.DataFrame
            StormTors modified to include motion relative coordinates.
        """
        
        #Check to make sure there's enough tornadoes
        if len(stormTors) == 0:
            stormTors['rot_xdist_s'] = []
            stormTors['rot_xdist_e'] = []
            stormTors['rot_ydist_s'] = []
            stormTors['rot_ydist_e'] = []
            return stormTors
        
        #Get storm dict from object
        stormdict = storm.to_dict()
        
        #Temporal interpolation of storm track
        dx = np.gradient(stormdict['lon'])
        dy = np.gradient(stormdict['lat'])
        
        f = interp1d(mdates.date2num(stormdict['date']),dx)
        interp_dx = f(mdates.date2num(stormTors['UTC_time']))
        f = interp1d(mdates.date2num(stormdict['date']),dy)
        interp_dy = f(mdates.date2num(stormTors['UTC_time']))
        
        ds = np.hypot(interp_dx,interp_dy)
        
        # Rotation matrix for +x pointing 90deg right of storm heading
        ds[ds == 0.0] = ds[ds == 0.0] + 0.01 #avoid warnings for divide by zero
        rot = np.array([[interp_dy,-interp_dx],[interp_dx,interp_dy]])/ds
        
        oldvec_s = np.array([stormTors['xdist_s'].values,stormTors['ydist_s'].values])
        newvec_s = [np.dot(rot[:,:,i],v) for i,v in enumerate(oldvec_s.T)]
        
        oldvec_e = np.array([stormTors['xdist_e'].values,stormTors['ydist_e'].values])
        newvec_e = [np.dot(rot[:,:,i],v) for i,v in enumerate(oldvec_e.T)]
        
        #Enter motion relative coordinates into stormTors dict
        stormTors['rot_xdist_s'] = [v[0] for v in newvec_s]
        stormTors['rot_xdist_e'] = [v[0] for v in newvec_e]
        stormTors['rot_ydist_s'] = [v[1] for v in newvec_s]
        stormTors['rot_ydist_e'] = [v[1] for v in newvec_e]
        
        #return modified stormtors
        return stormTors
        

    def plot_TCtors_rotated(self,storm,dist_thresh=1000,return_ax=False):
        
        r"""
        Plot tracks of tornadoes relative to the storm motion vector of the tropical cyclone.
        
        Parameters
        ----------
        storm : tropycal.tracks.Storm
            Instance of a Storm object.
        dist_thresh : int
            Distance threshold (in kilometers) from the tropical cyclone track over which to attribute tornadoes to the TC.
        return_ax : bool
            Whether to return the axis plotted. Default is False.
        
        Notes
        -----
        The motion vector is oriented upwards (in the +y direction).
        """
        
        #Retrieve tornadoes for the requested storm
        try:
            stormTors = storm.StormTors
        except:
            stormTors = self.get_storm_tornadoes(storm,dist_thresh)
        
        #Add motion vector relative coordinates
        stormTors = self.rotateToHeading(storm,stormTors)
        
        #Create figure for plotting
        plt.figure(figsize=(9,9),dpi=150)
        ax = plt.subplot()
        
        #Default EF color scale
        EFcolors = ef_colors('default')
        
        #Plot all tornado tracks in motion relative coords
        for _,row in stormTors.iterrows():
            plt.plot([row['rot_xdist_s'],row['rot_xdist_e']+.01],[row['rot_ydist_s'],row['rot_ydist_e']+.01],\
                     lw=2,c=EFcolors[row['mag']])
            
        #Plot dist_thresh radius
        ax.set_facecolor('#F6F6F6')
        circle = plt.Circle((0,0), dist_thresh, color='w')
        ax.add_artist(circle)
        an = np.linspace(0, 2 * np.pi, 100)
        ax.plot(dist_thresh * np.cos(an), dist_thresh * np.sin(an),'k')
        ax.plot([-dist_thresh,dist_thresh],[0,0],'k--',lw=.5)
        ax.plot([0,0],[-dist_thresh,dist_thresh],'k--',lw=.5)
        
        #Plot motion vector
        plt.arrow(0, -dist_thresh*.1, 0, dist_thresh*.2, length_includes_head=True,
          head_width=45, head_length=45,fc='k',lw=2)
        
        #Labels
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('Left/Right of Storm Heading (km)',fontsize=13)
        ax.set_ylabel('Behind/Ahead of Storm Heading (km)',fontsize=13)
        ax.set_title(f'{storm.name} {storm.year} tornadoes relative to heading',fontsize=17)
        ax.tick_params(axis='both', which='major', labelsize=11.5)
        
        #Add legend
        handles=[]
        for ef,color in enumerate(EFcolors):
            count = len(stormTors[stormTors['mag']==ef])
            handles.append(mlines.Line2D([], [], linestyle='-',color=color,label=f'EF-{ef} ({count})'))
        ax.legend(handles=handles,loc='lower left',fontsize=11.5)
        
        #Return axis or show figure
        if return_ax == True:
            return ax
        else:
            plt.show()
            plt.close()
        

    def plot_tors(self,tor_info,zoom="conus",plotPPF=False,\
                  ax=None,return_ax=False,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of tornado tracks and Practically Perfect Forecast (PPF).
        
        Parameters
        ----------
        tor_info : pandas.DataFrame / dict / datetime.datetime / list
            Requested tornadoes to plot. Can be one of the following:
            
            * **Pandas DataFrame** containing the requested tornadoes to plot.
            * **dict** entry containing the requested tornadoes to plot.
            * **datetime.datetime** object for a single day to plot tornadoes.
            * **list** with 2 datetime.datetime entries, a start date and end date for plotting over a range of dates.
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the tornado track(s) plotted.
            * **conus** - Contiguous United States
            * **east_conus** - Eastern CONUS
            * **lonW/lonE/latS/latN** - Custom plot domain
        plotPPF : bool or str
            Whether to plot practically perfect forecast (PPF). True defaults to "daily". Default is False.
        
            * **False** - no PPF plot.
            * **True** - defaults to "daily".
            * **"total"** - probability of a tornado within 25mi of a point during the period of time selected.
            * **"daily"** - average probability of a tornado within 25mi of a point during a day starting at 12 UTC.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of tornado tracks.
        map_prop : dict
            Property of cartopy map.
        """
        
        prop = kwargs.pop("prop",{})
        map_prop = kwargs.pop("map_prop",{})
        
        #Get plot data
        
        if isinstance(tor_info,pd.core.frame.DataFrame):
            dfTors = tor_info
        elif isinstance(tor_info,dict):
            dfTors = pd.DataFrame.from_dict(tor_info)
        else:
            dfTors = self.__getTimeTors(tor_info)
            if isinstance(tor_info,list):
                try:
                    if prop['PPFcolors']=='SPC':
                        warning_message = 'SPC colors only allowed for daily PPF. Defaulting to plasma colormap.'
                        warnings.warn(warning_message)
                        prop['PPFcolors']='plasma'
                except:
                    warning_message = 'SPC colors only allowed for daily PPF. Defaulting to plasma colormap.'
                    warnings.warn(warning_message)
                    prop['PPFcolors']='plasma'
                    
                if plotPPF!='total':
                    try:
                        prop['PPFlevels']
                    except:
                        t_int = (max(tor_info)-min(tor_info)).days
                        if t_int>1:
                            new_levs=[i*t_int**-.7 \
                                for i in [2,5,10,15,30,45,60,100]]
                            for i,_ in enumerate(new_levs[:-1]):
                                new_levs[i] = max([new_levs[i],0.1])
                                new_levs[i+1] = new_levs[i]+max([new_levs[i+1]-new_levs[i],.1])
                            prop['PPFlevels']=new_levs
    
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


    def plot_climo(self,date_range,year_range,zoom="conus",plotPPF='Daily',\
                  ax=None,return_ax=False,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of tornado tracks and Practically Perfect Forecast (PPF).
        
        Parameters
        ----------
        tor_info : pandas.DataFrame / dict / datetime.datetime / list
            Requested tornadoes to plot. Can be one of the following:
            
            * **Pandas DataFrame** containing the requested tornadoes to plot.
            * **dict** entry containing the requested tornadoes to plot.
            * **datetime.datetime** object for a single day to plot tornadoes.
            * **list** with 2 datetime.datetime entries, a start date and end date for plotting over a range of dates.
        zoom : str
            Zoom for the plot. Can be one of the following:
            
            * **dynamic** - default. Dynamically focuses the domain using the tornado track(s) plotted.
            * **conus** - Contiguous United States
            * **east_conus** - Eastern CONUS
            * **lonW/lonE/latS/latN** - Custom plot domain
        plotPPF : bool or str
            Whether to plot practically perfect forecast (PPF). True defaults to "daily". Default is False.
        
            * **False** - no PPF plot.
            * **True** - defaults to "daily".
            * **"total"** - probability of a tornado within 25mi of a point during the period of time selected.
            * **"daily"** - average probability of a tornado within 25mi of a point during a day starting at 12 UTC.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of tornado tracks.
        map_prop : dict
            Property of cartopy map.
        """
        
        prop = kwargs.pop("prop",{})
        map_prop = kwargs.pop("map_prop",{})
        
        #Get plot data
        
        climo = {}
        climo['date_range']=date_range
        climo['year_range']=year_range
        dfTors = self.__getClimo(date_range,year_range)
        if plotPPF!='maximum':
            warning_message = 'SPC colors only allowed for daily PPF. Defaulting to plasma colormap.'
            warnings.warn(warning_message)
            prop['PPFcolors']='plasma'
        if plotPPF!='maximum':
            try:
                prop['PPFlevels']
            except:
                t_int = 50
                if t_int>1:
                    new_levs=[i*t_int**-.7 \
                        for i in [2,5,10,15,30,45,60,100]]
                    for i,_ in enumerate(new_levs[:-1]):
                        new_levs[i] = max([new_levs[i],0.1])
                        new_levs[i+1] = new_levs[i]+max([new_levs[i+1]-new_levs[i],.1])
                    prop['PPFlevels']=new_levs

        #Create instance of plot object
        self.plot_obj = TornadoPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot tornadoes
        plot_info = self.plot_obj.plot_tornadoes(dfTors,zoom,plotPPF,ax,return_ax,climo=climo,prop=prop,map_prop=map_prop)
        
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

    def __getClimo(self,date_range,year_range):
        
        year_min,year_max = year_range
        date_min,date_max = [dt.strptime(i,'%m/%d') for i in date_range]
        date_min += timedelta(hours=12)
        date_max += timedelta(days=1,hours=12,seconds=-1)
        date_max = date_max.replace(year=date_min.year)
        
        def date_range_test(t,t_min,t_max,yr_min,yr_max):
            if t_min<t_max:
                test1 = (t>=t_min.replace(year=t.year))
                test2 = (t<=t_max.replace(year=t.year))
                return (test1 & test2) & (t.year>=yr_min) & (t.year<=yr_max)
            else:
                test1 = (t_min.replace(year=t.year)<=t<dt(t.year+1,1,1))
                test2 = (dt(t.year,1,1)<=t<=t_max.replace(year=t.year))
                return (test1 | test2) & (t.year>=yr_min) & (t.year<=yr_max)
        
        TorDateTest = np.array([date_range_test(t,date_min,date_max,year_min,year_max) for t in self.Tors['UTC_time']])
        subTors = self.Tors.loc[TorDateTest]
        return subTors

