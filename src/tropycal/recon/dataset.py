import os
import sys
import numpy as np
from datetime import datetime as dt,timedelta
import pandas as pd
import requests
import pickle
import copy

from scipy.interpolate import interp1d,splrep,splev
from scipy.ndimage import gaussian_filter as gfilt,gaussian_filter1d as gfilt1d
from scipy.ndimage.filters import minimum_filter
import matplotlib.dates as mdates

try:
    import matplotlib as mlib
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.gridspec as gridspec
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

from .plot import ReconPlot
from ..tracks.plot import TrackPlot

#Import tools
from .tools import *
from ..utils import *

class ReconDataset:

    r"""
    Creates an instance of a ReconDataset object containing all recon data for a single storm.
    
    Parameters
    ----------
    stormtuple : tuple or list
        Requested storm. Can be either tuple or list containing storm name and year (e.g., ("Matthew",2016)).
    save_path : str, optional
        Filepath to save recon data in. Recommended in order to avoid having to re-read in the data.
    read_path : str, optional
        Filepath to read saved recon data from. If specified, "save_path" cannot be passed as an argument.
        
    Returns
    -------
    Dataset
        An instance of ReconDataset, initialized with the following:
        
        * **missiondata** - A dictionary of missions.
            Each entry is a dateframe from a single mission.
            Dictionary keys are given by mission number and agency (e.g. '15_NOAA').
        * **recentered** - A dataframe with all missions concatenated together, and columns 'xdist' and 'ydist'
            indicating the distance (km) of the ob from the interpolated center of the storm.
    
    Notes
    -----
    Recon data is currently read in via Tropical Atlantic. Future releases of Tropycal will incorporate NHC recon archives.
    """

    def __repr__(self):
        
        info = []
        for name in ['hdobs','dropsondes','vdms']:
            try:
                info.append(self.__dict__[name].__repr__())
            except:
                info.append('')
        return '\n'.join(info)
    
    def __init__(self, storm):
        
        self.source = 'https://www.nhc.noaa.gov/archive/recon/'
        self.storm = storm

    def get_hdobs(self,data=None):
        self.hdobs = hdobs(self.storm,data)
    
    def get_dropsondes(self,data=None):
        self.dropsondes = dropsondes(self.storm,data)
        
    def get_vdms(self,data=None):
        self.vdms = vdms(self.storm,data)
        
    def update(self):
        for name in ['hdobs','dropsondes','vdms']:
            try:
                self.__dict__[name].update()
            except:
                print(f'No {name} object to update')

    def get_track(self,time=None):
        
        r"""
        Creates model for finding recon+btk track at a given time.
        
        Parameters
        ----------
        time : datetime.datetime or list
            Datetime object or list of datetime objects representing the time of the requested mission.
        
        Returns
        -------
        tuple
            (lon,lat) coordinates.
        """
        
        #Get track best fit model
        if False:
            btk = self.storm.to_dataframe()[['date','lon','lat']].rename(columns={'date':'time'})
            try:
                recon = pd.DataFrame([{k:d[k] for k in ('time','lon','lat')} for d in self.vdms])
            else:
                recon = self.sel(iscenter=1).data
            except:
                recon = None
          
            #Interpolate center position to time of each ob
            f1 = interp1d(mdates.date2num(btk['time']),btk['lon'],fill_value='extrapolate',kind='quadratic')
            ibtk_lon = f1(mdates.date2num(hdob['time']))
            f2 = interp1d(mdates.date2num(btk['time']),btk['lat'],fill_value='extrapolate',kind='quadratic')
            ibtk_lat = f2(mdates.date2num(hdob['time']))
            dist = [great_circle( (ibtk_lat[i],ibtk_lon[i]), \
                (ilat,ilon) ).kilometers for i,(ilat,ilon) in enumerate(zip(hdob['lat'],hdob['lon']))]

            hdob_lat = [l for l,d in zip(hdob['lat'],dist) if d<30]
            hdob_lon = [l for l,d in zip(hdob['lon'],dist) if d<30]
            hdob_time = [l for l,d in zip(hdob['time'],dist) if d<30]
            btk_lat = self.storm.dict['lat']
            btk_lon = self.storm.dict['lon']
            btk_time = self.storm.dict['date']

            oldtimes = [mdates.date2num(t) for t in hdob_time*2+btk_time]
            oldlons = hdob_lon*2+btk_lon
            oldlats = hdob_lat*2+btk_lat

            inlons = sorted([(t,l) for t,l in zip(oldtimes,oldlons)], key=lambda x: x[0])
            inlats = sorted([(t,l) for t,l in zip(oldtimes,oldlats)], key=lambda x: x[0])

            self.tck_lon = splrep([i[0] for i in inlons], [i[1] for i in inlons], s=.5)
            self.tck_lat = splrep([i[0] for i in inlats], [i[1] for i in inlats], s=.5)
        
        if time is not None:
            if isinstance(time,list):
                time = np.array(time)
            time = mdates.date2num(np.array(time))
            lonnew = splev(time, self.tck_lon, der=0)
            latnew = splev(time, self.tck_lat, der=0)
            return (latnew,lonnew)
        
    def center_relative(self):
        
        r"""
        Calculates center relative coordinates based on recon-btk track.
        """ 
        
        if 'track' not in self.__dict__.keys():
            self.get_track()
        
        for name in ['hdobs','dropsondes','vdms']:
            try:
                self.__dict__[name]._recenter()
            except:
                print(f'No {name} object to recenter')
                
    def find_mission(self,time,distance=None):
        
        r"""
        Returns the name of a mission or list of missions given a specified time.
        
        Parameters
        ----------
        time : datetime.datetime or list
            Datetime object or list of datetime objects representing the time of the requested mission.
        
        Returns
        -------
        list
            The names of any/all missions that had in-storm observations during the specified time.
        """
        
        if isinstance(time,list):
            t1=min(time)
            t2=max(time)
        else:
            t1 = t2 = time
        selected=[]
        if distance is None:
            data = self.hdobs.data
        else:
            data = self.hdobs.sel(distance=distance)

        mission_groups = data.groupby('mission')
        for g in mission_groups:
            t_start,t_end = (min(g[1]['time']),max(g[1]['time']))
            if t_start<=t1<=t_end or t_start<=t2<=t_end or t1<t_start<t2:
                selected.append(g[0])
        if len(selected)==0:
            msg = 'There were no recon missions during this time'
            if distance is not None:
                msg += f' within {distance} km of the storm'
            print(msg)
        else:
            return selected

    def plot_summary(self,mission=None):
        
        r"""
        Plot summary map of all recon data.
        
        Parameters
        ----------
        mission : string with mission name
            Will plot summary for the specified mission.
        """
        
        prop = {'hdobs':{'ms':5,'marker':'o'},\
                'dropsondes':{'ms':25,'marker':'v'},\
                'vdms':{'ms':100,'marker':'s'}}
        
        hdobs = self.hdobs.sel(mission=mission)
        dropsondes = self.dropsondes.sel(mission=mission)
        vdms = self.vdms.sel(mission=mission)
        
        ax,domain = hdobs.plot_points('pkwnd',\
                            prop={'cmap':{1:'firebrick',2:'tomato',4:'gold',6:'lemonchiffon'},'levels':(0,200),'ms':2})
        ax.scatter(*zip(*[(d['TOPlon'],d['TOPlat']) \
                        for d in dropsondes.data]),s=50,marker='v',edgecolor='w',linewidth=0.5,color='darkblue')
        ax.scatter(*zip(*[(d['lon'],d['lat']) for d in vdms.data]),s=80,marker='H',edgecolor='w',linewidth=1,\
              c=[d['Minimum Sea Level Pressure (hPa)'] for d in vdms.data],vmin=850,vmax=1020,cmap='Blues')
        
        title_left = ax.get_title(loc='left').split('\n')
        newtitle = title_left[0]+'\nRecon summary'+['',f' for mission {mission}'][mission is not None]
        ax.set_title(newtitle,fontsize=17,fontweight='bold',loc='left')
        
        return ax
    

class hdobs:

    r"""
    Creates an instance of a ReconDataset object containing all recon high density observations for a single storm.
    
    Parameters
    ----------
    stormtuple : tuple or list
        Requested storm. Can be either tuple or list containing storm name and year (e.g., ("Matthew",2016)).
    save_path : str, optional
        Filepath to save recon data in. Recommended in order to avoid having to re-read in the data.
    read_path : str, optional
        Filepath to read saved recon data from. If specified, "save_path" cannot be passed as an argument.
        
    Returns
    -------
    Dataset
        An instance of ReconDataset, initialized with a dataframe of HDOB
    """

    def __repr__(self):
         
        summary = ["<tropycal.recon.hdobs>"]
        
        #Find maximum wind and minimum pressure
        max_wspd = np.nanmax(self.data['wspd'])
        max_pkwnd = np.nanmax(self.data['pkwnd'])
        max_sfmr = np.nanmax(self.data['sfmr'])
        min_psfc = np.nanmin(self.data['p_sfc'])
        time_range = [pd.to_datetime(t) for t in (np.nanmin(self.data['time']),np.nanmax(self.data['time']))]

        #Add general summary
        emdash = '\u2014'
        summary_keys = {'Storm':f'{self.storm.name} {self.storm.year}',\
                        'Missions':len(set(self.data['mission'])),
                        'Time range':f"{time_range[0]:%b-%d %H:%M} {emdash} {time_range[1]:%b-%d %H:%M}",
                        'Max 30sec flight level wind':f"{max_wspd} knots",
                        'Max 10sec flight level wind':f"{max_pkwnd} knots",
                        'Max SFMR wind':f"{max_sfmr} knots",
                        'Min surface pressure':f"{min_psfc} hPa"}

        #Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')
        
        return "\n".join(summary)
    
    def __init__(self, storm, data=None, update=False):

        self.storm = storm
        self.archiveURL = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/AHONT1/'
        self.data = None

        if isinstance(data,str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        elif data is not None:
            self.data = data
        
        if data is None or update:
            try:
                start_time = max(self.data['time'])
            except:
                start_time = min(self.storm.dict['date'])-timedelta(days=1)
            end_time = max(self.storm.dict['date'])+timedelta(days=1)

            timestr = [f'{start_time:%Y%m%d}']+\
                        [f'{t:%Y%m%d}' for t in self.storm.dict['date'] if t>start_time]+\
                        [f'{end_time:%Y%m%d}']
            archive = pd.read_html(self.archiveURL)[0]
            linktimes = sorted([l.split('.') for l in archive['Name'] if isinstance(l,str) and 'txt' in l],key=lambda x: x[1])
            linksub = [self.archiveURL+'.'.join(l) for l in linktimes if l[1][:8] in timestr]
            timer_start = dt.now()
            print(f'Searching through recon HDOB files between {timestr[0]} and {timestr[-1]} ...')
            unreadable = 0
            for link in linksub:
                content = requests.get(link).text
                missionname = [i.split() for i in content.split('\n')][3][1]
                if missionname[2:5] == self.storm.id[2:4]+self.storm.id[0]:
                    try:
                        tmp = self._decode_hdob(content)
                    except:
                        unreadable+=1
                    if self.data is None:
                        self.data = copy.copy(tmp)
                    elif max(tmp['time'])>start_time:
                        self.data = pd.concat([self.data,tmp])
                    else:
                        pass
            print(f'--> Completed reading in recon HDOB files ({(dt.now()-timer_start).total_seconds():.1f} seconds)'+\
                  f'\nUnable to decode {unreadable} files')
            #self.data = self._find_centers()
            #self._get_track()
            #self.data = self._recenter()

        self.keys = list(self.data.keys())

    def update(self):
        newobj = hdobs(storm=self.storm,data=self.data,update=True)
        return newobj
    
    def _decode_hdob(self,content):
        tmp = [i.split() for i in content.split('\n')]
        tmp = [i for j,i in enumerate(tmp) if len(i)>0]
        items = []
        for j,i in enumerate(tmp):
            if j<=3:
                items.append(i)
            if j>3 and i[0][0].isdigit():
                items.append(i)
            
        missionname = items[2][1]
        data = {}
        data['time'] = [dt.strptime(items[2][-1]+i[0],'%Y%m%d%H%M%S') for i in items[3:]]
        if data['time'][0].hour>12 and data['time'][-1].hour<12:
            data['time'] = [t+timedelta(days=[0,1][t.hour<12]) for t in data['time']]

        data['lat'] = [np.nan if '/' in i[1] else round((float(i[1][:-3])+float(i[1][-3:-1])/60)*[-1,1][i[1][-1]=='N'],2) \
                       for i in items[3:]]
        data['lon'] = [np.nan if '/' in i[2] else round((float(i[2][:-3])+float(i[2][-3:-1])/60)*[-1,1][i[2][-1]=='E'],2) \
                       for i in items[3:]]
        data['plane_p'] = [np.nan if '/' in i[3] else round(float(i[3])*0.1+[0,1000][float(i[3])<1000],1) for i in items[3:]]
        data['plane_z'] = [np.nan if '/' in i[4] else round(float(i[4]),0) for i in items[3:]]
        data['p_sfc'] = [np.nan if (('/' in i[5]) | (p<550)) \
                         else round(float(i[5])*0.1+[0,1000][float(i[5])<1000],1) for i,p in zip(items[3:],data['plane_p'])]
        data['temp'] = [np.nan if '/' in i[6] else round(float(i[6])*0.1,1) for i in items[3:]]
        data['dwpt'] = [np.nan if '/' in i[7] else round(float(i[7])*0.1,1) for i in items[3:]]
        data['wdir'] = [np.nan if '/' in i[8][:3] else round(float(i[8][:3]),0) for i in items[3:]]
        data['wspd'] = [np.nan if '/' in i[8][3:] else round(float(i[8][3:]),0) for i in items[3:]]
        data['pkwnd'] = [np.nan if '/' in i[9] else round(float(i[9]),0) for i in items[3:]]
        data['sfmr'] = [np.nan if '/' in i[10] else round(float(i[10]),0) for i in items[3:]]
        data['rain'] = [np.nan if '/' in i[11] else round(float(i[11]),0) for i in items[3:]]

        data['flag']=[]
        for i in items[3:]:
            flag = []
            if int(i[12][0]) in [1,3]:
                flag.extend(['lat','lon'])
            if int(i[12][0]) in [2,3]:
                flag.extend(['plane_p','plane_z'])
            if int(i[12][1]) in [1,4,5,9]:
                flag.extend(['temp','dwpt'])
            if int(i[12][1]) in [2,4,6,9]:
                flag.extend(['wdir','wspd','pkwnd'])
            if int(i[12][1]) in [3,5,6,9]:
                flag.extend(['sfmr','rain'])
            data['flag'].append(flag)
        
        #QC p_sfc
        if any(abs(np.gradient(data['p_sfc'],np.array(data['time']).astype('datetime64[s]').astype(float)))>1):
            data['p_sfc']=[np.nan]*len(data['p_sfc'])
            data['flag'] = [d.append('p_sfc') for d in data['flag']]

        data['mission'] = [missionname[:2]]*len(data['time'])

        return_data = pd.DataFrame.from_dict(data).reset_index()
        #remove nan's for lat/lon coordinates
        return_data = return_data.dropna(subset=['lat', 'lon'])

        return return_data 
        
    def _find_centers(self,data=None):
        
        if data is None:
            data = self.data
        data = data.sort_values(['mission','time'])
                
        def fill_nan(A):
            #Interpolate to fill nan values
            A = np.array(A)
            inds = np.arange(len(A))
            good = np.where(np.isfinite(A))
            good_grad = np.interp(inds,good[0],np.gradient(good[0]))
            if len(good[0])>=3:
                f = interp1d(inds[good], A[good],bounds_error=False,kind='quadratic')
                B = np.where((np.isfinite(A)[good[0][0]:good[0][-1]+1]) | (good_grad[good[0][0]:good[0][-1]+1]>3),
                             A[good[0][0]:good[0][-1]+1],
                             f(inds[good[0][0]:good[0][-1]+1]))
                return [np.nan]*good[0][0]+list(B)+[np.nan]*(inds[-1]-good[0][-1])
            else:
                return [np.nan]*len(A)

        missiondata = data.groupby('mission')
        dfs = []
        for group in missiondata:
            mdata = group[1]
            #Check that sfc pressure spread is big enough to identify real minima
            if np.nanpercentile(mdata['p_sfc'],95)-np.nanpercentile(mdata['p_sfc'],5)>8:
                p_sfc_interp = fill_nan(mdata['p_sfc']) #Interp p_sfc across missing data
                wspd_interp = fill_nan(mdata['wspd']) #Interp wspd across missing data
                #Smooth p_sfc and wspd
                p_sfc_smooth = [np.nan]*1+list(np.convolve(p_sfc_interp,[1/3]*3,mode='valid'))+[np.nan]*1
                wspd_smooth = [np.nan]*1+list(np.convolve(wspd_interp,[1/3]*3,mode='valid'))+[np.nan]*1
                #Add wspd to p_sfc to encourage finding p mins with wspd mins 
                #and prevent finding p mins in intense thunderstorms
                pw_test = np.array(p_sfc_smooth)+np.array(wspd_smooth)*.1
                #Find mins in 20-minute windows
                imin = np.nonzero(pw_test == minimum_filter(pw_test,40))[0]
                #Only use mins if below 10th %ile of mission p_sfc data and when plane p is 550-950mb
                #and not in takeoff and landing time windows
                plane_p = fill_nan(mdata['plane_p'])
                imin = [i for i in imin if 800<p_sfc_interp[i]<np.nanpercentile(mdata['p_sfc'],10) and \
                        550<plane_p[i]<950 and i>60 and i<len(mdata)-60]
            else:
                imin=[]
            mdata['iscenter'] = np.array([1 if i in imin else 0 for i in range(len(mdata))])
            dfs.append(mdata)

        data = pd.concat(dfs)
        numcenters = sum(data['iscenter'])
        print(f'Found {numcenters} center passes')
        return data
        
    def _recenter(self,use='btk'): 
        data = self.sel(iscenter=1).data.sort_values(by='time').reset_index(drop=True)
          
        #Interpolate center position to time of each ob
        interp_clat,interp_clon = self._get_track(data['time'])

        #Get x,y distance of each ob from coinciding interped center position
        data['xdist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
            (interp_clat[i],data['lon'][i]) ).kilometers* \
            [1,-1][int(data['lon'][i] < interp_clon[i])] for i in range(len(data))]
        data['ydist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
            (data['lat'][i],interp_clon[i]) ).kilometers* \
            [1,-1][int(data['lat'][i] < interp_clat[i])] for i in range(len(data))]
        data['distance'] = [(i**2+j**2)**.5 for i,j in zip(data['xdist'],data['ydist'])]

        print('Completed center-relative coordinates')
        
        return data
    
    def sel(self,mission=None,time=None,domain=None,plane_p=None,plane_z=None,p_sfc=None,\
            temp=None,dwpt=None,wdir=None,wspd=None,pkwnd=None,sfmr=None,noflag=None,\
            iscenter=None,distance=None):
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

        #Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission']==mission]

        #Apply time filter
        if time is not None:
            bounds = get_bounds(NEW_DATA['time'],time)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['time']>bounds[0]) & (NEW_DATA['time']<bounds[1])]
        
        #Apply domain filter
        if domain is not None:
            tmp = {k[0].lower():v for k,v in domain.items()}
            domain = {'n':90,'s':-90,'e':359.99,'w':0}
            domain.update(tmp)
            bounds = get_bounds(NEW_DATA['lon']%360,(domain['w']%360,domain['e']%360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon']%360>=bounds[0]) & (NEW_DATA['lon']%360<=bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'],(domain['s'],domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat']>=bounds[0]) & (NEW_DATA['lat']<=bounds[1])]
        
        #Apply flight pressure filter
        if plane_p is not None:
            bounds = get_bounds(NEW_DATA['plane_p'],plane_p)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['plane_p']>bounds[0]) & (NEW_DATA['plane_p']<bounds[1])]
            
        #Apply flight height filter
        if plane_z is not None:
            bounds = get_bounds(NEW_DATA['plane_z'],plane_z)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['plane_z']>bounds[0]) & (NEW_DATA['plane_z']<bounds[1])]
        
        #Apply surface pressure filter
        if p_sfc is not None:
            bounds = get_bounds(NEW_DATA['p_sfc'],p_sfc)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['p_sfc']>bounds[0]) & (NEW_DATA['p_sfc']<bounds[1])]
            
        #Apply temperature filter
        if temp is not None:
            bounds = get_bounds(NEW_DATA['temp'],temp)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['temp']>bounds[0]) & (NEW_DATA['temp']<bounds[1])]
        
        #Apply dew point filter
        if dwpt is not None:
            bounds = get_bounds(NEW_DATA['dwpt'],dwpt)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['dwpt']>bounds[0]) & (NEW_DATA['dwpt']<bounds[1])]
            
        #Apply wind direction filter
        if wdir is not None:
            bounds = get_bounds(NEW_DATA['wdir'],wdir)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['wdir']>bounds[0]) & (NEW_DATA['wdir']<bounds[1])]
            
        #Apply wind speed filter
        if wspd is not None:
            bounds = get_bounds(NEW_DATA['wspd'],wspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['wspd']>bounds[0]) & (NEW_DATA['wspd']<bounds[1])]
            
        #Apply peak wind filter
        if pkwnd is not None:
            bounds = get_bounds(NEW_DATA['pkwnd'],pkwnd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['pkwnd']>bounds[0]) & (NEW_DATA['pkwnd']<bounds[1])]
            
        #Apply sfmr filter
        if sfmr is not None:
            bounds = get_bounds(NEW_DATA['sfmr'],sfmr)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['sfmr']>bounds[0]) & (NEW_DATA['sfmr']<bounds[1])]
        
        #Apply iscenter filter
        if iscenter is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['iscenter']==iscenter]
        
        #Apply distance filter
        if distance is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['distance']<distance]
        
        NEW_OBJ = hdobs(storm=self.storm,data=NEW_DATA)
        
        return NEW_OBJ
        
    def to_pickle(self,filename):
        r"""
        Save HDOB data (Pandas dataframe) to a pickle file
        
        Parameters
        ----------
        filename : str
            name of file to save pickle file to.
        """
        
        with open(filename,'wb') as f:
            pickle.dump(self.data,f)
    
    def plot_points(self,varname='wspd',domain="dynamic",ax=None,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of recon data points.
        
        Parameters
        ----------
        recon_select : Requested recon data
            pandas.DataFrame or dict,
            or string referencing the mission name (e.g. '12_NOAA'), 
            or datetime or list of start/end datetimes.
        varname : str
            Variable to plot. Can be one of the following keys in recon_select dataframe:
            
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
            Customization properties of recon plot. Please refer to :ref:`options-prop-recon-plot` for available options.
        map_prop : dict
            Customization properties of Cartopy map. Please refer to :ref:`options-map-prop` for available options.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
                
        #Get plot data
        dfRecon = self.data
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_points(self.storm,dfRecon,domain,varname=varname,\
                                              ax=ax,prop=prop,map_prop=map_prop)
        #Return axis
        return plot_info
        
    def plot_hovmoller(self,varname='wspd',radlim=None,track_dict=None,\
                       window=6,align='center',ax=None,**kwargs):
        
        r"""
        Creates a hovmoller plot of azimuthally-averaged recon data.
        
        Parameters
        ----------
        varname : Variable to average and plot (e.g. 'wspd').
            String
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
            
        Other Parameters
        ----------------
        prop : dict
            Customization properties for recon plot. Please refer to :ref:`options-prop-recon-hovmoller` for available options.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        default_prop = {'cmap':'category','levels':None,'smooth_contourf':False}
        for key in default_prop.keys():
            if key not in prop.keys():
                prop[key]=default_prop[key]
            
        #Get recon data based on recon_select
        dfRecon = self.data

        #Retrieve track dictionary if none is specified
        if track_dict is None:
            track_dict = self.storm.dict
        
        #Interpolate recon data to a hovmoller
        iRecon = interpRecon(dfRecon,varname,radlim,window=window,align=align)
        Hov_dict = iRecon.interpHovmoller(track_dict)

        #title = get_recon_title(varname) #may not be necessary
        #If no contour levels specified, generate levels based on data min and max
        if prop['levels'] is None:
            prop['levels'] = (np.nanmin(Hov_dict['hovmoller']),np.nanmax(Hov_dict['hovmoller']))
        
        #Retrieve updated contour levels and colormap based on input arguments and variable type
        cmap,clevs = get_cmap_levels(varname,prop['cmap'],prop['levels'])
        
        #Retrieve hovmoller times, radii and data
        time = Hov_dict['time']
        radius = Hov_dict['radius']
        vardata = Hov_dict['hovmoller']
        
        #Error check time
        time = [dt.strptime((i.strftime('%Y%m%d%H%M')),'%Y%m%d%H%M') for i in time]
        
        #------------------------------------------------------------------------------
        
        #Create plot        
        #plt.figure(figsize=(9,11),dpi=150)
        plt.figure(figsize=(9,9),dpi=150) #CHANGE THIS OR ELSE
        ax = plt.subplot()
        
        #Plot surface category colors individually, necessitating normalizing colormap
        if varname in ['vmax','sfmr','fl_to_sfc'] and prop['cmap'] == 'category':
            norm = mcolors.BoundaryNorm(clevs,cmap.N)
            cf = ax.contourf(radius,time,gfilt1d(vardata,sigma=3,axis=1),
                             levels=clevs,cmap=cmap,norm=norm)
        
        #Multiple clevels or without smooth contouring
        elif len(prop['levels']) > 2 or prop['smooth_contourf'] == False:
            cf = ax.contourf(radius,time,gfilt1d(vardata,sigma=3,axis=1),
                             levels=clevs,cmap=cmap)
        
        #Automatically generated levels with smooth contouring
        else:
            cf = ax.contourf(radius,time,gfilt1d(vardata,sigma=3,axis=1),
                             cmap=cmap,levels=np.linspace(min(prop['levels']),max(prop['levels']),256))
        ax.axis([0,max(radius),min(time),max(time)])
        
        #Plot colorbar
        cbar = plt.colorbar(cf,orientation='horizontal',pad=0.1)
        
        #Format y-label ticks and labels as dates
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(14)
        
        #Set axes labels
        ax.set_ylabel('UTC Time (MM-DD HH)',fontsize=15)
        ax.set_xlabel('Radius (km)',fontsize=15)
        
        #--------------------------------------------------------------------------------------
        
        #Generate left and right title strings
        title_left, title_right = hovmoller_plot_title(self.storm,Hov_dict,varname)
        ax.set_title(title_left,loc='left',fontsize=16,fontweight='bold')
        ax.set_title(title_right,loc='right',fontsize=12)
        
        #Return axis
        return ax

    def plot_maps(self,varname='wspd',track_dict=None,recon_stats=None,domain="dynamic",\
                  window=6,align='center',radlim=None,ax=None,savetopath=None,cartopy_proj=None,**kwargs):
    
        #plot_time, plot_mission (only for dots)
        
        r"""
        Creates maps of interpolated recon data. 
        
        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the following keys in recon_select dataframe:
            
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
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
        
        #Get plot data
        ONE_MAP = False
        dfRecon = self.data        
        
        MULTIVAR=False
        if isinstance(varname,(tuple,list)):
            MULTIVAR=True                    
        
        if track_dict is None:
            track_dict = self.storm.dict
            
            #Error check for time dimension name
            if 'time' not in track_dict.keys():
                track_dict['time'] = track_dict['date']
                
        if ONE_MAP:
            f = interp1d(mdates.date2num(track_dict['time']),track_dict['lon'], fill_value='extrapolate')
            clon = f(mdates.date2num(recon_select))
            f = interp1d(mdates.date2num(track_dict['time']),track_dict['lat'], fill_value='extrapolate')
            clat = f(mdates.date2num(recon_select))
            
            #clon = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lon'])
            #clat = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lat'])
            track_dict = {'time':recon_select,'lon':clon,'lat':clat}
        
        if MULTIVAR:
            Maps=[]
            for v in varname:
                iRecon = interpRecon(dfRecon,v,radlim,window=window,align=align)
                tmpMaps = iRecon.interpMaps(track_dict)
                Maps.append(tmpMaps)
        else:
            iRecon = interpRecon(dfRecon,varname,radlim,window=window,align=align)
            Maps = iRecon.interpMaps(track_dict)
                
        #titlename,units = get_recon_title(varname)
        
        if 'levels' not in prop.keys() or 'levels' in prop.keys() and prop['levels'] is None:
            prop['levels'] = np.arange(np.floor(np.nanmin(Maps['maps'])/10)*10,
                             np.ceil(np.nanmax(Maps['maps'])/10)*10+1,10)
        
        if not ONE_MAP:
            
            if savetopath is True:
                #savetopath = f'{self.storm}{self.year}_{varname}_maps'
                savetopath = f'{self.storm}{self.year}_maps'
            try:
                os.system(f'mkdir {savetopath}')
            except:
                pass
            
            if MULTIVAR:
                Maps2 = Maps[1]
                Maps = Maps[0]
            
                print(np.nanmax(Maps['maps']),np.nanmin(Maps2['maps']))
            
            figs = []
            for i,t in enumerate(Maps['time']):
                Maps_sub = {'time':t,'grid_x':Maps['grid_x'],'grid_y':Maps['grid_y'],'maps':Maps['maps'][i],\
                            'center_lon':Maps['center_lon'][i],'center_lat':Maps['center_lat'][i],'stats':Maps['stats']}

                #Create instance of plot object
                self.plot_obj = ReconPlot()
                
                #Create cartopy projection
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                cartopy_proj = self.plot_obj.proj
                
                #Maintain the same lat / lon dimensions for all dynamic maps
                #Determined by the dynamic domain from the first map
                if i>0 and domain == 'dynamic':
                    d1 = {'n':Maps_sub['center_lat']+dlat,\
                          's':Maps_sub['center_lat']-dlat,\
                          'e':Maps_sub['center_lon']+dlon,\
                          'w':Maps_sub['center_lon']-dlon}
                else:
                    d1 = domain
                
                #Plot recon
                
                if MULTIVAR:
                    Maps_sub1 = dict(Maps_sub)
                    Maps_sub2 = dict(Maps_sub)
                    Maps_sub = [Maps_sub1,Maps_sub2]
                    Maps_sub[1]['maps'] = Maps2['maps'][i]
                    
                    print(np.nanmax(Maps_sub[0]['maps']),np.nanmin(Maps_sub[1]['maps']))
                    
                plot_ax,d0 = self.plot_obj.plot_maps(self.storm,Maps_sub,varname,recon_stats,\
                                                    domain=d1,ax=ax,return_domain=True,prop=prop,map_prop=map_prop)
                
                #Get domain dimensions from the first map
                if i==0:
                    dlat = .5*(d0['n']-d0['s'])
                    dlon = .5*(d0['e']-d0['w'])
                
                figs.append(plot_ax)
                
                if savetopath is not None:
                    plt.savefig(f'{savetopath}/{t.strftime("%Y%m%d%H%M")}.png',bbox_inches='tight')
                plt.close()
                
            if savetopath is None:
                return figs

        else:
            #Create instance of plot object
            self.plot_obj = ReconPlot()
            
            #Create cartopy projection
            if cartopy_proj is None:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
                cartopy_proj = self.plot_obj.proj
            
            #Plot recon
            plot_info = self.plot_obj.plot_maps(self.storm,Maps,varname,recon_stats,\
                                                domain,ax,prop=prop,map_prop=map_prop)
            
            #Return axis
            return plot_info

    def plot_swath(self,varname='wspd',swathfunc=None,track_dict=None,radlim=None,\
                   domain="dynamic",ax=None,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a map plot of a swath of interpolated recon data.
        
        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the following keys in recon_select dataframe:
            
            * **"sfmr"** = SFMR surface wind
            * **"wspd"** = 30-second flight level wind (default)
            * **"pkwnd"** = 10-second flight level wind
            * **"p_sfc"** = extrapolated surface pressure
        swathfunc : function
            Function to operate on interpolated recon data.
            e.g., np.max, np.min, or percentile function
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
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
        
        #Get plot data
        dfRecon = self.data

        if track_dict is None:
            track_dict = self.storm.dict
        
        if swathfunc is None:
            if varname == 'p_sfc':
                swathfunc = np.min
            else:
                swathfunc = np.max
        
        iRecon = interpRecon(dfRecon,varname)
        Maps = iRecon.interpMaps(track_dict,interval=.2)
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_swath(self.storm,Maps,varname,swathfunc,track_dict,radlim,\
                                             domain,ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        return plot_info
    
    
    def gridded_stats(self,request,thresh={},binsize=1,domain="dynamic",ax=None,
                      return_array=False,cartopy_proj=None,prop={},map_prop={}):
        
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
            Keywords include:
                
            * **sample_min** - minimum number of storms in a grid box for the request to be applied. For the functions 'percentile' and 'average', 'sample_min' defaults to 5 and will override any value less than 5.
            * **v_min** - minimum wind for a given point to be included in the request.
            * **p_max** - maximum pressure for a given point to be included in the request.
            * **dv_min** - minimum change in wind over dt_window for a given point to be included in the request.
            * **dp_max** - maximum change in pressure over dt_window for a given point to be included in the request.
            * **dt_window** - time window over which change variables are calculated (hours). Default is 24.
            * **dt_align** - alignment of dt_window for change variables -- 'start','middle','end' -- e.g. 'end' for dt_window=24 associates a TC point with change over the past 24 hours. Default is middle.
            
            Units of all wind variables = kt, and pressure variables = hPa. These are added to the subtitle.

        year_range : list or tuple, optional
            List or tuple representing the start and end years (e.g., (1950,2018)). Default is start and end years of dataset.
        year_range_subtract : list or tuple, optional
            A year range to subtract from the previously specified "year_range". If specified, will create a difference plot.
        year_average : bool, optional
            If True, both year ranges will be computed and plotted as an annual average.
        date_range : list or tuple, optional
            List or tuple representing the start and end dates as a string in 'month/day' format (e.g., ('6/1','8/15')). Default is ('1/1','12/31') or full year.
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

        default_prop = {'smooth':None}
        for key in prop.keys():
            default_prop[key] = prop[key]
        prop = default_prop
        
        #Update thresh based on input
        default_thresh={'sample_min':1,'p_max':np.nan,'v_min':np.nan,'dv_min':np.nan,'dp_max':np.nan,'dv_max':np.nan,'dp_min':np.nan,'dt_window':24,'dt_align':'middle'}
        for key in thresh:
            default_thresh[key] = thresh[key]
        thresh = default_thresh
        
        #Retrieve the requested function, variable for computing stats, and plot title. These modify thresh if necessary.
        thresh,func = find_func(request,thresh)
        thresh,varname = find_var(request,thresh)
        
        #---------------------------------------------------------------------------------------------------

        points = self.data
        #Round lat/lon points down to nearest bin
        to_bin = lambda x: np.floor(x / binsize) * binsize
        points["latbin"] = points.lat.map(to_bin)
        points["lonbin"] = points.lon.map(to_bin)

        #---------------------------------------------------------------------------------------------------

        #Group by latbin,lonbin,stormid
        print("--> Grouping by lat/lon")
        groups = points.groupby(["latbin","lonbin"])

        #Loops through groups, and apply stat func to obs
        #Constructs a new dataframe containing the lat/lon bins and plotting variable
        new_df = {'latbin':[],'lonbin':[],varname:[]}
        for g in groups:
            new_df[varname].append(func(g[1][varname].values))                    
            new_df['latbin'].append(g[0][0])
            new_df['lonbin'].append(g[0][1])
        new_df = pd.DataFrame.from_dict(new_df)

        #---------------------------------------------------------------------------------------------------

        #Group again by latbin,lonbin
        #Construct two 1D lists: zi (grid values) and coords, that correspond to the 2D grid
        groups = new_df.groupby(["latbin", "lonbin"])

        zi = [func(g[1][varname]) if len(g[1]) >= thresh['sample_min'] else np.nan for g in groups]

        #Construct a 1D array of coordinates
        coords = [g[0] for g in groups]

        #Construct a 2D longitude and latitude grid, using the specified binsize resolution
        if prop['smooth'] is not None:
            all_lats = [(round(l/binsize)*binsize) for l in self.data['lat']]
            all_lons = [(round(l/binsize)*binsize)%360 for l in self.data['lon']]
            xi = np.arange(min(all_lons)-binsize,max(all_lons)+2*binsize,binsize)
            yi = np.arange(min(all_lats)-binsize,max(all_lats)+2*binsize,binsize)
        else:
            xi = np.arange(np.nanmin(points["lonbin"])-binsize,np.nanmax(points["lonbin"])+2*binsize,binsize)
            yi = np.arange(np.nanmin(points["latbin"])-binsize,np.nanmax(points["latbin"])+2*binsize,binsize)
        grid_x, grid_y = np.meshgrid(xi,yi)

        #Construct a 2D grid for the z value, depending on whether vector or scalar quantity
        grid_z = np.ones(grid_x.shape)*np.nan
        for c,z in zip(coords,zi):
            grid_z[np.where((grid_y==c[0]) & (grid_x==c[1]))] = z

        #---------------------------------------------------------------------------------------------------

        #Create instance of plot object
        try:
            self.plot_obj
        except:
            self.plot_obj = TrackPlot()
        
        #Create cartopy projection using basin
        if cartopy_proj is None:
            if max(points['lon']) > 150 or min(points['lon']) < -150:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
            else:
                self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)

        prop['title_L'],prop['title_R'] = self.storm.name,'things'
        
        if domain == "dynamic":
            domain = {'W':min(self.data['lon']),'E':max(self.data['lon']),'S':min(self.data['lat']),'N':max(self.data['lat'])}
        
        #Plot gridded field
        plot_ax = self.plot_obj.plot_gridded(grid_x,grid_y,grid_z,varname,domain=domain,ax=ax,prop=prop,map_prop=map_prop)
        
        #Format grid into xarray if specified
        if return_array:
            try:
                #Import xarray and construct DataArray, replacing NaNs with zeros
                import xarray as xr
                arr = xr.DataArray(np.nan_to_num(grid_z),coords=[grid_y.T[0],grid_x[0]],dims=['lat','lon'])
                return arr
            except ImportError as e:
                raise RuntimeError("Error: xarray is not available. Install xarray in order to use the 'return_array' flag.") from e

        #Return axis
        if return_array:
            return {'ax':plot_ax,'array':arr}
        else:
            return plot_ax

        
class dropsondes:

    r"""
    Creates an instance of a ReconDataset object containing all recon data for a single storm.
    
    Parameters
    ----------
    stormtuple : tuple or list
        Requested storm. Can be either tuple or list containing storm name and year (e.g., ("Matthew",2016)).
    save_path : str, optional
        Filepath to save recon data in. Recommended in order to avoid having to re-read in the data.
    read_path : str, optional
        Filepath to read saved recon data from. If specified, "save_path" cannot be passed as an argument.
        
    Returns
    -------
    Dataset
        An instance of ReconDataset, initialized with the following:
        
        * **missiondata** - A dictionary of missions.
            Each entry is a dateframe from a single mission.
            Dictionary keys are given by mission number and agency (e.g. '15_NOAA').
        * **recentered** - A dataframe with all missions concatenated together, and columns 'xdist' and 'ydist'
            indicating the distance (km) of the ob from the interpolated center of the storm.

    """

    def __repr__(self):
        
        summary = ["<tropycal.recon.dropsondes>"]
        
        def isNA(x,units):
            if np.isnan(x):
                return 'N/A'
            else:
                return f'{x} {units}'
        #Find maximum wind and minimum pressure
        max_MBLspd = isNA(np.nanmax([i['MBLspd'] for i in self.data]),'knots')
        max_DLMspd = isNA(np.nanmax([i['DLMspd'] for i in self.data]),'knots')
        max_WL150spd = isNA(np.nanmax([i['WL150spd'] for i in self.data]),'knots')
        min_slp = isNA(np.nanmin([i['slp'] for i in self.data]),'hPa')
        missions = set([i['mission'] for i in self.data])
        
        #Add general summary
        emdash = '\u2014'
        summary_keys = {'Storm':f'{self.storm.name} {self.storm.year}',\
                        'Missions':len(missions),
                        'Dropsondes':len(self.data),
                        'Max 500m-avg wind':max_MBLspd,
                        'Max 150m-avg wind':max_WL150spd,
                        'Min sea level pressure':min_slp}

        #Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')
        
        return "\n".join(summary)
    
    def __init__(self, storm, data=None, update=False):

        self.storm = storm
        self.archiveURL = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPNT3/'
        self.data = None

        if isinstance(data,str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        elif data is not None:
            self.data = data
        
        if data is None or update:
            try:
                start_time = max(self.data['time'])
            except:
                start_time = min(self.storm.dict['date'])-timedelta(days=1)
            end_time = max(self.storm.dict['date'])+timedelta(days=1)

            timeboundstrs = [f'{t:%Y%m%d%H%M}' for t in (start_time,end_time)]
            archive = pd.read_html(self.archiveURL)[0]
            linktimes = sorted([l.split('.') for l in archive['Name'] if isinstance(l,str) and 'txt' in l],key=lambda x: x[1])
            linksub = [self.archiveURL+'.'.join(l) for l in linktimes if l[1]>=min(timeboundstrs) and l[1]<=max(timeboundstrs)]
            timer_start = dt.now()
            print(f'Searching through recon dropsonde files between {timeboundstrs[0]} and {timeboundstrs[-1]} ...')
            for link in linksub:
                #print(link)
                content = requests.get(link).text
                datestamp = dt.strptime(link.split('.')[-2],'%Y%m%d%H%M')
                missionname,tmp = self._decode_dropsonde(content,date=datestamp)
                testkeys = ('TOPtime','lat','lon')
                if missionname[2:5] == self.storm.id[2:4]+self.storm.id[0]:
                    if self.data is None:
                        self.data = [copy.copy(tmp)]
                    elif [tmp[k] for k in testkeys] not in [[d[k] for k in testkeys] for d in self.data]:
                        self.data.append(tmp)
                    else:
                        pass
            print('--> Completed reading in recon missions (%.1f seconds)' % (dt.now()-timer_start).total_seconds())
            self.data = self._recenter()

        self.keys = sorted(list(set([k for d in self.data for k in d.keys()])))

    def update(self):
        newobj = dropsondes(storm=self.storm,data=self.data,update=True)
        return newobj
    
    def _decode_dropsonde(self,content,date):
        
        NOLOCFLAG = False
        missionname = '_____'
        
        delimiters = ['XXAA','31313','51515','61616','62626','XXBB','21212','_____']
        sections = {}
        for i,d in enumerate(delimiters[:-1]):
            a = content.split('\n'+d)
            if len(a)>1:
                a = ('\n'+d).join(a[1:]) if len(a)>2 else a[1] 
                b = a.split('\n'+delimiters[i+1])[0]
                sections[d] = b

        for k,v in sections.items():
            tmp = copy.copy(v)
            for d in delimiters:
                tmp=tmp.split('\n'+d)[0]
            tmp = [i for i in tmp.split(' ') if len(i)>0]
            tmp = [j.replace('\n','') if '\n' in j and (len(j)<(7+j.count('\n')) or len(j)==(11+j.count('\n'))) else j for j in tmp]
            tmp = [i for j in tmp for i in j.split('\n') if len(i)>0]
            sections[k] = tmp
        
        def _time(timestr):
            if timestr < f'{date:%H%M}':
                return date.replace(hour=int(timestr[:2]),minute=int(timestr[2:4]))
            else:
                return date.replace(hour=int(timestr[:2]),minute=int(timestr[2:4]))-timedelta(days=1)
        
        def _tempdwpt(item):
            if '/' in item[:3]:
                temp = np.nan
                dwpt = np.nan
            elif '/' in item[4:]:
                z = round(float(item[:3]),0)
                temp = round(z*0.1,1) if z%2==0 else round(z*-0.1,1)
                dwpt = np.nan
            else:
                z = round(float(item[:3]),0)
                temp = round(z*0.1,1) if z%2==0 else round(z*-0.1,1)
                z = round(float(item[3:]),0)
                dwpt = temp-(round(z*0.1,1) if z<=50 else z-50)
            return temp,dwpt

        def _wdirwspd(item):
            wdir = round(np.floor(float(item[:3])/5)*5,0) if '/' not in item else np.nan
            wspd = round(float(item[3:])+100*(float(item[2])%5),0) if '/' not in item else np.nan
            return wdir,wspd

        def _standard(I3):
            levkey = I3[0][:2]
            levdict = {'99':-1,'00':1000,'92':925,'85':850,'70':700,'50':500,'40':400,'30':300,'25':250,'20':200,'15':150,'10':100,'__':None}
            pres = float(levdict[levkey])
            output = {}
            output['pres'] = pres
            if pres==-1:
                output['pres'] = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:])+[0,1000][float(I3[0][2:])<100],1)
                output['hgt'] = 0.0
            elif pres==1000:
                z = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:]),0)
                output['hgt'] = round(500-z,0) if z>=500 else z
            elif pres==925:
                output['hgt'] = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:]),0)
            elif pres==850:
                output['hgt'] = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:])+1000,0)
            elif pres==700:
                z = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:]),0)
                output['hgt'] = round(z+3000,0) if z<500 else round(z+2000,0)       
            elif pres in (500,400,300):
                output['hgt'] = np.nan if '/' in I3[0][2:] else round(float(I3[0][2:])*10,0)
            else:
                output['hgt'] = np.nan if '/' in I3[0][2:] else round(1e4+float(I3[0][2:])*10,0)
            output['temp'],output['dwpt'] = _tempdwpt(I3[1])
            if I3[2][:2]=='88':
                output['wdir'],output['wspd'] = np.nan,np.nan
                skipflag = 0
            elif I3[2][:2]==list(levdict.keys())[list(levdict.keys()).index(levkey)+1] and \
            I3[3][:2]!=list(levdict.keys())[list(levdict.keys()).index(levkey)+1]:
                output['wdir'],output['wspd'] = np.nan,np.nan
                skipflag = 1
            else:
                output['wdir'],output['wspd'] = _wdirwspd(I3[2])  
                skipflag = 0 
            endflag = True if '88' in [i[:2] for i in I3] else False
            return output,skipflag,endflag

        data = {k:np.nan for k in ('lat','lon','slp',\
                                   'TOPlat','TOPlon','TOPtime',\
                                   'BOTTOMlat','BOTTOMlon','BOTTOMtime',\
                                   'MBLdir','MBLspd','DLMdir','DLMspd',\
                                   'WL150dir','WL150spd','top','LSThgt','software','levels')}
        
        for sec,items in sections.items():

            if sec == '61616' and len(items)>0:
                missionname = items[1]
                data['mission'] = items[1][:2]
                data['stormname'] = items[2]
                try:
                    data['obsnum'] = int(items[-1])
                except:
                    data['obsnum'] = items[-1]

            if sec == 'XXAA' and len(items)>0 and not NOLOCFLAG:
                if '/' in items[1]+items[2]:
                    NOLOCFLAG = True
                else:
                    octant = int(items[2][0])
                    data['lat'] = round(float(items[1][2:])*0.1*[-1,1][octant in (2,3,7,8)],1)
                    data['lon'] = round(float(items[2][1:])*0.1*[-1,1][octant in (0,1,2,3)],1)
                    data['slp'] = np.nan if '/' in items[4][2:] else round(float(items[4][2:])+[0,1000][float(items[4][2:])<100],1)

                    standard = {k:[] for k in ['pres','hgt','temp','dwpt','wdir','wspd']}
                    skips = 0
                    for jj,item in enumerate(items[4::3]):
                        if items[4+jj*3-skips][:2]=='88':
                            break
                        output,skipflag,endflag = _standard(items[4+jj*3-skips:8+jj*3-skips])
                        skips += skipflag
                        for k in standard.keys():
                            standard[k].append(output[k])
                        if endflag:
                            break
                    standard = pd.DataFrame.from_dict(standard).sort_values('pres',ascending=False)

            if sec == '62626' and len(items)>0 and not NOLOCFLAG:
                if items[0] in ['CENTER','MXWNDBND','RAINBAND','EYEWALL']:
                    data['location'] = items[0]
                    if items[0]=='EYEWALL':
                        data['octant'] = {'000':'N','045':'NE','090':'E','135':'SE',\
                                  '180':'S','225':'SW','270':'W','315':'NW'}[items[1]]
                if 'REL' in items:
                    tmp = items[items.index('REL')+1]
                    data['TOPlat'] = round(float(tmp[:4])*.01*[-1,1][tmp[4]=='N'],2)
                    data['TOPlon'] = round(float(tmp[5:10])*.01*[-1,1][tmp[10]=='E'],2)
                    tmp = items[items.index('REL')+2]
                    data['TOPtime'] = _time(tmp) #date + timedelta(hours=int(tmp[:2]),minutes=int(tmp[2:4]),seconds=int(tmp[4:6]))                    
                if 'SPG' in items:
                    tmp = items[items.index('SPG')+1]
                    data['BOTTOMlat'] = round(float(tmp[:4])*.01*[-1,1][tmp[4]=='N'],2)
                    data['BOTTOMlon'] = round(float(tmp[5:10])*.01*[-1,1][tmp[10]=='E'],2)
                    tmp = items[items.index('SPG')+2]
                    data['BOTTOMtime'] = _time(tmp) #date + timedelta(hours=int(tmp[:2]),minutes=int(tmp[2:4]),seconds=int(tmp[4:6]))                
                elif 'SPL' in items:
                    tmp = items[items.index('SPL')+1]
                    data['BOTTOMlat'] = round(float(tmp[:4])*.01*[-1,1][tmp[4]=='N'],2)
                    data['BOTTOMlon'] = round(float(tmp[5:10])*.01*[-1,1][tmp[10]=='E'],2)
                    tmp = items[items.index('SPL')+2]
                    data['BOTTOMtime'] = _time(tmp) #date + timedelta(hours=int(tmp[:2]),minutes=int(tmp[2:4]))                  
                if 'MBL' in items:
                    tmp = items[items.index('MBL')+2]
                    wdir,wspd = _wdirwspd(tmp)
                    data['MBLdir'] = wdir
                    data['MBLspd'] = wspd
                if 'DLM' in items:
                    tmp = items[items.index('DLM')+2]
                    wdir,wspd = _wdirwspd(tmp)
                    data['DLMdir'] = wdir
                    data['DLMspd'] = wspd                   
                if 'WL150' in items:
                    tmp = items[items.index('WL150')+1]
                    wdir,wspd = _wdirwspd(tmp)
                    data['WL150dir'] = wdir
                    data['WL150spd'] = wspd
                if 'LST' in items:
                    tmp = items[items.index('LST')+2]
                    data['LSThgt'] = round(float(tmp),0)
                if 'AEV' in items:
                    tmp = items[items.index('AEV')+1]
                    data['software'] = 'AEV '+tmp

            if sec == 'XXBB' and len(items)>0 and not NOLOCFLAG:
                sigtemp = {k:[] for k in ['pres','temp','dwpt']}
                for jj,item in enumerate(items[6::2]):
                    z = np.nan if '/' in items[6+jj*2][2:] else round(float(items[6+jj*2][2:]),0)
                    sigtemp['pres'].append(round(z+1000,0) if z<100 else z)
                    temp,dwpt = _tempdwpt(items[7+jj*2])
                    sigtemp['temp'].append(temp)
                    sigtemp['dwpt'].append(dwpt)
                sigtemp = pd.DataFrame.from_dict(sigtemp).sort_values('pres',ascending=False)

            if sec == '21212' and len(items)>0 and not NOLOCFLAG:
                sigwind = {k:[] for k in ['pres','wdir','wspd']}
                for jj,item in enumerate(items[2::2]):
                    z = np.nan if '/' in items[2+jj*2][2:] else round(float(items[2+jj*2][2:]),0)
                    sigwind['pres'].append(round(z+1000,0) if z<100 else z)
                    wdir,wspd = _wdirwspd(items[3+jj*2])
                    sigwind['wdir'].append(wdir)
                    sigwind['wspd'].append(wspd)
                sigwind = pd.DataFrame.from_dict(sigwind).sort_values('pres',ascending=False)

        if not NOLOCFLAG:
            def _justify(a, axis=0):    
                mask = pd.notnull(a)
                arg_justified = np.argsort(mask,axis=0)[-1]
                anew = [col[i] for i,col in zip(arg_justified,a.T)]
                return anew
            df = pd.concat([standard,sigtemp,sigwind],ignore_index=True, sort=False).sort_values('pres',ascending=False)
            data['levels'] = pd.DataFrame(np.vstack(df.groupby('pres', sort=False)
                              .apply(lambda gp: _justify(gp.to_numpy()))), columns=df.columns)

            data['top'] = np.nanmin(data['levels']['pres'])
            
        return missionname,data

    def _recenter(self,use='btk'): 
        data = self.data
        if use=='btk':
            centers = self.storm.to_dataframe()[['date','lon','lat']].rename(columns={'date':'time'})

        if len(centers)<2:
            print('Sorry, less than 2 center passes')
        else:
            for stage in ('TOP','BOTTOM'):
                #Interpolate center position to time of each ob
                f1 = interp1d(mdates.date2num(centers['time']),centers['lon'],fill_value='extrapolate',kind='quadratic')
                interp_clon = f1([mdates.date2num(d[f'{stage}time']) for d in data])
                f2 = interp1d(mdates.date2num(centers['time']),centers['lat'],fill_value='extrapolate',kind='quadratic')
                interp_clat = f2([mdates.date2num(d[f'{stage}time']) for d in data])

                #Get x,y distance of each ob from coinciding interped center position
                for i,d in enumerate(data):
                    d.update({f'{stage}xdist':great_circle( (interp_clat[i],interp_clon[i]), \
                        (interp_clat[i],d[f'{stage}lon']) ).kilometers* \
                        [1,-1][int(d[f'{stage}lon'] < interp_clon[i])]})
                    d.update({f'{stage}ydist':great_circle( (interp_clat[i],interp_clon[i]), \
                        (d[f'{stage}lat'],interp_clon[i]) ).kilometers* \
                        [1,-1][int(d[f'{stage}lat'] < interp_clat[i])]})
                    d.update({f'{stage}distance':(d[f'{stage}xdist']**2+d[f'{stage}ydist']**2)**.5})
                        
            print('Completed center-relative coordinates')
        return data
    
    def isel(self,i):
        r"""
        Select a single dropsonde by index of the list.
        """
        
        NEW_DATA = copy.copy(self.data)
        NEW_DATA = [NEW_DATA[i]]
        NEW_OBJ = dropsondes(storm = self.storm, data = NEW_DATA)
        
        return NEW_OBJ
        
    
    def sel(self,mission=None,time=None,domain=None,location=None,top=None,\
            slp=None,MBLspd=None,WL150spd=None,DLMspd=None):
        r"""
        Select a subset of dropsondes by parameter ranges.
        """

        NEW_DATA = copy.copy(pd.DataFrame(self.data))
        
        #Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission']==mission]

        #Apply time filter
        if time is not None:
            bounds = get_bounds(NEW_DATA['TOPtime'],time)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['TOPtime']>=bounds[0]) & (NEW_DATA['TOPtime']<=bounds[1])]
        
        #Apply domain filter
        if domain is not None:
            tmp = {k[0].lower():v for k,v in domain.items()}
            domain = {'n':90,'s':-90,'e':359.99,'w':0}
            domain.update(tmp)
            bounds = get_bounds(NEW_DATA['lon']%360,(domain['w']%360,domain['e']%360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon']%360>=bounds[0]) & (NEW_DATA['lon']%360<=bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'],(domain['s'],domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat']>=bounds[0]) & (NEW_DATA['lat']<=bounds[1])]
        
        #Apply location filter
        if location is not None:
            NEW_DATA = NEW_DATA.loc[NEW_DATA['location']==location.upper()]
        
        #Apply top standard level filter
        if top is not None:
            bounds = get_bounds(NEW_DATA['top'],top)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['top']>=bounds[0]) & (NEW_DATA['top']<=bounds[1])]
        
        #Apply surface pressure filter
        if slp is not None:
            bounds = get_bounds(NEW_DATA['slp'],slp)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['slp']>=bounds[0]) & (NEW_DATA['slp']<=bounds[1])]
            
        #Apply MBL wind speed filter
        if MBLspd is not None:
            bounds = get_bounds(NEW_DATA['MBLspd'],MBLspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['MBLspd']>=bounds[0]) & (NEW_DATA['MBLspd']<=bounds[1])]
            
        #Apply DLM wind speed filter
        if DLMspd is not None:
            bounds = get_bounds(NEW_DATA['DLMspd'],DLMspd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['DLMspd']>=bounds[0]) & (NEW_DATA['DLMspd']<=bounds[1])]
            
        #Apply WL150 wind speed filter
        if WL150spd is not None:
            bounds = get_bounds(NEW_DATA['WL150spd'],WL150spd)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['WL150spd']>=bounds[0]) & (NEW_DATA['WL150spd']<=bounds[1])]
            
        NEW_OBJ = dropsondes(storm=self.storm,data=list(NEW_DATA.T.to_dict().values()))
        
        return NEW_OBJ
        
    def to_pickle(self,filename):
        r"""
        Save dropsonde data (list of dictionaries) to a pickle file
        
        Parameters
        ----------
        filename : str
            name of file to save pickle file to.
        """
        
        with open(filename,'wb') as f:
            pickle.dump(self.data,f)
            
    def plot_points(self,varname='slp',level=None,domain="dynamic",\
                    ax=None,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of recon data points.
        
        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the keys in the dropsonde dictionary.
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
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
                
        #Get plot data
        if level is not None:
            plotdata = [m['levels'].loc[m['levels']['pres']==level][varname].to_numpy()[0] \
            if 'levels' in m.keys() and level in m['levels']['pres'].to_numpy() else np.nan \
            for m in self.data]
        else:
            plotdata = [m[varname] if varname in m.keys() else np.nan for m in self.data]
            
        dfRecon = pd.DataFrame.from_dict({'time':[m['BOTTOMtime'] for m in self.data],\
                                          'lat':[m['BOTTOMlat'] for m in self.data],\
                                          'lon':[m['BOTTOMlon'] for m in self.data],\
                                          varname:plotdata})
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_points(self.storm,dfRecon,domain,varname=(varname,level),\
                                              ax=ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        return plot_info

    def plot_skewt(self):
        r"""
        Plot skew-t for selected dropsondes
        
        returns a list of figures.
        """

        def time2text(time):
            try:
                return f'{data["TOPtime"]:%H:%M UTC %d %b %Y}'
            except:
                return 'N/A'
        def location_text(indict):
            try:
                loc = indict['location'].lower()
            except:
                return ''
            if loc == 'eyewall':
                return r"$\bf{"+indict['octant']+'}$ '+r"$\bf{"+loc.capitalize()+'}$, '
            else:
                return r"$\bf{"+loc.capitalize()+'}$, '
        degsym = u"\u00B0"
        def latlon2text(lat,lon):
            NA = False
            if lat<0:
                lattx = f'{abs(lat)}{degsym}S'
            elif lat>=0:
                lattx = f'{lat}{degsym}N'
            else:
                NA = True
            if lon<0:
                lontx = f'{abs(lon)}{degsym}W'
            elif lon>=0:
                lontx = f'{lon}{degsym}E'
            else:
                NA = True
            if NA:
                return 'N/A'
            else:
                return lattx+' '+lontx
                
        def mission2text(x):
            try:
                return int(x[:2])
            except:
                return x[:2]
        def wind_components(speed,direction):
            u = -speed * np.sin(direction*np.pi/180)
            v = -speed * np.cos(direction*np.pi/180)
            return u,v
        def deg2dir(x):
            dirs = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
            try:
                idx = int(round(x*16/360,0)%16)
                return dirs[idx]
            except:
                return 'N/A'
        def rh_from_dp(t,td):
            rh = np.exp(17.67 * (td) / (td+273.15 - 29.65)) / np.exp(17.67 * (t) / (t+273.15 - 29.65))
            return rh*100
        def cellcolor(color,value):
            if np.isnan(value):
                return 'w'
            else:
                return list(color[:3])+[.5]
        def skew_t(t,p):
            t0 = np.log(p/1050)*80/np.log(100/1050)
            return t0+t,p

        storm_data = self.storm.dict
        
        figs = []
        for data in self.data:
            # Loop through dropsondes
            df = data['levels'].sort_values('pres',ascending=True)
            Pres = df['pres']
            Temp = df['temp']
            Dwpt = df['dwpt']
            wind_speed = df['wspd']
            wind_dir = df['wdir']
            U,V = wind_components(wind_speed, wind_dir)

            ytop = int(np.nanmin(Pres)-50)
            yticks = np.arange(1000,ytop,-100)
            xticks = np.arange(-30,51,10)

            # Get mandatory and significant wind sub-dataframes
            dfmand = df.loc[df['pres'].isin((1000,925,850,700,500,400,300,250,200,150,100))]
            sfc = df.loc[df['hgt']==0]
            if len(sfc)>0:
                SLP = sfc['pres'].values[0]
                dfmand = pd.concat([dfmand,sfc])
                dfmand = dfmand.loc[dfmand['pres']<=SLP]
            else:
                SLP = None
            dfwind = df.loc[df['pres']>=700]

            # Start figure
            fig = plt.figure(figsize=(17,11))
            gs = gridspec.GridSpec(2,3,width_ratios=(2,.2,1.1),height_ratios=(len(dfmand)+3,len(dfwind)+3), wspace=0.0)

            ax1 = fig.add_subplot(gs[:,0])

            #Add titles
            type_array = np.array(storm_data['type'])
            idx = np.where((type_array == 'SD') | (type_array == 'SS') | (type_array == 'TD') | (type_array == 'TS') | (type_array == 'HU'))
            if ('invest' in storm_data.keys() and storm_data['invest'] == False) or len(idx[0]) > 0:
                tropical_vmax = np.array(storm_data['vmax'])[idx]

                add_ptc_flag = False
                if len(tropical_vmax) == 0:
                    add_ptc_flag = True
                    idx = np.where((type_array == 'LO') | (type_array == 'DB'))
                tropical_vmax = np.array(storm_data['vmax'])[idx]

                subtrop = classify_subtropical(np.array(storm_data['type']))
                peak_idx = storm_data['vmax'].index(np.nanmax(tropical_vmax))
                peak_basin = storm_data['wmo_basin'][peak_idx]
                storm_type = get_storm_classification(np.nanmax(tropical_vmax),subtrop,peak_basin)
                if add_ptc_flag == True: storm_type = "Potential Tropical Cyclone"

            ax1.set_title(f'{storm_type} {storm_data["name"]}'+\
                          f'\nDropsonde {data["obsnum"]}, Mission {mission2text(data["missionname"])}',\
                          loc='left',fontsize=17,fontweight='bold')
            ax1.set_title(f'Drop time: {time2text(data["TOPtime"])}'+\
                          f'\nDrop location: {location_text(data)}{latlon2text(data["lat"],data["lon"])}',loc='right',fontsize=13)
            plt.yscale('log')
            plt.yticks(yticks,[f'{i:d}' for i in yticks],fontsize=12)
            plt.xticks(xticks,[f'{i:d}' for i in xticks],fontsize=12)
            for y in range(1000,ytop,-50):plt.plot([-30,50],[y]*2,color='0.5',lw=0.5)
            for x in range(-30-80,50,10):plt.plot([x,x+80],[1050,100],color='0.5',linestyle='--',lw=0.5)

            plt.plot(*skew_t(Temp.loc[~np.isnan(Temp)],Pres.loc[~np.isnan(Temp)]),'o-',color='r')
            plt.plot(*skew_t(Dwpt.loc[~np.isnan(Dwpt)],Pres.loc[~np.isnan(Dwpt)]),'o-',color='g')
            plt.xlabel(f'Temperature ({degsym}C)',fontsize=13)
            plt.ylabel('Pressure (hPa)',fontsize=13)
            plt.axis([-30,50,1050,ytop])

            lim = max([i for stage in ('TOP','BOTTOM') for i in [1.5*abs(data[f'{stage}xdist'])+.1,1.5*abs(data[f'{stage}ydist'])+.1]])
            iscoords = np.isnan(lim)
            if iscoords:
                lim = 1
            for stage,ycoord in zip(('TOP','BOTTOM'),(.8,.05)):
                ax1in1 = ax1.inset_axes([0.05, ycoord, 0.15, 0.15])
                if iscoords:
                    ax1in1.set_title('distance N/A')
                else:
                    ax1in1.scatter(0,0,c='k')
                    ax1in1.scatter(data[f'{stage}xdist'],data[f'{stage}ydist'],c='w',marker='v',edgecolor='k')
                    ax1in1.set_title(f'{data[f"{stage}distance"]:0.0f} km {deg2dir(90-math.atan2(data[f"{stage}ydist"],data[f"{stage}xdist"])*180/np.pi)}')                    
                ax1in1.axis([-lim,lim,-lim,lim])
                ax1in1.xaxis.set_major_locator(plt.NullLocator())
                ax1in1.yaxis.set_major_locator(plt.NullLocator())
                
            ax4 = fig.add_subplot(gs[:,1],sharey=ax1)
            barbs = {k:[v.values[-1]] for k,v in zip(('p','u','v'),(Pres,U,V))}
            for p,u,v in zip(Pres.values[::-1],U.values[::-1],V.values[::-1]):
                if abs(p-barbs['p'][-1])>10 and not np.isnan(u):
                    for k,v in zip(('p','u','v'),(p,u,v)):
                        barbs[k].append(v)
            plt.barbs([.4]*len(barbs['p']),barbs['p'],barbs['u'],barbs['v'], pivot='middle')
            ax4.set_xlim(0,1)
            ax4.axis('off')

            RH = [rh_from_dp(i,j) for i,j in zip(dfmand['temp'],dfmand['dwpt'])]
            cellText = np.array([['' if np.isnan(i) else f'{int(i)} hPa' for i in dfmand['pres']],\
                        ['' if np.isnan(i) else f'{int(i)} m' for i in dfmand['hgt']],\
                        ['' if np.isnan(i) else f'{i:.1f} {degsym}C' for i in dfmand['temp']],\
                        ['' if np.isnan(i) else f'{int(i)} %' for i in RH],\
                        ['' if np.isnan(i) else f'{deg2dir(j)} at {int(i)} kt' for i,j in zip(dfmand['wspd'],dfmand['wdir'])]]).T
            colLabels = ['Pressure','Height','Temp','RH','Wind']

            cmap_rh = mlib.cm.get_cmap('BrBG')
            cmap_temp = mlib.cm.get_cmap('RdBu_r')
            cmap_wind = mlib.cm.get_cmap('Purples')

            colors = [['w','w',cellcolor(cmap_temp(t/120+.5),t),\
                                cellcolor(cmap_rh(r/100),r),\
                                cellcolor(cmap_wind(w/200),w)] for t,r,w in zip(dfmand['temp'],RH,dfmand['wspd'])]
            
            ax2 = fig.add_subplot(gs[0,2])
            ax2.xaxis.set_visible(False)  # hide the x axis
            ax2.yaxis.set_visible(False)  # hide the y axis
            TB = ax2.table(cellText=cellText,colLabels=colLabels,cellColours=colors,cellLoc='center',bbox = [0, .05, 1, .95])
            if SLP is not None:
                TB[(len(cellText), 0)].get_text().set_weight('bold')
            ax2.axis('off')
            TB.auto_set_font_size(False)
            TB.set_fontsize(9)
            #TB.scale(3,1.2)
            try:
                ax2.text(0,.05,f'\nDeep Layer Mean Wind: {deg2dir(data["DLMdir"])} at {int(data["DLMspd"])} kt',va='top',fontsize=12)
            except:
                ax2.text(0,.05,f'\nDeep Layer Mean Wind: N/A',va='top',fontsize=12)

            ax2.set_title('Generated using Tropycal \n',fontsize=12,fontweight='bold',color='0.7',loc='right')

            cellText = np.array([[f'{int(i)} hPa' for i,j in zip(dfwind['pres'],dfwind['wspd']) if not np.isnan(j)],\
                        [f'{deg2dir(j)} at {int(i)} kt' for i,j in zip(dfwind['wspd'],dfwind['wdir']) if not np.isnan(i)]]).T
            colLabels = ['Pressure','Wind']
            colors = [['w',cellcolor(cmap_wind(i/200),i)] for i in dfwind['wspd'] if not np.isnan(i)]

            ax3 = fig.add_subplot(gs[1,2])

            try:
                TB = ax3.table(cellText=cellText,colLabels=colLabels,cellColours=colors,cellLoc='center',bbox = [0, .1, 1, .9])
                TB.auto_set_font_size(False)
                TB.set_fontsize(9)
                meanwindoffset = 0
            except:
                meanwindoffset = 0.9
            #TB.scale(2,1.2)
            ax3.xaxis.set_visible(False)  # hide the x axis
            ax3.yaxis.set_visible(False)  # hide the y axis
            ax3.axis('off')
            
            try:
                ax3.text(0,.1+meanwindoffset,\
                         f'\nMean Wind in Lowest 500 m: {deg2dir(data["MBLdir"])} at {int(data["MBLspd"])} kt',va='top',fontsize=12)
            except:
                ax3.text(0,.1+meanwindoffset,\
                         f'\nMean Wind in Lowest 500 m: N/A',va='top',fontsize=12)
            try:
                ax3.text(0,.1+meanwindoffset,\
                         f'\n\nMean Wind in Lowest 150 m: {deg2dir(data["WL150dir"])} at {int(data["WL150spd"])} kt',va='top',fontsize=12)
            except:
                ax3.text(0,.1+meanwindoffset,\
                         f'\n\nMean Wind in Lowest 150 m: N/A',va='top',fontsize=12)

            figs.append(fig)
            plt.close()
            
        if len(figs)>1:
            return figs
        elif len(figs)==1:
            return fig
        else:
            print("No dropsondes in selection")
    
    
class vdms:
    
    def __repr__(self):
        summary = ["<tropycal.recon.vdms>"]

        #Find maximum wind and minimum pressure
        time_range = (np.nanmin([i['time'] for i in self.data]),np.nanmax([i['time'] for i in self.data]))
        time_range = list(set(time_range))
        min_slp = np.nanmin([i['Minimum Sea Level Pressure (hPa)'] for i in self.data])
        min_slp = 'N/A' if np.isnan(min_slp) else min_slp
        missions = set([i['mission'] for i in self.data])
        
        #Add general summary
        emdash = '\u2014'
        summary_keys = {'Storm':f'{self.storm.name} {self.storm.year}',\
                        'Missions':len(missions),
                        'VDMs':len(self.data),
                        'Min sea level pressure':f"{min_slp} hPa"}

        #Add dataset summary
        summary.append("Dataset Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')
        
        return "\n".join(summary)
            
    def __init__(self, storm, data=None):

        self.storm = storm
        archiveURL = f'https://www.nhc.noaa.gov/archive/recon/{self.storm.year}/REPNT2/'
        timestr = [f'{t:%Y%m%d}' for t in self.storm.dict['date']]
        archive = pd.read_html(archiveURL)[0]
        linktimes = sorted([l.split('.') for l in archive['Name'] if isinstance(l,str) and 'txt' in l],key=lambda x: x[1])
        linksub = [archiveURL+'.'.join(l) for l in linktimes if l[1][:8] in timestr]
        self.data = []

        if data is None:
            timer_start = dt.now()
            print(f'Searching through recon VDM files between {timestr[0]} and {timestr[-1]} ...')
            for link in linksub:
                content = requests.get(link).text
                #print(link)
                date = link.split('.')[-2]
                year = int(date[:4])
                month = int(date[4:6])
                day = int(date[6:8])
                missionname,tmp = self._decode_vdm(content,date=dt(year,month,day))
                if missionname[2:5] == self.storm.id[2:4]+self.storm.id[0]:
                    self.data.append(tmp)
            print('--> Completed reading in recon missions (%.1f seconds)' % (dt.now()-timer_start).total_seconds())
        elif isinstance(data,str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
        else:
            self.data = data
        self.keys = sorted(list(set([k for d in self.data for k in d.keys()])))
        
    def _decode_vdm(self,content,date):
        data = {}
        lines = content.split('\n')
        RemarksNext = False
        LonNext = False
        FORMAT = 1 if date.year<2018 else 2
        
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
                data['Remarks'] += (' '+line)
            if LonNext:
                info = line.split()
                data['lon'] = np.round((float(info[0])+float(info[2])/60)*[-1,1][info[4]=='E'],2)
                LonNext = False

            if 'VORTEX DATA MESSAGE' in line:
                stormid = line.split()[-1]
            if line[:2] == 'A.':
                info = line[3:].split('/')
                day = int(info[0])
                month = (date.month-int(day-date.day>15)-1)%12+1
                year = date.year-int(date.month-int(day-date.day>15)==0)
                hour,minute,second = [int(i) for i in info[1][:-1].split(':')]
                data['time'] = dt(year,month,day,hour,minute,second)

            if line[:2] == 'B.':
                info = line[3:].split()
                if FORMAT==1:
                    data['lat'] = np.round((float(info[0])+float(info[2])/60)*[-1,1][info[4]=='N'],2)
                    LonNext = True
                if FORMAT==2:
                    data['lat'] = float(info[0])*[-1,1][info[2]=='N']
                    data['lon'] = float(info[3])*[-1,1][info[5]=='E']

            if line[:2] == 'C.':
                info = line[3:].split()*5
                data[f'Standard Level (hPa)']=isNA(info[0])
                data[f'Minimum Height at Standard Level (m)']=isNA(info[2])

            if line[:2] == 'D.':
                info = line[3:].split()*5
                if FORMAT==1:
                    data['Estimated Maximum Surface Wind Inbound (kt)'] = isNA(info[0])
                if FORMAT==2:
                    data['Minimum Sea Level Pressure (hPa)']=isNA(info[-2])                    

            if line[:2] == 'E.':
                info = line[3:].split()*5
                if FORMAT==1:
                    data['Dropsonde Surface Wind Speed at Center (kt)']=isNA(info[2])
                    data['Dropsonde Surface Wind Direction at Center (deg)']=isNA(info[0])
                if FORMAT==2:
                    data['Location of Estimated Maximum Surface Wind Inbound']=isNA(line[3:])

            if line[:2] == 'F.':
                info = line[3:]
                if FORMAT==1:
                    data['Maximum Flight Level Wind Inbound']=isNA(info)
                if FORMAT==2:
                    data['Eye character']=isNA(info)

            if line[:2] == 'G.':
                info = line[3:]
                if FORMAT==1:
                    data['Location of the Maximum Flight Level Wind Inbound']=isNA(info)
                if FORMAT==2:
                    if isNA(info) == np.nan:
                        data.update({'Eye Shape':np.nan,'Eye Diameter (nmi)':np.nan})
                    else:
                        shape = ''.join([i for i in info[:2] if not i.isdigit()])
                        size = info[len(shape):]
                        if shape=='C':
                            data.update({'Eye Shape':'circular','Eye Diameter (nmi)':float(size)})
                        elif shape=='CO':
                            data['Eye Shape']='concentric'
                            data.update({f'Eye Diameter {i+1} (nmi)':float(s) for i,s in enumerate(size.split('-'))})
                        elif shape=='E':
                            einfo = size.split('/')
                            data.update({'Eye Shape':'elliptical','Orientation':float(einfo[0])*10,\
                                         'Eye Major Axis (nmi)':float(einfo[1]),'Eye Minor Axis (nmi)':float(einfo[1])})
                        else:
                            data.update({'Eye Shape':np.nan,'Eye Diameter (nmi)':np.nan})

            if line[:2] == 'H.':
                info = line[3:].split()*5
                if FORMAT==1:
                    data['Minimum Sea Level Pressure (hPa)']=isNA(info[-2]) 
                if FORMAT==2:
                    data['Estimated Maximum Surface Wind Inbound (kt)']=isNA(info[0])

            if line[:2] == 'I.':
                info = line[3:]
                if FORMAT==1:
                    data['Maximum Flight Level Temp Outside Eye (C)']=isNA(info.split()[0])
                if FORMAT==2:
                    data['Location & Time of the Estimated Maximum Surface Wind Inbound']=isNA(info)

            if line[:2] == 'J.':
                info = line[3:]
                if FORMAT==1:
                    data['Maximum Flight Level Temp Inside Eye (C)']=isNA(info.split()[0])
                if FORMAT==2:
                    data['Maximum Flight Level Wind Inbound (kt)']=isNA(info)

            if line[:2] == 'K.':
                info = line[3:]
                if FORMAT==1:
                    data['Dew Point Inside Eye (C)']=isNA(info.split()[0])
                if FORMAT==2:
                    data['Location & Time of the Maximum Flight Level Wind Inbound']=isNA(info)

            if line[:2] == 'L.':
                info = line[3:]
                if FORMAT==1:
                    data['Eye character']=isNA(info)
                if FORMAT==2:
                    data['Estimated Maximum Surface Wind Outbound (kt)']=isNA(info)

            if line[:2] == 'M.':
                info = line[3:]
                if FORMAT==1:
                    if isNA(info) == np.nan:
                        data.update({'Eye Shape':np.nan,'Eye Diameter (nmi)':np.nan})
                    else:
                        shape = ''.join([i for i in info[:2] if not i.isdigit()])
                        size = info[len(shape):]
                        if shape=='C':
                            data.update({'Eye Shape':'circular','Eye Diameter (nmi)':float(size)})
                        elif shape=='CO':
                            data['Eye Shape']='concentric'
                            data.update({f'Eye Diameter {i+1} (nmi)':float(s) for i,s in enumerate(size.split('-'))})
                        elif shape=='E':
                            einfo = size.split('/')
                            data.update({'Eye Shape':'elliptical','Orientation':float(einfo[0])*10,\
                                         'Eye Major Axis (nmi)':float(einfo[1]),'Eye Minor Axis (nmi)':float(einfo[1])})
                        else:
                            data.update({'Eye Shape':np.nan,'Eye Diameter (nmi)':np.nan})
                if FORMAT==2:
                    data['Location & Time of the Estimated Maximum Surface Wind Outbound']=isNA(info)

            if line[:2] == 'N.':
                info = line[3:]
                if FORMAT==2:
                    data['Maximum Flight Level Wind Outbound (kt)']=isNA(info)

            if line[:2] == 'O.':
                info = line[3:]
                if FORMAT==2:
                    data['Location & Time of the Maximum Flight Level Wind Outbound']=isNA(info)

            if line[:2] == 'P.':
                info = line[3:]
                if FORMAT==1:
                    data['Aircraft'] = info.split()[0]
                    data['mission'] = info.split()[1]
                    data['Remarks'] = ''
                    RemarksNext = True
                if FORMAT==2:
                    data['Maximum Flight Level Temp & Pressure Altitude Outside Eye']=isNA(info)

            if line[:2] == 'Q.':
                info = line[3:]
                if FORMAT==2:
                    data['Maximum Flight Level Temp & Pressure Altitude Inside Eye']=isNA(info)

            if line[:2] == 'R.':
                info = line[3:]
                if FORMAT==2:
                    data['Dewpoint Temp (collected at same location as temp inside eye)']=isNA(info)

            if line[:2] == 'S.':
                info = line[3:]
                if FORMAT==2:
                    data['Fix']=isNA(info)

            if line[:2] == 'T.':
                info = line[3:]
                if FORMAT==2:
                    data['Accuracy']=isNA(info)

            if line[:2] == 'U.':
                info = line[3:]
                if FORMAT==2:
                    data['Aircraft'] = info.split()[0]
                    data['mission'] = info.split()[1][:2]
                    data['Remarks'] = ''
                    RemarksNext = True
        
        return data['mission'],data

    def isel(self,i):
        r"""
        Select a single VDM by index of the list.
        """
        
        NEW_DATA = copy.copy(self.data)
        NEW_DATA = [NEW_DATA[i]]
        NEW_OBJ = vdms(storm = self.storm, data = NEW_DATA)
        
        return NEW_OBJ

    def sel(self,mission=None,time=None,domain=None):
        r"""
        Select a subset of dropsondes by parameter ranges.
        """

        NEW_DATA = copy.copy(pd.DataFrame(self.data))
        
        #Apply mission filter
        if mission is not None:
            mission = str(mission)
            NEW_DATA = NEW_DATA.loc[NEW_DATA['mission']==mission]

        #Apply time filter
        if time is not None:
            bounds = get_bounds(NEW_DATA['time'],time)
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['time']>=bounds[0]) & (NEW_DATA['time']<=bounds[1])]
        
        #Apply domain filter
        if domain is not None:
            tmp = {k[0].lower():v for k,v in domain.items()}
            domain = {'n':90,'s':-90,'e':359.99,'w':0}
            domain.update(tmp)
            bounds = get_bounds(NEW_DATA['lon']%360,(domain['w']%360,domain['e']%360))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lon']%360>=bounds[0]) & (NEW_DATA['lon']%360<=bounds[1])]
            bounds = get_bounds(NEW_DATA['lat'],(domain['s'],domain['n']))
            NEW_DATA = NEW_DATA.loc[(NEW_DATA['lat']>=bounds[0]) & (NEW_DATA['lat']<=bounds[1])]
        
        NEW_OBJ = vdms(storm=self.storm,data=list(NEW_DATA.T.to_dict().values()))
        
        return NEW_OBJ
    
    def to_pickle(self,filename):
        r"""
        Save VDM data (list of dictionaries) to a pickle file
        
        Parameters
        ----------
        filename : str
            name of file to save pickle file to.
        """
        
        with open(filename,'wb') as f:
            pickle.dump(self.data,f)
            
    def plot_points(self,varname='Minimum Sea Level Pressure (hPa)',sizeby=None,domain="dynamic",\
                    ax=None,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of recon data points.
        
        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the keys in self.keys.

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
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
                
        #Get plot data
        plotdata = [m[varname] if varname in m.keys() else np.nan for m in self.data]
            
        dfRecon = pd.DataFrame.from_dict({'time':[m['time'] for m in self.data],\
                                          'lat':[m['lat'] for m in self.data],\
                                          'lon':[m['lon'] for m in self.data],\
                                          varname:plotdata})
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_points(self.storm,dfRecon,domain,varname=varname,\
                                              ax=ax,prop=prop,map_prop=map_prop)        
        #Return axis
        return plot_info
        