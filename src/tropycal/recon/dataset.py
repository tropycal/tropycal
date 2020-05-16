import os
import numpy as np
from datetime import datetime as dt,timedelta
import pandas as pd
import requests
import pickle

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter as gfilt,gaussian_filter1d as gfilt1d
from scipy.ndimage.filters import minimum_filter
from geopy.distance import great_circle
import matplotlib.dates as mdates

try:
    import matplotlib as mlib
    import matplotlib.lines as mlines
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except:
    warnings.warn("Warning: Matplotlib is not installed in your python environment. Plotting functions will not work.")

from .plot import ReconPlot
from .tools import * 
#from ..tracks import *

class ReconDataset:

    r"""
    Creates an instance of a ReconDataset object containing all recon data for a single storm.
    
    Parameters
    ----------
    stormtuple : tuple or list
        Requested storm. Can be either tuple or list containing storm name and year (e.g., ("Matthew",2016)).
        
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

    def __init__(self,storm,save_path="",read_path=""):
        
        if save_path != "" and read_path != "":
            raise ValueError("Error: Cannot read in and save a file at the same time.")
        
        self.url_prefix = 'http://tropicalatlantic.com/recon/recon.cgi?'
        self.storm_obj = storm
        self.storm = str(storm.name)
        self.year = str(storm.year)
        
        if read_path != "":
            self.missiondata = pickle.load(open(read_path,'rb'))
        else:
            self.missiondata = self.allMissions()
            if save_path != "": pickle.dump(self.missiondata,open(save_path,'wb'),-1)
        
        self.recentered = self.recenter()
            
        
    def getMission(self,agency,mission_num,url_mission=None):
        if url_mission is None:
            url_mission = f'{self.url_prefix}basin=al&year={self.year}&product=hdob&storm={self.storm}&mission={mission_num}&agency={agency}'
        content = np.array(requests.get(url_mission).content.decode("utf-8").split('\n'))
        obs = [line.split('\"')[1] for line in content if 'option value=' in line][::-1]
        for i,ob in enumerate(obs):
            url_ob = url_mission+'&ob='+ob
            data = pd.read_html(url_ob)[0]
            data = data.rename(columns = {[name for name in data if 'Time' in name][0]:'Time'})
            if i==0:
                mission = data[:-1]
                day0 = dt.strptime(self.year+ob[:5],'%Y%m-%d')
            else:
                mission = mission.append(data[:-1],ignore_index=True)
                
        def getVar(x,name):
            a = np.nan
            if x!='-' and '*' not in x and x!='No Wind':
                if name == 'Time':
                    a = x
                if name == 'Coordinates':
                    lat,lon = x.split(' ')
                    lat = float(lat[:-1])*[1,-1][lat[-1]=='S']
                    lon = float(lon[:-1])*[1,-1][lon[-1]=='W']
                    a = np.array((lon,lat))
                elif name == 'Aircraft Static Air Pressure':
                    a=float(x.split(' mb')[0])
                elif name == 'Aircraft Geo. Height':
                    a=float(x.split(' meters')[0].replace(',', ''))
                elif name == 'Extrapolated Sfc. Pressure':
                    a=float(x.split(' mb')[0])
                elif name == 'Flight Level Wind (30 sec. Avg.)':
                    a=x.split(' ')
                    wdir = float(a[1][:-1])
                    wspd = float(a[3])
                    a = np.array((wdir,wspd))
                elif name == 'Peak (10 sec. Avg.) Flight Level Wind':
                    a=float(x.split(' knots')[0])
                elif name == 'SFMR Peak (10s Avg.) Sfc. Wind':
                    a=x.split(' knots')
                    a=float(a[0])
            if name in ['Coordinates','Flight Level Wind (30 sec. Avg.)'] and type(a)==float:
                a=np.array([a]*2)
            return a
    
        varnames = ['Time','Coordinates','Aircraft Static Air Pressure','Aircraft Geo. Height',
                    'Extrapolated Sfc. Pressure','Flight Level Wind (30 sec. Avg.)',
                    'Peak (10 sec. Avg.) Flight Level Wind','SFMR Peak (10s Avg.) Sfc. Wind']
        mission = {name:[getVar(item,name) for item in mission[name]] for name in varnames}
        for i,t in enumerate(mission['Time']):
            mission['Time'][i] = day0.replace(hour=int(t[:2]),minute=int(t[3:5]),second=int(t[6:8]))
            if i>0 and (mission['Time'][i]-mission['Time'][i-1]).total_seconds()<0:
                mission['Time'][i]+=timedelta(days=1)
        data={}
        data['lon'],data['lat'] = zip(*mission['Coordinates'])
        data['time'] = mission['Time']
        data['p_sfc'] = mission['Extrapolated Sfc. Pressure']
        data['wdir'],data['wspd'] = zip(*mission['Flight Level Wind (30 sec. Avg.)'])
        data['pkwnd'] = mission['Peak (10 sec. Avg.) Flight Level Wind']
        data['sfmr'] = mission['SFMR Peak (10s Avg.) Sfc. Wind']
        data['plane_p'] = mission['Aircraft Static Air Pressure']
        data['plane_z'] = mission['Aircraft Geo. Height']
        return_data = pd.DataFrame.from_dict(data)
        return_data['time'] = [pd.to_datetime(i) for i in return_data['time']]
        
        #remove nan's for lat/lon coordinates
        return_data = return_data.dropna(subset=['lat', 'lon'])
        
        return return_data
    

    def allMissions(self):
        url_storm = f'{self.url_prefix}basin=al&year={self.year}&storm={self.storm}&product=hdob'
        missions = pd.read_html(url_storm)[0]
        missiondata={}
        timer_start = dt.now()
        print(f'--> Starting to read in recon missions')
        for i_mission in range(len(missions)):
            mission_num = str(missions['MissionNumber'][i_mission]).zfill(2)
            agency = ''.join(filter(str.isalpha, missions['Agency'][i_mission]))
            missiondata[f'{mission_num}_{agency}'] = self.getMission(agency,mission_num)
            print(f'{mission_num}_{agency}')
        print('--> Completed reading in recon missions (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        return missiondata

    def find_centers(self,data):
        
        def fill_nan(A):
            #Interpolate to fill nan values
            A = np.array(A)
            inds = np.arange(len(A))
            good = np.where(np.isfinite(A))
            good_grad = np.gradient(good[0])
            if len(good[0])>=3:
                f = interp1d(inds[good], A[good],bounds_error=False,kind='quadratic')
                B = np.where(np.isfinite(A)[good[0][0]:good[0][-1]+1],
                             A[good[0][0]:good[0][-1]+1],
                             f(inds[good[0][0]:good[0][-1]+1]))
                return [np.nan]*good[0][0]+list(B)+[np.nan]*(inds[-1]-good[0][-1])
            else:
                return [np.nan]*len(A)
        
        #Check that sfc pressure spread is big enough to identify real minima
        if np.nanpercentile(data['p_sfc'],90)-np.nanpercentile(data['p_sfc'],10)>8:
            data['p_sfc'][:20]=[np.nan]*20 #NaN out the first 10 minutes of the flight
            p_sfc_interp = fill_nan(data['p_sfc']) #Interp p_sfc across missing data
            wspd_interp = fill_nan(data['wspd']) #Interp wspd across missing data
            #Smooth p_sfc and wspd
            p_sfc_smooth = [np.nan]*1+list(np.convolve(p_sfc_interp,[1/3]*3,mode='valid'))+[np.nan]*1
            wspd_smooth = [np.nan]*1+list(np.convolve(wspd_interp,[1/3]*3,mode='valid'))+[np.nan]*1
            #Add wspd to p_sfc to encourage finding p mins with wspd mins 
            #and prevent finding p mins in intense thunderstorms
            pw_test = np.array(p_sfc_smooth)+np.array(wspd_smooth)*.1
            #Find mins in 15-minute windows
            imin = np.nonzero(pw_test == minimum_filter(pw_test,30))[0]
            #Only use mins if below 15th %ile of mission p_sfc data and when plane p is 500-900mb
            imin = [i for i in imin if 800<p_sfc_interp[i]<np.nanpercentile(data['p_sfc'],15) and \
                    550<data['plane_p'][i]<900]
        else:
            imin=[]
        data['iscenter'] = np.zeros(len(data['p_sfc']))
        for i in imin:
            j = data.index.values[i]
            data['iscenter'][j] = 1
        return data

    def recenter(self,use='all'): 
        self.use = use        
        def stitchMissions():
            list_of_dfs=[]
            for name in self.missiondata:
                if self.use == 'all' or self.use in name:
                    mission = self.missiondata[name]
                    tmp = self.find_centers(mission)
                    list_of_dfs.append( tmp )
            data_concat = pd.concat(list_of_dfs,ignore_index=True)
            data_chron = data_concat.sort_values(by='time').reset_index(drop=True)
            return data_chron

        data = stitchMissions()
        centers = data.loc[data['iscenter']>0]
        
        if len(centers)<2:
            print('Sorry, less than 2 center passes')
        else:
            print(f'Found {len(centers)} center passes!')
            timer_start = dt.now()
            
            #Interpolate center position to time of each ob
            f1 = interp1d(mdates.date2num(centers['time']),centers['lon'],fill_value='extrapolate',kind='linear')
            interp_clon = f1(mdates.date2num(data['time']))
            f2 = interp1d(mdates.date2num(centers['time']),centers['lat'],fill_value='extrapolate',kind='linear')
            interp_clat = f2(mdates.date2num(data['time']))

            #Get x,y distance of each ob from coinciding interped center position
            data['xdist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
                (interp_clat[i],data['lon'][i]) ).kilometers* \
                [1,-1][int(data['lon'][i] < interp_clon[i])] for i in range(len(data))]
            data['ydist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
                (data['lat'][i],interp_clon[i]) ).kilometers* \
                [1,-1][int(data['lat'][i] < interp_clat[i])] for i in range(len(data))]
                        
            print('--> Completed recentering recon data (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        return data
        
    def __getSubTime(self,time):
        
        if isinstance(time,(tuple,list)):
            t1=min(time)
            t2=max(time)
        else:
            t1 = time-timedelta(hours=6)
            t2 = time+timedelta(hours=6)
        subRecon = self.recentered.loc[(self.recentered['time']>=t1) & \
                               (self.recentered['time']<t2)]
        return subRecon
    
    
    def findMission(self,time):
        
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
        for name in self.missiondata:
            t_start = min(self.missiondata[name]['time'])
            t_end = max(self.missiondata[name]['time'])
            if (t_start<t1<t_end) or (t_start<t2<t_end) or (t1<t_start<t2):
                selected.append(name)
        if len(selected)==0:
            print('There were no in-storm recon missions during this time')
        return selected


    #PLOT FUNCTION FOR RECON POINTS
    #Passed!
    def plot_points(self,recon_select=None,varname='wspd',domain="dynamic",ax=None,return_ax=False,cartopy_proj=None,**kwargs):
        
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
            "sfmr" - SFMR surface wind
            "wspd" - 30-second flight level wind 
            "pkwnd" - 10-second flight level wind
            "p_sfc" - extrapolated surface pressure
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic" - default - dynamically focuses the domain using the tornado track(s) plotted, 
            "north_atlantic" - North Atlantic Ocean basin, 
            "lonW/lonE/latS/latN" - Custom plot domain.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
            
        Additional Parameters
        ---------------------
        prop : dict
            Property of recon plot.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
        
        #Get plot data
        
        if recon_select is None:
            dfRecon = self.recentered
        elif isinstance(recon_select,pd.core.frame.DataFrame):
            dfRecon = recon_select
        elif isinstance(recon_select,dict):
            dfRecon = pd.DataFrame.from_dict(recon_select)
        elif isinstance(recon_select,str):
            dfRecon = self.missiondata[recon_select]
        else:
            dfRecon = self.__getSubTime(recon_select)
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj == None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_points(self.storm_obj,dfRecon,domain,varname=varname,\
                                              ax=ax,return_ax=return_ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax==True:
            return plot_info


    #PLOT FUNCTION FOR RECON HOVMOLLER
    #Passed!
    def plot_hovmoller(self,recon_select=None,varname='wspd',radlim=None,track_dict=None,\
                       ax=None,return_ax=False,**kwargs):
        
        r"""
        Creates a hovmoller plot of azimuthally-averaged recon data.
        
        Parameters
        ----------
        recon_select : Requested recon data
            pandas.DataFrame or dict,
            or datetime or list of start/end datetimes.
        varname : Variable to average and plot (e.g. 'wspd').
            String
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
            
        Additional Parameters
        ---------------------
        prop : dict
            Property of recon plot.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{'cmap':'category','levels':None})  

        #Get plot data
        
        if recon_select is None:
            dfRecon = self.recentered
        elif isinstance(recon_select,pd.core.frame.DataFrame):
            dfRecon = recon_select
        elif isinstance(recon_select,dict):
            dfRecon = pd.DataFrame.from_dict(recon_select)
        else:
            dfRecon = self.__getSubTime(recon_select)
        
        if track_dict is None:
            track_dict = self.storm_obj.dict
        
        iRecon = interpRecon(dfRecon,varname,radlim)
        Hov_dict = iRecon.interpHovmoller(track_dict)

        title = get_recon_title(varname)
        if prop['levels'] is None:
            prop['levels'] = np.arange(np.floor(np.nanmin(Hov_dict['hovmoller'])/10)*10,
                            np.ceil(np.nanmax(Hov_dict['hovmoller'])/10)*10+1,10)
        cmap,levels = get_cmap_levels(varname,prop['cmap'],prop['levels'],linear=True)
        
        time = Hov_dict['time']
        radius = Hov_dict['radius']
        vardata = Hov_dict['hovmoller']
        
        #Error check time
        time = [dt.strptime((i.strftime('%Y%m%d%H%M')),'%Y%m%d%H%M') for i in time]
        
        #Create plot        
        plt.figure(figsize=(9,11),dpi=150)
        ax=plt.subplot()
        cf=ax.contourf(radius,time,gfilt1d(vardata,sigma=3,axis=1),\
                     levels=levels,cmap=cmap)
        ax.axis([0,max(radius),min(time),max(time)])
        
        ax.yaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H'))
        
        ax.set_ylabel('UTC Time (MM-DD HH)')
        ax.set_xlabel('Radius (km)')
        plt.colorbar(cf,orientation='horizontal',pad=0.1)

        mlib.rcParams.update({'font.size': 16})
        
        #--------------------------------------------------------------------------------------
        
        title_left,title_right = ReconPlot().plot_hovmoller(self.storm_obj,Hov_dict,varname)
        ax.set_title(title_left,loc='left',fontsize=17,fontweight='bold')
        ax.set_title(title_right,loc='right',fontsize=13)
        
        #Return axis
        if return_ax:
            return ax


    #PLOT FUNCTION FOR RECON MAPS
    def plot_maps(self,recon_select=None,varname='wspd',track_dict=None,recon_stats=None,domain="dynamic",\
                  ax=None,return_ax=False,savetopath=None,cartopy_proj=None,**kwargs):
    
        #plot_time, plot_mission (only for dots)
        
        r"""
        Creates maps of interpolated recon data. 
        
        Parameters
        ----------
        recon_select : Requested recon data
            pandas.DataFrame or dict,
            or string referencing the mission name (e.g. '12_NOAA'), 
            or datetime or list of start/end datetimes.
        varname : str
            Variable to plot. Can be one of the following keys in recon_select dataframe:
            "sfmr" - SFMR surface wind
            "wspd" - 30-second flight level wind 
            "pkwnd" - 10-second flight level wind
            "p_sfc" - extrapolated surface pressure
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic" - default - dynamically focuses the domain around , 
            "north_atlantic" - North Atlantic Ocean basin, 
            "lonW/lonE/latS/latN" - Custom plot domain.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
            
        Additional Parameters
        ---------------------
        prop : dict
            Property of recon plot.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
        
        #Get plot data
        ONE_MAP = False
        if recon_select is None:
            dfRecon = self.recentered        
        elif isinstance(recon_select,pd.core.frame.DataFrame):
            dfRecon = recon_select
        elif isinstance(recon_select,dict):
            dfRecon = pd.DataFrame.from_dict(recon_select)
        elif isinstance(recon_select,str):
            dfRecon = self.missiondata[recon_select]
        else:
            dfRecon = self.__getSubTime(recon_select)
            if not isinstance(recon_select,(tuple,list)):
                ONE_MAP = True
        
        if track_dict is None:
            track_dict = self.storm_obj.dict
            
            #Error check for time dimension name
            if 'time' not in track_dict.keys():
                track_dict['time'] = track_dict['date']
                
        if ONE_MAP:
            clon = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lon'])
            clat = np.interp(mdates.date2num(recon_select),mdates.date2num(track_dict['time']),track_dict['lat'])
            track_dict = {'time':recon_select,'lon':clon,'lat':clat}
        
        iRecon = interpRecon(dfRecon,varname)
        Maps = iRecon.interpMaps(track_dict)
                
        titlename,units = get_recon_title(varname)
        
        if 'levels' not in prop.keys() or 'levels' in prop.keys() and prop['levels'] is None:
            prop['levels'] = np.arange(np.floor(np.nanmin(Maps['maps'])/10)*10,
                             np.ceil(np.nanmax(Maps['maps'])/10)*10+1,10)
        
        if not ONE_MAP:
            
            if savetopath is True:
                savetopath = f'{self.storm}{self.year}_{varname}_maps'
            try:
                os.system(f'mkdir {savetopath}')
            except:
                pass
            
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
                if i>0 and domain is 'dynamic':
                    d1 = {'n':Maps_sub['center_lat']+dlat,\
                          's':Maps_sub['center_lat']-dlat,\
                          'e':Maps_sub['center_lon']+dlon,\
                          'w':Maps_sub['center_lon']-dlon}
                else:
                    d1 = domain
                
                #Plot recon
                plot_ax,d0 = self.plot_obj.plot_maps(self.storm_obj,Maps_sub,varname,recon_stats,\
                                                    domain=d1,ax=ax,return_ax=True,return_domain=True,prop=prop,map_prop=map_prop)
                
                #Get domain dimensions from the first map
                if i==0:
                    dlat = .5*(d0['n']-d0['s'])
                    dlon = .5*(d0['e']-d0['w'])
                
                figs.append(plot_ax)
                
                if savetopath is not None:
                    plt.savefig(f'{savetopath}/{t.strftime("%Y%m%d%H%M")}',bbox_inches='tight')
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
            plot_info = self.plot_obj.plot_maps(self.storm_obj,Maps,varname,recon_stats,\
                                                domain,ax,return_ax,prop=prop,map_prop=map_prop)
            
            #Return axis
            if ax is not None or return_ax:
                return plot_info
            
 
    
    #PLOT FUNCTION FOR RECON SWATH
    def plot_swath(self,recon_select=None,varname='wspd',swathfunc=None,track_dict=None,radlim=None,\
                   domain="dynamic",ax=None,return_ax=False,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a map plot of a swath of interpolated recon data.
        
        Parameters
        ----------
        recon_select : Requested recon data
            pandas.DataFrame or dict,
            or string referencing the mission name (e.g. '12_NOAA'), 
            or datetime or list of start/end datetimes.
        varname : str
            Variable to plot. Can be one of the following keys in recon_select dataframe:
            "sfmr" - SFMR surface wind
            "wspd" - 30-second flight level wind 
            "pkwnd" - 10-second flight level wind
            "p_sfc" - extrapolated surface pressure
        swathfunc : function
            Function to operate on interpolated recon data.
            e.g., np.max, np.min, or percentile function
        domain : str
            Domain for the plot. Can be one of the following:
            "dynamic" - default - dynamically focuses the domain using the tornado track(s) plotted, 
            "north_atlantic" - North Atlantic Ocean basin, 
            "lonW/lonE/latS/latN" - Custom plot domain.
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        return_ax : bool
            If True, returns the axes instance on which the plot was generated for the user to further modify. Default is False.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
            
        Additional Parameters
        ---------------------
        prop : dict
            Property of recon plot.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
        
        #Get plot data
        if recon_select is None:
            dfRecon = self.recentered        
        elif isinstance(recon_select,pd.core.frame.DataFrame):
            dfRecon = recon_select
        elif isinstance(recon_select,dict):
            dfRecon = pd.DataFrame.from_dict(recon_select)
        elif isinstance(recon_select,str):
            dfRecon = self.missiondata[recon_select]
        else:
            dfRecon = self.__getSubTime(recon_select)

        if track_dict is None:
            track_dict = self.storm_obj.dict
        
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
        if cartopy_proj == None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_info = self.plot_obj.plot_swath(self.storm_obj,Maps,varname,swathfunc,track_dict,radlim,\
                                             domain,ax,return_ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None or return_ax==True:
            return plot_info
