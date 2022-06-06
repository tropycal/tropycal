import urllib3
import requests
import pandas as pd
from datetime import datetime as dt,timedelta
import matplotlib.pyplot as plt

from .plot import *
from .tools import *

class RealtimeRecon():
    
    r"""
    Creates an instance of a RealtimeRecon object.
    
    Parameters
    ----------
    hours : int
        Number of hours to search back for recon missions. Default is 12 hours. Max allowed is 48 hours.
    
    Returns
    -------
    RealtimeRecon
        Instance of a RealtimeRecon object.
    
    Notes
    -----
    The ``RealtimeRecon`` Class and accompanying ``Mission`` class make up the realtime part of the recon module. Unlike the ``ReconDataset`` Class and its accompanying ``hdobs``, ``dropsondes`` and ``vdms`` classes which are **storm-centric**, the realtime recon functionality is **mission-centric**.
    
    This mission-centric functionality means realtime recon missions lack the storm-centering functionality that the more comprehensive full recon functionality has, but is also much faster at reading recon data for realtime purposes and includes non-tropical cyclone recon missions.
    
    The following example shows how to use realtime recon functionality. First, create an instance of ``RealtimeRecon``, which retrieves all active missions within the specified hour window (default is the most recent 12 hours)
    
    .. code-block:: python
    
        from tropycal import recon
        realtime_obj = recon.RealtimeRecon()
        
    We can now quickly retrieve a list of all active mission IDs:
    
    >>> realtime_obj.get_mission_ids()
    ['AF306-0801A-ALEX', 'AF303-0901A-ALEX']
    
    Now use the ``get_mission()`` method to retrieve a Mission object containing all available data (HDOBs, VDMs & dropsondes) for this mission. The resulting object can access all of the methods and attributes of the ``Mission`` class.
    
    .. code-block:: python
        
        mission = realtime_obj.get_mission('AF303-0901A-ALEX')
    
    """
    
    def __repr__(self):
        summary = ["<tropycal.recon.RealtimeRecon>"]

        #Add dataset summary
        summary.append("Dataset Summary:")
        summary.append(f'{" "*4}Numbers of active missions: {len(self.missions)}')
        
        return "\n".join(summary)
    
    def __init__(self,hours=12):

        #Error check
        self.hours = hours
        if hours > 48 or hours <= 0:
            raise ValueError("Maximum allowed search is 48 hours back.")
        
        #Start timing
        print("--> Searching for active missions")
        timer_start = dt.now()

        #Set URLs for reading data
        self.urls = {
            'hdobs':f'https://www.nhc.noaa.gov/archive/recon/{dt.utcnow().year}/AHONT1/',
            'dropsondes':f'https://www.nhc.noaa.gov/archive/recon/{dt.utcnow().year}/REPNT3/',
            'vdms':f'https://www.nhc.noaa.gov/archive/recon/{dt.utcnow().year}/REPNT2/'
        }

        #Start time set by hour window
        start_time_request = dt.utcnow() - timedelta(hours=hours)
        start_time = dt.utcnow() - timedelta(hours=hours+12)
        self.start_time_request = start_time_request

        #Retrieve list of files in URL and filter by storm dates
        files = {'hdobs':[],'dropsondes':[],'vdms':[]}
        for key in files.keys():
            page = requests.get(self.urls[key]).text
            content = page.split("\n")
            file_list = []
            for line in content:
                if ".txt" in line: file_list.append(((line.split('txt">')[1]).split("</a>")[0]).split("."))
            del content
            file_list = sorted([i for i in file_list if dt.strptime(i[1][:10],'%Y%m%d%H') >= start_time],key=lambda x: x[1])
            files[key] = [self.urls[key]+'.'.join(l) for l in file_list]
        self.files = files

        #Retrieve all active missions & read HDOBs
        urllib3.disable_warnings()
        http = urllib3.PoolManager()
        self.missions = {}
        for file in files['hdobs']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = '-'.join((content_split[3].replace("  "," ")).split(" ")[:3])
            if mission_id not in self.missions:
                self.missions[mission_id] = {'hdobs':decode_hdob(content),
                                        'vdms':[],
                                        'dropsondes':[],
                                        'aircraft':mission_id.split("-")[0],
                                        'storm_name':mission_id.split("-")[2]
                                       }
            else:
                self.missions[mission_id]['hdobs'] = pd.concat([self.missions[mission_id]['hdobs'],decode_hdob(content)])

        #Retrieve VDMs
        for file in files['vdms']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = ['-'.join(i.split("U. ")[1].replace("  "," ").split(" ")[:3]) for i in content_split if i[:2] == "U."][0]
            date = dt.strptime((file.split('.')[-2])[:8],'%Y%m%d')
            blank, data = decode_vdm(content,date)
            if mission_id in self.missions.keys(): self.missions[mission_id]['vdms'].append(data)

        #Retrieve dropsondes
        for file in files['dropsondes']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = ['-'.join(i.split("61616 ")[1].replace("  "," ").split(" ")[:3]) for i in content_split if i[:5] == "61616"][0]
            date = dt.strptime((file.split('.')[-2])[:8],'%Y%m%d')
            blank, data = decode_dropsonde(content,date)
            if mission_id in self.missions.keys(): self.missions[mission_id]['dropsondes'].append(data)

        #Temporally filter missions
        keys = [k for k in self.missions.keys()]
        for key in keys:
            end_date = pd.to_datetime(self.missions[key]['hdobs']['time'].values[-1])
            if end_date < start_time_request: del self.missions[key]
        
        print(f"--> Completed retrieving active missions ({(dt.now()-timer_start).total_seconds():.1f} seconds)")
        
    def update(self):
        
        r"""
        Updates RealtimeRecon with the latest available data.
        
        Notes
        -----
        This function has no return value, but simply updates RealtimeRecon with the latest available recon data.
        """
        
        #Start timing
        timer_start = dt.now()
        
        #Start time set by hour window
        start_time = dt.utcnow() - timedelta(hours=24)
        if start_time < self.start_time_request: start_time = self.start_time_request

        #Retrieve list of files in URL and filter by storm dates
        files = {'hdobs':[],'dropsondes':[],'vdms':[]}
        for key in files.keys():
            page = requests.get(self.urls[key]).text
            content = page.split("\n")
            file_list = []
            for line in content:
                if ".txt" in line: file_list.append(((line.split('txt">')[1]).split("</a>")[0]).split("."))
            del content
            file_list = sorted([i for i in file_list if dt.strptime(i[1][:10],'%Y%m%d%H') >= start_time],key=lambda x: x[1])
            files[key] = [self.urls[key]+'.'.join(l) for l in file_list if self.urls[key]+'.'.join(l) not in self.files[key]]
            self.files[key] += files[key]
        
        #Retrieve all active missions & read HDOBs
        urllib3.disable_warnings()
        http = urllib3.PoolManager()
        for file in files['hdobs']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = '-'.join((content_split[3].replace("  "," ")).split(" ")[:3])
            if mission_id not in self.missions:
                self.missions[mission_id] = {'hdobs':decode_hdob(content),
                                        'vdms':[],
                                        'dropsondes':[],
                                        'aircraft':mission_id.split("-")[0],
                                        'storm_name':mission_id.split("-")[2]
                                       }
            else:
                self.missions[mission_id]['hdobs'] = pd.concat([self.missions[mission_id]['hdobs'],decode_hdob(content)])

        #Retrieve VDMs
        for file in files['vdms']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = ['-'.join(i.split("U. ")[1].replace("  "," ").split(" ")[:3]) for i in content_split if i[:2] == "U."][0]
            date = dt.strptime((file.split('.')[-2])[:8],'%Y%m%d')
            blank, data = decode_vdm(content,date)
            if mission_id in self.missions.keys(): self.missions[mission_id]['vdms'].append(data)

        #Retrieve dropsondes
        for file in files['dropsondes']:

            #Retrieve content
            response = http.request('GET',file)
            content = response.data.decode('utf-8')
            content_split = content.split("\n")

            #Construct mission ID
            mission_id = ['-'.join(i.split("61616 ")[1].replace("  "," ").split(" ")[:3]) for i in content_split if i[:5] == "61616"][0]
            date = dt.strptime((file.split('.')[-2])[:8],'%Y%m%d')
            blank, data = decode_dropsonde(content,date)
            if mission_id in self.missions.keys(): self.missions[mission_id]['dropsondes'].append(data)
        
        #Temporally filter missions
        keys = [k for k in self.missions.keys()]
        for key in keys:
            end_date = pd.to_datetime(self.missions[key]['hdobs']['time'].values[-1])
            start_time_request = dt.utcnow() - timedelta(hours=self.hours)
            if end_date < start_time_request: del self.missions[key]
        
        print(f"--> Completed updating mission data ({(dt.now()-timer_start).total_seconds():.1f} seconds)")
    
    def get_mission(self,mission_id):
        
        r"""
        Retrieve a Mission object given the mission ID.
        
        Parameters
        ----------
        mission_id : str
            String denoting requested mission ID. All active mission IDs can be retrieved using ``get_mission_ids()``.
        
        Returns
        -------
        tropycal.recon.Mission
            An instance of a Mission object for this mission.
        """
        
        return Mission(self.missions[mission_id],mission_id)
    
    def get_mission_ids(self,storm_name=None):
        
        r"""
        Retrieve a list of all active mission IDs.
        
        Parameters
        ----------
        storm_name : str, optional
            Storm name (case-insensitive) to filter the search by. If None, all active missions are searched.
        
        Returns
        -------
        list
            List containing all active mission IDs.
        """
        
        mission_ids = [key for key in self.missions.keys()]
        if storm_name is not None:
            mission_ids = [i for i in mission_ids if i.split("-")[2].lower() == storm_name.lower()]
        
        return mission_ids
    
class PseudoStorm():
    
    r"""
    Creates a dummy Storm object for functions that require a Storm object.
    """
    
    def __init__(self):
        
        self.dict = {'type':['TS'],'vmax':[50],'wmo_basin':'north_atlantic','name':'Test'}

class Mission():
    
    r"""
    Creates an instance of a Mission object.
    
    Parameters
    ----------
    data : dict
        Dictionary containing mission data. This is passed automatically from ``RealtimeRecon.get_mission()``.
    
    Returns
    -------
    tropycal.recon.Mission
        An instance of a Mission object.
    
    Notes
    -----
    Mission objects are retrieved directly from ``RealtimeRecon`` objects, provided a mission ID string. For this example below, we'll retrieve the latest mission available:
    
    .. code-block:: python
    
        #Read in all recent mission data
        from tropycal import recon
        realtime_obj = recon.RealtimeRecon()
        
        #Get latest mission ID
        latest_mission_id = realtime_obj.get_mission_ids()[-1]
        
        #Retrieve an instance of Mission
        mission = realtime_obj.get_mission(latest_mission_id)
        
    This instance of Mission can now access all of the attributes and methods of the Mission class. Note that since the realtime recon functionality is mission-centric, not storm-centric, much of the functionality available in the storm-centric recon classes is not available here, though some of the plotting functionality (e.g., plotting HDOB points or dropsondes) is available.
    """
    
    def __repr__(self):
        summary = ["<tropycal.recon.Mission>"]
        
        #Find maximum wind and minimum pressure
        max_wspd = np.nanmax(self.hdobs['wspd'])
        max_pkwnd = np.nanmax(self.hdobs['pkwnd'])
        max_sfmr = np.nanmax(self.hdobs['sfmr'])
        min_psfc = np.nanmin(self.hdobs['p_sfc'])
        time_range = [pd.to_datetime(t) for t in (np.nanmin(self.hdobs['time']),np.nanmax(self.hdobs['time']))]

        #Add summary text
        emdash = '\u2014'
        summary_keys = {'Dropsondes':len(self.dropsondes),
                        'VDMs':len(self.vdms),
                        'Time range':f"{time_range[0]:%b-%d %H:%M} {emdash} {time_range[1]:%b-%d %H:%M}",
                        'Max 30sec flight level wind':f"{max_wspd} knots",
                        'Max 10sec flight level wind':f"{max_pkwnd} knots",
                        'Max SFMR wind':f"{max_sfmr} knots",
                        'Min surface pressure':f"{min_psfc} hPa"}
        
        #Add text to output
        summary.append("Mission Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{summary_keys[key]}')

        return "\n".join(summary)
    
    def __init__(self,data,mission_id):
        
        #Retrieve variables
        self.vdms = data['vdms']
        self.hdobs = data['hdobs']
        self.dropsondes = data['dropsondes']
        
        #Retrieve attributes
        self.aircraft = data['aircraft']
        self.storm_name = data['storm_name']
        self.mission_id = mission_id
    
    def get_hdobs(self):
        
        r"""
        Returns High Density Observations (HDOBs) for this mission.
        
        Returns
        -------
        Pandas.DataFrame
            DataFrame containing HDOBs entries for this mission.
        """
        
        return self.hdobs
        
    def get_dropsondes(self):
        
        r"""
        Returns dropsondes for this mission.
        
        Returns
        -------
        dict
            Dictionary containing dropsonde data.
        """
        
        return self.dropsondes
    
    def get_vdms(self):
        
        r"""
        Returns Vortex Data Messages (VDMs) for this mission.
        
        Returns
        -------
        dict
            Dictionary containing VDM data.
        """
        
        return self.vdms
    
    def plot_points(self,varname='wspd',domain="dynamic",ax=None,cartopy_proj=None,**kwargs):
        
        r"""
        Creates a plot of High Density Observations (HDOBs) data points.
        
        Parameters
        ----------
        varname : str
            Variable to plot. Can be one of the following keys in dataframe:
            
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
        
        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        """
        
        #Pop kwargs
        prop = kwargs.pop('prop',{})
        map_prop = kwargs.pop('map_prop',{})
                
        #Get plot data
        dfRecon = self.hdobs
        
        #Create instance of plot object
        self.plot_obj = ReconPlot()
        
        #Create cartopy projection
        if cartopy_proj is None:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            cartopy_proj = self.plot_obj.proj
        
        #Plot recon
        plot_ax = self.plot_obj.plot_points(PseudoStorm(),dfRecon,domain,varname=varname,radlim=None,ax=ax,prop=prop,map_prop=map_prop)
        
        #Edit title
        plot_ax.set_title(f"Mission ID: {self.mission_id}",loc='left',fontsize=17,fontweight='bold')
        
        #Return axis
        return plot_ax
        
    def plot_skewt(self,number=None):
        
        r"""
        Plot a single dropsonde Skew-T for a given dropsonde number.
        
        Parameters
        ----------
        number : int
            Number of dropsonde during mission. If None (default), plots the latest available dropsonde.
        
        Returns
        -------
        fig
            Figure instance containing the plot.
        """
        
        #Retrieve data for dropsonde number
        try:
            if number == -1: number = len(self.dropsondes)
            if number is None: number = len(self.dropsondes)
            data = [self.dropsondes[number-1]]
        except:
            raise ValueError("Requested dropsonde number not available.")
        
        #Format title string
        title_string = f'Mission ID: {self.mission_id}\nDropsonde #{number}'
        
        return plot_skewt(data,title_string)
    
    def plot_time_series(self,varname=('p_sfc','wspd'),time=None,realtime=False,**kwargs):
        
        r"""
        Plots a time series of one or two variables on an axis.
        
        Parameters
        ----------
        varname : str or tuple
            If one variable to plot, varname is a string of the variable name. If two variables to plot, varname is a tuple of the left and right variable names, respectively. Available varnames are:
            
            * **p_sfc** - Mean Sea Level Pressure (hPa)
            * **temp** - Flight Level Temperature (C)
            * **dwpt** - Flight Level Dewpoint (C)
            * **wspd** - Flight Level Wind (kt)
            * **sfmr** - Surface Wind (kt)
            * **pkwnd** - Peak Wind Gust (kt)
            * **rain** - Rain Rate (mm/hr)
            * **plane_z** - Geopotential Height (m)
            * **plane_p** - Pressure (hPa)
        time : tuple
            Tuple of start and end times (datetime.datetime) to plot. If None, all times available are plotted.
        realtime : bool
            If True, the most recent 2 hours of the mission will plot, overriding the time argument. Default is False.
        
        Other Parameters
        ----------------
        left_prop : dict
            Dictionary of properties for the left line. Scroll down for more information.
        right_prop : dict
            Dictionary of properties for the right line. Scroll down for more information.
        
        Returns
        -------
        ax
            Instance of axes containing the plot is returned.
        
        Notes
        -----
        The following properties are available for customizing the plot, via ``left_prop`` and ``right_prop``.

        .. list-table:: 
           :widths: 25 75
           :header-rows: 1

           * - Property
             - Description
           * - ms
             - Marker size. If zero, none will be plotted. Default is zero.
           * - color
             - Color of lines (and markers if used). Default varies per varname.
           * - linewidth
             - Line width. Default is 1.0.
        """
        
        #Pop kwargs
        left_prop = kwargs.pop('left_prop',{})
        right_prop = kwargs.pop('right_prop',{})
        
        #Retrieve variables
        twin_ax = False
        if isinstance(varname,tuple):
            varname_right = varname[1]
            varname = varname[0]
            twin_ax = True
            varname_right_info = time_series_plot(varname_right)
        varname_info = time_series_plot(varname)
        
        #Get data
        df = self.hdobs
        
        #Filter by time or realtime flag
        if realtime:
            end_time = pd.to_datetime(df['time'].values[-1])
            df = df.loc[(df['time'] >= end_time-timedelta(hours=2)) & (df['time'] <= end_time)]
        elif time is not None:
            df = df.loc[(df['time'] >= time[0]) & (df['time'] <= time[1])]
        if len(df) == 0: raise ValueError("Time range provided is invalid.")
        
        #Filter by default kwargs
        left_prop_default = {'ms':0,'color':varname_info['color'],'linewidth':1}
        for key in left_prop.keys(): left_prop_default[key] = left_prop[key]
        left_prop = left_prop_default
        if twin_ax:
            right_prop_default = {'ms':0,'color':varname_right_info['color'],'linewidth':1}
            for key in right_prop.keys(): right_prop_default[key] = right_prop[key]
            right_prop = right_prop_default
        
        #----------------------------------------------------------------------------------
        
        #Create figure
        fig,ax = plt.subplots(figsize=(9,6),dpi=200)
        if twin_ax:
            ax.grid(axis='x')
        else:
            ax.grid()
        
        #Plot line
        line1 = ax.plot(df['time'],df[varname],color=left_prop['color'],linewidth=left_prop['linewidth'],label=varname_info['name'])
        ax.set_ylabel(varname_info['full_name'])
        
        #Plot dots
        if left_prop['ms'] >= 1:
            plot_times = df['time'].values
            plot_var = df[varname].values
            plot_times = [plot_times[i] for i in range(len(plot_times)) if varname not in df['flag'].values[i]]
            plot_var = [plot_var[i] for i in range(len(plot_var)) if varname not in df['flag'].values[i]]
            ax.plot(plot_times,plot_var,'o',color=left_prop['color'],ms=left_prop['ms'])
        
        #Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%Mz\n%m/%d'))
        
        #Add twin axis
        if twin_ax:
            ax2 = ax.twinx()
            
            #Plot line
            line2 = ax2.plot(df['time'],df[varname_right],color=right_prop['color'],linewidth=right_prop['linewidth'],label=varname_right_info['name'])
            ax2.set_ylabel(varname_right_info['full_name'])
            
            #Plot dots
            if right_prop['ms'] >= 1:
                plot_times = df['time'].values
                plot_var = df[varname_right].values
                plot_times = [plot_times[i] for i in range(len(plot_times)) if varname_right not in df['flag'].values[i]]
                plot_var = [plot_var[i] for i in range(len(plot_var)) if varname_right not in df['flag'].values[i]]
                ax2.plot(plot_times,plot_var,'o',color=right_prop['color'],ms=right_prop['ms'])

            #Add legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines,labels)
        
            #Special handling if both are in units of Celsius
            same_unit = False
            if varname in ['temp','dwpt'] and varname_right in ['temp','dwpt']: same_unit = True
            if varname in ['sfmr','wspd','pkwnd'] and varname_right in ['sfmr','wspd','pkwnd']: same_unit = True
            if same_unit:
                min_val = np.nanmin([np.nanmin(df[varname]),np.nanmin(df[varname_right])])
                max_val = np.nanmax([np.nanmax(df[varname]),np.nanmax(df[varname_right])])*1.05
                min_val = min_val * 1.05 if min_val < 0 else min_val * 0.95
                if np.isnan(min_val): min_val = 0
                if np.isnan(max_val): max_val = 0
                if min_val == max_val:
                    min_val = 0
                    max_val = 10
                ax.set_ylim(min_val,max_val)
                ax2.set_ylim(min_val,max_val)
        
        #Add titles
        title_string = f"\nRecon Aircraft HDOBs\nMission ID: {self.mission_id}"
        ax.set_title(title_string,loc='left',fontweight='bold')
        ax.set_title("Plot generated using Tropycal",fontsize=8,loc='right')
        
        #Return plot
        return ax