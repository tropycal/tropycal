import numpy as np
from datetime import datetime as dt,timedelta
import pandas
import requests

from scipy.interpolate import interp1d
from scipy.ndimage.filters import minimum_filter
from geopy.distance import great_circle
import matplotlib.dates as mdates


class Recon:
    
    r"""
    Retrieve and analyze recon data.
    """

    #init class
    def __init__(self,stormtuple):
        
        self.url_prefix = 'http://tropicalatlantic.com/recon/recon.cgi?'
        self.storm = str(stormtuple[0])
        self.year = str(stormtuple[1])
        self.missiondata = self.allMissions()
        self.recentered = self.recenter()
        
    def getMission(self,agency,mission_num,url_mission=None):
        if url_mission==None:
            url_mission = f'{self.url_prefix}basin=al&year={self.year}&product=hdob&storm={self.storm}&mission={mission_num}&agency={agency}'
        content = np.array(requests.get(url_mission).content.decode("utf-8").split('\n'))
        obs = [line.split('\"')[1] for line in content if 'option value=' in line][::-1]
        for i,ob in enumerate(obs):
            url_ob = url_mission+'&ob='+ob
            data = pandas.read_html(url_ob)[0]
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
        return pandas.DataFrame.from_dict(data)

    def allMissions(self):
        url_storm = f'{self.url_prefix}basin=al&year={self.year}&storm={self.storm}&product=hdob'
        missions = pandas.read_html(url_storm)[0]
        missiondata={}
        timer_start = dt.now()
        print(f'--> Start reading in recon missions')
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
            if len(good[0])>=3:
                f = interp1d(inds[good], A[good],bounds_error=False,kind='cubic')
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
            #Only use mins if below 20th %ile of mission p_sfc data and when plane p is 500-900mb
            imin = [i for i in imin if 800<p_sfc_interp[i]<np.nanpercentile(data['p_sfc'],20) and \
                    500<data['plane_p'][i]<900]
        else:
            imin=[]
        data['iscenter'] = np.zeros(len(data['p_sfc']))
        for i in imin:
            data['iscenter'][i] = 1
        return data

    def recenter(self): 
        
        def stitchMissions():
            list_of_dfs=[]
            for name in self.missiondata:
                mission = self.missiondata[name]
                tmp = self.find_centers(mission)
                list_of_dfs.append( tmp )
            data_concat = pandas.concat(list_of_dfs,ignore_index=True)
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
            f = interp1d(mdates.date2num(centers['time']),centers['lon'],fill_value='extrapolate',kind='linear')
            interp_clon = f(mdates.date2num(data['time']))
            f = interp1d(mdates.date2num(centers['time']),centers['lat'],fill_value='extrapolate',kind='linear')
            interp_clat = f(mdates.date2num(data['time']))

            #Get x,y distance of each ob from coinciding interped center position
            data['xdist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
                (interp_clat[i],data['lon'][i]) ).kilometers* \
                [1,-1][int(data['lon'][i] < interp_clon[i])] for i in range(len(data))]
            data['ydist'] = [great_circle( (interp_clat[i],interp_clon[i]), \
                (data['lat'][i],interp_clon[i]) ).kilometers* \
                [1,-1][int(data['lat'][i] < interp_clat[i])] for i in range(len(data))]
            
        print('--> Completed recentering recon data (%.2f seconds)' % (dt.now()-timer_start).total_seconds())
        return data
        


    
    