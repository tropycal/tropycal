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
    
    def __init__(self,dfRecon,varname,radlim=None):
        
        self.dfRecon = dfRecon
        self.varname = varname
        if radlim is None:
            self.radlim = 200 #km
        else:
            self.radlim = radlim
    
    
    def interpPol(self):
        
        # read in recon data
        data = [k for i,j,k in zip(self.dfRecon['xdist'],self.dfRecon['ydist'],self.dfRecon[self.varname]) if not np.isnan([i,j,k]).any()]
        path = [(i,j) for i,j,k in zip(self.dfRecon['xdist'],self.dfRecon['ydist'],self.dfRecon[self.varname]) if not np.isnan([i,j,k]).any()]

        # polar
        def cart2pol(x, y, offset=0):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return(rho, phi+offset)
        
        pol_path = [cart2pol(*p) for p in path]

        #Give more weight to newest obs in data subset
#        time = [mdates.date2num(i) for i,j in zip(self.dfRecon['time'],self.dfRecon[self.varname]) if not np.isnan(j).any()]
#        time_std = [(t-np.min(time))/np.std(time) for t in time]
#        wts = [np.array([self._cdist(p1,p2,t1) for p1,t1 in zip(pol_path,time_std)]) for p2 in pol_path]
#        data = [np.sum(np.array(data)*w)/np.sum(w) for w in wts]
        
        pol_path_wrap = [cart2pol(*p,offset=-2*np.pi) for p in path]+pol_path+\
                    [cart2pol(*p,offset=2*np.pi) for p in path]
        data_wrap = np.concatenate([data]*3)
        
        grid_rho, grid_phi = np.meshgrid(np.arange(0,self.radlim+.1,.5),np.linspace(-np.pi,np.pi,181))
    
        grid_z_pol = griddata(pol_path_wrap,data_wrap,(grid_rho,grid_phi),method='linear')
        rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]
        filleye = np.where((grid_rho<rmw) & (np.isnan(grid_z_pol)))
        grid_z_pol[filleye]=np.nanmin(grid_z_pol[np.where(grid_rho<rmw)])
    
        return grid_rho, grid_phi, grid_z_pol      
        
    
    def interpCart(self):
        
        grid_rho, grid_phi, grid_z_pol = self.interpPol()
        rmw = grid_rho[0,np.nanargmax(np.mean(grid_z_pol,axis=0))]
        
        grid_z_pol_wrap = np.concatenate([grid_z_pol]*3)
        
        # smooth
        grid_z_pol_final = np.array([gfilt(grid_z_pol_wrap,(6,3+abs(r-rmw)/10))[:,i] \
                                     for i,r in enumerate(grid_rho[0,:])]).T[len(grid_phi):2*len(grid_phi)]
        
        # back to cartesian
        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return(x, y)
        
        pinterp_grid = [pol2cart(i,j) for i,j in zip(grid_rho.flatten(),grid_phi.flatten())]
        pinterp_z = grid_z_pol_final.flatten()
        
        grid_x, grid_y = np.meshgrid(np.linspace(-self.radlim,self.radlim,self.radlim*2+1),\
                                     np.linspace(-self.radlim,self.radlim,self.radlim*2+1))
        grid_z = griddata(pinterp_grid,pinterp_z,(grid_x,grid_y),method='linear')
    
        return grid_x, grid_y, grid_z
    

    def interpHovmoller(self,target_track,window=4,align='backward'):
    
        tmpRecon = self.dfRecon.copy()
        window = timedelta(seconds=int(window*3600))
        
        if 'time' not in target_track.keys():
            target_track['time']=target_track['date']
        
        centerTimes=tmpRecon[tmpRecon['iscenter']==1]['time']
        spaceInterpTimes=[t+window/2 for t in centerTimes]
        trackTimes=[t for t in target_track['time'] if min(spaceInterpTimes)<t<max(spaceInterpTimes)]
        
        spaceInterpData={}
        for time in spaceInterpTimes:
            self.dfRecon = tmpRecon[(tmpRecon['time']>time-window) & (tmpRecon['time']<=time)]
            print(time)
            grid_rho, grid_phi, grid_z_pol = self.interpPol()
            grid_azim_mean = np.mean(grid_z_pol,axis=0)
            spaceInterpData[time] = grid_azim_mean
        self.dfRecon = tmpRecon
        reconArray=np.array([i for i in spaceInterpData.values()])

        #Time duration to interpolate
        start_time = dt.now()
        print("--> Starting to interpolate to target_track times")
        newTimes = np.arange(mdates.date2num(trackTimes[0]),mdates.date2num(trackTimes[-1])+1e-3,1/48)    
        oldTimes = mdates.date2num(spaceInterpTimes)
        reconTimeInterp=np.apply_along_axis(lambda x: np.interp(newTimes,oldTimes,x),
                                 axis=0,arr=reconArray)
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed interpolation ({tsec} seconds)")
        
        self.rmw = grid_rho[0,np.nanargmax(reconTimeInterp,axis=1)]
        self.Hovmoller = {'time':mdates.num2date(newTimes),'radius':grid_rho[0,:],'hovmoller':reconTimeInterp}
        return self.Hovmoller
        

    def interpMaps(self,target_track,interval=0.5,window=6,align='center',stat_vars=None):
        
        tmpRecon = self.dfRecon.copy()
        window = timedelta(seconds=int(window*3600))
 
        if 'time' not in target_track.keys():
            target_track['time']=target_track['date']
       
        if isinstance(target_track['time'],(tuple,list,np.ndarray)):
            centerTimes=tmpRecon[tmpRecon['iscenter']==1]['time']
            spaceInterpTimes=[t for t in centerTimes]
            trackTimes=[t for t in target_track['time'] if min(spaceInterpTimes)-window/2<t<max(spaceInterpTimes)+window/2]
        else:
            spaceInterpTimes=list([target_track['time']])
            trackTimes=spaceInterpTimes.copy()
        
        spaceInterpData={}
        recon_stats=None
        if stat_vars is not None:
            recon_stats={name:[] for name in stat_vars.keys()}
        for time in spaceInterpTimes:
            print(time)
            self.dfRecon = tmpRecon[(tmpRecon['time']>time-window/2) & (tmpRecon['time']<=time+window/2)]
            grid_x,grid_y,grid_z = self.interpCart()
            spaceInterpData[time] = grid_z
            if stat_vars is not None:
                for name in stat_vars.keys():
                    recon_stats[name].append(stat_vars[name](self.dfRecon[name]))
        self.dfRecon = tmpRecon        
        reconArray=np.array([i for i in spaceInterpData.values()])

        #Time duration to interpolate
        start_time = dt.now()
        print("--> Starting to interpolate to target_track times")
        
        if len(trackTimes)>1:
            newTimes = np.arange(mdates.date2num(trackTimes[0]),mdates.date2num(trackTimes[-1])+interval/24,interval/24)    
            oldTimes = mdates.date2num(spaceInterpTimes)
            reconTimeInterp=np.apply_along_axis(lambda x: np.interp(newTimes,oldTimes,x),
                                 axis=0,arr=reconArray)
            clon = np.interp(newTimes,mdates.date2num(target_track['time']),target_track['lon'])
            clat = np.interp(newTimes,mdates.date2num(target_track['time']),target_track['lat'])
        else:
            newTimes = mdates.date2num(trackTimes)[0]
            reconTimeInterp = reconArray[0]
            clon = target_track['lon']
            clat = target_track['lat']

        if stat_vars is not None:
            for varname in recon_stats.keys():
                recon_stats[varname] = np.interp(newTimes,oldTimes,recon_stats[varname])
            
        #Determine time elapsed
        time_elapsed = dt.now() - start_time
        tsec = str(round(time_elapsed.total_seconds(),2))
        print(f"--> Completed interpolation ({tsec} seconds)")            
            

    
        self.Maps = {'time':mdates.num2date(newTimes),'grid_x':grid_x,'grid_y':grid_y,'maps':reconTimeInterp,
                           'center_lon':clon,'center_lat':clat,'stats':recon_stats}
        return self.Maps
    
    
    @staticmethod
    def _cdist(pt1,pt2,t_sd):
        dx = pt1[0]-pt2[0]
        dy = min((pt1[1]-pt2[1])%(2*np.pi),(pt2[1]-pt1[1])%(2*np.pi))
        dist2 = (1-np.cos(dy))**2 + (dx*.2)**2
        w1 = np.exp(-dist2*5)
        w = [0,w1+.1*t_sd**3][int(w1>.8)]
        return w

    @staticmethod
    def _interpFunc(data1, times1, times2):
        f = interp1d(mdates.date2num(times1),data1)
        data2 = f(mdates.date2num(times2))
        return data2



#------------------------------------------------------------------------------
# TOOLS FOR PLOTTING
#------------------------------------------------------------------------------

#Title generator
        
def get_recon_title(varname):
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
    return titlename,unitname

#Convert between wind and category
def wind2cat(wind):
    w2c = {5:-1,\
           34:0,\
           64:1,\
           83:2,\
           96:3,\
           113:4,\
           137:5}
    return w2c[wind]

def cat2wind(cat):
    c2w = {-1:5,\
           0:34,\
           1:64,\
           2:83,\
           3:96,\
           4:113,\
           5:137}
    return c2w[cat]


def category_color(vmax):
    
    if isinstance(vmax,str) == True:
        vmax = category_to_wind(vmax)
    
    if vmax < 5:
        return '#FFFFFF'
    elif vmax < 34:
        return '#8FC2F2' #'#7DB7ED'
    elif vmax < 64:
        return '#3185D3'
    elif vmax < 83:
        return '#FFFF00'
    elif vmax < 96:
        return '#FF9E00'
    elif vmax < 113:
        return '#DD0000'
    elif vmax < 137:
        return '#FF00FC'
    else:
        return '#8B0088'

def make_colormap(colors,whiten=0):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, ColorConverter
    
    z  = np.array(sorted(colors.keys()))
    n  = len(z)
    z1 = min(z)
    zn = max(z)
    x0 = (z - z1) / (zn - z1)
    
    CC = ColorConverter()
    R = []
    G = []
    B = []
    for i in range(n):
        Ci = colors[z[i]]
        if type(Ci) == str:
            RGB = CC.to_rgb(Ci)
        else:
            RGB = Ci
        R.append(RGB[0] + (1-RGB[0])*whiten)
        G.append(RGB[1] + (1-RGB[1])*whiten)
        B.append(RGB[2] + (1-RGB[2])*whiten)
    
    cmap_dict = {}
    cmap_dict['red']   = [(x0[i],R[i],R[i]) for i in range(len(R))]
    cmap_dict['green'] = [(x0[i],G[i],G[i]) for i in range(len(G))]
    cmap_dict['blue']  = [(x0[i],B[i],B[i]) for i in range(len(B))]
    mymap = LinearSegmentedColormap('mymap',cmap_dict)
    
    return mymap

def get_cmap_levels(varname,x,clevs,linear=False):
    
    colors = ({0.0:'w',
           2/8:'#72FF78',
           3/8:'#EFFF0F',
           5/8:'r',
           6/8:'#6012D5',
           1.0:'#85FFFF'})
    cmap_fischer = make_colormap(colors)
    
    if x=='category':
        if varname in ['sfmr','fl_to_sfc']:
#            clevs = [34,64,83,96,113,137,200]
#            colors = ['#8FC2F2','#3185D3','#FFFF00','#FF9E00','#DD0000','#FF00FC','#8B0088']
            clevs = [cat2wind(c) for c in range(-1,6)]+[200]
            if linear:
                colorstack = [mcolors.to_rgba(category_color(lev)) \
                               for c,lev in enumerate(clevs[:-1]) for _ in range(clevs[c+1]-clevs[c])]
            else:
                colorstack = [mcolors.to_rgba(category_color(lev)) \
                               for c,lev in enumerate(clevs[:-1])]
            cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colorstack)
        else:
            warnings.warn('Saffir Simpson category colors allowed only for surface winds')
            x = 'plasma'
    if x!='category':
        if isinstance(x,str):
            cmap = mlib.cm.get_cmap(x)
            norm = mlib.colors.Normalize(vmin=0, vmax=len(clevs)-1)
            colors = cmap(norm(np.arange(len(clevs))))
        elif isinstance(x,list):
            colors = x
        else:
            norm = mlib.colors.Normalize(vmin=0, vmax=len(clevs)-1)
            colors = x(norm(np.arange(len(clevs))))
        cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',colors)
    return cmap,clevs
    