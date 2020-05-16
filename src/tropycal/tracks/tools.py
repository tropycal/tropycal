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

def findvar(cmd,thresh):
    cmd=cmd.lower()
    if cmd.find('count')>=0 or cmd.find('num')>=0:
        return thresh,'date'
    if cmd.find('wind')>=0 or cmd.find('vmax')>=0:
        if cmd.find('change')>=0:
            try:
                thresh['dt_window'] = int(''.join([c for i,c in enumerate(cmd) \
                      if c.isdigit() and i>cmd.find('hour')-4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh,'dvmax_dt'
        else:
            return thresh,'vmax'
    elif cmd.find('pressure')>=0 or cmd.find('slp')>=0:
        if cmd.find('change')>=0:
            try:
                thresh['dt_window'] = int(''.join([c for i,c in enumerate(cmd) \
                      if c.isdigit() and i>cmd.find('hour')-4]))
            except:
                raise RuntimeError("Error: specify time interval (hours)")
            return thresh,'dmslp_dt'
        else:
            return thresh,'mslp'
    elif cmd.find('heading')>=0 or cmd.find('movement')>=0 or cmd.find('motion')>=0:
        return thresh,('dx_dt','dy_dt')
    else:
        raise RuntimeError("Error: Could not decipher variable")
        
def findfunc(cmd,thresh):
    cmd=cmd.lower()
    if cmd.find('max')==0:
        return thresh,lambda x: np.nanmax(x)
    if cmd.find('min')==0:
        return thresh,lambda x: np.nanmin(x)
    elif cmd.find('mean')>=0 or cmd.find('average')>=0 or cmd.find('avg')>=0:
        thresh['sample_min']=max([5,thresh['sample_min']])
        return thresh,lambda x: np.nanmean(x)
    elif cmd.find('percentile')>=0:
        ptile = int(''.join([c for i,c in enumerate(cmd) if c.isdigit() and i<cmd.find('percentile')]))
        thresh['sample_min']=max([5,thresh['sample_min']])
        return thresh,lambda x: np.nanpercentile(x,ptile)
    elif cmd.find('count')>=0 or cmd.find('num')>=0:
        return thresh,lambda x: len(x)
    else:
        raise RuntimeError("Error: Could not decipher function")

def construct_title(thresh):
    plot_subtitle = []
    gteq = u"\u2265"
    lteq = u"\u2264"
    if not np.isnan(thresh['sample_min']):
        plot_subtitle.append(f"{gteq} {thresh['sample_min']} storms/bin")
    else:
        thresh['sample_min']=0
        
    if not np.isnan(thresh['v_min']):
        plot_subtitle.append(f"{gteq} {thresh['v_min']}kt")
    else:
        thresh['v_min']=0
        
    if not np.isnan(thresh['p_max']):
        plot_subtitle.append(f"{lteq} {thresh['p_max']}hPa")            
    else:
        thresh['p_max']=9999

    if not np.isnan(thresh['dv_min']):
        plot_subtitle.append(f"{gteq} {thresh['dv_min']}kt / {thresh['dt_window']}hr")            
    else:
        thresh['dv_min']=-9999

    if not np.isnan(thresh['dp_max']):
        plot_subtitle.append(f"{lteq} {thresh['dp_max']}hPa / {thresh['dt_window']}hr")            
    else:
        thresh['dp_max']=9999
    
    if not np.isnan(thresh['dv_max']):
        plot_subtitle.append(f"{lteq} {thresh['dv_max']}kt / {thresh['dt_window']}hr")            
    else:
        thresh['dv_max']=9999

    if not np.isnan(thresh['dp_min']):
        plot_subtitle.append(f"{gteq} {thresh['dp_min']}hPa / {thresh['dt_window']}hr")            
    else:
        thresh['dp_min']=-9999
    
    if len(plot_subtitle)>0:
        plot_subtitle = '\n'+', '.join(plot_subtitle)
    else:
        plot_subtitle = ''
    return thresh,plot_subtitle


def interp_storm(storm_dict,timeres=1,dt_window=24,dt_align='middle'):
    new_storm = {}
    for name in ['date','vmax','mslp','lat','lon','type']:
        new_storm[name]=[]
    times = mdates.date2num(storm_dict['date'])
    storm_dict['type']=np.asarray(storm_dict['type'])
    storm_dict['lon'] = np.array(storm_dict['lon'])%360
    try:
        targettimes = np.arange(times[0],times[-1]+timeres/24,timeres/24)
        new_storm['date'] = [t.replace(tzinfo=None) for t in mdates.num2date(targettimes)]
        stormtype = np.ones(len(storm_dict['type']))*-99
        stormtype[np.where((storm_dict['type']=='TD') | (storm_dict['type']=='SD') | (storm_dict['type']=='TS') | \
                           (storm_dict['type']=='SS') | (storm_dict['type']=='HU'))] = 0
        new_storm['type'] = np.interp(targettimes,times,stormtype)
        new_storm['type'] = np.where(new_storm['type']<0,'NT','TD')
        for name in ['vmax','mslp','lat','lon']:
            new_storm[name] = np.interp(targettimes,times,storm_dict[name])
        
        new_storm['dvmax_dt'] = [np.nan]+list((new_storm['vmax'][1:]-new_storm['vmax'][:-1])/timeres)

        new_storm['dmslp_dt'] = [np.nan]+list((new_storm['mslp'][1:]-new_storm['mslp'][:-1])/timeres)

        rE = 6.371e3 #km
        d2r = np.pi/180.
        new_storm['dx_dt'] = [np.nan]+list(d2r*(new_storm['lon'][1:]-new_storm['lon'][:-1])* \
                 rE*np.cos(d2r*np.mean([new_storm['lat'][1:],new_storm['lat'][:-1]],axis=0))/timeres)
        new_storm['dy_dt'] = [np.nan]+list(d2r*(new_storm['lat'][1:]-new_storm['lat'][:-1])* \
                 rE/timeres)
        
        for name in ['dvmax_dt','dmslp_dt']:
            tmp = np.round(np.convolve(new_storm[name],[1]*int(dt_window/timeres),mode='valid'),1)         
            if dt_align=='end':
                new_storm[name] = [np.nan]*(len(new_storm[name])-len(tmp))+list(tmp)
            if dt_align=='middle':
                tmp2 = [np.nan]*int((len(new_storm[name])-len(tmp))//2)+list(tmp)
                new_storm[name] = tmp2+[np.nan]*(len(new_storm[name])-len(tmp2))
            if dt_align=='start':
                new_storm[name] = list(tmp)+[np.nan]*(len(new_storm[name])-len(tmp))
                
        for name in ['dx_dt','dy_dt']:
            tmp = np.convolve(new_storm[name],[timeres/dt_window]*int(dt_window/timeres),mode='valid')
            if dt_align=='end':
                new_storm[name] = [np.nan]*(len(new_storm[name])-len(tmp))+list(tmp)
            if dt_align=='middle':
                tmp2 = [np.nan]*int((len(new_storm[name])-len(tmp))//2)+list(tmp)
                new_storm[name] = tmp2+[np.nan]*(len(new_storm[name])-len(tmp2))
            if dt_align=='start':
                new_storm[name] = list(tmp)+[np.nan]*(len(new_storm[name])-len(tmp))
            
        return new_storm
    except:
        for name in new_storm.keys():
            try:
                storm_dict[name]
            except:
                storm_dict[name]=np.ones(len(new_storm[name]))*np.nan
        return storm_dict


def filter_storms_vp(trackdata,year_min=0,year_max=9999,subset_domain=None):
    r"""
    trackdata : tracks.Dataset object
    subset_domain : str
        String representing either a bounded region 'latW/latE/latS/latN', or a basin name.
    """
    if subset_domain == None:
        lon_min,lon_max,lat_min,lat_max = [0,360,-90,90]
    else:
        lon_min,lon_max,lat_min,lat_max = [float(i) for i in subset_domain.split("/")]
    vp=[]
    for key in trackdata.keys:
        istorm = trackdata.data[key]
        for i,(iwind,imslp,itype,ilat,ilon,itime) in \
        enumerate(zip(istorm['vmax'],istorm['mslp'],istorm['type'],istorm['lat'],istorm['lon'],istorm['date'])):
            if np.nan not in [iwind,imslp] and itype in ['TD','TS','SS','HU'] \
                   and lat_min<=ilat<=lat_max and lon_min<=ilon%360<=lon_max \
                   and year_min<=itime.year<=year_max:
                vp.append([imslp,iwind])
    return vp

def testfit(data,x,order):
    if len(data)>50:
        f=np.polyfit([i[1] for i in data],[i[0] for i in data],order)
        y=sum([f[i]*x**(order-i) for i in range(order+1)])
        return y
    else:
        return np.nan
    
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def convert_to_julian(date):
    year = date.year
    return ((date - dt(year,1,1,0)).days + (date - dt(year,1,1,0)).seconds/86400.0) + 1

def months_in_julian(year):
    length_of_year = convert_to_julian(dt(year,12,31,0))+1.0
    months = range(1,13,1)
    months_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    months_dates = [dt(year,i,1,0) for i in months]
    months_julian = [int(convert_to_julian(i)) for i in months_dates]
    midpoint_julian = (np.array(months_julian) + np.array(months_julian[1:]+[length_of_year]))/2.0
    return {'start':months_julian,'midpoint':midpoint_julian.tolist(),'name':months_names}

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


def convert_category(vmax):
    if vmax >= 137:
        return 5
    elif vmax >= 113:
        return 4
    elif vmax >= 96:
        return 3
    elif vmax >= 83:
        return 2
    elif vmax >= 64:
        return 1
    elif vmax >= 34:
        return 0
    return -1

def category_to_wind(cat):
    cat2 = cat.lower()
    if cat2 == 'td' or cat2 == 'sd':
        return 33
    elif cat2 == 'ts' or cat2 == 'ss':
        return 34
    elif cat2 == 'c1':
        return 64
    elif cat2 == 'c2':
        return 83
    elif cat2 == 'c3':
        return 96
    elif cat2 == 'c4':
        return 113
    else:
        return 137

def classify_subtrop(storm_type):
    """
    SD purely - yes
    SD then SS then TS - no
    SD then TS - no
    """
    if 'SD' in storm_type:
        if 'SD' in storm_type and True not in np.isin(storm_type,['TD','TS','HU']):
            return True
    if 'SS' in storm_type and True not in np.isin(storm_type,['TD','TS','HU']):
        return True
    else:
        return False

def get_storm_type(vmax,subtrop_flag,basin):
    
    if basin in ['north_atlantic','east_pacific']:
        if vmax == 0:
            return "Unknown"
        elif vmax < 34:
            if subtrop_flag == True:
                return "Subtropical Depression"
            else:
                return "Tropical Depression"
        elif vmax < 63:
            if subtrop_flag == True:
                return "Subtropical Storm"
            else:
                return "Tropical Storm"
        else:
            return "Hurricane"
     
    elif basin == 'west_pacific':
        if vmax == 0:
            return "Unknown"
        elif vmax < 34:
            if subtrop_flag == True:
                return "Subtropical Depression"
            else:
                return "Tropical Depression"
        elif vmax < 63:
            if subtrop_flag == True:
                return "Subtropical Storm"
            else:
                return "Tropical Storm"
        elif vmax < 120:
            return "Typhoon"
        else:
            return "Super Typhoon"
    elif basin == 'australia' or basin == 'south_pacific':
        if vmax == 0:
            return "Unknown"
        elif vmax < 63:
            return "Tropical Cyclone"
        else:
            return "Severe Tropical Cyclone"
    elif basin == 'north_indian':
        if vmax == 0:
            return "Unknown"
        elif vmax < 28:
            return "Depression"
        elif vmax < 34:
            return "Deep Depression"
        elif vmax < 48:
            return "Cyclonic Storm"
        elif vmax < 64:
            return "Severe Cyclonic Storm"
        elif vmax < 90:
            return "Very Severe Cyclonic Storm"
        elif vmax < 120:
            return "Extremely Severe Cyclonic Storm"
        else:
            return "Super Cyclonic Storm"
    elif basin == 'south_indian':
        if vmax == 0:
            return "Unknown"
        elif vmax < 28:
            return "Tropical Disturbance"
        elif vmax < 34:
            return "Tropical Depression"
        elif vmax < 48:
            return "Moderate Tropical Storm"
        elif vmax < 64:
            return "Severe Tropical Storm"
        elif vmax < 90:
            return "Tropical Cyclone"
        elif vmax < 115:
            return "Intense Tropical Cyclone"
        else:
            return "Very Intense Tropical Cyclone"
    else:
        return "Cyclone"
    
def get_type(vmax,subtrop_flag):
    
    if vmax < 34:
        if subtrop_flag == True:
            return "SD"
        else:
            return "TD"
    elif vmax < 63:
        if subtrop_flag == True:
            return "SS"
        else:
            return "TS"
    else:
        return "HU"
    
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
    
    if x=='category':
        if varname == 'vmax':
#            clevs = [34,64,83,96,113,137,200]
#            colors = ['#8FC2F2','#3185D3','#FFFF00','#FF9E00','#DD0000','#FF00FC','#8B0088']
            clevs = [cat2wind(c) for c in range(-1,6)]+[200]
            if linear:
                colors = [mcolors.to_rgba(category_color(lev)) \
                               for c,lev in enumerate(clevs[:-1]) for _ in range(clevs[c+1]-clevs[c])]
            else:
                clevs = [cat2wind(c) for c in range(-1,6)]+[200]
                colors = [category_color(lev) for lev in clevs[:-1]]
                cmap = mcolors.ListedColormap(colors)
        else:
            warnings.warn('Saffir Simpson category colors allowed only for surface winds')
            x = 'plasma'
    if x!='category':
        if isinstance(x,str):
            cmap = mlib.cm.get_cmap(x)
        elif isinstance(x,list):
            cmap = mcolors.ListedColormap(x)
        elif isinstance(x,dict):
            cmap = make_colormap(x)
        else:
            cmap = x
        norm = mlib.colors.Normalize(vmin=0, vmax=len(clevs)-1)
        if len(clevs)>2:
            colors = cmap(norm(np.arange(len(clevs)-1)))
            cmap = mcolors.ListedColormap(colors)
        else:
            colors = cmap(norm(np.linspace(0,1,256)))
            cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap',colors)
            
            y0 = min(clevs)
            y1 = max(clevs)
            dy = (y1-y0)/8
            scalemag = int(np.log(dy)/np.log(10))
            dy_scaled = dy*10**-scalemag
            dc = min([1,2,5,10], key=lambda x:abs(x-dy_scaled))
            dc = dc*10**scalemag
            c0 = np.ceil(y0/dc)*dc
            c1 = np.floor(y1/dc)*dc
            clevs = np.arange(c0,c1+dc,dc)
            
    return cmap,clevs
    


def str2(a):
    if a < 10:
        return f'0{a}'
    return str(a)

def plot_credit():
    return "Plot generated using troPYcal"

def add_credit(ax,text):
    import matplotlib.patheffects as path_effects    
    a = ax.text(0.99,0.01,text,fontsize=9,color='k',alpha=0.7,fontweight='bold',
            transform=ax.transAxes,ha='right',va='bottom',zorder=10)
    a.set_path_effects([path_effects.Stroke(linewidth=5, foreground='white'),
                   path_effects.Normal()])
        

def all_nan(arr):
    arr_copy = np.array(arr)
    if len(arr_copy[~np.isnan(arr_copy)]) == 0:
        return True
    else:
        return False

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

def knots_to_mph(wind):

    kts = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185]
    mphs = [15,20,25,30,35,40,45,50,60,65,70,75,80,85,90,100,105,110,115,120,125,130,140,145,150,155,160,165,175,180,185,190,195,200,205,210]
    
    if wind in kts:
        return mphs[kts.index(wind)]
    return wind

def ef_colors(x):
    import matplotlib as mlib
    if x == 'default':
        colors = ['lightsalmon','tomato','red','firebrick','darkred','purple']
    elif isinstance(x,str):
        try:
            cmap = mlib.cm.get_cmap(x)
            norm = mlib.colors.Normalize(vmin=0, vmax=5)
            colors = cmap(norm([0,1,2,3,4,5]))
        except:
            colors = [x]*6
    elif isinstance(x,list):
        if len(x) == 6:
            colors = x
    else:
        colors = ['lightsalmon','tomato','red','firebrick','darkred','purple']
    return colors

def num_to_str(num):
    d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
          6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
          11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
          15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
          19 : 'nineteen', 20 : 'twenty',
          30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
          70 : 'seventy', 80 : 'eighty', 90 : 'ninety' }

    assert(0 <= num)

    if (num < 20):
        return d[num]

    if (num < 100):
        if num % 10 == 0:
            return d[num]
        else:
            return d[num // 10 * 10] + '-' + d[num % 10]