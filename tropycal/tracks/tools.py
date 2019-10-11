import os, sys
import numpy as np
import pandas as pd
from datetime import datetime as dt,timedelta
import requests
import urllib

def filter_storms(hurdat,keys,year_min=0,year_max=9999,lat_min=-90,lat_max=90,lon_min=0,lon_max=360):
    vp=[]
    for storm in keys:
        istorm = hurdat[storm]
        for i,(iwind,imslp,itype,ilat,ilon,itime) in enumerate(zip(istorm['vmax'],istorm['mslp'],istorm['type'],istorm['lat'],istorm['lon'],istorm['date'])):
            if np.nan not in [iwind,imslp] and itype in ['TD','TS','SS','HU'] \
                   and lat_min<=ilat<=lat_max and lon_min<=360+ilon<=lon_max \
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

#Convert vmax to category
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

def str2(a):
    if a < 10:
        return f'0{a}'
    return str(a)

def plot_credit():
    return "Plot generated using troPYcal"

def all_nan(arr):
    arr_copy = np.array(arr)
    if len(arr_copy[~np.isnan(arr_copy)]) == 0:
        return True
    else:
        return False

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
    mphs = [15,20,25,30,35,40,45,50,60,65,70,75,80,85,90,100,105,110,115,120,125,130,140,145,150,155,160,165,170,180,185,190,195,200,205,210]
    
    if wind in kts:
        return mphs[kts.index(wind)]
    return wind

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