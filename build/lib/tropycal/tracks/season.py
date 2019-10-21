r"""Functionality for storing and analyzing a year/season of cyclones."""

import calendar
import numpy as np
import pandas as pd
import re
import scipy.interpolate as interp
import urllib
import warnings
from datetime import datetime as dt,timedelta

from .plot import TrackPlot
from .tools import *
    
class Season:
    
    r"""
    Initializes an instance of Season, retrieved via ``TrackDataset.get_season()``.

    Parameters
    ----------
    season : dict
        Dict entry containing all storms within the requested season.
    info : dict
        Dict entry containing general information about the season.

    Returns
    -------
    Season
        Instance of a Season object.
    """
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):
         
        #Label object
        summary = ["<tropycal.tracks.Season>"]
        
        #Format keys for summary
        season_summary = self.annual_summary()
        summary_keys = {'Total Storms':season_summary['season_storms'],
                        'Named Storms':season_summary['season_named'],
                        'Hurricanes':season_summary['season_hurricane'],
                        'Major Hurricanes':season_summary['season_major'],
                        'Season ACE':season_summary['season_ace']}

        #Add season summary
        summary.append("Season Summary:")
        add_space = np.max([len(key) for key in summary_keys.keys()])+3
        for key in summary_keys.keys():
            key_name = key+":"
            val = '%0.1f'%(summary_keys[key]) if key == 'Season ACE' else summary_keys[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')
        
        #Add additional information
        summary.append("\nMore Information:")
        add_space = np.max([len(key) for key in self.coords.keys()])+3
        for key in self.coords.keys():
            key_name = key+":"
            val = '%0.1f'%(self.coords[key]) if key == 'ace' else self.coords[key]
            summary.append(f'{" "*4}{key_name:<{add_space}}{val}')

        return "\n".join(summary)
    
    def __init__(self,season,info):
        
        #Save the dict entry of the season
        self.dict = season
        
        #Add other attributes about the storm
        keys = info.keys()
        self.coords = {}
        for key in keys:
            if isinstance(info[key], list) == False and isinstance(info[key], dict) == False:
                self[key] = info[key]
                self.coords[key] = info[key]
    
    def to_dataframe(self):
        
        r"""
        Converts the season dict into a pandas DataFrame object.
        
        Returns
        -------
        `pandas.DataFrame`
            A pandas DataFrame object containing information about the season.
        """
        
        #Try importing pandas
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("Error: pandas is not available. Install pandas in order to use this function.") from e
        
        #Get season info
        season_info = self.annual_summary()
        season_info_keys = season_info['id']
        
        #Set up empty dict for dataframe
        ds = {'id':[],'name':[],'vmax':[],'mslp':[],'category':[],'ace':[],'start_time':[],'end_time':[]}
        
        #Add every key containing a list into the dict
        keys = [k for k in self.dict.keys()]
        for key in keys:
            if key in season_info_keys:
                sidx = season_info_keys.index(key)
                ds['id'].append(key)
                ds['name'].append(self.dict[key]['name'])
                ds['vmax'].append(season_info['max_wspd'][sidx])
                ds['mslp'].append(season_info['min_mslp'][sidx])
                ds['category'].append(season_info['category'][sidx])
                ds['start_time'].append(self.dict[key]['date'][0])
                ds['end_time'].append(self.dict[key]['date'][-1])
                ds['ace'].append(np.round(season_info['ace'][sidx],1))
                    
        #Convert entire dict to a DataFrame
        ds = pd.DataFrame(ds)

        #Return dataset
        return ds
        
    def plot(self,ax=None,cartopy_proj=None,prop={},map_prop={}):
        
        r"""
        Creates a plot of this season.
        
        Parameters
        ----------
        ax : axes
            Instance of axes to plot on. If none, one will be generated. Default is none.
        cartopy_proj : ccrs
            Instance of a cartopy projection to use. If none, one will be generated. Default is none.
        prop : dict
            Property of storm track lines.
        map_prop : dict
            Property of cartopy map.
        """
        
        #Create instance of plot object
        self.plot_obj = TrackPlot()
        
        if self.basin in ['east_pacific','west_pacific','south_pacific','australia','all']:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=180.0)
        else:
            self.plot_obj.create_cartopy(proj='PlateCarree',central_longitude=0.0)
            
            
        #Plot storm
        return_ax = self.plot_obj.plot_season(self,ax,prop=prop,map_prop=map_prop)
        
        #Return axis
        if ax != None: return return_ax
        
    def annual_summary(self):
        
        r"""
        Generates a summary for this season with various cumulative statistics.
        
        Returns
        -------
        dict
            Dictionary containing various statistics about this season.
        """
        
        #Initialize dict with info about all of year's storms
        hurdat_year = {'id':[],'operational_id':[],'name':[],'max_wspd':[],'min_mslp':[],'category':[],'ace':[]}
        
        #Search for corresponding entry in keys
        count_ss_pure = 0
        count_ss_partial = 0
        iterate_id = 1
        for key in self.dict.keys():

            #Retrieve info about storm
            temp_name = self.dict[key]['name']
            temp_vmax = np.array(self.dict[key]['vmax'])
            temp_mslp = np.array(self.dict[key]['mslp'])
            temp_type = np.array(self.dict[key]['type'])
            temp_time = np.array(self.dict[key]['date'])
            temp_ace = self.dict[key]['ace']

            #Get indices of all tropical/subtropical time steps
            idx = np.where((temp_type == 'SS') | (temp_type == 'SD') | (temp_type == 'TD') | (temp_type == 'TS') | (temp_type == 'HU'))

            #Get times during existence of trop/subtrop storms
            if len(idx[0]) == 0: continue
            trop_time = temp_time[idx]
            if 'season_start' not in hurdat_year.keys():
                hurdat_year['season_start'] = trop_time[0]
            hurdat_year['season_end'] = trop_time[-1]

            #Get max/min values and check for nan's
            np_wnd = np.array(temp_vmax[idx])
            np_slp = np.array(temp_mslp[idx])
            if len(np_wnd[~np.isnan(np_wnd)]) == 0:
                max_wnd = np.nan
                max_cat = -1
            else:
                max_wnd = int(np.nanmax(temp_vmax[idx]))
                max_cat = convert_category(np.nanmax(temp_vmax[idx]))
            if len(np_slp[~np.isnan(np_slp)]) == 0:
                min_slp = np.nan
            else:
                min_slp = int(np.nanmin(temp_mslp[idx]))

            #Append to dict
            hurdat_year['id'].append(key)
            hurdat_year['name'].append(temp_name)
            hurdat_year['max_wspd'].append(max_wnd)
            hurdat_year['min_mslp'].append(min_slp)
            hurdat_year['category'].append(max_cat)
            hurdat_year['ace'].append(temp_ace)
            hurdat_year['operational_id'].append(self.dict[key]['operational_id'])
            
            #Handle operational vs. non-operational storms

            #Check for purely subtropical storms
            if 'SS' in temp_type and True not in np.isin(temp_type,['TD','TS','HU']):
                count_ss_pure += 1

            #Check for partially subtropical storms
            if 'SS' in temp_type:
                count_ss_partial += 1

        #Add generic season info
        hurdat_year['season_storms'] = len(hurdat_year['name'])
        narray = np.array(hurdat_year['max_wspd'])
        narray = narray[~np.isnan(narray)]
        hurdat_year['season_named'] = len(narray[narray>=34])
        hurdat_year['season_hurricane'] = len(narray[narray>=65])
        hurdat_year['season_major'] = len(narray[narray>=100])
        hurdat_year['season_ace'] = np.sum(hurdat_year['ace'])
        hurdat_year['season_subtrop_pure'] = count_ss_pure
        hurdat_year['season_subtrop_partial'] = count_ss_partial
                
        #Return object
        return hurdat_year
