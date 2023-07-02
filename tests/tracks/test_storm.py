"""Tests for the `tracks` module"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import tropycal.tracks as tracks

def read_storm():
    """For functions involving non-internet related functionality, use this function"""
    
    #Read sample subset of HURDATv2 dataset, and retrieve sample storm data
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_hurdat.txt')
    basin = tracks.TrackDataset('north_atlantic', atlantic_url=filepath)
    
    #Retrieve Hurricane Sam
    return basin.get_storm(('sam',2021))

def read_short_storm():
    """For functions involving fetching data online, use this function to speed up testing"""
    
    #Read sample subset of HURDATv2 dataset, and retrieve sample storm data
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_hurdat.txt')
    basin = tracks.TrackDataset('north_atlantic', atlantic_url=filepath)
    
    #Retrieve a short-lived storm
    return basin.get_storm(('colin',2022))

#==========================================================================================
# Test functions not involving internet connection
#==========================================================================================

def test_storm_reading():
    """Tests storm reading"""
    
    #Retrieve data for Hurricane Sam (2021)
    storm = read_storm()
    
    #Check stats
    expected_output = {
        'id': 'AL182021',
        'operational_id': 'AL182021',
        'name': 'SAM',
        'year': 2021,
        'season': 2021,
        'basin': 'north_atlantic',
        'source_info': 'NHC Hurricane Database',
        'source': 'hurdat',
        'ace': 54.0075,
        'realtime': False,
        'invest': False,
        'subset': False,
    }
    assert storm.attrs == expected_output

def test_interp():
    """Tests interpolation functionality"""
    
    storm = read_storm()
    
    #Interpolate to hourly data linearly
    new_storm = storm.interp(hours=1, method='linear')
    
    #Ensure interpolation did not modify original storm object
    assert len(storm.lon) == 59
    assert len(new_storm.lon) == 349
    assert new_storm.lon[140] == 305.7
    assert new_storm.lat[140] == 17.43
    
    #Interpolate to hourly data with quadratic fit
    new_storm = storm.interp(hours=1, method='quadratic')
    assert new_storm.lon[280] == 311.89
    assert new_storm.lat[280] == 40.82

def test_sel():
    """Tests storm subsetting functionality"""
    
    storm = read_storm()
    
    #Subset to latitude band
    new_storm = storm.sel(lat=(20,30))
    
    #Ensure original storm object wasn't modified
    assert max(storm.lat) > 30
    
    #Check new storm is correct
    assert max(new_storm.lat) <= 30
    assert min(new_storm.lat) >= 20

def test_to_dataframe():
    """Test storm can be converted to a pandas DataFrame"""
    
    storm = read_storm()
    
    assert isinstance(storm.to_dataframe(),pd.DataFrame) is True

def test_to_xarray():
    """Test storm can be converted to an xarray Dataset"""
    
    storm = read_storm()
    
    assert isinstance(storm.to_xarray(),xr.Dataset) is True
    
def test_to_dict():
    """Test storm can be converted to a dictionary"""
    
    storm = read_storm()
    
    assert isinstance(storm.to_dict(),dict) is True

#==========================================================================================
# Test functions involving fetching data from online
#==========================================================================================

def test_get_nhc_discussion():
    """Test fetching NHC discussion"""
    
    storm = read_short_storm()
    
    #Get discussion #1
    discussion = storm.get_nhc_discussion(forecast=dt.datetime(2022,7,2,14))
    assert discussion['id'] == 1
    assert discussion['time_issued'] == dt.datetime(2022, 7, 2, 8, 54)
    
    #Get discussion #2
    discussion = storm.get_nhc_discussion(forecast=dt.datetime(2022,7,2,15))
    assert discussion['id'] == 2
    assert discussion['time_issued'] == dt.datetime(2022, 7, 2, 14, 48)
    
    #Get discussion #2 using alternative method
    discussion = storm.get_nhc_discussion(forecast=2)
    assert discussion['id'] == 2
    assert discussion['time_issued'] == dt.datetime(2022, 7, 2, 14, 48)

def test_get_nhc_forecast():
    """Test fetching NHC forecast dict"""
    
    storm = read_short_storm()
    
    #Fetch forecast dict
    forecast = storm.get_nhc_forecast_dict(time=dt.datetime(2022,7,2,18))
    expected_output = {
        'init': dt.datetime(2022, 7, 2, 18, 0),
        'fhr': [0, 3, 12, 24, 36],
        'lat': [33.8, 34.0, 34.5, 35.7, 37.1],
        'lon': [-78.9, -78.6, -77.5, -75.2, -71.4],
        'vmax': [35, 35, 35, 35, 35],
        'mslp': [np.nan, 1014, np.nan, np.nan, np.nan],
        'type': ['TS', 'TS', 'TS', 'TS', 'EX'],
    }
    for key in expected_output.keys():
        assert forecast[key] == expected_output[key]

def test_get_operational_forecasts():
    """Test fetching all operational model data"""
    
    storm = read_short_storm()
    
    #Fetch all forecasts
    forecasts = storm.get_operational_forecasts()
    
    #Check important expected keys are found
    expected_keys = ['CARQ', 'AEMN', 'AVNO', 'IVCN', 'CMC', 'UKM', 'UKX', 'HMON', 'HWRF', 'OFCL']
    for key in expected_keys:
        assert key in forecasts.keys()
    
    #Check for NHC forecasts
    expected_keys = ['2022070206', '2022070212', '2022070218', '2022070300']
    assert list(forecasts['OFCL'].keys()) == expected_keys

def test_list_nhc_discussions():
    """Test fetching NHC discussion list"""
    
    storm = read_short_storm()
    
    #Fetch all discussions
    output = storm.list_nhc_discussions()
    
    #Check output is valid
    expected_output = {
        'id': [1, 2, 3, 4, 5],
        'utc_time': [dt.datetime(2022, 7, 2, 8, 54),
                     dt.datetime(2022, 7, 2, 14, 48),
                     dt.datetime(2022, 7, 2, 20, 35),
                     dt.datetime(2022, 7, 3, 2, 34),
                     dt.datetime(2022, 7, 3, 8, 44)],
        'url': ['https://ftp.nhc.noaa.gov/atcf/archive/2022/messages/al032022.discus.001.07020854',
                'https://ftp.nhc.noaa.gov/atcf/archive/2022/messages/al032022.discus.002.07021448',
                'https://ftp.nhc.noaa.gov/atcf/archive/2022/messages/al032022.discus.003.07022035',
                'https://ftp.nhc.noaa.gov/atcf/archive/2022/messages/al032022.discus.004.07030234',
                'https://ftp.nhc.noaa.gov/atcf/archive/2022/messages/al032022.discus.005.07030844'],
        'mode': 0,
    }
    assert output == expected_output
