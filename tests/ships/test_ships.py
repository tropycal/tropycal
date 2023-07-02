"""Tests for the `ships` module"""

import os
import pandas as pd
import xarray as xr
import datetime as dt
import tropycal.tracks as tracks

def read_ships():

    #Read sample subset of HURDATv2 dataset, and retrieve sample storm data
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_hurdat.txt')
    basin = tracks.TrackDataset('north_atlantic', atlantic_url=filepath)

    #Retrieve Hurricane Ida SHIPS data
    storm = basin.get_storm(('ida',2021))
    return storm.get_ships(dt.datetime(2021, 8, 27, 0))

#==========================================================================================
# Test functions not involving internet connection
#==========================================================================================

def test_get_ships():
    """Tests fetching SHIPS data"""
    
    #Retrieve data for Hurricane Ida (2021)
    ships = read_ships()
    
    #Check attributes
    expected_output = {
        'storm_bearing_deg': 320,
        'storm_motion_kt': 10,
        'max_wind_t-12_kt': 30,
        'steering_level_pres_hpa': 500,
        'steering_level_pres_mean_hpa': 620,
        'brightness_temp_stdev': 21.5,
        'brightness_temp_stdev_mean': 14.5,
        'pixels_below_-20c': 60.0,
        'pixels_below_-20c_mean': 65.0,
        'lat': 18.0,
        'lon': -80.1,
        'storm_name': 'IDA',
        'forecast_init': dt.datetime(2021, 8, 27, 0, 0),
    }
    assert ships.attrs == expected_output

def test_get_snapshot():
    """Tests retrieving Ships data valid at a single time"""
    
    ships = read_ships()
    
    output = ships.get_snapshot(hour=48)
    
    expected_output = {
        'fhr': 48,
        'vmax_noland_kt': 65,
        'vmax_land_kt': 64,
        'vmax_lgem_kt': 59,
        'storm_type': 'TROP',
        'shear_kt': 6,
        'shear_adj_kt': 2,
        'shear_dir': 221,
        'sst_c': 30.1,
        'vmax_pot_kt': 172,
        '200mb_temp_c': -52.0,
        'thetae_dev_c': 7,
        '700_500_rh': 69,
        'model_vortex_kt': 21,
        '850mb_env_vort': 2,
        '200mb_div': 1,
        '700_850_tadv': 10,
        'dist_land_km': 378,
        'lat': 25.8,
        'lon': -88.0,
        'storm_speed_kt': 12,
        'heat_content': 128
    }
    assert output == expected_output

def test_get_ri_prob():
    """Tests fetching RI probability"""
    
    ships = read_ships()
    
    output = ships.get_ri_prob()
    
    expected_output = {
        '20kt/12hr': {'probability': 5, 'climo_mean': 4.9, 'prob / climo': 1.0},
        '25kt/24hr': {'probability': 16, 'climo_mean': 10.9, 'prob / climo': 1.5},
        '30kt/24hr': {'probability': 10, 'climo_mean': 6.8, 'prob / climo': 1.4},
        '35kt/24hr': {'probability': 8, 'climo_mean': 3.9, 'prob / climo': 2.2},
        '40kt/24hr': {'probability': 6, 'climo_mean': 2.4, 'prob / climo': 2.6},
        '45kt/36hr': {'probability': 11, 'climo_mean': 4.6, 'prob / climo': 2.3},
        '55kt/48hr': {'probability': 20, 'climo_mean': 4.7, 'prob / climo': 4.3},
        '65kt/72hr': {'probability': 44, 'climo_mean': 5.3, 'prob / climo': 8.4}
    }
    assert output == expected_output

def test_to_dataframe():
    """Test object can be converted to a pandas DataFrame"""
    
    ships = read_ships()
    
    assert isinstance(ships.to_dataframe(),pd.DataFrame) is True

def test_to_xarray():
    """Test object can be converted to an xarray Dataset"""
    
    ships = read_ships()
    
    assert isinstance(ships.to_xarray(),xr.Dataset) is True
