"""Tests for the `tracks` module"""

import os
import tropycal.tracks as tracks

def read_dataset():
    
    #Read sample subset of HURDATv2 dataset
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_hurdat.txt')
    return tracks.TrackDataset('north_atlantic', atlantic_url=filepath)

def test_season_reading():
    """Tests season reading"""
    
    #Read sample of HURDATv2 dataset
    basin = read_dataset()
    
    #Retrieve 2021 season
    season = basin.get_season(2021)
    
    #Check stats
    expected_output = {
        'year': 2021,
        'basin': 'north_atlantic',
        'source_basin': 'north_atlantic',
        'source': 'hurdat',
        'source_info': 'NHC Hurricane Database',
    }
    assert season.attrs == expected_output

def test_get_storm_id():
    """Test retrieving storm ID from storm tuple"""
    
    basin = read_dataset()
    season = basin.get_season(2021)
    
    #Test output
    assert season.get_storm_id(('elsa',2021)) == 'AL052021'

def test_summary():
    """Test retrieving season summary"""
    
    basin = read_dataset()
    season = basin.get_season(2021)
    
    #Test output
    summary = season.summary()
    
    #Check validity
    expected_output = {
        'season_storms': 21,
        'season_named': 21,
        'season_hurricane': 7,
        'season_major': 4,
        'season_ace': 145.3,
        'season_subtrop_pure': 1,
        'season_subtrop_partial': 3,
    }
    for key in expected_output.keys():
        assert summary[key] == expected_output[key]
    