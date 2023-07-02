"""Tests for the `tracks` module"""

import os
import datetime as dt
import tropycal.tracks as tracks

def read_dataset():
    
    #Read sample subset of HURDATv2 dataset
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_hurdat.txt')
    return tracks.TrackDataset('north_atlantic', atlantic_url=filepath)

def test_dataset_reading():
    """Tests dataset reading"""
    
    #Read sample of HURDATv2 dataset
    basin = read_dataset()
    
    #Check data validity
    assert len(basin.data) == 37
    
    #Check stats
    expected_output = {
        'basin': 'north_atlantic',
        'source': 'hurdat',
        'ibtracs_mode': '',
        'start_year': 2021,
        'end_year': 2022,
        'max_wind': ('IAN', 2022),
        'min_mslp': ('SAM', 2021)
    }
    assert basin.attrs == expected_output

def test_analogs_from_point():
    """Test deriving analogs from a single point"""
    
    basin = read_dataset()
    
    #Test analogs using default functionality
    output = basin.analogs_from_point(point=(40,-70), radius=300)
    expected_output = {
        'AL032021': 132.1,
        'AL052021': 210.8,
        'AL082021': 95.7,
        'AL152021': 245.1,
    }
    assert output == expected_output
    
    #Test analogs including non-tropical cyclones
    output = basin.analogs_from_point(point=(40,-70), radius=300, non_tropical=True)
    expected_output = {
        'AL032021': 123.7,
        'AL052021': 210.8,
        'AL082021': 95.7,
        'AL092021': 140.0,
        'AL152021': 234.7,
        'AL212021': 14.0,
    }
    assert output == expected_output
    
    #Test analogs including a date range
    output = basin.analogs_from_point(point=(40,-70), radius=300, date_range=('8/1','8/31'))
    expected_output = {
        'AL082021': 95.7,
    }
    assert output == expected_output
    
def test_analogs_from_shape():
    """Test deriving analogs from a polygon"""
    
    basin = read_dataset()
    
    #Create list of sample points
    points = [
        (38.9, -74.7),
        (39.3, -74.2),
        (40.4, -73.8),
        (41.0, -71.8),
        (41.2, -72.2),
        (40.8, -73.7),
        (40.6, -74.3),
        (39.7, -74.4),
        (39.0, -74.9)
    ]

    #Retrieve list of storms that meet this criteria
    output = basin.analogs_from_shape(points)
    assert output == ['AL052021']

def test_climatology():
    """Test creating a climatology"""
    
    basin = read_dataset()
    
    #Create a climatology
    output = basin.climatology()
    expected_output = {
        'all_storms': 21.0,
        'named_storms': 21.0,
        'hurricanes': 7.0,
        'major_hurricanes': 4.0,
        'ace': 145.3,
        'start_time': dt.datetime(2023, 5, 22, 4, 48),
        'end_time': dt.datetime(2023, 11, 7, 4, 48),
    }
    assert output == expected_output

def test_filter_storms():
    """Test filtering storms by various thresholds"""
    
    basin = read_dataset()
    
    #Filter by date range
    output = basin.filter_storms(date_range=('6/1','6/30'))
    expected_output = ['AL012022', 'AL022021', 'AL032021', 'AL042021', 'AL052021']
    assert output == expected_output
    
    #Filter by wind speed
    output = basin.filter_storms(thresh={'v_min':120})
    expected_output = ['AL072022', 'AL092021', 'AL092022', 'AL182021']
    assert output == expected_output
    
    #Filter by domain
    domain = {'w':-80,'e':-75,'s':20,'n':30}
    output = basin.filter_storms(domain=domain)
    expected_output = ['AL012022', 'AL052021', 'AL062021', 'AL092022', 'AL172022']
    assert output == expected_output

def test_get_storm_id():
    """Test retrieving storm ID from storm tuple"""
    
    basin = read_dataset()
    
    #Test output
    assert basin.get_storm_id(('elsa',2021)) == 'AL052021'

def test_get_storm_tuple():
    """Test retrieving storm tuple from storm ID"""
    
    basin = read_dataset()
    
    #Test output
    assert basin.get_storm_tuple('AL052021') == ('ELSA',2021)
    
def test_rank_storm():
    """Test storm ranking by various thresholds"""
    
    basin = read_dataset()
    
    #Rank storms by ACE
    top_rank = dict(basin.rank_storm(metric='ace').iloc[0])
    expected_output = {
        'ace': 54.0075,
        'id': 'AL182021',
        'name': 'SAM',
        'year': 2021,
    }
    assert top_rank == expected_output
    
    #Rank storms by ACE and domain
    top_rank = dict(basin.rank_storm(metric='ace', domain={'w':-70,'e':-60,'s':20,'n':30}).iloc[0])
    expected_output = {
        'ace': 8.195,
        'id': 'AL182021',
        'name': 'SAM',
        'year': 2021,
    }
    assert top_rank == expected_output
    
def test_search_name():
    """Test searching for storms by name"""
    
    basin = read_dataset()

    assert basin.search_name('elsa') == [2021]
