"""Tests for the `utils` module"""

import os
import numpy as np
import datetime as dt
import tropycal.utils as utils
import tropycal.tracks as tracks

def test_accumulated_cyclone_energy():
    """Test accumulated cyclone energy calculations"""
    
    #Test single values (0 ACE)
    wind_speed = 30
    np.testing.assert_almost_equal(utils.accumulated_cyclone_energy(wind_speed=wind_speed), 0, decimal=4)
    
    #Test single values (0.125 ACE)
    wind_speed = 35
    np.testing.assert_almost_equal(utils.accumulated_cyclone_energy(wind_speed=wind_speed), 0.1225, decimal=4)
    
    #Test single values with different hour interval
    wind_speed = 100
    hours = 3
    np.testing.assert_almost_equal(utils.accumulated_cyclone_energy(wind_speed=wind_speed,hours=hours), 0.5, decimal=4)
    
    #Test array
    wind_speed = np.array([30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160])
    validate_array = np.array([0.0, 0.1225, 0.16, 0.2025, 0.25, 0.3025, 0.36, 0.4225, 0.49, 0.5625, 0.64, 0.7225, 0.81, 0.9025, 1.0, 1.1025, 1.21, 1.3225, 1.44, 1.5625, 1.69, 1.8225, 1.96, 2.1025, 2.25, 2.4025, 2.56])
    np.testing.assert_almost_equal(utils.accumulated_cyclone_energy(wind_speed=wind_speed), validate_array, decimal=4)

def test_wind_to_category():
    """Test conversion of wind speed to category"""
    
    #Test various wind speed thresholds
    wind_speeds = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160]
    categories = [-1,-1,-1,0,0,0,0,0,0,1,1,1,1,2,2,2,3,3,3,4,4,4,4,4,5,5,5,5,5]
    for i,(wind_speed,category) in enumerate(zip(wind_speeds,categories)):
        np.testing.assert_almost_equal(utils.wind_to_category(wind_speed), category, decimal=1)

def test_category_to_wind():
    """Test conversion of category to wind speed"""
    
    #Test various category thresholds
    categories = [-1,0,1,2,3,4,5]
    wind_speeds = [5,34,64,83,96,113,137]
    for i,(wind_speed,category) in enumerate(zip(wind_speeds,categories)):
        np.testing.assert_almost_equal(utils.category_to_wind(category), wind_speed, decimal=1)

def test_calc_ensemble_ellipse():
    """Test calculation of ensemble ellipse"""
    
    #Test what should be a circle
    member_lons = [-60,-50,-50,-60]
    member_lats = [30,30,40,40]
    
    #Retrieve output
    ellipse = utils.calc_ensemble_ellipse(member_lons, member_lats)
    
    #Check for type and dimensions
    assert isinstance(ellipse,dict) is True
    assert list(ellipse.keys()) == ['ellipse_lon', 'ellipse_lat']
    assert len(ellipse['ellipse_lon']) == 361
    assert len(ellipse['ellipse_lat']) == 361
    
    #Check for longitude values
    np.testing.assert_almost_equal(ellipse['ellipse_lon'][0], -45.4625, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lon'][90], -55.0, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lon'][180], -64.5375, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lon'][270], -55.0, decimal=7)
    
    #Check for latitude values
    np.testing.assert_almost_equal(ellipse['ellipse_lat'][0], 35.0, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lat'][90], 44.5375, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lat'][180], 35.0, decimal=7)
    np.testing.assert_almost_equal(ellipse['ellipse_lat'][270], 25.4625, decimal=7)

def test_classify_subtropical():
    """Test classification of pure subtropical storms"""
    
    #Test pure tropical cyclone
    test_data = ['TD','TS','HU','EX']
    assert utils.classify_subtropical(test_data) is False
    
    #Test partial subtropical cyclone (only SD)
    test_data = ['SD','TD','TS','HU','EX']
    assert utils.classify_subtropical(test_data) is False
    
    #Test partial subtropical cyclone (only SS)
    test_data = ['SS','TS','HU','EX']
    assert utils.classify_subtropical(test_data) is False
    
    #Test pure subtropical cyclone (only SS)
    test_data = ['SS','SS','EX']
    assert utils.classify_subtropical(test_data) is True
    
    #Test pure subtropical cyclone (only SD)
    test_data = ['SD','SD','EX']
    assert utils.classify_subtropical(test_data) is True
    
    #Test pure subtropical cyclone (both SD and DD)
    test_data = ['SD','SS','EX']
    assert utils.classify_subtropical(test_data) is True

def test_create_storm_dict():
    """Test creating a storm dictionary"""
    
    #Read sample storm file
    test_data_dir = os.path.join(os.path.dirname(__file__), '../data')
    filepath = os.path.join(test_data_dir, 'sample_storm.txt')
    
    #Create storm dict
    storm_dict = utils.create_storm_dict(
        filepath,
        storm_name = 'Alex',
        storm_id = 'AL012021',
    )
    
    #Check for types
    assert isinstance(storm_dict,dict) is True
    assert isinstance(storm_dict['lat'],list) is True
    
    #Convert to Storm object
    storm = tracks.Storm(storm_dict)
    
    #Check ACE calculation
    np.testing.assert_almost_equal(storm.ace, 3.7825, decimal=7)
    
    #Check basin
    assert storm.basin == 'north_atlantic'
    
    #Check year
    assert storm.year == 2021
    
    #Check name
    assert storm.name == 'Alex'
    
    #Check mslp extrem
    np.testing.assert_almost_equal(min(storm.mslp), 988, decimal=2)
    assert min(storm.mslp) 

def test_dropsonde_mslp_estimate():
    """Test estimate of MSLP from dropsonde"""
    
    #Test various cases
    np.testing.assert_almost_equal(utils.dropsonde_mslp_estimate(mslp=942, surface_wind=0), 942, decimal=1)
    np.testing.assert_almost_equal(utils.dropsonde_mslp_estimate(mslp=942, surface_wind=15), 940.5, decimal=1)
    np.testing.assert_almost_equal(utils.dropsonde_mslp_estimate(mslp=942, surface_wind=30), 939, decimal=1)

def test_generate_nhc_cone():
    """Test NHC-style cone generation"""
    
    #Create sample forecast dict
    forecast = {
        'init': dt.datetime(2010, 6, 26, 0, 0),
        'fhr': [0, 3, 12, 24, 36, 48, 72, 96, 120],
        'lat': [16.6, 16.7, 17.4, 18.4, 19.5, 20.8, 22.5, 24.0, 24.5],
        'lon': [-83.9, -84.4, -85.8, -87.7, -89.1, -90.5, -92.0, -93.5, -94.5],
        'vmax': [30, 30, 40, 45, 35, 35, 45, 50, 50],
        'mslp': [np.nan, 1004, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        'type': ['TD', 'TD', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS', 'TS'],
    }
    
    #Create cone dict using default settings
    cone = utils.generate_nhc_cone(
        forecast,
        basin = 'north_atlantic',
        shift_lons = False,
        cone_days = 5,
        cone_year = None,
        return_xarray = False,
    )
    
    #Check dict was properly constructed
    assert isinstance(cone,dict) is True
    expected_keys = ['lat', 'lon', 'lat2d', 'lon2d', 'cone', 'center_lon', 'center_lat', 'year']
    for key in expected_keys:
        assert key in cone.keys()
    
    #Check cone generated properly
    np.testing.assert_almost_equal(np.max(cone['cone']), 1.0, decimal=1)
    assert len(cone['cone'][cone['cone']==1]) == 38892
    
    #Check center line was properly generated
    np.testing.assert_almost_equal(np.max(cone['center_lat']), 24.5032, decimal=4)
    np.testing.assert_almost_equal(np.max(cone['center_lon']), -84.4, decimal=2)

def test_get_basin():
    """Test determining basin from coordinates"""
    
    #Test South Atlantic basin
    assert utils.get_basin(lat=-20, lon=0) == 'south_atlantic'
    assert utils.get_basin(lat=-20, lon=19) == 'south_atlantic'
    assert utils.get_basin(lat=-20, lon=280) == 'south_atlantic'
    
    #Test South Pacific basin
    assert utils.get_basin(lat=-20, lon=279) == 'south_pacific'
    assert utils.get_basin(lat=-20, lon=160) == 'south_pacific'
    
    #Test Australia basin
    assert utils.get_basin(lat=-20, lon=159) == 'australia'
    assert utils.get_basin(lat=-20, lon=90) == 'australia'
    
    #Test South Indian basin
    assert utils.get_basin(lat=-20, lon=89) == 'south_indian'
    assert utils.get_basin(lat=-20, lon=20) == 'south_indian'

    #Test North Indian basin
    assert utils.get_basin(lat=30, lon=1) == 'north_indian'
    assert utils.get_basin(lat=30, lon=99) == 'north_indian'
    
    #Test regions north of North Indian
    assert utils.get_basin(lat=50, lon=1) == 'north_atlantic'
    assert utils.get_basin(lat=50, lon=69) == 'north_atlantic'
    assert utils.get_basin(lat=50, lon=70) == 'west_pacific'
    assert utils.get_basin(lat=50, lon=99) == 'west_pacific'
    
    #Test West Pacific basin
    assert utils.get_basin(lat=30, lon=100) == 'west_pacific'
    assert utils.get_basin(lat=30, lon=180) == 'west_pacific'
    
    #Test classic East Pacific basin
    assert utils.get_basin(lat=30, lon=181, source_basin='east_pacific') == 'east_pacific'
    
    #Test classic North Atlantic basin
    assert utils.get_basin(lat=30, lon=-50, source_basin='north_atlantic') == 'north_atlantic'
    assert utils.get_basin(lat=30, lon=-1, source_basin='north_atlantic') == 'north_atlantic'
    
    #Test Atlantic to Pacific crossover
    assert utils.get_basin(lat=13.5, lon=-85.5, source_basin='north_atlantic') == 'north_atlantic'
    assert utils.get_basin(lat=13.5, lon=-85.5, source_basin='east_pacific') == 'east_pacific'

def test_get_storm_classification():
    """Test generation of basin-specific storm classification titles"""
    
    #Test for NHC area of responsibility
    output = utils.get_storm_classification(wind_speed=25, subtropical_flag=False, basin='north_atlantic')
    assert output == 'Tropical Depression'
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=False, basin='north_atlantic')
    assert output == 'Tropical Storm'
    output = utils.get_storm_classification(wind_speed=64, subtropical_flag=False, basin='north_atlantic')
    assert output == 'Hurricane'
    output = utils.get_storm_classification(wind_speed=30, subtropical_flag=True, basin='north_atlantic')
    assert output == 'Subtropical Depression'
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=True, basin='north_atlantic')
    assert output == 'Subtropical Storm'
    
    #Test for West Pacific
    output = utils.get_storm_classification(wind_speed=25, subtropical_flag=False, basin='west_pacific')
    assert output == 'Tropical Depression'
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=False, basin='west_pacific')
    assert output == 'Tropical Storm'
    output = utils.get_storm_classification(wind_speed=64, subtropical_flag=False, basin='west_pacific')
    assert output == 'Typhoon'
    output = utils.get_storm_classification(wind_speed=130, subtropical_flag=False, basin='west_pacific')
    assert output == 'Super Typhoon'
    
    #Test for North Indian basin
    output = utils.get_storm_classification(wind_speed=25, subtropical_flag=False, basin='north_indian')
    assert output == 'Depression'
    output = utils.get_storm_classification(wind_speed=30, subtropical_flag=False, basin='north_indian')
    assert output == 'Deep Depression'
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=False, basin='north_indian')
    assert output == 'Cyclonic Storm'
    output = utils.get_storm_classification(wind_speed=48, subtropical_flag=False, basin='north_indian')
    assert output == 'Severe Cyclonic Storm'
    output = utils.get_storm_classification(wind_speed=64, subtropical_flag=False, basin='north_indian')
    assert output == 'Very Severe Cyclonic Storm'
    output = utils.get_storm_classification(wind_speed=90, subtropical_flag=False, basin='north_indian')
    assert output == 'Extremely Severe Cyclonic Storm'
    output = utils.get_storm_classification(wind_speed=120, subtropical_flag=False, basin='north_indian')
    assert output == 'Super Cyclonic Storm'
    
    #Test for South Indian basin
    output = utils.get_storm_classification(wind_speed=25, subtropical_flag=False, basin='south_indian')
    assert output == 'Tropical Disturbance'
    output = utils.get_storm_classification(wind_speed=30, subtropical_flag=False, basin='south_indian')
    assert output == 'Tropical Depression'
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=False, basin='south_indian')
    assert output == 'Moderate Tropical Storm'
    output = utils.get_storm_classification(wind_speed=48, subtropical_flag=False, basin='south_indian')
    assert output == 'Severe Tropical Storm'
    output = utils.get_storm_classification(wind_speed=64, subtropical_flag=False, basin='south_indian')
    assert output == 'Tropical Cyclone'
    output = utils.get_storm_classification(wind_speed=90, subtropical_flag=False, basin='south_indian')
    assert output == 'Intense Tropical Cyclone'
    output = utils.get_storm_classification(wind_speed=115, subtropical_flag=False, basin='south_indian')
    assert output == 'Very Intense Tropical Cyclone'
    
    #Test for Australian basin
    output = utils.get_storm_classification(wind_speed=34, subtropical_flag=False, basin='australia')
    assert output == 'Tropical Cyclone'
    output = utils.get_storm_classification(wind_speed=64, subtropical_flag=False, basin='australia')
    assert output == 'Severe Tropical Cyclone'

def test_get_storm_type():
    """Test deriving 2-character storm type string"""
    
    #Test various cases
    output = utils.get_storm_type(wind_speed=25, subtropical_flag=False, typhoon=False)
    assert output == 'TD'
    output = utils.get_storm_type(wind_speed=34, subtropical_flag=False, typhoon=False)
    assert output == 'TS'
    output = utils.get_storm_type(wind_speed=25, subtropical_flag=True, typhoon=False)
    assert output == 'SD'
    output = utils.get_storm_type(wind_speed=34, subtropical_flag=True, typhoon=False)
    assert output == 'SS'
    output = utils.get_storm_type(wind_speed=64, subtropical_flag=False, typhoon=False)
    assert output == 'HU'
    output = utils.get_storm_type(wind_speed=64, subtropical_flag=False, typhoon=True)
    assert output == 'TY'
    output = utils.get_storm_type(wind_speed=140, subtropical_flag=False, typhoon=True)
    assert output == 'ST'

def test_get_two_current():
    """Test retrieving current TWO"""
    
    #Retrieve TWO dict
    two_dict = utils.get_two_current()
    
    #Assert type and content
    assert isinstance(two_dict,dict) is True
    expected_keys = ['areas', 'lines', 'points']
    for key in expected_keys:
        assert key in two_dict.keys()

def skip_test_get_two_archive():
    """Test retrieving archived TWO"""
    """This function is *very* time consuming due to time to download data from NHC website. Only use when necessary"""
    
    #Retrieve archive TWO - seven day version
    two_dict = utils.get_two_archive(time=dt.datetime(2023,6,1,0))
    
    #Assert type and content
    assert isinstance(two_dict,dict) is True
    expected_keys = ['areas', 'lines', 'points']
    for key in expected_keys:
        assert key in two_dict.keys()
    
    #Retrieve archive TWO - five day version
    two_dict = utils.get_two_archive(time=dt.datetime(2016,7,1,0))
    
    #Assert type and content
    assert isinstance(two_dict,dict) is True
    for key in expected_keys:
        assert key in two_dict.keys()
    
    #Retrieve archive TWO - two day version
    two_dict = utils.get_two_archive(time=dt.datetime(2012,7,1,0))
    
    #Assert type and content
    assert isinstance(two_dict,dict) is True
    for key in expected_keys:
        assert key in two_dict.keys()

def test_knots_to_mph():
    """Test conversion of knots to mph"""

    #Define kt to mph conversions
    kts = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185]
    mphs = [0,10,15,20,25,30,35,40,45,50,60,65,70,75,80,85,90,100,105,110,115,120,125,130,140,145,150,155,160,165,175,180,185,190,195,200,205,210]
    
    #Assert all values
    for i,(mph,knot) in enumerate(zip(mphs,kts)):
        assert utils.knots_to_mph(knot) == mph

def test_nhc_cone_radii():
    """Test obtaining NHC cone radii by year"""
    
    #Test North Atlantic year
    output = utils.nhc_cone_radii(year=2021, basin='north_atlantic', forecast_hour=None)
    expected_output = {
        3: 16,
        12: 27,
        24: 40,
        36: 55,
        48: 69,
        60: 86,
        72: 102,
        96: 148,
        120: 200,
    }
    assert output == expected_output
    
    #Test East Pacific year
    output = utils.nhc_cone_radii(year=2021, basin='east_pacific', forecast_hour=None)
    expected_output = {
        3: 16,
        12: 25,
        24: 37,
        36: 51,
        48: 64,
        60: 77,
        72: 89,
        96: 114,
        120: 138,
    }
    assert output == expected_output

def test_calc_distance():
    """Test calculating gridded distance from a coordinate"""
    
    #Create sample coordinate data
    lats = np.array([30, 35, 40, 45, 50])
    lons = np.array([-70, -65, -60, -55, -50])

    #Create 2D arrays
    lons2d, lats2d = np.meshgrid(lons, lats)

    #Calculate distance from sample point
    output = utils.calc_distance(lats2d=lats2d, lons2d=lons2d, lat=40, lon=-60)
    
    #Check for output type
    assert len(output) == 2
    assert isinstance(output[1],np.ndarray)
    assert np.max(output[0]) == 0
    
    #Check distance was calculated correctly
    expected_output = np.array(
        [[1426.31201748, 1195.7406459, 1107.73929169, 1195.7406459, 1426.31201748],
         [1038.27901088, 708.32228018, 555.44613899, 708.32228018, 1038.27901088],
         [ 849.46114113, 425.60779557, 0,  425.60779557, 849.46114113],
         [ 986.43616274, 689.42894491, 555.44613899, 689.42894491, 986.43616274],
         [1351.58597175, 1173.70145504, 1107.73929169, 1173.70145504, 1351.58597175]])
    np.testing.assert_almost_equal(output[1], expected_output, decimal=5)

def test_add_radius():
    """Test calculating whether a point is in a specified radius"""
    
    #Create sample coordinate data
    lats = np.array([38, 39, 40, 41, 42])
    lons = np.array([-62, -61, -60, -59, -58])

    #Create 2D arrays
    lons2d, lats2d = np.meshgrid(lons, lats)

    #Calculate distance from sample point
    output = utils.add_radius(lats2d=lats2d, lons2d=lons2d, lat=40, lon=-60, rad=125)
    
    #Compare against expected output
    expected_output = np.array(
        [[0., 0., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 1., 1., 1., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0.]]
    )
    np.testing.assert_almost_equal(output, expected_output, decimal=1)

def test_all_nan():
    """Test checking if an array is entirely np.nan"""
    
    #Expected false
    data = [5,7,10,12]
    assert utils.all_nan(data) is False
    
    #Expected false
    data = [np.nan,np.nan,5,np.nan]
    assert utils.all_nan(data) is False
    
    #Expected true
    data = [np.nan,np.nan,np.nan,np.nan]
    assert utils.all_nan(data) is True

def category_label_to_wind():
    """Test converting category to minimum wind threshold"""
    
    #Test thresholds
    assert utils.category_label_to_wind('TD') == 33
    assert utils.category_label_to_wind('TS') == 34
    assert utils.category_label_to_wind('C1') == 64
    assert utils.category_label_to_wind('C2') == 83
    assert utils.category_label_to_wind('C3') == 96
    assert utils.category_label_to_wind('C4') == 113
    assert utils.category_label_to_wind('C5') == 137
