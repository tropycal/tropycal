# Tropycal
Tropycal intends to simplify the process of retrieving and analyzing cyclone data, specifically for tropical cyclones.

## Installation
The currently recommended method of installation is via cloning the repository:

```
git clone https://github.com/tropycal/tropycal
cd tropycal
python setup.py install
```

## Dependencies
- matplotlib >= 3.0.0
- numpy >= 1.15.4
- scipy >= 1.1.0
- pandas >= 0.23.4
- xarray >= 0.11.2
- geopy >= 1.18.1

To fully leverage tropycal's plotting capabilities, it is strongly recommended to have cartopy >= 0.17.0 installed.

## Documentation
For full documentation, please refer to [Tropycal Documentation](https://tropycal.readthedocs.io/en/latest/).

## Sample Usage
As an example, read in the North Atlantic HURDAT2 reanalysis dataset, excluding Best Track (current year's storms):

```python
import tropycal.tracks as tracks

hurdat_atl = tracks.Dataset(basin='north_atlantic')
```

### Individual Storm Analysis

Individual storms can be retrieved from the dataset by calling the `get_storm` function, which returns an instance of a Storm object. This can be done by either entering a tuple containing the storm name and year, or by the standard tropical cyclone ID (e.g., AL012019).

Let's retrieve an instance of Hurricane Michael from 2018:

```python
storm = hurdat_atl.get_storm(('michael',2018))
```

This instance of Storm contains several methods that return the storm data back in different data types. The following examples will show how to retrieve 3 different data types.

Retrieve Michael's data in different data formats:

```python
storm.to_dict()
storm.to_xarray()
storm.to_dataframe()
```

Visualize Michael's observed track with the `plot` function:

Note that you can pass various arguments to the plot function, such as customizing the map and track aspects. The only cartopy projection currently offered is PlateCarree. Read through the documentation for more customization options.

```python
storm.plot()
```

If this storm was ever in NHC's area of responsibility, you can retrieve operational forecast data for this event provided it is available. Forecast discussions date back to 1992, and forecast tracks date back to 1950.

Retrieve a single forecast discussion for Michael - both of these methods will yield an identical result:

```python
#Method 1: Specify date closest to desired discussion
disco = storm.get_nhc_discussion(forecast=dt.datetime(2018,10,7,0))
print(disco['text'])

#Method 2: Specify forecast discussion ID
disco = storm.get_nhc_discussion(forecast=2)
print(disco['text'])
```

NHC also archives forecast tracks, albeit in a different format than the official advisory data, so the operational forecast IDs here differ from the discussion IDs. As such, the forecast cone is not directly retrieved from NHC, but is generated using an algorithm that yields a cone closely resembling the official NHC cone.

Let's plot Michael's second forecast cone:

```python
storm.plot_nhc_forecast(forecast=2)
```

Now let's look at the 12th forecast for Michael.

Note that the observed track here differs from the HURDAT2 track plotted previously! This is because this plot displays the operationally analyzed location and intensity, rather than the post-storm analysis data. This is done to account for differences between HURDAT2 and operational data.

```python
storm.plot_nhc_forecast(forecast=12)
```
