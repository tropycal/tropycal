"""
===========================
Historical TC Track Analogs
===========================
This sample script shows how to use Tropycal to retrieve and plot historical tropical cyclone track analogs.
"""

from tropycal import tracks

###########################################
# Reading In HURTDAT2 Dataset
# ---------------------------
#
# Let's start with the HURDAT2 dataset by loading it into memory. By default, this reads in the HURDAT dataset from the National Hurricane Center (NHC) website, unless you specify a local file path using either ``atlantic_url`` for the North Atlantic basin on ``pacific_url`` for the East & Central Pacific basin.
# 
# HURDAT data is not available for the current year. To include the latest data up through today, the "include_btk" flag needs to be set to True, which reads in preliminary best track data from the NHC website. For this example, we'll set this to False.
# 
# Let's create an instance of a TrackDataset object, which will store the North Atlantic HURDAT2 dataset in memory. Once we have this we can use its methods for various types of analyses.

basin = tracks.TrackDataset(basin='north_atlantic',include_btk=False)

###########################################
# Historical Tropical Cyclone Analogs
# -----------------------------------
# 
# One popular tool for finding historical tropical cyclone track analogs is via NOAA: https://coast.noaa.gov/hurricanes/#map=4/32/-80
# 
# Tropycal now has similar functionality, with 4 new analog functions added to `TrackDataset`:
# 
# - ``basin.analogs_from_point()`` - Retrieve storms within a radius of a point and their closest approach to the point
# - ``basin.analogs_from_shape()`` - Retrieve storms within a bounded shape provided by a list of lat/lon coordinates
# - ``basin.plot_analogs_from_point()`` - Plot output from analog_from_point()
# - ``basin.plot_analogs_from_shape()`` - Plot output from analog_from_shape()
# 
# Let's start out with `analogs_from_point` by looking at all tropical cyclone tracks within 50 kilometers of NYC. For this sample script we'll use kilometers, but if you want to use miles, add a ``units='miles'`` argument.
#
# Note that the first time you run an analog function, if storms in `basin` haven't been interpolated to hourly yet, this will automatically perform that interpolation on the back end, and future calls within the same kernel won't need to re-interpolate making them much faster.

basin.analogs_from_point((40.7,-74.0),radius=50)

###########################################
# The output from this function is a dictionary, with the **key** the storm ID and **value** the distance from the point in kilometers.
# 
# The default ordering of the dict is by chronological order. We can resort it to be ordered by distance from the point as follows:

analogs = basin.analogs_from_point((40.7,-74.0),radius=50)
dict(sorted(analogs.items(), key=lambda item: item[1]))

###########################################
# We can then plot these storms relative to the point using `plot_analogs_from_point`:

basin.plot_analogs_from_point((40.7,-74.0),radius=50)

###########################################
# We can further customize the analogs by adding thresholds by year range, time of year, sustained wind and MSLP.
# 
# Let's test this out by expanding the radius to 100 km, adding a minimum sustained wind of 65 kt (i.e., Category 1 hurricane), and from May to October:

#Print storms
storms = basin.analogs_from_point((40.7,-74.0),radius=100,date_range=('5/1','10/1'),thresh={'v_min':65})
print(storms)

#Plot storms
basin.plot_analogs_from_point((40.7,-74.0),radius=100,date_range=('5/1','10/1'),thresh={'v_min':65})

###########################################
# Let's say we want to automatically plot the closest storm to a point that meets our threshold. The below code automates this for any lat/lon coordinate.
# 
# This example is for NYC - feel free to play around with any lat/lon coordinate of your choice!

point = (40.7,-74.0) #NYC lat/lon

#Retrieve dict of analogs
analogs = basin.analogs_from_point(point,radius=100,date_range=('5/1','10/1'),thresh={'v_min':65})

#Sort by ascending value, meaning the first entry is the smallest distance from the point
analogs_sorted = sorted(analogs.items(), key=lambda item: item[1])

#Get ID of closest storm, which will be the first item of the first entry of analogs_sorted
closest_storm = analogs_sorted[0][0]

#Plot storm
basin.plot_storm(closest_storm)

###########################################
# Lastly, we can also use a custom domain created by a list of lat/lon coordinate pairs.
# 
# The example below plots all* tropical cyclones that passed through the New Jersey to Long Island coastline between 1950 and 2022, with additional plotting properties of (1) not plotting dots and (2) coloring lines by SSHWS category.
# 
# *Note: Tropical cyclone tracks are interpolated to hourly; therefore, a point only counts if its hourly track passed through the specified domain. Note the 1938 "Long Island Express" Hurricane doesn't appear below, as it moved at an anomalously fast forward speed.*

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
storms = basin.analogs_from_shape(points,year_range=(1950,2022))
print(storms)

#Plot storms
basin.plot_analogs_from_shape(points,year_range=(1950,2022),prop={'dots':False,'linecolor':'category'})
