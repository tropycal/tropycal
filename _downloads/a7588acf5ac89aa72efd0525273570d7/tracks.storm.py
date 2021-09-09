"""
=========================
Individual Storm Analysis
=========================
This sample script illustrates how to retrieve a single storm from the HURDAT2 dataset, and make plots and analyses of this storm.

For documentation generation purposes, return_ax must be set True for plotting functions. You don't need to have this extra argument in every plotting function call (e.g., "storm.plot(return_ax=True)" will produce the same output as "storm.plot()").
"""

import tropycal.tracks as tracks
import datetime as dt

###########################################
# HURTDAT2 Dataset
# ----------------
# Let's start by creating an instance of a TrackDataset object. By default, this reads in the HURDAT2 dataset from the National Hurricane  Center (NHC) website. For this example we'll be using the HURDAT2 dataset over the North Atlantic basin.
# 
# HURDAT data is not available for the most recent hurricane seasons. To include the latest data up through today, the "include_btk" flag  would need to be set to True, which reads in preliminary best track data from the NHC website.

hurdat_atl = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=False)

###########################################
# Individual storm analysis
# -------------------------
# Individual storms can be retrieved from the dataset by calling the ``get_storm()`` function, which returns an instance of a Storm object. This can be done by either entering a tuple containing the storm name and year, or by the standard tropical cyclone ID (e.g., "AL012019").
# 
# Let's retrieve an instance of Hurricane Michael from 2018:

storm = hurdat_atl.get_storm(('michael',2018))

###########################################
# This instance of Storm contains several methods that return the storm data back in different data types. The following examples will show # how to retrieve 3 different data types.
# 
# Retrieve a dictionary of Michael's data:

print(storm.to_dict())

###########################################
# Retrieve xarray Dataset object with Michael's data:

print(storm.to_xarray())

###########################################
# Retrieve pandas DataFrame object with Michael's data:

print(storm.to_dataframe())

###########################################
# Visualize Michael's observed track with the "plot" function:
# 
# Note that you can pass various arguments to the plot function, such as customizing the map and track aspects. The only cartopy projection # currently offered is PlateCarree. Read through the documentation for more customization options.

storm.plot(return_ax=True)

###########################################
# Plot the tornado tracks associated with Michael, along with the accompanying daily practically perfect forecast (PPH):
# 
# Note: There is currently a bug with this function that outputs 2 axes, a filled one and a blank one. This will be fixed in future updates.

storm.plot_tors(plotPPH=True)

###########################################
# If this storm was ever in NHC's area of responsibility, you can retrieve operational NHC forecast data for this event provided it is available. Forecast discussions date back to 1992, and forecast tracks date back to 1954.
# 
# Retrieve a single forecast discussion for Michael:

#Method 1: Specify date closest to desired discussion
disco = storm.get_nhc_discussion(forecast=dt.datetime(2018,10,7,0))
print(disco['text'])

#Method 2: Specify forecast discussion ID
disco = storm.get_nhc_discussion(forecast=2)
#print(disco['text']) printing this would show the same output

###########################################
# NHC also archives forecast tracks, albeit in a different format than the official advisory data, so the operational forecast IDs here differ from the discussion IDs. As such, the forecast cone is not directly retrieved from NHC, but is generated using an algorithm that yields a cone closely resembling the official NHC cone.
# 
# Let's plot Michael's second forecast cone:

storm.plot_nhc_forecast(forecast=2,return_ax=True)

###########################################
# Now let's look at the 12th forecast for Michael.
# 
# Note that the observed track here differs from the HURDAT2 track plotted previously! This is because this plot displays the operationally analyzed location and intensity, rather than the post-storm analysis data. This is done to account for differences between HURDAT2 and operational data.

storm.plot_nhc_forecast(forecast=12,return_ax=True)

###########################################
# IBTrACS Dataset
# ---------------
# 
# We can also read in IBTrACS data and use it the same way as we would use HURDAT2 data. There are caveats to using IBTrACS data, however, which are described more in depth in the :doc:`../data` page. We'll retrieve the global IBTrACS dataset, using the Joint Typhoon Warning Center (JTWC) data, modified with the Neumann reanalysis for southern hemisphere storms, and including a special reanalysis for Cyclone Catarina (2004) in Brazil.
# 
# .. warning::
# 
#     By default, IBTrACS data is read in from an online source. If you're reading in the global IBTrACS dataset, this could be quite slow.  For global IBTrACS, it is recommended to have the CSV file saved locally (`link to data`_), then set the flag ``ibtracs_url="local_path"``.
# 
# .. _link to data: https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r00/access/csv/ibtracs.ALL.list.v04r00.csv

ibtracs = tracks.TrackDataset(basin='all',source='ibtracs',ibtracs_mode='jtwc_neumann',catarina=True)

###########################################
# The functionality for handling storms in IBTrACS is the same as with using HURDAT2, the only limitation being no NHC and operational model data can be accessed when using IBTrACS as the data source.
# 
# `Super Typhoon Haiyan`_ (2013) was a catastrophic storm in the West Pacific basin, having made landfall in the Philippines. With estimated sustained winds of 195 mph (170 kt), it is among one of the most powerful tropical cyclones in recorded history. We can illustrate this by making a plot of Haiyan's observed track and intensity, from JTWC data:
# 
# .. _Super Typhoon Haiyan: https://en.wikipedia.org/wiki/Typhoon_Haiyan

storm = ibtracs.get_storm(('haiyan',2013))
storm.plot(return_ax=True)

###########################################
# `Cyclone Catarina`_ (2004) was an extremely rare hurricane-force tropical cyclone that developed in the South Atlantic basin, which normally doesn't see tropical cyclone activity, and subsequently made landfall in Brazil. The "Catarina" name is unofficial; it was not assigned a name in real time, and JTWC assigned it the ID "AL502004". Recall that when reading in the IBTrACS dataset previously, we set ``Catarina=True``. This read in data for Cyclone Catarina from a special post-storm reanalysis from McTaggart-Cowan et al. (2006). Let's make a plot of Catarina's observed track and intensity per this reanalysis:
# 
# .. _Cyclone Catarina: https://en.wikipedia.org/wiki/Hurricane_Catarina

storm = ibtracs.get_storm(('catarina',2004))
storm.plot(return_ax=True)

###########################################
# If we were to read in IBTrACS without setting ``Catarina=True`` (which sets it to False by default) and plot the track for "AL502004", we would get a noticeably different (shorter) and weaker track.
