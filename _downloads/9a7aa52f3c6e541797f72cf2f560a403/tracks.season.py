"""
==========================
Individual Season Analysis
==========================
This sample script illustrates how to retrieve, visualize and analyze seasons from the HURDAT2 dataset.
"""

import tropycal.tracks as tracks

###########################################
# HURTDAT2 Dataset
# ----------------
# Let's start by creating an instance of a TrackDataset object. By default, this reads in the HURDAT2 dataset from the National Hurricane  Center (NHC) website. For this example we'll be using the HURDAT2 dataset over the North Atlantic basin.
# 
# HURDAT data is not available for the most recent hurricane seasons. To include the latest data up through today, the "include_btk" flag  would need to be set to True, which reads in preliminary best track data from the NHC website.

basin = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=False)

###########################################
# Individual season analysis
# --------------------------
# Individual seasons can be retrieved from the dataset by calling the ``get_season()`` function, which returns an instance of a Season object.
# 
# Let's retrieve an instance of the 2017 Atlantic Hurricane Season:

season = basin.get_season(2017)

###########################################
# We can quickly visualize what the Season object contains by printing it:

print(season)

###########################################
# The Season object can be converted to a Pandas DataFrame, which lists a summary of storms during the season:

season.to_dataframe()

###########################################
# A more detailed summary of the season can be retrieved using the `summary()` method:

print(season.summary())

###########################################
# Plot Season
# -----------
# Plotting a Season object can be quickly done using the ``plot()`` method.
# 
# Note that you can pass various arguments to the plot function, such as customizing the map and track aspects. The "Customizing Storm Plots" example script has more examples on how to customize such plots. Read through the documentation for more customization options.

season.plot()

###########################################
# Multiple Season Analysis
# ------------------------
# Seasons can also be combined for multi-season analyses by simply adding multiple season objects together.

season1 = basin.get_season(2017)
season2 = basin.get_season(2018)
season3 = basin.get_season(2021)

combined = season1 + season2 + season3

print(combined)

###########################################
# The combined seasons can then be plotted on the same map:

combined.plot()

###########################################
# The summary method also generates summaries for all seasons in this object:

print(combined.summary())