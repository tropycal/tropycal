"""
=========================
Tornado Analyses
=========================
This sample script illustrates how to retrieve and analyze the Storm Prediction Center (SPC) tornado database (1950-present), using both the tornado and tracks modules.

For documentation generation purposes, return_ax must be set True for plotting functions. You don't need to have this extra argument in every plotting function call (e.g., "storm.plot(return_ax=True)" will produce the same output as "storm.plot()").
"""

import tropycal.tracks as tracks
import tropycal.tornado as tornado
import datetime as dt

###########################################
# Using TornadoDataset
# --------------------
# Let's start by creating an instance of a TornadoDataset object. By default, this reads in the SPC tornado database from their website.

tor_data = tornado.TornadoDataset()
    
###########################################
# We can use a TornadoDataset object to analyze both tornadoes associated with tropical cyclones and non-TC tornadoes. As an example of the latter, we can make a plot of all tornadoes during the 27 April 2011 tornado outbreak, along with the Practically Perfect Forecast (PPH) in filled contours:

tor_ax,domain,leg_tor = tor_data.plot_tors(dt.datetime(2011,4,27),plotPPH=True,return_ax=True)
tor_ax

###########################################
# Using TrackDataset
# ------------------
# We can also use TornadoDataset to assess tornadoes associated with tropical cyclones. First off let's get an instance of TrackDataset for the North Atlantic HURDAT2 basin:

hurdat_atl = tracks.TrackDataset(basin='north_atlantic',source='hurdat',include_btk=False)

###########################################
# This instance of Storm contains several methods that return the storm data back in different data types. The following examples will show how to retrieve 3 different data types.
# 
# Now we want to attribute tornadoes from the SPC database to all tropical cyclones which produced tornadoes. We do so using the ``assign_storm_tornadoes()`` method of TrackDataset. The main input parameter is "dist_thresh", which controls the distance from the tropical cyclone center over which to attribute tornadoes to. For this example we'll use 750 kilometers as the threshold.
# 
# This code block will take a while to run, as it will iterate over every storm in HURDAT2 and match tornadoes to those that produced them.

hurdat_atl.assign_storm_tornadoes(dist_thresh=750)

###########################################
# Once the above block is done running, we can now look at a climatology of tornadoes associated with North Atlantic tropical cyclones. The current method of analysis is via the ``plot_TCtors_rotated()`` method, which rotates tropical cyclones to a storm motion relative framework.
# 
# Most tornadoes associated with tropical cyclones occur in the front right quadrant (i.e., forward and right of the storm track). We can visualize this by plotting all tornadoes associated with tropical cyclones in a motion relative framework:

hurdat_atl.plot_TCtors_rotated('all')

###########################################
# We can also make the same plot for a composite subset of tropical cyclones, given either their IDs (e.g., "AL052004"), or a storm tuple. For instance, let's composite the four hurricanes that made landfall in Florida in 2004:

hurdat_atl.plot_TCtors_rotated(storms=[('charley',2004),('frances',2004),('ivan',2004),('jeanne',2004)])

###########################################
# Using a Storm object
# --------------------
# 
# Tropical cyclone tornado analyses can also be done via a Storm object. Let's get the data for Hurricane Ivan from 2004, which produced a major tornado outbreak:
# 
# .. warning::
# 
#     If you retrieve an instance of a Storm object without first running ``TrackDataset.assign_storm_tornadoes()`` method, doing tornado analyses with a Storm object will require re-downloading the tornado database for each new instance of Storm. If you plan to analyze multiple storms with tornadoes, it is recommended to run ``assign_storm_tornadoes()`` first.

storm = hurdat_atl.get_storm(('ivan',2004))

###########################################
# Let's plot all the tornado tracks, and daily PPH, associated with Hurricane Ivan:

storm.plot_tors(plotPPH=True)

###########################################
# Let's make a plot of the tornadoes in storm motion relative coordinates:

storm.plot_TCtors_rotated()
