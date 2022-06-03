"""
======================
Realtime Functionality
======================
This sample script shows how to use Tropycal to retrieve and plot real time tropical cyclones and potential formation. As this script was written on 2 June 2022, the data below is valid for the time this script was written.
"""

from tropycal import realtime

###########################################
# Reading In Realtime Dataset
# ---------------------------
#
# Let's start with the Realtime dataset by loading it into memory. By default, this reads in Best Track data from the National Hurricane Center (NHC) and filters for all active tropical cyclones and invests over the last 18 hours.
# 
# The default is to not include global storms in the Joint Typhoon Warning Center (JTWC)'s area of responsibility. To read in invests and storms within JTWC's domain, add a ``jtwc=True`` argument, followed by a ``jtwc_source`` argument which can be either "jtwc", "noaa", or "ucar". Read through the documentation to see the limits and pros/cons of each method.
# 
# To keep this demonstration simple, we'll solely focus on the North Atlantic basin.

realtime_obj = realtime.Realtime()

###########################################
# View Current Activity
# ---------------------
# 
# To quickly view the content of the Realtime object, we can simply print it:

realtime_obj

###########################################
# Alternatively, we can also list all active storms, along with an optional filter by basin.

realtime_obj.list_active_storms(basin='north_atlantic')

###########################################
# A new functionality with Tropycal v0.4 is to plot a summary of all ongoing activity, whether globally or by basin. This function can be highly customized, with more details in the documentation.
#
# At the time this script was written, Potential Tropical Cyclone #1 is active in the western Caribbean, with a 90% of development into a tropical cyclone. This will be plotted in the summary below:

realtime_obj.plot_summary(domain='north_atlantic')

###########################################
# Realtime Storms
# ---------------
# To retrieve a storm from a Realtime object, simply use its ``get_storm()`` method and provide an ID as listed in ``list_active_storms()``:

storm = realtime_obj.get_storm('AL012022')

###########################################
# This now returns a RealtimeStorm object. RealtimeStorm objects inherit the same functionality as Storm objects, but have additional functions unique to realtime storms. Additionally, as these can also be valid for invests, certain functionality that is only available for tropical cyclones (e.g., NHC forecasts or discussions) is unavailable for invests.
#
# Let's view what this RealtimeStorm object contains:

storm

###########################################
# A quick and easy way to check if a storm is an invest is by checking its ``invest`` attribute. This will let you know if you can use the full set of functionality available for tropical cyclones or not.

storm.invest

###########################################
# The next few blocks will overview functions unique to RealtimeStorm objects. We can easily retrieve the latest available forecast dictionary from NHC or JTWC, depending on what area of responsibility the storm is in.
#
# This function also calculates the forecast Accumulated Cyclone Energy (ACE), derived by combining its observed ACE through the current time plus the forecast ACE using linearly interpolated forecast sustained wind.

storm.get_forecast_realtime()

###########################################
# We can also plot it using the ``plot_forecast_realtime()`` method:

storm.plot_forecast_realtime()

###########################################
# Storm forecast discussions can also be retrieved for storms in NHC's area of responsibility, or the Prognostic Reasoning product for JTWC's area of responsibility.

storm.get_discussion_realtime()

###########################################
# Lastly, RealtimeStorms also provide the latest available information using the ``get_realtime_info()`` method.
#
# The default argument is ``source='all'``, which returns the latest available data whether from Best Track or NHC Public Advisories. Other possible values are "public_advisory", which only returns the latest public advisory, or "best_track", which only returns the latest best track data.

storm.get_realtime_info()
