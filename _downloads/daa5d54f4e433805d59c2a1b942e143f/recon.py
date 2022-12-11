"""
============================
Aircraft Recon Data Analysis
============================
This sample script shows how to use Tropycal to retrieve, analyze and plot aircraft reconnaissance missions.
"""

from tropycal import tracks, recon
import numpy as np
import datetime as dt

###########################################
# Reading In Recon Dataset
# ------------------------
# There are several ways of retrieving a ReconDataset object. The first step is to retrieve a TrackDataset object, which for this example we'll use the HURDATv2 database for the North Atlantic basin.

basin = tracks.TrackDataset('north_atlantic')

###########################################
# We'll now retrieve a Storm object for Hurricane Irma (2017), which will be used for this example code.

storm = basin.get_storm(('Irma',2017))

###########################################
# Now that we have a Storm object, there are several ways of retrieving recon data.
#
# 1. The first and easiest way is through the Storm object, which already contains an empty instance of ReconDataset stored as an attribute ``storm.recon``. This method will be highlighted in this script. Its methods can then be accessed as ``storm.recon.plot_summary()``, for example.
#
# 2. The second method is through retrieving an instance of ReconDataset, providing the storm object as an argument (e.g., ``recon_obj = recon.ReconDataset(storm)``). Its methods can then be accessed as ``recon_obj.plot_summary()``, for example.
#
# 3. The individual sub-classes (to be discussed later) can also be created individually of a ReconDataset object, providing the storm object as an argument (e.g., ``dropsondes = recon.dropsondes(storm)``).
#
# For the rest of this script, we'll be using the first method.
#
# -----------------
# Recon Sub-Classes
# -----------------
# The three primary sub-classes of the recon module are:
#
# .. list-table:: 
#    :widths: 25 75
#    :header-rows: 1
#
#    * - Class
#      - Description
#    * - hdobs
#      - Class containing all High Density Observations (HDOBs) for this Storm.
#    * - dropsondes
#      - Class containing all dropsondes for this Storm.
#    * - vdms
#      - Class containing all Vortex Data Messages (VDMs) for this Storm.
#
# The following functionality is used to retrieve data for each sub-class. Each class has a ``to_pickle()`` method, which can be used to save the data once it's been read in as a local pickle file, which can be re-read in later.

storm.recon.get_vdms()
#Save pickle file of VDM data (list of dictionaries)
storm.recon.vdms.to_pickle(f'{storm.name}{storm.year}_vdms.pickle')

storm.recon.get_dropsondes()
#Save pickle file of Dropsonde data (list of dictionaries)
storm.recon.dropsondes.to_pickle(f'{storm.name}{storm.year}_dropsondes.pickle')

storm.recon.get_hdobs()
#Save pickle file of HDOB data (Pandas dataframe)
storm.recon.hdobs.to_pickle(f'{storm.name}{storm.year}_hdobs.pickle')

###########################################
# These can be initialized again any time later with the saved pickle files:
#
# >>> storm.recon.get_vdms(f'{storm.name}{storm.year}_vdms.pickle')
# >>> storm.recon.get_dropsondes(f'{storm.name}{storm.year}_dropsondes.pickle')
# >>> storm.recon.get_hdobs(f'{storm.name}{storm.year}_hdobs.pickle')
#

###########################################
# Visualizing ReconDataset
# ------------------------
# Print the recon object to see a summary of the data in recon from the three objects:

storm.recon

###########################################
# A summary of recon data for this storm can also be plotted:

storm.recon.plot_summary()

###########################################
# Find the mission numbers that were active during a given time, within a distance (in km) from the storm:

storm.recon.find_mission(dt.datetime(2017,9,7,12), distance=200)

###########################################
# Then plot a summary from just that mission:

storm.recon.plot_summary(mission=17)

###########################################
# High Density Observations (HDOBs)
# ---------------------------------
# The first class we'll be reviewing is the HDOBs class, which is the largest containing the most data given the high frequency of observations. Let's start by viewing the HDOB summary:

storm.recon.hdobs

###########################################
# And view HDOB Pandas DataFrame data:

storm.recon.hdobs.data

###########################################
# Plot a summary of the recon data for this storm, using peak wind gusts with a custom colormap:

storm.recon.hdobs.plot_points('pkwnd',prop={'cmap':{1:'dodgerblue',2:'gold',3:'firebrick'},'levels':np.arange(20,161,10)})

###########################################
# Plot a hovmoller from recon data interpolated to time and radius - note the eyewall replacement cycles:

storm.recon.hdobs.plot_hovmoller(varname='pkwnd',prop={'cmap':{1:'dodgerblue',2:'gold',3:'firebrick'},'levels':np.arange(20,161,10)})

###########################################
# Plot a map valid at 1200 UTC 6 September 2017 interpolated to time and space:

time = dt.datetime(2017,9,6,12)
storm.recon.hdobs.plot_maps(time=time,varname='pkwnd',prop={'cmap':{1:'dodgerblue',2:'gold',3:'firebrick'},'levels':np.arange(20,161,10)})

###########################################
# Dropsonde Data
# --------------
# Next we'll take a look at the dropsonde data for Hurricane Irma. First, let's take a look at the dropsonde summary:

storm.recon.dropsondes

###########################################
# Now use the ``sel`` function to subset to only dropsondes released in the eyewall:

storm.recon.dropsondes.sel(location='eyewall')

###########################################
# We can view data from one of the dropsondes by using the ``isel`` method to select a dropsonde number:

storm.recon.dropsondes.sel(location='eyewall').isel(23).data

###########################################
# Select one of the eyewall dropsondes and plot the Skew-T:

storm.recon.dropsondes.sel(location='eyewall').isel(23).plot_skewt()

###########################################
# Plot a map of dropsonde points colored by 850mb temperature:

storm.recon.dropsondes.plot_points('temp',level=850,prop={'cmap':{1:'dodgerblue',2:'gold',3:'firebrick'},'ms':20})

###########################################
# Plot a map of only upper-air dropsondes released at 300mb and above, colored by wind speed at 300mb:

storm.recon.dropsondes.sel(top=(None,300)).plot_points('wspd',level=300,prop={'cmap':{1:'dodgerblue',2:'gold',3:'firebrick'},'ms':20})

###########################################
# Vortex Data Messages (VDMs)
# ---------------------------
# The last class we'll look at is the VDM class. Let's start off by viewing a summary of VDM data for Hurricane Irma:

storm.recon.vdms

###########################################
# Let's look at decoded VDMs for a specific pass:

storm.recon.vdms.isel(10).data

###########################################
#Plot a map of VDM center location, colored by minimum pressure (default):

storm.recon.vdms.plot_points(prop={'cmap':{3:'dodgerblue',2:'gold',1:'firebrick'},'ms':40})
