"""
==================
Reading SHIPS Data
==================
This sample script illustrates how to leverage Tropycal's SHIPS reading capability.
"""

import json
import datetime as dt
import matplotlib.pyplot as plt
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
# Reading in SHIPS data
# ---------------------
# SHIPS data can be read either using the standalone ``ships.Ships`` class by providing the SHIPS text data, or via Storm objects. The more conventional method is via Storm objects, as this automatically locates the SHIPS text files from UCAR's online archive dating back to 2011.
#
# Let's retrieve an instance of Hurricane Ida from 2021 for our example:

storm = basin.get_storm(('ida',2021))

###########################################
# Next, let's search for times where SHIPS forecasts are available:

available_times = storm.search_ships()

for time in available_times:
    print(time)

###########################################
# Now that we know what times are available, let's use the SHIPS forecast from 1800 UTC 27 August 2021. The following line retrieves an instance of a Ships object and stores it in the variable ``ships``:

ships = storm.get_ships(dt.datetime(2021, 8, 27, 18))

###########################################
# We now have a Ships object containing the forecast initialized at this time. Let's peek at our Ships object:

print(ships)

###########################################
# Retrieving Data
# ---------------
#
# There's a lot of data in this object, as SHIPS files provide a lot of variables over many forecast hours. If we want data only valid at a specific forecast hour, we can use the following method:

output_dict = ships.get_snapshot(hour=48)

# Format nicely for documentation purposes
print(json.dumps(output_dict, indent=4))

###########################################
# We can also fetch the rapid intensification probabilities that SHIPS provides:

output_dict = ships.get_ri_prob()

# Format nicely for documentation purposes
print(json.dumps(output_dict, indent=4))

###########################################
# Ships objects also allow us to convert data to other formats, such as xarray Datasets:

ds = ships.to_xarray()
print(ds)

###########################################
# Visualizing Data
# ----------------
#
# Tropycal's Ships class comes with a built-in function to plot a basic summary of the SHIPS forecast and key diagnostics. We can use it as follows:

ships.plot_summary()

###########################################
# Let's say we want to make a plot of several metrics that affect a storm's intensity:

# Create figure
fig,ax = plt.subplots(figsize=(9,6), dpi=200)
ax.set_facecolor('#f6f6f6')
ax.grid()

# Plot variables
ax.plot(ships.fhr, ships.shear_kt, color='blue', label='Shear (kt)')
ax.plot(ships.fhr, ships.sst_c, color='red', label='SSTs (C)')
ax.plot(ships.fhr, ships['700_500_rh'], color='green', label='700-500mb RH (%)')
ax.set_ylabel('Shear, SST, RH')

# Add twin axes for wind speed
ax2 = ax.twinx()
ax2.plot(ships.fhr, ships.vmax_land_kt, color='k', linewidth=2.5)
ax2.set_ylabel('Wind Speed (kt)')

# Format and label x-axis
ax.set_xticks(range(0,ships.fhr[-1]+1,24))
ax.set_xlabel('Forecast Hour')

# Add legend and title
ax.legend()
ax.set_title(f"SHIPS Forecast for {ships.attrs['storm_name']}",
             loc='left', fontsize=14, fontweight='bold')
ax.set_title(f"Initialized: {ships.attrs['forecast_init'].strftime('%H%M UTC %d %b %Y')}",
             loc='right', fontsize=10)
