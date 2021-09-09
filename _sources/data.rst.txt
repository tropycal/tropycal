############
Data Sources
############

Tropycal currently makes use of two well-established tropical cyclone reanalysis datasets:

HURTDAT2
--------
The HURDAT2_ reanalysis is an actively updated reanalysis dataset for tropical cyclones in the North Atlantic, East Pacific and Central Pacific basins. The responsible WMO Agency is the **National Hurricane Center (NHC)** in the North Atlantic and East Pacific basins, and the **Central Pacific Hurricane Center (CPHC)** in the Central Pacific basin. Sustained wind is 1-minute averaged, and data is available back to 1851 for the North Atlantic basin and 1949 for the East & Central Pacific basins.

**Caveats:**

* The "Satellite Era" is generally considered to have begun in 1979 with consistent, reliable satellite observations. Satellite data was generally available before then but on a more limited basis. As such, there are more storms in earlier years that were detected post-operationally from the `HURDAT2 Re-analysis Project`_. Because of a higher likelihood of missed storms in earlier years, some climatological analyses, such as accumulated cyclone energy (ACE) or count of major hurricanes, should be started from a more recent year (e.g., 1950) as opposed to 1851.

* The methodology for identifying and naming subtropical cyclones has changed over the years. Some decades had more frequent subtropical cyclones identified in HURDAT2 (e.g., 1970s) than others. The current methodology of naming subtropical storms using the same naming list as tropical storms dates back to 2002. Keep this caveat in mind when using tropycal to analyze subtropical cyclones.

.. _HURDAT2: https://www.nhc.noaa.gov/data/#hurdat
.. _HURDAT2 Re-analysis Project: https://www.aoml.noaa.gov/hrd/data_sub/re_anal.html

----

.. _ibtracs-caveats:

IBTrACS
-------
The IBTrACS_ reanalysis is a comprehensive dataset combining tropical cyclone data from multiple WMO agencies across the world. IBTrACS contains many data sources, some which have advantages over others. This results in discrepancies in storm tracks, storms that were or were not tropical, or sustained wind measurement that vary between agency. tropycal offers 3 modes of reading in IBTrACS data, with the pros and cons of each method listed below.

**General Caveats:**

* Subtropical cyclones are not identified by every WMO agency. For the North Atlantic basin, refer to the HURDAT2 subtropical cyclone caveats listed above.

* Cyclone Catarina (2004) was not officially tracked by any WMO agency in real time. A more thorough reanalysis exists in literature using McTaggart-Cowan et al. (2006). This data can be read into IBTrACS using the ``catarina=True`` flag when declaring an instance of TrackDataset.

.. _IBTrACS: https://www.ncdc.noaa.gov/ibtracs/

----

World Meteorological Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The WMO section of IBTrACS lists data from each WMO agency for each tropical cyclone within its area of responsibility. For instance, if a hurricane was to cross from the Central Pacific into the West Pacific, the first half of its life cycle would consist of data from CPHC, and the second half would consist of data from the Japan Meteorological Agency (JMA). This can be accessed via TrackDataset using the ``ibtracs_mode="wmo"`` keyword argument.

**Pros:**

* The WMO section misses relatively few storms, is more comprehensive than the other methods, and is considered to be official.

* Minimum sea level pressure data is more reliable for past storms with less missing data.

**Cons:**

* The sustained wind threshold differs by WMO agency. The NHC/CPHC use 1-minute sustained wind, while JMA uses 10-minute sustained wind. To resolve this discrepancy, wind is provided both in its original format, and using the generally accepted conversion value of 0.88. However, it is not recommended to use this data for calculations utilizing sustained wind such as accumulated cyclone energy (ACE).

* TCs outside of the NHC/CPHC domain often lack sustained wind data at the start and/or end of their life cycles.

----

Joint Typhoon Warning Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The JTWC provides unofficial reanalysis tropical cyclone data for all tropical cyclones globally outside of the NHC/CPHC's area of responsibility. This can be accessed via TrackDataset using the ``ibtracs_mode="jtwc"`` keyword argument.

**Pros:**

* Sustained wind is reported using 1-minute averages, making it consistent with NHC/CPHC and rendering it useful for calculations such as accumulated cyclone energy (ACE).

**Cons:**

* Especially in earlier years, numerous storms are missing, and storms that do exist are lacking wind and/or mean sea level pressure data which is readily availabie via other reanalysis datasets.
* Some storm track data is inaccurate; for instance, JTWC has `Cyclone Tracy (1974)`_ dissipating off the coast of Australia, when in reality it made landfall as a Category 3-equivalent hurricane.

.. _Cyclone Tracy (1974): https://en.wikipedia.org/wiki/Cyclone_Tracy

----

Neumann
~~~~~~~
Another reanalysis dataset provided for the Southern Hemisphere via IBTrACS is Neumann. This dataset is generally more reliable for older storms in the Southern Hemisphere than JTWC data, containing storms that were either missing in JTWC data or with correct track evolution that was incorrect in JTWC data. This can be accessed via TrackDataset using the ``ibtracs_mode="jtwc_neumann"`` keyword argument, which obtains JTWC data and replaces it with Neumann data for storms where such data is available.

**Pros:**

* Sustained wind is reported using 1-minute averages, making it consistent with NHC/CPHC and rendering it useful for calculations such as accumulated cyclone energy (ACE).
* This dataset improves on missing or incorrect data from JTWC, such as Cyclone Tracy with a correct evolution and landfall.

**Cons:**

* Some storms are still missing from this dataset that exist in the WMO dataset.
* Some cyclone tracks are either broken into multiple segments, or slightly altered from their official WMO or JTWC coordinates.