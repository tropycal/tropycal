============
Data Sources
============

Tropycal currently makes use of two well-established tropical cyclone reanalysis datasets:

HURTDAT2
--------
The HURDAT2_ reanalysis is an actively updated reanalysis dataset for tropical cyclones in the North Atlantic, East Pacific and Central Pacific basins. The responsible WMO Agency is the **National Hurricane Center (NHC)** in the North Atlantic and East Pacific basins, and the **Central Pacific Hurricane Center (CPHC)** in the Central Pacific basin.

.. _HURDAT2: https://www.nhc.noaa.gov/data/#hurdat

----

ibtracs
-------
The ibtracs_ reanalysis is a comprehensive dataset combining tropical cyclone data from multiple worldwide WMO agencies across the world. ibtracs contains many data sources, some which have advantages over others, which can result in discrepancies such as storm tracks, storms that were or were not tropical, or sustained wind measurement that vary between agency. tropycal offers 3 modes of reading in ibtracs data, with the pros and cons of each method listed below:

.. _ibtracs: https://www.ncdc.noaa.gov/ibtracs/


World Meteorological Organization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The WMO section of ibtracs lists data from each WMO agency for each tropical cyclone within its area of responsibility. For instance, if a hurricane was to cross from the Central Pacific into the West Pacific, the first half of its life cycle would consist of data from CPHC, and the second half would consist of data from the Japan Meteorological Agency (JMA). This can be accessed via TrackDataset using the `ibtracs_mode="wmo"` keyword argument.

**Pros**:

* The WMO section misses relatively few storms, is more comprehensive than the other methods, and is considered to be official.

* Minimum sea level pressure data is more reliable for past storms with less missing data.

**Cons**:

* The sustained wind threshold differs by WMO agency. The NHC/CPHC use 1-minute sustained wind, while JMA uses 10-minute sustained wind. To resolve this discrepancy, wind is provided both in its original format, and using the generally accepted conversion value of 0.88. However, it is not recommended to use this data for calculations utilizing sustained wind such as accumulated cyclone energy (ACE).

* TCs outside of the NHC/CPHC domain often lack sustained wind data at the start and/or end of their life cycles.

Joint Typhoon Warning Center
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The JTWC provides unofficial reanalysis tropical cyclone data for all tropical cyclones globally outside of the NHC/CPHC's area of responsibility. This can be accessed via TrackDataset using the `ibtracs_mode="jtwc"` keyword argument.

**Pros**:

* Sustained wind is reported using 1-minute averages, making it consistent with NHC/CPHC and rendering it useful for calculations such as accumulated cyclone energy (ACE).

**Cons**

* Especially in earlier years, numerous storms are missing, and storms that do exist are lacking wind and/or mean sea level pressure data which is readily availabie via other reanalysis datasets.
* Some storm track data is inaccurate; for instance, JTWC has `Cyclone Tracy (1974)`_ dissipating off the coast of Australia, when in reality it made landfall as a Category 3-equivalent hurricane.

.. _Cyclone Tracy (1974): https://en.wikipedia.org/wiki/Cyclone_Tracy

Neumann
~~~~~~~
Another reanalysis dataset provided for the Southern Hemisphere is Neumann. This dataset is generally more reliable for older storms in the Southern Hemisphere than JTWC data. This can be accessed via TrackDataset using the `ibtracs_mode="jtwc_neumann"` keyword argument.

**Pros**:

* Sustained wind is reported using 1-minute averages, making it consistent with NHC/CPHC and rendering it useful for calculations such as accumulated cyclone energy (ACE).
* This dataset improves on missing or incorrect data from JTWC, such as Cyclone Tracy with a correct evolution and landfall.

**Cons**

* Some storms are still missing from this dataset that exist in the WMO dataset.
* Some cyclone tracks are either broken into multiple segments, or slightly altered from their official WMO or JTWC coordinates.