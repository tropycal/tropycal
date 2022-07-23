.. _options-domain:

##################
Map Domain Options
##################

Tropycal offers a variety of pre-defined map domains to use for plotting, passed via the "domain" argument. These are listed below:

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Domain
     - Description
   * - "all"
     - Global projection, centered on the Prime Meridian.
   * - "north_atlantic"
     - North Atlantic Ocean basin.
   * - "east_pacific"
     - East Pacific Ocean basin.
   * - "west_pacific"
     - West Pacific Ocean basin.
   * - "north_indian"
     - North Indian Ocean basin.
   * - "south_indian"
     - South Indian Ocean basin.
   * - "australia"
     - Australian basin.
   * - "south_pacific"
     - South Pacific Ocean basin.
   * - "south_atlantic"
     - South Atlantic Ocean basin.
   * - "conus"
     - Continental United States.
   * - "east_conus"
     - Eastern United States.

A custom map projection can be defined for all functions using a dictionary, with keyword arguments corresponding to the map bounds. For example, use "n", "north", "N", or "North" to specify the north bound of the map. Sample usage is listed below:

.. code-block:: python
    
    #One method of creating a custom domain
    storm.plot(domain={'n':50,'s':30,'w':-70,'e':-30})
    
    #Another method of creating a custom domain
    storm.plot(domain={'north':50,'south':30,'west':-70,'east':-30})

The following domains are available for some, but not all plotting functions. Please refer to the specific function documentation to see if these projections are available.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Domain
     - Description
   * - "dynamic"
     - Dynamic map projection, centered on the feature of interest (e.g., storm track, tornadoes, recon data).
   * - "dynamic_forecast"
     - Dynamic map projection, focused entirely on the forecast track. Available only for forecast track plotting functions.
   * - "dynamic_tropical"
     - Dynamic map projection, excluding extratropical cyclone points. Available only for track plotting functions.

