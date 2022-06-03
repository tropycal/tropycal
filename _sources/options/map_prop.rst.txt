.. _options-prop-all:

####################
Map Plotting Options
####################

Various plotting functions in Tropycal make use of properties to allow the user to customize the plots. These properties are listed below.

.. _options-map-prop:

Map prop
========

The following table lists options that can be passed to the "map_prop" argument as a dictionary. A sample usage block is included below:

.. code-block:: python
    
    storm.plot(map_prop={'figsize':(14,9),'linewidth':1.0})

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - figsize
     - Figure size in inches, horizontal by vertical. Default is (14,9).
   * - dpi
     - Figure resolutions in pixels per inch. Default is 200.
   * - res
     - Resolution of political and geographic boundaries. Options are 'l', 'm', 'h'. Default is 'm'.
   * - linewidth
     - Line width for political and geographic boundaries. Default is 0.5.
   * - linecolor
     - Line color for political and geographic boundaries. Default is black.
   * - land_color
     - Color used to fill land. Default is '#FBF5EA'.
   * - ocean_color
     - Color used to fill oceans and lakes. Default is '#EDFBFF'.

.. _options-prop:

Tracks Properties
=================

The following sections lists options that can be passed to the "prop" argument as a dictionary. A sample usage block is included below:

.. code-block:: python
    
    storm.plot(prop={'dots':True,'linecolor':'k'})

Generic prop options
--------------------

The following properties are available for any function that involves plotting storm tracks.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - dots
     - Boolean whether to plot storm track dots along the storm track. Default is False.
   * - fillcolor
     - Fill color for storm track dots. Refer to table below for special color options.
   * - linecolor
     - Line color for storm track. Refer to table below for special color options.
   * - cmap
     - Colormap used for fill and/or line color, if not a single color nor 'category'. Can be a string or a dict of values and corresponding colors.
   * - levels
     - List of levels corresponding to cmap, if not coloring by a single color nor 'category'.
   * - linewidth
     - Line width for storm track. Default varies by function.
   * - ms
     - Size of storm track dots. Default is 7.5.
   * - plot_names
     - For plotting multiple storms or seasons, determines whether to plot storm name labels.

The following special options are available for ``linecolor`` or ``fillcolor``:

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Value
     - Description
   * - "category"
     - Default color map for SSHWS category.
   * - "vmax"
     - Color by maximum sustained wind.
   * - "mslp"
     - Color by minimum MSLP.
   * - "dvmax_dt"
     - Fill color by change in sustained wind speed. Only available for interpolated storm objects, retrieved using ``storm.interp()``.
   * - "speed"
     - Fill color by forward speed of tropical cyclone. Only available for interpolated storm objects, retrieved using ``storm.interp()``.

.. _options-prop-nhc:

plot_nhc_forecast
-----------------

The following properties are available only for the ``tropycal.tracks.Storm.plot_nhc_forecast()`` function.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cone_lw
     - Line width for the cone of uncertainty. Default is 1.0.
   * - cone_alpha
     - Transparency for the cone of uncertainty. Default is 0.6.

.. _options-prop-gridded:

gridded_stats
-------------

The following properties are available only for the ``tropycal.tracks.TrackDataset.gridded_stats()`` function.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cmap
     - Colormap to use for the plot. If string 'category' is passed (default), uses a pre-defined color scale corresponding to the Saffir-Simpson Hurricane Wind Scale.
   * - clevs
     - Contour levels for the plot. Default is minimum and maximum values in the grid.
   * - left_title
     - Title string for the left side of the plot. Default is the string passed via the 'request' keyword argument.
   * - right_title
     - Title string for the right side of the plot. Default is 'All storms'.

.. _options-summary:

Realtime Summary
================

The following properties are available only for the ``tropycal.realtime.Realtime.plot_summary()`` function.

prop_two
--------

The following properties are available for plotting NHC Tropical Weather Outlook (TWO).

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - plot
     - Boolean to determine whether to plot NHC TWO. Default is True.
   * - days
     - Number of days for TWO. Can be either 2 or 5. Default is 5.
   * - fontsize
     - Font size for text label. Default is 12.

prop_invest
-----------

The following properties are available for plotting invests.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - plot
     - Boolean to determine whether to plot active invests. Default is True.
   * - linewidth
     - Line width for past track. Default is 0.8. Set to zero to not plot line.
   * - linecolor
     - Line color for past track. Default is black.
   * - linestyle
     - Line style for past track. Default is dotted.
   * - fontsize
     - Font size for invest name label. Default is 12.
   * - ms
     - Marker size for invest location. Default is 14.

prop_storm
----------

The following properties are available for plotting storms.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - plot
     - Boolean to determine whether to plot active storms. Default is True.
   * - linewidth
     - Line width for past track. Default is 0.8. Set to zero to not plot line.
   * - linecolor
     - Line color for past track. Default is black.
   * - linestyle
     - Line style for past track. Default is dotted.
   * - fontsize
     - Font size for storm name label. Default is 12.
   * - fillcolor
     - Fill color for storm location marker. Default is color by SSHWS category ("category").
   * - label_category
     - Boolean for whether to plot SSHWS category on top of storm location marker. Default is True.
   * - ms
     - Marker size for storm location. Default is 14.

prop_cone
---------

The following properties are available for plotting realtime cone of uncertainty.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - plot
     - Boolean to determine whether to plot cone of uncertainty & forecast track for active storms. Default is True.
   * - linewidth
     - Line width for forecast track. Default is 1.5. Set to zero to not plot line.
   * - alpha
     - Opacity for cone of uncertainty. Default is 0.6.
   * - days
     - Number of days for cone of uncertainty, from 2 through 5. Default is 5.
   * - fillcolor
     - Fill color for forecast dots. Default is color by SSHWS category ("category").
   * - label_category
     - Boolean for whether to plot SSHWS category on top of forecast dots. Default is True.
   * - ms
     - Marker size for forecast dots. Default is 12.

.. _options-prop-recon-plot:

Recon Properties
================

plot_points
-----------

The following properties are available only for the ``tropycal.recon.ReconDataset.plot_points()`` function.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cmap
     - Colormap to use for the plot. If string 'category' is passed (default), uses a pre-defined color scale corresponding to the Saffir-Simpson Hurricane Wind Scale.
   * - levels
     - Levels for the color scale. If None (default), these are automatically generated.
   * - sortby
     - Variable to sort observations by. Default is the variable specified for plotting.
   * - ms
     - Size of observation dots. Default is 7.5.

.. _options-prop-recon-swath:

plot_swath and plot_map
-----------------------

The following properties are available only for the ``tropycal.recon.ReconDataset.plot_swath()`` and ``tropycal.recon.ReconDataset.plot_map()`` functions.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cmap
     - Colormap to use for the plot. If string 'category' is passed (default), uses a pre-defined color scale corresponding to the Saffir-Simpson Hurricane Wind Scale.
   * - levels
     - Levels for the color scale. If None (default), these are automatically generated.
   * - left_title
     - Title string for the left side of the plot. Default is automatically generated based on the requested variable.
   * - right_title
     - Title string for the right side of the plot. Default is 'All storms'.
   * - pcolor
     - Boolean for whether to use ``matplotlib.pyplot.pcolor()`` if set to True (default). If False, uses ``matplotlib.pyplot.contourf()``.

.. _options-prop-recon-hovmoller:

plot_hovmoller
--------------

The following properties are available only for the ``tropycal.recon.ReconDataset.plot_hovmoller()`` function.

.. list-table:: 
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cmap
     - Colormap to use for the plot. If string 'category' is passed (default), uses a pre-defined color scale corresponding to the Saffir-Simpson Hurricane Wind Scale.
   * - levels
     - Levels for the color scale. If None (default), these are automatically generated.
   * - smooth_contourf
     - Boolean determining whether to draw a smooth contourfill plot (True, default) or discrete intervals (False).
