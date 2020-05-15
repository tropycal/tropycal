####################
Map Plotting Options
####################

Various plotting functions in Tropycal make use of additional properties in the keyword arguments 

Map prop
========

The following table lists keys that can be passed to the "map_prop" argument as a dictionary.

.. list-table:: Title
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

Prop
====

Generic prop options
--------------------

The following table lists keys that can be passed to the "prop" argument as a dictionary.

.. list-table:: Title
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - dots
     - Boolean whether to plot storm track dots along the storm track. Default is False.
   * - fillcolor
     - Fill color for storm track dots. If 'category', then uses a color scale varying by category.
   * - linecolor
     - Line color for storm track. If 'category', then uses a color scale varying by category.
   * - category_colors
     - Color scale to use for categories. Default is 'default'. Currently no other option is available.
   * - linewidth
     - Line width for storm track. Default varies by function.
   * - ms
     - Size of storm track dots. Default is 7.5.

plot_nhc_forecast
-----------------

The following properties are available only for the ``plot_nhc_forecast()`` function.

.. list-table:: Title
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - cone_lw
     - Line width for the cone of uncertainty. Default is 1.0.
   * - cone_alpha
     - Transparency for the cone of uncertainty. Default is 0.6.

gridded_stats
-------------

The following properties are available only for the ``gridded_stats()`` function.

.. list-table:: Title
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

More coming soon.

