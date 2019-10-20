Tropycal
============================

Tropycal is a Python package intended to simplify the process of retrieving and analyzing tropical cyclone data, both for past storms and in real time, and is geared towards the research and operational meteorology sectors.

Tropycal can read in HURDAT2 and ibtracs reanalysis data and operational National Hurricane Center (NHC) Best Track data and conform them to the same format, which can be used to perform climatological, seasonal and individual storm analyses. For each individual storm, operational NHC forecasts, aircraft reconnaissance data, and any associated tornado activity can be retrieved and plotted.

For an example on how to use Tropycal in a Python script, please refer to the :doc:`sample_usage` page. Additional information about the data sources used and their caveats is available in the :doc:`data` page.

.. warning::
  Tropycal is a new package. The syntax of classes and methods in the library is subject to change in future releases, which will also significantly optimize performance and speed of some functionalities.

.. _Github: https://github.com/tropycal/tropycal/issues

.. toctree::
   :maxdepth: 2
   :hidden:

   install
   support
   api/index
   sample_usage
   data


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
