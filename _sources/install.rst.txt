==================
Installation Guide
==================

------------
Requirements
------------

The primary dependencies of tropycal are as follows:

* matplotlib >= 2.2.2
* numpy >= 1.14.3
* scipy >= 1.1.0
* pandas >= 0.23.0
* geopy >= 1.18.1
* xarray >= 0.10.7
* networkx >= 2.0.0

To fully leverage tropycal's plotting capabilities, it is strongly recommended to have cartopy >= 0.17.0 installed.

------------
Installation
------------

The currently recommended method of installation is via pip::

    pip install tropycal

Tropycal can also be installed via cloning it from github. Running the following commands
will build and install tropycal into your python installation::

    git clone https://github.com/tropycal/tropycal
    cd tropycal
    python setup.py install
