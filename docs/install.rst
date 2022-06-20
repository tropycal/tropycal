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
* xarray >= 0.10.7
* networkx >= 2.0.0
* pyshp >= 2.1

To fully leverage tropycal's plotting capabilities, it is strongly recommended to have cartopy >= 0.17.0 installed.

------------
Installation
------------

From Conda
~~~~~~~~~~

As of v0.3, Tropycal can be installed via conda::

    conda install -c conda-forge tropycal

From Pip
~~~~~~~~

As with before, Tropycal can also be installed via pip::

    pip install tropycal

From Source
~~~~~~~~~~~

Tropycal can also be installed via cloning it from github. Running the following commands
will build and install tropycal into your python installation::

    git clone https://github.com/tropycal/tropycal
    cd tropycal
    python setup.py install
