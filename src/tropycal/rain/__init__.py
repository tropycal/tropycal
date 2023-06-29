r"""Functionality for reading and analyzing tropical cyclone rain data."""

from .dataset import RainDataset

import sys
if 'sphinx' not in sys.modules:
    from .plot import RainPlot
