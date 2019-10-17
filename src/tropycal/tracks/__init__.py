r"""Functionality for reading and analyzing storm tracks."""

from .dataset import TrackDataset
from .storm import Storm
from .season import Season

import sys
if 'sphinx' not in sys.modules:
    from .plot import TrackPlot