r"""Functionality for reading and analyzing recon data."""

from .dataset import ReconDataset, hdobs, dropsondes, vdms
from .realtime import RealtimeRecon, Mission

import sys
if 'sphinx' not in sys.modules:
    from .plot import ReconPlot
