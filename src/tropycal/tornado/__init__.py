r"""Functionality for reading and analyzing tornado data."""
print("in tornado")
from .dataset import TornadoDataset

import sys
if 'sphinx' not in sys.modules:
    from .plot import TornadoPlot