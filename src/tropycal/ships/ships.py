r"""Functionality for storing and analyzing SHIPS data."""

import numpy as np
import xarray as xr
import pandas as pd
from ..utils import ships_parser

class Ships():
    r"""
    Initializes an instance of Ships.
    
    Parameters
    ----------
    content : str
        SHIPS file content. If initialized via a tropycal.tracks.Storm object, this does not need to be provided.
    storm_name : str, optional
        Storm name to associate with this SHIPS data. If initialized via a ``tropycal.tracks.Storm`` object, this does not need to be provided.
    
    Returns
    -------
    tropycal.ships.Ships
        Instance of a Ships object.
    
    Notes
    -----
    A Ships object is best retrieved from a Storm object's ``get_ships()`` method. For example, if the dataset read in is the default North Atlantic and the desired storm is Hurricane Michael (2018), SHIPS data would be retrieved as follows:
    
    .. code-block:: python

        from tropycal import tracks
        import datetime as dt
        
        # Retrieve storm object for Hurricane Michael
        basin = tracks.TrackDataset()
        storm = basin.get_storm(('michael',2018))
        
        # Retrieve instance of a Ships object
        ships = storm.get_ships(dt.datetime(2018, 10, 8, 0))

    """
    
    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __repr__(self):

        # Label object
        summary = ["<tropycal.ships.Ships>"]

        # Format keys for coordinates
        variable_keys = {}
        for key in self.dict.keys():
            dtype = type(self.dict[key][0]).__name__
            dtype = dtype.replace("_", "")
            variable_keys[key] = f"({dtype}) [{self.dict[key][0]} .... {self.dict[key][-1]}]"

        # Add data
        summary.append('\nVariables:')
        add_space = np.max([len(key) for key in variable_keys.keys()]) + 3
        for key in variable_keys.keys():
            key_name = key
            summary.append(
                f'{" "*4}{key_name:<{add_space}}{variable_keys[key]}')
        
        # Add additional information
        summary.append('\nRI Probabilities:')
        add_space = np.max([len(key) for key in self.dict_ri.keys()]) + 3
        for key in self.dict_ri.keys():
            key_name = key + ":"
            entry = f'{self.dict_ri[key]["probability"]}% ({self.dict_ri[key]["prob / climo"]}x climo mean)'
            summary.append(f'{" "*4}{key_name:<{add_space}}{entry}')
            
        # Add additional information
        summary.append('\nAttributes:')
        add_space = np.max([len(key) for key in self.attrs.keys()]) + 3
        for key in self.attrs.keys():
            key_name = key + ":"
            summary.append(f'{" "*4}{key_name:<{add_space}}{self.attrs[key]}')

        return "\n".join(summary)
    
    def __init__(self, content, storm_name=None):
        
        ships = ships_parser(content)
        self.dict = ships['data']
        self.dict_ri = ships['data_ri']
        self.attrs = ships['data_attrs']
        
        # Add storm name if provided
        self.attrs['name'] = 'UNKNOWN'
        if storm_name is not None: self.attrs['name'] = storm_name
        
        # Set data variables as attributes of this object
        for key in self.dict:
            self[key] = np.array(self.dict[key])
    
    def get_variables(self):
        r"""
        Return list of available forecast variables.
        
        Returns
        -------
        list
            List containing available forecast variables.
        
        Notes
        -----
        These variables can be accessed from two methods:
        
        >>> # 1. Access directly from object
        >>> ships.fhr
        array([  0,   6,  12,  18,  24,  36,  48,  60,  72,  84,  96, 108, 120])
        
        >>> # 2. Access from internal data dictionary
        >>> ships.dict['fhr']
        array([  0,   6,  12,  18,  24,  36,  48,  60,  72,  84,  96, 108, 120])
        
        """
        
        return list(self.dict.keys())
    
    def get_snapshot(self, hour):
        r"""
        Return all variables valid at a single forecast hour.
        
        Parameters
        ----------
        hour : int
            Requested forecast hour.
        
        Returns
        -------
        dict
            Dictionary containing all variables valid at this forecast hour.
        """
        
        # Error check
        if hour not in self.dict['fhr']:
            raise ValueError('Requested forecast hour is not available.')
        idx = self.dict['fhr'].index(hour)
        
        # Format dict
        data = {}
        for key in self.dict:
            data[key] = self.dict[key][idx]
        
        return data
    
    def get_ri_prob(self):
        r"""
        Return rapid intensification probabilities.
        
        Returns
        -------
        dict
            Dictionary containing rapid intensification probabilities.
        """
        
        return self.dict_ri
    
    def to_xarray(self):
        r"""
        Convert data to an xarray Dataset.
        
        Returns
        -------
        xarray.Dataset
            SHIPS data converted to an xarray Dataset.
        """
        
        # Set up empty dict for dataset
        time = self.dict['fhr']
        ds = {}

        # Add every key containing a list into the dict, otherwise add as an attribute
        keys = [k for k in self.dict.keys() if k != 'fhr']
        for key in keys:
            ds[key] = xr.DataArray(self.dict[key],
                                   coords = [time],
                                   dims = ['fhr'])

        # Convert entire dict to a Dataset
        ds = xr.Dataset(ds, attrs=self.attrs)

        # Return dataset
        return ds
    
    def to_dataframe(self, attrs_as_columns=False):
        r"""
        Convert data to a pandas DataFrame object.

        Parameters
        ----------
        attrs_as_columns : bool
            If True, adds Ships object attributes as columns in the DataFrame returned. Default is False.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame object containing SHIPS data.
        """

        # Set up empty dict for dataframe
        df = {}

        # Add every key containing a list into the dict
        for key in self.dict:
            df[key] = self.dict[key]
        if attrs_as_columns:
            for key in self.attrs:
                df[key] = self.attrs[key]

        # Convert entire dict to a DataFrame
        df = pd.DataFrame(df)

        # Return dataset
        return df
