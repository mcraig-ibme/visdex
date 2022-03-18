"""
Cache for parsed/processed data using Apache Feather files
"""
import logging
import os
import tempfile

import pandas as pd

from .data_store import DataStore
from .feather_cache import FeatherCache
 
DATADIR = "/home/martin/nda/abcd"
DICTDIR = os.path.join(DATADIR, "abcd-4.0-data-dictionaries")

class AbcdData(DataStore):
    def __init__(self):
        DataStore.__init__(self, FeatherCache())
        self._datasets = pd.read_csv(os.path.join(DATADIR, "datasets.tsv"), sep="\t", quotechar='"')
        print(self._datasets)

    def get_main(self):
        raise NotImplementedError()

    def get_columns(self):
        raise NotImplementedError()

    def set_columns(self, cols):
        """
        Set the chosen fields
        
        :param cols: Dictionary of dataset short name : sequence of field names
        """
        raise NotImplementedError()

    def get_datasets(self):
        return self._datasets

    def get_fields_in_datasets(self, datasets):
        """
        :return: Sequence of field names for a set of data sets
        
        :param datasets: Sequence of data set short names
        """
        dfs = []
        for short_name in datasets:
            dict_fname = os.path.join(DICTDIR, "%s.csv" % short_name)
            df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        else:
            return pd.concat(dfs, axis=0)
