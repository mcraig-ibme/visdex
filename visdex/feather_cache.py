"""
Cache for parsed/processed data using Apache Feather files
"""
import logging
import os
import tempfile

import pandas as pd
import numpy as np

class FeatherCache:

    def __init__(self, cachedir=None):
        """
        :param cachedir: Name of dir to use for cache - will create temporary dir if not specified
        """
        self.log = logging.getLogger(__name__)
        if cachedir is not None:
            self._cachedir = cachedir
        else:
            self._cachedir = tempfile.TemporaryDirectory(prefix="visdex")
        self.log.info("cache at {}".format(self._cachedir.name))

    def load(self, name, keep_index_cols=False):
        """
        Load data from the cache

        :param name: Name of data
        :param keep_index_cols: If True, index columns will remain as columns in the returned data frame
        :return: pd.DataFrame containing cached data. If no cached data found, empty data
                 frame will be returned.
        """
        fname = self._fname(name)
        self.log.info(f"load {name} {fname}")

        try:
            df = pd.read_feather(fname)
            self.log.debug(f"df: {df}")
            try:
                index_df = pd.read_feather(self._index_fname(name))
                self.log.debug(f"index df: {index_df}")
                index_cols = [c for c in index_df["col_name"]]
                self.log.debug(f"Restoring index cols: {index_cols}")
                if index_cols == ["index"]:
                    keep_index_cols = False
                df.set_index(index_cols, inplace=True, verify_integrity=True, drop=not keep_index_cols)
                self.log.debug(f"df (index restored): {df}")
            except FileNotFoundError:
                self.log.warning(f"No index found")
        except FileNotFoundError:
            self.log.warning(f"Not found: {fname}")
            df = pd.DataFrame()

        return df

    def store(self, name, df):
        """
        Store data in the cache

        :param name: Name of data
        :param df: pd.DataFrame containing data. May be empty or None if there
                   is no data to be stored. In this case, subsequent calls to
                   ``load`` for this data will return an empty pd.DataFrame
        """
        fname = self._fname(name)
        self.log.info(f"store {name} {fname}")
        if df is None:
            # Don't bother storing empty DF. Just remove any existing file
            # so load will just return a empty DF
            if os.path.exists(fname):
                os.remove(fname)
        else:
            self.log.debug(f"df: {df}")
            index_cols = [col if col is not None else "index" for col in df.index.names]
            self.log.debug(f"Resetting index cols: {index_cols}")
            df = df.reset_index()
            self.log.debug(f"df (reset): {df}")
            df.to_feather(fname)
            if index_cols:
                self.log.debug(f"Storing index: {index_cols}")
                index_df = pd.DataFrame(index_cols, columns=["col_name"])
                index_df.to_feather(self._index_fname(name))

    def _fname(self, name):
        return os.path.join(self._cachedir.name, name + ".feather")

    def _index_fname(self, name):
        return self._fname(name + "_index")
