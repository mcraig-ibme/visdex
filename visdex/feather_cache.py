import logging
import os
import tempfile

import pandas as pd
import numpy as np

class FeatherCache:

    CACHE_FILES = {
        "cluster": "cluster.feather",
        "parsed": "df_parsed.feather",
        "columns": "df_columns.feather",
        "df": "df.feather",
        "filtered": "df_filtered.feather",
        "corr": "corr.feather",
        "pval": "pval.feather",
        "logs": "logs.feather",
        "flattened_logs": "flattened_logs.feather"
    }

    def __init__(self, cachedir=None):
        self.log = logging.getLogger(__name__)
        if cachedir is not None:
            self._cachedir = cachedir
        else:
            self._cachedir = tempfile.TemporaryDirectory(prefix="visdex")
        self.log.info("cache at {}".format(self._cachedir.name))

    def load(self, name):
        fname = os.path.join(self._cachedir.name, name + ".feather")
        self.log.debug(f"load {name} {fname}")

        try:
            df = pd.read_feather(fname)
            self.log.debug(f"df: {df}")
            try:
                index_df = pd.read_feather(os.path.join(self._cachedir.name, name + "_index.feather"))
                self.log.debug(f"index df: {index_df}")
                df.set_index([c for c in index_df["col_name"]], inplace=True, verify_integrity=True, drop=True)
            except FileNotFoundError:
                self.log.warn(f"No index found")
        except FileNotFoundError:
            self.log.warn(f"Not found: {fname}")
            df = pd.DataFrame()

        return df

    def store(self, name, df):
        fname = os.path.join(self._cachedir.name, name + ".feather")
        self.log.debug(f"store {name} {fname}")
        if df is None:
            # Don't bother storing empty DF - load will just return a empty DF if
            # the file does not exist
            if os.path.exists(fname):
                os.remove(fname)
        else:
            self.log.debug(f"df: {df}")
            self.log.debug(f"index: {df.index}")
            index_cols = [col if col is not None else "index" for col in df.index.names]
            self.log.debug(f"index cols: {index_cols}")
            df = df.reset_index()
            self.log.debug(f"df (reset): {df}")
            df.to_feather(fname)
            if index_cols:
                self.log.debug(f"storing index: {index_cols}")
                index_df = pd.DataFrame(index_cols, columns=["col_name"])
                index_df.to_feather(os.path.join(self._cachedir.name, name + "_index.feather"))
