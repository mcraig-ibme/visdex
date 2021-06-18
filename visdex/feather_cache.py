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
        self.log.info("Feather cache at {}".format(self._cachedir.name))
        
    def load(self, name):
        if name == "cluster":
            return self._load_cluster_feather()
        if name == "parsed":
            return self._load_parsed_feather()
        if name == "columns":
            return self._load_columns_feather()
        if name == "df":
            return self._load_feather()
        if name == "filtered":
            return self._load_filtered_feather()
        if name == "corr":
            return self._load_corr()
        if name == "pval":
            return self._load_pval()
        if name == "logs":
            return self._load_logs()
        if name == "flattened_logs":
            return self._load_flattened_logs()
        raise KeyError(name)

    def store(self, name, df):
        if df is None:
            df = pd.DataFrame()

        try:
            df.reset_index().to_feather(os.path.join(self._cachedir.name, self.CACHE_FILES[name]))
        except KeyError:
            raise KeyError(name)

    def _load_cluster_feather(self):
        """
        Utility function for reading the cluster DF from feather file, and setting the
        index.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "cluster.feather"))

        if len(dff) > 0:
            dff.set_index("index", inplace=True)
        return dff


    def _load_parsed_feather(self):
        """
        Utility function for reading raw DF from feather file - the MultiIndex has not been
        set, and this contains all the columns, not just the filtered subset. This is called
        just once, and converted to the main DF.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "df_parsed.feather"))

        return dff


    def _load_columns_feather(self):
        """
        Utility function for reading the column-names DF from feather file, and setting the
        index.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "df_columns.feather"))

        if len(dff) > 0:
            dff.set_index("index", inplace=True)
        return dff


    def _load_feather(self):
        """
        Utility function for the common task of reading DF from feather file, and setting
        the MultiIndex. This is called every time the main DF needs to be accessed.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "df.feather"))

        #if len(dff) > 0:
        #    dff.set_index(indices, inplace=True)
        return dff


    def _load_filtered_feather(self):
        """
        Utility function for the common task of reading DF from feather file, and setting
        the MultiIndex. This is called every time the main DF, filtered by missing values,
        needs to be accessed.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "df_filtered.feather"))

        #if len(dff) > 0:
        #    dff.set_index(indices, inplace=True)
        return dff


    def _load_corr(self):
        """
        Utility function for the common task of reading correlation DF from feather file,
        and setting the index.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "corr.feather"))

        if len(dff) > 0:
            dff.set_index("index", inplace=True)
        return dff



    def _load_pval(self):
        """
        Utility function for the common task of reading p-value DF from feather file, and
        setting the index.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "pval.feather"))

        if len(dff) > 0:
            dff.set_index("index", inplace=True)
        return dff


    def _load_logs(self):
        """
        Utility function for the common task of reading manhattan logs DF from feather file,
        and setting the index.
        """
        # The use of replace is required because the DF usually contains a column that is
        # all-Nones, which remain as Nones instead of being converted to np.nan unless we
        # do that explicitly here.
        dff = pd.read_feather(os.path.join(self._cachedir.name, "logs.feather")).replace([None], np.nan)

        if len(dff) > 0:
            dff.set_index("index", inplace=True)
        return dff


    def _load_flattened_logs(self):
        """
        Utility function for the common task of reading flattened manhattan logs DF from
        feather file, and setting the MultiIndex.
        """
        dff = pd.read_feather(os.path.join(self._cachedir.name, "flattened_logs.feather"))

        if len(dff) > 0:
            dff.set_index(["first", "second"], inplace=True)
        return dff["value"]
