"""
Functions for loading/saving data frame objects using Feather
"""
import logging
import pandas as pd
import numpy as np
from visdex.app import indices, cache, use_redis

logging.getLogger(__name__)

def load_cluster_feather():
    """
    Utility function for reading the cluster DF from feather file, and setting the
    index.
    """
    dff = pd.read_feather("cluster.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff

def load_parsed_feather():
    """
    Utility function for reading raw DF from feather file - the MultiIndex has not been
    set, and this contains all the columns, not just the filtered subset. This is called
    just once, and converted to the main DF.
    """
    dff = pd.read_feather("df_parsed.feather")
    return dff

def load_columns_feather():
    """
    Utility function for reading the column-names DF from feather file, and setting the
    index.
    """
    dff = pd.read_feather("df_columns.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff

def load_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting
    the MultiIndex. This is called every time the main DF needs to be accessed.
    """
    dff = pd.read_feather("df.feather")

    logging.debug(f"load_feather:\n{dff}")
    if len(dff) > 0:
        dff.set_index("index", inplace=True)
        # FIXME cut/paste put into helper function
        for index in known_indices:
            if all([col in dff for col in index]):
                dff.set_index(index, inplace=True, verify_integrity=True, drop=False)
                break

    logging.debug(f"load_feather2:\n{dff}")
    return dff

def load_filtered_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting
    the MultiIndex. This is called every time the main DF, filtered by missing values,
    needs to be accessed.
    """
    dff = pd.read_feather("df_filtered.feather")

    logging.debug(f"load_filtered_feather:\n{dff}")
    if len(dff) > 0:
        dff.set_index("index", inplace=True)
        # FIXME cut/paste put into helper function
        for index in known_indices:
            if all([col in dff for col in index]):
                dff.set_index(index, inplace=True, verify_integrity=True, drop=False)
                break

    logging.debug(f"load_filtered_feather2:\n{dff}")
    return dff

def load_corr():
    """
    Utility function for the common task of reading correlation DF from feather file,
    and setting the index.
    """
    dff = pd.read_feather("corr.feather")
    l = len(dff)
    logging.debug(f"load_corr: {dff} {l}")
    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    logging.debug(f"load_corr2: {dff} {l}")
    return dff

def load_pval():
    """
    Utility function for the common task of reading p-value DF from feather file, and
    setting the index.
    """
    dff = pd.read_feather("pval.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff

def load_logs():
    """
    Utility function for the common task of reading manhattan logs DF from feather file,
    and setting the index.
    """
    # The use of replace is required because the DF usually contains a column that is
    # all-Nones, which remain as Nones instead of being converted to np.nan unless we
    # do that explicitly here.
    dff = pd.read_feather("logs.feather").replace([None], np.nan)

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff

def load_flattened_logs():
    """
    Utility function for the common task of reading flattened manhattan logs DF from
    feather file, and setting the MultiIndex.
    """
    dff = pd.read_feather("flattened_logs.feather")

    if len(dff) > 0:
        dff.set_index(["first", "second"], inplace=True)
    return dff["value"]

def load(name):
    if use_redis:
        logging.debug(f"get cache {name}")
        try:
            df = cache.get(name)
            if df is None:
                return pd.DataFrame()
            return df
        except KeyError:
            return pd.DataFrame()
    else:
        # use feather
        logging.debug(f"loading feather: {name}")
        try:
            if name == "cluster":
                return load_cluster_feather()
            if name == "parsed":
                return load_parsed_feather()
            if name == "columns":
                return load_columns_feather()
            if name == "df":
                return load_feather()
            if name == "filtered":
                return load_filtered_feather()
            if name == "corr":
                return load_corr()
            if name == "pval":
                return load_pval()
            if name == "logs":
                return load_logs()
            if name == "flattened_logs":
                return load_flattened_logs()
            raise KeyError(name)
        except FileNotFoundError:
            return pd.DataFrame()

def store(name, df):
    if use_redis:
        logging.debug(f"set cache {name}")
        cache.set(name, df)
    else:
        # use feather
        if df is None:
            df = pd.DataFrame()

        # Map from file nickname to filename
        feather_filenames_dict = {
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

        try:
            logging.debug(f"store: {name}\n{df}")
            df = df.reset_index()
            df.to_feather(feather_filenames_dict[name])
            logging.debug(f"store2: {name}\n{df}")
        except KeyError:
            raise KeyError(name)

