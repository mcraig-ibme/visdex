import logging
import pandas as pd
import numpy as np
from psych_dashboard.app import indices, cache, use_redis

logging.getLogger(__name__)


def load_cluster_feather():
    """
    Utility function for reading the cluster DF from feather file, and setting the index.
    """
    dff = pd.read_feather("cluster.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff


def load_parsed_feather():
    """
    Utility function for reading raw DF from feather file - the MultiIndex has not been set, and this contains all the
    columns, not just the filtered subset. This is called just once, and converted to the main DF.
    """
    dff = pd.read_feather("df_parsed.feather")

    return dff


def load_columns_feather():
    """
    Utility function for reading the column-names DF from feather file, and setting the index.
    """
    dff = pd.read_feather("df_columns.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff


def load_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting the MultiIndex. This is called
    every time the main DF needs to be accessed.
    """
    dff = pd.read_feather("df.feather")

    if len(dff) > 0:
        dff.set_index(indices, inplace=True)
    return dff


def load_filtered_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather("df_filtered.feather")

    if len(dff) > 0:
        dff.set_index(indices, inplace=True)
    return dff


def load_corr():
    """
    Utility function for the common task of reading correlation DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather("corr.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff


def load(name):
    if use_redis:
        logging.debug(f"get cache {name}")
        try:
            df = cache.get(name)
            if df is None:
                return pd.DataFrame()
            else:
                return df
        except KeyError:
            return pd.DataFrame()
    else:
        # use feather
        if name == "cluster":
            return load_cluster_feather()
        elif name == "parsed":
            return load_parsed_feather()
        elif name == "columns":
            return load_columns_feather()
        elif name == "df":
            return load_feather()
        elif name == "filtered":
            return load_filtered_feather()
        elif name == "corr":
            return load_corr()
        elif name == "pval":
            return load_pval()
        elif name == "logs":
            return load_logs()
        elif name == "flattened_logs":
            return load_flattened_logs()
        else:
            raise KeyError(name)


def store(name, df):
    if use_redis:
        logging.debug(f"set cache {name}")
        cache.set(name, df)
        return None
    else:
        # use feather
        if df is None:
            df = pd.DataFrame()
        if name == "cluster":
            df.reset_index().to_feather("cluster.feather")
        elif name == "parsed":
            df.reset_index().to_feather("df_parsed.feather")
        elif name == "columns":
            df.reset_index().to_feather("df_columns.feather")
        elif name == "df":
            df.reset_index().to_feather("df.feather")
        elif name == "filtered":
            df.reset_index().to_feather("df_filtered.feather")
        elif name == "corr":
            df.reset_index().to_feather("corr.feather")
        elif name == "pval":
            df.reset_index().to_feather("pval.feather")
        elif name == "logs":
            df.reset_index().to_feather("logs.feather")
        elif name == "flattened_logs":
            df.reset_index().to_feather("flattened_logs.feather")
        else:
            raise KeyError(name)


def load_pval():
    """
    Utility function for the common task of reading p-value DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather("pval.feather")

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff


def load_logs():
    """
    Utility function for the common task of reading manhattan logs DF from feather file, and setting the index.
    """
    # The use of replace is required because the DF usually contains a column that is all-Nones, which remain
    # as Nones instead of being converted to np.nan unless we do that explicitly here.
    dff = pd.read_feather("logs.feather").replace([None], np.nan)

    if len(dff) > 0:
        dff.set_index("index", inplace=True)
    return dff


def load_flattened_logs():
    """
    Utility function for the common task of reading flattened manhattan logs DF from feather file, and setting the index.
    """
    dff = pd.read_feather("flattened_logs.feather")

    if len(dff) > 0:
        dff.set_index(["first", "second"], inplace=True)
    return dff["value"]
