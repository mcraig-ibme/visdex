import pandas as pd
import numpy as np
from psych_dashboard.app import indices


def load_cluster_feather():
    """
    Utility function for reading the cluster DF from feather file, and setting the index.
    """
    dff = pd.read_feather('cluster.feather')

    if len(dff) > 0:
        dff.set_index('index', inplace=True)
    return dff


def load_parsed_feather():
    """
    Utility function for reading raw DF from feather file - the MultiIndex has not been set, and this contains all the
    columns, not just the filtered subset. This is called just once, and converted to the main DF.
    """
    dff = pd.read_feather('df_parsed.feather')

    return dff


def load_columns_feather():
    """
    Utility function for reading the column-names DF from feather file, and setting the index.
    """
    dff = pd.read_feather('df_columns.feather')

    if len(dff) > 0:
        dff.set_index('index', inplace=True)
    return dff


def load_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting the MultiIndex. This is called
    every time the main DF needs to be accessed.
    """
    dff = pd.read_feather('df.feather')

    if len(dff) > 0:
        dff.set_index(indices, inplace=True)
    return dff


def load_filtered_feather():
    """
    Utility function for the common task of reading DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather('df_filtered.feather')

    if len(dff) > 0:
        dff.set_index(indices, inplace=True)
    return dff


def load_corr():
    """
    Utility function for the common task of reading correlation DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather('corr.feather')

    if len(dff) > 0:
        dff.set_index('index', inplace=True)
    return dff


def load_pval():
    """
    Utility function for the common task of reading p-value DF from feather file, and setting the MultiIndex. This is called
    every time the main DF, filtered by missing values, needs to be accessed.
    """
    dff = pd.read_feather('pval.feather')

    if len(dff) > 0:
        dff.set_index('index', inplace=True)
    return dff


def load_logs():
    """
    Utility function for the common task of reading manhattan logs DF from feather file, and setting the index.
    """
    # The use of replace is required because the DF usually contains a column that is all-Nones, which remain
    # as Nones instead of being converted to np.nan unless we do that explicitly here.
    dff = pd.read_feather('logs.feather').replace([None], np.nan)

    if len(dff) > 0:
        dff.set_index('index', inplace=True)
    return dff


def load_flattened_logs():
    """
    Utility function for the common task of reading flattened manhattan logs DF from feather file, and setting the index.
    """
    dff = pd.read_feather('flattened_logs.feather')

    if len(dff) > 0:
        dff.set_index(['first', 'second'], inplace=True)
    return dff['value']
