import pandas as pd
from psych_dashboard.app import indices


def load_feather(df_loaded):
    """
    Utility function for the common task of reading DF from feather file, and setting the MultiIndex. This is called
    every time the main DF needs to be accessed.
    """
    dff = pd.read_feather('df.feather')

    if df_loaded and len(dff) > 0:
        dff.set_index(indices, inplace=True)
    return dff
