"""
Cache for parsed/processed data using Apache Feather files
"""
import logging
import os
import tempfile
import shutil

import pandas as pd

from .data_store import DataStore
from .feather_cache import FeatherCache

# Possible sets of index columns. 
# This is very data specific so we just try them in order and use the first set
# that matches the DF
KNOWN_INDICES = [
    ["SUBJECTKEY", "EVENTNAME"],
]

class UserData:

    def __init__(self):
        """
        :param cachedir: Name of dir to use for cache - will create temporary dir if not specified
        """
        DataStore.__init__(self, FeatherCache())

    def get_main(self, keep_index_cols=False):
        return self.load("parsed", keep_index_cols)

    def set_main(self, df):
        """
        Set the main data source in the cache
        """
        self.store("parsed", df)
        self.set_columns()

    def get_columns(self):
        return self.load("columns")

    def set_columns(self, cols=None):
        """
        Set the column filter which defines what main data columns 
        the user is interested in
        """
        if cols is None:
            cols = self.load("parsed").columns
        df = pd.DataFrame(cols, columns=["names"])
        self.store("columns", df)
        return self._filter_cols()
    
    def get_filtered(self, keep_index_cols=False):
        raise NotImplementedError()

    def _filter_cols(self):
        """
        Apply the column filter
        """
        df = self.load("parsed")
        warning =""

        # Read in column DataFrame, or just use all the columns in the DataFrame
        if len(df) > 0:
            variables_of_interest = list(self.load("columns")["names"])

            # Remove variables that don't exist in the dataframe
            missing_vars = [var for var in variables_of_interest if var not in df.columns]
            present_vars = [var for var in variables_of_interest if var in df.columns]

            if not variables_of_interest:
                warning = "No columns found in filter file - ignored"
            elif missing_vars and not present_vars:
                warning = "No columns found in filter file are present in main data - filter file ignored"
            elif missing_vars:
                warning = "Some columns found in filter file not present in main data - these columns ignored"

            # Keep only the columns listed in the filter file
            if present_vars:
                df = df[present_vars]

        self.log.info("df\n: %s" % str(df))

        # Reformat SUBJECTKEY if it doesn't have the underscore
        # TODO: remove this when unnecessary
        #if "SUBJECTKEY" in df:
        #    df["SUBJECTKEY"] = df["SUBJECTKEY"].apply(standardise_subjectkey)

        # Set certain columns to have more specific types.
        # if 'SEX' in df.columns:
        #     df['SEX'] = df['SEX'].astype('category')
        #
        # for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
        #     if column in df.columns:
        #         df[column] = df[column].astype('string')

        # Set index. We try all known indices and apply the first
        # one where all the columns are present.
        for index in KNOWN_INDICES:
            if all([col in df for col in index]):
                df.set_index(index, inplace=True, verify_integrity=True, drop=True)
                break

        # Store the combined DF, and set df-loaded-div to [True]
        self.store("df", df)
        return warning
