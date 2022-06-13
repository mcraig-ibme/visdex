"""
Base class for data store with functions for creating and
retrieving per-session instances
"""
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from flask import session

LOG = logging.getLogger(__name__)

DATA_STORES = {}

def init(std_data=None):
    uid = session["uid"].hex
    if uid in DATA_STORES:
        # Is this useful?
        DATA_STORES[uid].free()

    if std_data is None:
        from .user_data import UserData
        DATA_STORES[uid] = UserData()
    elif std_data in ("abcd", "earlypsychosis", "hcpageing"):
        from .nda_data import NdaData
        DATA_STORES[uid] = NdaData(std_data)
    else:
        raise RuntimeError(f"Unknown standard data type: '{std_data}'")

def get():
    uid = session["uid"].hex
    if uid not in DATA_STORES:
        LOG.error("No data store for session: %s" % uid)
        init()
    return DATA_STORES[uid]

def remove():
    uid = session["uid"].hex
    if uid in DATA_STORES:
        LOG.info("Removing data store for session: %s" % uid)
        del DATA_STORES[uid]

MAIN_DATA = "df"
FILTERED = "filtered"

class DataStore:
    """
    Stores and retrieves the data the user is working with

    The class has a storage backend that can store DataFrame objects.
    However it can define and return the main data however it wants.
    """

    def __init__(self, backend):
        self.log = logging.getLogger(__name__)
        self.backend = backend
        self._datasets = []
        self._fields = []
        self._missing_values_threshold = 101
        self._predicates = []

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, value):
        self.log.info("Set datasets: %s", value)
        if value != self._datasets:
            self._datasets = value
            self.update()
            self.filter()

    @property
    def fields(self):
        return self._fields

    @fields.setter
    def fields(self, value):
        self.log.info("Set fields: %s", value)
        if value != self._fields:
            self._fields = value
            self.update()
            self.filter()

    @property
    def missing_values_threshold(self):
        return self._missing_values_threshold

    @datasets.setter
    def missing_values_threshold(self, value):
        if value is None:
            value = 101
        self.log.info("Set missing_values_threshold: %s", value)
        if value != self._missing_values_threshold:
            self._missing_values_threshold = value
            self.filter()

    @property
    def predicates(self):
        return self._predicates

    @predicates.setter
    def predicates(self, value):
        if value is None:
            value = []
        self.log.info("Set predicates: %s", value)
        if value != self._predicates:
            self._predicates = value
            self.filter()

    @property
    def imaging_types(self):
        """
        :return: Sequence of imaging data types available for subjects (if any)
        """
        return []

    def filter(self):
        """
        Do row/column filtering on main data
        """
        df = self.load(MAIN_DATA)
        if df.empty:
            self.log.info("No data to filter")
            return

        self.log.info("Filter: pre-filter %s", df.shape)

        # Apply row filters
        for column, operator, value in self._predicates:
            if operator == "random":
                self.log.info("Doing random sample of %s", value)
                df = df.sample(value)
            elif column not in df:
                self.log.warn("%s not found in data - ignoring row filter", column)
            elif operator not in ('==', '>', '<', '<-', '>=', "Not empty", "contains"):
                self.log.warn("%s not a supported operator - ignoring row filter", operator)
            else:
                if operator == "Not empty":
                    df = df[df[column].notna()]
                else:
                    if not value.strip():
                        self.log.warn("No value given - ignoring row filter")
                    else:
                        if is_string_dtype(df[column]):
                            value = f'"{value}"'
                        if operator == "contains":
                            query = f'`{column}`.str.contains({value})'
                        else:
                            query = f'`{column}` {operator} {value}'
                        df = df.query(query, engine="python")

        # Apply missing values threshold to remove columns with too many missing values
        if self._missing_values_threshold > 0:
            percent_missing = df.isnull().sum() * 100 / len(df)
            keep_cols = [col for col, missing in zip(list(df.columns), percent_missing) if missing <= self._missing_values_threshold]
            self.log.info("Filter: keep cols %s", keep_cols)
            df = df[keep_cols]

        self.log.info("Filter: post-filter %s", df.shape)
        self.store(FILTERED, df)

    @property
    def description(self):
        """
        :return: DataFrame containing description of the selected data. There is a row for
                 each field with count, basic summary stats and % of missing values
        """
        df = self.load(MAIN_DATA)
        if df.empty:
            return pd.DataFrame()

        description_df = df.describe().transpose()

        # Add largely empty rows to the summary table for non-numeric columns.
        for col in df.columns:
            if col not in description_df.index:
                description_df.loc[col] = [df.count()[col]] + [np.nan] * 7

        # Calculate percentage of missing values
        description_df["% missing values"] = 100 * (1 - description_df["count"] / len(df.index))

        # Add the index back in as a column so we can see it in the table preview
        description_df.insert(loc=0, column="column name", value=description_df.index)

        # Reorder the columns so that 50% centile is next to 'mean'
        description_df = description_df.reindex(
            columns=[
                "column name",
                "count",
                "mean",
                "50%",
                "std",
                "min",
                "25%",
                "75%",
                "max",
                "% missing values",
            ]
        )

        return description_df

    def update(self):
        """
        Called to update the main data selection after datasets/fields changed
        """
        raise NotImplementedError()

    def store(self, name, df):
        """
        Store a generic data frame
        
        Useful for caching custom data
        """
        self.backend.store(name, df)
    
    def load(self, name, keep_index_cols=False):
        """
        Load a previously stored data frame
        """
        return self.backend.load(name, keep_index_cols)

    def free(self):
        """
        Called when data store is no longer required to
        free any files/in-memory data if required
        """
        self.backend.free()
