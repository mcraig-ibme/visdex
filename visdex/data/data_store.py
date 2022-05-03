"""
Base class for data store with functions for creating and
retrieving per-session instances
"""
import logging

import numpy as np
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
    elif std_data == "abcd":
        from .abcd_data import AbcdData
        DATA_STORES[uid] = AbcdData()
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

    def filter(self):
        df = self.load(MAIN_DATA)
        self.log.info("Filter: pre-filter %s", df.shape)

        # Apply row filters
        for column, operator, value in self._predicates:
            if column not in df:
                self.log.warn("%s not found in data - ignoring row filter", column)
            elif operator not in ('==', '>', '<', '<-', '>='):
                self.log.warn("%s not a supported operator - ignoring row filter", operator)
            elif not value.strip():
                self.log.warn("No value given - ignoring row filter", operator)
            else:
                if is_string_dtype(df[column]):
                    query = f'`{column}` {operator} "{value}"'
                    self.log.info("Filtering string col: %s", query)
                else:
                    query = f'`{column}` {operator} {value}'
                    self.log.info("Filtering numeric col: %s", query)
                df = df.query(query)

        # Apply missing values threshold
        if self._missing_values_threshold > 0:
            percent_missing = df.isnull().sum() * 100 / len(df)
            keep_cols = [col for col, missing in zip(list(df.columns), percent_missing) if missing <= self._missing_values_threshold]
            self.log.info("Filter: keep cols %s", keep_cols)
            df = df[keep_cols]

        self.log.info("Filter: post-filter %s", df.shape)
        self.store(FILTERED, df)

    @property
    def description(self):
        df = self.load(MAIN_DATA)
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
