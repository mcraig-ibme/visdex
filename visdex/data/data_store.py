"""
Base class for data store with functions for creating and
retrieving per-session instances
"""
import logging
import os
from datetime import datetime
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from flask import current_app, session

LOG = logging.getLogger(__name__)
DATA_STORE_TIMEOUT_LAG_MINUTES = 5

# Data store implementing classes
DATA_STORE_CLASSES = {
    "user" : "UserData"
}

# Data store instances keyed by session UID
LOG.info("Clearing session stores")
SESSION_DATA_STORES = {
}

def load_config(fname):
    from . import data_stores
    global DATA_STORE_CLASSES
    LOG.info(f"start data store configuration: {DATA_STORE_CLASSES}")
    LOG.info("Load_config: %s", fname)
    try:
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                config = json.load(f)
            LOG.info(config)
            for cat, cat_defn in config.items():
                LOG.info(cat)
                LOG.info(cat_defn)
                mod = getattr(data_stores, cat, None)
                if mod is None:
                    LOG.warn("No module for data store category: {cat}")
                    continue

                class_name = None
                for k, v in cat_defn.items():
                    LOG.info("Attr: %s=%s", k, v)
                    if k == "class_name" :
                        LOG.info("Found class name: %s" % v)
                        class_name = v
                        cls = getattr(mod, v, None)
                        if cls is None:
                            LOG.warn("Class not found: {cat}.{class_name}")
                    else:
                        LOG.info("Setting attribute: %s=%s", k, v)
                        setattr(mod, k, v)
                
                LOG.info("Adding module")
                if class_name:
                    LOG.info("Setting class name: %s" % class_name)
                    DATA_STORE_CLASSES[cat] = class_name
                else:
                    LOG.warn(f"Class name not defined for data store category {cat}")
            LOG.info(f"Data store configuration: {DATA_STORE_CLASSES}")
        else:
            LOG.warn(f"Failed to load data store config from {fname} - no such file")
    except Exception as exc:
        LOG.warn(exc)

def prune_sessions():
    """
    Flag current session's last used time to now and remove sessions that have timed out

    We need this because although Flask can expire sessions itself, all that means is
    that the session data is removed, i.e. we will no longer get a session object with
    the expired uid, and we can't remove the associated data store without that uid.

    We remove data stores 5 minutes after the corresponding session timeout.
    """
    current_uid = session["uid"].hex
    if current_uid in SESSION_DATA_STORES:
        SESSION_DATA_STORES[uid].last_used = datetime.now()

    timeout_minutes = current_app.config.get("TIMOUT_MINUTES", 1) + DATA_STORE_TIMEOUT_LAG_MINUTES
    for uid in list(SESSION_DATA_STORES.keys()):
        ds = SESSION_DATA_STORES[uid]
        dt = datetime.now() - ds.last_used
        LOG.info(f"Session data store {uid} last used {dt.minutes} mins ago")
        if dt.minutes > timeout_minutes:
            LOG.info(f"Cleaning up expired session data store: {uid}")
            remove(uid)

def set_session_data_store(name="user"):
    """
    Set the data store for the current session
    
    :param name: Name of data store in form <category>.<name>
                 or just <category>
    """
    uid = session["uid"].hex
    remove(uid)

    from . import data_stores
    parts = name.split(".", 1)
    cat = parts[0]
    class_name = DATA_STORE_CLASSES.get(cat, None)
    mod = getattr(data_stores, cat, None)
    if mod is None or class_name is None:
        raise RuntimeError(f"Unknown data store category: '{cat}'")

    cls = getattr(mod, class_name, None)
    if cls is not None:
        name = "" if len(parts) == 1 else parts[1]
        LOG.info(f"Setting data store for session {uid} to {cls} {class_name}")
        SESSION_DATA_STORES[uid] = cls(name)
        LOG.info(SESSION_DATA_STORES)
    else:
        raise RuntimeError(f"Unknown data store class: '{cat}' '{class_name}'")

def get():
    """
    Get the DataStore for the current session
    """
    LOG.info("get")
    LOG.info(SESSION_DATA_STORES)
    uid = session["uid"].hex
    if uid not in SESSION_DATA_STORES:
        LOG.error("No data store for session: %s" % uid)
        set_session_data_store()
    return SESSION_DATA_STORES[uid]

def remove(uid=None):
    """
    Remove the DataStore for the current session and free any associated resources
    """
    if uid is None:
        uid = session["uid"].hex
    if uid in SESSION_DATA_STORES:
        LOG.info("Removing data store for session: %s" % uid)
        SESSION_DATA_STORES[uid].free()
        del SESSION_DATA_STORES[uid]

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
        self.last_used = datetime.now()
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
        df = self.load(MAIN_DATA, keep_index_cols=True)
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

        df.drop(columns=df.index.names, inplace=True)
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
