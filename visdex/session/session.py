"""
Class for sever-side session data storage

Session data is stored in a subdir of a global session cache dir and can include
Pandas data frames and generic string properties. It also maintains a last_used
property containing the date time last accessed so old sessions can be cleaned up
"""
import datetime
import os
import logging
import pathlib
import shutil

import pandas as pd
import numpy as np

class Session:
    """
    A Session object is instantiated on *each request* and provides access to
    cached server-side data and properties
    """

    def __init__(self, global_session_dir, uid):
        """
        :param global_session_dir: Global session cache directory
        :param uid: Session unique ID
        """
        self.log = logging.getLogger(__name__)
        self._sessdir = os.path.join(global_session_dir, uid)
        if not os.path.exists(self._sessdir):
            os.makedirs(self._sessdir)

    def get_prop(self, prop):
        """
        Get a string property
        """
        try:
            with open(os.path.join(self._sessdir, prop + ".prop"), "r") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def set_prop(self, prop, value):
        """
        Set a string property
        """
        with open(os.path.join(self._sessdir, prop + ".prop"), "w") as f:
            f.write(value)

    def last_used(self):
        """
        :return: Date/time this session was last accessed
        """
        return datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(self._sessdir, "last_used")))

    def touch(self):
        """
        Mark this session as having been accessed
        """
        self.log.info(f"Session dir {self._sessdir} accessed")
        pathlib.Path(os.path.join(self._sessdir, "last_used")).touch()

    def load(self, name, keep_index_cols=False):
        """
        Load data from the cache

        :param name: Name of data
        :param keep_index_cols: If True, index columns will remain as columns in the returned data frame
        :return: pd.DataFrame containing cached data. If no cached data found, empty data
                 frame will be returned.
        """
        fname = self._data_fname(name)
        self.log.info(f"load {name} {fname}")

        try:
            df = pd.read_feather(fname)
            self.log.debug(f"df: {df}")
            try:
                index_df = pd.read_feather(self._index_fname(name))
                self.log.debug(f"index df: {index_df}")
                index_cols = [c for c in index_df["col_name"]]
                self.log.debug(f"Restoring index cols: {index_cols}")
                if index_cols == ["index"]:
                    keep_index_cols = False
                df.set_index(index_cols, inplace=True, drop=not keep_index_cols)
                self.log.debug(f"df (index restored): {df}")
            except FileNotFoundError:
                self.log.warning(f"No index found")
        except FileNotFoundError:
            self.log.warning(f"Not found: {fname}")
            df = pd.DataFrame()

        return df

    def store(self, name, df):
        """
        Store data in the cache

        :param name: Name of data
        :param df: pd.DataFrame containing data. May be empty or None if there
                   is no data to be stored. In this case, subsequent calls to
                   ``load`` for this data will return an empty pd.DataFrame
        """
        fname = self._data_fname(name)
        self.log.info(f"store {name} {fname}")
        if df is None:
            # Don't bother storing empty DF. Just remove any existing file
            # so load will just return a empty DF
            if os.path.exists(fname):
                os.remove(fname)
        else:
            self.log.debug(f"df: {df}")
            index_cols = [col if col is not None else "index" for col in df.index.names]
            self.log.debug(f"Resetting index cols: {index_cols}")
            df = df.reset_index()
            self.log.debug(f"df (reset): {df}")
            df.to_feather(fname)
            if index_cols:
                self.log.debug(f"Storing index: {index_cols}")
                index_df = pd.DataFrame(index_cols, columns=["col_name"])
                index_df.to_feather(self._index_fname(name))

    def free(self):
        shutil.rmtree(self._sessdir)

    def _data_fname(self, name):
        return os.path.join(self._sessdir, name + ".feather")

    def _index_fname(self, name):
        return self._data_fname(name + "_index")
       
    def filter(self, df, missing_values_cutoff, predicates):
        if df.empty:
            self.log.info("No data to filter")
            return

        self.log.info("Filter: pre-filter %s", df.shape)
        if missing_values_cutoff is None:
            missing_values_cutoff = 101

        # Apply row filters
        for column, operator, value in predicates:
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
                        if pd.api.types.is_string_dtype(df[column]):
                            value = f'"{value}"'
                        if operator == "contains":
                            query = f'`{column}`.str.contains({value})'
                        else:
                            query = f'`{column}` {operator} {value}'
                        df = df.query(query, engine="python")

        # Apply missing values threshold to remove columns with too many missing values
        if missing_values_cutoff > 0:
            percent_missing = df.isnull().sum() * 100 / len(df)
            keep_cols = [col for col, missing in zip(list(df.columns), percent_missing) if missing <= missing_values_cutoff]
            self.log.info("Filter: keep cols %s", keep_cols)
            df = df[keep_cols]

        self.log.info("Filter: post-filter %s", df.shape)

        to_drop = [c for c in df.index.names if c in df]
        df.drop(columns=to_drop, inplace=True)
        return df

    def description(self, df):
        """
        :return: DataFrame containing description of the selected data. There is a row for
                 each field with count, basic summary stats and % of missing values
        """
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
