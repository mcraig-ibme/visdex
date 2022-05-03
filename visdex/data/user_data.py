"""
Manages access to user-uploaded data sets
"""
import base64
import csv
import io

import pandas as pd

from .data_store import DataStore
from .feather_cache import FeatherCache

# Possible sets of index columns. 
# This is very data specific so we just try them in order and use the first set
# that matches the DF
KNOWN_INDICES = [
    ["SUBJECTKEY", "EVENTNAME"],
]

class UserData(DataStore):

    def __init__(self):
        DataStore.__init__(self, FeatherCache())

    def update(self):
        if len(self._datasets) == 0:
            return
        elif len(self._datasets) > 1:
            raise ValueError("User data can only have a single source data set")

        contents, filename, date = self._datasets[0]
        if contents is None:
            return

        # Load the main data
        _content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        try:
            if filename.endswith("csv") or filename.endswith("txt"):
                # Assume that the user uploaded a delimited file
                dialect = csv.Sniffer().sniff(decoded.decode("utf-8")[:1024])
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), sep=dialect.delimiter, quotechar=dialect.quotechar, low_memory=False)
            elif filename.endswith("xlsx"):
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                raise NotImplementedError
        except Exception as e:
            raise ValueError(f"Error loading {filename}: {e}")
            
        # Filter the columns
        if self._fields:
            # Remove fields that don't exist in the source data
            missing_vars = [var for var in self._fields if var not in df.columns]
            present_vars = [var for var in self._fields if var in df.columns]

            if missing_vars and not present_vars:
                self.log.warn("No columns found in filter file are present in main data - filter file ignored")
                self._fields = []
            elif missing_vars:
                self.log.warn("Some columns found in filter file not present in main data - these columns ignored")
                self._fields = present_vars

        if self._fields:
            df = df[self._fields]

        # Filter the rows
        if self._predicates:
            pass

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
        self.store(MAIN_DATA, df)
