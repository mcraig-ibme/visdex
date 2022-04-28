"""
Code to access ABCD data sets
"""
import os
import logging

import pandas as pd

from .data_store import DataStore, MAIN_DATA
from .feather_cache import FeatherCache
 
DATADIR = "/home/martin/nda/abcd"
DICTDIR = os.path.join(DATADIR, "abcd-4.0-data-dictionaries")
STD_FIELDS = ["subjectkey", "src_subject_id", "interview_date", "interview_age", "sex", "eventname"]

class AbcdData(DataStore):
    def __init__(self):
        DataStore.__init__(self, FeatherCache())
        self._all_datasets = pd.read_csv(os.path.join(DATADIR, "datasets.tsv"), sep="\t", quotechar='"')
        self._fields = {}
        self._datasets = []
        print(self._datasets)

    def get_all_datasets(self):
        """
        :return: pd.DataFrame containing data set information for all known data sets
        """
        return self._all_datasets

    def select_datasets(self, datasets):
        """
        Select data sets of interest

        :param datasets: Sequence of data set short names
        """
        self._datasets = datasets

    def get_all_fields(self):
        """
        :return: DataFrame containing all fields in selected data sets
        """
        dfs = []
        for short_name in self._datasets:
            dict_fname = os.path.join(DICTDIR, "%s.csv" % short_name)
            df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            df.drop(df[df.ElementName.isin(STD_FIELDS)].index, inplace=True)
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        else:
            return pd.concat(dfs, axis=0)

    def select_fields(self, fields):
        """
        Set the chosen fields

        Once called, the main data will be available
        
        :param fields: Dictionary of dataset short name : sequence of field names
        """
        self._fields = [f["ElementName"] for f in fields]
        self.log.info(self._fields)

        total_df = None
        for short_name in self._datasets:
            dict_fname = os.path.join(DICTDIR, "%s.csv" % short_name)
            df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            df_fields = df["ElementName"]
            self.log.info(f"Dataset {short_name} has fields {df_fields}")
            fields_in_dataset = [f for f in self._fields if df_fields.str.contains(f).any()]
            if fields_in_dataset:
                self.log.info(f"Found fields {fields_in_dataset} in {short_name}")
                df = self._get_dataset(short_name)
                fields_of_interest = fields_in_dataset
                df = df[fields_of_interest]
 
        self.store(MAIN_DATA, df)

    def _get_dataset(self, short_name):
        self.log.info(f"_get_dataset {short_name}")
        data_fname = os.path.join(DATADIR, "%s.txt" % short_name)
        df = pd.read_csv(data_fname, sep="\t", quotechar='"', skiprows=[1])
        print(df)
        #print(df.columns)
        df.set_index('subjectkey')
        if not df.index.is_unique:
            self.log.info(f"Dataset {short_name} is not subjectkey only")
            df.reset_index()
            if 'eventname' in df.columns:
                df.set_index(['subjectkey', 'eventname'])
            else:
                df.set_index(['subjectkey', 'visit'])
            if not df.index.is_unique:
                self.log.warn(f"Dataset {short_name} still doesn't have a unique index")
        else:
            self.log.info(f"Dataset {short_name} is indexed by subjectkey")
        print(df)
        return df
