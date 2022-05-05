"""
Code to access ABCD data sets
"""
import os

import pandas as pd

from .data_store import DataStore, MAIN_DATA
from .feather_cache import FeatherCache
 
GLOBAL_DATADIR = "/home/martin/nda/"
GLOBAL_DICTDIR = os.path.join(GLOBAL_DATADIR, "dictionary")
STD_FIELDS = ["subjectkey", "src_subject_id", "interview_date", "interview_age", "sex", "eventname"]

class NdaData(DataStore):
    def __init__(self, study_name):
        DataStore.__init__(self, FeatherCache())
        self._study_name = study_name
        self._datadir = os.path.join(GLOBAL_DATADIR, study_name)
        self._dataset_names = [fname.split(".")[0] for fname in os.listdir(self._datadir) if fname.endswith(".txt")]
        self.log.info(self._dataset_names)
        df = pd.read_csv(os.path.join(GLOBAL_DATADIR, "datasets.tsv"), sep="\t", quotechar='"')
        self.log.info(df)
        self._all_datasets = df[df['shortname'].isin(self._dataset_names)]
        self._fields = {}
        self._datasets = []
        self.log.info("NDA data source for study %s", self._study_name)
        self.log.info("Found %i data sets", len(self._all_datasets))
        self.log.info(self._all_datasets)

    def get_all_datasets(self):
        """
        :return: pd.DataFrame containing data set information for all known data sets
        """
        return self._all_datasets

    def get_all_fields(self):
        """
        :return: DataFrame containing all fields in selected data sets
        """
        dfs = []
        for short_name in self._datasets:
            dict_fname = os.path.join(GLOBAL_DICTDIR, "%s.csv" % short_name)
            df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            df.drop(df[df.ElementName.isin(STD_FIELDS)].index, inplace=True)
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        else:
            return pd.concat(dfs, axis=0)

    def update(self):
        """
        Set the chosen fields

        Once called, the main data will be available
        
        :param fields: Dictionary of dataset short name : sequence of field names
        """

        # Build a data frame containing the selected fields from the corresponding datasets
        total_df = None
        for short_name in self._datasets:
            dict_fname = os.path.join(GLOBAL_DICTDIR, "%s.csv" % short_name)
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
        """
        Retrieve a dataset as a data frame
        """
        self.log.info(f"_get_dataset {short_name}")
        data_fname = os.path.join(self._datadir, "%s.txt" % short_name)
        df = pd.read_csv(data_fname, sep="\t", quotechar='"', skiprows=[1])
        #print("_get_dataset")
        print(df)
        #print(df.columns)
        df.set_index('subjectkey', inplace=True)
        if not df.index.is_unique:
            self.log.info(f"Dataset {short_name} is not subjectkey only")
            df.reset_index(inplace=True)
            if 'eventname' in df.columns:
                df.set_index(['subjectkey', 'eventname'], inplace=True)
            else:
                df.set_index(['subjectkey', 'visit'], inplace=True)
            if not df.index.is_unique:
                self.log.warn(f"Dataset {short_name} still doesn't have a unique index")
        else:
            self.log.info(f"Dataset {short_name} is indexed by subjectkey")
        print(df)
        return df
