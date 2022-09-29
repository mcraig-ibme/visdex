"""
visdex: Manages access to generic CSV-based data sets
"""
import os

import pandas as pd
from flask_login import current_user

from .data_store import DataStore

class GenericCsv(DataStore):
    def __init__(self, flask_app, datadir, **kwargs):
        DataStore.__init__(self, flask_app, **kwargs)
        self._datadir = datadir
        self._dictdir = os.path.join(datadir, kwargs.get("dictdir", "dictionary"))
        self._extension = kwargs.get("extension", "csv")
        self._sep = kwargs.get("separator", ",")
        self._quotechar = kwargs.get("quotechar", '"')
        self._key_fields = [kwargs.get("key", "subjid"),]
        self._skiprows = [kwargs.get("skiprows", 1)]

        self.datasets = self._get_known_datasets()
        self.imaging_types = []

        self.log.info("Generic CSV data store at %s", self._datadir)
        self.log.info("Found %i data sets", len(self.datasets))
        self.log.info(self.datasets)

        if not os.path.exists(self._dictdir):
            self.log.warn(f"No dictionary directory found {self._dictdir} - will not use")
            self._dictdir = None

    def get_fields(self, *dataset_names):
        """
        :return: DataFrame containing all fields and descriptions in list of data sets
        """
        dict_dfs = []
        for name in dataset_names:
            dict_df = self._dictionary(name)
            dict_df.drop(dict_df[dict_df.ElementName.isin(self._key_fields)].index, inplace=True)
            dict_dfs.append(dict_df)

        if not dict_dfs:
            return pd.DataFrame()
        else:
            return pd.concat(dict_dfs, axis=0)

    def get_data(self, dataset, fields):
        """
        Build a data frame containing the selected fields from the corresponding datasets

        FIXME only supports single dataset for now
        """
        if not fields:
            self.log.info("No fields selected")
            return

        main_df = None
        for name in [dataset]:
            dict_df = self._dictionary(name)
            dataset_fields = dict_df["ElementName"]
            self.log.info(f"Dataset {name} has fields:")
            self.log.info(dataset_fields.tolist())
            fields_in_dataset = [f for f in fields if dataset_fields.str.contains(f).any()]
            if fields_in_dataset:
                self.log.info(f"Found fields {fields_in_dataset} in {name}")
                main_df = self._load_dataset(name)
                fields_of_interest = fields_in_dataset
                main_df = main_df[fields_of_interest]
 
        if main_df is not None:
            return main_df
        else:
            raise RuntimeError(f"Fields {fields} not found in dataset {dataset}")

    def _dictionary(self, dataset_name):
        if self._dictdir:
            dict_fname = os.path.join(self._dictdir, f"{dataset_name}.{self._extension}")
            dict_df = pd.read_csv(dict_fname, sep=self._sep, quotechar=self._quotechar)
            return dict_df
        else:
            dataset_df = self._load_dataset(dataset_name)
            return pd.DataFrame({"ElementName" : dataset_df.columns, "ElementDescription" : dataset_df.columns})

    def _get_known_datasets(self):
        """
        :return: pd.DataFrame containing data set information for all known data sets
        """
        dataset_names = [fname.split(".")[0] for fname in os.listdir(self._datadir) if fname.endswith(self._extension)]
        return pd.DataFrame({"title" : dataset_names, "shortname" : dataset_names})

    def _load_dataset(self, name):
        """
        Retrieve a dataset as a data frame
        """
        self.log.info(f"_get_dataset {name}")
        data_fname = os.path.join(self._datadir, f"{name}.{self._extension}")
        df = pd.read_csv(data_fname, sep=self._sep, quotechar=self._quotechar, skiprows=self._skiprows, low_memory=False)
        if self._key_fields[0] in df.columns:
            df.set_index(self._key_fields[0], inplace=True)
            if not df.index.is_unique:
                self.log.info(f"Dataset {name} is not uniquely keyed on {self._key_fields[0]}")
        else:
            self.log.info(f"Dataset {name} does not contain {self._key_fields[0]}")
        return df
