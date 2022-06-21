"""
visdex: Manages access to NDA data sets
"""
import os

import pandas as pd

from .data_store import DataStore

STD_FIELDS = ["subjectkey", "src_subject_id", "visit", "interview_date", "interview_age", "sex", "eventname"]
IMG_FIELDS = [
    "image_description", "scan_type", 
    #"magnetic_field_strength", "mri_repetition_time_pd", "mri_echo_time_pd", "flip_angle", 
    #"image_extent1", "image_extent2", "image_extent3", "image_extent4", "extent4_type", 
    #"image_unit1", "image_unit2", "image_unit3", "image_unit4", 
    "image_resolution1", "image_resolution2", "image_resolution3", "image_resolution4",
    #"image_slice_thickness", "image_orientation",
]
IMG_ROUND_FIELDS = [
    "image_resolution1", "image_resolution2", "image_resolution3", "image_resolution4",
]
IMG_ROUND_DPS = [
    2, 2, 2, 2
]

class NdaData(DataStore):
    def __init__(self, global_datadir, study_name, **kwargs):
        DataStore.__init__(self)
        self._global_datadir = global_datadir
        self._study_name = study_name
        self._dictdir = os.path.join(global_datadir, "dictionary")
        self._datadir = os.path.join(global_datadir, study_name)

        self.datasets = self._get_known_datasets()
        self.imaging_types = self._get_known_imaging_types()

        self.log.info("NDA data source for study %s", self._study_name)
        self.log.info("Found %i data sets", len(self.datasets))
        self.log.info(self.datasets)

    def get_fields(self, *dataset_short_names):
        """
        :return: DataFrame containing all fields in data sets
        """
        dfs = []
        for short_name in dataset_short_names:
            dict_fname = os.path.join(self._dictdir, "%s.csv" % short_name)
            df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            df.drop(df[df.ElementName.isin(STD_FIELDS)].index, inplace=True)
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        else:
            return pd.concat(dfs, axis=0)

    def get_data(self, dataset, fields):
        """
        Build a data frame containing the selected fields from the corresponding datasets
        """
        if not fields:
            self.log.info("No fields selected")
            return

        main_df = None
        for short_name in [dataset]:
            dict_fname = os.path.join(self._dictdir, "%s.csv" % short_name)
            dict_df = pd.read_csv(dict_fname, sep=",", quotechar='"')
            df_fields = dict_df["ElementName"]
            self.log.info(f"Dataset {short_name} has fields:")
            self.log.info(df_fields.tolist())
            fields_in_dataset = [f for f in fields if df_fields.str.contains(f).any()]
            if fields_in_dataset:
                self.log.info(f"Found fields {fields_in_dataset} in {short_name}")
                main_df = self._load_dataset(short_name)
                fields_of_interest = fields_in_dataset
                main_df = main_df[fields_of_interest]
 
        if main_df is not None:
            return main_df
        else:
            raise RuntimeError(f"Fields {fields} not found in dataset {dataset}")

    def imaging_links(self, ids, imaging_types):
        """
        :param ids: Data frame containing subject IDs
        :param imaging_types: Data frame containing selected imaging data types
        """
        ids.reset_index(drop=False, inplace=True)

        images = self._load_dataset("image03")
        images.reset_index(drop=False, inplace=True)

        selected_types = imaging_types.set_index("image_description").index
        image_types = images.set_index('image_description').index
        images = images[image_types.isin(selected_types)]

        # FIXME ids may include visit
        ids = ids.set_index('subjectkey').index
        image_ids = images.set_index('subjectkey').index
        images = images[image_ids.isin(ids)]

        return images[["subjectkey", "visit", "image_file"]]

    def _get_known_datasets(self):
        """
        :return: pd.DataFrame containing data set information for all known data sets
        """
        dataset_names = [fname.split(".")[0] for fname in os.listdir(self._datadir) if fname.endswith(".txt")]
        df = pd.read_csv(os.path.join(self._global_datadir, "datasets.tsv"), sep="\t", quotechar='"')
        return df[df['shortname'].isin(dataset_names)]

    def _get_known_imaging_types(self):
        """
        :return: DataFrame containing imaging types - as a minimum should contain the 'text' column
        """
        def format(r):
            return "%s (%s)" % (r.scan_type, r.image_description)
            #return "%s (%s) %.1f %.1f %.1f %.1fmm" % (r.scan_type, r.image_description, r.image_resolution1, r.image_resolution2, r.image_resolution3, r.image_resolution4)

        df = self._load_dataset("image03")
        df.reset_index(drop=True, inplace=True)
        df = df[IMG_FIELDS]
        df.fillna(0, inplace=True)
        for f, dps in zip(IMG_ROUND_FIELDS, IMG_ROUND_DPS):
            df[f] = df[f].round(dps)
        df.drop_duplicates(inplace=True)
        df['text'] = df.apply(format, axis=1)
        return df[['image_description', 'text']]

    def _load_dataset(self, short_name):
        """
        Retrieve a dataset as a data frame
        """
        self.log.info(f"_get_dataset {short_name}")
        data_fname = os.path.join(self._datadir, "%s.txt" % short_name)
        df = pd.read_csv(data_fname, sep="\t", quotechar='"', skiprows=[1], low_memory=False)
        df.set_index('subjectkey', inplace=True)
        if not df.index.is_unique:
            self.log.info(f"Dataset {short_name} is not subjectkey only")
            df.reset_index(inplace=True)
            if 'eventname' in df.columns:
                df.set_index(['subjectkey', 'eventname'], inplace=True)
            elif 'visit' in df.columns:
                df.set_index(['subjectkey', 'visit'], inplace=True)
            else:
                df.set_index('subjectkey', inplace=True)

            if not df.index.is_unique:
                self.log.warn(f"Dataset {short_name} still doesn't have a unique index")
        else:
            self.log.info(f"Dataset {short_name} is indexed by subjectkey")
        return df
