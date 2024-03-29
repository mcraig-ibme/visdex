"""
visdex: Manages access to NDA data sets
"""
import os

import pandas as pd
from flask_login import current_user

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
    def __init__(self, flask_app, global_datadir, study_name, subject_table="ndar_subject01", subject_field="subjectkey", **kwargs):
        DataStore.__init__(self, flask_app, **kwargs)
        self._global_datadir = global_datadir
        self._study_name = study_name
        self._dictdir = os.path.join(global_datadir, "dictionary")
        self._datadir = os.path.join(global_datadir, study_name)
        self._subject_table = subject_table
        self._subject_field = subject_field
        self.datasets = self._get_known_datasets()
        self.imaging_types = self._get_known_imaging_types()

        self.log.info("NDA data source for study %s", self._study_name)
        self.log.info("Found %i data sets", len(self.datasets))
        #self.log.info(",".join(self.datasets['name'])
        self.log.info("Found %i imaging data types", len(self.imaging_types))
        #self.log.info(self.imaging_types)

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
        if not dataset:
            dataset = self._subject_table
            fields = [self._subject_field]

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
                if fields_in_dataset == [self._subject_field]:
                    main_df.reset_index(drop=False, inplace=True)
                main_df = main_df[fields_in_dataset]
                if fields_in_dataset == [self._subject_field]:
                    main_df.set_index(fields_in_dataset, inplace=True)
 
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
        ids = ids.set_index(self._subject_field).index
        image_ids = images.set_index(self._subject_field).index
        images = images[image_ids.isin(ids)]

        if self._study_name == "abcd":
            return images[[self._subject_field, "visit", "image_file"]]
        else:
            # Different for HCP! We get S3 links from the datastructure_manifest
            # table. Also we have images in fmriresults01 for processed data
            # and maybe imagingcollection01
            manifest_dfs = []

            image_manifests = images[[self._subject_field, "manifest"]]
            image_manifests.set_index("manifest")
            manifest_dfs.append(image_manifests)

            for table, img_types_col, manifest_col in [
                ("fmriresults01", "job_name", "manifest"),
                ("imagingcollection01", "image_collection_name", "image_manifest"),
            ]:
                try:
                    df = self._load_dataset(table)
                    df.reset_index(drop=False, inplace=True)
                    image_types = df.set_index(img_types_col).index
                    df_images = df[image_types.isin(selected_types)]
                    df_ids = df_images.set_index(self._subject_field).index
                    df_images = df_images[df_ids.isin(ids)]
                    df_manifests = df_images[[self._subject_field, manifest_col]]
                    df_manifests.rename(columns={manifest_col : "manifest"}, inplace=True)
                    df_manifests.set_index("manifest")
                    manifest_dfs.append(df_manifests)
                except:
                    self.log.exception(f"Failed to get manifests from {table}")
            
            manifests = pd.concat(manifest_dfs)
            manifest_data = self._load_dataset("datastructure_manifest")[["manifest_name", "associated_file"]]
            manifest_data.rename(columns={"manifest_name" : "manifest"}, inplace=True)
            manifest_data.set_index('manifest')
            manifest_data = manifest_data.merge(manifests, on="manifest", how="inner")
            return manifest_data[[self._subject_field, "associated_file"]]

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

        try:
            cache_file = os.path.join(self._datadir, "_visdex_imaging_types.csv")
            return pd.read_csv(cache_file)
        except IOError:
            self.log.info("Imaging types cache file not found - recreating")
            imaging_types = []
            try:
                df = self._load_dataset("image03")
                df.reset_index(drop=True, inplace=True)
                df = df[IMG_FIELDS]
                df.fillna(0, inplace=True)
                for f, dps in zip(IMG_ROUND_FIELDS, IMG_ROUND_DPS):
                    df[f] = df[f].round(dps)
                df.drop_duplicates(inplace=True)
                df['text'] = df.apply(format, axis=1)
                imaging_types.append(df[['image_description', 'text']])
            except:
                self.log.exception(f"Error getting imaging types from image03")

            for table, img_types_col, manifest_col in [
                ("fmriresults01", "job_name", "manifest"),
                ("imagingcollection01", "image_collection_name", "manifest_name"),
            ]:
                try:
                    self.log.info(f"Loading imaging results from {table}")
                    df = self._load_dataset(table)
                    df.reset_index(drop=True, inplace=True)
                    df = df[[img_types_col]]
                    df.drop_duplicates(inplace=True)
                    df['image_description'] = df[img_types_col]
                    df['text'] = df[img_types_col]
                    imaging_types.append(df[['image_description', 'text']])
                except:
                    self.log.exception(f"Error getting imaging types from {table}")

            imaging_types = pd.concat(imaging_types)
            imaging_types.to_csv(cache_file)
            return imaging_types
        except:
            self.log.exception(f"Error getting imaging types")

    def _load_dataset(self, short_name):
        """
        Retrieve a dataset as a data frame
        """
        data_fname = os.path.join(self._datadir, "%s.txt" % short_name)
        self.log.info(f"_get_dataset {short_name} {data_fname}")
        df = pd.read_csv(data_fname, sep="\t", quotechar='"', skiprows=[1], low_memory=False)
        if self._subject_field in df.columns:
            df.set_index(self._subject_field, inplace=True)
            if not df.index.is_unique:
                self.log.info(f"Dataset {short_name} is not {self._subject_field} only")
                df.reset_index(inplace=True)
                if 'eventname' in df.columns:
                    df.set_index([self._subject_field, 'eventname'], inplace=True)
                elif 'visit' in df.columns:
                    df.set_index([self._subject_field, 'visit'], inplace=True)
                else:
                    df.set_index(self._subject_field, inplace=True)

                if not df.index.is_unique:
                    self.log.warn(f"Dataset {short_name} still doesn't have a unique index")
            else:
                self.log.info(f"Dataset {short_name} is indexed by {self._subject_field}")
        else:
            self.log.warn(f"Dataset {short_name} does not contain '{self._subject_field}'")
        return df
