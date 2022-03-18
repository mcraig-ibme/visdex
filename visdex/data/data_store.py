"""
Base class for data store
"""
import logging
import os
import tempfile
import shutil

import pandas as pd

class DataStore:

    def __init__(self, backend):
        """
        """
        self.log = logging.getLogger(__name__)
        self.backend = backend

    def get_main(self, keep_index_cols=False):
        raise NotImplementedError()

    def get_columns(self):
        raise NotImplementedError()

    def get_filtered(self, keep_index_cols=False):
        raise NotImplementedError()

    def store(self, name, df):
        self.backend.store(name, df)
    
    def load(self, name, keep_index_cols=False):
        return self.backend.load(name, keep_index_cols)
