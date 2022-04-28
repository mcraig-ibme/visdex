"""
Base class for data store with functions for creating and
retrieving per-session instances
"""
import logging

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

MAIN_DATA = "df"
FILTERED = "filtered"

def remove():
    uid = session["uid"].hex
    if uid in DATA_STORES:
        LOG.info("Removing data store for session: %s" % uid)
        del DATA_STORES[uid]

class DataStore:
    """
    Stores and retrieves the data the user is working with

    The class has a storage backend that can store DataFrame objects.
    However it can define and return the main data however it wants.
    """

    def __init__(self, backend):
        self.log = logging.getLogger(__name__)
        self.backend = backend

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
