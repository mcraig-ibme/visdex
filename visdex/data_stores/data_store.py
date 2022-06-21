"""
visdex: Base class for data stores
"""
import logging

class DataStore:
    """
    A data store is a stateless object which provides access to a particular type of data
    """

    def __init__(self):
        self.log = logging.getLogger(__name__)
