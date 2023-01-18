"""
visdex: Data stores

A data store is a source of data and is stateless so there is only one class per server instance.
Note that in multiprocess servers (e.g. apache with mod_wsgi) there may be multiple instances
of each, this is pretty harmless so long as data stores are not computationally expensive to create.
"""
import logging

from .nda import NdaData
from .user import UserData
from .generic_csv import GenericCsv

LOG = logging.getLogger(__name__)

DATA_STORES = {
}

def init(flask_app):
    """
    Load data store configuration from the configured .json file
    
    Format is:
    
    id : {config}
    
    config should include a human readable label and a class
    name as the minimum
    """
    global DATA_STORES
    LOG.info(f"Loading data store configuration")
    DATA_STORES = flask_app.config.get("DATA_STORES", {})
    if not isinstance(DATA_STORES, dict):
        raise ValueError("Invalid data store configuration - DATA_STORES must be a dictionary")

    for id in list(DATA_STORES.keys()):
        LOG.info("Creating data store: %s: %s" % (id, DATA_STORES[id]))
        impl = _get_impl(flask_app, id)
        if impl is not None:
            DATA_STORES[id]["impl"] = impl
        else:
            LOG.warn("Failed to create data store: %s: %s" % (id, DATA_STORES[id]))
            del DATA_STORES[id]

    LOG.info(f"Configured data stores: {list(DATA_STORES.keys())}")

def _get_impl(flask_app, id):
    """
    Get the implementation of a data store by ID
    """
    global DATA_STORES
    ds_conf = DATA_STORES.get(id, None)
    if ds_conf is None:
        raise RuntimeError(f"No such data store '{id}'")
    class_name = ds_conf.get("class", None)

    if class_name is None:
        raise RuntimeError(f"No class name for data store '{id}'")
    cls = globals().get(class_name, None)

    if cls is None:
        raise RuntimeError(f"Can't find class {class_name} for data store '{id}'")

    try:
        return cls(flask_app, **ds_conf)
    except:
        LOG.exception("Exception in constructor for data store")
        return None
