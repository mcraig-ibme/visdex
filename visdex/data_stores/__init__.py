"""
visdex: Data stores

A data store is a source of data and is stateless so there is only one class per server instance.
Note that in multiprocess servers (e.g. apache with mod_wsgi) there may be multiple instances
of each, this is pretty harmless.
"""
import json
import logging
import os
import traceback

from .nda import NdaData
from .user import UserData

LOG = logging.getLogger(__name__)

DATA_STORES = {
    "user" : {
        "label" : "Upload a CSV/TSV data set",
        "class" : "UserData",
    }
}

def init(flask_app):
    """
    Load data store configuration from a .json file
    
    Format is:
    
    id : {config}
    
    config should include a human readable label and a class
    name as the minimum
    """
    global DATA_STORES
    LOG.info(f"Loading data store configuration")
    fname = flask_app.config.get("DATA_STORE_CONFIG", None)
    try:
        if os.path.isfile(fname):
            LOG.info(f"Config file: {fname}")
            with open(fname, "r") as f:
                config = json.load(f)
            LOG.debug(config)
            DATA_STORES.update(config)
        elif fname is not None:
            LOG.warn(f"Failed to load data store config from {fname} - no such file")
        for id in list(DATA_STORES.keys()):
            impl = _get_impl(id)
            if impl is not None:
                DATA_STORES[id]["impl"] = impl
                LOG.info("Created data store: %s: %s" % (id, DATA_STORES[id]))
            else:
                LOG.warn("Failed to create data store: %s: %s" % (id, DATA_STORES[id]))
                del DATA_STORES[id]
    except Exception as exc:
        LOG.warn(exc)

def _get_impl(id):
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
        return cls(**ds_conf)
    except:
        traceback.print_exc()
        return None

__all__= ["DATA_STORES", "load_config"]
