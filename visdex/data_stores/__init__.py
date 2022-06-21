import json
import logging
import os

from .nda import NdaData
from .user import UserData

LOG = logging.getLogger(__name__)

DATA_STORES = {
    "user" : {
        "label" : "Upload a CSV/TSV data set",
        "class" : "UserData",
    }
}

def load_config(fname):
    global DATA_STORES
    LOG.info(f"start data store configuration: {DATA_STORES}")
    LOG.info("Load_config: %s", fname)
    try:
        if os.path.isfile(fname):
            with open(fname, "r") as f:
                config = json.load(f)
            LOG.info(config)
            DATA_STORES.update(config)
        else:
            LOG.warn(f"Failed to load data store config from {fname} - no such file")
        for id in list(DATA_STORES.keys()):
            DATA_STORES[id]["impl"] = _get(id)

    except Exception as exc:
        LOG.warn(exc)
    LOG.info(f"end data store configuration: {DATA_STORES}")

def _get(id):
    global DATA_STORES
    LOG.info(f"data store configuration: {DATA_STORES}")
    ds_conf = DATA_STORES.get(id, None)
    if ds_conf is None:
        raise RuntimeError(f"No such data store '{id}'")
    class_name = ds_conf.get("class", None)
    if class_name is None:
        raise RuntimeError(f"No class name for data store '{id}'")
    cls = globals().get(class_name, None)
    if cls is None:
        raise RuntimeError(f"Can't find class {class_name} for data store '{id}'")
    return cls(**ds_conf)

__all__= ["DATA_STORES", "load_config", "get"]
