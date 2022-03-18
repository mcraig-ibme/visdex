"""
Creates a cache instance which stores and retieves data frames
for the application
"""
import logging

from flask import session

from .user_data import UserData
from .abcd_data import AbcdData

LOG = logging.getLogger(__name__)

# Default to use Feather files for data caching rather than redis
USE_REDIS = False
CACHES = {}

def remove_cache():
    uid = session["uid"].hex
    if uid in CACHES:
        LOG.info("Removing data cache for session: %s" % uid)
        del CACHES[uid]

# def use_std_data(name):
#     """
#     Set the data cache to use 'standard' predefined data, e.g. ABCD
#     """
#     uid = session["uid"].hex
#     if uid in CACHES:
#         LOG.info("Removing data cache for session: %s" % uid)
#         del CACHES[uid]
#     LOG.info("Creating standard data cache for %s data in session: %s" % (name, uid))
#     CACHES[uid] = FeatherCache()

def init_cache(std_data=None):
    uid = session["uid"].hex
    if uid in CACHES:
        # Is this useful?
        CACHES[uid].free()

    if USE_REDIS:
        raise NotImplementedError()
        #import os
        #from flask_caching import Cache
        #CACHE_CONFIG = {
        #    "CACHE_TYPE": "redis",
        #    "CACHE_REDIS_URL": os.environ.get("REDIS_URL", "redis://localhost:6379"),
        #}
        #cache = Cache()
        #cache.init_app(app.server, config=CACHE_CONFIG)
    else:
        if std_data is None:
            CACHES[uid] = UserData()
        elif std_data == "abcd":
            CACHES[uid] = AbcdData()
        else:
            raise RuntimeError(f"Unknown standard data type: '{std_data}'")

def get_cache():
    uid = session["uid"].hex
    if uid not in CACHES:
        LOG.error("No data cache for session: %s" % uid)
    return CACHES[uid]
