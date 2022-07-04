"""
visdex: Configuration handling
"""
import logging
import os
from datetime import timedelta

import visdex.data_stores as data_stores

LOG = logging.getLogger(__name__)

def init(flask_app):
    # Identify the configuration file
    if "VISDEX_CONFIG" in os.environ:
        config_file = os.environ["VISDEX_CONFIG"]
    elif "HOME" in os.environ:
        config_file = os.path.join(os.environ["HOME"], ".visdex.conf")
    else:
        config_file = "/etc/visdex/visdex.conf"

    if os.path.isfile(config_file):
        LOG.info(f"Using config file: {config_file}")
        flask_app.config.from_pyfile(config_file)
    else:
        LOG.info(f"Config file: {config_file} not found")
