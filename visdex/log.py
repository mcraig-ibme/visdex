"""
visdex: Set up logging
"""
import logging

LOG = logging.getLogger(__name__)

def init(flask_app):
    """
    Called once on server startup
    """
    global KNOWN_USERS
    logfile = flask_app.config.get("LOG_FILE", None)
    level = flask_app.config.get("LOG_LEVEL", "INFO")
    if logfile:
        logging.basicConfig(
            level=getattr(logging, level), 
            filename=logfile
        )
        
        LOG.info(f"Starting logfile")
