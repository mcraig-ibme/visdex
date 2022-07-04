import datetime
import logging
import os
import flask

from .session import Session

SESSION_TIMEOUT_LAG_MINUTES = 5

MAIN_DATA = "df"
FILTERED = "filtered"

LOG = logging.getLogger(__name__)

def init(flask_app):
    timeout_minutes = flask_app.config.get("TIMEOUT_MINUTES", 5)
    LOG.info(f"Session timeout {timeout_minutes} minutes")

    @flask_app.before_request
    def set_session_timeout():
        flask.session.permanent = True
        flask_app.permanent_session_lifetime = datetime.timedelta(minutes=timeout_minutes)

def get():
    """
    Get the current session
    """
    session_dir = flask.current_app.config.get("DATA_CACHE_DIR", None)
    if not session_dir:
        raise RuntimeError("No session cache dir defined")
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    elif not os.path.isdir(session_dir):
        raise RuntimeError("Session cache dir already exists and is not a directory")

    uid = flask.session["uid"].hex
    sess = Session(session_dir, uid)
    sess.touch()
    expire_old_sessions(session_dir)
    return sess

def expire_old_sessions(session_dir):
    """
    Flag current session's last used time to now and remove sessions that have timed out

    We need this because although Flask can expire sessions itself, all that means is
    that the session data is removed, i.e. we will no longer get a session object with
    the expired uid, and we can't remove the associated data store without that uid.

    We remove data stores 5 minutes after the corresponding session timeout.
    """
    timeout_minutes = flask.current_app.config.get("TIMOUT_MINUTES", 1) + SESSION_TIMEOUT_LAG_MINUTES
    now = datetime.datetime.now()
    for uid in os.listdir(session_dir):
        sess = Session(session_dir, uid)
        dt = (now - sess.last_used()).seconds//60
        LOG.info(f"Session data store {uid} last used {dt} mins ago")
        if dt > timeout_minutes:
            LOG.info(f"Cleaning up expired session data store: {uid}")
            sess.free()
