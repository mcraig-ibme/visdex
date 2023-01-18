import datetime
import logging
import os
import uuid

import flask

from .session import Session

TIMEOUT_LAG_MINUTES = 5
TIMEOUT_MINUTES = 5
CACHE_DIR = None

MAIN_DATA = "df"
FILTERED = "filtered"

LOG = logging.getLogger(__name__)

def init(flask_app):
    """
    Set session configuration - timeout, cache dir etc

    Called once on server initialization
    """
    global TIMEOUT_MINUTES, CACHE_DIR
    TIMEOUT_MINUTES = flask_app.config.get("TIMEOUT_MINUTES", 5)
    LOG.info(f"Session timeout {TIMEOUT_MINUTES} minutes")

    CACHE_DIR = flask_app.config.get("DATA_CACHE_DIR", None)
    if not CACHE_DIR:
        raise RuntimeError("No session cache dir defined")
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    elif not os.path.isdir(CACHE_DIR):
        raise RuntimeError("Session cache dir already exists and is not a directory")
    LOG.info(f"Session cache dir: {CACHE_DIR}")

    @flask_app.before_request
    def set_session_timeout():
        flask.session.permanent = True
        flask_app.permanent_session_lifetime = datetime.timedelta(minutes=TIMEOUT_MINUTES)

def create():
    """
    Create a new session
    """
    session_id = uuid.uuid4()
    flask.session['uid'] = session_id
    LOG.info(f"Created session with UID {session_id.hex} {flask.session['uid']}.hex")

def get():
    """
    Get the current session

    FIXME we expect the uid will always be there but it isn't - why?
    """
    global CACHE_DIR
    if "uid" not in flask.session:
        LOG.warn(f"Session UID not found")
        create()
    uid = flask.session["uid"].hex
    sess = Session(CACHE_DIR, uid)
    sess.touch()
    expire_old_sessions()
    return sess

def expire_old_sessions():
    """
    Flag current session's last used time to now and remove sessions that have timed out

    We need this because although Flask can expire sessions itself, all that means is
    that the session data is removed, i.e. we will no longer get a session object with
    the expired uid, and we can't remove the associated data store without that uid.

    We remove data stores 5 minutes after the corresponding session timeout.
    """
    global TIMEOUT_MINUTES, TIMEOUT_LAG_MINUTES, CACHE_DIR
    timeout_minutes = TIMEOUT_MINUTES + TIMEOUT_LAG_MINUTES
    now = datetime.datetime.now()
    for uid in os.listdir(CACHE_DIR):
        sess = Session(CACHE_DIR, uid)
        dt = (now - sess.last_used()).seconds//60
        LOG.info(f"Session data store {uid} last used {dt} mins ago")
        if dt > timeout_minutes:
            LOG.info(f"Cleaning up expired session data store: {uid}")
            sess.free()
