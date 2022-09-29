"""
visdex: Base class for data stores
"""
import logging

from flask_login import current_user

class DataStore:
    """
    A data store is a stateless object which provides access to a particular type of data
    """

    def __init__(self, flask_app, **kwargs):
        self.log = logging.getLogger(__name__)
        self.users = kwargs.get("users", None)
        self.config = flask_app.config

    def check_user(self):
        if not self.config["USING_AUTH"]:
            return True
        else:
            self.log.info("Checking user %s vs %s" % (current_user.get_id(), self.users))
            return self.users is None or (current_user.is_authenticated and current_user.get_id() in self.users)
