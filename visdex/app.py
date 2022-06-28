"""
visdex: Dashboard data explorer for CSV trial data

Originally written for ABCD data

This module defines the basic Dash application that serves the
appropriate page in response to requests
"""
import logging
import os
from datetime import timedelta

import flask
from flask_login import logout_user, current_user

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import html

import visdex.common.header as header
import visdex.visdex as visdex
import visdex.login as login
import visdex.data_stores as data_stores

LOG = logging.getLogger(__name__)

# Create the Flask application
flask_app = flask.Flask(__name__)
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

# It should be possible to handle URL prefixes transparently but
# have not managed to get this to work yet - so set this to whatever
# prefix Apache is using
prefix = flask_app.config.get("PREFIX", "/")
LOG.info(f"Using prefix: {prefix}")

timeout_minutes = flask_app.config.get("TIMOUT_MINUTES", 1)
LOG.info(f"Session timeout {timeout_minutes} minutes")

session_dir = flask_app.config.get("DATA_CACHE_DIR", None)
LOG.info(f"Session cache dir: {session_dir}")

data_stores.load_config(flask_app.config.get("DATA_STORE_CONFIG", None))

@flask_app.before_request
def set_session_timeout():
    flask.session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=5)

login.init_login(flask_app)

# Create the Dash application
app = dash.Dash(
    __name__,
    server=flask_app,
    requests_pathname_prefix=prefix,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Main layout - all pages have header
app.layout = html.Div([
    header.get_layout(app),
    html.Div(id='page-content'),
])

visdex_layout = visdex.get_layout(app)
login_layout =  login.get_layout(app),

@app.callback(
    Output('page-content', 'children'), 
    Output('redirect', 'pathname'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    """
    Display the appropriate page based on URL requested and login status
    """
    view = None
    url = dash.no_update
    LOG.info(f"Request: {pathname}")
    if pathname == f'{prefix}login':
        view = login_layout
    elif pathname == f'{prefix}logout':
        if "uid" in flask.session:
            #data_store.remove() FIXME free server side session
            pass
        if current_user.is_authenticated:
            logout_user()
        url = f'{prefix}login'
    elif pathname == f'{prefix}app':
        if current_user.is_authenticated and "uid" in flask.session:
            view = visdex_layout
        else:
            # Not authenticated - redirect to login page
            url = f'{prefix}login'
    else:
        # Redirect any other page to the main app
        url = f'{prefix}app'

    if url != dash.no_update:
        LOG.info(f"Redirect: {url}")
    return view, url
