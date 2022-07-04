"""
visdex: Dashboard data explorer for CSV trial data

Originally written for ABCD data

This module defines the basic Dash application that serves the
appropriate page in response to requests
"""
import logging

import flask
from flask_login import logout_user, current_user

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import html

import visdex.common.header as header
import visdex.visdex as visdex
import visdex.login as login
import visdex.config as config
import visdex.session as session
import visdex.data_stores as data_stores

LOG = logging.getLogger(__name__)

# Create the Flask application
flask_app = flask.Flask(__name__)
config.init(flask_app)
login.init(flask_app)
session.init(flask_app)
data_stores.init(flask_app)

# It should be possible to handle URL prefixes transparently but
# have not managed to get this to work yet - so set this to whatever
# prefix Apache is using
prefix = flask_app.config.get("PREFIX", "/")
LOG.info(f"Using prefix: {prefix}")

# Create the Dash application
dash_app = dash.Dash(
    __name__,
    server=flask_app,
    requests_pathname_prefix=prefix,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Main layout - all pages have header
dash_app.layout = html.Div([
    header.get_layout(dash_app),
    html.Div(id='page-content'),
])

visdex_layout = visdex.get_layout(dash_app)
login_layout =  login.get_layout(dash_app),

@dash_app.callback(
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
