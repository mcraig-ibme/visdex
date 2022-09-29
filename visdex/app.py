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

# Create the Flask application and load configuration file
flask_app = flask.Flask(__name__)
config.init(flask_app)
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
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Main layout - all pages have header
dash_app.layout = html.Div([
    header.Header(dash_app),
    html.Div(id='page-content'),
])

visdex_page = visdex.VisdexPage(dash_app)
flask_app.config["USING_AUTH"] = flask_app.config.get("AUTH", {}).get("type", "none") != "none"
if not flask_app.config["USING_AUTH"]:
    LOG.warn(f"Not using authorization - any user can access")
    login_page = None
else:
    login_page =  login.LoginPage(dash_app)

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
    using_auth = flask_app.config["USING_AUTH"]
    LOG.info(f"Request: {pathname} {flask.session}")
    if using_auth:
        if pathname == f'{prefix}login':
            view = login_page
        elif pathname == f'{prefix}logout':
            if current_user.is_authenticated:
                logout_user()
            url = f'{prefix}login'
        elif pathname == f'{prefix}app':
            if current_user.is_authenticated and "uid" in flask.session:
                view = visdex_page
            else:
                # Not authenticated - redirect to login page
                url = f'{prefix}login'
        else:
            url = f'{prefix}app'
    else:
        if pathname != f'{prefix}app':
            url = f'{prefix}app'
        else:
            view = visdex_page
            if "uid" not in flask.session:
                session.create()
                LOG.info(f"Session: {flask.session['uid'].hex}")
                url = f'{prefix}app'

    if url != dash.no_update:
        LOG.info(f"Redirect: {url}")
    return view, url
