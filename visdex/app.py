"""
visdex: Dashboard data explorer for CSV trial data

This module defines the overalllayout of 
the main application page.

Originally written for ABCD data
"""
import logging

import flask
from flask import abort
from flask_login import login_user, logout_user, current_user

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import html, dcc

import visdex.data
import visdex.summary
import visdex.export
import visdex.header
import visdex.exploratory_graphs
import visdex.login

LOG = logging.getLogger(__name__)

flask_app = flask.Flask(__name__)
flask_app.secret_key = 'visdexsecret'

visdex.login.init_login(flask_app)

# Create the Dash application
app = dash.Dash(
    __name__,
    server=flask_app,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div([
    visdex.header.get_layout(app),
    html.Div(id='page-content'),
])

visdex_layout = html.Div(
    children=[
        # Data selection
        visdex.data.get_layout(app),

        # Export component
        #visdex.export.get_layout(app),

        # Summary section
        visdex.summary.get_layout(app),

        # Exploratory graphs
        visdex.exploratory_graphs.get_layout(app),
    ]
)

login_layout =  visdex.login.get_layout(app),

@app.callback(
    Output('page-content', 'children'), 
    Output('redirect', 'pathname'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    """
    Display the appropriate page based on login/logout status
    """
    view = None
    url = dash.no_update
    if pathname == '/login':
        view = login_layout
    elif pathname == '/visdex':
        if current_user.is_authenticated:
            view = visdex_layout
        else:
            # Not authenticated - redirect to login page
            view = login_layout
            url = '/login'
    elif pathname == '/logout':
        if current_user.is_authenticated:
            logout_user()
        view = login_layout
        url = '/login'
    else:
        # Just redirect other pages to the login screen
        view = login_layout
        url = '/login'

    return view, url
