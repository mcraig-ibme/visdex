"""
visdex: Main page header with UoN logo, version etc
"""
import logging

from flask_login import current_user

from dash import html, dcc
from dash.dependencies import Input, Output

from ._version import __version__
from .common import hstack

LOG = logging.getLogger(__name__)
HEADER_IMAGE = "UoN_Primary_Logo_RGB.png"

def get_layout(app):
        
    @app.callback(
        Output('user-status-label', 'children'), 
        Output('login-status', 'data'), 
        [Input('url', 'pathname')]
    )
    def login_status(url):
        """
        Display login/logout link at the top of the page
        """
        if getattr(current_user, 'is_authenticated', False):
            user_id = current_user.get_id()
            LOG.info(user_id)
            return [html.Label(f"User: {user_id}", style=hstack), html.A('logout', href=f'{app.config.get("PREFIX", "/")}logout', style=hstack)], user_id
        else:
            return [], ''

    return html.Div(children=[
        dcc.Location(id='url', refresh=False),
        dcc.Location(id='redirect', refresh=True),
        dcc.Store(id='login-status', storage_type='session'),
        html.A(
            children=[html.Img(src=app.get_asset_url(HEADER_IMAGE), height=100)],
            href="https://www.nottingham.ac.uk/",
        ),
        html.Div(
            id="app-header",
            children=[
                html.A(
                    id="drs-link",
                    children=[
                        html.H1("DRS |"),
                    ],
                    href="https://digitalresearch.nottingham.ac.uk/",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "margin-right": "10px",
                        "text-decoration": "none",
                        "align": "center",
                    },
                ),
                html.H2(
                    "Visual Data Explorer v%s" % __version__,
                    style={
                        "display": "inline-block",
                    },
                ),
                html.Div(
                    id="user-status-label", 
                    style={
                         "display": "inline-block",
                         "position" : "absolute",
                         "top" : "0",
                         "right": "0",
                         "margin-right": "10px",
                         "color": "#505050",
                     },
                )
            ],
            style={"width": "100%", "display": "inline-block", "color": "white"},
        ),
    ])
