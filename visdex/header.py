"""
visdex: Main page header with UoN logo, version etc
"""
import logging

from flask_login import current_user

from dash import html, dcc
from dash.dependencies import Input, Output

from ._version import __version__

LOG = logging.getLogger(__name__)
HEADER_IMAGE = "/assets/UoN_Primary_Logo_RGB.png"

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
        if getattr(current_user, 'is_authenticated', False) and url != '/logout':
            user_id = current_user.get_id()
            return [dcc.Link(user_id, href="/profile"), dcc.Link('logout', href='/logout', style={"margin-left": "15px"})], user_id
        else:
            return [], 'logged out'

    return html.Div(children=[
        dcc.Location(id='url', refresh=False),
        dcc.Location(id='redirect', refresh=True),
        dcc.Store(id='login-status', storage_type='session'),
        html.Img(src=HEADER_IMAGE, height=100),
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
                         "text-decoration": "none",
                         "color": "#FFFFFF",
                     },
                )
                # html.A(
                #     id="uni-link",
                #     children=[
                #         html.H1("UoN"),
                #     ],
                #     href="https://www.nottingham.ac.uk/",
                #     style={
                #         "display": "inline-block",
                #         "float": "right",
                #         "margin-right": "10px",
                #         "text-decoration": "none",
                #         "color": "#FFFFFF",
                #     },
                # ),
            ],
            style={"width": "100%", "display": "inline-block", "color": "white"},
        ),
    ])
