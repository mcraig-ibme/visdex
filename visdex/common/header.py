"""
visdex: Page header with UoN logo, version etc
"""
import logging

from flask import session
from flask_login import current_user

from dash import html, dcc
from dash.dependencies import Input, Output

from visdex._version import __version__
from visdex.common import Component

LOG = logging.getLogger(__name__)
HEADER_IMAGE = "UoN_Primary_Logo_RGB.png"

class Header(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
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
                        children=[html.H2("DRS | ", className="inline")],
                        href="https://digitalresearch.nottingham.ac.uk/",
                    ),
                    html.H2("Visual Data Explorer v%s" % __version__, className="inline"),
                    html.Div(id="user-status-label")
                ],
            ),
        ])

        self.register_cb(app, "login_status", 
            Output('user-status-label', 'children'), 
            Output('login-status', 'data'), 
            [Input('url', 'pathname')]
        )

    def login_status(self, url):
        """
        Display login/logout link at the top of the page
        """
        if getattr(current_user, 'is_authenticated', False):
            user_id = current_user.get_id()
            LOG.info(current_user)
            LOG.info(session)
            return [f"User: {user_id} ", html.A('logout', href=f'{self.config.get("PREFIX", "/")}logout', className="inline")], user_id
        else:
            return [], ''
