"""
visdex: Main page header with UoN logo, version etc
"""
import logging

import dash_html_components as html

from ._version import __version__

LOG = logging.getLogger(__name__)
HEADER_IMAGE = "/assets/UoN_Primary_Logo_RGB.png"

def get_layout(app):
    return html.Div(children=[
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
                html.A(
                    id="uni-link",
                    children=[
                        html.H1("UoN"),
                    ],
                    href="https://www.nottingham.ac.uk/",
                    style={
                        "display": "inline-block",
                        "float": "right",
                        "margin-right": "10px",
                        "text-decoration": "none",
                        "color": "#FFFFFF",
                    },
                ),
            ],
            style={"width": "100%", "display": "inline-block", "color": "white"},
        ),
    ])
