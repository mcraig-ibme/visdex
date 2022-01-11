"""
visdex: Dashboard data explorer for CSV trial data

This module defines the overalllayout of 
the main application page.

Originally written for ABCD data
"""
import logging

import dash
import dash_bootstrap_components as dbc
import dash_html_components as html

import visdex.data
import visdex.summary
import visdex.export
import visdex.header
import visdex.exploratory_graphs

LOG = logging.getLogger(__name__)

# Create the Dash application
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div(
    children=[
        # Page header
        visdex.header.get_layout(app),
        
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

