"""
visdex: Dashboard data explorer for CSV trial data

This module defines the layout of the data explorer application
"""
import logging

from dash import html

import visdex.data
import visdex.summary
import visdex.export
import visdex.exploratory_graphs

LOG = logging.getLogger(__name__)

def get_layout(app):
    return html.Div(
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
