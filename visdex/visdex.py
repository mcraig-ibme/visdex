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
            visdex.data.DataSelection(app),

            # Export component
            #visdex.export.get_layout(app),

            # Summary section
            visdex.summary.Summary(app),

            # Exploratory graphs
            visdex.exploratory_graphs.ExploratoryGraphs(app),
        ]
    )
