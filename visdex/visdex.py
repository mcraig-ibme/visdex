"""
visdex: Dashboard data explorer for CSV trial data

This module defines the layout of the data explorer application
"""
from visdex.common import Component
import visdex.data
import visdex.summary
import visdex.export
import visdex.exploratory_graphs

class VisdexPage(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="data-", children=[
            # Data selection
            visdex.data.DataSelection(app),

            # Export component
            #visdex.export.Export(app),

            # Summary section
            visdex.summary.Summary(app),

            # Exploratory graphs
            visdex.exploratory_graphs.ExploratoryGraphs(app),
        ]
    )
