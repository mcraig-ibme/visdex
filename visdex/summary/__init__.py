"""
visdex: Summary section

The summary section defines basic data tables and visualisations that are
available by default
"""
from visdex.common import Component
from . import (
    raw_preview,
    data_preview,
    filter,
    download,
    heatmap,
    manhattan,
    kde,
)

class Summary(Component):
    def __init__(self, app):
        Component.__init__(self, app, id_prefix="summary-", children=[
            raw_preview.RawPreview(app),
            filter.DataFilter(app),
            download.Download(app),
            heatmap.SummaryHeatmap(app),
            #manhattan.get_layout(app),
            kde.SummaryKdes(app),
        ])
