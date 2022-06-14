"""
visdex: Summary section

The summary section defines basic data visualisations that are
available by default
"""
from visdex.common import Component
from . import (
    data_preview,
    filter,
    download,
    heatmap,
    manhattan,
    kde,
)

class Summary(Component):
    def __init__(self, app):
        Component.__init__(self, id_prefix="summary-", children=[
            data_preview.RawPreview(app),
            filter.DataFilter(app),
            download.Download(app),
            heatmap.SummaryHeatmap(app),
            #manhattan.get_layout(app),
            kde.SummaryKdes(app),
        ])
