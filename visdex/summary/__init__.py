"""
visdex: Summary section

The summary section defines basic data visualisations that are
available by default
"""
from dash import html

from . import (
    data_preview,
    filter,
    download,
    heatmap,
    manhattan,
    kde,
)

def get_layout(app):
    return html.Div(children=[
        data_preview.RawPreview(app),
        filter.DataFilter(app),
        download.Download(app),
        heatmap.SummaryHeatmap(app),
        #manhattan.get_layout(app),
        kde.SummaryKdes(app),
        html.Div(id="filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),
    ])
