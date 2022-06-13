"""
visdex: Summary section

The summary section defines basic data visualisations that are
available by default
"""
import logging

from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from visdex.common import standard_margin_left
from visdex.data import data_store
from . import (
    data_preview,
    filter,
    download,
    heatmap,
    manhattan,
    kde,
)

LOG = logging.getLogger(__name__)

def get_layout(app):
    layout = html.Div(children=[
        html.Div(
            [
                dbc.Button(
                    "-",
                    id="collapse-summary-button",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "width": "40px",
                        "vertical-align" : "middle",
                    },
                ),
                html.H2(
                    "Summary",
                    style={
                        "display": "inline-block",
                        "margin-left": standard_margin_left,
                        "vertical-align" : "middle",
                        "margin-bottom" : "0",
                        "padding" : "0",
                    },
                ),
            ],
        ),
        dbc.Collapse(
            id="summary-collapse",
            children=[
                data_preview.DataPreview(app, "Data preview (unfiltered)"),
                filter.RowFilter(app),
                data_preview.DataPreview(app, "Data preview (filtered)", id_prefix="preview-filtered", update_div_id="filtered-loaded-div", data_id=data_store.FILTERED),
                download.Download(app),
             ],
             is_open=True,
        ),
        heatmap.SummaryHeatmap(app),
        #manhattan.get_layout(app),
        kde.SummaryKdes(app),
        html.Div(id="filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),
    ])

    @app.callback(
        [
            Output("summary-collapse", "is_open"),
            Output("collapse-summary-button", "children"),
        ],
        [Input("collapse-summary-button", "n_clicks")],
        [State("summary-collapse", "is_open")],
        prevent_initial_call=True,
    )
    def _toggle_collapse_summary(n_clicks, is_open):
        """
        Handle click on the 'Summary' expand/collapse button
        """
        LOG.info(f"toggle_collapse_summary {n_clicks} {is_open}")
        if n_clicks:
            return not is_open, "+" if is_open else "-"
        return is_open, "-"

    return layout
