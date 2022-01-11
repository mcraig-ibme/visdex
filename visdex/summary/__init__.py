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
from . import (
    preview_table,
    summary_stats,
    heatmap,
    manhattan,
    kde,
)

LOG = logging.getLogger(__name__)

def get_layout(app):
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

    return html.Div(children=[
        html.Div(
            [
                html.H1(
                    "Summary",
                    style={
                        "display": "inline-block",
                        "margin-left": standard_margin_left,
                    },
                ),
                dbc.Button(
                    "-",
                    id="collapse-summary-button",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "width": "40px",
                    },
                ),
            ],
        ),
        dbc.Collapse(
            id="summary-collapse",
            children=[
                preview_table.PreviewTable(app),
                summary_stats.get_layout(app),
                heatmap.get_layout(app),
                manhattan.get_layout(app),
                kde.get_layout(app),
             ],
             is_open=True,
        ),
    ])
