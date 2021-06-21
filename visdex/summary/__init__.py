"""
visdex: Summary section

The summary section defines basic data visualisations that are
available by default
"""

import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from . import (
    preview_table,
    summary_table,
    summary_heatmap,
    summary_manhattan,
    summary_kde,
)

from visdex.common import standard_margin_left

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
    def toggle_collapse_summary(n_clicks, is_open):
        """
        Handle click on the 'Summary' expand/collapse button
        """
        logging.info(f"toggle_collapse_summary {n_clicks} {is_open}")
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
                preview_table.get_layout(app),
                summary_table.get_layout(app),
                summary_heatmap.get_layout(app),
                summary_manhattan.get_layout(app),
                summary_kde.get_layout(app),

             ],
             is_open=True,
        ),
    ])
