"""
visdex: preview table

Shows a basic summary of the first few rows in the data 
"""
import logging

import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

from visdex.common import div_style, TABLE_WIDTH
from visdex.cache import cache

LOG = logging.getLogger(__name__)

def get_layout(app):
    
    @app.callback(
        Output("table_preview", "children"),
        [Input("df-loaded-div", "children")],
        prevent_initial_call=True,
    )
    def _update_preview_table(df_loaded):
        LOG.info(f"update_preview_table")

        # We want to be able to see the index columns in the preview table
        dff = cache.load("df", keep_index_cols=True)

        if dff.size > 0:
            return html.Div(
                dash_table.DataTable(
                    id="table",
                    columns=[{"name": i, "id": i} for i in dff.columns],
                    data=dff.head().to_dict("records"),
                    style_table={"overflowX": "auto"},
                ),
            )

        return html.Div(dash_table.DataTable(id="table"))

    layout = html.Div(children=[
            html.H2(children="Table Preview", style=div_style),
            dcc.Loading(
                id="loading-table-preview",
                children=[
                    html.Div(
                        id="table_preview",
                        style={
                            "width": TABLE_WIDTH,
                            "margin-left": "10px",
                            "margin-right": "10px",
                        },
                    )
                ],
            ),
        ])

    return layout