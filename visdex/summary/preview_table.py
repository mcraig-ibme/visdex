"""
visdex: preview table

Shows a basic summary of the first few rows in the data 
"""
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output

from visdex.common import Component, vstack
from visdex.data.cache import get_cache

class PreviewTable(Component):
    """
    Component that displays the first few lines of a data frame
    """

    def __init__(self, app, id_prefix="preview-", df_name="df", update_div_id="df-loaded-div"):
        """
        :param app: Dash application
        :param id_prefix: Prefix string for HTML component identifiers
        :param df_name: Cache name of data frame to preview
        :param update_div_id: ID of div that signals when to update
        """
        Component.__init__(self, app, id_prefix, children=[
            html.H3(children="Table Preview", style=vstack),
            dcc.Loading(
                id=id_prefix + "loading",
                children=[
                    html.Div(
                        id=id_prefix + "table",
                        style={
                            "width": "95%",
                            "margin-left": "10px",
                            "margin-right": "10px",
                        },
                    )
                ],
            ),
        ])
        self.df_name = df_name

        self.register_cb(app, "update",
            Output(self.id_prefix + "table", "children"),
            [Input(update_div_id, "children")],
            prevent_initial_call=True,
        )

    def update(self, df_loaded):
        self.log.debug("Update preview table")
        cache = get_cache()

        # We want to be able to see the index columns in the preview table
        dff = cache.load(self.df_name, keep_index_cols=True)

        if dff.size > 0:
            return html.Div(
                dash_table.DataTable(
                    id=self.id_prefix + "data-table",
                    columns=[{"name": i, "id": i} for i in dff.columns],
                    data=dff.head().to_dict("records"),
                    style_table={"overflowX": "auto"},
                ),
            )
        else:
            return html.Div(dash_table.DataTable(self.id_prefix + "data-table"))
