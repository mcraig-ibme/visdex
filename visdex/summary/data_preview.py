"""
visdex: data preview table

Shows a basic summary of the first few rows in a data frame
"""
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output

from visdex.common import Component
import visdex.session

class DataPreview(Component):
    """
    Component that displays the first few lines of a data frame
    """

    def __init__(self, app, title, id_prefix="preview-", update_div_id="df-loaded-div", data_id=visdex.session.MAIN_DATA):
        """
        :param app: Dash application
        :param id_prefix: Prefix string for HTML component identifiers
        :param update_div_id: ID of div that signals when to update
        """
        self.data_id = data_id
        self.title = title

        Component.__init__(self, app, id_prefix, children=[
            html.H3(id=id_prefix+"title"),
            dcc.Loading(
                id=id_prefix + "loading",
                children=[
                    html.Div(id=id_prefix + "table")
                ],
            ),
        ])

        self.register_cb(app, "update",
            [
                Output(self.id_prefix + "title", "children"),
                Output(self.id_prefix + "table", "children"),
            ],
            [
                Input(update_div_id, "children")
            ],
            prevent_initial_call=True,
        )

    def update(self, df_loaded):
        self.log.info("Update preview table")
        ds = visdex.session.get()

        # We want to be able to see the index columns in the preview table
        dff = ds.load(self.data_id, keep_index_cols=True)

        if dff.size > 0:
            return self.title, html.Div(children=[
                dash_table.DataTable(
                    id=self.id_prefix + "data-table",
                    columns=[{"name": i, "id": i} for i in dff.columns],
                    data=dff.head().to_dict("records"),
                    #className="data-table",
                ),
                html.Div([
                    html.Div("Rows: " + str(dff.shape[0])),
                    html.Div("Columns: " + str(dff.shape[1])),
                ])
            ])
        else:
            return "", html.Div(dash_table.DataTable(self.id_prefix + "data-table"))
