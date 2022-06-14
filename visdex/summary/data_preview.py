"""
visdex: data preview table

Shows a basic summary of the first few rows in the data 
"""
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output

from visdex.common import Component, Collapsible, vstack
from visdex.data import data_store

class RawPreview(Collapsible):
    def __init__(self, app, id_prefix="rawpreview-"):
        """
        :param app: Dash application
        """
        Collapsible.__init__(self, app, id_prefix, title="Raw data preview", children=[
            html.H3(children="Column Summary", style=vstack),
            dcc.Loading(
                id=id_prefix+"loading-fields",
                children=[
                    html.Div(
                        id=id_prefix+"fields",
                        style={
                            "width": "95%",
                            "margin-left": "10px",
                            "margin-right": "10px",
                        },
                    ),
                ],
            ),
            DataPreview(app, "Data preview (unfiltered)"),
        ], is_open=False)

        self.register_cb(app, "expand_on_data_load",
            Output(id_prefix+"force-collapse", "children"),
            Input("df-loaded-div", "children"),
            prevent_initial_call=True,
        )

        self.register_cb(app, "update_field_summary",
            Output(id_prefix+"fields", "children"),
            Input("df-loaded-div", "children"),
            Input("filter-missing-values-input", "value"),
            prevent_initial_call=True,
        )

    def expand_on_data_load(self, df_loaded):
        return not df_loaded

    def update_field_summary(self, df_loaded, missing_value_cutoff):
        self.log.info(f"update_field_summary")
        ds = data_store.get()

        df = ds.load(data_store.MAIN_DATA)
        if df.empty:
            return [], False

        description_df = ds.description
        specifiers = ["s", "d", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
        fields_table_layout = html.Div(
            dash_table.DataTable(
                id="table",
                columns=[
                    {
                        "name": description_df.columns[0].upper(),
                        "id": description_df.columns[0],
                        "type": "text",
                        "format": {"specifier": "s"},
                    }
                ]
                + [
                    {
                        "name": i.upper(),
                        "id": i,
                        "type": "numeric",
                        "format": {"specifier": j},
                    }
                    for i, j in zip(description_df.columns[1:], specifiers[1:])
                ],
                data=description_df.to_dict("records"),
                page_size=20,
                # Highlight any columns that do not have a complete set of records,
                # by comparing count against the length of the DF.
                style_data_conditional=[
                    {
                        "if": {
                            "filter_query": "{{count}} < {}".format(df.shape[0]),
                            "column_id": "count",
                        },
                        "backgroundColor": "FireBrick",
                        "color": "white",
                    },
                    {
                        "if": {"filter_query": "{{% missing values}} > {}".format(missing_value_cutoff)},
                        "backgroundColor": "Grey",
                        "color": "white",
                    }
                ],
            ),
        )

        return fields_table_layout

class DataPreview(Component):
    """
    Component that displays the first few lines of a data frame
    """

    def __init__(self, app, title, id_prefix="preview-", update_div_id="df-loaded-div", data_id=data_store.MAIN_DATA):
        """
        :param app: Dash application
        :param id_prefix: Prefix string for HTML component identifiers
        :param update_div_id: ID of div that signals when to update
        """
        self.data_id = data_id
        self.title = title

        Component.__init__(self, app, id_prefix, children=[
            html.H3(id=id_prefix+"title", style=vstack),
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
        ds = data_store.get()

        # We want to be able to see the index columns in the preview table
        dff = ds.load(self.data_id, keep_index_cols=True)

        if dff.size > 0:
            return self.title, html.Div(children=[
                dash_table.DataTable(
                    id=self.id_prefix + "data-table",
                    columns=[{"name": i, "id": i} for i in dff.columns],
                    data=dff.head().to_dict("records"),
                    style_table={"overflowX": "auto"},
                ),
                html.Div([
                    html.Div("Rows: " + str(dff.shape[0])),
                    html.Div("Columns: " + str(dff.shape[1])),
                ])
            ])
        else:
            return "", html.Div(dash_table.DataTable(self.id_prefix + "data-table"))
