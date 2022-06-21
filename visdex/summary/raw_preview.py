"""
visdex: raw preview table

Shows the first few rows in the raw data and a listing of fields
"""
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output

from visdex.common import Collapsible
import visdex.session

class RawPreview(Collapsible):
    def __init__(self, app, id_prefix="rawpreview-"):
        """
        :param app: Dash application
        """
        import visdex.summary.data_preview as data_preview
        Collapsible.__init__(self, app, id_prefix, title="Raw data preview", children=[
            html.H3(children="Column Summary"),
            dcc.Loading(
                id=id_prefix+"loading-fields",
                children=[
                    html.Div(
                        id=id_prefix+"fields",
                        className="data-table",
                    ),
                ],
            ),
            data_preview.DataPreview(app, "Data preview (unfiltered)"),
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
        ds = visdex.session.get()

        df = ds.load(visdex.session.MAIN_DATA)
        if df.empty:
            return [], False

        description_df = ds.description(df)
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
