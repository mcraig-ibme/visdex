"""
visdex: Summary statistics table

Displays summary statistics about the data, e.g. mean values of numerical
data, % of missing data items
"""
import logging

from dash import html, dcc, dash_table, callback_context
from dash.dependencies import Input, Output, State, ALL

from visdex.common import vstack, hstack, standard_margin_left
from visdex.data import data_store
from visdex.common.timing import timing

LOG = logging.getLogger(__name__)

def get_layout(app):
        
    @app.callback(
        [
            Output("table-summary", "children"),
            Output("filtered-loaded-div", "children"),
        ],
        [
            Input("df-loaded-div", "children"),
            Input("missing-values-input", "value"),
            Input({"type": "predicate-filter-column", "index": ALL}, "value"),
            Input({"type": "predicate-filter-op", "index": ALL}, "value"),
            Input({"type": "predicate-filter-value", "index": ALL}, "value"),
            Input("predicate-filter-container", "children"),
        ],
        prevent_initial_call=True,
    )
    @timing
    def update_summary_table(df_loaded, missing_value_cutoff, pred_cols, pred_ops, pred_values, _pred_children):
        LOG.info(f"update_summary_table")
        ds = data_store.get()

        description_df = ds.description
        df = ds.load(data_store.MAIN_DATA)

        specifiers = ["s", "d", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]
        table_summary_layout = html.Div(
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

        ds.missing_values_threshold = missing_value_cutoff
        ds.predicates= list(zip(pred_cols, pred_ops, pred_values))

        return table_summary_layout, True

    @app.callback(
        Output("predicate-filter-container", "children"),
        Input("df-loaded-div", "children"),
        Input("add-row-predicate-button", "n_clicks"),
        State("predicate-filter-container", "children"),
        prevent_initial_call=True,
    )
    def add_row_predicate(n_clicks, df_loaded, children): 
        which_input = callback_context.triggered[0]['prop_id'].split('.')[0]
        if which_input == "df-loaded-div":
            LOG.info(f"reset row predicates")
            return []
        else:
            LOG.info(f"add_row_predicate %s", n_clicks)
            if n_clicks:
                df = data_store.get().load(data_store.FILTERED)
                LOG.info(list(df.columns))
                col_options = [
                    {'label': c, 'value': c}
                    for c in list(df.columns)
                ]
                op_options = [
                    {'label': c, 'value': c}
                    for c in ["==", ">", ">=", "<", "<=" "!="]
                ]
                new_predicate = html.Div(
                    [
                        html.Label("Row filter", style={"display" : "inline-block", "verticalAlign" : "middle"}),
                        dcc.Dropdown(
                            id={"type" : "predicate-filter-column", "index" : n_clicks},
                            options=col_options, value=None,
                            style={"width" : "300px", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                        dcc.Dropdown(
                            id={"type" : "predicate-filter-op", "index" : n_clicks},
                            options=op_options, value="=",
                            style={"width" : "50px", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                        dcc.Input(
                            id={"type" : "predicate-filter-value", "index" : n_clicks},
                            style={"width" : "20%", "display" : "inline-block", "verticalAlign" : "middle"},
                        ),
                    ],
                    id="predicate-filter-%i" % n_clicks,
                )

                children.append(new_predicate)
        
            return children

    layout = html.Div(children=[
        html.H3(children="Column Summary", style=vstack),
        dcc.Loading(
            id="loading-table-summary",
            children=[
                html.Div(
                    id="table-summary",
                    style={
                        "width": "95%",
                        "margin-left": "10px",
                        "margin-right": "10px",
                    },
                ),
            ],
        ),
        html.H3(children="Row/column filter", style=vstack),
        html.Div(
            id="missing-values-filter",
            children=[
                html.Label("Filter out all columns missing at least X percentage of rows:", style=hstack),
                dcc.Input(
                    id="missing-values-input",
                    type="number",
                    min=0,
                    max=100,
                    debounce=True,
                    value=None,
                    style=hstack,
                ),
            ],
        ),        
        html.Div(
            id="predicate-filters",
            children=[
                html.Div(id="predicate-filter-container", children=[]),
                html.Button(
                    "Add row condition",
                    id="add-row-predicate-button",
                    style={
                        "margin-top": "10px",
                        "margin-left": standard_margin_left,
                        "margin-bottom": "40px",
                    },
                ),
            ],
        ),
    ])

    return layout
