"""
visdex: Summary statistics table

Displays summary statistics about the data, e.g. mean values of numerical
data, % of missing data items
"""
import logging

import numpy as np

from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

from visdex.common import vstack, hstack, standard_margin_left
from visdex.data import data_store
from visdex.common.timing import timing

LOG = logging.getLogger(__name__)

def get_layout(app):
        
    @app.callback(
        [
            Output("other-summary", "children"),
            Output("table-summary", "children"),
            Output("filtered-loaded-div", "children"),
        ],
        [
            Input("df-loaded-div", "children"),
            Input("missing-values-input", "value")
        ],
        prevent_initial_call=True,
    )
    @timing
    def update_summary_table(df_loaded, missing_value_cutoff):
        LOG.info(f"update_summary_table")
        ds = data_store.get()
        # Keep index columns in the summary table
        dff = ds.load(data_store.MAIN_DATA, keep_index_cols=True)

        # If empty, return an empty Div
        if dff.size == 0:
            return html.Div(), html.Div(), False

        description_df = dff.describe().transpose()

        # Add largely empty rows to the summary table for non-numeric columns.
        for col in dff.columns:
            if col not in description_df.index:
                description_df.loc[col] = [dff.count()[col]] + [np.nan] * 7

        # Calculate percentage of missing values
        description_df["% missing values"] = 100 * (
            1 - description_df["count"] / len(dff.index)
        )

        # Create a filtered version of the DF which doesn't have the index columns.
        dff_filtered = dff.drop([col for col in dff.index.names if col in dff], axis=1)

        # Take out the columns which are filtered out by failing the 'missing values'
        # threshold.
        dropped_columns = []
        if missing_value_cutoff not in [None, ""]:
            for col in description_df.index:
                if description_df["% missing values"][col] > float(missing_value_cutoff):
                    dff_filtered.drop(col, axis=1, inplace=True)
                    dropped_columns.append(col)

        # Save the filtered dff to feather file. This is the file that will be used for
        # all further processing.
        ds.store(data_store.FILTERED, dff_filtered)

        # Add the index back in as a column so we can see it in the table preview
        description_df.insert(loc=0, column="column name", value=description_df.index)

        # Reorder the columns so that 50% centile is next to 'mean'
        description_df = description_df.reindex(
            columns=[
                "column name",
                "count",
                "mean",
                "50%",
                "std",
                "min",
                "25%",
                "75%",
                "max",
                "% missing values",
            ]
        )
        specifiers = ["s", "d", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f", ".2f"]

        other_summary_layout = html.Div(
            [
                html.Div("#rows: " + str(dff_filtered.shape[0])),
                html.Div("#columns: " + str(dff_filtered.shape[1])),
            ]
        )

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
                            "filter_query": "{{count}} < {}".format(dff.shape[0]),
                            "column_id": "count",
                        },
                        "backgroundColor": "FireBrick",
                        "color": "white",
                    }
                ]
                + [
                    {
                        "if": {"filter_query": "{{column name}} = {}".format(i)},
                        "backgroundColor": "Grey",
                        "color": "white",
                    }
                    for i in dropped_columns
                ],
            ),
        )

        return other_summary_layout, table_summary_layout, True

    @app.callback(
        Output("predicate-filter", "children"),
        [Input("add-row-predicate-button", "n_clicks")],
        [State("predicate-filter", "children")],
        prevent_initial_call=True,
    )
    def add_row_predicate(n_clicks, children):
        # Add a new graph group each time the button is clicked. The if None guard stops
        # there being an initial graph.
        LOG.info(f"add_row_predicate")
        if n_clicks is not None:
            new_predicate = html.Div(
                id={"type": "predicate-filter", "index": n_clicks},
                children=[
                    html.Label("Row filter predicate", style=hstack),
                    dcc.Input(
                        id="predicate-filter-input",
                        value=None,
                        style=hstack,
                    ),
                ],
            ),     

            children.insert(len(children)-1, new_predicate)

        return children

    layout = html.Div(children=[
        html.H3(children="Table Summary and Filter", style=vstack),
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
            id="predicate-filter",
            children=[
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
                html.Div(
                    id="other-summary",
                    style={
                        "width": "100%",
                        "margin-left": "10px",
                        "margin-right": "10px",
                    },
                ),
            ],
        ),
    ])

    return layout
