import logging
import dash_html_components as html
import dash_table
import numpy as np
from dash.dependencies import Input, Output
from psych_dashboard.app import app, indices
from psych_dashboard.load_feather import load, store
from psych_dashboard.timing import timing


@app.callback(
    [
        Output("other_summary", "children"),
        Output("table_summary", "children"),
        Output("df-filtered-loaded-div", "children"),
    ],
    [Input("df-loaded-div", "children"), Input("missing-values-input", "value")],
    prevent_initial_call=True,
)
@timing
def update_summary_table(df_loaded, missing_value_cutoff):
    logging.info(f"update_summary_table")
    dff = load("df")

    # If empty, return an empty Div
    if dff.size == 0:
        return html.Div(), html.Div(), False

    for index_level, index in enumerate(indices):
        dff.insert(
            loc=index_level, column=index, value=dff.index.get_level_values(index_level)
        )

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
    dff_filtered = dff.drop(indices, axis=1)
    # Take out the columns which are filtered out by failing the 'missing values' threshold.
    dropped_columns = []
    if missing_value_cutoff not in [None, ""]:
        for col in description_df.index:
            if description_df["% missing values"][col] > float(missing_value_cutoff):
                dff_filtered.drop(col, axis=1, inplace=True)
                dropped_columns.append(col)

    # Save the filtered dff to feather file. This is the file that will be used for all further processing.
    store("filtered", dff_filtered)

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
    return (
        html.Div(
            [
                html.Div("#rows: " + str(dff.shape[0])),
                html.Div("#columns: " + str(dff.shape[1])),
            ]
        ),
        html.Div(
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
                # style_table={'height': '300px'},
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
            # style={'width': '90%'}
        ),
        True,
    )
