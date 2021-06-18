import logging
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output
from visdex.app import app, cache

logging.getLogger(__name__)


@app.callback(
    Output("table_preview", "children"),
    [Input("df-loaded-div", "children")],
    prevent_initial_call=True,
)
def update_preview_table(df_loaded):
    logging.info(f"update_preview_table")

    dff = cache.load("df")

    if dff.size > 0:
        # Add the index back in as columns so we can see them in the table preview
        # FIXME MSC removed as we are no longer dropping index columns
        #for index_level, index in enumerate([i for i in indices if i in dff]):
        #    dff.insert(
        #        loc=index_level,
        #        column=index,
        #        value=dff.index.get_level_values(index_level),
        #    )

        return html.Div(
            dash_table.DataTable(
                id="table",
                columns=[{"name": i, "id": i} for i in dff.columns],
                data=dff.head().to_dict("records"),
                style_table={"overflowX": "auto"},
            ),
        )

    return html.Div(dash_table.DataTable(id="table"))
