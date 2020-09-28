import logging
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output
from psych_dashboard.app import app, indices
from psych_dashboard.load_feather import load

logging.getLogger(__name__)


@app.callback(
    Output('table_preview', 'children'),
    [Input('df-loaded-div', 'children')])
def update_preview_table(df_loaded):
    logging.info(f'update_preview_table')

    dff = load('df')

    # Add the indices back in as columns so we can see them in the table preview
    if dff.size > 0:
        for index_level, index in enumerate(indices):
            dff.insert(loc=index_level, column=index, value=dff.index.get_level_values(index_level))

        return html.Div(dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in dff.columns],
            data=dff.head().to_dict('records'),
        ),
                        )

    return html.Div(dash_table.DataTable(id='table'))
