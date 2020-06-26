import json
import pandas as pd
import dash_table
import dash_html_components as html
from dash.dependencies import Input, Output
from psych_dashboard.app import app


@app.callback(
    Output('table_preview', 'children'),
    [Input('json-df-div', 'children')])
def update_preview_table(input_json_df):
    print('update_preview_table')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    # Add the index back in as a column so we can see it in the table preview
    if dff.size > 0:
        dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
    return html.Div(dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in dff.columns],
        data=dff.head().to_dict('record'),
    ),
                    )
