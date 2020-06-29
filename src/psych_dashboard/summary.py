import json
import math
import pandas as pd
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from psych_dashboard.app import app
from scipy.cluster.vq import kmeans, vq, whiten


@app.callback(
    Output('table_summary', 'children'),
    [Input('json-df-div', 'children')])
def update_summary_table(input_json_df):
    print('update_summary_table')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    # Add the index back in as a column so we can see it in the table preview
    if dff.size > 0:
        dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
        description_df = dff.describe().transpose()
        description_df.insert(loc=0, column='', value=description_df.index)
        return html.Div([html.Div(['nrows:' + str(dff.shape[0]),
                                   'ncols:' + str(dff.shape[1])]
                                  ),
                         html.Div(
                             dash_table.DataTable(
                                 id='table',
                                 columns=[{'name': i.upper(),
                                           'id': i,
                                           'type': 'numeric',
                                           'format': {'specifier': '.2f'}} for i in description_df.columns],
                                 data=description_df.to_dict('record'),
                                 fixed_rows={'headers': True},
                                 style_table={'height': '300px', 'overflowY': 'auto'},
                                 # Highlight any columns that do not have a complete set of records,
                                 # by comparing count against the length of the DF.
                                 style_data_conditional=[
                                     {
                                         'if': {
                                             'filter_query': '{{count}} < {}'.format(dff.shape[0]),
                                             'column_id': 'count'
                                         },
                                         'backgroundColor': 'FireBrick',
                                         'color': 'white'
                                     }
                                 ]
                             )
                         )
                         ]
                        )

    return html.Div()


@app.callback(
    Output('heatmap-div', 'children'),
    [Input('json-df-div', 'children')]
)
def update_heatmap_dropdown(input_json_df):
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]
    return [html.Div(["Select variables to display:", dcc.Dropdown(id='heatmap-dropdown',
                                                                   options=options,
                                                                   value=[col for col in dff.columns if dff[col].dtype in [np.int64, np.float64]],
                                                                   multi=True,
                                                                   style={'height': '100px', 'overflowY': 'auto'})])]


@app.callback(
    Output('heatmap', 'figure'),
    [Input('json-df-div', 'children'),
     Input('heatmap-dropdown', 'value')])
def update_summary_heatmap(input_json_df, dropdown_values):
    print('update_summary_heatmap')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    # Add the index back in as a column so we can see it in the table preview
    if dff.size > 0 and dropdown_values != []:
        dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
        selected_columns = list(dropdown_values)
        print('selected_columns', selected_columns)
        cov = dff[selected_columns].corr()
        print('cov', cov)
        # K-means clustering of covariance values.
        w_cov = whiten(cov)
        print(w_cov)
        centroids, _ = kmeans(w_cov, min(7,int(math.sqrt(w_cov.shape[0]))))
        print(centroids)
        # Generate the indices for each column, which cluster they belong to.
        clx, _ = vq(w_cov, centroids)
        print(clx)
        print([x for _,x in sorted(zip(clx,selected_columns))])
        # TODO: what would be good here would be to rename the clusters based on the average variance (diags) within
        # TODO: each cluster - that would reduce the undesirable behaviour whereby currently the clusters can jump about
        # TODO: when re-calculating the clustering.
        # Sort cov columns
        half_sorted_cov = cov[[x for _,x in sorted(zip(clx,selected_columns))]]
        print(half_sorted_cov)
        # Sort cov rows
        sorted_cov = half_sorted_cov.reindex([x for _, x in sorted(zip(clx, selected_columns))])
        return go.Figure(go.Heatmap(z=sorted_cov,
                                    x=sorted_cov.columns,
                                    y=sorted_cov.columns,
                                    zmin=-1,
                                    zmax=1,
                                    colorscale='RdBu')
                         )

    return go.Figure(go.Heatmap())