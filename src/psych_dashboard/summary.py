import math
import numpy as np
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import scipy.stats as stats
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from psych_dashboard.app import app, indices
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.cluster import AgglomerativeClustering
from psych_dashboard.load_feather import load_feather


@app.callback(
    Output('table_summary', 'children'),
    [Input('df-loaded-div', 'children')])
def update_summary_table(df_loaded):
    print('update_summary_table')
    dff = load_feather(df_loaded)

    # Add the index back in as a column so we can see it in the table preview
    if dff.size > 0:
        for index_level, index in enumerate(indices):
            dff.insert(loc=index_level, column=index, value=dff.index.get_level_values(index_level))

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
    [Input('df-loaded-div', 'children')]
)
def update_heatmap_dropdown(df_loaded):
    dff = load_feather(df_loaded)

    options = [{'label': col,
                'value': col} for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]
    return [html.Div(["Select variables to display:", dcc.Dropdown(id='heatmap-dropdown',
                                                                   options=options,
                                                                   value=[col for col in dff.columns if dff[col].dtype in [np.int64, np.float64]],
                                                                   multi=True,
                                                                   style={'height': '100px', 'overflowY': 'auto'}
                                                                   )])]


@app.callback(
    Output('heatmap', 'figure'),
    [Input('heatmap-dropdown', 'value')],
    [State('df-loaded-div', 'children')])
def update_summary_heatmap(dropdown_values, df_loaded):
    print('update_summary_heatmap')

    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load_feather(df_loaded)

        # Add the index back in as a column so we can see it in the table preview
        if dff.size > 0 and dropdown_values != []:
            dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
            selected_columns = list(dropdown_values)
            print('selected_columns', selected_columns)
            corr = dff[selected_columns].corr(min_periods=1)
            print('corr', corr)
            corr.fillna(0, inplace=True)
            cluster_method = 'hierarchical'
            if cluster_method == 'Kmeans':
                # K-means clustering of correlation values.
                w_corr = whiten(corr)
                centroids, _ = kmeans(w_corr, min(7, int(math.sqrt(w_corr.shape[0]))))
                # Generate the indices for each column, which cluster they belong to.
                clx, _ = vq(w_corr, centroids)

            elif cluster_method == 'hierarchical':
                cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
                cluster.fit_predict(corr)
                clx = cluster.labels_
            else:
                raise ValueError

            print(clx)
            print([x for _,x in sorted(zip(clx, selected_columns))])

            # TODO: what would be good here would be to rename the clusters based on the average variance (diags) within
            # TODO: each cluster - that would reduce the undesirable behaviour whereby currently the clusters can jump about
            # TODO: when re-calculating the clustering.
            # Sort corr columns
            half_sorted_corr = corr[[x for _, x in sorted(zip(clx, selected_columns))]]
            print(half_sorted_corr)
            # Sort corr rows
            sorted_corr = half_sorted_corr.reindex([x for _, x in sorted(zip(clx, selected_columns))])

            return go.Figure(go.Heatmap(z=np.fliplr(np.triu(sorted_corr)),
                                        x=sorted_corr.columns[::-1],
                                        y=sorted_corr.columns,
                                        zmin=-1,
                                        zmax=1,
                                        colorscale='RdBu')
                             )

    return go.Figure(go.Heatmap())


@app.callback(
    Output('kde-figure', 'figure'),
    [Input('heatmap-dropdown', 'value')],
    [State('df-loaded-div', 'children')])
def update_summary_kde(dropdown_values, df_loaded):
    print('update_summary_kde')

    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load_feather(df_loaded)

        n_columns = len(dropdown_values) if dropdown_values is not None else 0
        if n_columns > 0:
            # Use a maximum of 5 columns
            n_cols = min(5, math.ceil(math.sqrt(n_columns)))
            n_rows = math.ceil(n_columns / n_cols)
            fig = make_subplots(n_rows, n_cols)

            # For each column, calculate its KDE and then plot that over the histogram.
            for j in range(n_cols):
                for i in range(n_rows):
                    if i*n_cols+j < n_columns:
                        col_name = dropdown_values[i*n_cols+j]
                        this_col = dff[col_name]
                        col_min = this_col.min()
                        col_max = this_col.max()
                        data_range = col_max - col_min
                        # Guard against a singular matrix in the KDE when a column contains only a single value
                        if data_range > 0:

                            # Generate KDE kernel
                            kernel = stats.gaussian_kde(this_col.dropna())
                            pad = 0.1
                            # Generate linspace
                            x = np.linspace(col_min - pad*data_range, col_max + pad*data_range, num=101)
                            # Sample kernel
                            y = kernel(x)
                            # Plot KDE line graph using sampled data
                            fig.add_trace(go.Scatter(x=x, y=y, name=col_name),
                                          i+1,
                                          j+1)
                        # Plot normalised histogram of data, regardless of KDE completion or not
                        fig.add_trace(go.Histogram(x=this_col, name=col_name, histnorm='probability density'),
                                      i+1,
                                      j+1)
            fig.update_layout(height=200*n_rows)
            return fig

    return go.Figure(go.Scatter())
