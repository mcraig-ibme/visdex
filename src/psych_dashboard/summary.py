import math
import numpy as np
import pandas as pd
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import scipy.stats as stats
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_bio import ManhattanPlot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from psych_dashboard.app import app, indices
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.cluster import AgglomerativeClustering
from psych_dashboard.load_feather import load_feather, load_filtered_feather, load_pval
from itertools import combinations_with_replacement


@app.callback(
    [Output('table_summary', 'children'),
     Output('df-filtered-loaded-div', 'children')],
    [Input('df-loaded-div', 'children'),
     Input('missing-values-input', 'value')])
def update_summary_table(df_loaded, missing_value_cutoff):
    print('update_summary_table')
    dff = load_feather(df_loaded)

    # If empty, return an empty Div
    if dff.size == 0:
        return html.Div(), False

    for index_level, index in enumerate(indices):
        dff.insert(loc=index_level, column=index, value=dff.index.get_level_values(index_level))

    description_df = dff.describe().transpose()

    # Add largely empty rows to the summary table for non-numeric columns.
    for col in dff.columns:
        if col not in description_df.index:
            description_df.loc[col] = [dff.count()[col]] + [np.nan] * 7

    # Create a filtered version of the DF which doesn't have the index columns.
    dff_filtered = dff.drop(indices, axis=1)
    # Take out the columns which are filtered out by failing the 'missing values' threshold.
    dropped_columns = []
    if missing_value_cutoff not in [None, '']:
        for col in description_df.index:
            if 100*description_df['count'][col]/len(dff.index) < 100. - float(missing_value_cutoff):
                dff_filtered.drop(col, axis=1, inplace=True)
                dropped_columns.append(col)
    # Save the filtered dff to feather file. This is the file that will be used for all further processing.
    dff_filtered.reset_index().to_feather('df_filtered.feather')

    # Add the index back in as a column so we can see it in the table preview
    description_df.insert(loc=0, column='column name', value=description_df.index)

    # Reorder the columns so that 50% centile is next to 'mean'
    description_df = description_df.reindex(columns=['column name', 'count', 'mean', '50%', 'std', 'min', '25%', '75%', 'max'])

    return html.Div([html.Div('#rows: ' + str(dff.shape[0])),
                     html.Div('#columns: ' + str(dff.shape[1])),
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
                                  }] +
                                 [{
                                     'if': {
                                        'filter_query': '{{column name}} = {}'.format(i)
                                     },
                                     'backgroundColor': 'Grey',
                                     'color': 'white'
    }
                                  for i in dropped_columns
                                  ]
                         )
                     )
                     ]
                    ), True


@app.callback(
    [Output('heatmap-dropdown', 'options'),
     Output('heatmap-dropdown', 'value')],
    [Input('df-filtered-loaded-div', 'children')]
)
def update_heatmap_dropdown(df_loaded):
    print('update_heatmap_dropdown', df_loaded)
    dff = load_filtered_feather(df_loaded)

    options = [{'label': col,
                'value': col} for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]
    return options, \
        [col for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]


@app.callback(
    Output('heatmap', 'figure'),
    [Input('heatmap-dropdown', 'value')],
    [State('df-loaded-div', 'children')])
def update_summary_heatmap(dropdown_values, df_loaded):
    print('update_summary_heatmap')

    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load_filtered_feather(df_loaded)

        # Add the index back in as a column so we can see it in the table preview
        if dff.size > 0 and dropdown_values != []:
            dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
            selected_columns = list(dropdown_values)
            print('selected_columns', selected_columns)
            dff.dropna(inplace=True)

            # Create and populate correlation matrix and p-values matrix using stats.pearsonr
            corr = pd.DataFrame(columns=selected_columns, index=selected_columns)
            pvalues = pd.DataFrame(columns=selected_columns, index=selected_columns)
            for v1, v2 in combinations_with_replacement(selected_columns, 2):
                corr[v1][v2], pvalues[v1][v2] = stats.pearsonr(dff[v1].values, dff[v2].values)
                # Populate the other half of the matrix
                if v1 != v2:
                    corr[v2][v1] = corr[v1][v2]
                    pvalues[v2][v1] = pvalues[v1][v2]

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
            half_sorted_pval = pvalues[[x for _, x in sorted(zip(clx, selected_columns))]]
            print(half_sorted_corr)
            # Sort corr rows
            sorted_corr = half_sorted_corr.reindex([x for _, x in sorted(zip(clx, selected_columns))])
            sorted_pval = half_sorted_pval.reindex([x for _, x in sorted(zip(clx, selected_columns))])

            sorted_corr.reset_index().to_feather('corr.feather')
            sorted_pval.reset_index().to_feather('pval.feather')
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
        dff = load_filtered_feather(df_loaded)

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


# TODO: currently only allows int64 and float64
valid_manhattan_dtypes = [np.int64, np.float64]


@app.callback(
    Output('manhattan-dd', 'options'),
    [Input('df-filtered-loaded-div', 'children')],
)
def update_manhattan_dropdown(df_loaded):
    if df_loaded:
        dff = load_filtered_feather(df_loaded)

        return [{'label': col,
                 'value': col} for col in dff.columns if dff[col].dtype in valid_manhattan_dtypes]
    else:
        return []


def calculate_manhattan_data(dff, manhattan_variable, ref_pval):
    # Filter columns to those with valid types.

    if manhattan_variable is None:
        manhattan_variables = dff.columns
    else:
        if isinstance(manhattan_variable, list):
            manhattan_variables = manhattan_variable
        else:
            manhattan_variables = [manhattan_variable]

    # Create DF to hold the results of the calculations, and perform the log calculation
    logs = pd.DataFrame(columns=dff.columns, index=manhattan_variables)
    for variable in dff:
        logs[variable] = -np.log10(dff[variable])

    # Now blank out any duplicates including the diagonal
    for ind in logs.index:
        for col in logs.columns:
            if ind in logs.columns and col in logs.index and logs[col][ind] == logs[ind][col]:
                logs[ind][col] = np.nan

    # Divide reference p-value by number of variable pairs to get corrected p-value
    corrected_ref_pval = ref_pval / (logs.notna().sum().sum())
    # Transform corrected p-value by -log10
    transformed_corrected_ref_pval = -np.log10(corrected_ref_pval)

    return logs, transformed_corrected_ref_pval


@app.callback(
    Output('manhattan-figure', 'figure'),
    [Input('manhattan-dd', 'value'),
     Input('manhattan-pval-input', 'value'),
     Input('manhattan-logscale-check', 'value')],
    [State('df-filtered-loaded-div', 'children')]
)
def plot_manhattan(manhattan_variable, pvalue, logscale, df_loaded):
    print('plot_manhattan', manhattan_variable)
    if not df_loaded or manhattan_variable is None or manhattan_variable == []:
        return go.Figure()

    if pvalue <= 0. or pvalue is None:
        raise PreventUpdate

    dff = load_filtered_feather(df_loaded).dropna()

    # Calculate p-value of corr coeff per variable against the manhattan variable, and the significance threshold
    logs, transformed_corrected_ref_pval = calculate_manhattan_data(load_pval(df_loaded), manhattan_variable, float(pvalue))

    fig = px.scatter(logs, log_y=logscale == ['LOG'])

    fig.update_layout(shapes=[
        dict(
            type='line',
            yref='y', y0=transformed_corrected_ref_pval, y1=transformed_corrected_ref_pval,
            xref='x', x0=0, x1=len(logs)-1
        )
    ],
        annotations=[
            dict(
                x=0,
                y=transformed_corrected_ref_pval if logscale != ['LOG'] else np.log10(transformed_corrected_ref_pval),
                xref='x',
                yref='y',
                text='{:f}'.format(transformed_corrected_ref_pval),
                showarrow=True,
                arrowhead=7,
                ax=-50,
                ay=0
            ),
            ],
        xaxis_title='variable',
        yaxis_title='-log10(p)',
    )
    return fig


@app.callback(
    Output('manhattan-all-figure', 'figure'),
    [Input('manhattan-all-pval-input', 'value'),
     Input('df-filtered-loaded-div', 'children'),
     Input('manhattan-all-logscale-check', 'value')]
)
def plot_all_manhattan(pvalue, df_loaded, logscale):
    # TODO: reuse the correlation/pvalue calculations between heatmap and manhattan plot
    print('plot_manhattan')
    if not df_loaded:
        return go.Figure()

    if pvalue <= 0. or pvalue is None:
        raise PreventUpdate

    dff = load_filtered_feather(df_loaded).dropna()

    # Calculate p-value of corr coeff per variable against the manhattan variable, and the significance threshold
    logs, transformed_corrected_ref_pval = calculate_manhattan_data(load_pval(df_loaded), None, float(pvalue))

    fig = px.scatter(logs, log_y=logscale == ['LOG'])

    fig.update_layout(shapes=[
        dict(
            type='line',
            yref='y', y0=transformed_corrected_ref_pval, y1=transformed_corrected_ref_pval,
            xref='x', x0=0, x1=len(logs)-1
        )
    ],
        annotations=[
            dict(
                x=0,
                y=transformed_corrected_ref_pval if logscale != ['LOG'] else np.log10(transformed_corrected_ref_pval),
                xref='x',
                yref='y',
                text='{:f}'.format(transformed_corrected_ref_pval),
                showarrow=True,
                arrowhead=7,
                ax=-50,
                ay=0
            ),
            ],
        xaxis_title='variable',
        yaxis_title='-log10(p)',
    )
    return fig