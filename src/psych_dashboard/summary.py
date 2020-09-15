import math
import numpy as np
import pandas as pd
import time
import dash_table
import dash
import dash_html_components as html
import dash_core_components as dcc
import scipy.stats as stats
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from psych_dashboard.app import app, indices
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.cluster import AgglomerativeClustering
from psych_dashboard.load_feather import load_feather, load_filtered_feather, load_pval, load_corr, load_logs, load_flattened_logs
from psych_dashboard.exploratory_graphs.manhattan_graph import calculate_transformed_corrected_pval, calculate_manhattan_data, flattened
from itertools import product
from functools import wraps

timing_dict = dict()


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('#### func:%r took: %2.4f sec' %
              (f.__name__, te-ts))
        timing_dict[f.__name__] = te-ts
        return result
    return wrap


@app.callback(
    [Output('table_summary', 'children'),
     Output('df-filtered-loaded-div', 'children')],
    [Input('df-loaded-div', 'children'),
     Input('missing-values-input', 'value')])
@timing
def update_summary_table(df_loaded, missing_value_cutoff):
    print('update_summary_table')
    dff = load_feather()

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

    # Calculate percentage of missing values
    description_df['% missing values'] = 100*(1-description_df['count']/len(dff.index))

    # Create a filtered version of the DF which doesn't have the index columns.
    dff_filtered = dff.drop(indices, axis=1)
    # Take out the columns which are filtered out by failing the 'missing values' threshold.
    dropped_columns = []
    if missing_value_cutoff not in [None, '']:
        for col in description_df.index:
            if description_df['% missing values'][col] > float(missing_value_cutoff):
                dff_filtered.drop(col, axis=1, inplace=True)
                dropped_columns.append(col)

    # Save the filtered dff to feather file. This is the file that will be used for all further processing.
    dff_filtered.reset_index().to_feather('df_filtered.feather')

    # Add the index back in as a column so we can see it in the table preview
    description_df.insert(loc=0, column='column name', value=description_df.index)

    # Reorder the columns so that 50% centile is next to 'mean'
    description_df = description_df.reindex(columns=['column name', 'count', 'mean', '50%', 'std', 'min', '25%', '75%', 'max', '% missing values'])

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
@timing
def update_heatmap_dropdown(df_loaded):
    print('update_heatmap_dropdown', df_loaded)
    dff = load_filtered_feather()

    options = [{'label': col,
                'value': col} for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]
    return options, \
        [col for col in dff.columns if dff[col].dtype in [np.int64, np.float64]]


@app.callback(
    [Output('heatmap', 'figure'),
     Output('corr-loaded-div', 'children'),
     Output('pval-loaded-div', 'children'),
     ],
    [Input('heatmap-dropdown', 'value'),
     Input('heatmap-clustering-input', 'value')],
    [State('df-loaded-div', 'children')])
@timing
def update_summary_heatmap(dropdown_values, clusters, df_loaded):
    print('update_summary_heatmap', dropdown_values, clusters)
    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load_filtered_feather()
        corr_dff = load_corr()
        pval_dff = load_pval()

        ts = time.time()

        # Add the index back in as a column so we can see it in the table preview
        if dff.size > 0 and dropdown_values != []:
            dff.insert(loc=0, column='SUBJECTKEY(INDEX)', value=dff.index)
            dff.dropna(inplace=True)

            # The columns we want to have calculated
            selected_columns = list(dropdown_values)
            print('selected_columns', selected_columns)

            # Work out which columns/rows are needed anew, and which are already populated
            # TODO: note that if we load in a new file with some of the same column names, then this old correlation
            # TODO: data may be used erroneously.
            previous_cols = corr_dff.columns
            overlap = set(selected_columns).intersection(set(previous_cols))
            print('these are needed and already available:', overlap)
            required_new = set(selected_columns).difference(set(previous_cols))
            print('these are needed and not already available:', required_new)

            # If there is no overlap, then skip this step and create a brand new empty dataframe instead.
            if len(overlap) != 0:
                # Copy across existing corr and pval data rather than recalculating
                corr = corr_dff[overlap][overlap]
                pvalues = pval_dff[overlap][overlap]

                # Create nan elements in correlation matrix and p-values matrix for those values which
                # will be calculated
                if len(required_new) != 0:
                    corr = corr.append([pd.Series(np.nan, index=corr.columns, name=col) for col in required_new])
                    pvalues = pvalues.append([pd.Series(np.nan, index=pvalues.columns, name=col) for col in required_new])
                    for col in required_new:
                        corr[col] = np.nan
                        pvalues[col] = np.nan

            # Reorder the corr and pvalues matrices to match the input order from the dropdown
                corr = corr.reindex(selected_columns)
                corr = corr[selected_columns]
                pvalues = pvalues.reindex(selected_columns)
                pvalues = pvalues[selected_columns]

            else:
                corr = pd.DataFrame(index=selected_columns, columns=selected_columns)
                pvalues = pd.DataFrame(index=selected_columns, columns=selected_columns)

            te = time.time()
            timing_dict['update_summary_heatmap-init-corr'] = te - ts
            ts = time.time()
            # Populate missing elements in correlation matrix and p-values matrix using stats.pearsonr
            # Firstly, convert the dff columns needed to numpy (far faster than doing it each iteration)
            np_dff_sel = dff[selected_columns].to_numpy()
            np_dff_req = dff[required_new].to_numpy()
            counter = 0
            for v1, v2 in product(selected_columns, required_new):
                # Use counter to work out the indexing into the pre-numpyed arrays.
                v2_counter = counter % len(selected_columns)
                v1_counter = math.floor(counter / len(selected_columns))
                # Calculate corr and p-val
                if v1 < v2:
                    corr.at[v1, v2], pvalues.at[v1, v2] = stats.pearsonr(np_dff_req[:, v1_counter], np_dff_sel[:, v2_counter])
                    # Populate the other half of the matrix
                    corr.at[v2, v1] = corr.at[v1, v2]
                    pvalues.at[v2, v1] = pvalues.at[v1, v2]
                counter += 1

            te = time.time()
            timing_dict['update_summary_heatmap-corr'] = te - ts
            ts = time.time()

            corr.fillna(0, inplace=True)
            cluster_method = 'hierarchical'
            if cluster_method == 'Kmeans':
                # K-means clustering of correlation values.
                w_corr = whiten(corr)
                centroids, _ = kmeans(w_corr, min(7, int(math.sqrt(w_corr.shape[0]))))
                # Generate the indices for each column, which cluster they belong to.
                clx, _ = vq(w_corr, centroids)

            elif cluster_method == 'hierarchical':
                cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')
                cluster.fit_predict(corr)
                clx = cluster.labels_
            else:
                raise ValueError

            te = time.time()
            timing_dict['update_summary_heatmap-cluster'] = te - ts
            ts = time.time()

            print(clx)
            print([x for _,x in sorted(zip(clx, selected_columns))])

            # TODO: what would be good here would be to rename the clusters based on the average variance (diags) within
            # TODO: each cluster - that would reduce the undesirable behaviour whereby currently the clusters can jump about
            # TODO: when re-calculating the clustering.
            # Sort corr columns
            sorted_column_order = [x for _, x in sorted(zip(clx, selected_columns))]

            half_sorted_corr = corr[sorted_column_order]
            half_sorted_pval = pvalues[sorted_column_order]

            # Sort corr rows
            sorted_corr = half_sorted_corr.reindex(sorted_column_order)
            sorted_pval = half_sorted_pval.reindex(sorted_column_order)

            te = time.time()
            timing_dict['update_summary_heatmap-reorder'] = te - ts
            ts = time.time()

            sorted_corr.reset_index().to_feather('corr.feather')
            sorted_pval.reset_index().to_feather('pval.feather')
            te = time.time()
            timing_dict['update_summary_heatmap-save'] = te - ts
            ts = time.time()
            # Remove the upper triangle and diagonal
            triangular = sorted_corr.to_numpy()
            triangular[np.tril_indices(triangular.shape[0], 0)] = np.nan
            triangular_pval = sorted_pval.to_numpy()
            triangular_pval[np.tril_indices(triangular_pval.shape[0], 0)] = np.nan
            te = time.time()
            timing_dict['update_summary_heatmap-triangular'] = te - ts

            fig = go.Figure(go.Heatmap(z=np.fliplr(triangular),
                                       x=sorted_corr.columns[-1::-1],
                                       y=sorted_corr.columns[:-1],
                                       zmin=-1,
                                       zmax=1,
                                       colorscale='RdBu',
                                       customdata=np.fliplr(triangular_pval),
                                       hovertemplate="%{x}<br>vs.<br>%{y}<br>      r: %{z:.2g}<br> pval: %{customdata:.2g}<extra></extra>",
                                       colorbar_title_text='r',
                                       ),
                            )

            fig.update_layout(xaxis_showgrid=False,
                              yaxis_showgrid=False,
                              plot_bgcolor='rgba(0,0,0,0)')

            # Find the indices where the sorted classes from the clustering change. Use these indices to plot vertical
            # lines on the heatmap to demarcate the different categories visually
            y = np.concatenate((np.array([0]), np.diff(sorted(clx))))
            fig.update_layout(shapes=[
                                  dict(
                                      type='line',
                                      yref='y',
                                      y0=-0.5,
                                      y1=len(sorted_corr.columns)-1.5,
                                      xref='x',
                                      x0=len(sorted_corr.columns)-float(i)-0.5,
                                      x1=len(sorted_corr.columns)-float(i)-0.5
                                  )
                              for i in np.where(y)[0]
                              ]
                              )
            return fig, True, True

    fig = go.Figure(go.Heatmap())
    fig.update_layout(xaxis_showgrid=False,
                      xaxis_zeroline=False,
                      xaxis_range=[0, 1],
                      yaxis_showgrid=False,
                      yaxis_zeroline=False,
                      yaxis_range=[0, 1],
                      plot_bgcolor='rgba(0,0,0,0)')

    return fig, False, False


@app.callback(
    Output('kde-figure', 'figure'),
    [Input('heatmap-dropdown', 'value')],
    [State('df-loaded-div', 'children')])
@timing
def update_summary_kde(dropdown_values, df_loaded):
    print('update_summary_kde')

    # Guard against the second argument being an empty list, as happens at first invocation
    if df_loaded is True:
        dff = load_filtered_feather()

        n_columns = len(dropdown_values) if dropdown_values is not None else 0
        if n_columns > 0:
            # Use a maximum of 5 columns
            n_cols = min(5, math.ceil(math.sqrt(n_columns)))
            n_rows = math.ceil(n_columns / n_cols)
            fig = make_subplots(n_rows, n_cols, subplot_titles = dropdown_values)

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
                            x = np.linspace(col_min - pad*data_range, col_max + pad*data_range, num=21)
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
            fig.update_layout(height=200*n_rows,
                              showlegend=False)
            return fig

    return go.Figure(go.Scatter())


# TODO: currently only allows int64 and float64
valid_manhattan_dtypes = [np.int64, np.float64]


@app.callback(
    Output('manhattan-figure', 'figure'),
    [Input('manhattan-pval-input', 'value'),
     Input('manhattan-logscale-check', 'value'),
     Input('df-filtered-loaded-div', 'children'),
     Input('pval-loaded-div', 'children')]
)
@timing
def plot_manhattan(pvalue, logscale, df_loaded, pval_loaded):
    print('plot_manhattan')
    dff = load_pval()
    manhattan_variable = [col for col in dff.columns if dff[col].dtype in valid_manhattan_dtypes]

    if not pval_loaded or manhattan_variable is None or manhattan_variable == []:
        return go.Figure()

    if pvalue <= 0. or pvalue is None:
        raise PreventUpdate

    ctx = dash.callback_context

    # Calculate p-value of corr coeff per variable against the manhattan variable, and the significance threshold.
    # Save logs and flattened logs to feather files
    # Skip this and reuse the previous values if we're just changing the log scale.
    if ctx.triggered[0]['prop_id'] not in ['manhattan-pval-input.value', 'manhattan-logscale-check.value']:
        logs = calculate_manhattan_data(dff, manhattan_variable)
        logs.reset_index().to_feather('logs.feather')
        flattened_logs = flattened(logs).dropna()
        flattened_logs.reset_index().to_feather('flattened_logs.feather')
        transformed_corrected_ref_pval = calculate_transformed_corrected_pval(float(pvalue), logs)
    else:
        print('using logscale shortcut')
        logs = load_logs()
        flattened_logs = load_flattened_logs()
        transformed_corrected_ref_pval = calculate_transformed_corrected_pval(float(pvalue), logs)

    fig = go.Figure(go.Scatter(x=[[item[i] for item in flattened_logs.index[::-1]] for i in range(0, 2)],
                               y=np.flip(flattened_logs.values),
                               mode='markers'
                               ),
                    )

    fig.update_layout(shapes=[
        dict(
            type='line',
            yref='y', y0=transformed_corrected_ref_pval, y1=transformed_corrected_ref_pval,
            xref='x', x0=0, x1=len(flattened_logs)-1
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
        yaxis_type='log' if logscale == ['LOG'] else None
    )
    print(pd.DataFrame(timing_dict.items()))
    return fig
