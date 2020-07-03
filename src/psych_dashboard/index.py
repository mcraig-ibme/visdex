import glob
import json
import itertools
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State, MATCH
from dash.exceptions import PreventUpdate
import dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from psych_dashboard import preview_table, summary, single_scatter
from psych_dashboard.app import app, indices, graph_types, scatter_graph_dimensions, bar_graph_dimensions
from psych_dashboard.load_feather import load_feather




global_width = '100%'
default_marker_color = "crimson"
file_extensions = ['*.txt', '*.xlsx']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }


def generate_graph_button(graph_no):
    return html.Button('New Graph', id='add-graph-button'+str(graph_no))


app.layout = html.Div(children=[
    html.H1(children='ABCD data exploration dashboard'),

    html.H2(children="File selection"),

    html.Label(children='Files Multi-Select'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in
                 [f for f_ in [glob.glob('../data/'+e) for e in file_extensions] for f in f_]
                 ],
        value=[],
        multi=True
    ),
    dcc.Loading(
        id='loading-filenames-div',
        children=[html.Div(id='filenames-div')],
    ),
    html.H2(children="Table Preview"),
    dcc.Loading(
        id='loading-table-preview',
        children=[
            html.Div(id='table_preview',
                     style={'width': global_width})
        ]
    ),
    html.H2(children="Table Summary"),
    dcc.Loading(
        id='loading-table-summary',
        children=[
            html.Div(id='table_summary',
                     style={'width': global_width})
            ]
    ),
    html.H2(children='Correlation Heatmap'),
    html.Div(id='heatmap-div',
             children=[html.Div(["Select variables to display:", dcc.Dropdown(id='heatmap-dropdown', options=([]))])]),
    dcc.Loading(
        id='loading-heatmap',
        children=[
            dcc.Graph(id='heatmap',
                      figure=go.Figure()
                      )
            ]
    ),
    html.H2(children='Per-variable Histograms and KDEs'),
    dcc.Loading(
        id='loading-kde-figure',
        children=[
            dcc.Graph(id='kde-figure',
                      figure=go.Figure()
                      )
            ]
    ),

    # Hidden div for holding the boolean identifying whether a DF is loaded
    html.Div(id='df-loaded-div', style={'display': 'none'}, children=[]),

    # html.H2(children="Scatter"),
    # # Div holds the graph selection elements
    # html.Div(id='graph1-selection',
    #          children=[html.Div([value + ":", dcc.Dropdown(id=key, options=[])], style=style_dict)
    #                    for key, value in scatter_graph_dimensions.items()
    #                    ]
    #          + [html.Div(["regression:", dcc.Input(id="regression",
    #                                                type='number',
    #                                                min=0,
    #                                                step=1
    #                                                )
    #                       ],
    #                      style=style_dict)
    #             ]
    #          ),
    #
    # dcc.Graph(id="graph1", style={"width": global_width}),

    # html.H2(children="Count per category"),
    # html.Div(id='bar-div',
    #          children=[html.Div(["Select variable to group by:", dcc.Dropdown(id='bar-dropdown', options=([]))])]),
    # dcc.Graph(
    #     id='bar-chart',
    #     figure=go.Figure(data=go.Bar()),
    # ),
    # html.H2(children="Boxplot per category"),
    # html.Div(id='box-div',
    #          children=[html.Div(["Select variables to view:", dcc.Dropdown(id='box-dropdown', options=([]))])]),
    # dcc.Graph(
    #     id='box-plot',
    #     figure=go.Figure(data=go.Bar()),
    # ),
    html.Div(id='graph-group-container', children=[]),
    generate_graph_button(1)
])


def generate_scatter_group(n_clicks):
    print('generate_scatter_group')
    return html.Div(id={'type': 'filter-graph-group-scatter',
                        'index': n_clicks
                        },
                    children=[html.Div([value + ":", dcc.Dropdown(id={'type': 'scatter_'+str(key), 'index': n_clicks},
                                                                  options=[])],
                                       id={'type': 'div_scatter_'+str(key), 'index': n_clicks},
                                       style=style_dict
                                       )
                              for key, value in scatter_graph_dimensions.items()
                              ]
                    + [html.Div(["regression:", dcc.Input(id={'type': 'scatter_regression', 'index': n_clicks},
                                                          type='number',
                                                          min=0,
                                                          step=1,
                                                          )
                                 ],
                                id={'type': 'div_scatter_regression', 'index': n_clicks},
                                style=style_dict)
                       ]
                    + [dcc.Graph(id={'type': 'gen_scatter_graph', 'index': n_clicks},
                                 figure=go.Figure(data=go.Scatter()))
                       ]
                    )


def generate_bar_group(n_clicks):
    print('generate_bar_group')
    return html.Div(id={'type': 'filter-graph-group-bar',
                        'index': n_clicks
                        },
                    children=[html.Div([value + ":", dcc.Dropdown(id={'type': 'bar_'+key, 'index': n_clicks},
                                                                  options=[])],
                                       id={'type': 'div_bar_'+str(key), 'index': n_clicks},
                                       style=style_dict
                                       )
                              for key, value in bar_graph_dimensions.items()
                              ]
                    + [dcc.Graph(id={'type': 'gen_bar_graph', 'index': n_clicks},
                                 figure=go.Figure(data=go.Bar()))
                       ]
                    )


@app.callback(
    Output({'type': 'divgraph-type-dd', 'index': MATCH}, 'children'),
    [Input({'type': 'graph-type-dd', 'index': MATCH}, 'value')],
    [State({'type': 'graph-type-dd', 'index': MATCH}, 'id'),
     State({'type': 'divgraph-type-dd', 'index': MATCH}, 'children')]
)
def change_graph_group_type(graph_type, id, children):
    print('change_graph_group_type', graph_type)
    if graph_type == 'Bar':
        children[-1] = generate_bar_group(id['index'])
    elif graph_type == 'Scatter':
        children[-1] = generate_scatter_group(id['index'])
    else:
        raise ValueError
    return children


@app.callback(
    Output('graph-group-container', 'children'),
    [Input('add-graph-button1', 'n_clicks')],
    [State('graph-group-container', 'children')])
def display_graph_groups(n_clicks, children):
    # Add a new graph group each time the button is clicked. The if None guard stops there being an initial graph.
    print('display_graph_groups')
    if n_clicks is not None:
        # This dropdown controls what type of graph-group to display next to it.
        new_graph_type_dd = html.Div(['Graph type:',
                                      dcc.Dropdown(id={'type': 'graph-type-dd',
                                                       'index': n_clicks
                                                       },
                                                   options=[{'label': value,
                                                             'value': value
                                                             }
                                                            for value in graph_types
                                                            ],
                                                   value='Scatter',
                                                   style={'width': '50%'}
                                                   ),
                                      # Default graph-group type is Scatter
                                      generate_scatter_group(n_clicks)
                                      ],
                                     id={'type': 'divgraph-type-dd', 'index': n_clicks},
                                     )

        children.append(new_graph_type_dd)

    return children


@app.callback(
    [Output({'type': 'div_scatter_'+str(t), 'index': MATCH}, 'children')
     for t in (list(scatter_graph_dimensions) + ['regression'])],
    [Input('df-loaded-div', 'children')],
    list(itertools.chain.from_iterable([State({'type': 'scatter_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'scatter_'+t, 'index': MATCH}, 'value')
                                        ] for t in (list(scatter_graph_dimensions) + ['regression'])))
    + [State({'type': 'div_scatter_x', 'index': MATCH}, 'style')]
)
def update_scatter_select_columns(df_loaded, x, xv, y, yv, color, colorv, facet_col, fcv, facet_row, frv, regression, regv,
                              style_dict):
    print('update_scatter_select_columns')
    print(x, xv, y, yv, color, colorv, facet_col, fcv, facet_row, frv, regression, regv)
    print('style_dict')
    print(style_dict)
    ctx = dash.callback_context
    ctx_msg = json.dumps(
        {
            'states': ctx.states,
            'triggered': ctx.triggered
        },
        indent=2)
    print(ctx_msg)
    dff = load_feather(df_loaded)
    options = [{'label': col,
                'value': col} for col in dff.columns]

    return ["x:", dcc.Dropdown(id={'type': 'scatter_x', 'index': x['index']}, options=options, value=xv)], \
           ["y:", dcc.Dropdown(id={'type': 'scatter_y', 'index': y['index']}, options=options, value=yv)], \
           ["color:", dcc.Dropdown(id={'type': 'scatter_color', 'index': color['index']}, options=options, value=colorv)], \
           ["split_horizontally:", dcc.Dropdown(id={'type': 'scatter_facet_col', 'index': facet_col['index']},
                                                options=options, value=fcv)], \
           ["split_vertically:", dcc.Dropdown(id={'type': 'scatter_facet_row', 'index': facet_row['index']},
                                              options=options, value=frv)], \
           ["regression degree:", dcc.Input(id={'type': 'scatter_regression', 'index': regression['index']},
                                            type='number',
                                            min=0,
                                            step=1,
                                            value=regv)]


@app.callback(
    Output({'type': 'gen_scatter_graph', 'index': MATCH}, "figure"),
    [Input('df-loaded-div', 'children'),
     *(Input({'type': 'scatter_'+d, 'index': MATCH}, "value") for d in (list(scatter_graph_dimensions) + ['regression']))]
)
def make_scatter_figure(df_loaded, x, y, color=None, facet_col=None, facet_row=None, regression_degree=None):
    dff = load_feather(df_loaded)
    print('make_scatter_figure', x, y, color, facet_col, facet_row, regression_degree)
    ctx = dash.callback_context
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered
    }, indent=2)
    print(ctx_msg)

    facet_row_cats = dff[facet_row].unique() if facet_row is not None else [None]
    facet_col_cats = dff[facet_col].unique() if facet_col is not None else [None]

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or x is None or y is None:
        return px.scatter()

    fig = make_subplots(len(facet_row_cats), len(facet_col_cats))
    # If color is not provided, then use default
    if color is None:
        color_to_use = default_marker_color
    # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
    else:
        dff.dropna(inplace=True, subset=[color])
        color_to_use = pd.DataFrame(dff[color])

        if dff[color].dtype == pd.CategoricalDtype:
            dff['color_to_use'] = map_color(dff[color])
        else:
            color_to_use.set_index(dff.index, inplace=True)
            dff['color_to_use'] = color_to_use
    for i in range(len(facet_row_cats)):
        for j in range(len(facet_col_cats)):
            working_dff = dff
            working_dff = filter_facet(working_dff, facet_row, facet_row_cats, i)
            working_dff = filter_facet(working_dff, facet_col, facet_col_cats, j)
            fig.add_trace(go.Scatter(x=working_dff[x],
                                     y=working_dff[y],
                                     mode='markers',
                                     marker=dict(color=working_dff[
                                         'color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                 coloraxis="coloraxis",
                                                 showscale=True),
                                     # hovertemplate=['SUBJECT_ID: ' + str(i) + ', DATASET_ID: ' + str(j) +
                                     #                '<extra></extra>'
                                     #                for i, j in zip(working_dff.index, working_dff.DATASET_ID)],
            ),
                          i + 1,
                          j + 1)

            # Add regression lines
            print('regression_degree', regression_degree)
            if regression_degree is not None:
                working_dff.dropna(inplace=True)
                # Guard against fitting an empty graph
                if len(working_dff) > 0:
                    working_dff.sort_values(by=x, inplace=True)
                    Y = working_dff[y]
                    X = working_dff[x]
                    model = Pipeline([('poly', PolynomialFeatures(degree=regression_degree)),
                                      ('linear', LinearRegression(fit_intercept=False))])
                    reg = model.fit(np.vstack(X), Y)
                    Y_pred = reg.predict(np.vstack(X))
                    fig.add_trace(go.Scatter(name='line of best fit', x=X, y=Y_pred, mode='lines'))

    fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, )
    fig.update_xaxes(matches='x')
    fig.update_yaxes(matches='y')
    return fig


@app.callback(
    [Output({'type': 'div_bar_'+str(t), 'index': MATCH}, 'children')
     for t in bar_graph_dimensions],
    [Input('df-loaded-div', 'children')],
    list(itertools.chain.from_iterable([State({'type': 'bar_'+t, 'index': MATCH}, 'id'),
                                        State({'type': 'bar_'+t, 'index': MATCH}, 'value')
                                        ] for t in bar_graph_dimensions))
    + [State({'type': 'div_bar_x', 'index': MATCH}, 'style')]
)
def update_bar_select_columns(df_loaded, x, xv, split_by, split_byv,
                              style_dict):
    print('update_bar_select_columns')
    print(x, xv, split_by, split_byv)
    print('style_dict')
    print(style_dict)
    ctx = dash.callback_context
    ctx_msg = json.dumps(
        {
            'states': ctx.states,
            'triggered': ctx.triggered
        },
        indent=2)
    print(ctx_msg)
    dff = load_feather(df_loaded)
    options = [{'label': col,
                'value': col} for col in dff.columns]

    return ["x:", dcc.Dropdown(id={'type': 'bar_x', 'index': x['index']}, options=options, value=xv)], \
           ["split by:", dcc.Dropdown(id={'type': 'bar_split_by', 'index': split_by['index']}, options=options, value=split_byv)],


@app.callback(
    Output({'type': 'gen_bar_graph', 'index': MATCH}, "figure"),
    [Input('df-loaded-div', 'children'),
     *(Input({'type': 'bar_'+d, 'index': MATCH}, "value") for d in bar_graph_dimensions)]
)
def make_bar_figure(df_loaded, x, split_by=None):
    dff = load_feather(df_loaded)
    print('make_bar_figure', x, split_by)
    ctx = dash.callback_context
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered
    }, indent=2)
    print(ctx_msg)

    # Return empty scatter if not enough options are selected, or the data is empty.
    if dff.columns.size == 0 or x is None:
        return go.Figure(go.Bar())

    fig = go.Figure()
    # If color is not provided, then use default
    # if color is None:
    #     color_to_use = default_marker_color
    # # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
    # else:
    #     dff.dropna(inplace=True, subset=[color])
    #     color_to_use = pd.DataFrame(dff[color])
    #
    #     if dff[color].dtype == pd.CategoricalDtype:
    #         dff['color_to_use'] = map_color(dff[color])
    #     else:
    #         color_to_use.set_index(dff.index, inplace=True)
    #         dff['color_to_use'] = color_to_use
    if split_by is not None:
        # get all unique values in split_by column. Filter by each of these, and add a go.Bar for each, and use these as names.
        split_by_names = sorted(dff[split_by].dropna().unique())

        for name in split_by_names:
            count_by_value = dff[dff[split_by] == name][x].value_counts()

            fig.add_trace(go.Bar(name=name, x=count_by_value.index, y=count_by_value.values))
    else:

        count_by_value = dff[x].value_counts()

        fig.add_trace(go.Bar(name=str(x), x=count_by_value.index, y=count_by_value.values))
    fig.update_layout(coloraxis=dict(colorscale='Bluered_r'))

    return fig

# @app.callback(
#     Output('box-div', 'children'),
#     [Input('df-loaded-div', 'children')])
# # Standard dropdown to select column of interest
# def update_box_columns(df_loaded):
#     print('update_box_columns')
#     dff = load_feather(df_loaded)
#     options = [{'label': col,
#                 'value': col} for col in dff.columns]
#     return html.Div(["Select variable:", dcc.Dropdown(id='box-dropdown', options=options)])
#
#
# @app.callback(
#     Output("box-plot", "figure"),
#     [Input('df-loaded-div', 'children'),
#      Input('box-dropdown', 'value')])
# def make_boxplot(df_loaded, x):
#     print('make_boxplot', x)
#     dff = load_feather(df_loaded)
#
#     if x is not None:
#         return go.Figure(go.Box(x=dff[x].dropna(), boxpoints="all"))
#
#     return go.Figure()
#
#
# @app.callback(
#     Output('bar-div', 'children'),
#     [Input('df-loaded-div', 'children')])
# # Standard dropdown to select column of interest
# def update_bar_columns(df_loaded):
#     print('update_bar_columns')
#     dff = load_feather(df_loaded)
#     options = [{'label': col,
#                 'value': col} for col in dff.columns]
#     return html.Div(["Select variable:", dcc.Dropdown(id='bar-dropdown', options=options)])
#
#
# @app.callback(
#     Output("bar-chart", "figure"),
#     [Input('df-loaded-div', 'children'),
#      Input('bar-dropdown', 'value')])
# def make_barchart(df_loaded, x):
#     print('make_barchart', x)
#     dff = load_feather(df_loaded)
#
#     # Manually group by the column selected - TODO is there an easier way to do this?
#     if x is not None:
#         grouped_df = dff[x].to_frame(0).groupby(0)
#
#         return go.Figure(data=go.Bar(x=[key for key, _ in grouped_df],
#                                      y=[item.size for key, item in grouped_df]))
#
#     return go.Figure(data=go.Bar())


def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        return dff[dff[facet] == facet_cats[i]]

    return dff


def map_color(dff):
    values = sorted(dff.unique())
    all_values = pd.Series(list(map(lambda x: values.index(x), dff)), index=dff.index)
    if all([value == 0 for value in all_values]):
        all_values = pd.Series([1 for _ in all_values])
    return all_values


def standardise_subjectkey(subjectkey):
    if subjectkey[4] == "_":
        return subjectkey

    return subjectkey[0:4]+"_"+subjectkey[4:]


def filename_and_number_to_abbr(filename, number):
    """
    Converts a given filename and filenumber (its position in the list of filenames) into a unique abbreviation for
    use in column names etc.
    """
    return 'F' + str(number)


@app.callback(
    [Output(component_id='df-loaded-div', component_property='children'),
     Output(component_id='filenames-div', component_property='children')],
    [Input(component_id='file-dropdown', component_property='value')]
)
def update_df_loaded_div(input_value):

    print('update_df_loaded_div', input_value)
    dfs = []
    # Read in files based upon their extension - assume .txt is a whitespace-delimited csv.
    for filename in input_value:

        if filename.endswith('.xlsx'):

            dfs.append(pd.read_excel(filename))

        elif filename.endswith('.txt'):

            dfs.append(pd.read_csv(filename, delim_whitespace=True))

    for df in dfs:

        # Reformat SUBJECTKEY if it doesn't have the underscore
        # TODO: remove this when unnecessary
        df['SUBJECTKEY'] = df['SUBJECTKEY'].apply(standardise_subjectkey)

        # Set certain columns to have more specific types.
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].astype('category')

        for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
            if column in df.columns:
                df[column] = df[column].astype('string')

        # Set SUBJECTKEY, EVENTNAME as MultiIndex in all dfs
        df.set_index(indices, inplace=True, verify_integrity=True, drop=True)

    # Join all DFs on their index (that is, SUBJECTKEY), and suffix columns by the filename (if there's
    # more than one file)
    if len(dfs) >= 1:

        total_df = dfs[0]

        if len(dfs) > 1:
            total_df = total_df.add_suffix('@'+filename_and_number_to_abbr(input_value[0], 0))

            for (df_number, (df, df_name)) in enumerate(zip(dfs[1:], input_value[1:]), start=1):
                total_df = total_df.join(df.add_suffix('@'+filename_and_number_to_abbr(df_name, df_number)),
                                         how='outer')
        # Fill df.feather with the combined DF, and set df-loaded-div to True

        total_df.reset_index().to_feather('df.feather')
        return True, \
            [html.H3('File abbreviations:')] + \
            [dash_table.DataTable(id='abbreviations-table',
                                  columns=[{"name": i, "id": i} for i in ['Filename', 'Abbreviation']],
                                  data=[{'Filename': filename,
                                         'Abbreviation': filename_and_number_to_abbr(filename, i)
                                         }
                                        for i, filename in enumerate(input_value)]
                                  )
             ]

    # If no DFs are selected, then fill df.feather with an empty DF, and set df-loaded-div to False
    pd.DataFrame().reset_index().to_feather('df.feather')
    return False, html.Div()


if __name__ == '__main__':
    app.run_server(debug=True)
