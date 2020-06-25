import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import glob
import json

graph_dimensions = {"x": "x", "y": "y", "color": "color (will drop NAs)", "facet_col": "split horizontally", "facet_row": "split vertically"}
global_width = '100%'
default_marker_color = "crimson"
file_extensions = ['*.txt','*.xlsx']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

style_dict = {
        'width': '13%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }

test_df = pd.read_excel('../data/ABCD_PSB01.xlsx', delim_whitespace=True).head(n=10)


def generate_graph_button(graph_no):
    return html.Button('Show Graph '+str(graph_no), id='add-graph-button'+str(graph_no))


app.layout = html.Div(children=[
    html.H1(children='ABCD data exploration dashboard'),

    html.H2(children="File selection"),

    html.Label(children='Files Multi-Select'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in [f for f_ in [glob.glob('../data/'+e) for e in file_extensions] for f in f_]],
        value=[],
        multi=True
    ),
    html.H2(children="Table Preview"),
    html.Div(id='table_preview',
             style={'width': global_width}),
    html.H2(children="Table Summary"),
    html.Div(id='table_summary',
             style={'width': global_width}),
    # dcc.Dropdown(
    #     id='column-dropdown',
    #     options=[{'label': 'hello',
    #               'value': 'placeholder'}],
    #     value=[]
    # ),
    # Hidden div for holding the jsonised combined DF in use
    html.Div(id='json-df-div', style={'display': 'none'}, children=[]),

    html.H2(children="Scatter"),
    # Div holds the graph selection elements
    html.Div(id='graph1-selection',
             children=[html.Div([value + ":", dcc.Dropdown(id=key, options=[])], style=style_dict) for key, value in graph_dimensions.items()]
             + [html.Div(["regression:", dcc.Input(id="regression",
                                                   type='number',
                                                   min=0,
                                                   step=1
                                                   )
                          ],
                         style=style_dict)
                ]
             ),

    dcc.Graph(id="graph1", style={"width": global_width}),

    html.H2(children="Count per category"),
    html.Div(id='bar-div',
             children=[html.Div(["Select variable to group by:", dcc.Dropdown(id='bar-dropdown', options=([]))])]),
    dcc.Graph(
        id='bar-chart',
        figure=go.Figure(data=go.Bar()),
    ),
    html.H2(children="Boxplot per category"),
    html.Div(id='box-div',
             children=[html.Div(["Select variables to view:", dcc.Dropdown(id='box-dropdown', options=([]))])]),
    dcc.Graph(
        id='box-plot',
        figure=go.Figure(data=go.Bar()),
    ),
    html.Div(id='graph-group-container', children=[]),
    generate_graph_button(1),
    # generate_graph_group('graph-group-1', go.Figure(data=go.Bar()), graph_no=1, style={'display': 'none'}),
])



@app.callback(
    Output('graph-group-container', 'children'),
    [Input('add-graph-button1', 'n_clicks')],
    [State('graph-group-container', 'children')])
def display_graph_groups(n_clicks, children):
    # Add a new graph group each time the button is clicked. The if None guard stops there being an initial graph.
    print('display_graph_groups')
    # print('children0', children)
    if n_clicks is not None:
        new_graph_group = html.Div(id={
                'type': 'filter-graph-group',
                'index': n_clicks
            },
            children=[html.Div([value + ":", dcc.Dropdown(id={'type': key, 'index': n_clicks}, options=[])],
                               id={'type': 'div'+str(key), 'index': n_clicks},
                               style=style_dict
                              )
                      for key, value in graph_dimensions.items()
                      ]
                      + [html.Div(["regression:", dcc.Input(id={'type': 'regression', 'index': n_clicks},
                                                            type='number',
                                                            min=0,
                                                            step=1,
                                                            )],
                                  id={'type': 'divregression', 'index': n_clicks},
                                  style=style_dict)]
                      + [dcc.Graph(id={'type': 'gen_graph', 'index': n_clicks},
                         figure=go.Figure(data=go.Bar()))]
        )
        children.append(new_graph_group)
        # print('children', children)
    return children


@app.callback(
    [Output({'type': 'divx', 'index': MATCH}, 'children'),
     Output({'type': 'divy', 'index': MATCH}, 'children'),
     Output({'type': 'divcolor', 'index': MATCH}, 'children'),
     Output({'type': 'divfacet_col', 'index': MATCH}, 'children'),
     Output({'type': 'divfacet_row', 'index': MATCH}, 'children'),
     Output({'type': 'divregression', 'index': MATCH}, 'children')],
    [Input('json-df-div', 'children')],
    [State({'type': 'x', 'index': MATCH}, 'id'),
     State({'type': 'x', 'index': MATCH}, 'value'),
     State({'type': 'y', 'index': MATCH}, 'id'),
     State({'type': 'y', 'index': MATCH}, 'value'),
     State({'type': 'color', 'index': MATCH}, 'id'),
     State({'type': 'color', 'index': MATCH}, 'value'),
     State({'type': 'facet_col', 'index': MATCH}, 'id'),
     State({'type': 'facet_col', 'index': MATCH}, 'value'),
     State({'type': 'facet_row', 'index': MATCH}, 'id'),
     State({'type': 'facet_row', 'index': MATCH}, 'value'),
     State({'type': 'regression', 'index': MATCH}, 'id'),
     State({'type': 'regression', 'index': MATCH}, 'value')
     ],
)
def update_any_select_columns(input_json_df, x, xv, y, yv, color, colorv, facet_col, fcv, facet_row, frv, regression, regv):
    print('update_any_select_columns')
    print(x, xv, y, yv, color, colorv, facet_col, fcv, facet_row, frv, regression, regv)
    ctx = dash.callback_context
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered
    }, indent=2)
    print(ctx_msg)
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns]
    # print(options)
    style_dict = {
        'width':'15%',
        'display':'inline-block',
        'verticalAlign':"middle",
        'margin-right':'2em',
    }
    return ["x:", dcc.Dropdown(id={'type': 'x', 'index': x['index']}, options=options, value=xv)], \
           ["y:", dcc.Dropdown(id={'type': 'y', 'index': y['index']}, options=options, value=yv)], \
           ["color:", dcc.Dropdown(id={'type': 'color', 'index': color['index']}, options=options, value=colorv)], \
           ["split_horizontally:", dcc.Dropdown(id={'type': 'facet_col', 'index': facet_col['index']}, options=options, value=fcv)], \
           ["split_vertically:", dcc.Dropdown(id={'type': 'facet_row', 'index': facet_row['index']}, options=options, value=frv)], \
           ["regression degree:", dcc.Input(id={'type': 'regression', 'index': regression['index']},
                                            type='number',
                                            min=0,
                                            step=1,
                                            value=regv)]


@app.callback(
    Output({'type': 'gen_graph', 'index': MATCH}, "figure"),
    [Input('json-df-div', 'children'),
     *(Input({'type': d, 'index': MATCH}, "value") for d in graph_dimensions),
     Input({'type': 'regression', 'index': MATCH}, "value")])
def make_any_figure(input_json_df, x, y, color=None, facet_col=None, facet_row=None, regression_degree=None):
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    print('make_any_figure', x, y, color, facet_col, facet_row, regression_degree)
    ctx = dash.callback_context
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered
    }, indent=2)
    print(ctx_msg)
    if facet_row is not None:
        facet_row_cats = dff[facet_row].unique()
    else:
        facet_row_cats = [None]
    if facet_col is not None:
        facet_col_cats = dff[facet_col].unique()
    else:
        facet_col_cats = [None]
    if dff.columns.size > 0 and x is not None and y is not None:
        fig = make_subplots(len(facet_row_cats), len(facet_col_cats))
        # If color is not provided, then use default
        if color is None:
            color_to_use = default_marker_color
        # Otherwise, if the dtype is categorical, then we need to map - otherwise, leave as it is
        else:
            dff.dropna(inplace=True, subset=[color])
            # print('color_to_use', color)
            color_to_use = pd.DataFrame(dff[color])
            # print('color_to_use2', color)

            if dff[color].dtype == pd.CategoricalDtype:
                # print("category detected")
                dff['color_to_use'] = map_color(dff[color])
                # print('color_to_use3', color_to_use)
                # print(dff, dff['color_to_use'])
            else:
                color_to_use.set_index(dff.index, inplace=True)
                dff['color_to_use'] = color_to_use
        for i in range(len(facet_row_cats)):
            for j in range(len(facet_col_cats)):
                working_dff = dff
                working_dff = filter_facet(working_dff, facet_row, facet_row_cats, i)
                working_dff = filter_facet(working_dff, facet_col, facet_col_cats, j)
                # print(working_dff.columns)
                # print('color_to_use4',
                #       working_dff['color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use)
                # print()
                # print(working_dff['color_to_use'])
                # print(color_to_use)
                # print('dffx', working_dff[x])
                # print('dffy', working_dff[y])
                fig.add_trace(go.Scatter(x=working_dff[x],
                                         y=working_dff[y],
                                         mode='markers',
                                         marker=dict(color=working_dff[
                                             'color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                     coloraxis="coloraxis",
                                                     showscale=True)),
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
    else:
        return px.scatter()


@app.callback(
    Output('box-div', 'children'),
    [Input('json-df-div', 'children')])
# Standard dropdown to select column of interest
def update_box_columns(input_json_df):
    print('update_box_columns')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns]
    return html.Div(["Select variable:", dcc.Dropdown(id='box-dropdown', options=options)])


@app.callback(
    Output("box-plot", "figure"),
    [Input('json-df-div', 'children'),
     Input('box-dropdown', 'value')])
def make_boxplot(input_json_df, x):
    print('make_boxplot', x)
    dff = pd.read_json(json.loads(input_json_df), orient='split')

    if x is not None:
        return go.Figure(go.Box(x=dff[x].dropna(), boxpoints="all"))
    else:
        return go.Figure()


@app.callback(
    Output('bar-div', 'children'),
    [Input('json-df-div', 'children')])
# Standard dropdown to select column of interest
def update_bar_columns(input_json_df):
    print('update_bar_columns')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns]
    return html.Div(["Select variable:", dcc.Dropdown(id='bar-dropdown', options=options)])


@app.callback(
    Output("bar-chart", "figure"),
    [Input('json-df-div', 'children'),
     Input('bar-dropdown', 'value')])
def make_barchart(input_json_df, x):
    print('make_barchart', x)
    dff = pd.read_json(json.loads(input_json_df), orient='split')

    # Manually group by the column selected - TODO is there an easier way to do this?
    if x is not None:
        grouped_df = dff[x].to_frame(0).groupby(0)

        return go.Figure(data=go.Bar(x=[key for key, _ in grouped_df],
                                     y=[item.size for key, item in grouped_df]))
    else:
        return go.Figure(data=go.Bar())


def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        return dff[dff[facet] == facet_cats[i]]
    else:
        return dff


def map_color(dff):
    values = sorted(dff.unique())
    all_values = pd.Series(list(map(lambda x: values.index(x), dff)), index=dff.index)
    if all([value == 0 for value in all_values]):
        all_values = pd.Series([1 for _ in all_values])
    return all_values


@app.callback(
    Output("graph1", "figure"),
    [Input('json-df-div', 'children'),
     *(Input(d, "value") for d in graph_dimensions),
     Input('regression', "value")])
def make_scatter(input_json_df, x, y, color=None, facet_col=None, facet_row=None, regression_degree=None):
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    print('make_scatter', x, y, color, facet_col, facet_row)
    if facet_row is not None:
        facet_row_cats = dff[facet_row].unique()
    else:
        facet_row_cats = [None]
    if facet_col is not None:
        facet_col_cats = dff[facet_col].unique()
    else:
        facet_col_cats = [None]
    if dff.columns.size > 0 and x is not None and y is not None:
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
                                         marker=dict(color=working_dff['color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                     coloraxis="coloraxis",
                                                     showscale=True)),
                              i + 1,
                              j + 1)

                # Add regression lines
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
    else:
        return px.scatter()


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
        return html.Div(dash_table.DataTable(
            id='table',
            columns=[{"name": i.upper(), "id": i, 'type': 'numeric', 'format': {'specifier': '.2f'}} for i in description_df.columns],
            data=description_df.to_dict('record'),
            # Highlight any columns that do not have a complete set of records,
            # by comparing count against the length of the DF.
            style_data_conditional=[{
                'if': {
                    'filter_query': '{{count}} < {}'.format(dff.shape[0]),
                    'column_id': 'count'
                },
                'backgroundColor': 'FireBrick',
                'color': 'white'
                  }
            ]
    ),
    )
    return html.Div()


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


@app.callback(
    dash.dependencies.Output('graph1-selection', 'children'),
    [dash.dependencies.Input('json-df-div', 'children')])
def update_select_columns(input_json_df):
    print('update_select_columns')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns]
    style_dict = {
        'width':'13%',
        'display':'inline-block',
        'verticalAlign':"middle",
        'margin-right':'2em',
    }
    return html.Div([
        html.Div([value + ":", dcc.Dropdown(id=key, options=options)],
                 style=style_dict
                 )
        for key, value in graph_dimensions.items()]
        + [html.Div(["regression:", dcc.Input(id="regression",
                                                            type='number',
                                                            min=0,
                                                            step=1
                                                           )
                     ], style=style_dict)]
        )


def standardise_subjectkey(subjectkey):
    if subjectkey[4] == "_":
        return subjectkey
    else:
        return subjectkey[0:4]+"_"+subjectkey[4:]

@app.callback(
    Output(component_id='json-df-div', component_property='children'),
    [Input(component_id='file-dropdown', component_property='value')]
)
def update_json_div(input_value):

    print('update_json_div', input_value)
    dfs = []
    # Read in files based upon their extension - assume .txt is a whitespace-delimited csv.
    for filename in input_value:

        if filename.endswith('.xlsx'):

            dfs.append(pd.read_excel('../data/'+filename))

        elif filename.endswith('.txt'):

            dfs.append(pd.read_csv('../data/' + filename, delim_whitespace=True))

    for df in dfs:

        # Reformat SUBJECTKEY if it doesn't have the underscore
        df['SUBJECTKEY'] = df['SUBJECTKEY'].apply(standardise_subjectkey)

        # Set SUBJECTKEY as index in all dfs
        df.set_index('SUBJECTKEY', inplace=True, verify_integrity=True)

        # Set certain columns to have more specific types.
        if 'SEX' in df.columns:
            df['SEX'] = df['SEX'].astype('category')

        for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
            if column in df.columns:
                df[column] = df[column].astype('string')

    # Join all DFs on their index (that is, SUBJECTKEY), and suffix columns by the filename (if there's
    # more than one file)
    if len(dfs) >= 1:

        total_df = dfs[0]

        if len(dfs) > 1:
            total_df = total_df.add_suffix('_'+input_value[0])

        for (df, df_name) in zip(dfs[1:], input_value[1:]):
            total_df = total_df.join(df.add_suffix('_'+df_name), how='outer')
        # print(total_df.dtypes)
        return json.dumps(total_df.to_json(orient='split', date_format='iso'))

    else:

        return json.dumps(pd.DataFrame().to_json(orient='split', date_format='iso'))


# @app.callback(
#     Output(component_id='column-dropdown', component_property='options'),
#     [Input(component_id='my-div', component_property='children')]
# )
# def update_available_columns(input_value):
#     print('update_available_columns')
#     df = pd.read_json(json.loads(input_value), orient='split')
#     return [{'label': col,
#              'value': col} for col in df.columns]


# @app.callback(
#     Output(component_id='example-graph', component_property='figure'),
#     [Input(component_id='my-div', component_property='children'),
#      Input(component_id='column-dropdown', component_property='value')]
# )
# def update_graph(input_value, column_value):
#     print('update_graph')
#     # print(input_value)
#     df = pd.read_json(json.loads(input_value), orient='split')
#     if df.columns.size > 0 and column_value != []:
#         if column_value in df.columns:
#             out = {
#                     'data': [
#                         {'x': df.index.tolist(), 'y': df[column_value].tolist(), 'type': 'bar', 'name': 'Tokyo'},
#                         # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#                     ],
#                     'layout': {
#                         'title': column_value
#                     }
#                 }
#             return out
#     return {
#             'data': [
#
#                 # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
#             ],
#             'layout': {
#                 # 'title': 'Dash Data Visualization'
#             }
#         }


if __name__ == '__main__':
    app.run_server(debug=True)
