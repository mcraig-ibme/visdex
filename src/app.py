import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import glob
import json

graph_dimensions = ["x", "y", "color", "facet_col", "facet_row"]
global_width = '100%'
default_marker_color = "crimson"
file_extensions=['*.txt','*.xlsx']

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

style_dict = {
        'width': '15%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }

test_df = pd.read_excel('../data/ABCD_PSB01.xlsx', delim_whitespace=True).head(n=10)

app.layout = html.Div(children=[
    html.H1(children='ABCD data exploration dashboard'),

    html.Div(children='''
        Select 1+ files, and then plot scatter plots!
    '''),

    html.Label('Files Multi-Select'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in [f for f_ in [glob.glob('../data/'+e) for e in file_extensions] for f in f_]],
        value=[],
        multi=True
    ),

    html.Div(id='table_preview',
                 style={'width': global_width}),
    # dcc.Dropdown(
    #     id='column-dropdown',
    #     options=[{'label': 'hello',
    #               'value': 'placeholder'}],
    #     value=[]
    # ),
    # Hidden div for holding the jsonised combined DF in use
    html.Div(id='my-div', style={'display': 'none'}, children=[]),

    # dcc.Graph(
    #     id='example-graph',
    #     figure = go.Figure(data=go.Scatter(x=test_df['PROSOCIAL_Q1_Y'],
    #                         y=test_df['PROSOCIAL_Q2_Y'],
    #                         mode='markers',
    #                         marker=dict(color=test_df['PROSOCIAL_Q3_Y'],
    #                                     coloraxis="coloraxis",
    #                                     showscale=True
    #                                     )
    #                         )),
        # fig.update_layout(coloraxis=dict(colorscale='Bluered_r'))
        # fig.update_xaxes(matches='x')
        # figure={
        #     'data': [
        #         {'x': test_df['PROSOCIAL_Q1_Y'], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
        #         # {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
        #     ],
        #     'layout': {
        #         'title': 'Dash Data Visualization'
        #     }
        # }
    # ),
    # Div holds the graph selection elements
    html.Div(id='graph1-selection',
             children=[html.Div([d + ":", dcc.Dropdown(id=d, options=[])], style=style_dict) for d in graph_dimensions]),

    dcc.Graph(id="graph1", style={"width": global_width}),

])


def filter_facet(dff, facet, facet_cats, i):
    if facet is not None:
        # if not isinstance(color, str):
        #     print('dff', dff)
        #     print('color', color)
        #     print('facet', facet)
        #     return dff[dff[facet] == facet_cats[i]], color[dff[facet] == facet_cats[i]]
        # else:
        return dff[dff[facet] == facet_cats[i]]
    else:
        return dff


def map_color(dff):
    values = sorted(dff.unique())
    all_values = pd.Series(list(map(lambda x: values.index(x), dff)), index=dff.index)
    print(dff)
    print(all_values)
    if all([value == 0 for value in all_values]):
        all_values = pd.Series([1 for _ in all_values])
    return all_values


@app.callback(
    Output("graph1", "figure"),
    [Input('my-div', 'children'),
     *(Input(d, "value") for d in graph_dimensions)])
def make_figure(input_json_df, x, y, color=None, facet_col=None, facet_row=None):
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    print('make_figure', x, y, color, facet_col, facet_row)
    # return go.Figure(data=go.Scatter(x=dff[x], y=dff[y], color=color, facet_row=facet_row, facet_col=facet_col, mode='markers'))
    # print('dffx', dff[x])
    # print('dffy', dff[y])
    print(x)
    print(y)
    print(dff.dropna())
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
            # dff.dropna(inplace=True, subset=[color])
            print('color_to_use', color)
            color_to_use = pd.DataFrame(dff[color])
            print('color_to_use2', color)

            if dff[color].dtype == pd.CategoricalDtype:
                print("category detected")
                dff['color_to_use'] = map_color(dff[color])
                print('color_to_use3', color_to_use)
                print(dff, dff['color_to_use'])
            else:
                color_to_use.set_index(dff.index, inplace=True)
                dff['color_to_use'] = color_to_use
        for i in range(len(facet_row_cats)):
            for j in range(len(facet_col_cats)):
                working_dff = dff
                working_dff = filter_facet(working_dff, facet_row, facet_row_cats, i)
                working_dff = filter_facet(working_dff, facet_col, facet_col_cats, j)
                print(working_dff.columns)
                print('color_to_use4', working_dff['color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use)
                print()
                # print(working_dff['color_to_use'])
                # print(color_to_use)
                print('dffx', working_dff[x])
                print('dffy', working_dff[y])
                fig.add_trace(go.Scatter(x=working_dff[x],
                                         y=working_dff[y],
                                         mode='markers',
                                         marker=dict(color=working_dff['color_to_use'] if 'color_to_use' in working_dff.columns else color_to_use,
                                                     coloraxis="coloraxis",
                                                     showscale=True)),
                              i + 1,
                              j + 1)
        fig.update_layout(coloraxis=dict(colorscale='Bluered_r'), showlegend=False, )
        fig.update_xaxes(matches='x')
        fig.update_yaxes(matches='y')
        # fig.show()
        # return px.scatter(
        #     x=dff.dropna()[x],
        #     y=dff.dropna()[y],
        #     color=dff.dropna()[color] if color is not None else None,
        #     # facet_col=facet_col,
        #     # facet_row=facet_row,
        #     height=700
        # )
        return fig
    else:
        return px.scatter()


@app.callback(
    Output('table_preview', 'children'),
    [Input('my-div', 'children')])
def update_preview_table(input_json_df):
    print('update_preview_table')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    # subset_df = dff[row_value[0]:row_value[1]]
    # if set(col_value).intersection(set(dff.columns)):
    #     subset_df = subset_df[col_value]
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
    [dash.dependencies.Input('my-div', 'children')])
def update_select_columns(input_json_df):
    print('update_select_columns')
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    options = [{'label': col,
                'value': col} for col in dff.columns]
    style_dict = {
        'width':'15%',
        'display':'inline-block',
        'verticalAlign':"middle",
        'margin-right':'2em',
    }
    return html.Div([
        html.Div([d + ":", dcc.Dropdown(id=d, options=options)], style=style_dict) for d in graph_dimensions],
        )

def standardise_subjectkey(subjectkey):
    if subjectkey[4] == "_":
        return subjectkey
    else:
        return subjectkey[0:4]+"_"+subjectkey[4:]

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='file-dropdown', component_property='value')]
)
def update_output_div(input_value):

    print('update_output_div', input_value)
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
        print(total_df.dtypes)
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
