import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os
import json

graph_dimensions = ["x", "y", "color", "facet_col", "facet_row"]
global_width = '100%'

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config['suppress_callback_exceptions'] = True

style_dict = {
        'width': '15%',
        'display': 'inline-block',
        'verticalAlign': "middle",
        'margin-right': '2em',
    }

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for Python.
    '''),

    html.Label('Multi-Select Dropdown'),
    dcc.Dropdown(
        id='file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in os.listdir('../data')],
        value=[],
        multi=True
    ),

    dcc.Dropdown(
        id='column-dropdown',
        options=[{'label': 'hello',
                  'value': 'placeholder'}],
        value=[]
    ),
    # Hidden div for holding the jsonised combined DF in use
    html.Div(id='my-div', style={'display': 'none'}, children=[]),

    # dcc.Graph(
    #     id='example-graph',
    #     figure={
    #         'data': [
    #             {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
    #             {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    #         ],
    #         'layout': {
    #             'title': 'Dash Data Visualization'
    #         }
    #     }
    # ),
    # Div holds the graph selection elements
    html.Div(id='graph1-selection',
             children=[html.Div([d + ":", dcc.Dropdown(id=d, options=[])], style=style_dict) for d in graph_dimensions]),

    dcc.Graph(id="graph1", style={"width": global_width}),

])





@app.callback(
    Output("graph1", "figure"),
    [Input('my-div', 'children'),
     *(Input(d, "value") for d in graph_dimensions)])
def make_figure(input_json_df, x, y, color, facet_col, facet_row):
    print('make_figure', x, y, color, facet_col, facet_row)
    dff = pd.read_json(json.loads(input_json_df), orient='split')
    return px.scatter(
        dff,
        x=x,
        y=y,
        color=color,
        facet_col=facet_col,
        facet_row=facet_row,
        height=700
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

        return json.dumps(total_df.to_json(orient='split', date_format='iso'))

    else:

        return json.dumps(pd.DataFrame().to_json(orient='split', date_format='iso'))


@app.callback(
    Output(component_id='column-dropdown', component_property='options'),
    [Input(component_id='my-div', component_property='children')]
)
def update_available_columns(input_value):
    print('update_available_columns')
    df = pd.read_json(json.loads(input_value), orient='split')
    return [{'label': col,
             'value': col} for col in df.columns]


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
