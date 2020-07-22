import glob
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from psych_dashboard import preview_table, summary, single_scatter, exploratory_graph_groups
from psych_dashboard.exploratory_graphs import scatter_graph, bar_graph
from psych_dashboard.app import app, indices


global_width = '100%'
data_file_extensions = ['*.txt', '*.xlsx']
filter_file_extensions = ['*.filter']
header_image = '/assets/UoN_Primary_Logo_RGB.png'

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_header(title):
    return html.Div(id='app-header', children=[
            html.A(
                id='drs-link', children=[
                    html.H1('DRS |'),
                ],
                href="https://digitalresearch.nottingham.ac.uk/",
                style={'display': 'inline-block',
                       'margin-left': '10px',
                       'margin-right': '10px',
                       'text-decoration': 'none',
                       'align': 'center'
                       }
            ),
            html.H2(
                title,
                style={'display': 'inline-block', }
            ),
            html.A(
                id='uni-link', children=[
                    html.H1('UoN'),
                ],
                href="https://www.nottingham.ac.uk/",
                style={'display': 'inline-block',
                       'float': 'right',
                       'margin-right': '10px',
                       'text-decoration': 'none',
                       'color': '#FFFFFF',
                       }
            ),
        ],
        style={'width': '100%', 'display': 'inline-block', 'color': 'white'}
        )


app.layout = html.Div(children=[
    html.Img(src=header_image, height=100),
    create_header('ABCD data exploration dashboard'),

    html.H1(children="File selection"),

    html.Label(children='Data File Selection'),
    dcc.Dropdown(
        id='data-file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in
                 [f for f_ in [glob.glob('../data/'+e) for e in data_file_extensions] for f in f_]
                 ],
        value=None,
        placeholder='Select data file'
    ),
    dcc.Loading(
        id='loading-filenames-div',
        children=[html.Div(id='filenames-div')],
    ),
    html.Label(children='Column-Filter File Selection'),
    dcc.Dropdown(
        id='column-filter-file-dropdown',
        options=[{'label': filename,
                  'value': filename} for filename in
                 [f for f_ in [glob.glob('../data/'+e) for e in filter_file_extensions] for f in f_]
                 ],
        value=None,
        placeholder='Leave blank to analyse all columns'
    ),
    dcc.Loading(
        id='loading-column-filter-filenames-div',
        children=[html.Div(id='column-filter-filenames-div')],
    ),
    html.Button('Load selected files', id='load-files-button'),
    html.Div([
        html.H1('Summary', style={'display': 'inline-block'}),
        dbc.Button(
            "-",
            id="collapse-summary-button",
            style={'display': 'inline-block', 'margin-left': '10px', 'width': '40px'}
        ),
        ],
    ),
    dbc.Collapse(id='summary-collapse',
                 children=[
                     html.H2(children="Table Preview"),
                     dcc.Loading(
                         id='loading-table-preview',
                         children=[
                             html.Div(id='table_preview',
                                      style={'width': global_width})
                         ]
                     ),
                     html.H3(children="Table Summary and Filter"),
                     html.Div('\nFilter out all columns missing at least X percentage of rows:'),
                     dcc.Input(id='missing-values-input',
                               type="number",
                               min=0,
                               max=100,
                               debounce=True,
                               value=None),
                     dcc.Loading(
                         id='loading-table-summary',
                         children=[
                             html.Div(id='table_summary',
                                      style={'width': global_width})
                             ]
                     ),
                     html.H2(children='Correlation Heatmap'),
                     html.Div(id='heatmap-div',
                              children=[html.Div(["Select (numerical) variables to display:",
                                                  dcc.Dropdown(id='heatmap-dropdown',
                                                               options=([]),
                                                               multi=True,
                                                               style={'height': '100px', 'overflowY': 'auto'}
                                                               )
                                                  ]
                                                 )
                                        ]
                              ),
                     dcc.Loading(
                         id='loading-heatmap',
                         children=[
                             dcc.Graph(id='heatmap',
                                       figure=go.Figure()
                                       )
                             ]
                     ),
                     html.H2('Manhattan Plot'),
                     dcc.Loading(
                         id='loading-manhattan-dd',
                         children=[dcc.Dropdown(id='manhattan-dd',
                                                multi=True,
                                                style={'display': 'inline-block', 'width': '80%'}),
                                   dcc.Checklist(id='manhattan-all-values-check',
                                                 options=[
                                                     {'label': 'select all', 'value': 'all'}
                                                 ],
                                                 value=['all'],
                                                 style={'display': 'inline-block', 'width': '10%'})
                                   ]
                     ),
                     dcc.Loading(
                         id='loading-manhattan-figure',
                         children=[
                             dcc.Input(id='manhattan-pval-input',
                                       type='number',
                                       value=0.05,
                                       step=0.0001,
                                       debounce=True,
                                       style={'display': 'inline-block'}),
                             dcc.Checklist(
                                 id='manhattan-logscale-check',
                                 options=[
                                     {'label': 'logscale y-axis', 'value': 'LOG'}
                                 ],
                                 value=[],
                                 style={'display': 'inline-block'}
                             ),
                             dcc.Graph(id='manhattan-figure',
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
                     ],
                 is_open=True
                 ),


    # Hidden div for holding the boolean identifying whether a DF is loaded
    html.Div(id='df-loaded-div', style={'display': 'none'}, children=[]),
    html.Div(id='df-filtered-loaded-div', style={'display': 'none'}, children=[]),
    html.Div(id='corr-loaded-div', style={'display': 'none'}, children=[]),
    html.Div(id='pval-loaded-div', style={'display': 'none'}, children=[]),

    # html.H1('Exploratory graphs', style={'margin-top': '10px', 'margin-bottom': '10px'}),
    html.Div([
        html.H1('Exploratory graphs', style={'display': 'inline-block', 'margin-top': '10px', 'margin-bottom': '10px'}),
        dbc.Button(
            "-",
            id="collapse-explore-button",
            style={'display': 'inline-block', 'margin-left': '10px', 'width': '40px'}
        ),
        ],
    ),
    dbc.Collapse(id='explore-collapse',
                 children=[
                     # Container to hold all the exploratory graphs
                     html.Div(id='graph-group-container', children=[]),
                     # Button at the page bottom to add a new graph
                     html.Button('New Graph', id='add-graph-button', style={'margin-top': '10px'})
                     ],
                 is_open=True)
])


@app.callback(
    [Output("summary-collapse", "is_open"),
     Output("collapse-summary-button", "children")],
    [Input("collapse-summary-button", "n_clicks")],
    [State("summary-collapse", "is_open")],
)
def toggle_collapse_summary(n, is_open):
    print('toggle_collapse_summary', n, is_open)
    if n:
        return not is_open, "+" if is_open else "-"
    return is_open, "-"


@app.callback(
    [Output("explore-collapse", "is_open"),
     Output("collapse-explore-button", "children")],
    [Input("collapse-explore-button", "n_clicks")],
    [State("explore-collapse", "is_open")],
)
def toggle_collapse_explore(n, is_open):
    print('toggle_collapse_explore', n, is_open)
    if n:
        return not is_open, "+" if is_open else "-"
    return is_open, "-"


def standardise_subjectkey(subjectkey):
    if subjectkey[4] == "_":
        return subjectkey

    return subjectkey[0:4]+"_"+subjectkey[4:]


@app.callback(
    [Output('df-loaded-div', 'children')],
    [Input('load-files-button', 'n_clicks')],
    [State('data-file-dropdown', 'value'),
     State('column-filter-file-dropdown', 'value')]
)
def update_df_loaded_div(n_clicks, data_file_value, filter_file_value):

    print('update_df_loaded_div', data_file_value, filter_file_value)
    # Return early if no data file selected, and fill df.feather with an empty DF, and set df-loaded-div to [False]
    if data_file_value is None:
        pd.DataFrame().reset_index().to_feather('df.feather')
        return [False]

    # Read in data file based upon the extension - assume .txt is a whitespace-delimited csv.
    if data_file_value.endswith('.xlsx'):
        df = pd.read_excel(data_file_value)
    elif data_file_value.endswith('.txt'):
        df = pd.read_csv(data_file_value, delim_whitespace=True)
    else:
        raise FileNotFoundError(data_file_value)

    # If filter file is selected, read in filter file, and add the index names if they are not present. Otherwise,
    # do no filtering
    if filter_file_value:
        with open(filter_file_value) as f:
            variables_of_interest = f.read().splitlines()
            print(variables_of_interest)
            # Add index names if they are not present
            variables_of_interest.extend(index for index in indices if index not in variables_of_interest)

        # Verify that the variables of interest exist.
        missing_vars = [var for var in variables_of_interest if var not in df.columns]

        if missing_vars:
            raise ValueError(str(missing_vars) + ' is in the filter file but not found in the data file.')

        # Keep only the columns listed in the filter file
        df = df[variables_of_interest]

    # Reformat SUBJECTKEY if it doesn't have the underscore
    # TODO: remove this when unnecessary
    df['SUBJECTKEY'] = df['SUBJECTKEY'].apply(standardise_subjectkey)

    # Set certain columns to have more specific types.
    if 'SEX' in df.columns:
        df['SEX'] = df['SEX'].astype('category')

    for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
        if column in df.columns:
            df[column] = df[column].astype('string')

    # Set SUBJECTKEY, EVENTNAME as MultiIndex
    df.set_index(indices, inplace=True, verify_integrity=True, drop=True)

    # Fill df.feather with the combined DF, and set df-loaded-div to [True]
    df.reset_index().to_feather('df.feather')

    return [True]


if __name__ == '__main__':
    pd.DataFrame().reset_index().to_feather('df.feather')
    pd.DataFrame().reset_index().to_feather('df_filtered.feather')
    app.run_server(debug=True)
