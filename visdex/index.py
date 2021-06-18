import io
import base64
import datetime
import logging
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from visdex import preview_table, exploratory_graph_groups, export
from visdex.load_feather import store, load
from visdex.exploratory_graphs import (
    scatter_graph,
    bar_graph,
    histogram_graph,
    manhattan_graph,
    violin_graph,
)
from visdex.summary import (
    summary_heatmap,
    summary_kde,
    summary_manhattan,
    summary_table,
)
<<<<<<< HEAD:visdex/index.py
from visdex.app import app, indices, standard_margin_left, div_style
=======
from psych_dashboard.app import app, known_indices, standard_margin_left, div_style
>>>>>>> msc:src/psych_dashboard/index.py

logging.getLogger(__name__)

GLOBAL_WIDTH = "100%"
TABLE_WIDTH = "95%"
HEADER_IMAGE = "/assets/UoN_Primary_Logo_RGB.png"
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def create_header(title):
    return html.Div(
        id="app-header",
        children=[
            html.A(
                id="drs-link",
                children=[
                    html.H1("DRS |"),
                ],
                href="https://digitalresearch.nottingham.ac.uk/",
                style={
                    "display": "inline-block",
                    "margin-left": "10px",
                    "margin-right": "10px",
                    "text-decoration": "none",
                    "align": "center",
                },
            ),
            html.H2(
                title,
                style={
                    "display": "inline-block",
                },
            ),
            html.A(
                id="uni-link",
                children=[
                    html.H1("UoN"),
                ],
                href="https://www.nottingham.ac.uk/",
                style={
                    "display": "inline-block",
                    "float": "right",
                    "margin-right": "10px",
                    "text-decoration": "none",
                    "color": "#FFFFFF",
                },
            ),
        ],
        style={"width": "100%", "display": "inline-block", "color": "white"},
    )


app.layout = html.Div(
    children=[
        html.Img(src=HEADER_IMAGE, height=100),
        create_header("ABCD data exploration dashboard"),
        html.H1(children="File selection", style=div_style),
        html.Label(
            children="Data File Selection (initial data read will happen immediately)",
            style=div_style,
        ),
        dcc.Upload(
            id="data-file-upload",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div(
            id="output-data-file-upload", children=["No file loaded"], style=div_style
        ),
        html.Label(
            children="Column Filter File Selection (initial data read will happen"
                     " immediately)",
            style=div_style,
        ),
        dcc.Upload(
            id="filter-file-upload",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div(
            id="output-filter-file-upload", children=["No file loaded"], style=div_style
        ),
        html.Button("Analyse", id="load-files-button", style=div_style),
        html.Button("Export to PDF", id="export-pdf-button", style=div_style),
        html.Div(children=[""], id="export-div", style=div_style),
        html.Div(
            [
                html.H1(
                    "Summary",
                    style={
                        "display": "inline-block",
                        "margin-left": standard_margin_left,
                    },
                ),
                dbc.Button(
                    "-",
                    id="collapse-summary-button",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "width": "40px",
                    },
                ),
            ],
        ),
        dbc.Collapse(
            id="summary-collapse",
            children=[
                html.H2(children="Table Preview", style=div_style),
                dcc.Loading(
                    id="loading-table-preview",
                    children=[
                        html.Div(
                            id="table_preview",
                            style={
                                "width": TABLE_WIDTH,
                                "margin-left": "10px",
                                "margin-right": "10px",
                            },
                        )
                    ],
                ),
                html.H3(children="Table Summary and Filter", style=div_style),
                html.Div(
                    "\nFilter out all columns missing at least X percentage of rows:",
                    style=div_style,
                ),
                dcc.Input(
                    id="missing-values-input",
                    type="number",
                    min=0,
                    max=100,
                    debounce=True,
                    value=None,
                    style=div_style,
                ),
                dcc.Loading(
                    id="loading-table-summary",
                    children=[
                        html.Div(
                            id="other_summary",
                            style={
                                "width": GLOBAL_WIDTH,
                                "margin-left": "10px",
                                "margin-right": "10px",
                            },
                        ),
                        html.Div(
                            id="table_summary",
                            style={
                                "width": TABLE_WIDTH,
                                "margin-left": "10px",
                                "margin-right": "10px",
                            },
                        ),
                    ],
                ),
                html.H2(children="Correlation Heatmap (Pearson's)", style=div_style),
                html.Div(
                    id="heatmap-div",
                    style=div_style,
                    children=[
                        dcc.Input(
                            id="heatmap-clustering-input",
                            type="number",
                            min=1,
                            debounce=True,
                            value=2,
                        ),
                        # TODO label the cluster input selection
                        html.Div(
                            [
                                "Select (numerical) variables to display:",
                                dcc.Dropdown(
                                    id="heatmap-dropdown",
                                    options=([]),
                                    multi=True,
                                    # style={'height': '100px', 'overflowY': 'auto'}
                                ),
                            ]
                        ),
                    ],
                ),
                dcc.Loading(
                    id="loading-heatmap",
                    children=[dcc.Graph(id="heatmap", figure=go.Figure())],
                ),
                html.H2("Manhattan Plot", style=div_style),
                dcc.Loading(
                    id="loading-manhattan-figure",
                    children=[
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="manhattan-active-check",
                                    options=[
                                        {
                                            "label": " Plot Manhattan",
                                            "value": "manhattan-active",
                                        }
                                    ],
                                    value=[],
                                    style={"display": "inline-block"},
                                ),
                            ]
                        ),
                        html.Div(
                            [
                                "p-value:  ",
                                dcc.Input(
                                    id="manhattan-pval-input",
                                    type="number",
                                    value=0.05,
                                    step=0.0001,
                                    debounce=True,
                                    style={"display": "inline-block"},
                                ),
                            ],
                            style=div_style,
                        ),
                        html.Div(
                            [
                                dcc.Checklist(
                                    id="manhattan-logscale-check",
                                    options=[
                                        {"label": "  logscale y-axis", "value": "LOG"}
                                    ],
                                    value=[],
                                    style={"display": "inline-block"},
                                ),
                            ],
                            style=div_style,
                        ),
                        dcc.Graph(id="manhattan-figure", figure=go.Figure()),
                    ],
                ),
                html.H2(children="Per-variable Histograms and KDEs", style=div_style),
                dcc.Checklist(
                    "kde-checkbox",
                    options=[{"label": " Run KDE analysis", "value": "kde-active"}],
                    value=[],
                    style=div_style,
                ),
                dcc.Loading(
                    id="loading-kde-figure",
                    children=[dcc.Graph(id="kde-figure", figure=go.Figure())],
                ),
            ],
            is_open=True,
        ),
        # Hidden divs for holding the booleans identifying whether a DF is loaded in
        # each case
        html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="df-filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),
        html.Div(
            [
                html.H1(
                    "Exploratory graphs",
                    style={
                        "display": "inline-block",
                        "margin-top": "10px",
                        "margin-bottom": "10px",
                        "margin-left": standard_margin_left,
                    },
                ),
                dbc.Button(
                    "-",
                    id="collapse-explore-button",
                    style={
                        "display": "inline-block",
                        "margin-left": "10px",
                        "width": "40px",
                    },
                ),
            ],
        ),
        dbc.Collapse(
            id="explore-collapse",
            children=[
                # Container to hold all the exploratory graphs
                html.Div(id="graph-group-container", children=[]),
                # Button at the page bottom to add a new graph
                html.Button(
                    "New Graph",
                    id="add-graph-button",
                    style={
                        "margin-top": "10px",
                        "margin-left": standard_margin_left,
                        "margin-bottom": "40px",
                    },
                ),
            ],
            is_open=True,
        ),
    ]
)


@app.callback(
    [
        Output("summary-collapse", "is_open"),
        Output("collapse-summary-button", "children"),
    ],
    [Input("collapse-summary-button", "n_clicks")],
    [State("summary-collapse", "is_open")],
    prevent_initial_call=True,
)
def toggle_collapse_summary(n_clicks, is_open):
    logging.info(f"toggle_collapse_summary {n_clicks} {is_open}")
    if n_clicks:
        return not is_open, "+" if is_open else "-"
    return is_open, "-"


@app.callback(
    [
        Output("explore-collapse", "is_open"),
        Output("collapse-explore-button", "children"),
    ],
    [Input("collapse-explore-button", "n_clicks")],
    [State("explore-collapse", "is_open")],
    prevent_initial_call=True,
)
def toggle_collapse_explore(n, is_open):
    logging.info(f"toggle_collapse_explore {n} {is_open}")
    if n:
        return not is_open, "+" if is_open else "-"
    return is_open, "-"


def standardise_subjectkey(subjectkey):
    if subjectkey[4] == "_":
        return subjectkey

    return subjectkey[0:4] + "_" + subjectkey[4:]


@app.callback(
    [Output("output-data-file-upload", "children")],
    [Input("data-file-upload", "contents")],
    [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
    prevent_initial_call=True,
)
# This function is triggered by the data file upload, and parses the contents of the
# triggering file, then saves them to the appropriate children
def parse_input_data_file(contents, filename, date):
    logging.info(f"parse data")

    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        try:
            if filename.endswith("csv"):
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            elif filename.endswith("txt"):
                # Assume that the user uploaded a whitespace-delimited CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode("utf-8")), delim_whitespace=True
                )
            elif filename.endswith("xlsx"):
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
            else:
                raise NotImplementedError
        except Exception as e:
            logging.error(f"{e}")
            return [html.Div(["There was an error processing this file."])]

        store("parsed", df)

        return [
            f"{filename} loaded, last modified "
            f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
        ]

    return ["No file loaded"]


@app.callback(
    [Output("output-filter-file-upload", "children")],
    [Input("filter-file-upload", "contents")],
    [
        State("filter-file-upload", "filename"),
        State("filter-file-upload", "last_modified"),
    ],
    prevent_initial_call=True,
)
# This function is triggered by either upload, and parses the contents of the
# triggering file, then saves them to the appropriate children
def parse_input_filter_file(contents, filename, date):
    logging.info(f"parse filter")

    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode()

        try:
            variables_of_interest = [str(item) for item in decoded.splitlines()]
            # Add index names if they are not present
            # MSC not sure if useful here
            #variables_of_interest.extend(
            #    index for index in indices if index not in variables_of_interest
            #)

        except Exception as e:
            logging.error(f"{e}")
            return html.Div(["There was an error processing this file."])
        df = pd.DataFrame(variables_of_interest, columns=["names"])
        store("columns", df)

        return [
            f"{filename} loaded, last modified "
            f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
        ]

    return ["No file loaded"]


@app.callback(
    [Output("df-loaded-div", "children")],
    [Input("load-files-button", "n_clicks")],
    [State("data-file-upload", "filename"), State("filter-file-upload", "filename")],
    prevent_initial_call=True,
)
# This function is triggered by the button, and takes the parsed values of the 1 or 2
# upload components, and outputs the resulting df to the div.
def update_df_loaded_div(n_clicks, data_file_value, filter_file_value):
    logging.info(f"update_df_loaded_div")
    # Read in main DataFrame
    if data_file_value is None:
        return [False]
    df = load("parsed")

    # Read in column DataFrame, or just use all the columns in the DataFrame
    if filter_file_value is not None:
        variables_of_interest = list(load("columns")["names"])

        # Verify that the variables of interest exist in the dataframe
        missing_vars = [var for var in variables_of_interest if var not in df.columns]

        if missing_vars:
            raise ValueError(
                str(missing_vars)
                + " is in the filter file but not found in the data file."
            )

        # Keep only the columns listed in the filter file
        df = df[variables_of_interest]

    df = df.drop(columns="index", errors="ignore")
    logging.info("df\n: %s" % str(df))

    # Reformat SUBJECTKEY if it doesn't have the underscore
    # TODO: remove this when unnecessary
    if "SUBJECTKEY" in df:
        df["SUBJECTKEY"] = df["SUBJECTKEY"].apply(standardise_subjectkey)

    # Set certain columns to have more specific types.
    # if 'SEX' in df.columns:
    #     df['SEX'] = df['SEX'].astype('category')
    #
    # for column in ['EVENTNAME', 'SRC_SUBJECT_ID']:
    #     if column in df.columns:
    #         df[column] = df[column].astype('string')

    # Set index. We try all known indices and apply the first
    # one where all the columns are present. Otherwise just
    # leave default index
    for index in known_indices:
        if all([col in df for col in index]):
            df.set_index(index, inplace=True, verify_integrity=True, drop=False)
            break

    # Store the combined DF, and set df-loaded-div to [True]
    store("df", df)

    return [True]


def main():
    logging.getLogger().setLevel(logging.INFO)
    app.run_server(debug=False)


if __name__ == "__main__":
    main()
