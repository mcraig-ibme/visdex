"""
visdex: Dashboard data explorer for CSV trial data

This module defines the layout and behaviour of 
the main application page.

Originally written for ABCD data
"""
import os
import logging
import io
import base64
import datetime

import pandas as pd

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.graph_objects as go

from .cache import cache
from .common import create_header, HEADER_IMAGE, div_style, standard_margin_left
from ._version import __version__
import visdex.summary
import visdex.export
import visdex.exploratory_graphs

LOG = logging.getLogger(__name__)

# Possible sets of index columns. 
# This is very data specific so we just try them in order and use the first set
# that matches the DF
known_indices = [
    ["SUBJECTKEY", "EVENTNAME"],
]

# Create the Dash application
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=False,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div(
    children=[
        html.Img(src=HEADER_IMAGE, height=100),
        create_header("Visual Data Explorer v%s" % __version__),
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

        # Hidden divs for holding the booleans identifying whether a DF is loaded in
        # each case
        html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="df-filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),

        # Export component
        visdex.export.get_layout(app),

        # Summary section
        visdex.summary.get_layout(app),

        # Exploratory graphs
        visdex.exploratory_graphs.get_layout(app),
    ]
)

@app.callback(
    [Output("output-data-file-upload", "children")],
    [Input("data-file-upload", "contents")],
    [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
    prevent_initial_call=True,
)
def parse_input_data_file(contents, filename, date):
    """
    Handle data file upload
    
    Parses the contents of the uploaded file into a DataFrame and stores
    it in the cache, updating the appropriate children
    """
    LOG.info(f"parse data")

    if contents is not None:
        _content_type, content_string = contents.split(",")
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
            LOG.error(f"{e}")
            return [html.Div(["There was an error processing this file."])]

        cache.store("parsed", df)

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
def parse_input_filter_file(contents, filename, date):
    """
    Handle upload of column filter file

    The file is parsed and stored in the cache. It is used to filter
    the columns of the main data file, and the filtered data frame is
    also stored in the cache
    """
    LOG.info(f"parse filter")

    if contents is not None:
        _content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string).decode()

        try:
            variables_of_interest = [str(item) for item in decoded.splitlines()]
            # Add index names if they are not present
            # MSC not sure if useful here
            #variables_of_interest.extend(
            #    index for index in indices if index not in variables_of_interest
            #)

        except Exception as e:
            LOG.error(f"{e}")
            return html.Div(["There was an error processing this file."])
        df = pd.DataFrame(variables_of_interest, columns=["names"])
        cache.store("columns", df)

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
def update_df_loaded_div(n_clicks, data_file_value, filter_file_value):
    """
    Handle click on the 'Analyse' button
    
    The main data file is parsed and filtered and the result stored in the cache
    """
    LOG.info(f"update_df_loaded_div")
    # Read in main DataFrame
    if data_file_value is None:
        return [False]
    df = cache.load("parsed")

    # Read in column DataFrame, or just use all the columns in the DataFrame
    if filter_file_value is not None:
        variables_of_interest = list(cache.load("columns")["names"])

        # Verify that the variables of interest exist in the dataframe
        missing_vars = [var for var in variables_of_interest if var not in df.columns]

        if missing_vars:
            raise ValueError(
                str(missing_vars)
                + " is in the filter file but not found in the data file."
            )

        # Keep only the columns listed in the filter file
        df = df[variables_of_interest]

    LOG.info("df\n: %s" % str(df))

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
    # one where all the columns are present.
    for index in known_indices:
        if all([col in df for col in index]):
            df.set_index(index, inplace=True, verify_integrity=True, drop=True)
            break

    # Store the combined DF, and set df-loaded-div to [True]
    cache.store("df", df)

    return [True]

def standardise_subjectkey(subjectkey):
    """
    Standardise the subject key

    FIXME does this actually do anything useful? It just looks
    like it doubles an underscore in a particular position if
    found
    """
    if subjectkey[4] == "_":
        return subjectkey

    return subjectkey[0:4] + "_" + subjectkey[4:]
