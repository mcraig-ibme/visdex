"""
visdex: Data handling components

These components handle the loading of the main data and filter files
"""
import logging
import io
import base64
import datetime

import pandas as pd

from dash import html, dcc
from dash.dependencies import Input, Output, State

from visdex.common import div_style, hstack
from visdex.cache import cache

LOG = logging.getLogger(__name__)

# Possible sets of index columns. 
# This is very data specific so we just try them in order and use the first set
# that matches the DF
known_indices = [
    ["SUBJECTKEY", "EVENTNAME"],
]

def get_layout(app):
    layout = html.Div(children=[
        html.H2(children="Data selection", style=div_style),
        html.Label(
            children="Data File Selection",
            style=div_style,
        ),
        dcc.Upload(
            id="data-file-upload",
            children=html.Div([html.A(id="output-data-file-upload", children="Drag and drop or click to select files")]),
            style={
                "height": "60px",
                "lineHeight" : "60px",
                "borderWidth" : "1px",
                "borderStyle" : "dashed",
                "borderRadius" : "5px",
                "textAlign" : "center",
                "margin" : "10px",
            },
        ),
        html.Label(
            children="Column Filter File Selection",
            style=div_style,
        ),
        dcc.Upload(
            id="filter-file-upload",
            children=html.Div([html.A(id="output-filter-file-upload", children="Drag and drop or click to select files")]),
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
            id="data-warning", children=[""], style=div_style
        ),
        html.Button("Analyse", id="load-files-button", style=div_style),

        # Hidden divs for holding the booleans identifying whether a DF is loaded in
        # each case
        html.Div(id="df-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="df-filtered-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="corr-loaded-div", style={"display": "none"}, children=[]),
        html.Div(id="pval-loaded-div", style={"display": "none"}, children=[]),
    ])

    @app.callback(
        [Output("output-data-file-upload", "children")],
        [Input("data-file-upload", "contents")],
        [State("data-file-upload", "filename"), State("data-file-upload", "last_modified")],
        prevent_initial_call=True,
    )
    def _data_file_changed(contents, filename, date):
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
                return ["There was an error processing this file"]

            cache.store("parsed", df)

            return [
                f"{filename} loaded, last modified "
                f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
            ]

        return ["No file loaded", ""]
    
    @app.callback(
        [Output("output-filter-file-upload", "children")],
        [Input("filter-file-upload", "contents")],
        [
            State("filter-file-upload", "filename"),
            State("filter-file-upload", "last_modified"),
        ],
        prevent_initial_call=True,
    )
    def _filter_file_changed(contents, filename, date):
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
                return ["There was an error processing this file"]
            df = pd.DataFrame(variables_of_interest, columns=["names"])
            cache.store("columns", df)

            return [
                f"{filename} loaded, last modified "
                f"{datetime.datetime.fromtimestamp(date).strftime('%Y-%m-%d %H:%M:%S')}"
            ]

        return ["No file loaded"]

    @app.callback(
        [Output("df-loaded-div", "children"), Output("data-warning", "children")],
        [Input("load-files-button", "n_clicks")],
        [State("data-file-upload", "filename"), State("filter-file-upload", "filename")],
        prevent_initial_call=True,
    )
    def _analyse_button_clicked(n_clicks, data_file_value, filter_file_value):
        """
        Handle click on the 'Analyse' button
        
        The main data file is parsed and filtered and the result stored in the cache
        """
        LOG.info(f"update_df_loaded_div")
        warning = ""

        # Read in main DataFrame
        if data_file_value is None:
            return [False, ""]
        df = cache.load("parsed")

        # Read in column DataFrame, or just use all the columns in the DataFrame
        if filter_file_value is not None:
            variables_of_interest = list(cache.load("columns")["names"])

            # Remove variables that don't exist in the dataframe
            missing_vars = [var for var in variables_of_interest if var not in df.columns]
            present_vars = [var for var in variables_of_interest if var in df.columns]

            if missing_vars:
                warning = "Variables found in filter file that are not present in main data"

            # Keep only the columns listed in the filter file
            df = df[present_vars]

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

        return [True, warning]

    return layout

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
